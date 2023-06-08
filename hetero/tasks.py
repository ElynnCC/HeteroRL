import attrs
import collections
import numpy as np

from typing import Iterable

from hetero.config import DataGenConfig, DTYPE
from hetero.datagen import generate_data_from_config
from hetero.utils import action_feature_prod
from hetero.algo import MCPImpl, BetaOptimizer, group, align_binary_labels


def compute_UV_truths(
    data_config_init,
    discount,
    pi_eval,
    num_repeats=10,
    num_trajectories=1000,
    seed=7531,
):
    us = []
    vs = []

    for i in range(num_repeats):
        data_config = DataGenConfig(
            num_trajectories=num_trajectories, seed=seed * (i + 1), **data_config_init
        )
        data = generate_data_from_config(data_config, pi_sample=pi_eval)

        pi_eval_actions = pi_eval.action(data.next_features)
        us.append(action_feature_prod(pi_eval_actions, data.next_features).mean(axis=0))

        discount_factors = np.array(
            [discount**i for i in data.time_step_indices], dtype=DTYPE
        )
        discounted_rewards = data.rewards * discount_factors
        group_values = list(set(x[0] for x in data.labels))
        group_values.sort()
        group_masks = [[x[0] == gv for x in data.labels] for gv in group_values]
        num_masked = [
            len({x[1] for x in data.labels if x[0] == gv}) for gv in group_values
        ]
        vs.append(
            np.array(
                [
                    discounted_rewards[mask].sum() / n
                    for mask, n in zip(group_masks, num_masked)
                ],
                dtype=DTYPE,
            )
        )

    return np.stack(us, axis=0), np.stack(vs, axis=0)


@(attrs.frozen)
class BetaEstimate(object):
    betas: Iterable[np.array]
    sigmas: Iterable[np.array]
    omegas: Iterable[np.array]
    num_samples: Iterable[int]
    mean_sqrt_residuals: Iterable[float]
    all_residuals: Iterable[np.array]


def beta_estimate_from(data, pi_eval, discount):
    Z = action_feature_prod(data.actions, data.current_features)
    pi_eval_actions = pi_eval.action(data.next_features)
    U = action_feature_prod(pi_eval_actions, data.next_features)

    v = Z - discount * U
    left = np.einsum("id,ie->ide", Z, v)
    right = Z * data.rewards.reshape([-1, 1])
    partitioner = data.partitioner

    betas = []
    sigmas = []
    omegas = []
    num_samples = []
    mean_sqrt_residuals = []
    all_residuals = []
    for mask in partitioner.index_to_mask_mapping():
        num_samples.append(mask.sum())
        sigma = left[mask, :, :].sum(axis=0)
        sigmas.append(sigma / num_samples[-1])
        right_i = right[mask, :].sum(axis=0)
        betas.append(np.linalg.pinv(sigma) @ right_i)
        residual_i = data.rewards[mask] - v[mask, :].dot(betas[-1])
        all_residuals.append(residual_i)
        mean_sqrt_residuals.append(np.sqrt((residual_i * residual_i).mean()))
        omega_i = np.einsum("id,ie,i->de", Z[mask, :], Z[mask, :], residual_i**2)
        omegas.append(omega_i / num_samples[-1])

    return BetaEstimate(
        betas, sigmas, omegas, num_samples, mean_sqrt_residuals, all_residuals
    )


def beta_estimate_from_new_labels(data, new_labels, pi_eval, discount):
    converted_data = data.replace_labels(new_labels)
    old_label_to_new_index = collections.defaultdict(set)
    for old_label, new_label in zip(data.labels, new_labels):
        old_label_to_new_index[old_label].add(
            converted_data.partitioner.label_to_index(new_label)
        )
    new_label_index_converter = {}
    for old_label, new_index in old_label_to_new_index.items():
        assert len(new_index) == 1
        new_label_index_converter[old_label] = list(new_index)[0]

    return (
        beta_estimate_from(converted_data, pi_eval, discount),
        new_label_index_converter,
    )


def beta_estimate_from_e2e_learning(
    data, algo_config, grouping_config, pi_eval, init_beta=None
):
    if algo_config.use_group_wise_regression_init:
        if init_beta is None:
            group_labels = [x[0] for x in data.labels]
            group_wise_beta, new_label_index_converter = beta_estimate_from_new_labels(
                data, group_labels, pi_eval, algo_config.discount
            )
            init_beta = np.stack(
                [
                    group_wise_beta.betas[new_label_index_converter[x]]
                    for x in data.partitioner.index_to_label_mapping()
                ],
                axis=0,
            )
            # print("group_wise_regression_init_beta_before_noise=", init_beta)
            init_beta_noise_rand = np.random.RandomState(algo_config.seed)
            init_beta += init_beta_noise_rand.normal(
                scale=algo_config.group_wise_regression_init_noise, size=init_beta.shape
            ).astype(DTYPE)
        else:
            print(
                """
!!! INCONSISTENCY in CONFIG !!!
The algo_config is set to use group wise regression to initialize beta,
but init_beta is also specified. Please double check your config before
proceeding. For now, we continue with the supplied init_beta.
!!! INCONSISTENCY in CONFIG !!!
"""
            )

    impl = MCPImpl(data.N(), algo_config)
    beta_opt = BetaOptimizer(data, algo_config, pi_eval, impl, init_beta)
    betas = beta_opt.compute()
    learned_labels = group(betas, grouping_config)
    truth = np.array([x[0] for x in data.partitioner.index_to_label_mapping()])
    aligned_labels = align_binary_labels(learned_labels, truth)
    assert np.all(aligned_labels == learned_labels) or np.all(
        aligned_labels == 1 - learned_labels
    )
    print("Label mismatch =", (aligned_labels != truth).sum())
    return beta_estimate_from_new_labels(
        data, aligned_labels, pi_eval, algo_config.discount
    )[0]


def beta_estimate_from_nongrouped(data, pi_eval, discount):
    new_labels = [0] * data.NT()
    return beta_estimate_from_new_labels(data, new_labels, pi_eval, discount)[0]


def compute_V_estimate(u_truth, beta_estimate):
    mus = [(u_truth * beta).sum() for beta in beta_estimate.betas]
    sigmas = [
        np.sqrt(
            u_truth.reshape([1, -1])
            @ np.linalg.inv(sigma)
            @ omega
            @ np.linalg.inv(sigma).T
            @ u_truth.reshape([-1, 1])
        )[0, 0]
        / np.sqrt(n)
        for sigma, omega, n in zip(
            beta_estimate.sigmas, beta_estimate.omegas, beta_estimate.num_samples
        )
    ]
    return mus, sigmas
