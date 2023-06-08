import numpy as np

from sklearn.cluster import KMeans

from hetero.config import DTYPE
from hetero.utils import (
    action_feature_prod,
    pairwise_diff,
    print_if_nan,
    remove_outlier,
    soft_thresh,
)


def beta_from_linreg(data, Z, U, discount):
    v = Z - discount * U
    left = np.einsum("id,ie->ide", Z, v)
    right = Z * data.rewards.reshape([-1, 1])
    partitioner = data.partitioner
    beta = np.empty(
        [
            partitioner.num_unique_labels(),
            Z.shape[1],
        ],
        dtype=DTYPE,
    )
    for i, mask in enumerate(partitioner.index_to_mask_mapping()):
        beta[i, :] = np.linalg.pinv(left[mask, :, :].sum(axis=0)) @ right[mask, :].sum(
            axis=0
        )
    return beta


class MCPImpl(object):
    def __init__(self, N, algo_config):
        self.N = N
        self.algo_config = algo_config

    def compute_new_delta(self, new_beta, nu):
        ret = np.zeros_like(nu)
        cut = self.algo_config.gam * self.algo_config.lam
        num_below, num_above = 0, 0
        for i in range(self.N - 1):
            for j in range(i + 1, self.N):
                v = new_beta[i, :] - new_beta[j, :] + nu[i, j, :] / self.algo_config.rho
                if np.linalg.norm(v) <= cut:
                    ret[i, j, :] = soft_thresh(
                        v, self.algo_config.lam / self.algo_config.rho
                    )
                    ret[i, j, :] /= 1 - 1 / (
                        self.algo_config.gam * self.algo_config.rho
                    )
                    num_below += 1
                else:
                    ret[i, j, :] = v
                    num_above += 1
                ret[j, i, :] = -ret[i, j, :]
        print(f"MCPImpl: num_above={num_above}, num_below={num_below}")
        return ret


def beta_solve(As, c, bs):
    """
    The equations for betas are A_i @ beta_i - c * sum(beta_i) = b_i.
    """
    inv_As = [np.linalg.pinv(A) for A in As]
    inv_Abs = [inv_A @ b for inv_A, b in zip(inv_As, bs)]
    left = np.eye(As[0].shape[0], dtype=DTYPE) - c * sum(inv_As)
    print("beta_solver, min eigen of left matrix =", np.linalg.eigvals(left).min())
    right = sum(inv_Abs)
    sum_betas = np.linalg.solve(left, right)
    return np.stack(
        [c * inv_A @ sum_betas + inv_Ab for inv_A, inv_Ab in zip(inv_As, inv_Abs)],
        axis=0,
    )


class BetaOptimizer(object):
    def __init__(
        self,
        data,
        algo_config,
        pi,
        impl,
        init_beta=None,
    ):
        self.data = data
        self.algo_config = algo_config
        self.impl = impl
        self.Z = action_feature_prod(data.actions, data.current_features)
        self.U = action_feature_prod(pi.action(data.next_features), data.next_features)
        self.G_deriv = self._compute_G_deriv()
        self.init_beta = init_beta

    def _compute_G_deriv(self):
        v = self.Z - self.algo_config.discount * self.U
        left = np.einsum("id,ie->ide", self.Z, v)
        right = self.Z * self.data.rewards.reshape([-1, 1])
        jnt_prod = self.data.J() * self.data.NT()
        c = 2.0 / (jnt_prod**2)
        ret = []
        for mask in self.data.partitioner.index_to_mask_mapping():
            ret.append(
                (c * left[mask, :, :].sum(axis=0), -c * right[mask, :].sum(axis=0))
            )
        return ret

    def _compute_nu_deriv(self, nu):
        c = 0.5 * self.algo_config.nu_coeff / (self.data.N() ** 2)
        return [
            c * (nu[i, :, :].sum(axis=0) - nu[:, i, :].sum(axis=0))
            for i in range(self.data.N())
        ]

    def _delta_deriv_constant(self):
        j_m_n2 = self.data.J() * self.data.M() * (self.data.N() ** 2)
        return 2.0 * self.algo_config.delta_coeff / (2.0 * j_m_n2)

    def _compute_delta_deriv(self, delta):
        c = self._delta_deriv_constant()
        return [
            (
                c * self.data.N() * np.eye(delta.shape[-1], dtype=DTYPE),
                -c * delta[i, :, :].sum(axis=0),
            )
            for i in range(self.data.N())
        ]

    def one_iter(self, beta, delta, nu):
        new_beta = np.zeros_like(beta)
        beta_As = []
        beta_bs = []
        for (g_left, g_right), nu_right, (delta_left, delta_right) in zip(
            self.G_deriv, self._compute_nu_deriv(nu), self._compute_delta_deriv(delta)
        ):
            beta_As.append(g_left + delta_left)
            beta_bs.append(-g_right - nu_right - delta_right)
            for var_name in [
                "g_left",
                "delta_left",
                "g_right",
                "nu_right",
                "delta_right",
            ]:
                print_if_nan(locals(), var_name)

        new_beta = beta_solve(beta_As, self._delta_deriv_constant(), beta_bs)
        print_if_nan(locals(), "new_beta")
        if self.algo_config.should_remove_outlier:
            new_beta = remove_outlier(
                new_beta,
                self.algo_config.outlier_lower_perc,
                self.algo_config.outlier_upper_perc,
            )
        new_delta = self.impl.compute_new_delta(new_beta, nu)
        print_if_nan(locals(), "new_delta")
        new_nu = np.zeros_like(nu)
        for i in range(self.data.N() - 1):
            for j in range(i + 1, self.data.N()):
                new_nu[i, j, :] = nu[i, j, :] + self.algo_config.rho * (
                    new_beta[i, :] - new_beta[j, :] - new_delta[i, j, :]
                )
                new_nu[j, i, :] = -new_nu[i, j, :]
        print_if_nan(locals(), "new_nu")

        return new_beta, new_delta, new_nu

    def _init_beta_delta_nu(self):
        beta = np.empty([self.data.N(), self.data.num_beta_dims()], dtype=DTYPE)
        if self.init_beta is None:
            for i, (g_left, g_right) in enumerate(self.G_deriv):
                beta[i, :] = np.linalg.pinv(g_left) @ (-g_right)
            if self.algo_config.should_remove_outlier:
                beta = remove_outlier(
                    beta,
                    self.algo_config.outlier_lower_perc,
                    self.algo_config.outlier_upper_perc,
                )
        else:
            beta = self.init_beta
        delta = pairwise_diff(beta)
        nu = np.zeros_like(delta)
        return beta, delta, nu

    def compute(self):
        beta, delta, nu = self._init_beta_delta_nu()
        for i in range(self.algo_config.max_num_iters):
            beta, delta, nu = self.one_iter(beta, delta, nu)
        return beta


def group(betas, config):
    assert config.method == "kmeans"
    kmeans = KMeans(
        n_clusters=config.num_clusters,
        random_state=config.seed,
        init=config.kmeans_init_method,
        n_init=config.kmeans_num_inits,
        max_iter=config.kmeans_max_iter,
        tol=config.kmeans_tol,
    ).fit(betas)
    print(f"kmeans center = {kmeans.cluster_centers_} and inertia = {kmeans.inertia_}")
    return kmeans.labels_


def align_binary_labels(labels, truth, threshold=1.0 / 2):
    assert (
        len(labels.shape) == 1
        and len(truth.shape) == 1
        and labels.shape[0] == truth.shape[0]
    )
    num_matched = (labels == truth).sum()
    min_matched = labels.shape[0] * threshold
    if num_matched < min_matched:
        num_reverted = ((1 - labels) == truth).sum()
        assert num_reverted >= min_matched
        return 1 - labels
    else:
        return labels
