from typing import Iterable

import attrs
import numpy as np

DTYPE = np.float32


@(attrs.frozen)
class DimConfig(object):
    # Following params are hardcoded. If changed, please double check all implementation.
    NUM_ACTIONS: int = 2
    NUM_GROUPS: int = 2
    NUM_STATE_DIMS: int = 2

    num_trajectories: int = attrs.field(default=100)
    # If num_time_steps = 2, we have s0 -> s1 -> s2
    num_time_steps: int = attrs.field(default=10)

    basis_expansion_method: str = attrs.field(default="NONE")
    basis_expansion_factor: int = attrs.field(default=2)

    add_intercept_column: bool = attrs.field(default=False)

    def num_beta_dims(self):
        return self.NUM_ACTIONS * self.J()

    def num_feature_dims_without_intercept(self):
        return (
            self.NUM_STATE_DIMS
            if self.basis_expansion_method == "NONE"
            else self.NUM_STATE_DIMS * self.basis_expansion_factor
        )

    def num_feature_dims(self):
        return self.num_feature_dims_without_intercept() + (
            1 if self.add_intercept_column else 0
        )

    def J(self):
        return self.num_feature_dims()

    def K(self):
        return self.NUM_GROUPS

    def M(self):
        return self.NUM_ACTIONS

    def N(self):
        return self.NUM_GROUPS * self.num_trajectories

    def T(self):
        return self.num_time_steps


DEFAULT_GROUP_REWARD_COEFF_OVERRIDE = np.array([[2.0, -1.0], [-2.0, 1.0]], dtype=DTYPE)


@(attrs.frozen)
class DataGenConfig(DimConfig):
    seed: int = attrs.field(default=7531)

    # state transition
    # X_i,t+1 = M_i, y X_i, t + eps_i,t
    # M_i = [ action_transition_coeff * (2 * action - 1.0),  0,
    #         0,  -action_transition_coeff * (2 * action - 1.0)]
    init_state_mu: DTYPE = attrs.field(default=0.0)
    init_state_sigma: DTYPE = attrs.field(default=1.0)
        
    noise_type: str = attrs.field(default="NORMAL") # Change to STUDENT for t   
    noise_sigma: DTYPE = attrs.field(default=0.5)
    noise_student_degree: int = attrs.field(default=4)
        
    prob_action_1: DTYPE = attrs.field(default=0.5)
    action_transition_coeff: DTYPE = attrs.field(default=0.75)
    num_burnin_steps: int = attrs.field(default=0)

    # reward = X . group_reward_coeff[k,:] + action_reward_coeff * (2 * action - 1.0)
    group_reward_coeff_override: np.array = attrs.field(default=None)
    group_reward_coeff_range: Iterable[DTYPE] = attrs.field(default=[-3, 3])
    action_reward_coeff: Iterable[DTYPE] = attrs.field(default=[-1, 1])
    use_01_action_values: bool = attrs.field(default=True)

    bspline_num_degrees: int = attrs.field(default=2)

    transformation_method: str = attrs.field(default="NONE")

    def num_legendre_bases(self):
        return self.basis_expansion_factor

    def num_sine_bases(self):
        return self.basis_expansion_factor

    def num_total_steps(self):
        return self.num_time_steps + self.num_burnin_steps

    def bspline_num_freedoms(self):
        return self.bspline_num_degrees + 1

    def __attrs_post_init__(self):
        assert (
            self.group_reward_coeff_override is None
            or self.group_reward_coeff_override.shape
            == (self.NUM_GROUPS, self.num_feature_dims_without_intercept())
        )

        assert (
            self.basis_expansion_method != "BSPLINE"
            or self.bspline_num_degrees >= self.basis_expansion_factor
        )

        assert len(self.action_reward_coeff) == self.NUM_GROUPS


@(attrs.frozen)
class AlgoConfig(object):
    max_num_iters: int = attrs.field(default=10)
    discount: DTYPE = attrs.field(default=0.6)
    gam: DTYPE = attrs.field(default=2.0)
    lam: DTYPE = attrs.field(default=1.0)
    rho: DTYPE = attrs.field(default=2.0)
    nu_coeff: DTYPE = attrs.field(default=0.0)
    delta_coeff: DTYPE = attrs.field(default=0.0)

    should_remove_outlier: bool = attrs.field(default=False)
    outlier_lower_perc: int = attrs.field(default=2)
    outlier_upper_perc: int = attrs.field(default=98)

    use_group_wise_regression_init: bool = attrs.field(default=False)
    group_wise_regression_init_noise: DTYPE = attrs.field(default=0.1)

    seed: int = attrs.field(default=7531)

    def __attrs_post_init__(self):
        assert self.gam * self.rho > 1.0
        assert 0 <= self.outlier_lower_perc < self.outlier_upper_perc <= 100


@(attrs.frozen)
class GroupingConfig(object):
    method: str = attrs.field(default="kmeans")
    num_clusters: int = attrs.field(default=2)
    seed: int = attrs.field(default=7531)
    kmeans_init_method: str = attrs.field(default="k-means++")
    kmeans_num_inits: int = attrs.field(default=3)
    kmeans_max_iter: int = attrs.field(default=30)
    kmeans_tol: DTYPE = attrs.field(default=1e-3)
