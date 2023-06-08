import numpy as np
from statsmodels.gam.smooth_basis import BSplines
from scipy.special import legendre

from hetero.config import DTYPE


def bspline_expand(states, config):
    assert len(states) == config.num_total_steps() + 1
    assert states[0].shape == (
        config.NUM_GROUPS,
        config.num_trajectories,
        config.NUM_STATE_DIMS,
    )

    factor = config.basis_expansion_factor
    ret = [
        np.empty(
            [
                config.NUM_GROUPS,
                config.num_trajectories,
                config.num_feature_dims_without_intercept(),
            ],
            dtype=DTYPE,
        )
        for _ in range(config.num_total_steps() + 1)
    ]
    for t in range(config.num_total_steps() + 1):
        for k in range(config.NUM_GROUPS):
            for i in range(config.NUM_STATE_DIMS):
                x = states[t][k, :, i]
                bspl = BSplines(
                    x,
                    df=config.bspline_num_freedoms(),
                    degree=config.bspline_num_degrees,
                    include_intercept=True,
                )
                interleave_mask = [factor * j + i for j in range(factor)]
                left = ret[t]
                right = bspl.transform(x)[:, 1 : (factor + 1)]  # noqa E203
                left[k : (k + 1), :, interleave_mask] = right

    return ret


def _stack_and_swap(x):
    ret = np.stack(x, axis=3)
    ret = np.swapaxes(ret, 2, 3)
    ret_shape = ret.shape
    return ret.reshape(list(ret_shape[:-2]) + [-1])


def legendre_bases(x, max_order):
    return _stack_and_swap([legendre(k)(x) for k in range(1, max_order + 1)])


def legendre_expand(states, config):
    assert len(states) == config.num_total_steps() + 1
    assert states[0].shape == (
        config.NUM_GROUPS,
        config.num_trajectories,
        config.NUM_STATE_DIMS,
    )
    return [legendre_bases(x, config.num_legendre_bases()) for x in states]


def sine_expand(states, config):
    assert len(states) == config.num_total_steps() + 1
    assert states[0].shape == (
        config.NUM_GROUPS,
        config.num_trajectories,
        config.NUM_STATE_DIMS,
    )
    # output = sin(a * x + b)
    a_list = (
        np.arange(1, config.num_sine_bases() + 1)
        * np.pi
        * 2
        / (config.num_sine_bases() + 1)
    )
    b_rand = np.random.RandomState(config.seed)
    b_list = b_rand.uniform(low=0, high=np.pi * 2, size=[config.num_sine_bases()])
    ret = []
    for x in states:
        expanded_list = [np.sin(x * a + b) for a, b in zip(a_list, b_list)]
        ret.append(_stack_and_swap(expanded_list))
    return ret
