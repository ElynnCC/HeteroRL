'''
policies.py contains the implementation of policy functions.

Most variables and functions are self-explanatory in their names, indicating their intended purposes.

Comments are provided for some functions or variables that are not self-explanatory.
'''

import attrs

import numpy as np
from scipy.special import softmax

from hetero.config import DTYPE
from hetero.utils import lookup_inner


class AllPosPolicy(object):
    def action(self, features):
        assert features.shape[-1] == 2
        pos = features > 0
        split = np.split(pos, 2, axis=-1)
        return np.logical_and(*split).astype(DTYPE)


class XORPolicy(object):
    def action(self, features):
        assert features.shape[-1] == 2
        pos = features > 0
        split = np.split(pos, 2, axis=-1)
        return (split[0] != split[1]).astype(DTYPE)


class AlternativePolicy(object):
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def action(self, features):
        features_shape = features.shape
        features = features.reshape([-1, features_shape[-1]])
        num_pos = (features > 0).sum(axis=-1)
        action_indices = np.remainder(
            np.arange(features.shape[0]) + num_pos, self.num_actions
        ).reshape(features_shape[:-1])
        return lookup_inner(np.eye(self.num_actions, dtype=DTYPE), action_indices)


class UniformRandomPolicy(object):
    def __init__(self, num_actions, rand):
        self.num_actions = num_actions
        self.rand = rand

    def action(self, features):
        action_shape = features.shape[:-1]
        action_indices = self.rand.choice(
            np.arange(self.num_actions), size=action_shape, replace=True
        )
        return lookup_inner(np.eye(self.num_actions, dtype=DTYPE), action_indices)


@(attrs.frozen)
class SoftmaxPolicy(object):
    weights: np.array

    def __attrs_post_init__(self):
        assert len(self.weights.shape) == 2

    def action(self, features):
        assert self.weights.shape[-1] == features.shape[-1]
        logits = features @ self.weights.T
        return softmax(logits, axis=-1)
