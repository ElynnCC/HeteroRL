'''
offline_rl.py contains the implementation of problem configurations.

Most variables and functions are self-explanatory in their names, indicating their intended purposes.

Comments are provided for some functions or variables that are not self-explanatory.
'''

import attrs

import numpy as np
import torch

from hetero.config import DTYPE
from hetero.policies import SoftmaxPolicy
from hetero.utils import LabelPartitioner, action_feature_prod


def _eval(pi_eval, kernel, features):
    return np.einsum(
        "ia, as, is -> i",
        pi_eval.action(features),
        kernel,
        features,
    ).mean()


class SARSSamplePolicy(object):
    def __init__(self, current_features, actions, rewards, next_features):
        self.current_features = current_features
        self.actions = actions
        self.rewards = rewards
        self.next_features = next_features

        assert len(self.rewards.shape) == 1
        assert (
            len(self.current_features.shape)
            == len(self.actions.shape)
            == len(self.next_features.shape)
            == 2
        )
        self.num_samples = self.rewards.shape[0]
        assert self.current_features.shape == self.next_features.shape
        assert (
            self.current_features.shape[0]
            == self.actions.shape[0]
            == self.next_features.shape[0]
            == self.num_samples
        )
        self.num_feature_dims = self.current_features.shape[1]
        self.num_actions = self.actions.shape[1]
        self.Z = action_feature_prod(self.actions, self.current_features)
        self.right = self.Z * self.rewards.reshape([-1, 1])

    def _U(self, pi_eval):
        u_actions = pi_eval.action(self.next_features)
        return action_feature_prod(u_actions, self.next_features)

    def features(self):
        return self.current_features

    def fit_Q_kernel(self, pi_eval, discount):
        U = self._U(pi_eval)
        v = self.Z - discount * U
        left = np.einsum("id, ie -> de", self.Z, v)
        right = self.right.sum(axis=0)
        return (np.linalg.pinv(left) @ right).reshape(
            [self.num_actions, self.num_feature_dims]
        )

    def fit_Q_kernel_by_label(self, pi_eval, discount, labels):
        U = self._U(pi_eval)
        v = self.Z - discount * U
        left = np.einsum("id, ie -> ide", self.Z, v)

        lp = LabelPartitioner(labels)
        ret = {}
        for label in lp.index_to_label_mapping():
            label_mask = lp.label_to_mask(label)
            label_left = left[label_mask, :, :].sum(axis=0)
            label_right = self.right[label_mask, :].sum(axis=0)
            ret[label] = (np.linalg.pinv(label_left) @ label_right).reshape(
                [self.num_actions, self.num_feature_dims]
            )

        return ret

    def fit_then_eval(self, pi_eval, discount):
        kernel = self.fit_Q_kernel(pi_eval, discount)
        return _eval(pi_eval, kernel, self.current_features)


def split_sars_data_for_rl(data, cluster_labels):
    clp = LabelPartitioner(cluster_labels)
    converter = dict(zip(data.partitioner.index_to_label_mapping(), cluster_labels))
    ret = {}

    for label in clp.index_to_label_mapping():
        record_mask = [converter[x] == label for x in data.labels]
        ret[label] = SARSSamplePolicy(
            data.current_features[record_mask, :],
            data.actions[record_mask, :],
            data.rewards[record_mask],
            data.next_features[record_mask, :],
        )
    return ret


@(attrs.frozen)
class ActorCriticLearnerParams(object):
    # Params for initializing softmax weights
    seed: int = attrs.field(default=7531)
    # Close to 0, i.e., policy close to uniform random
    init_weight_bound: DTYPE = attrs.field(default=0.01)
    rel_weight_eps: DTYPE = attrs.field(default=0.01)

    # Number of policy-eval => actor-learning iters
    num_eval_learn_iters: int = attrs.field(default=100)

    discount: DTYPE = attrs.field(default=0.6)

    # Params for ActorLearner
    lr: DTYPE = attrs.field(default=0.1)
    num_steps: int = attrs.field(default=100)
    rel_y_eps: DTYPE = attrs.field(default=0.01)
    G_eps: DTYPE = attrs.field(default=0.1)
    decay: DTYPE = attrs.field(default=0.001)
    grad_norm_clip: DTYPE = attrs.field(default=0)


class ActorLearner(object):
    def __init__(self, kernel, features, init_weights, params=None):
        self._kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        self._features = torch.tensor(
            features, dtype=torch.float32, requires_grad=False
        )
        self._weights = torch.tensor(
            init_weights, dtype=torch.float32, requires_grad=True
        )

        self.params = params or ActorCriticLearnerParams()
        self.G = (
            torch.ones_like(self._weights, dtype=torch.float32, requires_grad=False)
            * self.params.G_eps
        )

    def _forward(self):
        logits = self._features @ self._weights.T
        actions = torch.nn.Softmax(dim=-1)(logits)
        return torch.einsum(
            "ia, as, is -> i", actions, self._kernel, self._features
        ).sum()

    def _one_step(self, i):
        y = self._forward()
        y.backward()
        with torch.no_grad():
            g = self._weights.grad
            g_norm = torch.norm(g).numpy()
            g_norm_clipped = (
                self.params.grad_norm_clip and g_norm > self.params.grad_norm_clip
            )
            lr_clip = self.params.grad_norm_clip / g_norm if g_norm_clipped else 1.0
            self.G += g * g
            self._weights += (
                self.params.lr * lr_clip * g / torch.sqrt(self.G)
                - self.params.decay * self._weights
            )
            self._weights.grad.zero_()
            ret = y.numpy()
            clip_info = f"(CLIP@{self.params.grad_norm_clip})" if g_norm_clipped else ""
            print(
                f"  * At actor-learn step {i}, g_norm={g_norm:0.3f}{clip_info}, y={ret:0.3f}"
            )
            return ret

    def learn(self):
        prev_y = None
        for i in range(self.params.num_steps):
            y = self._one_step(i)
            if prev_y is not None:
                relative = np.abs(y - prev_y) / (np.abs(prev_y) + self.params.rel_y_eps)
                if relative < self.params.rel_y_eps:
                    print(
                        f"  * At actor-learn step {i}, stopped at y={y:0.3f}, "
                        f"prev_y={prev_y:0.3f}, relative_error={relative:0.3f}"
                    )
                    break
            prev_y = y

    def weights(self):
        return self._weights.detach().numpy()


class ActorCriticLeaner(object):
    def __init__(self, sample_pi, params=None):
        self.sample_pi = sample_pi
        self.params = params or ActorCriticLearnerParams()
        rand = np.random.RandomState(self.params.seed)
        self._weights = rand.uniform(
            low=-self.params.init_weight_bound,
            high=self.params.init_weight_bound,
            size=[sample_pi.num_actions, sample_pi.num_feature_dims],
        ).astype(DTYPE)
        actor_pi = SoftmaxPolicy(self._weights)
        self._kernel = self.sample_pi.fit_Q_kernel(actor_pi, self.params.discount)

    def _one_iter(self):
        actor_learner = ActorLearner(
            self._kernel, self.sample_pi.features(), self._weights, self.params
        )
        actor_learner.learn()
        new_weights = actor_learner.weights()
        return new_weights

    def learn(self):
        for i in range(self.params.num_eval_learn_iters):
            new_weights = self._one_iter()
            diff = new_weights - self._weights
            diff_norm = np.linalg.norm(diff)
            cur_norm = np.linalg.norm(self._weights)
            relative = diff_norm / (cur_norm + self.params.rel_weight_eps)
            print(
                f"* At alter step {i}, diff_norm={diff_norm:0.3f}, "
                f"cur_norm={cur_norm:0.3f}, relative_error={relative:0.3f}"
            )
            self._weights = new_weights
            if relative < self.params.rel_weight_eps:
                break

            actor_pi = SoftmaxPolicy(self._weights)
            self._kernel = self.sample_pi.fit_Q_kernel(actor_pi, self.params.discount)

    def weights(self):
        return self._weights

    def kernel(self):
        return self._kernel

    def eval(self):
        learned_pol = SoftmaxPolicy(self._weights)
        return _eval(learned_pol, self._kernel, self.sample_pi.features())
