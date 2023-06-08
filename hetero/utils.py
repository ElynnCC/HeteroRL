import numpy as np

from hetero.config import DTYPE


def pairwise_diff(beta):
    d = beta.shape[0]
    left = np.stack([beta] * d, axis=1)
    right = np.stack([beta] * d, axis=0)
    return left - right


def soft_thresh(x, c):
    return np.sign(x) * np.clip(np.abs(x) - c, 0, None)


def print_if_nan(var_dict, var_name):
    var_val = var_dict[var_name]
    assert not np.any(np.isnan(var_val)), f"NaN found in {var_name}: {var_val}"


def remove_outlier(flat_beta, lower_perc=0, upper_perc=100):
    upper = np.percentile(flat_beta, upper_perc, axis=0)
    lower = np.percentile(flat_beta, lower_perc, axis=0)
    return np.clip(flat_beta, lower, upper)


def lookup_inner(values, indices):
    index_shape = list(indices.shape)
    assert len(values.shape) == 2
    flat = values[indices.ravel()]
    return flat.reshape(index_shape + [values.shape[-1]])


def action_feature_prod(action, feature):
    assert len(action.shape) == len(feature.shape) == 2
    assert action.shape[0] == feature.shape[0]
    ret = np.einsum("ia, is -> ias", action, feature)
    return ret.reshape([action.shape[0], -1])


class LabelPartitioner(object):
    def __init__(self, labels):
        self._unique_labels = list(set(labels))
        self._unique_labels.sort()
        self._label_to_index = {label: i for i, label in enumerate(self._unique_labels)}
        self._index_to_mask = [
            np.array([x == label for x in labels], dtype=np.bool_)
            for label in self._unique_labels
        ]

    def num_unique_labels(self):
        return len(self._unique_labels)

    def index_to_label_mapping(self):
        return self._unique_labels

    def label_to_index_mapping(self):
        return self._label_to_index

    def index_to_mask_mapping(self):
        return self._index_to_mask

    def index_to_label(self, i):
        return self._unique_labels[i]

    def label_to_index(self, label):
        return self._label_to_index[label]

    def index_to_mask(self, i):
        return self._index_to_mask[i]

    def label_to_mask(self, label):
        return self._index_to_mask[self._label_to_index[label]]


def generate_decay_matrix(m, n, decay):
    row = [decay**i for i in range(n)]
    return np.array([row] * m, dtype=DTYPE)
