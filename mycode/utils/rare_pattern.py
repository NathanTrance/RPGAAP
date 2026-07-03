import torch
import numpy as np


def make_rare_pattern_weights(features, labels, train_mask,
                               num_bins=5, top_k_features=10,
                               max_weight=3.0, fraud_boost=1.5):
    """
    Compute rare-pattern sample weights for graph fraud detection.

    Args:
        features: Tensor (N, d) -- node features
        labels: Tensor (N,) -- node labels (0 = normal, 1 = fraud)
        train_mask: Tensor (N,) -- boolean mask of training nodes
        num_bins: number of quantile bins per feature
        top_k_features: number of top features by variance to use
        max_weight: maximum sample weight
        fraud_boost: extra multiplier for fraud nodes

    Returns:
        weights: Tensor (N,) -- sample weights for all nodes
    """
    N = features.shape[0]
    features = features.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_mask = train_mask.detach().cpu().numpy().astype(bool)

    train_features = features[train_mask]
    if train_features.shape[0] == 0:
        return torch.ones(N, dtype=torch.float32)

    var = np.var(train_features, axis=0)
    top_k = min(top_k_features, len(var))
    top_indices = np.argsort(var)[-top_k:]

    patterns = np.zeros(train_features.shape[0], dtype=np.int64)
    multiplier = 1
    for col in top_indices:
        col_vals = train_features[:, col]
        bins = np.quantile(col_vals, np.linspace(0, 1, num_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        digitized = np.digitize(col_vals, bins[1:-1], right=False)
        patterns += digitized * multiplier
        multiplier *= num_bins

    unique_patterns, inverse_indices, counts = np.unique(
        patterns, return_inverse=True, return_counts=True)

    max_count = counts.max()
    min_count = counts.min()
    if max_count == min_count:
        normalized_freq = np.ones_like(counts, dtype=np.float64)
    else:
        normalized_freq = (max_count - counts) / (max_count - min_count)

    train_weights = np.zeros(train_features.shape[0], dtype=np.float64)
    for i in range(len(unique_patterns)):
        mask_i = inverse_indices == i
        train_weights[mask_i] = normalized_freq[i]

    train_weights = 1.0 + (max_weight - 1.0) * train_weights

    is_fraud = labels[train_mask] == 1
    train_weights[is_fraud] *= fraud_boost

    train_weights = np.clip(train_weights, 1.0, max_weight * fraud_boost)

    weights = np.ones(N, dtype=np.float64)
    weights[train_mask] = train_weights

    return torch.tensor(weights, dtype=torch.float32)
