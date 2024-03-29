#!/usr/bin/env python3
# coding=utf-8

import torch
from scipy.optimize import linear_sum_assignment
import numpy as np


@torch.no_grad()
def match_label(target, matching, shape, device, compute_mask=True):
    idx = _get_src_permutation_idx(matching)

    target_classes = torch.zeros(shape, dtype=torch.long, device=device)
    target_classes[idx] = torch.cat([t[J] for t, (_, J) in zip(target, matching)])

    return target_classes


@torch.no_grad()
def match_anchor(anchor, matching, shape, device):
    target, _ = anchor

    idx = _get_src_permutation_idx(matching)
    target_classes = torch.zeros(shape, dtype=torch.long, device=device)
    target_classes[idx] = torch.cat([t[J, :] for t, (_, J) in zip(target, matching)])

    matched_mask = torch.ones(shape[:2], dtype=torch.bool, device=device)
    matched_mask[idx] = False

    return target_classes, matched_mask


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


@torch.no_grad()
def get_matching(cost_matrices):
    output = []
    for cost_matrix in cost_matrices:
        indices = linear_sum_assignment(cost_matrix, maximize=True)
        #indices = linear_sum_assignment_with_inf(cost_matrix)
        indices = (torch.tensor(indices[0], dtype=torch.long), torch.tensor(indices[1], dtype=torch.long))
        output.append(indices)

    return output

def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        min_values = values.min()
        max_values = values.max()
        m = min(cost_matrix.shape)

        positive = m * (max_values - min_values + np.abs(max_values) + np.abs(min_values) + 1)
        if max_inf:
            place_holder = (max_values + (m - 1) * (max_values - min_values)) + positive
        elif min_inf:
            place_holder = (min_values + (m - 1) * (min_values - max_values)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder
    return linear_sum_assignment(cost_matrix, maximize=True)


def sort_by_target(matchings):
    new_matching = []
    for matching in matchings:
        source, target = matching
        target, indices = target.sort()
        source = source[indices]
        new_matching.append((source, target))
    return new_matching


def reorder(hidden, matchings, max_length):
    batch_size, _, hidden_dim = hidden.shape
    matchings = sort_by_target(matchings)

    result = torch.zeros(batch_size, max_length, hidden_dim, device=hidden.device)
    for b in range(batch_size):
        indices = matchings[b][0]
        result[b, : len(indices), :] = hidden[b, indices, :]

    return result
