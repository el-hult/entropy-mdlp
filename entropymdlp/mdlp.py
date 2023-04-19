import numpy as np
from numba import njit


@njit(cache=True)
def entropy(variable):
    """
    Compute the Shannon entropy of an input variable with categorical values

    Assumes that all values the variable can take are present in the vector of data

    Example:
    >>> entropy(np.array([1]))
    0
    """
    assert variable.size > 0
    values = np.unique(variable)
    counts = np.zeros(len(values))
    for k, v in enumerate(values):
        counts[k] = (variable == v).sum()
    N = len(variable)
    ent = np.log2(N) - (counts @ np.log2(counts)) / N
    return ent


@njit(cache=True)
def should_partition(cut_idx, yo, ent):
    """Should we partition yo at cut_idx?

    The acceptance criterion in MDLP partitioning.

    Args:
        cut_idx, an index into yo
        yo a vector of class indices

    Notes:
        If the gain in further splitting, i.e. the decrease in entropy
        is too small, the algorithm will return -1,

    Returns:
        +1 if we should partition
        0 if we should not
    """
    n = len(yo)
    yo_entropy = entropy(yo)
    gain = yo_entropy - ent

    k = len(np.unique(yo))
    k1 = len(np.unique(yo[:cut_idx]))
    k2 = len(np.unique(yo[cut_idx:]))

    delta = np.log2(3 ** k - 2) - (
        k * yo_entropy - k1 * entropy(yo[:cut_idx]) - k2 * entropy(yo[cut_idx:])
    )
    cond = np.log2(n - 1) / n + delta / n
    if gain >= cond:
        return +1
    else:
        return 0


@njit(cache=True)
def find_cut_index(yo, cut_point_candidates):
    """Find best cutting point for binary partition of yo

    Args:
        cut_point_candidates: potential indices into yo where
            it makes sense to make a cut

    Returns:
        (cut_index,current_entropy) if there is a cut point that improves the measure
            cut_index is an index into cut_point_candidates
        (-1,0) if no split improves the target function

    """
    n = len(yo)
    current_entropy = np.inf
    cut_index = -1
    for cut_candidate_index, cut_point_candidate in enumerate(cut_point_candidates):
        weight_cutx = cut_point_candidate / n
        left_entropy = weight_cutx * entropy(yo[:cut_point_candidate])
        right_entropy = (1 - weight_cutx) * entropy(yo[cut_point_candidate:])
        temp = left_entropy + right_entropy
        if temp < current_entropy:
            current_entropy = temp
            cut_index = cut_candidate_index
    if cut_index == -1:
        return (-1, 0)
    else:
        return (cut_index, current_entropy)


@njit(cache=True)
def next_cut(yo, cut_point_candidates):
    """Helper to `part`

    Args
        cut_point_candidates: indices into xo where we might take cuts

    Returns
        ci such that there is a cut point at cut_point_candidates[ci] that improves the score
        -1 if no good cut exists"""
    cut_index, current_entropy = find_cut_index(yo, cut_point_candidates)
    if cut_index == -1:
        return -1

    cut_point = cut_point_candidates[cut_index]
    if should_partition(cut_point, yo, current_entropy):
        return cut_index
    else:
        return -1


def partition(yo, cut_point_candidates):
    """Helper to `cut_points` that is recursivel acting on partitions

    Args:
        cut_point_candidates is a list of indices into yo for potential cut points

    Returns:
        integer valued array with indices IN SORTED ORDER
    """
    empty = np.array([], dtype=cut_point_candidates.dtype)
    if cut_point_candidates.size == 0:
        return empty

    ci = next_cut(yo=yo, cut_point_candidates=cut_point_candidates)
    if ci == -1:
        return empty

    cut_index = cut_point_candidates[ci]
    left_cuts = partition(
        yo=yo[:cut_index], cut_point_candidates=cut_point_candidates[:ci]
    )
    right_cuts = (
        partition(
            yo=yo[cut_index:],
            cut_point_candidates=cut_point_candidates[ci + 1 :] - cut_index,
        )
        + cut_index
    )
    out = np.concatenate(
        (left_cuts, [cut_index], right_cuts), dtype=cut_point_candidates.dtype
    )
    return out


def cut_points(x, y):
    """
    Main function for the MDLP algorithm. A feature vector x
    and a target vector y are given as input, the algorithm
    computes a list of cut-values used for binning the variable x.
    """
    assert x.ndim == 1
    assert x.shape == y.shape
    sorted_index = np.argsort(x)
    xo = x[sorted_index]
    yo = y[sorted_index]

    cut_point_candidates = (
        np.argwhere((np.diff(xo) != 0) & (np.diff(yo) != 0))
    ).flatten() + 1  #

    cut_indices = partition(yo=yo, cut_point_candidates=cut_point_candidates)
    cut_values = (xo[cut_indices - 1] + xo[cut_indices]) / 2.0
    return cut_values
