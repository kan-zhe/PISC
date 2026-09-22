"""DTIC: Distance-Threshold Iterative Clustering.

Core prototype-selection algorithm of PISC. Given an overcomplete set of
KMeans pseudo-prototypes, DTIC groups nearby prototypes under a distance
threshold (greedy clique cover) and retains at most two representatives per
group.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Set, Tuple

import numpy as np
from sklearn.cluster import KMeans

PrototypePair = Tuple[int, int, float]


def initialize_kmeans_prototypes(
    data: np.ndarray, n_prototypes: int, seed: int = 42
) -> np.ndarray:
    """Use KMeans centroids as the initial pseudo-prototype candidates."""
    kmeans = KMeans(n_clusters=n_prototypes, n_init=10, random_state=seed)
    kmeans.fit(data)
    return kmeans.cluster_centers_


def find_close_prototype_pairs(
    prototypes: np.ndarray, distance_threshold: float
) -> List[PrototypePair]:
    """Return prototype index pairs whose Euclidean distance is below the threshold."""
    close_pairs: List[PrototypePair] = []
    for left in range(len(prototypes) - 1):
        for right in range(left + 1, len(prototypes)):
            distance = float(np.linalg.norm(prototypes[left] - prototypes[right]))
            if distance < distance_threshold:
                close_pairs.append((left, right, distance))
    return close_pairs


def _build_edge_lookup(close_pairs: Sequence[PrototypePair]) -> Set[Tuple[int, int]]:
    """Build an undirected edge set for constant-time pair lookup."""
    return {(min(l, r), max(l, r)) for l, r, _ in close_pairs}


def _is_connected_to_group(
    candidate: int, group: Sequence[int], edge_lookup: Set[Tuple[int, int]]
) -> bool:
    """Check whether a candidate is close to every prototype already in the group."""
    return all(
        (min(candidate, g), max(candidate, g)) in edge_lookup for g in group
    )


def _expand_prototype_group(
    initial_group: Sequence[int], close_pairs: Sequence[PrototypePair]
) -> List[int]:
    """Greedily expand a group into a clique (each member close to all others)."""
    group = list(initial_group)
    edge_lookup = _build_edge_lookup(close_pairs)
    while True:
        added = False
        for left, right, _ in close_pairs:
            for candidate in (left, right):
                if candidate in group:
                    continue
                if _is_connected_to_group(candidate, group, edge_lookup):
                    group.append(candidate)
                    added = True
                    break
            if added:
                break
        if not added:
            return sorted(group)


def group_similar_prototypes(
    prototypes: np.ndarray, distance_threshold: float, seed: int = 42
) -> List[List[int]]:
    """Group close prototypes with a greedy clique-cover heuristic."""
    close_pairs = find_close_prototype_pairs(prototypes, distance_threshold)
    if not close_pairs:
        return [[i] for i in range(len(prototypes))]

    rng = random.Random(seed)
    remaining_pairs = close_pairs[:]
    initial_idx = rng.randrange(len(remaining_pairs))
    current_group = [remaining_pairs[initial_idx][0], remaining_pairs[initial_idx][1]]

    prototype_groups: List[List[int]] = []
    while remaining_pairs:
        expanded = _expand_prototype_group(current_group, remaining_pairs)
        prototype_groups.append(expanded)
        members = set(expanded)
        remaining_pairs = [
            p for p in remaining_pairs if p[0] not in members and p[1] not in members
        ]
        if remaining_pairs:
            current_group = [remaining_pairs[0][0], remaining_pairs[0][1]]

    grouped = {i for group in prototype_groups for i in group}
    prototype_groups.extend([i] for i in range(len(prototypes)) if i not in grouped)
    return prototype_groups


def _select_closest_pair(
    prototypes: np.ndarray, group: Sequence[int]
) -> Tuple[int, int]:
    """Return the two indices in a group that are closest together."""
    best_pair = (group[0], group[1])
    best_distance = float("inf")
    for i in range(len(group) - 1):
        for j in range(i + 1, len(group)):
            d = float(np.linalg.norm(prototypes[group[i]] - prototypes[group[j]]))
            if d < best_distance:
                best_distance = d
                best_pair = (group[i], group[j])
    return best_pair


def _select_farthest_pair(
    prototypes: np.ndarray, group: Sequence[int]
) -> Tuple[int, int]:
    """Return the two indices in a group that are farthest apart."""
    best_pair = (group[0], group[1])
    best_distance = -1.0
    for i in range(len(group) - 1):
        for j in range(i + 1, len(group)):
            d = float(np.linalg.norm(prototypes[group[i]] - prototypes[group[j]]))
            if d > best_distance:
                best_distance = d
                best_pair = (group[i], group[j])
    return best_pair


def select_representative_prototypes(
    prototypes: np.ndarray,
    prototype_groups: Sequence[Sequence[int]],
    max_representatives_per_group: int = 2,
    strategy: str = "closest",
) -> Tuple[np.ndarray, List[int]]:
    """Select representatives from each threshold-consistent group.

    Groups larger than ``max_representatives_per_group`` keep the closest (or
    farthest) pair; smaller groups keep all members.
    """
    selected: List[int] = []
    for group in prototype_groups:
        group = list(group)
        if len(group) > max_representatives_per_group:
            if strategy == "farthest":
                selected.extend(_select_farthest_pair(prototypes, group))
            else:
                selected.extend(_select_closest_pair(prototypes, group))
        else:
            selected.extend(group)
    selected = sorted(selected)
    return prototypes[selected], selected
