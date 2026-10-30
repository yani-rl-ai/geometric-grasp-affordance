#!/usr/bin/env python3
"""
DBL-GNG: batch Growing Neural Gas.

Ported from the user's existing grasp_ws implementation
(grasp_ws/src/camera_grasp/scripts/dbl_gng_segmentor.py), stripped of its
ROS/grasp-candidate machinery -- just the core topology-learning algorithm,
used here to build a node/edge graph over each segmented object's 3D points.

Unlike textbook GNG (one input point processed per update), this variant
processes ALL points together each epoch: the winner node and its neighbors
move toward the mean of the points currently closest to that winner, and
edge ages increment once per epoch rather than once per point. Both choices
exist to make repeated re-fits (this gets refit from scratch on new point
sets each time, since node ids aren't tracked across frames) land on nearly
the same topology when the input points are nearly the same -- an online,
per-point GNG refit from scratch every frame visibly "jumps" between calls
even for a near-static scene.
"""
import numpy as np


class DBLGrowingNeuralGas:
    def __init__(self, max_nodes=30, n_epochs=120, eps_b=0.1, eps_n=0.006,
                 max_age=30, insert_every=4, alpha=0.5, beta=0.995):
        self.max_nodes = max_nodes
        self.n_epochs = n_epochs
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.max_age = max_age
        self.insert_every = insert_every
        self.alpha = alpha
        self.beta = beta

    def fit(self, points):
        """points: (N,3) float array. Returns (positions, edges).

        positions -- dict {node_id: np.ndarray(3,)}
        edges     -- dict {frozenset({id_a, id_b}): age}
        """
        n = len(points)
        if n < 2:
            return {}, {}

        # Deterministic init (not 2 random points): the FIRST point (image
        # scan order, stable across frames for a mostly-static scene) and
        # the point FARTHEST from it. Random seeding would make the fitted
        # topology jump between calls even when the input points are nearly
        # identical.
        idx0 = 0
        d2_from_p0 = np.sum((points - points[idx0][None, :]) ** 2, axis=1)
        idx1 = int(np.argmax(d2_from_p0))
        positions = {
            0: points[idx0].astype(np.float64).copy(),
            1: points[idx1].astype(np.float64).copy(),
        }
        errors = {0: 0.0, 1: 0.0}
        edges = {frozenset({0, 1}): 0}
        next_id = 2

        for epoch in range(self.n_epochs):
            node_ids = np.fromiter(positions.keys(), dtype=int)
            pos_arr = np.stack([positions[i] for i in node_ids])
            k = len(node_ids)

            diff = points[:, None, :] - pos_arr[None, :, :]
            d2 = np.einsum('nkd,nkd->nk', diff, diff)

            if k >= 2:
                part = np.argpartition(d2, 1, axis=1)[:, :2]
                rows = np.arange(n)
                first_smaller = d2[rows, part[:, 0]] <= d2[rows, part[:, 1]]
                s1 = np.where(first_smaller, part[:, 0], part[:, 1])
                s2 = np.where(first_smaller, part[:, 1], part[:, 0])
            else:
                s1 = np.zeros(n, dtype=int)
                s2 = s1

            sums = np.zeros((k, 3))
            counts = np.zeros(k)
            np.add.at(sums, s1, points)
            np.add.at(counts, s1, 1)
            err_add = np.zeros(k)
            np.add.at(err_add, s1, d2[np.arange(n), s1])

            neighbor_map = {int(i): set() for i in node_ids}
            for e in edges:
                a, b = tuple(e)
                if a in neighbor_map:
                    neighbor_map[a].add(b)
                if b in neighbor_map:
                    neighbor_map[b].add(a)

            new_positions = dict(positions)
            for i, nid in enumerate(node_ids):
                nid = int(nid)
                errors[nid] = errors.get(nid, 0.0) + float(err_add[i])
                if counts[i] == 0:
                    continue
                mean_pt = sums[i] / counts[i]
                new_positions[nid] = positions[nid] + self.eps_b * (mean_pt - positions[nid])
                for nb in neighbor_map.get(nid, ()):
                    new_positions[nb] = new_positions.get(nb, positions[nb]) \
                        + self.eps_n * (mean_pt - new_positions.get(nb, positions[nb]))
            positions = new_positions

            for e in list(edges.keys()):
                edges[e] += 1
            if k >= 2:
                pairs = np.unique(np.stack([s1, s2], axis=1), axis=0)
                for a, b in pairs:
                    a, b = int(node_ids[a]), int(node_ids[b])
                    if a != b:
                        edges[frozenset({a, b})] = 0

            for e in [e for e, age in edges.items() if age > self.max_age]:
                del edges[e]

            if len(positions) > 2:
                connected = set()
                for e in edges:
                    connected.update(e)
                for nid in [i for i in positions if i not in connected]:
                    if len(positions) <= 2:
                        break
                    del positions[nid]
                    errors.pop(nid, None)

            if (epoch + 1) % self.insert_every == 0 and len(positions) < self.max_nodes and errors:
                candidates = {nid: e for nid, e in errors.items() if nid in positions}
                if candidates:
                    q = max(candidates, key=candidates.get)
                    nbrs = set()
                    for e in edges:
                        a, b = tuple(e)
                        if a == q:
                            nbrs.add(b)
                        elif b == q:
                            nbrs.add(a)
                    if nbrs:
                        f = max(nbrs, key=lambda x: errors.get(x, 0.0))
                        r = next_id
                        next_id += 1
                        positions[r] = 0.5 * (positions[q] + positions[f])
                        edges.pop(frozenset({q, f}), None)
                        edges[frozenset({q, r})] = 0
                        edges[frozenset({r, f})] = 0
                        errors[q] *= self.alpha
                        errors[f] *= self.alpha
                        errors[r] = errors[q]

            for nid in errors:
                errors[nid] *= self.beta

        return positions, edges


def deterministic_subsample(n, max_n):
    """Evenly-strided indices (not random) so the same-ish input points pick
    the same-ish subset across frames -- see DBLGrowingNeuralGas docstring."""
    if n <= max_n:
        return np.arange(n)
    return np.linspace(0, n - 1, max_n).round().astype(np.int64)
