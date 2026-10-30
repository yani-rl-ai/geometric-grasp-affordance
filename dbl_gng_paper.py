#!/usr/bin/env python3
"""
DBL-GNG following Siow, Saputra, Obo, Kubota, "Distributed Batch Learning of
Growing Neural Gas for Quick and Efficient Clustering", Mathematics 2024,
12(12), 1909 (Algorithms 1-6, Eq. 19-41, Sec. 4.2 defaults).

Differences from dbl_gng.py (the earlier implementation, now kept only as the
"legacy" baseline for ablation): that one was a batch-update GNG with 2-node
first+farthest initialization, one node inserted per 4 epochs and edge aging.
This one has the paper's four ingredients:

  1. Distributed initialization (Alg. 5): eta start points spread over the
     data (random point from the farthest remaining batch + its 3rd-nearest
     neighbour, then drop the nearest batch).
  2. Matrix batch learning (Alg. 1, Eq. 19-27): distance matrix, winner /
     second winner for every datum at once, accumulated delta movement and
     activation counts, per-epoch average update (Eq. 28). No edge age: edges
     are rebuilt every epoch from the winner pairs (Eq. 29), with edge strength
     S accumulated for the final cut.
  3. Multi-node growth (Eq. 34, Alg. 3): g = #nodes whose error exceeds the
     p-th percentile of node errors, all inserted in the same epoch.
  4. Edge cutting (Eq. 33, Alg. 4): edges with strength <= the percentile of
     the positive strengths are removed, then isolated nodes are removed.

Defaults follow Sec. 4.2 of the paper: winner rate 0.5, neighbor rate 0.01,
rho = delta = 0.5, growth percentile 0.85, cut percentile 0.15, eps = 1e-4,
10% chance per epoch of removing inactive nodes, stop at max nodes + 10 more
epochs (Sec. 4.3).

Documented implementation choices where the paper is silent / ambiguous:
  - eta (number of start points) is a free parameter; the paper sets it to the
    cluster count of each benchmark. Here a single object surface is one
    cluster, so eta is a tuned hyper-parameter (default 4).
  - the random start point uses a seeded numpy Generator so a trial is
    reproducible from the logged seed.
  - nodes created in the last epoch (after the final batch step) have no
    activation count yet; they are exempt from the final inactive-node removal.
  - safety cap on total epochs (max_epochs) in case deletions keep the node
    count below the maximum.
"""
import numpy as np


class PaperDBLGNG:
    def __init__(self, max_nodes=30, n_start=4, alpha=0.5, beta=0.01,
                 rho=0.5, delta=0.5, grow_pct=85.0, cut_pct=15.0,
                 extra_epochs=10, max_epochs=200, inactive_prob=0.1,
                 eps=1e-4, seed=0, do_cut=True):
        self.max_nodes = max_nodes
        self.n_start = n_start
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.delta = delta
        self.grow_pct = grow_pct
        self.cut_pct = cut_pct
        self.extra_epochs = extra_epochs
        self.max_epochs = max_epochs
        self.inactive_prob = inactive_prob
        self.eps = eps
        self.seed = seed
        self.do_cut = do_cut
        self.last_info = {}

    # ---------------------------------------------------------------- init
    def _init_nodes(self, X, rng):
        """Alg. 5 / Eq. 35-41. Returns (W (m,F), C (m,m) bool)."""
        D = len(X)
        eta = max(1, min(self.n_start, D // 3))
        B = max(3, D // eta)
        Xr = X
        nodes = []
        pairs = []
        for _ in range(eta):
            if len(Xr) < 3:
                break
            lo = max(0, len(Xr) - B)
            i = int(rng.integers(lo, len(Xr)))          # Eq. 36
            w = Xr[i]
            d = (Xr ** 2).sum(1) - 2.0 * (Xr @ w)       # Eq. 37 (ordering only)
            order = np.argsort(d, kind="stable")
            Xs = Xr[order]                              # Eq. 38
            a = len(nodes)
            nodes.append(w.copy())
            nodes.append(Xs[2].copy())                  # Eq. 39: 3rd closest
            pairs.append((a, a + 1))                    # Eq. 40
            Xr = Xs[B:]                                 # Eq. 41
        W = np.asarray(nodes, dtype=np.float64)
        m = len(W)
        C = np.zeros((m, m), dtype=bool)
        for a, b in pairs:
            C[a, b] = C[b, a] = True
        return W, C

    # ------------------------------------------------------------- helpers
    def _grow(self, W, E, C, S, A1):
        """Eq. 34 + Alg. 3 (Eq. 13-16, 18, 32), g nodes in one epoch."""
        m = len(W)
        if m >= self.max_nodes or m < 2:
            return W, E, C, S, A1
        g = int(np.sum(E > np.percentile(E, self.grow_pct)))
        for _ in range(g):
            m = len(W)
            if m >= self.max_nodes:
                break
            has_nb = C.any(axis=1)
            if not has_nb.any():
                break
            q1 = int(np.argmax(np.where(has_nb, E, -np.inf)))      # Eq. 13
            nbrs = np.nonzero(C[q1])[0]
            q2 = int(nbrs[np.argmax(E[nbrs])])                      # Eq. 14
            w3 = 0.5 * (W[q1] + W[q2])                              # Eq. 15
            W = np.vstack([W, w3])
            C = np.pad(C, ((0, 1), (0, 1)))
            S = np.pad(S, ((0, 1), (0, 1)))
            q3 = m
            C[q1, q2] = C[q2, q1] = False                           # Eq. 16
            S[q1, q2] = S[q2, q1] = 0.0
            for q in (q1, q2):
                C[q, q3] = C[q3, q] = True
                S[q, q3] = S[q3, q] = 1.0                           # Eq. 32
            E = E.copy()
            E[q1] *= self.rho                                       # Eq. 18
            E[q2] *= self.rho
            E = np.append(E, 0.5 * (E[q1] + E[q2]))
            A1 = np.append(A1, 1.0)   # fresh node counts as active
        return W, E, C, S, A1

    # ----------------------------------------------------------------- fit
    def fit(self, points):
        """points: (N,F) array. Returns (positions, edges):
        positions {node_id: ndarray(F,)}, edges {frozenset({a,b}): strength}."""
        X = np.asarray(points, dtype=np.float64)
        n = len(X)
        if n < 2:
            return {}, {}
        rng = np.random.default_rng(self.seed)
        W, C = self._init_nodes(X, rng)
        if len(W) < 2:   # degenerate (n < 3): fall back to the two points
            W = X[:2].copy()
            C = np.array([[False, True], [True, False]])
        m = len(W)
        E = np.zeros(m)
        S = C.astype(np.float64)
        A1 = np.ones(m)
        x2 = (X ** 2).sum(1)
        rows = np.arange(n)
        epochs = 0
        extra_left = None

        while epochs < self.max_epochs:
            epochs += 1
            m = len(W)
            # ---- Alg. 1: batch learning (Eq. 19-27)
            d2 = np.maximum(x2[:, None] - 2.0 * (X @ W.T) + (W ** 2).sum(1)[None, :], 0.0)
            s1 = d2.argmin(axis=1)
            d2m = d2.copy()
            d2m[rows, s1] = np.inf
            s2 = d2m.argmin(axis=1)
            d_s1 = np.sqrt(d2[rows, s1])
            E = E + np.bincount(s1, weights=d_s1, minlength=m) * self.alpha       # Eq. 22
            A1 = np.bincount(s1, minlength=m).astype(np.float64)                   # Eq. 25
            sumX = np.zeros_like(W)
            np.add.at(sumX, s1, X)
            dW1 = (sumX - A1[:, None] * W) * self.alpha                            # Eq. 23
            Cs = C[s1].astype(np.float64)                                          # n x m
            A2 = Cs.sum(axis=0)                                                    # Eq. 26
            dW2 = (Cs.T @ X - A2[:, None] * W) * self.beta                         # Eq. 24
            S = np.zeros((m, m))
            np.add.at(S, (s1, s2), 1.0)                                            # Eq. 27
            S = S + S.T

            # ---- Alg. 2: network update
            W = W + dW1 / (A1 + self.eps)[:, None] + dW2 / (A2 + self.eps)[:, None]  # Eq. 28
            C = S > 0                                                              # Eq. 29
            keep = C.any(axis=1)                                                   # Eq. 10
            if keep.sum() < 2:
                keep = np.ones(m, dtype=bool)
            W, E, A1 = W[keep], E[keep], A1[keep]
            C, S = C[np.ix_(keep, keep)], S[np.ix_(keep, keep)]                    # Eq. 30
            E = E * self.delta                                                     # Eq. 11
            if rng.random() > 1.0 - self.inactive_prob:                            # 10 % chance
                keep = A1 > 0                                                      # Eq. 31
                if keep.sum() >= 2:
                    W, E, A1 = W[keep], E[keep], A1[keep]
                    C, S = C[np.ix_(keep, keep)], S[np.ix_(keep, keep)]

            # ---- growth (Eq. 34, Alg. 3), then stopping rule (Sec. 4.3)
            W, E, C, S, A1 = self._grow(W, E, C, S, A1)
            if extra_left is None:
                if len(W) >= self.max_nodes:
                    extra_left = self.extra_epochs
            else:
                extra_left -= 1
            if extra_left is not None and extra_left <= 0:
                break

        nodes_pre_cut = len(W)
        # ---- Alg. 4: edge cutting
        if self.do_cut:
            keep = A1 > 0
            if keep.sum() >= 2:
                W, E, A1 = W[keep], E[keep], A1[keep]
                C, S = C[np.ix_(keep, keep)], S[np.ix_(keep, keep)]
            pos = S[np.triu_indices(len(W), 1)]
            pos = pos[pos > 0]
            if len(pos):
                thr = np.percentile(pos, self.cut_pct)                             # Eq. 33
                C = C & (S > thr)
            keep = C.any(axis=1)
            if keep.sum() >= 2:
                W = W[keep]
                C, S = C[np.ix_(keep, keep)], S[np.ix_(keep, keep)]

        self.last_info = dict(epochs=epochs, nodes_pre_cut=nodes_pre_cut,
                              nodes=len(W), seed=self.seed)
        positions = {i: W[i].copy() for i in range(len(W))}
        edges = {}
        for a, b in zip(*np.nonzero(np.triu(C, 1))):
            edges[frozenset({int(a), int(b)})] = float(S[a, b])
        return positions, edges
