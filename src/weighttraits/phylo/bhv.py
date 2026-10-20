"""BHV tree-space geometry: geodesics, Frechet (Sturm) mean, log map, confidence sets.

Ported unchanged in numerics from ELLMTrees ``analysis/reconstruction/tree_space.py``, which
produced the paper's all-weight estimates (Table 2) and the documented-family estimates.

Implements:

  * Owen & Provan (2011), "A fast algorithm for computing geodesic distances in
    tree space" -- the GTP geodesic via iterated min-weight vertex cover.
  * Sturm (2003) / Bacak (2014) -- the proximal-point Frechet ("average") tree.
  * Barden, Le & Owen (2018) -- the modified log map T_{m+3} -> R^m.
  * Willis (2019, JASA) "Confidence Sets for Phylogenetic Trees" -- the F-based
    confidence set / hypothesis test (Eq. 4); and
    Willis & Bell (2018, JCGS) "Uncertainty in Phylogenetic Tree Estimates" --
    the weighted Frechet mean and log-map projection conventions.

Trees are treated as UNROOTED with a common leaf set (the requirement for BHV).
Internal edges are bipartitions of the leaf set (length > 0); pendant edges are
per-leaf. Requires numpy, networkx (min-cut), and dendropy (Newick parsing and
split encoding); scipy is used for the F distribution when present.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# F distribution (no scipy dependency)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_cdf(x: float, d1: float, d2: float) -> float:
    """CDF of the F(d1, d2) distribution."""
    try:
        from scipy.stats import f as _f
        return float(_f.cdf(x, d1, d2))
    except Exception:
        if x <= 0:
            return 0.0
        return _betai(d1 / 2.0, d2 / 2.0, (d1 * x) / (d1 * x + d2))


def f_ppf(q: float, d1: float, d2: float) -> float:
    """Inverse CDF (quantile) of F(d1, d2) via bisection."""
    try:
        from scipy.stats import f as _f
        return float(_f.ppf(q, d1, d2))
    except Exception:
        lo, hi = 0.0, 1.0
        while f_cdf(hi, d1, d2) < q and hi < 1e12:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f_cdf(mid, d1, d2) < q:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Tree representation
# ---------------------------------------------------------------------------

# An internal split is keyed by a frozenset of leaf names: the side of the
# bipartition that does NOT contain the reference leaf (leaves[0]). This makes
# the key canonical and identical across trees sharing a leaf set.


@dataclass
class BHVTree:
    """An unrooted phylogenetic tree as a point in BHV space."""

    leaves: Tuple[str, ...]                       # sorted, canonical leaf order
    internal: Dict[frozenset, float] = field(default_factory=dict)  # split -> length
    pendant: Dict[str, float] = field(default_factory=dict)         # leaf  -> length

    # -- construction -------------------------------------------------------
    @property
    def leafset(self) -> frozenset:
        return frozenset(self.leaves)

    @property
    def ref(self) -> str:
        return self.leaves[0]

    def canonical(self, side: frozenset) -> frozenset:
        """Return the canonical representative of a bipartition `side|rest`."""
        if self.ref in side:
            return self.leafset - side
        return frozenset(side)

    def copy(self) -> "BHVTree":
        return BHVTree(self.leaves, dict(self.internal), dict(self.pendant))

    # -- diagnostics --------------------------------------------------------
    def splits(self) -> List[frozenset]:
        return list(self.internal.keys())

    def is_valid(self) -> bool:
        """True iff all internal splits are pairwise compatible (a real tree)."""
        ks = list(self.internal)
        L = self.leafset
        return all(
            compatible(ks[i], ks[j], L)
            for i in range(len(ks))
            for j in range(i + 1, len(ks))
        )


def split_sort_key(split: frozenset) -> Tuple[int, Tuple[str, ...]]:
    """Deterministic order for splits: by size, then by sorted leaf names."""
    return len(split), tuple(sorted(split))


def compatible(a: frozenset, b: frozenset, L: frozenset) -> bool:
    """Two bipartitions a|(L-a), b|(L-b) are compatible iff one of the four
    pairwise intersections is empty."""
    a2, b2 = L - a, L - b
    return not (a & b) or not (a & b2) or not (a2 & b) or not (a2 & b2)


# ---------------------------------------------------------------------------
# Newick / dendropy I/O
# ---------------------------------------------------------------------------

def from_newick(newick: str, taxon_namespace=None) -> BHVTree:
    """Parse a Newick string into a BHVTree (unrooted)."""
    import dendropy

    t = dendropy.Tree.get(
        data=newick, schema="newick", taxon_namespace=taxon_namespace,
        preserve_underscores=True, suppress_internal_node_taxa=True,
    )
    return from_dendropy(t)


def from_dendropy(t) -> BHVTree:
    t.is_rooted = False
    t.encode_bipartitions(suppress_unifurcations=True)
    leaves = tuple(sorted(lf.taxon.label for lf in t.leaf_node_iter()))
    leafset = frozenset(leaves)
    ref = leaves[0]
    label_for = {}  # bitmask -> leaf label, for pendant edges
    for lf in t.leaf_node_iter():
        label_for[lf.bipartition.leafset_bitmask] = lf.taxon.label

    internal: Dict[frozenset, float] = {}
    pendant: Dict[str, float] = {}
    # map each taxon to its bitmask bit
    bit_of = {lf.taxon.label: lf.bipartition.leafset_bitmask for lf in t.leaf_node_iter()}

    def mask_to_set(mask: int) -> frozenset:
        return frozenset(lab for lab, bit in bit_of.items() if mask & bit)

    for edge in t.preorder_edge_iter():
        if edge.head_node is None or edge.bipartition is None:
            continue
        length = float(edge.length) if edge.length is not None else 0.0
        side = mask_to_set(edge.bipartition.leafset_bitmask)
        if len(side) == 0 or len(side) == len(leaves):
            continue
        if len(side) == 1:                      # pendant edge
            (leaf,) = tuple(side)
            pendant[leaf] = length
        else:                                   # internal edge
            key = side if ref not in side else (leafset - side)
            if length > 0:
                internal[key] = internal.get(key, 0.0) + length
    # ensure every leaf has a pendant entry (0 if missing)
    for lf in leaves:
        pendant.setdefault(lf, 0.0)
    return BHVTree(leaves, internal, pendant)


def from_biopython(bp_tree, taxon_namespace=None) -> BHVTree:
    """Convert a Bio.Phylo tree to a BHVTree (via Newick round-trip)."""
    import io
    from Bio import Phylo

    from weighttraits.phylo.reconstruct import NEWICK_BRANCH_LENGTH_FORMAT

    buf = io.StringIO()
    Phylo.write(bp_tree, buf, "newick", format_branch_length=NEWICK_BRANCH_LENGTH_FORMAT)
    return from_newick(buf.getvalue().strip(), taxon_namespace=taxon_namespace)


def to_newick(tree: BHVTree) -> str:
    """Serialize a BHVTree back to Newick (builds a tree from its split set)."""
    import dendropy

    tns = dendropy.TaxonNamespace([dendropy.Taxon(label) for label in tree.leaves])
    taxon = {t.label: t for t in tns}
    star = dendropy.Tree(taxon_namespace=tns)
    for label in tree.leaves:
        star.seed_node.new_child(taxon=taxon[label], edge_length=tree.pendant.get(label, 0.0))

    # Insert internal edges SMALLEST-first, each under the lowest existing node
    # that already contains its leaf set. (Inserting largest-first and only
    # regrouping direct children of the seed silently dropped every nested
    # split, collapsing any resolved tree to a polytomy.)
    splits = sorted(tree.internal.items(), key=lambda kv: len(kv[0]))
    for side, length in splits:
        members = set(side)
        cand = star.seed_node
        moved = True
        while moved:                       # descend to lowest node containing members
            moved = False
            for c in cand.child_nodes():
                lu = leaves_under(c)
                if members <= lu and len(lu) < len(leaves_under(cand)):
                    cand = c
                    moved = True
                    break
        movable = [c for c in cand.child_nodes() if leaves_under(c) <= members]
        covered = set()
        for c in movable:
            covered |= leaves_under(c)
        if covered != members:
            # split incompatible with current partial topology; skip
            continue
        new_node = dendropy.Node(edge_length=length)
        for c in movable:
            cand.remove_child(c)
            new_node.add_child(c)
        cand.add_child(new_node)
    return star.as_string(schema="newick").strip()


def leaves_under(node) -> set:
    return {lf.taxon.label for lf in node.leaf_iter()}


# ---------------------------------------------------------------------------
# Owen-Provan geodesic (GTP via min-weight vertex cover)
# ---------------------------------------------------------------------------

@dataclass
class SupportPair:
    A: List[frozenset]      # splits in target (T) only -- "incoming"
    B: List[frozenset]      # splits in base   (S) only -- "outgoing"
    lenA: Dict[frozenset, float]
    lenB: Dict[frozenset, float]

    @property
    def normA(self) -> float:
        return math.sqrt(sum(self.lenA[a] ** 2 for a in self.A))

    @property
    def normB(self) -> float:
        return math.sqrt(sum(self.lenB[b] ** 2 for b in self.B))

    @property
    def ratio(self) -> float:
        nb = self.normB
        return self.normA / nb if nb > 0 else math.inf


@dataclass
class Geodesic:
    base: BHVTree                                   # S
    target: BHVTree                                 # T
    shared: List[Tuple[frozenset, float, float]]    # (key, lenS, lenT)
    pendant: List[Tuple[str, float, float]]         # (leaf, lenS, lenT)
    pairs: List[SupportPair]                         # ordered by ascending ratio

    def distance(self) -> float:
        d2 = 0.0
        for _, ls, lt in self.shared:
            d2 += (ls - lt) ** 2
        for _, ls, lt in self.pendant:
            d2 += (ls - lt) ** 2
        for p in self.pairs:
            d2 += (p.normA + p.normB) ** 2
        return math.sqrt(max(d2, 0.0))


def geodesic(S: BHVTree, T: BHVTree, tol: float = 1e-12) -> Geodesic:
    """Owen-Provan geodesic between trees S and T (common leaf set)."""
    import networkx as nx
    # networkx's default min-cut solver (preflow_push) crashes on some of our
    # incompatibility graphs with "min() arg is an empty sequence" (a node with
    # no residual out-edge during relabel). edmonds_karp is robust and agrees
    # with the other solvers on the graphs where preflow_push *does* succeed.
    from networkx.algorithms.flow import edmonds_karp

    assert S.leafset == T.leafset, "trees must share a leaf set"
    L = S.leafset

    shared_keys = set(S.internal) & set(T.internal)
    # Iterate shared splits in a fixed order. Set order follows Python's per-process string
    # hashing, which made branch lengths (and rare near-tie topologies) differ between runs.
    shared = [(k, S.internal[k], T.internal[k]) for k in sorted(shared_keys, key=split_sort_key)]
    pendant = [(lf, S.pendant.get(lf, 0.0), T.pendant.get(lf, 0.0)) for lf in S.leaves]

    Bset = [k for k in S.internal if k not in shared_keys]   # base-only (drop)
    Aset = [k for k in T.internal if k not in shared_keys]   # target-only (add)
    lenB = {k: S.internal[k] for k in Bset}
    lenA = {k: T.internal[k] for k in Aset}

    pairs: List[SupportPair] = []
    if Aset and Bset:
        # iterative splitting starting from the cone pair (A, B)
        stack = [(list(Aset), list(Bset))]
        final: List[Tuple[List[frozenset], List[frozenset]]] = []
        while stack:
            A, B = stack.pop()
            normA2 = sum(lenA[a] ** 2 for a in A)
            normB2 = sum(lenB[b] ** 2 for b in B)
            if normA2 <= tol or normB2 <= tol:
                final.append((A, B))
                continue
            # build incompatibility bipartite flow graph
            G = nx.DiGraph()
            SRC, SNK = "##s", "##t"
            for a in A:
                G.add_edge(SRC, ("A", a), capacity=lenA[a] ** 2 / normA2)
            for b in B:
                G.add_edge(("B", b), SNK, capacity=lenB[b] ** 2 / normB2)
            n_edges = 0
            for a in A:
                for b in B:
                    if not compatible(a, b, L):
                        G.add_edge(("A", a), ("B", b), capacity=float("inf"))
                        n_edges += 1
            if n_edges == 0:
                # A fully compatible with B -> two independent pairs, but the
                # cone here has cover weight 0 < 1: split into singletons set.
                # Mincut handles it; fall through to cut logic.
                pass
            cut_value, (reach, _) = nx.minimum_cut(G, SRC, SNK,
                                                    flow_func=edmonds_karp)
            # min vertex cover weight = cut_value. Split iff < 1 (strictly).
            if cut_value >= 1.0 - 1e-9:
                final.append((A, B))
                continue
            # cover = A-vertices whose src-edge is cut (NOT in reach) +
            #         B-vertices whose snk-edge is cut (in reach)
            CA = [a for a in A if ("A", a) not in reach]
            CB = [b for b in B if ("B", b) in reach]
            A1 = [a for a in A if a not in set(CA)]   # A \ CA
            B1 = list(CB)                              # CB
            A2 = list(CA)                              # CA
            B2 = [b for b in B if b not in set(CB)]    # B \ CB
            # guard against non-progress
            if (not A1 and not B1) or (not A2 and not B2):
                final.append((A, B))
                continue
            stack.append((A2, B2))
            stack.append((A1, B1))
        for A, B in final:
            if A and B:
                pairs.append(SupportPair(A, B, {a: lenA[a] for a in A},
                                         {b: lenB[b] for b in B}))
            elif B:               # only outgoing -> they just shrink to 0
                for b in B:
                    pairs.append(SupportPair([], [b], {}, {b: lenB[b]}))
            elif A:               # only incoming -> grow from 0
                for a in A:
                    pairs.append(SupportPair([a], [], {a: lenA[a]}, {}))
    else:
        # one side empty: each leftover edge is its own degenerate pair
        for b in Bset:
            pairs.append(SupportPair([], [b], {}, {b: lenB[b]}))
        for a in Aset:
            pairs.append(SupportPair([a], [], {a: lenA[a]}, {}))

    pairs.sort(key=lambda p: p.ratio)
    return Geodesic(S, T, shared, pendant, pairs)


def geodesic_distance(S: BHVTree, T: BHVTree) -> float:
    return geodesic(S, T).distance()


# ---------------------------------------------------------------------------
# Geodesic interpolation  (point at fraction lambda from base toward target)
# ---------------------------------------------------------------------------

def geodesic_point(S: BHVTree, T: BHVTree, lam: float,
                   geo: Optional[Geodesic] = None) -> BHVTree:
    """Tree at fraction `lam` in [0,1] along the geodesic from S to T."""
    if lam <= 0:
        return S.copy()
    if lam >= 1:
        return T.copy()
    g = geo or geodesic(S, T)
    internal: Dict[frozenset, float] = {}
    for k, ls, lt in g.shared:
        v = (1 - lam) * ls + lam * lt
        if v > 0:
            internal[k] = v
    for p in g.pairs:
        nA, nB = p.normA, p.normB
        denom = nA + nB
        t_i = nB / denom if denom > 0 else 0.0
        if lam <= t_i:                      # outgoing (B) side still present
            if nB > 0:
                scale = ((1 - lam) * nB - lam * nA) / nB
                for b in p.B:
                    v = scale * p.lenB[b]
                    if v > 0:
                        internal[b] = v
        else:                               # incoming (A) side present
            if nA > 0:
                scale = (lam * nA - (1 - lam) * nB) / nA
                for a in p.A:
                    v = scale * p.lenA[a]
                    if v > 0:
                        internal[a] = v
    pendant = {lf: (1 - lam) * ls + lam * lt for lf, ls, lt in g.pendant}
    return BHVTree(S.leaves, internal, pendant)


# ---------------------------------------------------------------------------
# Sturm / Bacak Frechet mean  (the "average tree")
# ---------------------------------------------------------------------------

def frechet_mean(trees: Sequence[BHVTree], weights: Optional[Sequence[float]] = None,
                 n_epochs: int = 30, seed: int = 0,
                 tol: float = 1e-9) -> BHVTree:
    """Proximal-point (Sturm) estimate of the BHV Frechet mean.

    Iterates  s_{k+1} = geodesic_point(s_k, T_{i_k}, w_{i_k}/(W_k))  with a
    decreasing step, cycling random shuffles of the trees over `n_epochs`.
    Converges to the unique Frechet mean (CAT(0) guarantee, Sturm 2003)."""
    trees = list(trees)
    n = len(trees)
    if n == 0:
        raise ValueError("no trees")
    if n == 1:
        return trees[0].copy()
    if weights is None:
        weights = [1.0] * n
    weights = list(weights)

    rng = random.Random(seed)
    s = trees[0].copy()
    k = 0
    wsum = 0.0
    prev = None
    for epoch in range(n_epochs):
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            wsum += weights[i]
            step = weights[i] / wsum            # 1/k style, weighted
            s = geodesic_point(s, trees[i], step)
            k += 1
        if prev is not None:
            if geodesic_distance(prev, s) < tol:
                break
        prev = s.copy()
    return s


def frechet_variance(mean: BHVTree, trees: Sequence[BHVTree],
                     weights: Optional[Sequence[float]] = None) -> float:
    trees = list(trees)
    if weights is None:
        weights = [1.0] * len(trees)
    num = sum(w * geodesic_distance(mean, t) ** 2 for w, t in zip(weights, trees))
    return num / sum(weights)


# ---------------------------------------------------------------------------
# Barden-Le-Owen modified log map   T_{m+3} -> R^m  (indexed by base edges)
# ---------------------------------------------------------------------------

def log_map(base: BHVTree, T: BHVTree,
            index: Optional[List[frozenset]] = None) -> np.ndarray:
    """Modified log map of tree T relative to `base` (T*), in R^m where the m
    coordinates are the internal edges of `base` (order = `index`).

    Cases (Willis & Bell 2018, sec 2.3), all reproduced exactly by
    Phi = base_lengths + first_segment_velocity:
      * shared internal edge      -> length of that edge in T
      * base-only edge (dropped)  -> -|f|*||A_i||/||B_i||  (single NNI: -|new edge|)
    """
    if index is None:
        index = list(base.internal.keys())
    g = geodesic(base, T)
    pos = {k: i for i, k in enumerate(index)}
    phi = np.array([base.internal[k] for k in index], dtype=float)

    # shared edges -> target length
    for k, ls, lt in g.shared:
        if k in pos:
            phi[pos[k]] = lt
    # base-only (outgoing) edges -> negative coordinate from first-segment velocity
    for p in g.pairs:
        nB = p.normB
        if nB <= 0:
            continue
        nA = p.normA
        for b in p.B:
            if b in pos:
                phi[pos[b]] = -p.lenB[b] * nA / nB
    return phi


# ---------------------------------------------------------------------------
# Confidence set  (Willis 2019, Eq. 4)
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSet:
    mean: BHVTree
    index: List[frozenset]          # m base edges defining the coordinates
    phi_bar: np.ndarray             # (m,) mean of replicate log maps
    cov: np.ndarray                 # (m,m) sample covariance S
    n: int
    m: int
    alpha: float
    replicate_phi: np.ndarray       # (n,m)

    def _stat(self, phi0: np.ndarray) -> float:
        diff = self.phi_bar - phi0
        Sinv = np.linalg.pinv(self.cov)
        return float(diff @ Sinv @ diff)

    def threshold(self) -> float:
        n, m = self.n, self.m
        return m * (n - 1) / (n * (n - m)) * f_ppf(1 - self.alpha, m, n - m)

    def contains(self, tree: BHVTree) -> Tuple[bool, float, float]:
        """Is `tree` in the 100(1-alpha)% set? Returns (in_set, stat, p_value)."""
        phi0 = log_map(self.mean, tree, self.index)
        stat = self._stat(phi0)
        n, m = self.n, self.m
        f_stat = stat * n * (n - m) / (m * (n - 1))
        p = float(1 - f_cdf(f_stat, m, n - m))
        return stat < self.threshold(), stat, p

    def split_support(self) -> List[dict]:
        """Per-split summary: presence frequency across replicates, mean
        coordinate, and marginal 95% CI (positive & excluding 0 => supported)."""
        out = []
        for j, key in enumerate(self.index):
            col = self.replicate_phi[:, j]
            freq = float(np.mean(col > 0))           # present (positive coord)
            mu = float(np.mean(col))
            se = float(np.std(col, ddof=1) / math.sqrt(self.n)) if self.n > 1 else float("nan")
            lo, hi = mu - 1.96 * se, mu + 1.96 * se
            out.append({
                "split": tuple(sorted(key)),
                "size": len(key),
                "presence_freq": freq,
                "mean_coord": mu,
                "ci_low": lo,
                "ci_high": hi,
                "supported": lo > 0,
            })
        return out


def confidence_set(trees: Sequence[BHVTree], alpha: float = 0.05,
                   mean: Optional[BHVTree] = None,
                   weights: Optional[Sequence[float]] = None,
                   **mean_kwargs) -> ConfidenceSet:
    """Build the Willis (2019) confidence set from replicate trees.

    n must exceed m (number of internal edges of the mean) for the F-test.
    """
    trees = list(trees)
    if mean is None:
        mean = frechet_mean(trees, weights=weights, **mean_kwargs)
    index = list(mean.internal.keys())
    m = len(index)
    n = len(trees)
    phi = np.vstack([log_map(mean, t, index) for t in trees]) if m > 0 \
        else np.zeros((n, 0))
    phi_bar = phi.mean(axis=0) if m > 0 else np.zeros(0)
    if n > 1 and m > 0:
        cov = np.cov(phi, rowvar=False, ddof=1)
        cov = np.atleast_2d(cov)
    else:
        cov = np.zeros((m, m))
    return ConfidenceSet(mean, index, phi_bar, cov, n, m, alpha, phi)


# ---------------------------------------------------------------------------
# RF distance (topology only) -- convenience for comparing to majority consensus
# ---------------------------------------------------------------------------

def rf_distance(t1: BHVTree, t2: BHVTree) -> Tuple[int, int]:
    s1, s2 = set(t1.internal), set(t2.internal)
    rf = len(s1 ^ s2)
    rf_max = len(s1) + len(s2)
    return rf, rf_max
