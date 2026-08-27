"""prerelation.scan — pairwise screening of a whole attribute set.

``scan`` takes an (n_persons, n_attributes) matrix of trait values on a
common anchored scale and returns

* a tidy table with Pi, the reverse Pi, Delta and a permutation p-value for
  every ordered pair;
* the edge set surviving Benjamini-Hochberg control of the false discovery
  rate;
* a cycle report; and
* the transitive reduction of the edge set when it is acyclic
  (Aho, Garey and Ullman, 1972: for a directed acyclic graph the transitive
  reduction is unique and is a subgraph of the original).

The reduction is what turns a screen into a readable diagram: it removes
the edges that are already implied by a longer path, leaving a Hasse-like
picture of the recovered ordering.

What the scan recovers is a **dominance preorder** over the attributes —
the ordering induced by which attributes act as ceilings on which others —
not a direct-prerequisite DAG. Indirect dominance produces edges of its
own (removed only along chains by the transitive reduction), and siblings
that share a common ceiling can be linked to each other even though
neither is a prerequisite for the other. A disagreement between the
recovered order and an expert-specified prerequisite graph is therefore a
difference between two concepts, not by itself an error in either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import DELTA, prereq_index

__all__ = ["ScanResult", "scan", "bh_fdr", "find_cycles", "transitive_reduction"]


def bh_fdr(pvalues):
    """Benjamini-Hochberg adjusted p-values (step-up, monotonised)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def _reachable(adj: Dict[str, set], src: str, dst: str, skip: Tuple[str, str]) -> bool:
    """Is dst reachable from src without traversing the single edge `skip`?"""
    stack = [src]
    seen = {src}
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if (u, v) == skip:
                continue
            if v == dst:
                return True
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def transitive_reduction(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]):
    """Transitive reduction of a directed acyclic graph.

    An edge is dropped when the same ordering is already implied by a path of
    length two or more. Raises if the graph has a cycle, where the reduction
    is not unique.
    """
    cycles = find_cycles(nodes, edges)
    if cycles:
        raise ValueError(
            "transitive reduction is only defined for acyclic graphs; "
            f"found cycle {cycles[0]}"
        )
    adj = {u: set() for u in nodes}
    for u, v in edges:
        adj[u].add(v)
    kept = []
    for u, v in edges:
        if not _reachable(adj, u, v, skip=(u, v)):
            kept.append((u, v))
    return kept


def find_cycles(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]):
    """Return the directed cycles found by depth-first search (may be empty)."""
    adj = {u: [] for u in nodes}
    for u, v in edges:
        adj[u].append(v)
    colour = {u: 0 for u in nodes}  # 0 white, 1 grey, 2 black
    stack: List[str] = []
    cycles: List[List[str]] = []

    def visit(u):
        colour[u] = 1
        stack.append(u)
        for v in adj[u]:
            if colour[v] == 0:
                visit(v)
            elif colour[v] == 1:
                cycles.append(stack[stack.index(v):] + [v])
        stack.pop()
        colour[u] = 2

    for u in nodes:
        if colour[u] == 0:
            visit(u)
    return cycles


@dataclass
class ScanResult:
    """Outcome of a pairwise scan.

    Attributes
    ----------
    records : list of dict
        One row per ordered pair; always available (no pandas needed).
    edges : list of (source, target)
        Pairs surviving the FDR threshold and the direction rule.
    reduced_edges : list of (source, target) or None
        Transitive reduction of ``edges``; None when a cycle was found.
    cycles : list of list of str
    names : list of str
    alpha : float
    """

    records: List[dict]
    edges: List[Tuple[str, str]]
    reduced_edges: Optional[List[Tuple[str, str]]]
    cycles: List[List[str]]
    names: List[str]
    alpha: float
    meta: dict = field(default_factory=dict)

    @property
    def table(self):
        """Tidy pandas DataFrame of ``records`` (pandas imported lazily)."""
        import pandas as pd

        return pd.DataFrame.from_records(self.records)

    def __repr__(self):  # pragma: no cover - display only
        return (
            f"ScanResult(attributes={len(self.names)}, tests={len(self.records)}, "
            f"edges={len(self.edges)}, "
            f"reduced={'n/a' if self.reduced_edges is None else len(self.reduced_edges)}, "
            f"cycles={len(self.cycles)})"
        )


def scan(
    theta_matrix,
    alpha=0.05,
    names=None,
    n_perm=999,
    seed=0,
    delta=DELTA,
    min_pi=0.0,
    require_positive_delta=True,
):
    """Screen every ordered pair of attributes for ceiling dominance.

    The edge set (and its transitive reduction) is read as a *dominance
    preorder* over the attributes, not as a recovered direct-prerequisite
    DAG; see the module docstring for what separates the two.

    **Design floor on permutation replicates.** With ``k`` attributes there
    are ``K = k (k - 1)`` ordered pairs, and the smallest attainable
    permutation p-value is ``1 / (n_perm + 1)``. For any pair to survive
    Benjamini-Hochberg control at level ``alpha`` the replicate count must
    satisfy ``n_perm >= K / alpha - 1`` (e.g. ``K = 6``, ``alpha = 0.05``
    requires ``n_perm >= 119``). Below the floor the scan cannot return any
    edge, regardless of the data.

    Parameters
    ----------
    theta_matrix : array_like of shape (n_persons, n_attributes)
        Trait values on a common anchored scale.
    alpha : float
        Target false discovery rate for the BH step-up procedure, applied
        jointly to all ordered pairs.
    names : sequence of str, optional
        Attribute labels; defaults to ``A1, A2, ...``.
    n_perm : int
        Permutation replicates per ordered pair. Each pair gets its own
        stream, seeded as ``seed + pair_position``, so results are
        reproducible and independent of the order of evaluation.
    min_pi : float
        Additional floor on Pi for an edge to be kept; 0 by default, since
        significance is the primary rule.
    require_positive_delta : bool
        Keep an edge only when the forward direction dominates the reverse.

    Returns
    -------
    ScanResult
    """
    theta = np.asarray(theta_matrix, dtype=float)
    if theta.ndim != 2:
        raise ValueError("theta_matrix must be two-dimensional (persons x attributes)")
    n, k = theta.shape
    if k < 2:
        raise ValueError("need at least two attributes to scan")
    if names is None:
        names = [f"A{j + 1}" for j in range(k)]
    names = list(names)
    if len(names) != k:
        raise ValueError("names must have one entry per attribute")

    pi_matrix = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i != j:
                pi_matrix[i, j] = prereq_index(theta[:, i], theta[:, j], delta)["PI"]

    records = []
    pair_position = 0
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            x, y = theta[:, i], theta[:, j]
            res = prereq_index(x, y, delta)
            rng = np.random.default_rng(seed + pair_position)
            obs = res["PI"]
            cnt = 0
            for _ in range(n_perm):
                cnt += prereq_index(x, rng.permutation(y), delta)["PI"] >= obs
            p_value = (cnt + 1) / (n_perm + 1)
            records.append(
                {
                    "source": names[i],
                    "target": names[j],
                    "pi": res["PI"],
                    "pi_reverse": pi_matrix[j, i],
                    "delta": res["PI"] - pi_matrix[j, i],
                    "A1": res["A1"],
                    "A2": res["A2"],
                    "q": res["q"],
                    "ell": res["ell"],
                    "p_value": p_value,
                    "n": n,
                    "n_perm": n_perm,
                }
            )
            pair_position += 1

    p_adj = bh_fdr([r["p_value"] for r in records])
    edges = []
    for rec, pa in zip(records, p_adj):
        rec["p_adj"] = float(pa)
        keep = pa <= alpha and rec["pi"] >= min_pi
        if require_positive_delta:
            keep = keep and rec["delta"] > 0
        rec["edge"] = bool(keep)
        if keep:
            edges.append((rec["source"], rec["target"]))

    cycles = find_cycles(names, edges)
    reduced = None if cycles else transitive_reduction(names, edges)

    return ScanResult(
        records=records,
        edges=edges,
        reduced_edges=reduced,
        cycles=cycles,
        names=names,
        alpha=alpha,
        meta={"n": n, "n_perm": n_perm, "seed": seed, "delta": delta},
    )
