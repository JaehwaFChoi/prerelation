"""prerelation.core — the prerelation coefficient and its direction statistic.

The coefficient
---------------
For a pair of traits reported on a common anchored scale [0, 1],

    Pi(X -> Y) = A1 * A2,

    A1 = max(0, 1 - v / v0)      emptiness of the corner {Y > X}, measured
                                 against the independence baseline v0
    A2 = q * ell                 conditional freedom, censoring-aware
    q                            uniformity of the interior of
                                 W = min(Y / X, 1) below the ceiling band
    ell                          legitimacy of the ceiling: censoring must
                                 thin out at high x, which is what separates
                                 a genuine ceiling from equivalence

    Delta = Pi(X -> Y) - Pi(Y -> X).

Anchored scales are an interpretability requirement, not a claim about the
measurement precision of any scoring model: the ratio Y / X and the corner
moment (Y - X)_+ only carry the reading "how much of the ceiling granted by X
is used by Y" when both endpoints are substantive anchors. On an unanchored
scale Pi carries no prerequisite interpretation.

Correctness standard
--------------------
``tests/oracle/prereq_index_v2.py`` is kept verbatim and is the permanent
oracle for this package. The functions below are vectorised, but every
branch of the definition is transcribed literally and the tests pin the
outputs to the oracle within 1e-12. The following are part of the
*definition*, not implementation detail:

* the independence baseline is a V-statistic — the double sum runs over all
  n**2 ordered pairs, the diagonal i = j included;
* ``v0 <= 1e-9`` forces ``A1 = 0``;
* fewer than ``max(10, 0.05 n)`` interior points forces ``q = 0``;
* an empty top-x stratum forces ``p1_top = 1`` (hence ``ell = 0``);
* the ratio is clipped to [0, 1] and its denominator floored at 1e-9.

Only the summation order of the baseline is allowed to differ, and only
above ``DENSE_MAX_N`` (see ``_baseline_mean``).

The defaults ``delta = 0.05``, ``TOP_Q = 0.8`` and
``MIN_INTERIOR = max(10, 0.05 n)`` are fixed. A small-sample adaptation of
TOP_Q is an open question and is deliberately *not* implemented here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DELTA",
    "TOP_Q",
    "MIN_INTERIOR",
    "prereq_index",
    "direction",
    "perm_pvalue",
]

DELTA = 0.05          # ceiling band width
TOP_Q = 0.8           # top-x quantile for the legitimacy check
MIN_INTERIOR = 10     # minimum interior points before freedom is credited

_EPS_DEN = 1e-9       # floor of the ratio denominator
_EPS_V0 = 1e-9        # guard on the independence baseline

# Above this sample size the baseline is accumulated by sorting instead of
# forming the n x n matrix. The two paths agree to floating-point rounding
# (they differ only in summation order); below the threshold the dense
# expression of the oracle is used verbatim, so the agreement is exact.
DENSE_MAX_N = 3000


def _as_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if x.size != y.size:
        raise ValueError(f"x and y must have equal length, got {x.size} and {y.size}")
    if x.size == 0:
        raise ValueError("x and y must be non-empty")
    return x, y


def _baseline_mean(x, y, dense_max_n=DENSE_MAX_N):
    """Independence baseline v0 = mean over all n**2 ordered pairs of (y_j - x_i)_+.

    The diagonal is included: this is a V-statistic, not a U-statistic.
    """
    n = x.size
    if n <= dense_max_n:
        # Verbatim oracle expression, hence bit-identical below the threshold.
        return float(np.mean(np.maximum(y[None, :] - x[:, None], 0.0)))

    # Sort-and-accumulate: for each x_i, sum_j (y_j - x_i)_+ equals the sum of
    # the y above x_i minus x_i times how many there are. O(n log n) instead of
    # O(n**2), which keeps large scans and permutation loops feasible.
    ys = np.sort(y)
    tail = np.concatenate((np.cumsum(ys[::-1])[::-1], [0.0]))  # tail[k] = sum(ys[k:])
    k = np.searchsorted(ys, x, side="right")                   # count of y_j <= x_i
    inner = tail[k] - x * (n - k)
    return float(np.mean(inner) / n)


def prereq_index(x, y, delta=DELTA, dense_max_n=DENSE_MAX_N):
    """Prerelation coefficient of the ordered pair (x -> y).

    Parameters
    ----------
    x, y : array_like of shape (n,)
        Trait values on a common anchored scale; x is the candidate
        prerequisite. Values are expected in [0, 1].
    delta : float
        Ceiling band width. The default 0.05 is the fixed convention.
    dense_max_n : int
        Sample size up to which the independence baseline is formed as a
        dense n x n matrix (bit-identical to the oracle).

    Returns
    -------
    dict with keys ``PI``, ``A1``, ``A2``, ``q``, ``ell``.

    Notes
    -----
    The statistic is deliberately not invariant to monotone rescaling of
    either axis, and it is not symmetric: ``prereq_index(y, x)`` answers a
    different question.
    """
    x, y = _as_pair(x, y)
    n = x.size

    # A1: corner emptiness relative to the independence baseline.
    v = float(np.mean(np.maximum(y - x, 0.0)))
    v0 = _baseline_mean(x, y, dense_max_n)
    a1 = max(0.0, 1.0 - v / v0) if v0 > _EPS_V0 else 0.0

    # A2: conditional freedom with a censoring-aware benchmark.
    u = np.clip(y / np.maximum(x, _EPS_DEN), 0.0, 1.0)
    ceil_mask = u >= 1.0 - delta
    interior = u[~ceil_mask]

    if interior.size < max(MIN_INTERIOR, 0.05 * n):
        q = 0.0
    else:
        t = np.sort(interior / (1.0 - delta))
        F = np.arange(1, t.size + 1) / t.size
        q = 1.0 - float(np.max(np.abs(F - t)))

    x_top = x >= np.quantile(x, TOP_Q)
    p1_top = float(np.mean(ceil_mask[x_top])) if x_top.sum() > 0 else 1.0
    ell = 1.0 - max(0.0, p1_top - delta) / (1.0 - delta)

    a2 = q * ell
    return {"PI": a1 * a2, "A1": a1, "A2": a2, "q": q, "ell": ell}


def direction(x, y, delta=DELTA, dense_max_n=DENSE_MAX_N):
    """Directional contrast of the pair.

    Returns
    -------
    (delta_stat, pi_xy, pi_yx) : tuple of float
        ``delta_stat = pi_xy - pi_yx``. The order of the tuple follows the
        oracle so that parity tests can compare element by element.
    """
    pi_xy = prereq_index(x, y, delta, dense_max_n)["PI"]
    pi_yx = prereq_index(y, x, delta, dense_max_n)["PI"]
    return pi_xy - pi_yx, pi_xy, pi_yx


def perm_pvalue(x, y, n_perm=1000, seed=0, delta=DELTA, dense_max_n=DENSE_MAX_N):
    """Permutation test of independence for the forward statistic.

    Under the null of independence the joint law of the sample is invariant
    under permutations of the y-labels, so the test is exact for the full
    group; the Monte-Carlo version below inherits validity because the
    observed configuration is counted in the reference set (the add-one
    rule).

    Returns
    -------
    (observed, p_value) : tuple of float

    Notes
    -----
    The random draws are consumed exactly as the oracle consumes them — one
    ``rng.permutation(y)`` per replicate, in order — so the same seed gives
    the same reference set. The count uses ``>=``, which makes it sensitive
    to exact ties; ties occur in degenerate configurations where many
    permutation replicates give an identical value (typically 0), and there
    a rounding difference of 1e-16 in the observed statistic can move the
    p-value by one step of 1 / (n_perm + 1).
    """
    x, y = _as_pair(x, y)
    rng = np.random.default_rng(seed)
    obs = prereq_index(x, y, delta, dense_max_n)["PI"]
    cnt = 0
    for _ in range(n_perm):
        cnt += prereq_index(x, rng.permutation(y), delta, dense_max_n)["PI"] >= obs
    return obs, (cnt + 1) / (n_perm + 1)
