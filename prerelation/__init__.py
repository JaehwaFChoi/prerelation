"""prerelation — a coefficient for prerequisite relations on anchored scales.

Two abilities can be strongly associated without one being a prerequisite
for the other, and they can be ordered without being distinct. The
prerelation coefficient Pi in [0, 1] asks a narrower question than
correlation and a different one from necessity analysis: does the candidate
prerequisite X act as a *ceiling* on Y, and does Y retain the freedom below
that ceiling which a prerequisite structure implies?

    Pi(X -> Y) = A1 * A2

    A1  the corner {Y > X} is empty relative to what independence would give
    A2  below the ceiling, Y varies as a free component should, and the
        censoring thins out at high x

The product structure is what lets one number separate the four extremes:
independence is annihilated by A1 alone, exact equivalence by A2 alone.

Scope
-----
Pi is defined for traits on a common bounded scale whose endpoints are
substantive anchors (0 = absence of the ability, 1 = full mastery). This is
an interpretability requirement on the scale, not a claim about the
measurement precision of any scoring model: the ratio Y / X simply has no
prerequisite reading on a location-scale standardised latent trait. The
bounded trait model of Choi (2022) is one scale that satisfies the
requirement, and ``prerelation.pv`` can consume its posteriors, but the
coefficient itself is model-free.

Modules
-------
``core``     Pi, Delta and the permutation test (numpy only)
``scan``     all pairs, BH-FDR, cycle check, transitive reduction
``ceiling``  monotone quantile ceiling and the ceiling-referenced variant
``pv``       plausible-value correction for scoring error (optional extra)
``study``    the simulation frame: one config dict in, tidy table out

Correctness standard
--------------------
``tests/oracle/prereq_index_v2.py`` is kept verbatim as the permanent oracle
and the implementation is pinned to it within 1e-12.

Quick start
-----------
    >>> import numpy as np
    >>> from prerelation import prereq_index, direction
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 1, 500)
    >>> y = x * rng.uniform(0, 1, 500)          # X is a ceiling on Y
    >>> round(prereq_index(x, y)["PI"], 2)
    0.94
    >>> d, fwd, rev = direction(x, y)
    >>> rev                                      # the reverse reading is empty
    0.0
"""

from __future__ import annotations

from . import ceiling, core, pv, scan, study
from .ceiling import CeilingFit, ceiling_fit, prereq_index_referenced
from .core import DELTA, MIN_INTERIOR, TOP_Q, direction, perm_pvalue, prereq_index
from .pv import ctm_posterior, draw_pv, pv_correct
from .scan import ScanResult, bh_fdr, find_cycles, scan as scan_pairs, transitive_reduction
from .study import EXAMPLE_CONFIG, run_study

__version__ = "0.2.0"

__all__ = [
    # coefficient
    "prereq_index",
    "direction",
    "perm_pvalue",
    "DELTA",
    "TOP_Q",
    "MIN_INTERIOR",
    # screening
    "scan_pairs",
    "ScanResult",
    "bh_fdr",
    "find_cycles",
    "transitive_reduction",
    # ceiling
    "ceiling_fit",
    "CeilingFit",
    "prereq_index_referenced",
    # plausible values
    "pv_correct",
    "draw_pv",
    "ctm_posterior",
    # simulation frame
    "run_study",
    "EXAMPLE_CONFIG",
    # submodules
    "core",
    "scan",
    "ceiling",
    "pv",
    "study",
    "__version__",
]

# ``scan`` is both a module and the function most users want. The module is
# reachable as ``prerelation.scan``; the function is exported under the
# unambiguous name ``scan_pairs`` and also as ``prerelation.scan.scan``.
