"""Boundary cases where the guards, not the formulas, decide the answer."""

import numpy as np
import pytest

from prerelation import core
from tests.oracle import prereq_index_v2 as oracle

TOL = 1e-12


def test_exact_equivalence_is_exactly_zero():
    """Y = X: no interior and a fully occupied ceiling — annihilated twice."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    r = core.prereq_index(x, x.copy())
    assert r["q"] == 0.0 and r["ell"] == 0.0 and r["PI"] == 0.0
    assert r["A1"] == 1.0  # the corner is empty, which alone proves nothing


def test_degenerate_zero_hits_the_v0_guard():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 300)
    r = core.prereq_index(x, np.zeros(300))
    assert r["A1"] == 0.0 and r["q"] == 0.0 and r["PI"] == 0.0


def test_degenerate_constant_is_zero_only_to_machine_epsilon():
    """v = v0 is an identity of real arithmetic, not of floating point.

    With y constant the two averages sum the same numbers in a different
    order, so A1 lands within a few ulp of zero rather than on it. Exact-zero
    assertions belong to the branches that are structurally zero; everywhere
    else the tolerance is 1e-12.
    """
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 5000)
    r = core.prereq_index(x, np.full(5000, 0.3))
    assert r["A1"] < 1e-12
    assert r["PI"] < 1e-12
    o = oracle.prereq_index(x, np.full(5000, 0.3))
    assert abs(r["A1"] - o["A1"]) <= TOL


def test_interior_shortage_zeroes_q():
    """Fewer than max(10, 0.05 n) interior points means freedom is not credited."""
    rng = np.random.default_rng(3)
    n = 200
    x = rng.uniform(0.5, 1, n)
    y = x * rng.uniform(0.98, 1.0, n)  # almost everything sits in the band
    r = core.prereq_index(x, y)
    interior = (np.clip(y / np.maximum(x, 1e-9), 0, 1) < 1 - core.DELTA).sum()
    assert interior < max(core.MIN_INTERIOR, 0.05 * n)
    assert r["q"] == 0.0 and r["PI"] == 0.0


def test_tiny_sample_below_min_interior():
    x = np.linspace(0.1, 1.0, 8)
    y = x * 0.5
    r = core.prereq_index(x, y)
    assert r["q"] == 0.0  # 8 points cannot clear MIN_INTERIOR = 10
    assert r["PI"] == 0.0
    assert abs(r["A1"] - oracle.prereq_index(x, y)["A1"]) <= TOL


def test_reverse_direction_of_a_ceiling_is_exactly_zero():
    """With Y <= X the reverse ratio is pinned at 1, so q and ell both vanish."""
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 1, 800)
    y = x * rng.uniform(0, 1, 800)
    d, fwd, rev = core.direction(x, y)
    assert rev == 0.0
    assert d == fwd


def test_zero_x_is_absorbed_by_the_denominator_floor():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, 300)
    x[:5] = 0.0
    y = x * rng.uniform(0, 1, 300)
    r = core.prereq_index(x, y)
    o = oracle.prereq_index(x, y)
    for key in ("PI", "A1", "A2", "q", "ell"):
        assert abs(r[key] - o[key]) <= TOL


def test_input_validation():
    with pytest.raises(ValueError):
        core.prereq_index([0.1, 0.2], [0.1])
    with pytest.raises(ValueError):
        core.prereq_index([], [])
    with pytest.raises(ValueError):
        core.prereq_index(np.zeros((2, 2)), np.zeros((2, 2)))


def test_permutation_pvalue_is_bounded_and_add_one():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 1, 200)
    y = rng.uniform(0, 1, 200)
    obs, p = core.perm_pvalue(x, y, n_perm=49, seed=1)
    assert 1 / 50 <= p <= 1.0
    assert 0.0 <= obs <= 1.0
