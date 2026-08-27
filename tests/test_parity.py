"""Parity with the permanent oracle.

``tests/oracle/prereq_index_v2.py`` is a verbatim copy of the reference
implementation that produced the recorded results. Nothing in the package is
allowed to disagree with it by more than 1e-12.
"""

import numpy as np
import pytest

from prerelation import core
from tests.oracle import prereq_index_v2 as oracle

TOL = 1e-12
COMPONENTS = ("PI", "A1", "A2", "q", "ell")


def oracle_scenarios(n=2000, seed=42):
    """The ten scenarios of the oracle's __main__ block, rebuilt verbatim."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, n)
    U = rng.uniform(0, 1, n)
    S = {
        "pure prereq   Y=X*U": (X, X * U),
        "independent": (X, rng.uniform(0, 1, n)),
        "equivalence   Y=X+e(.02)": (X, np.clip(X + rng.normal(0, 0.02, n), 0, 1)),
        "degenerate    Y~0": (X, 0.02 * rng.uniform(0, 1, n)),
        "noisy prereq  Y=X*U+e": (X, np.clip(X * U + rng.normal(0, 0.05, n), 0, 1)),
        "min model     Y=min(X,U)": (X, np.minimum(X, U)),
        "nonunif U     Y=X*Beta(2,2)": (X, X * rng.beta(2, 2, n)),
        "min + noise": (X, np.clip(np.minimum(X, U) + rng.normal(0, 0.05, n), 0, 1)),
        "equivalence   Y=X+e(.10)": (X, np.clip(X + rng.normal(0, 0.10, n), 0, 1)),
    }
    Yr = rng.uniform(0, 1, n)
    S["reverse       X=Y*U"] = (Yr * U, Yr)
    return S


SCENARIOS = oracle_scenarios()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_components_match_oracle(name):
    x, y = SCENARIOS[name]
    got = core.prereq_index(x, y)
    want = oracle.prereq_index(x, y)
    for key in COMPONENTS:
        assert abs(got[key] - want[key]) <= TOL, (name, key, got[key], want[key])


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_direction_matches_oracle(name):
    x, y = SCENARIOS[name]
    got = core.direction(x, y)
    want = oracle.direction(x, y)
    assert len(got) == len(want) == 3
    for g, w in zip(got, want):
        assert abs(g - w) <= TOL, (name, g, w)


@pytest.mark.parametrize("name", ["pure prereq   Y=X*U", "independent",
                                  "min model     Y=min(X,U)"])
def test_perm_pvalue_matches_oracle(name):
    """Same seed, same reference set, same p-value.

    Restricted to non-degenerate scenarios: the count uses ``>=``, so where
    many replicates land on an identical value a 1e-16 difference in the
    observed statistic could move the p-value by one step. That sensitivity
    is a property of the definition, not a defect of the implementation, and
    it is exercised deliberately in test_edge.py.
    """
    x, y = SCENARIOS[name]
    x, y = x[:400], y[:400]
    got = core.perm_pvalue(x, y, n_perm=60, seed=7)
    want = oracle.perm_pvalue(x, y, n_perm=60, seed=7)
    assert abs(got[0] - want[0]) <= TOL
    assert got[1] == want[1]


def test_baseline_is_a_v_statistic():
    """The independence baseline includes the diagonal i = j."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 1, 200)
    y = rng.uniform(0, 1, 200)
    dense = np.mean(np.maximum(y[None, :] - x[:, None], 0.0))
    n = x.size
    off = (dense * n * n - np.sum(np.maximum(y - x, 0.0))) / (n * (n - 1))
    assert abs(core._baseline_mean(x, y) - dense) <= TOL
    assert abs(core._baseline_mean(x, y) - off) > 1e-6  # the two differ


@pytest.mark.parametrize("case", ["product", "independent", "constant_y"])
def test_sorted_baseline_matches_dense_path(case):
    """Above DENSE_MAX_N the baseline switches algorithm but not value.

    ``constant_y`` is the worst case found: the running sum of the tail is
    accumulated sequentially, so a constant y at n in the thousands drifts by
    a few parts in 1e13 *relative* — still four orders inside the 1e-12
    absolute tolerance, because the baseline itself is O(0.05).
    """
    rng = np.random.default_rng(11)
    n = 2500
    x = rng.uniform(0, 1, n)
    y = {
        "product": lambda: x * rng.uniform(0, 1, n),
        "independent": lambda: rng.uniform(0, 1, n),
        "constant_y": lambda: np.full(n, 0.3),
    }[case]()
    dense = core._baseline_mean(x, y, dense_max_n=10_000)
    sorted_path = core._baseline_mean(x, y, dense_max_n=10)
    assert abs(dense - sorted_path) <= TOL
    assert abs(dense - sorted_path) / dense <= 1e-11


def test_sorted_path_handles_a_sample_the_dense_path_cannot():
    """n = 20000 needs a 400-million-element matrix in the dense form."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20_000)
    y = x * rng.uniform(0, 1, 20_000)
    r = core.prereq_index(x, y)
    assert 0.0 <= r["PI"] <= 1.0
    assert r["A1"] == 1.0  # y <= x everywhere, so the corner is empty


def test_large_sample_parity_against_oracle():
    """n above the dense threshold still matches the oracle within 1e-12."""
    rng = np.random.default_rng(20260826)
    x = rng.uniform(0, 1, 5000)
    y = 0.7 * x * rng.uniform(0, 1, 5000)
    got = core.prereq_index(x, y)
    want = oracle.prereq_index(x, y)
    for key in COMPONENTS:
        assert abs(got[key] - want[key]) <= TOL, (key, got[key], want[key])
