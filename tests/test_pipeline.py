"""Unit tests for the screening, ceiling, plausible-value and study layers."""

import numpy as np
import pytest

from prerelation import ceiling as ceilmod
from prerelation import pv as pvmod
from prerelation import study as studymod
from prerelation.core import prereq_index
from prerelation.scan import (
    bh_fdr,
    find_cycles,
    scan,
    transitive_reduction,
)


# ------------------------------------------------------------------ scan
def chain_traits(n=600, seed=0):
    """A -> B -> C: each step is a ceiling with its own free component."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, n)
    b = a * rng.uniform(0, 1, n)
    c = b * rng.uniform(0, 1, n)
    return np.column_stack([a, b, c])


def test_bh_fdr_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    adj = bh_fdr(p)
    assert np.all(np.diff(adj) >= -1e-15)
    assert np.all((adj >= p - 1e-15) & (adj <= 1.0))
    assert bh_fdr([]).size == 0


def test_bh_fdr_matches_hand_computation():
    p = np.array([0.01, 0.02, 0.04])
    # m/k * p = 0.03, 0.03, 0.04 -> monotonised from the right
    assert np.allclose(bh_fdr(p), [0.03, 0.03, 0.04])


def test_scan_recovers_a_known_chain():
    theta = chain_traits()
    res = scan(theta, alpha=0.05, names=["A", "B", "C"], n_perm=99, seed=1)
    assert len(res.records) == 6  # every ordered pair
    assert res.cycles == []
    assert ("A", "B") in res.edges and ("B", "C") in res.edges
    assert ("B", "A") not in res.edges and ("C", "B") not in res.edges
    # A -> C is implied by the chain and must not survive the reduction
    assert ("A", "C") not in res.reduced_edges
    assert set(res.reduced_edges) == {("A", "B"), ("B", "C")}


def test_scan_table_is_tidy():
    pd = pytest.importorskip("pandas")
    res = scan(chain_traits(n=300, seed=4), names=["A", "B", "C"], n_perm=49, seed=2)
    df = res.table
    assert isinstance(df, pd.DataFrame)
    assert {"source", "target", "pi", "delta", "p_value", "p_adj", "edge"} <= set(df.columns)
    assert len(df) == 6


def test_scan_finds_no_edges_under_independence():
    rng = np.random.default_rng(5)
    theta = rng.uniform(0, 1, (400, 3))
    res = scan(theta, alpha=0.05, n_perm=99, seed=3)
    assert res.edges == []
    assert res.reduced_edges == []


def test_transitive_reduction_drops_the_implied_edge():
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C"), ("A", "C")]
    assert set(transitive_reduction(nodes, edges)) == {("A", "B"), ("B", "C")}


def test_cycle_is_detected_and_blocks_reduction():
    nodes = ["A", "B", "C"]
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    cycles = find_cycles(nodes, edges)
    assert cycles
    with pytest.raises(ValueError):
        transitive_reduction(nodes, edges)


# --------------------------------------------------------------- ceiling
def test_ceiling_fit_recovers_a_proportional_ceiling():
    rng = np.random.default_rng(7)
    n = 4000
    x = rng.uniform(0, 1, n)
    y = 0.6 * x * rng.uniform(0, 1, n)
    fit = ceilmod.ceiling_fit(x, y, tau=0.98, seed=0)
    grid = np.linspace(0.2, 1.0, 9)
    pred = fit.predict(grid)
    assert np.all(np.diff(pred) >= -1e-12)          # monotone by construction
    assert np.max(np.abs(pred - 0.6 * grid)) < 0.08  # near the true ceiling


def test_referenced_variant_recomputes_every_component():
    """The referenced index is the same statistic on the transformed pair."""
    rng = np.random.default_rng(8)
    n = 3000
    x = rng.uniform(0, 1, n)
    y = 0.6 * x * rng.uniform(0, 1, n)
    fit = ceilmod.ceiling_fit(x, y, tau=0.98, seed=0)
    got = ceilmod.prereq_index_referenced(x, y, fit)
    sel = fit.eval_index
    want = prereq_index(np.clip(fit.predict(x[sel]), 0, 1), y[sel])
    assert got == want                      # identical dict, not a partial patch
    assert got["PI"] > prereq_index(x[sel], y[sel])["PI"]  # attenuation undone


def test_ceiling_fit_uses_a_held_out_half():
    rng = np.random.default_rng(9)
    x = rng.uniform(0, 1, 500)
    y = x * rng.uniform(0, 1, 500)
    fit = ceilmod.ceiling_fit(x, y, seed=3)
    assert fit.fit_index.size + fit.eval_index.size == 500
    assert not set(fit.fit_index) & set(fit.eval_index)


def test_ctm_ceiling_hits_both_anchors():
    c = ceilmod.ctm_ceiling(np.array([0.0, 1.0]), a=8.0, b=0.4)
    assert abs(c[0]) < 1e-12 and abs(c[1] - 1.0) < 1e-12


def test_ctm_method_fits_a_curved_ceiling():
    pytest.importorskip("scipy")
    rng = np.random.default_rng(10)
    n = 1500
    x = rng.uniform(0, 1, n)
    y = (x ** 2) * rng.uniform(0, 1, n)
    fit = ceilmod.ceiling_fit(x, y, tau=0.97, method="ctm", seed=1)
    pred = fit.predict(np.array([0.25, 0.5, 0.75]))
    assert np.all(np.diff(pred) > 0)
    assert np.all(pred <= 1.0) and np.all(pred >= 0.0)


# -------------------------------------------------------------------- pv
def test_draw_pv_reproduces_a_point_mass_posterior():
    nodes = np.linspace(0, 1, 61)
    post = np.zeros((4, 61))
    post[:, 30] = 1.0
    draws = pvmod.draw_pv(nodes, post, 5, np.random.default_rng(0))
    assert draws.shape == (5, 4)
    assert np.allclose(draws, nodes[30])


def test_pv_correct_reports_spread_across_draws():
    rng = np.random.default_rng(11)
    n = 400
    truth_x = rng.uniform(0, 1, n)
    truth_y = truth_x * rng.uniform(0, 1, n)
    nodes = np.linspace(0.001, 0.999, 61)

    def posterior_around(t, sd=0.05):
        w = np.exp(-0.5 * ((nodes[None, :] - t[:, None]) / sd) ** 2)
        return nodes, w / w.sum(axis=1, keepdims=True)

    out = pvmod.pv_correct(posterior_around(truth_x), posterior_around(truth_y),
                           n_draws=8, seed=2)
    assert 0.0 <= out["pi_mean"] <= 1.0
    assert out["pi_sd"] >= 0.0
    assert out["pi_draws"].size == 8
    assert out["pi_ci"][0] <= out["pi_mean"] <= out["pi_ci"][1]
    assert out["delta_mean"] > 0  # the ceiling direction survives scoring error


def test_ctm_posterior_wrapper_or_clear_error():
    """With the optional extra installed the wrapper returns a grid posterior."""
    ctm = pytest.importorskip("cogtraitmodel")
    rng = np.random.default_rng(12)
    theta = rng.uniform(0.1, 0.9, 50)
    alpha = np.full(10, 8.0)
    beta = np.linspace(0.2, 0.8, 10)
    Y = ctm.gen_responses(theta, alpha, beta, rng=np.random.default_rng(13))
    nodes, post = pvmod.ctm_posterior(Y, alpha, beta)
    assert nodes.shape == (61,)
    assert post.shape == (50, 61)
    assert np.allclose(post.sum(axis=1), 1.0)


# ----------------------------------------------------------------- study
def test_run_study_returns_a_tidy_frame():
    pd = pytest.importorskip("pandas")
    cfg = dict(studymod.EXAMPLE_CONFIG, n=200, reps=3)
    df = studymod.run_study(cfg)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert {"name", "rep", "seed", "PI", "A1", "A2", "q", "ell", "Delta"} <= set(df.columns)
    assert df["seed"].tolist() == [cfg["seed"] + r for r in range(3)]


def test_run_study_is_reproducible():
    pytest.importorskip("pandas")
    cfg = dict(studymod.EXAMPLE_CONFIG, n=200, reps=2)
    a = studymod.run_study(cfg)
    b = studymod.run_study(cfg)
    assert np.allclose(a["PI"].to_numpy(), b["PI"].to_numpy())


def test_study_models_behave_as_advertised():
    pytest.importorskip("pandas")
    prod = studymod.run_study(dict(studymod.EXAMPLE_CONFIG, n=600, reps=3))
    indep = studymod.run_study(dict(studymod.EXAMPLE_CONFIG, model="independent",
                                    n=600, reps=3))
    equiv = studymod.run_study(dict(studymod.EXAMPLE_CONFIG, model="equivalence",
                                    n=600, reps=3))
    assert prod["PI"].mean() > 0.8
    assert indep["PI"].mean() < 0.1
    assert equiv["PI"].max() == 0.0


# ------------------------------------------------- generators and guards
def test_study_ceiling_and_free_specs():
    """Phase C leans on these branches, so each one is exercised here."""
    pytest.importorskip("pandas")
    base = dict(studymod.EXAMPLE_CONFIG, n=800, reps=2)

    linear = studymod.run_study(dict(base, ceiling=("linear", 0.7)))
    power = studymod.run_study(dict(base, ceiling=("power", 2)))
    beta_free = studymod.run_study(dict(base, free=("beta", 30, 10)))
    weakest = studymod.run_study(dict(base, model="min"))
    noisy = studymod.run_study(dict(base, noise=0.05))

    # a proportional ceiling attenuates towards a / (1 - delta)
    assert abs(linear["PI"].mean() - 0.7 / (1 - 0.05)) < 0.05
    # a curved ceiling attenuates further
    assert power["PI"].mean() < linear["PI"].mean()
    # every configuration stays in range and keeps the forward direction
    for df in (linear, power, beta_free, weakest, noisy):
        assert df["PI"].between(0, 1).all()
        assert (df["Delta"] >= 0).all()


def test_study_rejects_unknown_specs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        studymod.generate_pair({"n": 10, "ceiling": ("quadratic", 2)}, rng)
    with pytest.raises(ValueError):
        studymod.generate_pair({"n": 10, "free": ("gamma", 2, 2)}, rng)
    with pytest.raises(ValueError):
        studymod.generate_pair({"n": 10, "model": "additive"}, rng)


def test_referenced_index_accepts_explicit_indices_and_plain_callables():
    rng = np.random.default_rng(14)
    n = 1200
    x = rng.uniform(0, 1, n)
    y = 0.5 * x * rng.uniform(0, 1, n)
    fit = ceilmod.ceiling_fit(x, y, tau=0.98, seed=0)

    everywhere = ceilmod.prereq_index_referenced(x, y, fit, indices="all")
    subset = ceilmod.prereq_index_referenced(x, y, fit, indices=np.arange(600))
    assert 0.0 <= everywhere["PI"] <= 1.0 and 0.0 <= subset["PI"] <= 1.0

    # a plain function is accepted as a ceiling, with no fitting at all
    known = ceilmod.prereq_index_referenced(x, y, lambda v: 0.5 * v, indices="all")
    assert known["PI"] > prereq_index(x, y)["PI"]


def test_ceiling_guards():
    rng = np.random.default_rng(15)
    x = rng.uniform(0, 1, 100)
    y = x * rng.uniform(0, 1, 100)
    with pytest.raises(ValueError):
        ceilmod.ceiling_fit(x, y[:50])
    with pytest.raises(ValueError):
        ceilmod.ceiling_fit(x, y, method="isotonic-spline")
    with pytest.raises(ValueError):
        ceilmod.prereq_index_referenced(x, y, ceilmod.ceiling_fit(x, y), indices="train")
    with pytest.raises(ValueError):
        ceilmod._fit_monotone_quantile(x[:3], y[:3], 0.95, n_bins=10)


def test_ceiling_fit_without_split_uses_every_point():
    rng = np.random.default_rng(16)
    x = rng.uniform(0, 1, 300)
    y = x * rng.uniform(0, 1, 300)
    fit = ceilmod.ceiling_fit(x, y, split=None)
    assert fit.fit_index.size == 300 and fit.eval_index.size == 300


def test_scan_input_validation():
    with pytest.raises(ValueError):
        scan(np.zeros(10))
    with pytest.raises(ValueError):
        scan(np.zeros((10, 1)))
    with pytest.raises(ValueError):
        scan(np.zeros((10, 3)), names=["A", "B"])


def test_pv_input_validation():
    nodes = np.linspace(0, 1, 5)
    good = np.full((3, 5), 0.2)
    with pytest.raises(ValueError):
        pvmod.draw_pv(nodes, np.full((3, 4), 0.25), 2, np.random.default_rng(0))
    with pytest.raises(ValueError):
        pvmod.pv_correct((nodes, good), (nodes, np.full((2, 5), 0.2)))
