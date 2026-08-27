"""Tests for the ceiling_fit postulate correction (D16).

Basis, Proposition T7: under the product form Y = c(X)U with U~Uniform(0,1)
independent of X, the tau-envelope limit is tau*c, and c_corr = min(c_hat/tau, 1)
is consistent for c. It does not hold for the min form (overestimation). The
oracle (prereq_index) is unchanged -- this option lives only in the ceiling_fit
layer and is irrelevant to JS parity (parity covers core closed forms and scan).
"""
import numpy as np
import pytest

from prerelation import ceiling as ceilmod


def _product_uniform(n, a=0.6, seed=11):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    u = rng.uniform(0.0, 1.0, n)
    return x, a * x * u


def test_correction_off_is_bit_identical():
    # Default (off) and explicit False must be bit-identical to prior outputs (regression).
    x, y = _product_uniform(4000)
    grid = np.linspace(0.05, 0.95, 40)
    fit_default = ceilmod.ceiling_fit(x, y, tau=0.95, seed=0)
    fit_false = ceilmod.ceiling_fit(x, y, tau=0.95, seed=0,
                                    postulate_correction=False)
    assert np.array_equal(fit_default.c_knots, fit_false.c_knots)
    assert np.array_equal(fit_default.predict(grid), fit_false.predict(grid))
    assert fit_default.truncation_rate is None
    assert fit_false.truncation_rate is None


def test_correction_recovers_the_product_uniform_ceiling():
    # T7 verification case: product x uniform, c = 0.6x, tau = 0.95.
    # The uncorrected envelope estimates tau*c (systematically low); correction recovers c.
    x, y = _product_uniform(20000, a=0.6, seed=21)
    grid = np.linspace(0.15, 0.9, 25)
    truth = 0.6 * grid
    raw = ceilmod.ceiling_fit(x, y, tau=0.95, seed=0)
    corr = ceilmod.ceiling_fit(x, y, tau=0.95, seed=0,
                               postulate_correction=True)
    err_raw = float(np.max(np.abs(raw.predict(grid) - truth)))
    err_corr = float(np.max(np.abs(corr.predict(grid) - truth)))
    # Dominant term of the uncorrected error: the (1-tau)*c = 0.03*x systematic bias.
    assert err_corr < err_raw
    assert err_corr < 0.04
    # Truncation-rate report: c_raw/tau <= 0.6/0.95 < 1, so truncation should be virtually absent.
    assert corr.truncation_rate is not None
    assert 0.0 <= corr.truncation_rate <= 0.02


def test_correction_overestimates_under_the_weakest_link_form():
    # Misapplication demo (docstring item (a)): min form with c = id gives corrected/c > 1.
    rng = np.random.default_rng(31)
    n = 20000
    x = rng.uniform(0.0, 1.0, n)
    t = rng.uniform(0.0, 1.0, n)
    y = np.minimum(x, t)
    grid = np.linspace(0.15, 0.85, 20)
    corr = ceilmod.ceiling_fit(x, y, tau=0.95, seed=0,
                               postulate_correction=True)
    ratio = corr.predict(grid) / grid  # c(x) = x
    # Theory: min(c/tau, 1)/c = 1/tau (= 1.0526) -- exceeds 1 across the whole grid.
    assert np.all(ratio > 1.0)
    assert abs(float(np.median(ratio)) - 1.0 / 0.95) < 0.06


def test_correction_reports_truncation_at_the_upper_bound():
    # c = id (a=1.0): envelope ~ 0.95x, correction ~ x; knots near the top that
    # exceed 1 are truncated, and the fraction is reported as truncation_rate.
    x, y = _product_uniform(20000, a=1.0, seed=41)
    corr = ceilmod.ceiling_fit(x, y, tau=0.95, seed=0,
                               postulate_correction=True)
    assert corr.truncation_rate is not None
    assert 0.0 <= corr.truncation_rate <= 1.0
    assert float(np.max(corr.predict(np.linspace(0, 1, 50)))) <= 1.0


def test_correction_applies_to_the_ctm_method_too():
    pytest.importorskip("scipy")
    x, y = _product_uniform(8000, a=0.7, seed=51)
    corr = ceilmod.ceiling_fit(x, y, tau=0.95, method="ctm", seed=0,
                               postulate_correction=True)
    raw = ceilmod.ceiling_fit(x, y, tau=0.95, method="ctm", seed=0)
    grid = np.linspace(0.2, 0.8, 10)
    assert np.all(corr.predict(grid) >= raw.predict(grid))
    assert corr.truncation_rate is not None
