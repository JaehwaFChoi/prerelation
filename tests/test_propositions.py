"""The propositions of the theory note, re-run against the package.

This is a port of the verification script that accompanies the theory note
(``verify_props_v1.py``): same seed, same draws, same twenty-six checks, but
the coefficient comes from ``prerelation.core`` instead of the reference
implementation. An assertion failure here is a signal to re-examine the
statement, not the code.

The numbers are preliminary. Manuscript numbers are regenerated from logged
runs; nothing in a test file is a result.
"""

import numpy as np
import pytest

from prerelation.core import DELTA, direction, perm_pvalue, prereq_index

SEED = 20260826
CHECKS = []


def log(tag, msg):
    CHECKS.append(f"[{tag}] {msg}")


@pytest.fixture(scope="module")
def state():
    """Draw exactly as the verification script draws, in the same order."""
    rng = np.random.default_rng(SEED)
    n = 5000
    X = rng.uniform(0, 1, n)
    U = rng.uniform(0, 1, n)
    return {"rng": rng, "n": n, "X": X, "U": U}


# ------------------------------------------------------------------ T1
def test_t1_extremes(state):
    rng, n, X, U = state["rng"], state["n"], state["X"], state["U"]

    r = prereq_index(X, X * U)
    assert r["A1"] == 1.0
    assert r["PI"] > 0.95
    assert abs(r["PI"] - 0.9740) < 1e-3
    log("T1a", f"pure product (U~Unif): PI={r['PI']:.4f} (pop=1)")

    Yind = rng.uniform(0, 1, n)
    r = prereq_index(X, Yind)
    assert r["PI"] < 0.05
    log("T1b", f"independent: PI={r['PI']:.4f} A1={r['A1']:.4f} (pop 0)")

    r = prereq_index(X, X.copy())
    assert r["PI"] == 0.0 and r["q"] == 0.0 and r["ell"] == 0.0
    log("T1c", "exact equivalence: PI=0 exactly")

    r = prereq_index(X, np.zeros(n))
    assert r["PI"] == 0.0 and r["A1"] == 0.0 and r["q"] == 0.0
    log("T1d", "degenerate Y=0: PI=0 exactly (guard + interior degeneracy)")

    r = prereq_index(X, np.full(n, 0.3))
    assert r["A1"] < 1e-12 and r["PI"] < 1e-12
    log("T1d", f"degenerate Y=0.3: A1={r['A1']:.2e} (v=v0 identity, fp-eps)")


# ------------------------------------------------------------------ T2
def test_t2_calibrated_attenuation(state):
    n, X, U = state["n"], state["X"], state["U"]

    a = 0.7
    Y = a * X * U
    r_id, r_c = prereq_index(X, Y), prereq_index(a * X, Y)
    pop_id = a / (1 - DELTA)
    assert r_c["PI"] > 0.95
    assert abs(r_id["PI"] - pop_id) < 0.03
    assert abs(r_id["PI"] - 0.7364) < 1e-3
    assert r_id["PI"] <= r_c["PI"]
    log("T2", f"c=0.7x: PI_id={r_id['PI']:.4f} (pop {pop_id:.4f}) <= PI_c={r_c['PI']:.4f}")

    Y = (X ** 2) * U
    r_id, r_c = prereq_index(X, Y), prereq_index(X ** 2, Y)
    assert r_c["PI"] > 0.95 and r_id["PI"] < r_c["PI"]
    log("T2", f"c=x^2: PI_id={r_id['PI']:.4f} (pop~0.632) <= PI_c={r_c['PI']:.4f}")

    for aa in (0.97, 0.90):
        r_id = prereq_index(X, aa * X * U)
        band = "inside" if aa >= 1 - DELTA else "outside"
        log("T2", f"c={aa}x ({band} band): PI_id={r_id['PI']:.4f}")
    assert prereq_index(X, 0.97 * X * U)["PI"] > 0.95


def test_t2_counterexamples_to_the_unconditional_form(state):
    """Without a uniform free component the attenuation inequality fails."""
    rng, n, X = state["rng"], state["n"], state["X"]

    log("T2", "-- counterexample search (non-uniform U) --")
    found = []
    for uname, udraw in [
        ("Beta(30,10)", lambda m: rng.beta(30, 10, m)),
        ("Beta(8,2)", lambda m: rng.beta(8, 2, m)),
        ("Beta(5,5)", lambda m: rng.beta(5, 5, m)),
        ("Beta(2,8)", lambda m: rng.beta(2, 8, m)),
    ]:
        for cname, cfun in [("x^2", lambda x: x ** 2), ("x^3", lambda x: x ** 3)]:
            Unu = udraw(n)
            Y = cfun(X) * Unu
            pid = prereq_index(X, Y)["PI"]
            pc = prereq_index(cfun(X), Y)["PI"]
            mark = "  << counterexample" if pid > pc + 0.01 else ""
            log("T2", f"  U~{uname:11s} c={cname}: PI_id={pid:.4f} PI_c={pc:.4f}{mark}")
            if pid > pc + 0.01:
                found.append((uname, cname, pid, pc))

    assert len(found) == 4, found
    assert abs(found[0][2] - 0.8104) < 1e-3 and abs(found[0][3] - 0.3828) < 1e-3
    log("T2", f"{len(found)} counterexamples -> T2 holds only for U ~ Uniform(0,1)")


# ------------------------------------------------------------------ T3
def test_t3_equivalence_is_not_identified(state):
    rng, n, X = state["rng"], state["n"], state["X"]

    assert np.array_equal(X.copy(), np.minimum(X, np.ones(n)))
    log("T3", "equivalence (X,X) == min model with T=1, sample for sample")

    T = rng.uniform(0, 1, n)
    frac = float(np.mean(np.minimum(X, T) < X))
    assert frac > 0.4
    log("T3", f"freedom postulate (T~Unif): P(Y<X) = {frac:.3f} (pop 0.5)")


# ------------------------------------------------------------------ T4
def test_t4_direction_asymmetry(state):
    rng, n, X, U = state["rng"], state["n"], state["X"], state["U"]

    d, pf, pr = direction(X, X * U)
    assert pr == 0.0 and d > 0.9
    log("T4", f"product: PI_fwd={pf:.4f} PI_rev={pr} (exactly 0) Delta={d:+.4f}")

    T = rng.uniform(0, 1, n)
    d2, pf2, pr2 = direction(X, np.minimum(X, T))
    assert pr2 == 0.0
    assert abs(pf2 - 0.9) < 0.04
    assert abs(pf2 - 0.8854) < 1e-3
    log("T4", f"min model: PI_fwd={pf2:.4f} (pop 0.9 = E[X|top]) Delta={d2:+.4f}")


# ------------------------------------------------------------------ T5
@pytest.mark.slow
def test_t5_permutation_test_holds_its_level():
    n5, nperm, reps = 200, 199, 400
    rej = 0
    for i in range(reps):
        rg = np.random.default_rng(50_000 + i)
        xs = rg.uniform(0, 1, n5)
        ys = rg.uniform(0, 1, n5)
        _, p = perm_pvalue(xs, ys, n_perm=nperm, seed=90_000 + i)
        rej += p <= 0.05
    rate = rej / reps
    band = 3 * np.sqrt(0.05 * 0.95 / reps)
    assert rate <= 0.05 + band
    assert abs(rate - 0.0600) < 1e-9
    log("T5", f"H0 independent: rejection {rate:.4f} <= {0.05 + band:.4f}")


# ------------------------------------------------------------------ T6
def test_t6_consistency():
    pop = 0.7 / (1 - DELTA)
    errs = []
    for nn in (500, 2000, 6000):
        e = []
        for s in range(3):
            rg = np.random.default_rng(100 + s)
            x = rg.uniform(0, 1, nn)
            u = rg.uniform(0, 1, nn)
            e.append(abs(prereq_index(x, 0.7 * x * u)["PI"] - pop))
        errs.append(float(np.mean(e)))
    assert errs[-1] < 0.02 and errs[0] > errs[-1]
    assert abs(errs[0] - 0.0028) < 1e-3 and abs(errs[2] - 0.0001) < 1e-3
    log("T6", f"c=0.7x mean |error|: {errs[0]:.4f} -> {errs[1]:.4f} -> {errs[2]:.4f}")

    tr = []
    for nn in (500, 2000, 6000):
        rg = np.random.default_rng(300)
        x = rg.uniform(0, 1, nn)
        u = rg.uniform(0, 1, nn)
        tr.append(prereq_index(x, x * u)["PI"])
    assert tr[-1] > tr[0] and tr[-1] > 0.97
    log("T6", f"pure product PI trend: {tr[0]:.4f} -> {tr[1]:.4f} -> {tr[2]:.4f} -> 1")


# ------------------------------------------------------------------ count
@pytest.mark.slow
def test_all_twenty_six_checks_ran():
    """The verification protocol is twenty-six items; all of them must run."""
    assert len(CHECKS) == 26, (len(CHECKS), CHECKS)
