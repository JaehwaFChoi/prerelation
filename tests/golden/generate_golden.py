"""Generate the golden fixtures and expected outputs.

This script is committed for provenance only. The CSV/JSON files it wrote
are the canonical fixtures: consumers (the JS and R implementations, and
``tests/test_golden.py``) read the files and never re-run this script.

Fixture provenance
------------------
Synthetic fixtures use ``numpy.random.default_rng(SeedSequence([20260827, k]))``
with the ``k`` recorded below. The real-data fixture ``ecpe_slice`` is the
first 200 rows (file order) of the ECPE skill-theta table (not committed;
the committed ``fixture_ecpe_slice.csv`` is the canonical data), pair
x = skill3, y = skill2. Ceiling for the synthetic ceilinged sets:
``c(x) = clip(0.15 + 0.85 x^0.8, 0, 1)``.

Permutation contract
--------------------
``perm_indices_n{N}.csv`` holds ``N_PERM`` rows; row r is the r-th
permutation of ``0..N-1`` drawn from ``numpy.random.default_rng(PERM_SEED)``
by ``rng.permutation(N)``, consuming the stream exactly as
``prerelation.core.perm_pvalue`` consumes it. Applying row r as
``y[P[r]]`` therefore reproduces the package's r-th replicate, and

    p = (1 + sum_r [ PI(x, y[P[r]]) >= PI(x, y) ]) / (N_PERM + 1)

must equal ``perm_pvalue(x, y, n_perm=N_PERM, seed=PERM_SEED)[1]`` exactly
-- not merely in distribution. Any implementation that reads the committed
index matrix and computes PI identically must land on the identical p-value.
"""

import csv
import json
import os

import numpy as np

from prerelation.core import direction, perm_pvalue, prereq_index
from prerelation.reference import pi_envelope
from prerelation.scan import condense

HERE = os.path.dirname(os.path.abspath(__file__))
N_PERM = 199
PERM_SEED = 20260827
DELTA = 0.05
TOP_Q = 0.8


# Graph fixtures for the condensation, shared with the JavaScript and R
# packages. Held as literals: the condensation is a property of a graph, so
# it needs no data file. The two cyclic ones are the cases the manuscript's
# equivalence classes rest on and the ones no acyclic reduction can reach.
GRAPHS = {
    "chain_isolate": (["a1", "a2", "a3", "a4"],
                      [("a1", "a2"), ("a1", "a3"), ("a2", "a3")]),
    "two_cycle": (["a", "b", "c"],
                  [("a", "b"), ("b", "a"), ("b", "c")]),
    "shared_node": (["a1", "a2", "a3", "a4", "a5"],
                    [("a1", "a2"), ("a2", "a1"), ("a2", "a3"),
                     ("a3", "a4"), ("a4", "a2"), ("a4", "a5")]),
    "no_edges": (["p", "q", "r"], []),
    "diamond_redundant": (["A", "B", "C", "D"],
                          [("A", "B"), ("B", "D"), ("A", "C"),
                           ("C", "D"), ("A", "D")]),
    "two_cycles_cross": (["a", "b", "c", "d"],
                         [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c"),
                          ("b", "c"), ("a", "d")]),
}

# Settings for the sensitivity block. Every value is a departure from the
# fixed convention and is pinned so that a change to the argument handling
# cannot pass unnoticed; none of them is a recommended setting.
SENS_TOP_Q = [0.6, 0.95]
SENS_MIN_INTERIOR = [0, 200]


def condensation_record(nodes, edges):
    """Canonical rendering, class ids offset to one as the ports render them."""
    r = condense(nodes, edges)
    return {
        "classes": "|".join("+".join(c) for c in r.classes),
        "class_of": ",".join("%s:%d" % (u, r.class_of[u] + 1) for u in nodes),
        "quotient": "-" if not r.quotient_edges else ",".join(
            "%d>%d" % (a + 1, b + 1) for a, b in r.quotient_edges),
        "hasse": "-" if not r.hasse_edges else ",".join(
            "%d>%d" % (a + 1, b + 1) for a, b in r.hasse_edges),
    }


def sensitivity_record(x, y):
    """PI and q away from the default top_q and min_interior.

    The defaults are the fixed convention; these entries exist so that the
    keyword arguments added in 0.4.0 are covered by the contract rather than
    merely present in the signature.
    """
    out = {}
    for t in SENS_TOP_Q:
        r = prereq_index(x, y, top_q=t)
        out["top_q_%g_PI" % t] = repr(r["PI"])
        out["top_q_%g_ell" % t] = repr(r["ell"])
    for mi in SENS_MIN_INTERIOR:
        r = prereq_index(x, y, min_interior=mi)
        out["min_interior_%d_PI" % mi] = repr(r["PI"])
        out["min_interior_%d_q" % mi] = repr(r["q"])
    return out


def ceiling(x):
    return np.clip(0.15 + 0.85 * x ** 0.8, 0.0, 1.0)


def synthetic_fixtures():
    out = {}
    n = 400

    def rng_for(k):
        return np.random.default_rng(np.random.SeedSequence([20260827, k]))

    r = rng_for(1)
    x = r.uniform(0.02, 0.98, n)
    out["product"] = (x, np.clip(ceiling(x) * r.uniform(0.0, 1.0, n), 0.0, 1.0))

    r = rng_for(2)
    x = r.uniform(0.02, 0.98, n)
    out["min"] = (x, np.clip(np.minimum(ceiling(x), r.uniform(0.0, 1.0, n)), 0.0, 1.0))

    r = rng_for(3)
    x = r.uniform(0.02, 0.98, n)
    out["independent"] = (x, r.uniform(0.0, 1.0, n))

    r = rng_for(4)
    x = r.uniform(0.02, 0.98, n)
    out["equivalence"] = (x, x.copy())

    r = rng_for(5)
    x = r.uniform(0.02, 0.98, n)
    out["partial_equivalence"] = (
        x,
        np.clip(0.7 * x + 0.3 * r.uniform(0.0, 1.0, n), 0.0, 1.0),
    )
    return out


def ecpe_fixture():
    path = os.path.join(HERE, "ecpe_theta_full.csv")
    if not os.path.exists(path):
        # The full theta table is not part of the repository. Regenerating
        # from the committed fixture is exact: the CSV stores repr(float64).
        path = os.path.join(HERE, "fixture_ecpe_slice.csv")
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))[1:]
        return (np.array([float(r[0]) for r in rows]),
                np.array([float(r[1]) for r in rows]))
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:201]  # first 200 persons, file order
    i_x = header.index("skill3")
    i_y = header.index("skill2")
    x = np.array([float(r[i_x]) for r in data])
    y = np.array([float(r[i_y]) for r in data])
    return x, y


def components(x, y):
    """Component-wise outputs, computed from the definition."""
    n = x.size
    v = float(np.mean(np.maximum(y - x, 0.0)))
    xs = x[:, None]
    v0 = float(np.mean(np.maximum(y[None, :] - xs, 0.0)))  # V-statistic, diag incl.
    res = prereq_index(x, y)
    rev = prereq_index(y, x)
    dl = direction(x, y)
    env = pi_envelope(x, y)
    assert env["attained"], "closed-form supremum not attained on a fixture"
    u = np.clip(y / np.maximum(x, 1e-12), 0.0, 1.0)
    ceil_mask = u >= 1.0 - DELTA
    x_top = x >= np.quantile(x, TOP_Q)
    p1_top = float(np.mean(ceil_mask[x_top])) if x_top.sum() > 0 else 1.0
    return {
        "n": n,
        "v": repr(v),
        "v0": repr(v0),
        "A1": repr(res["A1"]),
        "mass_ceiling_band": repr(float(np.mean(ceil_mask))),
        "mass_interior": repr(float(np.mean(~ceil_mask))),
        "n_interior": int(np.sum(~ceil_mask)),
        "p1_top": repr(p1_top),
        "q": repr(res["q"]),
        "ell": repr(res["ell"]),
        "A2": repr(res["A2"]),
        "PI": repr(res["PI"]),
        "PI_reverse": repr(rev["PI"]),
        "Delta": repr(dl[0]),
        # reference class and envelope (Propositions 2.20-2.21): the exact
        # supremum of q over the admissible class B, the vacuous infimum
        # 1/m, and the upper envelope PI_hi = A1 * ell * sup_q. All closed
        # forms; n_tail_band counts interior points with t >= 1 - delta.
        "n_tail_band": env["n_tail"],
        "D_star": repr(env["D_star"]),
        "sup_q": repr(env["sup_q"]),
        "inf_q": repr(env["inf_q"]),
        "PI_hi": repr(env["PI_hi"]),
    }


def main():
    fixtures = synthetic_fixtures()
    fixtures["ecpe_slice"] = ecpe_fixture()

    perm = {}
    for n in sorted({x.size for x, _ in fixtures.values()}):
        rng = np.random.default_rng(PERM_SEED)
        P = np.stack([rng.permutation(n) for _ in range(N_PERM)])
        perm[n] = P
        with open(os.path.join(HERE, f"perm_indices_n{n}.csv"), "w") as fh:
            for row in P:
                fh.write(",".join(str(int(i)) for i in row) + "\n")

    expected = {"_contract": {"n_perm": N_PERM, "perm_seed": PERM_SEED,
                              "delta": DELTA, "top_q": TOP_Q}}
    sensitivity = {}
    for name, (x, y) in fixtures.items():
        with open(os.path.join(HERE, f"fixture_{name}.csv"), "w") as fh:
            fh.write("x,y\n")
            for a, b in zip(x, y):
                fh.write(f"{float(a)!r},{float(b)!r}\n")
        comp = components(x, y)
        obs = prereq_index(x, y)["PI"]
        P = perm[x.size]
        cnt = sum(prereq_index(x, y[P[r]])["PI"] >= obs for r in range(N_PERM))
        p_matrix = (cnt + 1) / (N_PERM + 1)
        obs2, p_pkg = perm_pvalue(x, y, n_perm=N_PERM, seed=PERM_SEED)
        assert obs2 == obs
        assert p_pkg == p_matrix, (name, p_pkg, p_matrix)
        comp["perm_p"] = repr(p_matrix)
        expected[name] = comp
        sensitivity[name] = sensitivity_record(x, y)

    expected["_condense"] = {name: condensation_record(*g)
                             for name, g in GRAPHS.items()}
    expected["_sensitivity"] = sensitivity

    with open(os.path.join(HERE, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=2, sort_keys=True)
    print("fixtures:", ", ".join(sorted(fixtures)))
    print("perm matrices:", ", ".join(f"n{n}" for n in sorted(perm)))
    print("package p == matrix p for every fixture: OK")
    print("condensation fixtures:", ", ".join(sorted(GRAPHS)))


if __name__ == "__main__":
    main()
