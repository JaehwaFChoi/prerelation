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

HERE = os.path.dirname(os.path.abspath(__file__))
N_PERM = 199
PERM_SEED = 20260827
DELTA = 0.05
TOP_Q = 0.8


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

    with open(os.path.join(HERE, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=2, sort_keys=True)
    print("fixtures:", ", ".join(sorted(fixtures)))
    print("perm matrices:", ", ".join(f"n{n}" for n in sorted(perm)))
    print("package p == matrix p for every fixture: OK")


if __name__ == "__main__":
    main()
