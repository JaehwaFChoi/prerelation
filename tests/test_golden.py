"""Golden-vector pins: the committed fixtures must reproduce exactly.

Every quantity in ``tests/golden/expected.json`` is regenerated here from
the committed fixture CSVs and compared at 1e-12 (integer counts and the
permutation p-values must match exactly). This is what makes the Python
package the reference implementation: any change that moves these numbers
is a behavioral change and must fail loudly.
"""

import csv
import json
import os

import numpy as np
import pytest

from prerelation.core import direction, prereq_index
from prerelation.reference import pi_envelope
from prerelation.scan import condense

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
TOL = 1e-12

with open(os.path.join(GOLDEN, "expected.json")) as _fh:
    EXPECTED = json.load(_fh)
CONTRACT = EXPECTED["_contract"]
FIXTURES = sorted(k for k in EXPECTED if not k.startswith("_"))
CONDENSE = EXPECTED["_condense"]
SENSITIVITY = EXPECTED["_sensitivity"]

# Graph fixtures, held here as literals exactly as the generator holds them.
# A condensation is a property of a graph and needs no data file.
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


@pytest.mark.parametrize("name", sorted(CONDENSE))
def test_condensation_pinned(name):
    """The condensation is pinned in the same rendering the ports print.

    Class ids are offset to one here because the R package labels from one;
    the rendering is what the cross-language comparison reads, so it is the
    rendering that is pinned.
    """
    nodes, edges = GRAPHS[name]
    r = condense(nodes, edges)
    exp = CONDENSE[name]
    assert "|".join("+".join(c) for c in r.classes) == exp["classes"]
    assert ",".join("%s:%d" % (u, r.class_of[u] + 1) for u in nodes) == exp["class_of"]
    quotient = "-" if not r.quotient_edges else ",".join(
        "%d>%d" % (a + 1, b + 1) for a, b in r.quotient_edges)
    hasse = "-" if not r.hasse_edges else ",".join(
        "%d>%d" % (a + 1, b + 1) for a, b in r.hasse_edges)
    assert quotient == exp["quotient"]
    assert hasse == exp["hasse"]


@pytest.mark.parametrize("name", sorted(SENSITIVITY))
def test_non_default_top_q_and_min_interior_pinned(name):
    """The 0.4.0 keyword arguments are covered by the contract, not merely present.

    None of these settings is a recommended one: the fixed conventions are
    delta = 0.05, TOP_Q = 0.8 and MIN_INTERIOR = 10, and the values here are
    departures from them, pinned so that a change to how the arguments are
    threaded cannot pass unnoticed.
    """
    x, y = load_fixture(name)
    exp = SENSITIVITY[name]
    for key, value in exp.items():
        setting, _, component = key.rpartition("_")
        if setting.startswith("top_q_"):
            res = prereq_index(x, y, top_q=float(setting[len("top_q_"):]))
        else:
            res = prereq_index(x, y,
                               min_interior=int(setting[len("min_interior_"):]))
        assert abs(res[component] - float(value)) <= TOL, key


def test_the_defaults_are_still_the_fixed_conventions():
    """The arguments must not have moved the definition (P4-P1)."""
    from prerelation.core import DELTA, MIN_INTERIOR, TOP_Q
    assert (DELTA, TOP_Q, MIN_INTERIOR) == (0.05, 0.8, 10)
    assert CONTRACT["delta"] == DELTA and CONTRACT["top_q"] == TOP_Q
    x, y = load_fixture("product")
    assert prereq_index(x, y) == prereq_index(x, y, top_q=TOP_Q,
                                              min_interior=MIN_INTERIOR)


def load_fixture(name):
    with open(os.path.join(GOLDEN, f"fixture_{name}.csv"), newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    x = np.array([float(r[0]) for r in rows])
    y = np.array([float(r[1]) for r in rows])
    return x, y


def load_perm(n):
    path = os.path.join(GOLDEN, f"perm_indices_n{n}.csv")
    with open(path, newline="") as fh:
        P = np.array([[int(v) for v in row] for row in csv.reader(fh)])
    assert P.shape == (CONTRACT["n_perm"], n)
    return P


@pytest.mark.parametrize("name", FIXTURES)
def test_components_pinned(name):
    exp = EXPECTED[name]
    x, y = load_fixture(name)
    assert x.size == exp["n"]

    delta = CONTRACT["delta"]
    top_q = CONTRACT["top_q"]

    v = float(np.mean(np.maximum(y - x, 0.0)))
    v0 = float(np.mean(np.maximum(y[None, :] - x[:, None], 0.0)))
    assert abs(v - float(exp["v"])) <= TOL
    assert abs(v0 - float(exp["v0"])) <= TOL

    res = prereq_index(x, y)
    for key in ("A1", "q", "ell", "A2", "PI"):
        assert abs(res[key] - float(exp[key])) <= TOL, (name, key)
    # A1 must be consistent with the definitional components
    assert abs(res["A1"] - max(0.0, 1.0 - v / v0)) <= TOL

    u = np.clip(y / np.maximum(x, 1e-12), 0.0, 1.0)
    ceil_mask = u >= 1.0 - delta
    assert abs(float(np.mean(ceil_mask)) - float(exp["mass_ceiling_band"])) <= TOL
    assert abs(float(np.mean(~ceil_mask)) - float(exp["mass_interior"])) <= TOL
    assert int(np.sum(~ceil_mask)) == exp["n_interior"]
    x_top = x >= np.quantile(x, top_q)
    p1_top = float(np.mean(ceil_mask[x_top])) if x_top.sum() > 0 else 1.0
    assert abs(p1_top - float(exp["p1_top"])) <= TOL

    rev = prereq_index(y, x)
    assert abs(rev["PI"] - float(exp["PI_reverse"])) <= TOL
    dl = direction(x, y)
    assert abs(dl[0] - float(exp["Delta"])) <= TOL


@pytest.mark.parametrize("name", FIXTURES)
def test_permutation_contract_pinned(name):
    exp = EXPECTED[name]
    x, y = load_fixture(name)
    P = load_perm(x.size)
    obs = prereq_index(x, y)["PI"]
    n_perm = CONTRACT["n_perm"]
    cnt = sum(prereq_index(x, y[P[r]])["PI"] >= obs for r in range(n_perm))
    p = (cnt + 1) / (n_perm + 1)
    assert p == float(exp["perm_p"]), (name, p, exp["perm_p"])


@pytest.mark.parametrize("name", FIXTURES)
def test_envelope_pinned(name):
    """Reference class and envelope (closed forms) pinned per fixture."""
    exp = EXPECTED[name]
    x, y = load_fixture(name)
    env = pi_envelope(x, y)
    assert env["n_tail"] == exp["n_tail_band"]
    for key in ("D_star", "sup_q", "inf_q", "PI_hi"):
        assert abs(env[key] - float(exp[key])) <= TOL, (name, key)
    assert env["attained"]
