"""The admissible reference class and the exact upper envelope.

What is checked:

* the admissibility check is the class ``B`` (a ceiling condition), not
  first-order stochastic dominance by Uniform;
* ``interior_q`` at the Uniform reference reproduces ``core`` bit for bit;
* the closed-form supremum is an upper bound for every admissible
  reference tried, is attained by the exhibited member of ``B``, and moves
  when the exhibited member is perturbed;
* the infimum ``1/m`` is attained by the point mass at 0;
* ``reference.py`` imports ``core`` and nothing else from the package.
"""

import ast
import csv
import os
import pathlib

import numpy as np
import pytest

from prerelation import reference
from prerelation.core import prereq_index
from prerelation.reference import (
    admissibility,
    attaining_reference,
    beta_reference,
    interior_q,
    pi_envelope,
    point_mass_reference,
    prereq_index_family,
    uniform_reference,
)

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
with open(os.path.join(GOLDEN, "expected.json")) as _fh:
    _EXP = __import__("json").load(_fh)
EXPECTED_PI = {k: v["PI"] for k, v in _EXP.items() if not k.startswith("_")}
# The golden contract compares floats at 1e-12; see tests/golden/README.md.
GOLDEN_TOL = 1e-12
FIXTURES = ["product", "min", "independent", "equivalence", "partial_equivalence", "ecpe_slice"]
TOL = 1e-12


def load_fixture(name):
    with open(os.path.join(GOLDEN, f"fixture_{name}.csv"), newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    return np.array([float(r[0]) for r in rows]), np.array([float(r[1]) for r in rows])


# ---------------------------------------------------------------- admissibility

def test_uniform_is_the_boundary_point():
    a = admissibility(uniform_reference())
    assert a.admissible
    assert abs(a.tail_mass - 0.05) <= 1e-12
    assert abs(a.worst_slack) <= 1e-12


def test_point_mass_at_zero_is_admissible():
    assert admissibility(point_mass_reference(0.0)).admissible


@pytest.mark.parametrize("a, b, tail", [(2, 1, 0.0975), (8, 2, 0.0712), (20, 1, 0.6415)])
def test_ceiling_loaded_references_are_rejected(a, b, tail):
    r = admissibility(beta_reference(a, b))
    assert not r.admissible
    assert abs(r.tail_mass - tail) < 5e-5
    assert r.worst_slack < 0


def test_membership_is_weaker_than_dominance():
    """Beta(2,10) fails dominance by Uniform at the floor yet belongs to B."""
    F = beta_reference(2, 10)
    assert float(F(np.array([0.01]))[0]) < 0.01          # dominance fails
    r = admissibility(F)
    assert r.admissible                                    # B accepts it
    assert r.tail_mass < 1e-11


def test_admissibility_rejects_non_callable():
    with pytest.raises(TypeError):
        admissibility((2.0, 10.0))


# ---------------------------------------------------------------- interior_q

@pytest.mark.parametrize("name", FIXTURES)
def test_interior_q_at_uniform_matches_core_bitwise(name):
    x, y = load_fixture(name)
    assert interior_q(x, y) == prereq_index(x, y)["q"]
    assert interior_q(x, y, uniform_reference()) == prereq_index(x, y)["q"]


# ---------------------------------------------------------------- envelope

@pytest.mark.parametrize("name", FIXTURES)
def test_envelope_bounds_every_admissible_reference(name):
    x, y = load_fixture(name)
    e = pi_envelope(x, y)
    assert 0.0 <= e["sup_q"] <= 1.0
    assert e["PI_hi"] >= e["PI"] - TOL
    assert abs(e["PI_hi"] - e["A1"] * e["ell"] * e["sup_q"]) <= TOL
    for F in (uniform_reference(), beta_reference(2, 10), beta_reference(1, 2),
              beta_reference(5, 5), point_mass_reference(0.0)):
        assert admissibility(F).admissible
        assert interior_q(x, y, F) <= e["sup_q"] + TOL, F.__name__


@pytest.mark.parametrize("name", [n for n in FIXTURES if n != "equivalence"])
def test_supremum_is_attained_by_the_exhibited_member(name):
    x, y = load_fixture(name)
    e = pi_envelope(x, y)
    Fs = attaining_reference(x, y)
    assert admissibility(Fs).admissible
    assert abs(interior_q(x, y, Fs) - e["sup_q"]) <= TOL
    assert e["attained"]


@pytest.mark.parametrize("name", [n for n in FIXTURES if n != "equivalence"])
def test_infimum_is_one_over_m_via_point_mass(name):
    x, y = load_fixture(name)
    e = pi_envelope(x, y)
    assert abs(e["inf_q"] - 1.0 / e["m"]) <= TOL
    assert abs(interior_q(x, y, point_mass_reference(0.0)) - e["inf_q"]) <= TOL


def test_guard_case_is_all_zero():
    x, y = load_fixture("equivalence")
    e = pi_envelope(x, y)
    assert e["m"] == 0
    assert e["sup_q"] == 0.0 and e["inf_q"] == 0.0 and e["PI_hi"] == 0.0


def test_d_star_is_zero_without_tail_band_points():
    """A floor-loaded interior has nothing above 1 - delta: sup_q = 1."""
    rng = np.random.default_rng(np.random.SeedSequence([20260902, 1]))
    x = rng.uniform(0.2, 1.0, 3000)
    y = x * rng.beta(2, 10, 3000)
    e = pi_envelope(x, y)
    assert e["n_tail"] == 0
    assert e["D_star"] == 0.0 and e["sup_q"] == 1.0


def test_perturbed_attainer_does_not_attain():
    """A reference that is admissible but not the exhibited member falls
    short of the supremum: the closed form is not a tautology."""
    x, y = load_fixture("product")
    e = pi_envelope(x, y)
    Fs = attaining_reference(x, y)

    def F_pert(t):                       # push the tail band up by 0.01
        t = np.asarray(t, dtype=float)
        return np.clip(Fs(t) + 0.01 * (t >= 0.95), 0.0, 1.0)

    assert admissibility(F_pert).admissible
    assert interior_q(x, y, F_pert) < e["sup_q"] - 1e-6


# ---------------------------------------------------------------- coupling

def test_reference_imports_core_only():
    path = pathlib.Path(reference.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    intra = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            if node.module:
                intra.add(node.module)
            else:                        # ``from . import core``
                intra.update(a.name for a in node.names)
    assert intra <= {"core"}, intra


def test_positive_part_in_d_star_matters():
    """A tail-band point with t_(i) <= i/m contributes 0, not a negative
    amount: sup_q must not exceed 1. Built by hand so the tail band is
    non-empty and every tail-band point sits at or below its index."""
    m = 100
    t = np.concatenate([np.linspace(0.005, 0.90, m - 1), [0.96]])   # t_(m) = 0.96 < m/m
    x = np.full(m, 0.5)
    y = x * t * (1.0 - 0.05)                                          # u = t (1 - delta)
    e = pi_envelope(x, y)
    assert e["n_tail"] == 1
    assert e["D_star"] == 0.0
    assert e["sup_q"] == 1.0
    assert e["attained"]


# ---------------------------------------------------------------- family member

@pytest.mark.parametrize("name", FIXTURES)
def test_family_member_at_uniform_equals_pi_bitwise(name):
    """PI(Uniform) == prereq_index PI with ==, not a tolerance; the key PI
    is already in expected.json, so no new golden key is needed."""
    x, y = load_fixture(name)
    fam = prereq_index_family(x, y)
    ref = prereq_index(x, y)
    assert fam["PI"] == ref["PI"]
    assert fam["A2"] == ref["A2"] and fam["q"] == ref["q"]
    assert fam["PI"] == float(EXPECTED_PI[name])


@pytest.mark.parametrize("name", [n for n in FIXTURES if n != "equivalence"])
def test_family_member_is_bounded_by_the_envelope(name):
    x, y = load_fixture(name)
    e = pi_envelope(x, y)
    for F in (beta_reference(2, 10), beta_reference(1, 2), point_mass_reference(0.0)):
        assert prereq_index_family(x, y, F)["PI"] <= e["PI_hi"] + TOL
    assert abs(prereq_index_family(x, y, attaining_reference(x, y))["PI"] - e["PI_hi"]) <= TOL
