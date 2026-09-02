"""The admissible reference class and the exact upper envelope.

The interior component ``q`` of the prerelation coefficient compares the
rescaled interior of ``u = y / x`` with a reference law on ``[0, 1]``. The
package definition fixes that reference at Uniform(0, 1). This module
generalises the reference to an exogenously *declared* law ``F0`` and
supplies the two objects that the generalisation calls for:

* the **admissible reference class**

      B = { F0 on [0, 1] : F0(t) >= t for all t in [1 - delta, 1] },

  i.e. references that place no more mass near the ceiling than Uniform
  does. Uniform meets the condition with equality and is the boundary
  point of ``B``. Membership is strictly weaker than first-order
  stochastic dominance by Uniform: a floor-loaded reference such as
  Beta(2, 10) fails dominance at the floor and still belongs to ``B``.

* the **exact upper envelope** of the interior component over ``B``:

      sup_{F0 in B} q(F0) = 1 - D*,
      D* = max { (t_(i) - i/m)_+ : t_(i) >= 1 - delta }   (0 if empty),

  attained by ``F0*(t) = max(ECDF_m(t), t * 1{t >= 1 - delta})``, which is
  itself a member of ``B``. Here ``t_(1) <= ... <= t_(m)`` are the sorted
  rescaled interior values ``t = u_interior / (1 - delta)``. The envelope
  of the coefficient is ``PI_hi = A1 * ell * (1 - D*)``.

  The lower end is vacuous: the point mass at 0 belongs to ``B`` and gives
  ``q = 1/m`` exactly, a function of the interior sample size alone.

Both quantities are closed forms. No grid, no Beta sub-family sweep: a
maximum over a Beta grid *understates* the supremum over the full class.

Scale of ``delta``
------------------
``delta`` enters twice and both uses are on the rescaled scale ``t``:
the interior is ``u < 1 - delta`` (so the ceiling band is excluded before
rescaling), the rescaling is ``t = u / (1 - delta)``, and the admissibility
threshold ``t >= 1 - delta`` is then applied to ``t`` -- equivalently
``u >= (1 - delta)^2``. ``B`` is a class of laws on the ``t`` scale.

Representing ``F0``
-------------------
A reference is a *callable* ``F0(t)`` that accepts a float array on
``[0, 1]`` and returns the distribution function values. Convenience
constructors are provided (:func:`uniform_reference`,
:func:`beta_reference`, :func:`point_mass_reference`). A parametric
``(a, b)`` pair was rejected as the interface because the supremum is
over the full class and neither the attaining reference nor the point
mass at 0 is a Beta law; a sampled grid was rejected because interpolation
error lands exactly on the ceiling band where admissibility bites.

The reference must be declared before the data are seen and never fitted
from the same data.

This module depends on ``core`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import core
from .core import DELTA, DENSE_MAX_N, MIN_INTERIOR, prereq_index

__all__ = [
    "Admissibility",
    "admissibility",
    "interior_q",
    "prereq_index_family",
    "pi_envelope",
    "uniform_reference",
    "beta_reference",
    "point_mass_reference",
    "attaining_reference",
]

_TOL = 1e-12


# --------------------------------------------------------------------------
# reference constructors
# --------------------------------------------------------------------------

def uniform_reference():
    """The package default: ``F0(t) = t``. Boundary point of ``B``."""

    def F0(t):
        return np.clip(np.asarray(t, dtype=float), 0.0, 1.0)

    F0.__name__ = "Uniform(0,1)"
    return F0


def beta_reference(a, b):
    """Beta(a, b) distribution function as a reference callable."""
    from scipy.special import betainc  # scipy is a declared dependency

    a = float(a)
    b = float(b)
    if not (a > 0 and b > 0):
        raise ValueError("Beta parameters must be positive")

    def F0(t):
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        return betainc(a, b, t)

    F0.__name__ = f"Beta({a:g},{b:g})"
    return F0


def point_mass_reference(at=0.0):
    """Degenerate law at ``at``: ``F0(t) = 1{t >= at}``.

    The point mass at 0 is the admissible reference that attains the
    (vacuous) lower end ``q = 1/m``.
    """
    at = float(at)

    def F0(t):
        return (np.asarray(t, dtype=float) >= at).astype(float)

    F0.__name__ = f"PointMass({at:g})"
    return F0


def attaining_reference(x, y, delta=DELTA):
    """The member of ``B`` that attains the supremum for this pair.

    ``F0*(t) = max(ECDF_m(t), t * 1{t >= 1 - delta})`` built on the sorted
    rescaled interior of ``(x, y)``. Returned so that the closed form can
    be checked against a direct evaluation rather than trusted.
    """
    t_sorted, _ = _rescaled_interior(x, y, delta)
    m = t_sorted.size
    if m == 0:
        raise ValueError("no interior points")
    lo = 1.0 - delta

    def F0(t):
        t = np.asarray(t, dtype=float)
        ecdf = np.searchsorted(t_sorted, t, side="right") / m
        return np.maximum(ecdf, np.where(t >= lo, t, 0.0))

    F0.__name__ = "F0star"
    return F0


# --------------------------------------------------------------------------
# admissibility
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Admissibility:
    """Result of the admissibility check.

    Attributes
    ----------
    admissible : bool
        ``F0(t) >= t`` on the checked points of ``[1 - delta, 1]``.
    tail_mass : float
        ``1 - F0(1 - delta)``: the mass ``F0`` places above ``1 - delta``.
        Admissible references have ``tail_mass <= delta``; Uniform has
        exactly ``delta``.
    worst_slack : float
        ``min_t (F0(t) - t)`` over the checked points; negative means a
        violation, zero means the boundary (Uniform).
    worst_t : float
        The ``t`` at which ``worst_slack`` occurs.
    """

    admissible: bool
    tail_mass: float
    worst_slack: float
    worst_t: float


def admissibility(F0, delta=DELTA, n_grid=2001, tol=_TOL):
    """Does the declared reference ``F0`` belong to the admissible class?

    Parameters
    ----------
    F0 : callable
        Distribution function on ``[0, 1]``, vectorised over a float array.
    delta : float
        Ceiling band width (the fixed convention is 0.05).
    n_grid : int
        Number of equally spaced points on ``[1 - delta, 1]`` (both endpoints
        included) at which ``F0(t) >= t`` is checked. The condition is
        pointwise on a continuum; a callable can only be checked on a
        finite set, and this is that set.
    tol : float
        ``F0(t) >= t - tol`` is accepted, so that a numerically evaluated
        Uniform (the boundary point) is not rejected by rounding.

    Returns
    -------
    Admissibility

    Notes
    -----
    The check is the defining pointwise condition, **not** first-order
    stochastic dominance by Uniform: nothing is checked below
    ``1 - delta``. ``tail_mass <= delta`` is a consequence of admissibility
    (it is the condition at the single point ``t = 1 - delta``), not an
    equivalent restatement, which is why both are reported.
    """
    if not callable(F0):
        raise TypeError("F0 must be a callable distribution function")
    lo = 1.0 - float(delta)
    t = np.linspace(lo, 1.0, int(n_grid))
    F = np.asarray(F0(t), dtype=float)
    if F.shape != t.shape:
        raise ValueError("F0 must return one value per input point")
    slack = F - t
    k = int(np.argmin(slack))
    tail = float(1.0 - np.asarray(F0(np.array([lo])), dtype=float)[0])
    return Admissibility(
        admissible=bool(slack[k] >= -tol),
        tail_mass=tail,
        worst_slack=float(slack[k]),
        worst_t=float(t[k]),
    )


# --------------------------------------------------------------------------
# interior statistic at a declared reference, and the envelope
# --------------------------------------------------------------------------

def _rescaled_interior(x, y, delta):
    """Sorted rescaled interior ``t`` and ``n``, exactly as ``core`` forms them."""
    x, y = core._as_pair(x, y)
    u = np.clip(y / np.maximum(x, core._EPS_DEN), 0.0, 1.0)
    ceil_mask = u >= 1.0 - delta
    interior = u[~ceil_mask]
    t = np.sort(interior / (1.0 - delta))
    return t, x.size


def _guard_fires(m, n):
    return m < max(MIN_INTERIOR, 0.05 * n)


def interior_q(x, y, F0=None, delta=DELTA):
    """The interior component ``q`` computed against a declared reference.

    ``q(F0) = 1 - max_i |i/m - F0(t_(i))|`` over the sorted rescaled
    interior. With ``F0=None`` (Uniform) this equals ``prereq_index(x, y)["q"]``
    bit for bit. Returns ``0.0`` when the interior guard of the definition
    fires (fewer than ``max(10, 0.05 n)`` interior points), as ``core`` does.
    """
    t, n = _rescaled_interior(x, y, delta)
    m = t.size
    if _guard_fires(m, n):
        return 0.0
    F = np.arange(1, m + 1) / m
    if F0 is None:
        s = t
    else:
        s = np.asarray(F0(t), dtype=float)
    return 1.0 - float(np.max(np.abs(F - s)))


def prereq_index_family(x, y, F0=None, delta=DELTA, dense_max_n=DENSE_MAX_N):
    """The family member ``PI(F0)``: the coefficient at a declared reference.

    Composed from the definition's own components rather than recomputed:
    ``A1`` and ``ell`` do not depend on the reference, so

        PI(F0) = A1 * (q(F0) * ell)

    with ``A1`` and ``ell`` taken from :func:`prereq_index` and ``q(F0)``
    from :func:`interior_q`. The product is associated exactly as ``core``
    associates it (``A2 = q * ell`` first), so at ``F0 = None`` (Uniform)
    the result equals ``prereq_index(x, y)["PI"]`` bit for bit.

    The reference must be declared before the data are seen; check it with
    :func:`admissibility`. This function does not check admissibility
    itself, because a non-admissible reference still defines a member of
    the family -- it is simply one the manuscript does not license.

    Returns
    -------
    dict with keys ``PI``, ``A1``, ``A2``, ``q``, ``ell``, ``reference``
    (the name of ``F0`` if it has one, else ``"Uniform(0,1)"``).
    """
    res = prereq_index(x, y, delta=delta, dense_max_n=dense_max_n)
    q = interior_q(x, y, F0, delta=delta)
    a2 = q * res["ell"]
    name = "Uniform(0,1)" if F0 is None else getattr(F0, "__name__", "F0")
    return {"PI": res["A1"] * a2, "A1": res["A1"], "A2": a2, "q": q,
            "ell": res["ell"], "reference": name}


def pi_envelope(x, y, delta=DELTA, dense_max_n=DENSE_MAX_N):
    """The exact upper envelope of the prerelation coefficient over ``B``.

    Returns
    -------
    dict with keys
        ``PI_hi``    ``A1 * ell * sup_q`` -- the identified upper bound
        ``sup_q``    ``1 - D*``, the exact supremum of ``q`` over ``B``
        ``inf_q``    ``1 / m``, the (vacuous) infimum, attained by the point
                     mass at 0; reported so that it can be *labelled* vacuous
        ``D_star``   ``max { (t_(i) - i/m)_+ : t_(i) >= 1 - delta }``
        ``n_tail``   number of interior points with ``t_(i) >= 1 - delta``
        ``m``        interior sample size
        ``attained`` ``q(F0*) == sup_q`` within 1e-12, by direct evaluation
        ``A1``, ``ell``, ``q``, ``PI``   the components at the Uniform
                     reference, from :func:`prereq_index`

    When the interior guard fires, ``q`` is ``0`` for every reference by
    definition, so ``sup_q = inf_q = PI_hi = 0``.
    """
    res = prereq_index(x, y, delta=delta, dense_max_n=dense_max_n)
    t, n = _rescaled_interior(x, y, delta)
    m = t.size
    out = {"A1": res["A1"], "ell": res["ell"], "q": res["q"], "PI": res["PI"], "m": m}

    if _guard_fires(m, n):
        out.update(
            {"PI_hi": 0.0, "sup_q": 0.0, "inf_q": 0.0, "D_star": 0.0,
             "n_tail": 0, "attained": True}
        )
        return out

    F = np.arange(1, m + 1) / m
    tail = t >= 1.0 - delta
    n_tail = int(tail.sum())
    if n_tail == 0:
        d_star = 0.0
    else:
        d_star = float(np.max(np.maximum(t[tail] - F[tail], 0.0)))
    sup_q = 1.0 - d_star

    # direct evaluation at the attaining reference, as a distribution
    # *function* (ECDF via searchsorted, so tied sample values receive one
    # common value) -- the closed form is verified on every call rather
    # than assumed. With ties ``1 - D*`` remains a valid upper bound but
    # need not be attained; ``attained`` reports which case holds.
    ecdf = np.searchsorted(t, t, side="right") / m
    s_star = np.maximum(ecdf, np.where(tail, t, 0.0))
    q_star = 1.0 - float(np.max(np.abs(F - s_star)))
    attained = abs(q_star - sup_q) <= _TOL

    out.update(
        {
            "PI_hi": res["A1"] * res["ell"] * sup_q,
            "sup_q": sup_q,
            "inf_q": 1.0 / m,
            "D_star": d_star,
            "n_tail": n_tail,
            "attained": bool(attained),
        }
    )
    return out
