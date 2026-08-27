"""prerelation.ceiling — estimating the ceiling c and the ceiling-referenced index.

The default coefficient takes the raw pair (x, y): the unfitted,
convention-anchored form, playing the role that the plain Pearson r plays
for association. When the ceiling is not the identity, that default is
attenuated, and a *ceiling-referenced* variant applies the identical
statistic to the transformed pair (c(x), y).

Two rules govern this module.

1. The transformation is applied to the pair, and every component of the
   coefficient is then recomputed on the transformed pair. Replacing only
   one component (say q) while leaving A1 and ell on the identity reference
   is not a variant of the coefficient — it is a different statistic, and it
   contradicts the definition. ``prereq_index_referenced`` therefore does
   nothing but call ``prereq_index`` on (c(x), y); the "recompute
   everything" rule is structural, not a convention to remember.

2. The ceiling is fitted on one half of the sample and the coefficient
   evaluated on the other. Fitting an upper envelope and then measuring the
   emptiness of the region above it on the same points is circular: the
   split is what keeps the estimate honest.

Methods
-------
``"monotone_quantile"``
    Bin x by its own quantiles, take the tau-quantile of y within each bin,
    then enforce monotonicity by rearrangement — sorting the fitted values.
    Rearranging an estimate of a monotone function can only reduce its
    estimation error, which is the result of Chernozhukov, Fernandez-Val and
    Galichon (2009). Bin quantiles are read off the empirical distribution
    (Koenker and Bassett, 1978, is the regression counterpart).

``"ctm"``
    Fit the normalised bounded-trait logistic
    c(x; a, b) = [S(a(x - b)) - S(-ab)] / [S(a(1 - b)) - S(-ab)], with S the
    logistic function, by minimising the pinball loss at level tau. The form
    passes through (0, 0) and (1, 1) by construction and is increasing, so
    monotonicity needs no post-hoc repair. Optimisation runs in the
    transformed space log(a), logit(b). Requires scipy.

Both estimate an upper envelope, so tau is chosen high (0.95 by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .core import DELTA, prereq_index

__all__ = ["CeilingFit", "ceiling_fit", "prereq_index_referenced", "ctm_ceiling"]

_FLOOR = 1e-9


def ctm_ceiling(x, a, b):
    """Normalised bounded-trait logistic ceiling; c(0) = 0 and c(1) = 1."""
    x = np.asarray(x, dtype=float)
    s = lambda t: 1.0 / (1.0 + np.exp(-t))  # noqa: E731 - local shorthand
    lo = s(-a * b)
    hi = s(a * (1.0 - b))
    return (s(a * (x - b)) - lo) / (hi - lo)


@dataclass
class CeilingFit:
    """A fitted ceiling function with its split-sample bookkeeping.

    When ``postulate_correction`` is True, ``predict`` returns the
    postulate-corrected ceiling ``min(c_raw / tau, 1)`` instead of the raw
    envelope (see ``ceiling_fit`` for the applicability conditions), and
    ``truncation_rate`` reports the share of fit-half points at which the
    corrected value was truncated at the upper bound 1.
    """

    method: str
    tau: float
    x_knots: Optional[np.ndarray] = None
    c_knots: Optional[np.ndarray] = None
    params: dict = field(default_factory=dict)
    fit_index: Optional[np.ndarray] = None
    eval_index: Optional[np.ndarray] = None
    postulate_correction: bool = False
    truncation_rate: Optional[float] = None

    def _predict_raw(self, x):
        if self.method == "ctm":
            c = ctm_ceiling(x, self.params["a"], self.params["b"])
        else:
            c = np.interp(x, self.x_knots, self.c_knots)
        return c

    def predict(self, x):
        """Evaluate the fitted ceiling, clipped to [0, 1].

        With ``postulate_correction`` the raw envelope is divided by ``tau``
        before clipping; without it, the computation is exactly the pre-0.1.1
        path (bit-identical outputs).
        """
        x = np.asarray(x, dtype=float)
        c = self._predict_raw(x)
        if self.postulate_correction:
            c = c / self.tau
        return np.clip(c, 0.0, 1.0)

    def __repr__(self):  # pragma: no cover - display only
        detail = (
            f"a={self.params['a']:.3f}, b={self.params['b']:.3f}"
            if self.method == "ctm"
            else f"knots={0 if self.x_knots is None else self.x_knots.size}"
        )
        return f"CeilingFit(method={self.method!r}, tau={self.tau}, {detail})"


def _split(n, split, seed):
    if not split:
        idx = np.arange(n)
        return idx, idx
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(float(split) * n))
    cut = min(max(cut, 1), n - 1)
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def _fit_monotone_quantile(x, y, tau, n_bins):
    n = x.size
    if n_bins is None:
        n_bins = int(np.clip(np.sqrt(n) / 1.5, 3, 20))
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-12
    xk, ck = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x > lo) & (x <= hi)
        if m.sum() < 2:
            continue
        xk.append(float(np.median(x[m])))
        ck.append(float(np.quantile(y[m], tau)))
    if len(xk) < 2:
        raise ValueError("not enough populated bins to fit a ceiling")
    xk = np.asarray(xk, dtype=float)
    # Rearrangement: sorting an estimate of a monotone function cannot
    # increase its error (Chernozhukov, Fernandez-Val and Galichon, 2009).
    ck = np.sort(np.asarray(ck, dtype=float))
    # Anchor at the origin; the scale reads 0 as absence of the ability.
    xk = np.concatenate(([0.0], xk))
    ck = np.concatenate(([0.0], ck))
    return xk, np.clip(ck, _FLOOR, 1.0)


def _pinball(resid, tau):
    return float(np.mean(np.maximum(tau * resid, (tau - 1.0) * resid)))


def _fit_ctm(x, y, tau):
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("method='ctm' needs scipy; install scipy or use "
                          "method='monotone_quantile'") from exc

    def unpack(z):
        a = float(np.exp(np.clip(z[0], -5.0, 6.0)))
        b = float(1.0 / (1.0 + np.exp(-np.clip(z[1], -8.0, 8.0))))
        return a, b

    def loss(z):
        a, b = unpack(z)
        return _pinball(y - ctm_ceiling(x, a, b), tau)

    best, best_val = None, np.inf
    for z0 in ([np.log(8.0), 0.0], [np.log(2.0), -1.0], [np.log(20.0), 1.0]):
        res = minimize(loss, np.asarray(z0, dtype=float), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 2000})
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    a, b = unpack(best)
    return {"a": a, "b": b, "pinball": best_val}


def ceiling_fit(x, y, tau=0.95, method="monotone_quantile", n_bins=None,
                split=0.5, seed=0, postulate_correction=False):
    """Fit the ceiling c on a random half of the sample.

    Parameters
    ----------
    x, y : array_like of shape (n,)
    tau : float
        Quantile level of the envelope.
    method : {"monotone_quantile", "ctm"}
    n_bins : int, optional
        Only for ``"monotone_quantile"``; defaults to a size-dependent value.
    split : float or None
        Fraction of the sample used for fitting. ``None`` or 0 fits and
        evaluates on the same points, which is optimistic and is meant for
        illustration only.
    seed : int
    postulate_correction : bool
        Default False. When True, ``predict`` returns the corrected ceiling
        ``c_corr = min(c_raw / tau, 1)`` and ``truncation_rate`` (the share
        of fit-half points truncated at 1) is reported on the returned
        object.

        **Applicability.** The correction is grounded only under the
        *calibrated product model*: ``Y = c(X) U`` with ``U ~ Uniform(0, 1)``
        independent of ``X`` (the uniform freedom postulate). Under that
        model the tau-quantile envelope estimates ``tau * c``, so dividing
        by ``tau`` is a consistent correction (Proposition T7).

        **Not applicable.** (a) *Weakest-link form* ``Y = min(c(X), T)``:
        the envelope limit is ``min(c(x), Q_T(tau))`` and division by
        ``tau`` strictly over-estimates the ceiling wherever
        ``0 < c(x) < 1`` (by the factor ``1/tau`` where the censoring
        binds). (b) *Non-uniform free components*: the envelope identifies
        ``c * Q_U(tau)``, and ``1/tau`` carries no information about
        ``Q_U(tau)`` -- the correction has no basis. In both cases leave
        the default False.

    Returns
    -------
    CeilingFit
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    fit_idx, eval_idx = _split(x.size, split, seed)
    xf, yf = x[fit_idx], y[fit_idx]

    if method == "monotone_quantile":
        xk, ck = _fit_monotone_quantile(xf, yf, tau, n_bins)
        out = CeilingFit(method=method, tau=tau, x_knots=xk, c_knots=ck,
                         fit_index=fit_idx, eval_index=eval_idx,
                         postulate_correction=postulate_correction)
    elif method == "ctm":
        params = _fit_ctm(xf, yf, tau)
        out = CeilingFit(method=method, tau=tau, params=params,
                         fit_index=fit_idx, eval_index=eval_idx,
                         postulate_correction=postulate_correction)
    else:
        raise ValueError(f"unknown method {method!r}")
    if postulate_correction:
        raw = out._predict_raw(xf)
        out.truncation_rate = float(np.mean(raw / tau > 1.0))
    return out


def prereq_index_referenced(x, y, ceiling, indices="eval", delta=DELTA):
    """Ceiling-referenced coefficient: the same statistic on (c(x), y).

    Parameters
    ----------
    x, y : array_like of shape (n,)
    ceiling : CeilingFit or callable
        Anything with ``predict`` or a plain function of x.
    indices : {"eval", "all"} or array_like of int
        Which points to evaluate on. The default uses the half of the sample
        that was held out when the ceiling was fitted.
    delta : float

    Returns
    -------
    dict, exactly as ``prereq_index``: every component is recomputed on the
    transformed pair.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    predict = getattr(ceiling, "predict", ceiling)

    if isinstance(indices, str):
        if indices == "all":
            sel = np.arange(x.size)
        elif indices == "eval":
            sel = getattr(ceiling, "eval_index", None)
            sel = np.arange(x.size) if sel is None else sel
        else:
            raise ValueError(f"unknown indices {indices!r}")
    else:
        sel = np.asarray(indices, dtype=int)

    cx = np.clip(predict(x[sel]), 0.0, 1.0)
    return prereq_index(cx, y[sel], delta)
