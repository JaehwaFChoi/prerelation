"""prerelation.pv — plausible-value correction for scoring error.

Pi is computed from trait *estimates*, and estimates carry error. Plugging
point estimates in and reporting the result as if they were the traits
understates the uncertainty and, because Pi is a nonlinear functional of the
pair, can bias it in either direction. The standard remedy is plausible
values: draw from each person's posterior, recompute the statistic on the
draw, and summarise across draws.

This module consumes a grid posterior — a set of nodes and a normalised
weight matrix of shape (n_persons, n_nodes) — and is therefore independent
of how that posterior was produced. ``ctm_posterior`` is a thin convenience
wrapper around the bounded-trait scoring package, which is an *optional*
dependency:

    pip install "prerelation[ctm]"

The wrapper reads the posterior only; it never modifies that package.

A note that has cost time before: in the scoring package ``map_theta``
returns a tuple ``(theta_hat, at_bound)``, not an array. This module does
not use it — the posterior itself is what plausible values need — but code
that mixes the two should unpack the tuple.
"""

from __future__ import annotations

import numpy as np

from .core import DELTA, prereq_index

__all__ = ["ctm_posterior", "draw_pv", "pv_correct"]

_CTM_HINT = (
    "cogtraitmodel is an optional dependency of prerelation; install it with "
    "pip install 'prerelation[ctm]' to score responses from this package."
)


def ctm_posterior(Y, alpha, beta, gamma=None, n_nodes=61, prior=(2.0, 2.0)):
    """Grid posterior of the bounded-trait model for a response matrix.

    Returns
    -------
    (nodes, post) : ndarray of shape (K,) and (n_persons, K)
    """
    try:
        from cogtraitmodel import core as ctm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_CTM_HINT) from exc
    nodes, post = ctm.posterior(Y, alpha, beta, gamma=gamma, n_nodes=n_nodes,
                                prior=prior)
    return np.asarray(nodes, dtype=float), np.asarray(post, dtype=float)


def draw_pv(nodes, post, n_draws, rng):
    """Draw plausible values by inverse-cdf sampling on the grid.

    Parameters
    ----------
    nodes : ndarray of shape (K,)
    post : ndarray of shape (n_persons, K)
        Rows are normalised (or will be normalised here).
    n_draws : int
    rng : numpy Generator

    Returns
    -------
    ndarray of shape (n_draws, n_persons)
    """
    nodes = np.asarray(nodes, dtype=float)
    post = np.asarray(post, dtype=float)
    if post.ndim != 2 or post.shape[1] != nodes.size:
        raise ValueError("post must have shape (n_persons, len(nodes))")
    w = post / post.sum(axis=1, keepdims=True)
    cdf = np.cumsum(w, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random((n_draws, post.shape[0]))
    idx = np.empty(u.shape, dtype=int)
    for i in range(post.shape[0]):
        idx[:, i] = np.searchsorted(cdf[i], u[:, i], side="left")
    idx = np.clip(idx, 0, nodes.size - 1)
    return nodes[idx]


def pv_correct(post_x, post_y, n_draws=20, seed=0, delta=DELTA, ci=(2.5, 97.5)):
    """Plausible-value summary of Pi and Delta for one ordered pair.

    Parameters
    ----------
    post_x, post_y : (nodes, post) tuples
        Grid posteriors of the two traits for the same persons, in the same
        row order.
    n_draws : int
        Plausible values per person. Twenty is a common convention; the
        Monte-Carlo error of the mean falls as 1/sqrt(n_draws).
    seed : int
    ci : tuple of float
        Percentile levels of the reported interval.

    Returns
    -------
    dict with the mean, SD and percentile interval of Pi and Delta across
    draws, plus the per-draw arrays under ``pi_draws`` and ``delta_draws``.

    Notes
    -----
    The interval describes the spread induced by *scoring error* under the
    fitted model. It is not a confidence interval for the population value,
    which also carries sampling error; the permutation test in
    ``prerelation.core`` addresses a different question again.
    """
    nodes_x, px = post_x
    nodes_y, py = post_y
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    if px.shape[0] != py.shape[0]:
        raise ValueError("the two posteriors must cover the same persons")

    rng = np.random.default_rng(seed)
    draws_x = draw_pv(nodes_x, px, n_draws, rng)
    draws_y = draw_pv(nodes_y, py, n_draws, rng)

    pi = np.empty(n_draws)
    pi_rev = np.empty(n_draws)
    for d in range(n_draws):
        pi[d] = prereq_index(draws_x[d], draws_y[d], delta)["PI"]
        pi_rev[d] = prereq_index(draws_y[d], draws_x[d], delta)["PI"]
    dlt = pi - pi_rev

    return {
        "pi_mean": float(pi.mean()),
        "pi_sd": float(pi.std(ddof=1)) if n_draws > 1 else 0.0,
        "pi_ci": tuple(float(v) for v in np.percentile(pi, ci)),
        "delta_mean": float(dlt.mean()),
        "delta_sd": float(dlt.std(ddof=1)) if n_draws > 1 else 0.0,
        "delta_ci": tuple(float(v) for v in np.percentile(dlt, ci)),
        "pi_draws": pi,
        "pi_reverse_draws": pi_rev,
        "delta_draws": dlt,
        "n_draws": n_draws,
        "seed": seed,
    }
