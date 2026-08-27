"""
Prerequisite index (generalized, censoring-aware mixture benchmark)

PI(X->Y) = A1 * A2*
  A1  : corner emptiness vs independence baseline (half-rectified moment)
  A2* : conditional freedom = q * ell
        q   = interior uniformity (KS), ceiling band treated as censored
        ell = ceiling legitimacy: censoring thins at high x; equivalence doesn't
Direction: Delta = PI(X->Y) - PI(Y->X)
"""
import numpy as np

DELTA = 0.05          # ceiling band width
TOP_Q = 0.8           # top-x quantile for legitimacy check
MIN_INTERIOR = 10     # minimum interior points to credit freedom


def prereq_index(x, y, delta=DELTA):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)

    # A1: corner emptiness
    v  = np.mean(np.maximum(y - x, 0))
    v0 = np.mean(np.maximum(y[None, :] - x[:, None], 0))
    a1 = max(0.0, 1 - v / v0) if v0 > 1e-9 else 0.0

    # A2*: conditional freedom with censoring-aware benchmark
    u = np.clip(y / np.maximum(x, 1e-9), 0, 1)
    ceil_mask = u >= 1 - delta
    interior = u[~ceil_mask]

    if len(interior) < max(MIN_INTERIOR, 0.05 * n):
        q = 0.0
    else:
        t = np.sort(interior / (1 - delta))
        F = np.arange(1, len(t) + 1) / len(t)
        q = 1 - np.max(np.abs(F - t))

    x_top = x >= np.quantile(x, TOP_Q)
    p1_top = np.mean(ceil_mask[x_top]) if x_top.sum() > 0 else 1.0
    ell = 1 - max(0.0, p1_top - delta) / (1 - delta)

    a2 = q * ell
    return {'PI': a1 * a2, 'A1': a1, 'A2': a2, 'q': q, 'ell': ell}


def direction(x, y, delta=DELTA):
    pi_xy = prereq_index(x, y, delta)['PI']
    pi_yx = prereq_index(y, x, delta)['PI']
    return pi_xy - pi_yx, pi_xy, pi_yx


def perm_pvalue(x, y, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    obs = prereq_index(x, y)['PI']
    cnt = sum(prereq_index(x, rng.permutation(y))['PI'] >= obs
              for _ in range(n_perm))
    return obs, (cnt + 1) / (n_perm + 1)


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.uniform(0, 1, n); U = rng.uniform(0, 1, n)
    S = {
        'pure prereq   Y=X*U':        (X, X * U),
        'independent':                (X, rng.uniform(0, 1, n)),
        'equivalence   Y=X+e(.02)':   (X, np.clip(X + rng.normal(0, .02, n), 0, 1)),
        'degenerate    Y~0':          (X, 0.02 * rng.uniform(0, 1, n)),
        'noisy prereq  Y=X*U+e':      (X, np.clip(X * U + rng.normal(0, .05, n), 0, 1)),
        'min model     Y=min(X,U)':   (X, np.minimum(X, U)),
        'nonunif U     Y=X*Beta(2,2)':(X, X * rng.beta(2, 2, n)),
        'min + noise':                (X, np.clip(np.minimum(X, U) + rng.normal(0, .05, n), 0, 1)),
        'equivalence   Y=X+e(.10)':   (X, np.clip(X + rng.normal(0, .10, n), 0, 1)),
    }
    Yr = rng.uniform(0, 1, n)
    S['reverse       X=Y*U'] = (Yr * U, Yr)

    hdr = f"{'scenario':30s} {'PI':>7s} {'A1':>6s} {'A2*':>6s} {'q':>6s} {'ell':>6s} {'Delta':>8s}"
    print(hdr); print('-' * len(hdr))
    for name, (x, y) in S.items():
        r = prereq_index(x, y)
        d, _, _ = direction(x, y)
        print(f"{name:30s} {r['PI']:7.3f} {r['A1']:6.3f} {r['A2']:6.3f} "
              f"{r['q']:6.3f} {r['ell']:6.3f} {d:+8.3f}")
