# JavaScript parity specification

The web application carries its own implementation of the coefficient. This
note fixes what "parity" means for it, so that the JS side can be written and
checked against a target rather than against a moving reference. Implementing
and running the JS side is a later phase; this file is the specification only.

## Reference of record

The Python package is the reference, and the Python package is itself pinned
to `tests/oracle/prereq_index_v2.py`. A JS result is correct when it matches
Python, which matches the oracle.

## Functions in scope

| function | signature | parity target |
|---|---|---|
| `prereqIndex(x, y, delta = 0.05)` | arrays of equal length | all five components `PI, A1, A2, q, ell` within `1e-12` |
| `direction(x, y, delta)` | arrays | `Delta, forward, reverse` within `1e-12` |
| `scan(theta, opts)` | matrix | identical edge set and identical transitive reduction, per commit |

The permutation test is **not** a parity target elementwise: the two
languages do not share a random number generator, so the reference sets
differ. What must agree is the observed statistic (`1e-12`) and, on shared
fixtures, the decision at a stated alpha.

## Closed-form fixtures

These are exact population values and can be checked in either language
without simulation. They are derived in the theory note; the finite-sample
figures are for orientation only.

| configuration | quantity | value |
|---|---|---|
| `Y = X * U`, `U ~ Uniform(0,1)`, `U` independent of `X` | `Pi` | `1` |
| `X` independent of `Y` | `A1`, hence `Pi` | `0` |
| `Y = X` almost surely | `q`, `ell`, hence `Pi` | `0` |
| `Y` constant | `A1`, hence `Pi` | `0` |
| `Y = a X U`, `0 < a < 1 - delta`, `U ~ Uniform(0,1)` | `Pi` on the identity reference | `a / (1 - delta)` |
| `Y = min(X, T)`, `X, T ~ Uniform(0,1)` | `Pi` | `E[X \| X >= x_0.8] = 0.9` |
| any `c(x) <= x`, either generating form | reverse `Pi` | exactly `0` |

## Definitional details that must be transcribed, not reinvented

Parity breaks here first, so the JS implementation must reproduce each of
these literally:

1. **The independence baseline is a V-statistic.** `v0` averages
   `(y_j - x_i)+` over all `n^2` ordered pairs, the diagonal `i = j`
   included. Excluding the diagonal changes the value.
2. **Guards are part of the definition.** `v0 <= 1e-9` forces `A1 = 0`;
   fewer than `max(10, 0.05 n)` interior points forces `q = 0`; an empty
   top-x stratum forces `p1top = 1`, hence `ell = 0`.
3. **Clipping.** `u = clip(y / max(x, 1e-9), 0, 1)`.
4. **The ceiling band is closed on the left**: a point is a ceiling point
   when `u >= 1 - delta`.
5. **The uniformity statistic is the grid form**
   `q = 1 - max_i |i/m - t_(i)|` over the sorted rescaled interior values,
   with `i` running from `1` to `m`. It differs from the classical
   two-sided Kolmogorov-Smirnov statistic by at most `1/m`; do not
   substitute a library KS routine.
6. **The top stratum uses the empirical 0.8-quantile with linear
   interpolation** — the default of `numpy.quantile`. JS has no default;
   implement the same convention explicitly.
7. **Fixed constants**: `delta = 0.05`, `TOP_Q = 0.8`, `MIN_INTERIOR = 10`.
   A small-sample adaptation of `TOP_Q` is an open question and must not be
   introduced on one side only.

## Exact-zero versus tolerance

Some zeros are structural (a branch returns `0.0`) and must be exactly zero
in both languages: exact equivalence, a degenerate `Y`, and the reverse
direction of any sub-identity ceiling. Others are identities of real
arithmetic that floating point only approximates — with `y` constant,
`v = v0` holds mathematically but the two averages sum in different orders,
so the residual lands within a few units in the last place. Assert exact
zero only in the first group; use `1e-12` everywhere else.

## Procedure

For each commit that touches either implementation: run the shared fixture
set in both languages, compare the closed forms above, and compare the scan
output on a fixed trait matrix shipped with the fixtures. A difference is a
defect in whichever side moved, and it is logged before it is fixed.
