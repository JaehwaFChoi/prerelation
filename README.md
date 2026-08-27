# prerelation

A coefficient for **prerequisite relations** between traits reported on a
common anchored scale.

Two abilities can be strongly correlated without either being a prerequisite
for the other, and they can be perfectly ordered without being distinct.
`prerelation` asks a narrower question than correlation and a different one
from necessity analysis:

> does the candidate prerequisite **X** act as a *ceiling* on **Y**, and does
> **Y** keep the freedom below that ceiling which a prerequisite structure
> implies?

```
Pi(X -> Y) = A1 * A2        in [0, 1]

A1   the corner {Y > X} is empty, relative to what independence would give
A2   below the ceiling Y varies as a free component should (q), and the
     censoring thins out at high x (ell)

Delta = Pi(X -> Y) - Pi(Y -> X)
```

The product structure is what lets a single number separate the four
extremes. Independence is annihilated by `A1` alone; exact equivalence is
annihilated by `A2` alone. A coefficient that scores only the emptiness of
the corner gives equivalence its maximal value, because equivalence has an
empty corner — necessity and prerequisiteness are different concepts.

## Install

```bash
pip install prerelation                # core + screening + ceiling + study
pip install "prerelation[ctm]"         # adds the bounded-trait scoring extra
```

## Quick start

```python
import numpy as np
from prerelation import prereq_index, direction, scan_pairs

rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 500)
y = x * rng.uniform(0, 1, 500)          # X is a ceiling on Y

prereq_index(x, y)["PI"]                 # -> about 0.94
direction(x, y)                          # (Delta, forward, reverse); reverse is 0.0

theta = np.column_stack([x, y, y * rng.uniform(0, 1, 500)])
res = scan_pairs(theta, names=["A", "B", "C"], n_perm=199)
res.edges                                # [('A','B'), ('A','C'), ('B','C')]
res.reduced_edges                        # [('A','B'), ('B','C')] — the implied edge is gone
res.table                                # tidy DataFrame of every ordered pair
```

The scan recovers a **dominance preorder** over the attributes — which
attributes act as ceilings on which others — not a direct-prerequisite
DAG. Indirect dominance produces edges of its own, and siblings under a
common ceiling can be linked even though neither is a prerequisite for the
other; the transitive reduction cleans up chains but cannot distinguish a
direct edge from a dominated one in general. When the recovered order
disagrees with an expert-specified prerequisite graph, the two are
answering different questions. The permutation screen also has a design
floor: with `K` ordered pairs, no edge can survive BH control at level
`alpha` unless `n_perm >= K / alpha - 1`.

Estimating the ceiling itself:

```python
from prerelation import ceiling_fit, prereq_index_referenced

fit = ceiling_fit(x, y, tau=0.95)        # monotone quantile envelope, split-half
pi_ref = prereq_index_referenced(x, y, fit)
```

`ceiling_fit(..., postulate_correction=True)` divides the raw envelope by
`tau`. Its validity domain, from the docstring: the correction is grounded
only under the *calibrated product model* `Y = c(X) U` with
`U ~ Uniform(0, 1)` independent of `X` (the uniform freedom postulate) —
under that model the tau-quantile envelope estimates `tau * c`, so dividing
by `tau` is a consistent correction. It is **not applicable** to the
weakest-link form `Y = min(c(X), T)`, where division by `tau` strictly
over-estimates the ceiling wherever the censoring binds, nor to non-uniform
free components, where `1/tau` carries no information about the quantile
being estimated. In both cases leave the default `False`.

Correcting for scoring error with plausible values:

```python
from prerelation import pv_correct   # requires: pip install "prerelation[ctm]"
```

`pv_correct` consumes grid posteriors from the bounded-trait scorer, which
is deliberately an *optional* extra — `prerelation.core` imports numpy and
nothing else, and a test enforces that.

## Scope, stated plainly

`Pi` is defined for traits on a common bounded scale whose endpoints are
substantive anchors — 0 means absence of the ability, 1 means full mastery.
This is an **interpretability requirement on the scale**, not a claim about
the measurement precision of any scoring model. The ratio `Y / X` and the
corner moment `(Y - X)+` carry the reading "how much of the ceiling granted
by X is used by Y", and that reading does not survive an arbitrary monotone
rescaling: on a location-scale standardised latent trait, `Pi` computed from
the numbers is not a prerequisite statistic at all. Accordingly `Pi` is
deliberately *not* invariant to rescaling either axis, and it is not
symmetric.

Any scale meeting the requirement will do. The bounded trait model of Choi
(2022) is one such scale and `prerelation.pv` can consume its posteriors,
but the coefficient itself is model-free: `prerelation.core` imports numpy
and nothing else, and a test enforces that. This package is part of the
research program around that bounded-trait framework (CTM/ALF), where the
anchored `[0, 1]` scale supplies the interpretive ground the coefficient
needs; the scale requirement here is about interpretability, not about the
measurement quality of any model.

## What is in the package

| module | contents |
|---|---|
| `core` | `prereq_index`, `direction`, `perm_pvalue` |
| `scan` | all ordered pairs, BH-FDR, cycle check, transitive reduction — read as a dominance preorder |
| `ceiling` | monotone quantile ceiling, CTM-logistic ceiling, referenced index |
| `pv` | plausible-value correction for scoring error (optional extra) |
| `study` | the simulation frame: one config dict in, one tidy table out |

### The ceiling-referenced variant

When the ceiling is far from the identity the default coefficient is
attenuated. `ceiling_fit` estimates `c` on one half of the sample and
`prereq_index_referenced` evaluates the coefficient on the other half, on the
transformed pair `(c(x), y)`. Every component is recomputed there — replacing
only one component while leaving the others on the identity reference is a
different statistic, not a variant, so the API gives no way to do it.

### Plausible values

`Pi` is computed from trait *estimates*, and estimates carry error. Because
`Pi` is a nonlinear functional of the pair, plugging in point estimates
understates the uncertainty. `pv_correct` consumes grid posteriors, draws
plausible values, recomputes `Pi` per draw, and reports the spread.

## Correctness standard

`tests/oracle/prereq_index_v2.py` is a verbatim copy of the reference
implementation that produced the recorded results and is kept permanently.
Every optimised path in the package is pinned by tests asserting agreement
with it to within `1e-12`, including the guards, the clipping, and the fact
that the independence baseline is a V-statistic whose double sum includes the
diagonal. Above `DENSE_MAX_N` the baseline is accumulated by sorting instead
of forming the `n x n` matrix; the two differ only in summation order, which
is what makes large scans and permutation loops possible at all.

## Reproducibility and golden vectors

`tests/golden/` contains fixed fixtures spanning the product, weakest-link,
independence, equivalence and partial-equivalence regimes plus a real-data
slice, with component-wise outputs (`v`, `v0`, `A1`, per-band interior
mass, `q`, `ell`, `A2`, `Pi`, `Delta`) written to machine-readable files.
The permutation index matrix itself is committed, generated once from a
recorded seed, so an implementation in any language can apply *identical*
permutations and match the p-values exactly rather than in distribution —
the contract is documented in `tests/golden/README.md`. A pytest module
regenerates every golden output from the fixtures and asserts equality at
`1e-12`, which pins this package as the reference implementation.

## Citation

The methodological paper introducing the coefficient is in preparation. To
cite the software, use the metadata in `CITATION.cff`; an archived DOI will
be added there upon release (Zenodo, DOI to follow).

## Status

`0.2.0` — first public release. The API is settled enough to build on; the
numbers in the test suite are verification fixtures rather than results.

## License

MIT.
