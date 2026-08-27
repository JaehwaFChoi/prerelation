# Changelog

## 0.2.0 — 2026-08-27

First public release. **No behavioral changes**: every computational output
is bit-identical to 0.1.1.dev0.

- Documentation pass over the public API; the pairwise scan is now described
  throughout as recovering a *dominance preorder* over the attributes rather
  than a direct-prerequisite DAG, and the `n_perm >= K / alpha - 1` design
  floor for the permutation screen is stated in the `scan` docstring.
- Golden vectors under `tests/golden/`: fixed fixtures spanning the product,
  weakest-link, independence, equivalence and partial-equivalence regimes
  plus a real-data slice, with component-wise outputs (v, v0, A1, per-band
  interior mass, q, ell, A2, Pi, Delta) and a committed permutation index
  matrix so downstream implementations can match p-values exactly. A pytest
  module regenerates all outputs from the fixtures and pins them at 1e-12.
- README rewritten for public consumption; CHANGELOG and packaging polish;
  version pinned consistently across `pyproject.toml`, `__init__`,
  `CITATION.cff` and the test suite.

## 0.1.1.dev0 — 2026-08-27

- `postulate_correction` option on `ceiling_fit` (D16): the corrected
  ceiling `min(c_raw / tau, 1)` with a `truncation_rate` diagnostic.
  Default **off**; the off path is bit-identical to 0.1.0.dev0, enforced by
  regression tests. The docstring states the validity domain (calibrated
  product model with a uniform free component) and the two regimes where the
  correction has no basis (weakest-link form, non-uniform free components).

## 0.1.0.dev0 — 2026-08-26

- Initial package (Phase B): `core` (`prereq_index`, `direction`,
  `perm_pvalue`), `scan` (BH-FDR, cycle check, transitive reduction),
  `ceiling` (monotone quantile and CTM-logistic ceilings, the
  ceiling-referenced variant), `pv` (plausible-value correction; the
  bounded-trait scorer is an optional extra), `study` (config dict in, tidy
  table out).
- Correctness standard: `tests/oracle/prereq_index_v2.py` kept verbatim;
  every optimised path pinned to the oracle within 1e-12.
