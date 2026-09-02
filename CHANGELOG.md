# Changelog

## 0.3.1 — 2026-09-02

Test-only. **The library is byte-identical to 0.3.0** and every computational
output is unchanged.

- `test_family_member_at_uniform_equals_pi_bitwise` compared a freshly
  computed `PI` against the literal stored in `tests/golden/expected.json`
  with `==`. That is a claim about the build that wrote the literal rather
  than about the definition: the dense double sum behind `v0` accumulates in
  a different order between NumPy generations, moving `A1` and hence `PI` by
  one unit in the last place on the `min` and `ecpe_slice` fixtures. Because
  NumPy 2.3 requires Python 3.11, the comparison held on 3.11 and 3.12 and
  failed on 3.9 and 3.10. It now uses the golden contract's tolerance of
  1e-12, which is what `tests/golden` applies to every float. The two
  same-process assertions are unchanged and still use `==`.
- The publish workflow now runs the test suite on the oldest and newest
  supported Python versions before uploading. 0.3.0 was published while the
  suite was failing on 3.9 and 3.10, because nothing connected the two
  workflows.
  
## 0.3.0 — 2026-09-02

Adds the admissible reference class and the exact upper envelope. **No
existing output changes**: the ninety-four quantities already in
`tests/golden/expected.json` are byte-identical to 0.2.0, and `delta`,
`TOP_Q` and `MIN_INTERIOR` are untouched.

- New module `prerelation.reference`, exporting `admissibility`,
  `Admissibility`, `interior_q`, `prereq_index_family`, `pi_envelope` and
  the reference constructors `uniform_reference`, `beta_reference`,
  `point_mass_reference` and `attaining_reference`.
- `admissibility(F0)` tests a declared reference against
  `B = { F0 : F0(t) >= t for all t >= 1 - delta }` on the rescaled interior
  scale, and reports the mass it places above `1 - delta` separately. The
  pointwise condition implies that mass is at most `delta`; the converse
  does not hold, and the pointwise form is the one the envelope rests on.
- `pi_envelope(x, y)` returns `sup_q = 1 - D*` with
  `D* = max{ (t_(i) - i/m)_+ : t_(i) >= 1 - delta }`, `inf_q = 1/m`, and
  `PI_hi = A1 * ell * sup_q`. `1 - D*` is an exactly computable **upper
  bound** on the interior component over all of `B`; it equals the supremum
  and is attained by `attaining_reference` when the rescaled interior values
  are distinct. Under ties the bound still holds but need not be attained,
  so the result carries an `attained` flag computed by evaluating the
  attaining reference on every call rather than assumed.
- `inf_q` depends only on the interior sample size and carries no
  information about the data. It is returned so that it is visible rather
  than quietly omitted.
- `prereq_index_family(x, y, F0)` is the coefficient at a declared
  reference, composed from the same core quantities. At `F0 = Uniform` it is
  bit-identical to `prereq_index`, which required matching core's
  multiplication order (`A2 = q * ell` first) rather than merely agreeing to
  a tolerance.
- Golden vectors gain `n_tail_band`, `D_star`, `sup_q`, `inf_q` and `PI_hi`
  for each of the six fixtures. No fixture was added or changed.
- The reference is declared in advance and is never fitted from the same
  data.
  
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
