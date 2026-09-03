# Changelog

## 0.4.0 — 2026-09-03

Additive. **Every one of the 124 golden keys is bit-identical to 0.3.2** — not
agreeing to a tolerance, bit-identical — and the fixture CSVs and permutation
index matrices are byte-identical. Two things are added and nothing is changed.

### Added

- `condense(nodes, edges)`, in `scan`, returning a `Condensation` with
  `classes`, `class_of`, `quotient_edges` and `hasse_edges`. Each strongly
  connected component becomes one class and the acyclic quotient is
  transitively reduced. `transitive_reduction` is unchanged: it is defined
  only for an acyclic graph and still raises on a cycle, and `condense` is
  what handles cycles.

  This function existed in the JavaScript and R packages but had **no Python
  original**, so the declared reference implementation was missing something
  the method's own description requires. It is written here from that
  description rather than translated from either existing copy, and it uses
  Kosaraju's two-pass search where both ports use an iterative Tarjan — a
  translation would have produced a third copy rather than a third answer.
  It is checked against the JavaScript port on the six shared graph fixtures,
  against the R package's own recorded expectations, and against an
  independent reachability-closure verifier that is not a component algorithm
  at all (`tests/test_condense.py`).

- `top_q` and `min_interior` as keyword arguments to `prereq_index`,
  `direction` and `perm_pvalue`, threaded through `reference.interior_q`,
  `reference.prereq_index_family`, `reference.pi_envelope` and
  `study.run_study`. `run_study` accepts them as config keys and carries them
  into every output row, so a table swept over them stays reproducible from
  its own contents.

  **The defaults are exactly the existing module constants** — `TOP_Q = 0.8`,
  `MIN_INTERIOR = 10`, with the interior floor still `max(min_interior,
  0.05 n)` — and `DELTA`, `TOP_Q` and `MIN_INTERIOR` keep their values. The
  arguments exist so that the sensitivity of the statistic to these choices
  can be measured without editing the module. They do not make the constants
  adjustable conventions: any value other than the default puts the statistic
  outside the definition the manuscript reports.

### Where the version number lives

`README.md` no longer states a version. It said `0.2.0` through the 0.3.0,
0.3.1 and 0.3.2 releases, and updating it here would have re-laid the same
trap for the next release, so the number was removed rather than corrected;
the file now points at this changelog and at the concept DOI
`10.5281/zenodo.22132819`, which resolves to every version and cannot go
stale. The README's per-version archive DOI went with it for the same reason.

After that sweep the version exists in exactly two places:

- **The distribution metadata (`pyproject.toml`) is authoritative.**
  `__version__` is derived from it and is not a second copy.
- **`CITATION.cff` is the one maintained copy**, and it is maintained
  deliberately because Zenodo requires the field. It is named here so that it
  is a recorded exception rather than an unmarked second source: when the
  version moves, this is the file that moves with `pyproject.toml`, and it is
  the only one.

No CI check compares the two. A check that compares two copies of a value is
not a check, and removing copies is what removes the need for one. Everything
else that looks like a version string is a dependency constraint, the CFF
schema version, the `0.0.0+unknown` sentinel for an uninstalled source tree,
or a historical statement about which release something arrived in -- none of
which goes stale.

### Cross-language scope

The JavaScript and R packages are **not** changed in this release and do not
carry the new keyword arguments. The parity contract is over computed
quantities at the default settings, not over API surface, so it is unaffected;
`prerelation-r` already declares a narrower scope. The asymmetry is stated
here rather than left to be discovered.

## 0.3.2 — 2026-09-02

Test-only and metadata. **The library's computational behaviour is unchanged.**

- `__version__` is now derived from the installed distribution metadata rather
  than written out a second time. It had read `0.2.0` since the 0.2.0 release
  while `pyproject.toml` moved to 0.3.0 and then 0.3.1, so the package
  disagreed with its own build in both of those releases.
- The test that existed for this compared `__version__` against a third
  hard-coded copy of the same number and stayed green throughout. It now
  compares the package against `importlib.metadata.version`, which is the
  authority it was always supposed to be checking.
  
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
