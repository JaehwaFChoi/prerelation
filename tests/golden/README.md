# Golden vectors

Fixed fixtures that pin `prerelation` (Python) as the **reference
implementation** and give downstream implementations (JavaScript, R) an
exact — not merely distributional — target.

## Files

| file | contents |
|---|---|
| `fixture_{name}.csv` | one `(x, y)` pair per row, full-precision `repr` of float64, exact round-trip |
| `perm_indices_n{N}.csv` | the committed permutation index matrix for sample size `N` (199 rows) |
| `expected.json` | component-wise expected outputs per fixture, as full-precision strings |
| `generate_golden.py` | provenance: the script that wrote the files (consumers never re-run it) |

Fixtures: `product`, `min`, `independent`, `equivalence`,
`partial_equivalence` (synthetic, n = 400, seeds recorded in
`generate_golden.py`) and `ecpe_slice` (real data: the first 200 persons of
the ECPE skill-theta table, x = skill3, y = skill2; the committed fixture
CSV is the canonical data — the full theta table is not part of the repo).
The real-data fixture carries ties and a discrete support, which exercises
code paths the smooth synthetic sets do not.

## Expected outputs

For each fixture, `expected.json` records `n`, `v`, `v0`, `A1`,
`mass_ceiling_band`, `mass_interior`, `n_interior`, `p1_top`, `q`, `ell`,
`A2`, `PI`, `PI_reverse`, `Delta` and `perm_p`. Every float is stored as a
Python `repr` string so it parses back to the identical float64. Component
definitions (`delta = 0.05`, `TOP_Q = 0.8`, the ceiling band
`u >= 1 - delta`, the V-statistic baseline with the diagonal included) are
in `prerelation/core.py` and `tests/oracle/prereq_index_v2.py`.

## The permutation contract

`perm_indices_n{N}.csv` row `r` is the `r`-th permutation of `0..N-1` drawn
from `numpy.random.default_rng(20260827)` via `rng.permutation(N)` —
consuming the stream exactly as `perm_pvalue` consumes it. The contract for
any implementation:

1. read the fixture `(x, y)` and the index matrix `P`;
2. compute the observed `PI(x, y)`;
3. for each row `r`, compute `PI(x, y[P[r]])`;
4. `p = (1 + #{r : PI_r >= PI_obs}) / (199 + 1)`.

Because the permutations are shared as data, the resulting p-value must
**equal** the value in `expected.json` bit-for-bit; a match "in
distribution" is not sufficient. `generate_golden.py` asserts that this
matrix path reproduces `perm_pvalue(x, y, n_perm=199, seed=20260827)`
exactly for every fixture.

`tests/test_golden.py` regenerates every expected quantity from the
committed fixtures on each test run and asserts agreement at `1e-12`
(exact for the permutation p-values and counts).
