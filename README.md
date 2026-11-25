# epilot

C++17 / CUDA tool for **exhaustive pairwise SNP screening** in binary (case/control)
genetic data, built to investigate statistical epistasis. It reads PLINK
`.bed/.bim/.fam` files, builds the 2×3×3 genotype contingency table of every SNP
pair, and applies a selectable test statistic on CPU or GPU.

> **Status: research baseline (v0.2.1, work in progress).** The scan engine is
> exact and tested for coverage and CPU/GPU agreement, but the *statistics* are
> not yet calibrated interaction tests. Do not interpret the output
> scientifically before reading [Known limitations](#known-limitations).
> Direction and open work are tracked in [PLAN.md](PLAN.md).

Formerly `cudaepi`. The `epiload` repository provides the PLINK reader bridge
(`bed-reader-cpp`).

---

## Contents

- [Features](#features)
- [Repository layout](#repository-layout)
- [Build](#build)
- [Usage](#usage)
- [Output format](#output-format)
- [Input contract](#input-contract)
- [Methods](#methods)
- [Architecture](#architecture)
- [Testing](#testing)
- [Known limitations](#known-limitations)
- [Roadmap and contributing](#roadmap-and-contributing)
- [References](#references)
- [License](#license)

## Features

- PLINK BED/BIM/FAM input; unknown phenotypes are dropped, missing genotypes are
  excluded per pair.
- Exhaustive pair coverage through chunked tiling (`GS` SNPs per chunk, padded so
  coverage is exact for any SNP count).
- **One implementation of the per-pair logic** (`test_cell<Test>()`), compiled
  for host and device, so the CPU and GPU back ends give the same hits.
- Statistics are small functors (`epi_stats.h`) selected by name at run time and
  instantiated per back end at compile time.
- GPU hit buffer overflow is detected and reported as an error (no silent loss).
- Arithmetic is `double` by default (`-DEPI_REAL_FLOAT` selects `float`).

## Repository layout

| Path | Role |
| --- | --- |
| `include/epi_core.h` | Core types: `real`, `Hit`, `TestOut`, `TileView`, `SnpSource`, `ScanConfig` |
| `include/epi_stats.h` | Test functors (`FastEpistasis`, `FastEpistasis2`, `Chi2Absolute`, `FastThenChi2`) and the `with_test()` dispatcher |
| `include/epi_cell.h` | `test_cell<Test>()`: builds one pair's table and applies the test and hit filters (shared by CPU and GPU) |
| `include/epi_scanner.h` | `ScanContext`, `Tile`, `TileScanner` interface, `run_scan()`, and the `brute_force()` reference scan |
| `include/epi_cpu.h` | `CpuTileScanner` (single-threaded) |
| `include/epi_gpu.cuh`, `src/epi_gpu.cu` | `GpuTileScanner` (one CUDA thread per SNP pair) |
| `include/bed_dataset.h` | `BedDataset`: PLINK adapter over `bed-reader-cpp`, plus single-SNP scores |
| `src/main.cpp` | Command-line driver |
| `src/test_epi.cpp` | Test program (built as `epitest.exe`) |
| `scripts/fetch_deps.py`, `deps.yaml` | Dependency fetcher and manifest |
| `data/` | Small PLINK fixtures and reference outputs (`all.*`: 5,000 SNPs × 500 samples; `testepi.*`: 2 SNPs × 6 samples) |
| `Makefile` | Windows x64 build (see below) |
| `Release` | Version string |

## Build

Current build target: **Windows x64** with GNU Make, MSVC C++ Build Tools, CUDA
Toolkit 12.x (`nvcc`), and Python 3.12+ (for the dependency fetcher). Linux and
CPU-only builds are not supported yet (see [PLAN.md](PLAN.md), milestone M0).

`CUDA_PATH` must point to the Toolkit installation.

```text
make setup     # create .venv and install PyYAML
make doctor    # check Python / Make / PyYAML
make           # fetch deps, then build epi.exe and epitest.exe
make debug     # build/debug/epi.exe and build/debug/epitest.exe
make clean     # remove build outputs
make deps-force  # re-fetch dependencies
```

`CUDA_ARCH` defaults to `native`; override with `make CUDA_ARCH=sm_XX`.
The `bed-reader-cpp` 0.1.0 Windows x64 release declared in `deps.yaml` is
downloaded to `3rdparty/` and verified by SHA-256. Even `--device cpu` currently
links the CUDA object, so the CUDA Toolkit is required to build.

## Usage

```powershell
.\epi.exe --files "C:\data\study.bed" --methods fast chi2 --device gpu --verbose --noprompt
```

| Option | Meaning |
| --- | --- |
| `--files a.bed [b.bed ...]` | One or more BED files (matching `.bim`/`.fam` required in the same directory) |
| `--methods m1 [m2 ...]` | Tests to run, default `fast` (see table below) |
| `--device cpu\|gpu` | Back end, default `cpu` |
| `--verbose` | Also print the number of SNP pairs tested |
| `--noprompt` | Do not wait for Enter before exiting |

| Method name | Test |
| --- | --- |
| `fast` | alias of `fastepistasis+Chi2` |
| `fastepistasis` | `FastEpistasis` |
| `chi2` | `Chi2Absolute` |
| `fastepistasis+Chi2` | `FastThenChi2` |

Fixed in code for now (no command-line option yet): `GS = 1024` SNPs per chunk,
`min_stat = 0`, screening threshold `3.0`, excluded chromosomes `6` and `23`,
tile memory budget 500 MB, GPU block side 16, GPU hit capacity 2²² per tile.

## Output format

Results are **appended** to `<input path>.cc.log` (tab-separated). A header line
is written before each method's block, so a file holds one header per run and
method.

| Column | Meaning |
| --- | --- |
| `Test` | Method name used |
| `EdgeName` | `R<i>`, index of the hit within this block |
| `CHR1`, `SNP1`, `CHR2`, `SNP2` | Chromosome and SNP identifiers from the `.bim` |
| `STAT` | Test statistic of the reported pair |
| `OR` | Odds ratio of the reported cell (`Chi2Absolute`, `FastThenChi2`); `0` for `FastEpistasis` |
| `Source`, `Target` | Zero-based SNP indices in the input dataset |

Hits are sorted by `(Source, Target)`, so output is deterministic. The selected
genotype cell is computed internally (`Hit.row`, `Hit.col`) but not yet written
out.

## Input contract

- Genotypes are the allele count `0`, `1`, `2` as returned by `bed-reader`;
  missing is `-1`. Any other value aborts the scan.
- PLINK phenotype `2` = case, `1` = control. Any other phenotype (including
  `0`/`-9`) marks the sample as unknown and it is not used.
- The whole genotype matrix is loaded in memory by `BedDataset`.

## Methods

Notation: `CT[g][a][b]` is the number of samples in group `g` (case/control)
with genotype `a` at SNP 1 and `b` at SNP 2. `N[g]` counts samples with both
genotypes known.

- **`FastEpistasis`** — collapses each group's 3×3 table to a 2×2 allele-pair
  table (cells `Σ n·(2−a)(2−b)`, `Σ n·(2−a)·b`, `Σ n·a·(2−b)`, `Σ n·a·b`),
  then `z = |log OR_case − log OR_control| / sqrt(Σ 1/cell)`. Returns 0 if any of the 8 cells is empty.
- **`FastEpistasis2`** — same idea with the allele-pair cells built as
  `4·n00 + 2·n01 + 2·n10 + n11` (and analogues), and separate variances for
  cases and controls. It is used inside
  `FastThenChi2` but cannot yet be selected on its own.
- **Statistic scale (fast statistics).** `FastEpistasis` and `FastEpistasis2`
compute the normally distributed statistic `z = |log OR_case − log OR_control| / sqrt(v_case + v_control)` and
report **z²**, which is χ² with 1 df. This is deliberate: it matches
`--fast-epistasis` in PLINK 1.07 (`CHISQ`/`STAT` columns), so results can be
compared with PLINK output directly. A two-sided p-value is `2·Φ(−|z|)`
(= P(χ²₁ ≥ z²)); epilot does not output it yet (PLAN.md, M1). Thresholds such
as `screen_min` are compared against z², not |z|.
- **`Chi2Absolute`** — Pearson χ² (1 df, no continuity correction) of each of
  the nine joint-genotype cells against all other cells, cases vs controls. Only
  cells with all four counts positive are tested. Reports the cell with the
  largest χ² (ties: first in row-major order) and its odds ratio.
- **`FastThenChi2`** — runs `FastEpistasis2` and, if its statistic is
  `≥ screen_min`, runs `Chi2Absolute`.

**Hit criterion.** A pair is reported when its statistic exceeds `min_stat` and
exceeds the single-SNP score of both SNPs. The single-SNP score is the largest
1-df χ² of a genotype class against the other two, computed in `BedDataset`.

## Architecture

```text
SnpSource (BedDataset) ──► make_context() ──► run_scan()
                                                │  chunks of GS SNPs, pairs a ≤ b
                                                ▼
                                      TileScanner::scan(tile)
                              ┌─────────────────┴───────────────┐
                    CpuTileScanner (loop)              GpuTileScanner (1 thread / pair)
                              └─────────────────┬───────────────┘
                                      test_cell<Test>()   (epi_cell.h)
                                                │
                                      Test::run(CT, N, opt)   (epi_stats.h)
```

A tile holds two chunks of `GS` SNPs. For a diagonal tile (chunk with itself)
only `c > r` is tested. Padding and excluded SNPs carry `valid = 0` and are
skipped. Adding a statistic means writing a struct with
`static EPI_HD TestOut run(const real CT[2][3][3], const real N[3], const TestOptions&)`
and adding one line to `with_test()`.

## Testing

`make` also builds `epitest.exe` (from `src/test_epi.cpp`); run it from the
repository root. It covers:

- statistics against independent references,
- the CPU scan against a brute-force scan without tiling (same hits, same number
  of tests, planted interaction recovered),
- the GPU scan against the CPU scan,
- error handling (unknown test, invalid genotype, tile over budget, GPU overflow,
  `GS` not a multiple of the block side).

Known test state: the `FastEpistasis` statistic checks compare against z²
(the convention above). There is no continuous integration yet and no test has
been calibrated against a null distribution.

## Known limitations

These are the findings that matter for interpreting results; the full list with
identifiers is in [PLAN.md](PLAN.md).

1. **The default output is not a significance filter.** `min_stat` is 0 and there
   is no command-line option for it. On a phenotype-permuted null dataset
   (first 1,500 SNPs of `data/all.bed`, 1.12 M pairs, `Chi2`) 65 % of pairs were
   reported at `min_stat = 0`, versus one pair at `min_stat = 25`.
2. **The statistics are not interaction tests.** `Chi2Absolute` tests a single
   genotype cell against the rest, i.e. joint-genotype association, and the
   "beats both single-SNP scores" rule is a heuristic, not a hypothesis test. The
   maximum over nine cells is compared with a maximum over three, with no
   p-values, no degrees-of-freedom accounting, and no multiple-testing control.
3. **`screen_min` semantics are unsettled.** The default is 3.0 in code and is
   compared against z² of `FastEpistasis2`; older notes called it a z score, a
   p-value or a χ² threshold (PLAN.md, F-04).
4. **Allele-based fast statistics may be mis-calibrated.** They follow the PLINK
   fast-epistasis approach, whose formulas were reported to contain minor
   errors by Ueki & Cordell (2012).
5. **No covariates or structure.** Population stratification, relatedness and
   LD between the two SNPs are not modelled; there is no LD or distance filter.
6. **Sparse tables.** There is no expected-count check, no continuity
   correction and no exact test.
7. **Scale.** Genotypes are held fully in memory (and temporarily as floats),
   hits are accumulated in memory, and GPU kernels read unpacked 1-byte
   genotypes per sample.
8. **Platform.** Windows x64 only; building requires CUDA even for the CPU
   back end.

## Roadmap and contributing

[PLAN.md](PLAN.md) is the living plan: milestones, algorithm specifications
(BOOST/GBOOST, case-only and case-control contrasts, Ueki–Cordell, MDR, MB-MDR),
decision log and change log. Update it in the same commit as the change it
describes, and keep this README limited to what the code does *now*.

## References

- Purcell S et al. (2007) PLINK. *Am J Hum Genet* 81:559–575.
- Piegorsch WW et al. (1994) Non-hierarchical logistic models and case-only
  designs. *Stat Med* 13:153–162.
- Wu X et al. (2010) A novel statistic for genome-wide interaction analysis.
  *PLoS Genet* 6(9):e1001131.
- Wan X et al. (2010) BOOST. *Am J Hum Genet* 87:325–340.
- Ueki M, Cordell HJ (2012) Improved statistics for genome-wide interaction
  analysis. *PLoS Genet* 8(4):e1002625.
- Cordell HJ (2009) Detecting gene–gene interactions that underlie human
  diseases. *Nat Rev Genet* 10:392–404.

## License

Apache License 2.0, see [LICENSE](LICENSE).
