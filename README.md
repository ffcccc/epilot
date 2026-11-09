# cudaepi (-->epilot)

Development of statistical methods for detecting epistasis in genetic datasets,
using C++ and, where useful, CUDA acceleration.

The starting point is a legacy CUDA prototype for pairwise SNP screening in
binary case/control data. The repository now includes a Windows x64 Makefile
build and a PLINK BED/BIM/FAM input adapter, but remains a research baseline,
not a validated analysis pipeline. Known correctness issues below must be
resolved before interpreting its output scientifically. No CPU-only executable,
automated tests, or sample dataset is included.

## Repository Layout

| File | Role |
| --- | --- |
| [src/epi.cu](src/epi.cu) | GPU contingency tables, screening statistics, pair scanning, and CUDA lifecycle |
| [include/epi.h](include/epi.h) | Thresholds, result structures, genotype type, and host-facing CUDA declarations |
| [src/main.cpp](src/main.cpp) | BED dataset loading, chunking, GPU orchestration, and tab-separated output |
| [include/bed_dataset.h](include/bed_dataset.h) | PLINK BED/BIM/FAM adapter and single-SNP score calculation |
| [include/cutil_compat.h](include/cutil_compat.h) | Minimal replacements for the removed CUDA CUTIL helpers |
| [Makefile](Makefile), [deps.yaml](deps.yaml) | Windows x64 build targets and the Rust BED-reader bridge dependency |
| [src/cuPrintf.cu](src/cuPrintf.cu), [src/cuPrintf.cuh](src/cuPrintf.cuh) | Bundled device-printing helper, not used by the active scan |

The Makefile builds `epi.exe` or `build/debug/epi.exe`; existing binaries and
PDBs are build artifacts, not evidence that the current source passes tests.

## Current Method

1. Read genotypes from a PLINK `.bed` file and SNP/sample metadata from matching
  `.bim` and `.fam` files. Single-SNP comparison scores are computed by the
  `BedDataset` adapter rather than loaded from an external reader.
2. Partition the SNP matrix using a hard-coded 500,000,000-byte genotype-memory
   budget. Scan within chunks and between pairs of chunks.
3. Assign one CUDA thread to each scheduled SNP pair. Build a `2 x 3 x 3`
   contingency table indexed by phenotype and the two genotype codes. Exclude
   samples missing either genotype and adjust case/control totals for that pair.
4. Compute a fast-epi-style screening statistic comparing allele-product odds
   ratios in cases and controls. The intended absolute standardized contrast is
   the difference in log odds ratios divided by the square root of the sum of
   reciprocal allele-product counts. These are weighted genotype counts, not
   inferred phased haplotypes. The current implementation returns zero because
   of variable shadowing; the p-value conversion is commented out.
5. For pairs passing the screen, test each of the nine joint-genotype cells
   against all other joint genotypes in a case/control 2x2 table. With `A` and
   `B` the case/control counts in the selected cell and `C` and `D` the remaining
   case/control counts, compute:

   ```text
   N  = A + B + C + D
   X2 = N * (A*D - B*C)^2 / ((A+C)*(A+B)*(C+D)*(B+D))
   OR = (A*D) / (B*C)
   ```

   Only tables with all four counts strictly positive are evaluated. There is
   no continuity correction or exact test for sparse tables.
6. Retain a pair if a cell exceeds `CHI10E9` (currently 25) and the retained
   statistic exceeds both supplied single-SNP scores. The current code keeps
   the last qualifying cell in traversal order, not the strongest cell.

The active path uses `ctEpiAbs`; `ctEpiRel` is an unused alternative comparing
cells against the double heterozygote. `VecAdd` is unrelated example code.

### Statistical Interpretation

The cell test detects joint-genotype association with phenotype. Comparing its
statistic with single-SNP scores does not establish interaction beyond main
effects; the adapter computes each single-SNP score as the maximum genotype-cell
chi-square statistic.
A formal interaction model, null hypothesis, calibration, and validation are
still needed before claiming detection of epistasis.

Despite its name, `CHI10E9` is a statistic cutoff of 25, approximately a
5.7e-7 upper-tail probability under a one-degree-of-freedom chi-square reference.
It is not a study-wide significance guarantee. The code applies no explicit
multiple-testing correction across SNP pairs or the nine cell comparisons and
does not check expected-count adequacy. Population structure, relatedness, and
other covariates are not modeled.

## Input Contract

- Genotypes are stored SNP-major: `M[snp * nSamples + sample]`.
- Valid genotype codes are `0`, `1`, and `2`; negative values indicate missing
  genotypes. Allele orientation must be consistent with the dataset reader.
- The adapter maps PLINK FAM phenotype `1` to `0` (control) and `2` to `1`
  (case). Other phenotype values become missing, which the kernel does not
  handle safely; use datasets containing only `1` and `2` phenotypes.
- `TF` is plain `char`; missing-value handling requires signed-char behavior
  so BED missing genotypes remain negative in the kernel.
- The adapter computes each SNP's univariate score from its genotype counts.

Pass the PLINK `.bed` path; matching same-basename `.bim` and `.fam` files must
be in the same directory. Genotypes are read by the Rust-backed
`bed-reader-cpp` bridge.

## Build Status

The current Makefile targets Windows x64 with GNU Make, MSVC C++ Build Tools,
and CUDA Toolkit 12.x (`nvcc` and the CUDA runtime). It uses C++17 and defaults
`CUDA_ARCH` to `native`, compiling for the GPU architecture detected by NVCC.
`CUDA_PATH` must point to the selected Toolkit installation for linking.
The old CUTIL dependency has been replaced by `include/cutil_compat.h`. The
Makefile fetches the Rust-backed `bed-reader-cpp` 0.1.0 release declared in
`deps.yaml`; Python 3.12+ and PyYAML are used by the dependency fetcher.

From the project directory, set up the dependency fetcher and build the
executable:

```text
make setup
make doctor
make
```

`make` fetches dependencies and builds `epi.exe`. To build the debug executable,
run `make debug` after dependencies are available; it writes
`build/debug/epi.exe`. Set `CUDA_ARCH` on the Make command line to override the
default architecture, for example `make CUDA_ARCH=sm_XX`. This Makefile does not
provide a Linux, cross-platform, or CPU-only build.

## Invocation And Output

With the required build tools installed and a PLINK dataset available:

```powershell
.\epi.exe --files "C:\data\study.bed" --noprompt
```

`--files` accepts one or more BED paths. Each BED file must have matching BIM
and FAM siblings. The program appends tab-separated results to
`<input filename>.cc.log`; it waits for Enter before exiting unless
`--noprompt` is supplied. Headers are repeated for each tile, and `EdgeName`
numbering restarts for each tile.

| Output | Meaning |
| --- | --- |
| `EdgeName` | Tile-local generated result label |
| `CHR1`, `SNP1`, `CHR2`, `SNP2` | Reader-provided chromosome and SNP identifiers |
| `X2 STAT` | Retained joint-genotype cell's Pearson statistic |
| `fast-epi P` | Misnamed screening field: assigned a z statistic, currently always zero |
| `OR` | Selected cell's case/control odds ratio; not a relative risk |
| `Source`, `Target` | Zero-based SNP indices mapped back to the input dataset |

The selected genotype cell itself is not recorded. Output ordering is not
deterministic because GPU threads reserve result slots atomically. The result
handling defects below mean even apparently populated rows are not trustworthy.

## Initial Code Review

These are source-review findings, not fixes. The documentation changes preserve
the executable behavior.

### Critical: Result Integrity And Input Safety

- **Uninitialized results:** `kernelLaunch` allocates `d_R` without initializing
  it, then copies the entire buffer back. Host-side initialization is not copied
  to the device. `printout` scans all entries using only `stat > 0.0001`, so
  unwritten slots can be mistaken for results and yield invalid metadata indices.
- **Uninitialized excluded SNPs:** `main1` skips loading chromosomes `6` and
  `23`, but still advances their matrix positions and schedules them for testing.
  Both genotype and single-SNP-score slots remain uninitialized. Invalid
  genotype values can also index outside the contingency table.
- **Unchecked phenotype/genotype codes:** `VecEpi` directly indexes `CT` and
  `NA` using input values. Missing or invalid phenotypes, or positive genotype
  codes above 2, can cause out-of-bounds device accesses.

### High: Screening And Pair Coverage

- **Disabled first-stage screen:** `fastepi` declares a second `z` inside the
  positive-count branch. The returned outer `z` is always zero. Simply removing
  the shadowing is insufficient: `VecEpi` uses `z < P_Epi`, where `P_Epi = 0.01`
  is a p-value-style cutoff. Define z-score versus p-value semantics and sparse
  count behavior together before fixing this stage.
- **Incomplete or incorrect pair scheduling:** grid dimensions use integer
  division by 16, dropping partial blocks and potentially launching zero-sized
  grids. The diagonal mapping assumes an even SNP count; cross-chunk mapping
  assumes two equally sized halves, although the final chunk can be shorter.
  Ceiling division alone is insufficient without bounds checks and explicit
  chunk lengths. Tiny datasets can also make `matrixPartition` divide by zero
  when the computed half-chunk length is zero.
- **Wrapping result reservation:** `atomicInc(ResIndex, NumRes)` wraps rather
  than reporting overflow. Slot zero is skipped, then earlier results can be
  overwritten. The host receives neither a valid result count nor an overflow
  flag. The allocation has 10,001 entries, with 10,000 intended result slots.

### Other Correctness And Validation Gaps

- `ctEpiAbs` overwrites the result for every passing cell, so a weaker later
  cell can mask a stronger cell and change the single-SNP comparison outcome.
  The inactive CPU version repeats this behavior and is not an independent oracle.
- `ctEpiRel`, if enabled, uses the whole pair's sample total in place of the
  four selected cells' total, inflating the restricted comparison's statistic.
- `NTest` starts at one and wraps at 10,000 via `atomicInc`; it is not an accurate
  count of tested pairs. Its allocation is not explicitly freed in `CudaCleanup`,
  although the subsequent legacy context teardown releases device resources.
- Sparse/degenerate tables have no explicit result status, and float arithmetic
  can lose count precision or overflow intermediate products on large cohorts.
- No automated correctness, CPU/GPU agreement, pair-coverage, or calibration
  tests are included. Historical build logs do not substitute for these checks.

## Development Priorities

1. Specify the interaction null hypothesis, genotype/phenotype encoding,
   screening statistic, missing-data policy, and multiple-testing strategy.
2. Establish small deterministic C++ reference tests for contingency tables,
   known statistics, null effects, main effects only, and simulated interactions.
3. Correct input validation, pair scheduling, result initialization, capacity
   handling, and screening semantics; check CPU/GPU agreement and exact pair
   coverage on odd, small, non-block-aligned, and unequal chunk sizes.
4. Introduce a reproducible modern C++ build with optional CUDA support and
   self-contained fixtures before optimizing throughput.
5. Evaluate calibration, power, and performance on documented simulations and
   appropriately controlled genetic datasets.
