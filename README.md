# cudaepi (-->epilot)

Development of statistical methods for detecting epistasis in genetic datasets,
using C++ and, where useful, CUDA acceleration.

The starting point is a legacy CUDA prototype for pairwise SNP screening in
binary case/control data. It is a research baseline, not a validated analysis
pipeline. Known correctness issues below must be resolved before interpreting
its output scientifically. No CPU-only executable, automated tests, example
dataset, or modern portable build configuration is currently included.

## Repository Layout

| File | Role |
| --- | --- |
| [epi.cu](epi.cu) | GPU contingency tables, screening statistics, pair scanning, and CUDA lifecycle |
| [epi.h](epi.h) | Thresholds, result structures, genotype type, and host-facing CUDA declarations |
| [main.cpp](main.cpp) | HDF5 input, chunking, GPU orchestration, and tab-separated output; also inactive CPU comparison code |
| [common.h](common.h) | Placeholder header |
| [cudaepi.vcproj](cudaepi.vcproj) | Legacy Visual Studio 2008 Win32 project |
| [cuPrintf.cu](cuPrintf.cu), [cuPrintf.cuh](cuPrintf.cuh) | Bundled device-printing helper, not used by the active scan |

`Debug/` and `Release/` contain historical build artifacts, not a reproducible
build or evidence that the current source passes tests.

## Current Method

1. Read genotypes, binary phenotypes, SNP metadata, and precomputed single-SNP
   scores through the external `hdfTPED` reader.
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
effects, and the external scores' definition is not available in this repository.
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
- Phenotypes must be exactly `0` (control) or `1` (case). The kernel directly
  uses them as array indices and does not handle missing phenotypes safely.
- `TF` is plain `char`; missing-value handling requires signed-char behavior
  and compatibility with the reader's external `T_GType` definition.
- Each SNP must have a supplied univariate score on a scale appropriate for
  the comparison performed by the kernel.

The input is an HDF5 file in the external `hugedatafile` reader's format, not a
generic HDF5 schema or a direct VCF/PLINK input. The reader and its schema are
not included, so dataset conversion and schema details cannot yet be specified
from this repository alone.

## Build Status

The checked-in project targets Visual Studio 2008, Win32, CUDA Toolkit/SDK 3.2,
and CUDA compute capability 2.0. It depends on:

- NVIDIA CUDA runtime and the old GPU Computing SDK helpers, including
  `cutil_inline.h` (not provided by current CUDA toolkits).
- HDF5 C and C++ libraries, with configuration-specific DLL import libraries.
- External `hugedatafile` headers and sources referenced from a sibling project.

The project also contains machine-specific include paths, environment variables
such as `MYWORKSPACE`, `CUDA_INC_PATH`, `CUDA_LIB_PATH32`, and
`NVSDKCOMPUTE_ROOT`, and a post-build copy to an external directory. Review
these before attempting a legacy build. Some additional historical include
paths are listed in the project; their necessity cannot be verified without
the external reader sources.

To reproduce the historical build, restore compatible external dependencies,
correct the paths and post-build event, then build `Debug|Win32` or
`Release|Win32` in the matching Visual Studio/CUDA environment. A current CUDA
installation alone is insufficient. A modern build will require replacing
obsolete SDK helpers and updating the project and GPU target; no working modern
build command is claimed here.

## Legacy Invocation And Output

With a working legacy executable and a compatible input file:

```powershell
.\cudaepi.exe "C:\data\study.h5"
```

The program appends tab-separated results to `<input filename>.cc.log` and
normally waits for Enter before exiting. Headers are repeated for each tile,
and `EdgeName` numbering restarts for each tile. Running with no input currently
processes no datasets. Although `--noprompt` is recognized, adding it to a file
argument makes `argc == 3`, which bypasses dataset processing; do not use that
combination until argument handling is corrected.

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
