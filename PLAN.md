# epilot — development plan

Living document. It records what is wrong, what is next, why decisions were
made, and what changed. The README describes what the code does *now*; this file
describes where it is going.

**Baseline:** commit `f038f33` (2026-11-25), version 0.2.1.
**Goal:** a validated, fast pairwise epistasis scanner for case/control GWAS data
that offers several published detection methods (BOOST/GBOOST, case-only and
case-control contrasts, Ueki–Cordell, MDR, MB-MDR) behind one engine, with
calibrated p-values and reproducible benchmarks.

## How to use this file

- Every finding has an ID (`F-xx`), every milestone an ID (`M<n>`), every
  decision an ID (`D-xx`). Reference them in commit messages (`M0: fix F-03`).
- Status markers: `[ ]` open, `[~]` in progress, `[x]` done, `[-]` dropped
  (keep the line and say why).
- Update this file in the same commit as the change. Add a line to the
  [Change log](#change-log); do not rewrite history in the other sections.
- When a finding is fixed, change its status here and leave the README
  limitation list in sync.

## 1. Baseline findings (2026-11-25)

Source: code review of `f038f33`, build of the test program, and an experiment on
`data/all.bed`. "Verified" means reproduced by running code.

### Statistics and science

| ID | Sev | Finding | Status |
| ---- | --- | --- | --- |
| F-01 | High | Default scan reports noise. `main.cpp` uses `ScanConfig` with `min_stat = 0` and exposes no option; the old `CHI10E9 = 25` cutoff is unused. Verified: on phenotype-permuted data (1,500 SNPs, 1.12 M pairs, `Chi2`) 65.5 % of pairs are reported at `min_stat = 0`, 1 at `min_stat = 25`. | [ ] |
| F-02 | High | `Chi2Absolute` is a joint-genotype *association* test (cell vs rest), not an interaction test. The "beats both single-SNP scores" filter compares a maximum over 9 cells with a maximum over 3 and is not calibrated. No p-value, no df, no multiple-testing control. | [ ] |
| F-03 | High | z vs z² mismatch. `FastEpistasis` returns z² (`out.stat = z*z`); its doc comment, the test and the README describe \|z\|. Verified: all 200 `FastEpistasis` checks in `test_statistics()` fail with got = want². | [ ] |
| F-04 | Med | `screen_min` semantics: default `3.0` in `ScanConfig`, `0.01` in old docs, `EpiScreeningTH`/`P_Epi` macros unused. Compared against z² of `FastEpistasis2`. | [ ] |
| F-05 | Med | `FastEpistasis2` is unreachable by name (`with_test` maps `fastepistasis` to `FastEpistasis`) and has no unit test. Two different allele-pair constructions exist without a documented rationale. | [ ] |
| F-06 | Med | Screening then testing the same data (`FastThenChi2`) invalidates stage-2 p-values unless the screen is independent of, or accounted for in, the second test. | [ ] |
| F-07 | Med | Single-SNP score (max genotype-vs-rest χ², float, in `BedDataset`) is a different statistic from the pair statistics it is compared with (z², cell χ²). | [ ] |
| F-08 | Med | Allele-based fast statistics assume independent alleles; Ueki & Cordell (2012) report formula errors in related statistics. Calibration unknown. | [ ] |
| F-09 | Med | No sparse-table policy (expected counts, continuity correction, exact test) and no result status (`tested/degenerate/skipped`). | [ ] |
| F-10 | Med | No covariates, population-structure adjustment, relatedness handling, or LD/distance filter between the two SNPs. | [ ] |
| F-11 | Low | Chromosomes `6` and `23` are silently excluded by default and cannot be changed from the command line. | [ ] |

### Engineering

| ID | Sev | Finding | Status |
| --- | --- | --- | --- |
| F-20 | High | Windows-only build (`cmd.exe`, `cl`, `vswhere`; Rust bridge only for Windows x64). No Linux build. | [ ] |
| F-21 | High | `--device cpu` still needs CUDA: `main.cpp` includes `epi_gpu.cuh` and the link line always includes `epi_gpu.obj`. | [ ] |
| F-22 | Med | No continuous integration; no `make test` target; test header comment still refers to `tests/test_epi.cpp`. | [ ] |
| F-23 | Med | `clean-deps` uses `rm -rf` while `SHELL := cmd.exe`; `OBJS` variable unused. | [ ] |
| F-24 | Med | `epi_gpu.cu` carries ~360 lines of commented legacy code; `epi_core.h` has unused `T_Res`, `T_Cel`, `TF`, `MAX_TEST`, `P_Epi`, `EpiScreeningTH`, `CHI10E9`. | [ ] |
| F-25 | Low | `cuda_device_info()` is declared `inline` in a header and defined in the `.cu`; it is not usable and not called. | [ ] |
| F-26 | Low | `bed_dataset.h` has `using namespace epi;` and includes via project-relative paths; `brute_force()` (a test oracle) lives in a production header and does not validate genotype codes. | [ ] |
| F-27 | Low | `.gitignore` ignores `*.log`, so default output files (`*.cc.log`) are ignored; `README.demo.md` is leftover. | [ ] |
| F-28 | Low | Output: header repeated per run and method, selected cell not written, no p-value/df columns, `EdgeName` is only a running index. | [ ] |

### Performance and scale

| ID | Sev | Finding | Status |
| --- | --- | --- | --- |
| F-40 | Med | `BedDataset` loads the whole matrix as Rust `Vec<float>`, then copies to bytes: peak memory about 5× the genotype matrix. | [ ] |
| F-41 | Med | GPU kernel: each thread loops over all samples reading two unpacked 1-byte rows (uncoalesced), uses `double` by default (slow on consumer GPUs), and does one `atomicAdd` per tested pair. | [ ] |
| F-42 | Low | Chunk A is re-uploaded for every tile; no copy/compute overlap; single GPU; CPU back end is single-threaded; hits accumulated in memory and sorted at the end. | [ ] |

## 2. Principles and decisions

| ID | Decision | Why |
| --- | --- | --- |
| D-01 | One per-pair code path (`test_cell<Test>`) for CPU and GPU. Keep. | Guarantees CPU/GPU agreement; CPU is the oracle in tests. |
| D-02 | Table-based statistics stay functors of `(CT, N)`; methods that need raw genotypes (MDR, MB-MDR) get a separate engine. | Per-thread 3×3 tables are GPU-friendly; MDR-type methods need cross-validation and permutation. |
| D-03 | Internal statistic scale is χ² with an explicit df; every test returns `{stat, df, p, status}`. | Makes tests comparable, enables one threshold rule and QQ plots. |
| D-04 | Interaction p-values are only claimed for tests with a stated null hypothesis and calibration evidence from simulations. Other outputs are labelled "screening score". | Avoids the current ambiguity (F-02). |
| D-05 | Every new method ships with: a written specification here, a unit test against an independent reference, a null-calibration run, and a benchmark entry. | Prevents repeating F-03. |
| D-06 | Open question: keep the Windows Makefile and add CMake, or replace with CMake. | Linux CI and CPU-only builds need CMake; decide in M0. |

## 3. Milestones

### M0 — Reproducible, honest baseline

Acceptance: a clean clone builds and passes tests on Windows **and** Linux (CPU
build without CUDA), and default output is no longer flooded with noise.

- [ ] Fix F-03: choose one convention (see D-03), update `FastEpistasis`, its doc
      comment, the test, and the README together.
- [ ] Fix F-04/F-05: one documented meaning of `screen_min`; expose
      `FastEpistasis2` by name (`fastepistasis2`); add its unit test against an
      independent reference.
- [ ] Fix F-01: add `--min-stat`, `--GS`, `--excluded-chr` (F-11), `--out`; set a
      sensible default and print the thresholds used.
- [ ] Fix F-20/F-21: `EPI_WITH_CUDA` switch; CPU-only link; decide D-06 (CMake
      with an optional CUDA language, or keep Make plus a CMake file for Linux).
- [ ] Fix F-22/F-23: `make test` (or `ctest`), move tests to `tests/`, fix
      `clean-deps`, remove unused variables.
- [ ] CI: Linux CPU build and tests on every push; Windows build if a runner is
      available. GPU tests run locally and are listed in the release checklist.
- [ ] Fix F-24/F-25/F-26/F-27: delete dead code and macros, move `brute_force`
      into the test code, tidy headers, un-ignore the output extension or change
      the default, remove `README.demo.md`.
- [ ] Output format v2 (F-28): one header per file, add cell, df, p-value,
      status; write hits as they arrive.

### M1 — Result schema and statistics framework

Acceptance: all tests return `{stat, df, p, status, cell}`; one threshold rule
(`--alpha`, with Bonferroni over tested pairs, or `--min-stat`); QQ/λ report.

- [ ] `TestOut` extension (status codes: ok, sparse, degenerate, skipped).
- [ ] Device-usable p-value helpers (χ² survival, normal tail).
- [ ] Sparse-table policy (F-09): configurable minimum expected count; skipped
      pairs counted and reported.
- [ ] Stage abstraction: screen then confirm, with a pre-specified screen and
      the correction for it documented (F-06).
- [ ] Replace the "beats both single-SNP scores" rule with a documented option;
      keep it only as an explicit `--marginal-filter` (F-02, F-07).
- [ ] Genomic-control λ and QQ data written for each run.

### M2 — Table-based methods (cheapest first)

Acceptance for each method: spec below is implemented, unit test matches an
independent reference (Python or R), and null calibration is within tolerance on
simulations (M4 harness).

- [ ] `CaseOnlyChi2` (4 df).
- [ ] `ContrastChi2`: case-control difference of the 3×3 association.
- [ ] Ueki–Cordell adjusted statistics (`UekiAdj*`) and joint-effects statistics
      (`UekiJoint*`).
- [ ] BOOST stage 1 (`BoostKSA`, 4 df) and stage 2 (`BoostLRT`, host side).
- [ ] Optional: logistic interaction test with covariates on survivors (host).

### M3 — Engines for data-based methods

- [ ] Pair-batch engine that keeps the pair's genotype columns, supports
      k-fold cross-validation and permutations (shared by MDR and MB-MDR).
- [ ] MDR with cross-validation consistency and balanced accuracy.
- [ ] MB-MDR with its permutation-based p-values.
- [ ] Run on survivors of an M2 screen by default; exhaustive mode behind a flag.

### M4 — Validation harness

- [ ] Simulator (Python script in `scripts/`): null, main effects only, LD
      between SNPs, planted interactions for several penetrance models, optional
      stratification.
- [ ] Cross-checks on identical data against PLINK `--fast-epistasis` /
      `--epistasis` and BOOST/GBOOST where available.
- [ ] Reports: type-1 error and λ under the null; power curves; runtime.
- [ ] Replace the tiny `testepi.*` fixture with a documented mid-size one and
      commit the PLINK reference outputs for it.

### M5 — Performance and scale

- [ ] Block-wise BED reading (F-40); no full float copy.
- [ ] 2-bit packed genotypes and bit-plane AND + popcount table building (CPU
      and GPU) (F-41).
- [ ] `float` GPU path validated against the `double` path; block-level
      reduction for counters; overlap copy and compute; upload chunk A once per
      row of tiles (F-42).
- [ ] Multi-threaded CPU back end (OpenMP or std threads).
- [ ] Streaming output with a bounded hit buffer.
- [ ] Benchmarks recorded in section 6.

### M6 — Release 0.3

- [ ] README reflects the code; limitations list pruned.
- [ ] Tagged release with binaries, benchmark table, and calibration report.

## 4. Algorithm specifications

Each entry is a contract for implementation. Formulas marked **verify** must be
transcribed from the cited paper and checked before coding; they are not copied
here from memory.

### 4.1 Case-only test (Piegorsch 1994)

- Input: `CT[AFF]` (3×3).
- Statistic: Pearson χ² of independence, df = 4.
- Null: the two SNPs are independent in the population. Invalid under LD or
  stratification; apply the LD/distance filter (M1) and document.
- Sparse policy: minimum expected count, else status `sparse`.

### 4.2 Case-control contrast

- Input: `CT[AFF]`, `CT[UNAFF]`.
- Statistic: difference of the SNP–SNP association between cases and controls
  (genotypic, 4 df, or allelic, 1 df). Exact form: **verify** (Wu 2010; Zhao 2006;
  Ueki–Cordell 2012 for the corrected variance).

### 4.3 Ueki–Cordell adjusted and joint-effects statistics

- Reference: Ueki & Cordell (2012), *PLoS Genet* 8(4):e1002625.
- The paper reports formula errors in Wu et al.'s statistics (inflated type-1
  error) and minor errors in PLINK's fast-epistasis and case-only statistics. It
  proposes adjusted versions of all four statistics and two "joint effects"
  statistics that respond to genuine interaction only (for a rare disease).
- Plan: implement the adjusted versions of the statistics currently in
  `FastEpistasis`/`FastEpistasis2` first (they replace them), then the joint
  effects statistics. Exact estimators and variances: **verify**.
- Acceptance: type-1 error at the nominal level on the null simulations, with
  LD between the SNPs; agreement with an independent implementation.

### 4.4 BOOST / GBOOST (Wan 2010; Yung 2011)

- Stage 1: interaction statistic from the 9×2 table using the Kirkwood
  superposition approximation of the no-interaction model; 4 df. Exact
  expression and zero-cell handling: **verify**.
- Stage 2: likelihood-ratio test of the saturated model against the
  main-effects-only logistic model; 4 df; iterative fit, run on the host for
  pairs that pass stage 1 (`--boost-threshold`, with the correction for the
  screen documented per F-06).
- GBOOST is the GPU implementation of BOOST. Here it is not a separate
  statistic: the work is the performance milestone (M5, bit-plane table
  construction).
- Acceptance: stage 1 reproduces BOOST output on the same data.

### 4.5 MDR (Ritchie 2001)

- Pair-level: label each of the 9 genotype combinations high/low risk by the
  case:control ratio, classify, estimate balanced accuracy with k-fold
  cross-validation, report cross-validation consistency; significance by
  permutation.
- Needs the batch engine (M3). Default folds and the tie rule are to be fixed in
  this section before coding.

### 4.6 MB-MDR (Calle 2008 and later work)

- Pair-level: per-genotype-combination association tests classify cells as
  high, low, or no evidence, then a statistic on the resulting two-group
  contrast; significance by permutation. Exact procedure and corrections for
  the main-effect adjustment: **verify** against the MB-MDR papers and the
  reference implementation.
- Needs the batch engine and permutation support (M3).

## 5. Risks and open questions

- Streaming exhaustive MDR/MB-MDR on millions of SNPs is infeasible; the default
  is screen-then-confirm, which interacts with F-06.
- Which dataset will be the real target (sample size, SNP count, binary vs
  quantitative trait, covariates)? This decides how far M5 must go and whether a
  quantitative-trait path is needed.
- `bed-reader-cpp` is distributed as a Windows-only release; a Linux asset (or a
  native BED reader) is needed for M0.
- GPU tests need a real device; decide whether to keep the CUDA emulation shim
  for CI.

## 6. Benchmarks (fill in as they are produced)

| Date | Commit | Dataset | Method | Device | Pairs | Time | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-25 | `4e4c197` | `data/all.bed`, first 1,500 SNPs | `Chi2` | CPU, 1 thread | 1.12 M | ~0.6 s | Emulation harness, not `epi.exe` |

## 7. Change log

- 2026-11-25 — Plan created from the review of `f038f33`. Added README rewrite.
  Findings F-01…F-42 recorded; milestones M0–M6 defined.
