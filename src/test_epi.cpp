// Tests for the epistasis scan. Build and run from the repository root:
//   g++ -std=c++17 -O2 -Wall -Wextra -I. -o /tmp/test_epi tests/test_epi.cpp && /tmp/test_epi
//
// Covers:
//  - the test statistics against independent references
//  - the CPU scan against a brute-force scan with no tiling (every pair once, same hits)
//  - the GPU scan (run through tests/cuda_shim.h) against the CPU scan
//  - error handling (unknown test, bad genotype, overflow, memory budget)

// #include "include/bed_dataset.h"
// #include "include/tests/cuda_shim.h"
#include <cuda_runtime.h>

#include "include/epi_core.h"
#include "include/epi_cpu.h"    // CPU back end declarations
#include "include/epi_gpu.cuh"  // GPU back end declarations
// "epi_gpu.cu" --> epi.obj // GPU back end, compiled by nvcc
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace epi;

namespace {

int failures = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            ++failures;                                     \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                       \
            std::printf("\n");                              \
        }                                                   \
    } while (0)

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

// In-memory dataset.
struct MockSource : SnpSource {
    std::vector<std::vector<int8_t>> G;  // [snp][sample]
    std::vector<int> pheno;              // per sample: 1, 0, or -1 (unknown)
    std::vector<std::string> chr;        // per SNP
    std::vector<float> sc;               // per SNP
    size_t numSnps() const override { return G.size(); }
    size_t numSamples() const override { return pheno.size(); }
    // size_t numPhenotypes() const override { return pheno.size(); }
    int8_t genotype(size_t i, size_t j) const override { return G[i][j]; }
    int phenotype(size_t s) const override { return pheno[s]; }
    std::string chromosome(size_t i) const override { return chr[i]; }
    void genotypes(size_t i, int8_t* out) const override { std::copy(G[i].begin(), G[i].end(), out); }
    float score(size_t i) const override { return sc[i]; }
};

// Random genotypes, ~2% missing, a planted interaction between SNP 3 and SNP 7,
// ~10% unknown phenotypes, and some SNPs on excluded chromosomes.
MockSource make_source(size_t nSnp, size_t nSamp, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    MockSource s;
    s.G.assign(nSnp, std::vector<int8_t>(nSamp));
    for (size_t i = 0; i < nSnp; ++i)
        for (size_t k = 0; k < nSamp; ++k)
            s.G[i][k] = u(rng) < 0.02 ? static_cast<int8_t>(-1) : static_cast<int8_t>(rng() % 3);
    s.pheno.resize(nSamp);
    for (size_t k = 0; k < nSamp; ++k) {
        const bool risk = nSnp > 7 && s.G[3][k] == 2 && s.G[7][k] == 2;
        s.pheno[k] = u(rng) < (risk ? 0.9 : 0.3) ? 1 : 0;
        if (u(rng) < 0.1) s.pheno[k] = -1;
    }
    s.chr.resize(nSnp);
    s.sc.resize(nSnp);
    for (size_t i = 0; i < nSnp; ++i) {
        s.chr[i] = i % 11 == 5 ? "6" : (i % 13 == 2 ? "23" : "1");
        s.sc[i] = static_cast<float>(4.0 * u(rng));
    }
    return s;
}

bool same_hits(const std::vector<Hit>& a, const std::vector<Hit>& b) {
    if (a.size() != b.size()) return false;
    for (size_t k = 0; k < a.size(); ++k) {
        if (a[k].s1 != b[k].s1 || a[k].s2 != b[k].s2 || a[k].row != b[k].row || a[k].col != b[k].col)
            return false;
        if (!near(a[k].stat, b[k].stat, 1e-9) || !near(a[k].risk, b[k].risk, 1e-9)) return false;
    }
    return true;
}

// Independent z-score for FastEpistasis, computed from raw genotypes.
double reference_fast_z(const std::vector<int>& g1, const std::vector<int>& g2,
                        const std::vector<int>& cc) {
    double T[2][4] = {};
    for (size_t k = 0; k < g1.size(); ++k) {
        const int a = g1[k], b = g2[k], grp = cc[k];
        T[grp][0] += (2 - a) * (2 - b);
        T[grp][1] += (2 - a) * b;
        T[grp][2] += a * (2 - b);
        T[grp][3] += a * b;
    }
    double lor[2], var = 0;
    for (int g = 0; g < 2; ++g) {
        const double A = T[g][0], B = T[g][1], C = T[g][2], D = T[g][3];
        if (A <= 0 || B <= 0 || C <= 0 || D <= 0) return 0.0;
        lor[g] = std::log((A * D) / (B * C));
        var += 1 / A + 1 / B + 1 / C + 1 / D;
    }
    return std::fabs(lor[AFF] - lor[UNAFF]) / std::sqrt(var);
}

const char* kTestNames[] = {"fastepistasis", "Chi2", "fastepistasis+Chi2"};

// ---------------------------------------------------------------------------------------
void test_statistics() {
    TestOptions opt = {0.0};
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 200; ++trial) {
        const int n = 200 + trial * 7;
        std::vector<int> g1(n), g2(n), cc(n);
        real CT[2][3][3] = {};
        for (int k = 0; k < n; ++k) {
            g1[k] = rng() % 3;
            g2[k] = rng() % 3;
            cc[k] = rng() % 2;
            CT[cc[k]][g1[k]][g2[k]] += 1;
        }
        const real N[3] = {0, 0, static_cast<real>(n)};  // FastEpistasis ignores N
        const double got = FastEpistasis::run(CT, N, opt).stat;
        const double want = reference_fast_z(g1, g2, cc);
        CHECK(near(got, want, 1e-12), "FastEpistasis trial %d: got %.15g want %.15g", trial, got, want);
    }
    // Chi2 on a table with a known answer (scipy.stats.chi2_contingency, no correction):
    // cell (0,0) has chi2 = 426.6026, cell (2,2) has 6.044, cell (1,1) has 388.555.
    real T[2][3][3] = {};
    T[AFF][0][0] = 1000;
    T[UNAFF][0][0] = 100;
    T[AFF][2][2] = 60;
    T[UNAFF][2][2] = 40;
    T[AFF][1][1] = 500;
    T[UNAFF][1][1] = 500;
    real N[3];
    N[AFF] = 1560;
    N[UNAFF] = 640;
    N[ALL] = 2200;
    const TestOut o = Chi2Absolute::run(T, N, opt);
    CHECK(near(o.stat, 426.6026, 1e-5), "Chi2 should be the max cell 426.6026, got %.6f", o.stat);
    CHECK(o.row == 0 && o.col == 0, "Chi2 cell should be (0,0), got (%d,%d)", o.row, o.col);
    CHECK(near(o.risk, 540000.0 / 56000.0, 1e-12), "Chi2 risk, got %.9f", o.risk);
    // Two-stage: screen below threshold returns nothing.
    TestOptions strict = {1e9};
    CHECK(FastThenChi2::run(T, N, strict).stat == 0, "two-stage must stop at a strict screen");
}
// CPU scan against brute force, for several chunk sizes (padding, one tile, many tiles).

void test_cpu_matches_brute_force() {
    const MockSource s = make_source(53, 300, 11);
    for (const char* name : kTestNames)
        for (size_t GS : {16, 64}) {
            ScanConfig cfg;
            cfg.test = name;
            cfg.GS = GS;
            const ScanContext ctx = make_context(s, cfg);
            CpuTileScanner cpu(ctx, cfg);
            ScanStats st;
            const std::vector<Hit> got = run_scan(s, ctx, cpu, cfg, &st);
            uint64_t n_wanted_tests = 0;
            const std::vector<Hit> want = brute_force(s, cfg, n_wanted_tests);
            CHECK(st.n_tests == n_wanted_tests, "%s GS=%zu: tests %llu, brute force %llu", name, GS,
                  (unsigned long long)st.n_tests, (unsigned long long)n_wanted_tests);
            CHECK(same_hits(got, want), "%s GS=%zu: CPU hits (%zu) differ from brute force (%zu)",
                  name, GS, got.size(), want.size());
            if (std::string(name) == "Chi2") {
                bool found37 = false;
                for (const Hit& h : got) found37 = found37 || (h.s1 == 3 && h.s2 == 7);
                CHECK(found37, "planted pair (3,7) not found by Chi2 (GS=%zu)", GS);
            }
        }
}

// GPU scan (emulated) against the CPU scan.
void test_gpu_matches_cpu() {
    const MockSource s = make_source(53, 300, 11);
    for (const char* name : kTestNames)
        for (size_t GS : {16, 64}) {
            ScanConfig cfg;
            cfg.test = name;
            cfg.GS = GS;
            const ScanContext ctx = make_context(s, cfg);
            CpuTileScanner cpu(ctx, cfg);
            ScanStats cst;
            const std::vector<Hit> chits = run_scan(s, ctx, cpu, cfg, &cst);
            const int BS = GS == 16 ? 4 : 16;  // BD = GS / BS
            GpuTileScanner gpu(ctx, cfg, BS);
            ScanStats gst;
            const std::vector<Hit> ghits = run_scan(s, ctx, gpu, cfg, &gst);
            CHECK(gst.n_tests == cst.n_tests, "%s GS=%zu: GPU tests %llu, CPU %llu", name, GS,
                  (unsigned long long)gst.n_tests, (unsigned long long)cst.n_tests);
            CHECK(same_hits(ghits, chits), "%s GS=%zu: GPU hits (%zu) differ from CPU (%zu)", name,
                  GS, ghits.size(), chits.size());
        }
}

void test_errors() {
    const MockSource s = make_source(20, 100, 3);
    ScanConfig cfg;
    cfg.test = "NoSuchTest";
    bool threw = false;
    try {
        const ScanContext ctx = make_context(s, cfg);
        CpuTileScanner cpu(ctx, cfg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "unknown test name must throw");
    cfg.test = "Chi2";
    cfg.GS = 16;
    threw = false;
    try {
        ScanConfig tiny = cfg;
        tiny.max_tile_bytes = 10;
        const ScanContext ctx = make_context(s, tiny);
        CpuTileScanner cpu(ctx, tiny);
        run_scan(s, ctx, cpu, tiny);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "tile over the memory budget must throw");
    MockSource bad = make_source(20, 100, 3);
    bad.G[4][10] = 3;  // invalid genotype
    threw = false;
    try {
        const ScanContext ctx = make_context(bad, cfg);
        CpuTileScanner cpu(ctx, cfg);
        run_scan(bad, ctx, cpu, cfg);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "genotype 3 must throw");
    // GPU hit buffer too small: must report, not drop hits silently.
    threw = false;
    try {
        const ScanContext ctx = make_context(s, cfg);
        GpuTileScanner gpu(ctx, cfg, 4, /*capacity=*/1);
        run_scan(s, ctx, gpu, cfg);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw, "GPU overflow must throw");
    // GS not a multiple of BS.
    threw = false;
    try {
        ScanConfig odd = cfg;
        odd.GS = 18;
        const ScanContext ctx = make_context(s, odd);
        GpuTileScanner gpu(ctx, odd, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw, "GS not a multiple of BS must throw");
}

}  // namespace

int main() {
    test_statistics();
    test_cpu_matches_brute_force();
    test_gpu_matches_cpu();
    test_errors();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all epi tests passed\n");
    return 0;
}