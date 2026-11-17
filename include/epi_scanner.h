// Shared loader for the CPU and GPU back ends.
//
// Replaces the old main(): it reads the dataset through SnpSource, splits the SNPs into
// chunks of GS (padded with invalid SNPs so GD = ceil(nSnp/GS) is exact), enumerates the
// GD(GD+1)/2 chunk pairs, and hands each one to a TileScanner. The scanner decides how
// the pairs of a tile are evaluated (CPU loop or GPU kernel); everything else is here.
//
// Wiring your reader (THDF5 is not in this repository, so the adapter is yours):
//
//   struct H5Source : SnpSource { THDF5* hf; ... forwards to hf-> ... };
//   H5Source src{hf};
//   epi::ScanConfig cfg;  cfg.test = "Chi2";  cfg.GS = 1024;
//   epi::ScanContext ctx = epi::make_context(src, cfg);
//   epi::CpuTileScanner cpu(ctx, cfg);            // or epi::GpuTileScanner gpu(ctx, cfg, 16);
//   epi::ScanStats st;
//   std::vector<epi::Hit> hits = epi::run_scan(src, ctx, cpu, cfg, &st);

#ifndef EPI_LOADER_H
#define EPI_LOADER_H

#include "epi_core.h"
#include "epi_cell.h"
#include "epi_stats.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace epi {

// Brute force: every valid pair once, no tiling, no padding, same test and filters.
inline std::vector<Hit> brute_force(const SnpSource& src, const ScanConfig& cfg, uint64_t& tests) {
    std::vector<size_t> kept;
    std::vector<uint8_t> cc;
    real nA = 0, nU = 0;
    for (size_t k = 0; k < src.numSamples(); ++k) {
        const int p = src.phenotype(k);
        if (p != 0 && p != 1) continue;
        kept.push_back(k);
        cc.push_back(static_cast<uint8_t>(p));
        (p == 1 ? nA : nU) += 1;
    }
    const size_t nSnp = src.numSnps(); //.G.size();
    std::vector<bool> ok(nSnp);
    std::vector<std::vector<int8_t>> g(nSnp);
    for (size_t i = 0; i < nSnp; ++i) {
        ok[i] = std::find(cfg.excluded_chromosomes.begin(), cfg.excluded_chromosomes.end(),
                          src.chromosome(i)) == cfg.excluded_chromosomes.end();
        if (!ok[i]) continue;
        g[i].resize(kept.size());
        for (size_t k = 0; k < kept.size(); ++k) g[i][k] = src.genotype(i, kept[k]);
    }
    std::vector<Hit> hits;
    tests = 0;
    for (size_t i = 0; i < nSnp; ++i)
        for (size_t j = i + 1; j < nSnp; ++j) {
            if (!ok[i] || !ok[j]) continue;
            ++tests;
            real CT[2][3][3] = {};
            real NA[2] = {0, 0};
            for (size_t k = 0; k < kept.size(); ++k) {
                const int a = g[i][k], b = g[j][k];
                if (a < 0 || b < 0) NA[cc[k]] += 1;
                else CT[cc[k]][a][b] += 1;
            }
            real N[3];
            N[AFF] = nA - NA[AFF];
            N[UNAFF] = nU - NA[UNAFF];
            N[ALL] = N[AFF] + N[UNAFF];
            const TestOut o = with_test(cfg.test, [&](auto t) {
                return decltype(t)::run(CT, N, cfg.opt);
            });
            if (o.stat > cfg.min_stat && o.stat > static_cast<real>(src.score(i)) &&
                o.stat > static_cast<real>(src.score(j))) {
                hits.push_back(Hit{static_cast<uint32_t>(i), static_cast<uint32_t>(j), o.stat, o.risk, o.row, o.col});
            }
        }
    return hits;
}

// Per-run data shared by all tiles. Built once from the source.
struct ScanContext {
    size_t nSnp = 0;
    int nSamp = 0;                  // samples with known phenotype
    std::vector<size_t> kept;       // source index of each kept sample
    std::vector<uint8_t> cc;        // UNAFF/AFF per kept sample
    real nAff = 0, nUnaff = 0;
    std::vector<uint8_t> snp_ok;    // 0 for SNPs on excluded chromosomes
    std::vector<real> snp_score;    // univariate score per SNP
};

// One tile: chunk A (rows [0,GS)) and chunk B (rows [GS,2GS)); see TileView.
struct Tile {
    int GS = 0;
    uint32_t snpA = 0, snpB = 0;
    bool diagonal = false;
    std::vector<int8_t> geno;
    std::vector<real> score;
    std::vector<uint8_t> valid;
};

// Back end interface. Scans every pair of one tile and returns the reported hits
// (global indices); *tests receives the number of pairs actually tested.
class TileScanner {
public:
    virtual ~TileScanner() {}
    virtual std::vector<Hit> scan(const Tile& t, uint64_t& tests) = 0;
};

struct ScanStats {
    uint64_t n_tiles = 0;
    uint64_t n_tests = 0;
    uint64_t n_hits = 0;
};

inline ScanContext make_context(const SnpSource& src, const ScanConfig& cfg) {
    ScanContext ctx;
    ctx.nSnp = src.numSnps();
    const size_t nSamp = src.numSamples();
    for (size_t s = 0; s < nSamp; ++s) {
        const int phe = src.phenotype(s);
        if (phe != 0 && phe != 1) continue;  // unknown phenotype: sample is not used
        ctx.kept.push_back(s);
        ctx.cc.push_back(static_cast<uint8_t>(phe));
        if (phe == AFF) ctx.nAff += 1;
        else ctx.nUnaff += 1;
    }
    ctx.nSamp = static_cast<int>(ctx.kept.size());
    ctx.snp_ok.resize(ctx.nSnp);
    ctx.snp_score.resize(ctx.nSnp);
    for (size_t i = 0; i < ctx.nSnp; ++i) {
        const std::string chr = src.chromosome(i);
        const bool excluded = std::find(cfg.excluded_chromosomes.begin(), cfg.excluded_chromosomes.end(), chr) != cfg.excluded_chromosomes.end();
        ctx.snp_ok[i] = excluded ? 0 : 1;
        ctx.snp_score[i] = static_cast<real>(src.score(i));
    }
    return ctx;
}

// Fills chunk `half` (0 = A, 1 = B) of the tile with SNPs first..first+GS-1.
// Padding and excluded SNPs get valid = 0 and all-missing genotypes.
inline void fill_chunk( const SnpSource& src,
                        const ScanContext& ctx,
                        Tile& t, int half,
                        size_t first,
                        std::vector<int8_t>& raw) {
    const size_t GS = static_cast<size_t>(t.GS);
    const size_t nSamp = static_cast<size_t>(ctx.nSamp);
    for (size_t r = 0; r < GS; ++r) {
        const size_t row = static_cast<size_t>(half) * GS + r;
        const size_t snp = first + r;
        int8_t* dst = &t.geno[row * nSamp];
        if (snp >= ctx.nSnp || !ctx.snp_ok[snp]) {
            std::fill(dst, dst + nSamp, static_cast<int8_t>(-1));
            t.valid[row] = 0;
            t.score[row] = 0;
            continue;
        }
        src.genotypes(snp, raw.data());
        for (size_t k = 0; k < nSamp; ++k) {
            const int v = raw[ctx.kept[k]];
            if (v < -1 || v > 2) throw std::runtime_error("genotype outside {-1,0,1,2}");
            dst[k] = static_cast<int8_t>(v);
        }
        t.valid[row] = 1;
        t.score[row] = ctx.snp_score[snp];
    }
}

inline TileView make_view(const Tile& t, const ScanContext& ctx, const ScanConfig& cfg) {
    TileView tv;
    tv.geno = t.geno.data();
    tv.valid = t.valid.data();
    tv.score = t.score.data();
    tv.cc = ctx.cc.data();
    tv.nSamp = ctx.nSamp;
    tv.GS = t.GS;
    tv.diagonal = t.diagonal;
    tv.nAff = ctx.nAff;
    tv.nUnaff = ctx.nUnaff;
    tv.min_stat = cfg.min_stat;
    tv.opt = cfg.opt;
    tv.snpA = t.snpA;
    tv.snpB = t.snpB;
    return tv;
}

// Runs the whole scan. Hits are returned sorted by (s1, s2).
inline std::vector<Hit> run_scan(const SnpSource& src, const ScanContext& ctx,
                                 TileScanner& scanner, const ScanConfig& cfg,
                                 ScanStats* stats = nullptr) {
    if (cfg.GS < 1) throw std::invalid_argument("GS must be >= 1");
    const size_t GS = cfg.GS;
    const size_t GD = (ctx.nSnp + GS - 1) / GS;
    const size_t nSamp = static_cast<size_t>(ctx.nSamp);
    const size_t bytes = 2 * GS * nSamp * sizeof(int8_t) + 2 * GS * (sizeof(real) + 1);
    if (bytes > cfg.max_tile_bytes)
        throw std::runtime_error("one tile does not fit max_tile_bytes; lower GS");
    Tile t;
    t.GS = static_cast<int>(GS);
    t.geno.assign(2 * GS * nSamp, -1);
    t.score.assign(2 * GS, 0);
    t.valid.assign(2 * GS, 0);
    std::vector<int8_t> raw(src.numSamples());
    ScanStats st;
    std::vector<Hit> hits;
    for (size_t a = 0; a < GD; ++a) {
        t.snpA = static_cast<uint32_t>(a * GS);
        fill_chunk(src, ctx, t, 0, t.snpA, raw);  // chunk A is reused for every b >= a
        for (size_t b = a; b < GD; ++b) {
            t.diagonal = (a == b);
            t.snpB = static_cast<uint32_t>(b * GS);
            if (!t.diagonal) fill_chunk(src, ctx, t, 1, t.snpB, raw);
            uint64_t tests = 0;
            const std::vector<Hit> h = scanner.scan(t, tests);
            hits.insert(hits.end(), h.begin(), h.end());
            st.n_tests += tests;
            st.n_tiles += 1;
        }
    }
    std::sort(hits.begin(), hits.end(), [](const Hit& x, const Hit& y) {
        return x.s1 != y.s1 ? x.s1 < y.s1 : x.s2 < y.s2;
    });
    st.n_hits = hits.size();
    if (stats) *stats = st;
    return hits;
}

}  // namespace epi
#endif  // EPI_LOADER_H