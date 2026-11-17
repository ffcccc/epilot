// One SNP pair ("cell" of the tile). This is the only per-pair code: the CPU loops over
// cells, the GPU runs one cell per thread.
#ifndef EPI_CELL_H
#define EPI_CELL_H
#include "epi_core.h"
#include "epi_stats.h"

namespace epi {

enum CellStatus : int {
    CELL_SKIP = 0,    // not a pair to test (padding, excluded SNP, or lower triangle)
    CELL_TESTED = 1,  // tested, not reported
    CELL_HIT = 2,     // tested and reported
};

// Tests the pair (row r of chunk A, column c of chunk B) of tile v.
template <class Test>
EPI_HD inline int test_cell(const TileView& v, int r, int c, Hit& hit) {
    const int colRow = (v.diagonal ? 0 : v.GS) + c;  // row of chunk-B SNP c in v.geno
    if (v.diagonal && c <= r) return CELL_SKIP;
    if (!v.valid[r] || !v.valid[colRow]) return CELL_SKIP;
    const int8_t* g1 = v.geno + static_cast<size_t>(r) * v.nSamp;
    const int8_t* g2 = v.geno + static_cast<size_t>(colRow) * v.nSamp;
    real CT[2][3][3] = {};
    real NA[2] = {0, 0};  // samples with a missing genotype, per group
    for (int k = 0; k < v.nSamp; ++k) {
        const int a = g1[k], b = g2[k];
        const int grp = v.cc[k];
        if (a < 0 || b < 0) NA[grp] += 1;
        else CT[grp][a][b] += 1;
    }
    real N[3];
    N[AFF] = v.nAff - NA[AFF];
    N[UNAFF] = v.nUnaff - NA[UNAFF];
    N[ALL] = N[AFF] + N[UNAFF];
    const TestOut o = Test::run(CT, N, v.opt);
    if (!(o.stat > v.min_stat)) return CELL_TESTED;
    if (!(o.stat > v.score[r] && o.stat > v.score[colRow])) return CELL_TESTED;
    hit.s1 = v.snpA + static_cast<uint32_t>(r);
    hit.s2 = v.snpB + static_cast<uint32_t>(c);
    hit.stat = o.stat;
    hit.risk = o.risk;
    hit.row = o.row;
    hit.col = o.col;
    return CELL_HIT;
}
}  // namespace epi
#endif  // EPI_CELL_H