// CPU back end: evaluates every cell of a tile on one thread.
// The per-pair work is test_cell() from epi_cell.h, the same function the GPU kernel
// calls, so the two back ends produce the same hits.
#ifndef EPI_CPU_H
#define EPI_CPU_H

#include "epi_cell.h"
#include "epi_scanner.h"
#include "epi_stats.h"

#include <string>
#include <vector>

namespace epi {

class CpuTileScanner : public TileScanner {
public:
    CpuTileScanner(const ScanContext& ctx, const ScanConfig& cfg) : ctx_(ctx), cfg_(cfg) {
        with_test(cfg_.test, [](auto) {});  // rejects unknown test names up front
    }
    
    std::vector<Hit> scan(const Tile& t, uint64_t& tests) override {
        const TileView v = make_view(t, ctx_, cfg_);
        return with_test(cfg_.test, [&](auto test) {
            return scan_view<decltype(test)>(v, tests);
        });
    }

private:
    template <class Test>
    static std::vector<Hit> scan_view(const TileView& tv, uint64_t& tests) {
        std::vector<Hit> out;
        uint64_t n = 0;
        Hit h = {0, 0, 0, 0, -1, -1};
        for (int r = 0; r < tv.GS; ++r)
            for (int c = 0; c < tv.GS; ++c) {
                const int st = test_cell<Test>(tv, r, c, h);
                if (st == CELL_SKIP) continue;
                ++n;
                if (st == CELL_HIT) out.push_back(h);
            }
        tests = n;
        return out;
    }
    ScanContext ctx_;  // copies: the scanner does not depend on the caller's lifetime
    ScanConfig cfg_;
};

}  // namespace epi
#endif  // EPI_CPU_H