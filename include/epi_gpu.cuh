#ifndef EPI_GPU_CUH
#define EPI_GPU_CUH

#include "epi_scanner.h"
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
	#include <cuda_runtime.h>
	#include <cstdio>
#endif

#ifndef EPI_LAUNCH
	#define EPI_LAUNCH(K, G, B, ...) K<<<G, B>>>(__VA_ARGS__)
#endif

#define EPI_CUDA_CHECK(call)                                                             \
    do {                                                                                 \
        cudaError_t e_ = (call);                                                         \
        if (e_ != cudaSuccess)                                                           \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + \
                                     " in " #call);                                      \
    } while (0)

namespace epi {

// GPU tile scanner class declaration
class GpuTileScanner : public TileScanner {
public:
    // BS: threads per block side (BS*BS <= 1024, and cfg.GS must be a multiple of BS).
    // capacity: max hits stored per tile.
    GpuTileScanner( const ScanContext& ctx,
                    const ScanConfig& cfg,
                    int BS = 16,
                    unsigned long long capacity = 1ULL << 22);
    ~GpuTileScanner() override;
    GpuTileScanner(const GpuTileScanner&) = delete;
    GpuTileScanner& operator=(const GpuTileScanner&) = delete;
    std::vector<Hit> scan(const Tile& t, uint64_t& tests) override;
private:
    void release();
    ScanContext ctx_;
    ScanConfig cfg_;
    int BS_;
    unsigned long long cap_;
    int8_t* d_geno_ = nullptr;
    uint8_t* d_valid_ = nullptr;
    real* d_score_ = nullptr;
    uint8_t* d_cc_ = nullptr;
    Hit* d_out_ = nullptr;
    unsigned long long* d_count_ = nullptr;
    unsigned long long* d_tests_ = nullptr;
};

#ifdef __CUDACC__
// Prints the properties of the current device (replaces the old CudaInfo()).
inline void cuda_device_info(std::FILE* out);
#endif

}  // namespace epi

#endif  // EPI_GPU_CUH
