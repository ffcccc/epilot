// Minimal CUDA stand-in for compiling and testing epi.cu without nvcc or a GPU.
// Kernels run sequentially: every (block, thread) pair is executed in turn with the
// blockIdx / threadIdx / blockDim globals set. This checks indexing, tile coverage and
// the host logic. It does not check races, shared memory or timing.
// Include this header before epi.cu.
#ifndef EPI_CUDA_SHIM_H
#define EPI_CUDA_SHIM_H
#include <cstdint>
#include <cstdlib>
#include <cstring>
#define __global__
#define __device__
#define __host__

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};

inline dim3 blockIdx, threadIdx, blockDim, gridDim;
template <class K, class... A>
void epi_emulate(K kernel, dim3 grid, dim3 block, A... args) {
    blockDim = block;
    gridDim = grid;
    for (unsigned by = 0; by < grid.y; ++by)
        for (unsigned bx = 0; bx < grid.x; ++bx)
            for (unsigned ty = 0; ty < block.y; ++ty)
                for (unsigned tx = 0; tx < block.x; ++tx) {
                    blockIdx = dim3(bx, by, 0);
                    threadIdx = dim3(tx, ty, 0);
                    kernel(args...);
                }
}

#define EPI_LAUNCH(K, G, B, ...) epi_emulate(K, G, B, __VA_ARGS__)

typedef int cudaError_t;
enum { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };

inline cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::malloc(n ? n : 1);
    return cudaSuccess;
}

inline cudaError_t cudaFree(void* p) {
    std::free(p);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    std::memcpy(dst, src, n);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* p, int v, size_t n) {
    std::memset(p, v, n);
    return cudaSuccess;
}

inline cudaError_t cudaGetLastError() { return cudaSuccess; }

inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

inline const char* cudaGetErrorString(cudaError_t) { return "emulated CUDA error"; }

inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) {
    const unsigned long long old = *p;
    *p += v;
    return old;
}
#endif  // EPI_CUDA_SHIM_H