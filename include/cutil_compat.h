#pragma once

// Minimal replacements for the legacy CUDA SDK cutil helpers (removed after CUDA 4.x).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define cutilSafeCall(call)                                                              \
	do {                                                                                 \
		cudaError_t err_ = (call);                                                       \
		if (err_ != cudaSuccess) {                                                       \
			fprintf(stderr, "%s(%d): CUDA error: %s\n", __FILE__, __LINE__,              \
			        cudaGetErrorString(err_));                                           \
			exit(EXIT_FAILURE);                                                          \
		}                                                                                \
	} while (0)

#define cutilCheckMsg(msg)                                                               \
	do {                                                                                 \
		cudaError_t err_ = cudaGetLastError();                                           \
		if (err_ != cudaSuccess) {                                                       \
			fprintf(stderr, "%s(%d): %s: %s\n", __FILE__, __LINE__, (msg),               \
			        cudaGetErrorString(err_));                                           \
			exit(EXIT_FAILURE);                                                          \
		}                                                                                \
	} while (0)

inline int cutGetMaxGflopsDeviceId() {
	int count = 0, best = 0;
	long long bestPerf = -1;
	cutilSafeCall(cudaGetDeviceCount(&count));
	for (int dev = 0; dev < count; ++dev) {
		int sms = 0, clock = 0;
		cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
		cudaDeviceGetAttribute(&clock, cudaDevAttrClockRate, dev);
		const long long perf = static_cast<long long>(sms) * clock;
		if (perf > bestPerf) {
			bestPerf = perf;
			best = dev;
		}
	}
	return best;
}
