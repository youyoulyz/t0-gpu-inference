#include <hip/hip_runtime.h>

// Kernel 1: Compute-bound — repeated MAD on registers
__global__ void compute_bound(float* out, const float* in, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = in[idx];
    for (int i = 0; i < iters; i++) {
        val = val * 1.0001f + 0.0001f;
    }
    out[idx] = val;
}

// Kernel 2: Memory-bound — indirect/strided access
__global__ void memory_bound(float* out, const int* indices, const float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int src = indices[idx];
    out[idx] = data[src];
}

// Kernel 3: Mixed — SAXPY + compute
__global__ void mixed_kernel(float* out, const float* a, const float* b, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float va = a[idx];
    float vb = b[idx];
    float result = alpha * va + vb;
    for (int i = 0; i < 4; i++) {
        result = result * result + vb;
    }
    out[idx] = result;
}
