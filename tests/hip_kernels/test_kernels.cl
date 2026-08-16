// Kernel 1: Compute-bound — repeated MAD on registers
__kernel void compute_bound(__global float* out, __global const float* in, int n, int iters) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    float val = in[idx];
    for (int i = 0; i < iters; i++) {
        val = val * 1.0001f + 0.0001f;
    }
    out[idx] = val;
}

// Kernel 2: Memory-bound — indirect/strided access
__kernel void memory_bound(__global float* out, __global const int* indices, __global const float* data, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    int src = indices[idx];
    out[idx] = data[src];
}

// Kernel 3: Mixed — SAXPY + compute
__kernel void mixed_kernel(__global float* out, __global const float* a, __global const float* b, float alpha, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    float va = a[idx];
    float vb = b[idx];
    float result = alpha * va + vb;
    for (int i = 0; i < 4; i++) {
        result = result * result + vb;
    }
    out[idx] = result;
}
