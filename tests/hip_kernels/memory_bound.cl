__kernel void memory_bound(__global float* out, __global const int* indices, __global const float* data, int n) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    int src = indices[idx];
    out[idx] = data[src];
}
