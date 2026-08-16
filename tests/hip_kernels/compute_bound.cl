__kernel void compute_bound(__global float* out, __global const float* in, int n, int iters) {
    int idx = get_global_id(0);
    if (idx >= n) return;
    float val = in[idx];
    for (int i = 0; i < iters; i++) {
        val = val * 1.0001f + 0.0001f;
    }
    out[idx] = val;
}
