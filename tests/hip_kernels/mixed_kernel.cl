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
