# GPU Attention 实现工作记录

## 日期: 2026-05-29

## 目标

将 `standard_attention` 从纯 CPU 实现迁移到 GPU，使用现有内核（bf16_matmul、softmax_large）。

## 实现结果

### ✅ 已完成

1. **GPU Softmax** — 成功集成 `softmax_large` 内核到 attention 流程
   - 首次编译: 21ms
   - 后续调用: 0.1ms/head（内核缓存）
   - 支持任意 kv_len（标准 softmax 限 256，large 版无限制）

2. **4 个 GPU Attention 测试** — 全部通过
   - `test_gpu_attention_decode_small`: seq=1, head_dim=4, kv_len=3
   - `test_gpu_attention_decode_multi_head`: seq=1, GQA (2 query, 1 kv)
   - `test_gpu_attention_prefill`: seq=3, causal mask
   - `test_gpu_attention_larger`: head_dim=32, kv_len=16, 4 heads

3. **性能数据**（decode, kv_len=1）:
   - 每层 attention: ~1.6ms (16 heads × 0.1ms softmax)
   - 28 层总计: ~45ms/token
   - 占总推理时间 ~2%

### ❌ 未完成：Q@K^T 和 weights@V 的 GPU GEMM

**尝试**: 用 `bf16_matmul` 实现 `scores = Q @ K^T` 和 `out = weights @ V`

**失败原因**: `bf16_matmul` 对 M > 1 的输入有 bug

**复现测试** (`test_gemm_m3_debug`):
```
输入: X[3,4] @ W[4,3] → 期望 Y[3,3]
GPU 输出: [1.40, 1.60, 1.80,  0, 0, 0,  0, 0, 0]
                                    ↑ 第 1、2 行全零
CPU 参考: [1.40, 1.60, 1.80,  3.16, 3.68, 4.20,  4.92, 5.76, 6.60]
```

M=1 时 GEMM 正确，M=3 时只有第 0 行有值。

**根因分析**:

GEMM 流程:
1. `f32_to_bf16_gpu_padded`: X[3,4] → pad 到 [16, k_pad] bf16
2. `f32_to_bf16_transpose_gpu_padded`: W[4,3] → transpose → [3,4] → pad 到 [n_pad, k_pad] bf16
3. GEMM 内核: 计算 [16, k_pad] @ [k_pad, n_pad] → [16, n_pad]
4. `unpad_f32`: 提取 [3, 3]

问题在步骤 3-4 之间。GEMM 内核通过 kernarg 接收 K 和 N，但不接收 M。M 仅用于 grid 计算。内核可能使用 tile_m 作为行步长，导致只有第一个 tile 的行被正确写入。

`build_kernargs` 布局:
```
[0..8]   X addr
[8..16]  WT addr
[16..24] Y addr
[24..28] K (u32)
[28..32] N (u32)
[32..36] split_k_shift (硬编码 0)
[36..40] y_split_stride
```

M 没有传递给内核。内核通过 `n * tile_m` 计算行偏移，但这个公式在 M < tile_m 时可能不正确。

**当前 workaround**: Q@K^T 和 weights@V 仍在 CPU 上计算。CPU 计算对于小矩阵（head_dim=128, kv_len<200）足够快（<0.1ms/head）。

### 涉及文件

| 文件 | 修改 |
|------|------|
| `src/ignis/ops/attention.rs` | 重写：GPU softmax + CPU scores + CPU weights@V + 4 个测试 |
| `src/ignis/ops/bf16_matmul.rs` | 添加 `precompute_wt_bf16`、M=3 debug 测试（测试 FAIL） |

### 后续工作

1. **修复 bf16_matmul M>1 bug**: 需要调试 `dispatch_gemm_forward`，确认 M 是否需要传递给内核，或 grid 计算逻辑是否正确
2. **Fused attention 内核**: 长期方案，一次 dispatch 完成 Q@K^T + mask + softmax + weights@V
3. **GPU head de-interleave**: 避免从 GPU 读回 Q/K/V 到 CPU 做 head 切片
