# bf16_matmul M>1 Bug 根因分析

## 现象

GEMM 对 M>1 的输入只有第 0 行有值，其余行全零：

```
X[3,4] @ W[4,3] → Y[3,3]
GPU: [1.40, 1.60, 1.80,  0, 0, 0,  0, 0, 0]
CPU: [1.40, 1.60, 1.80,  3.16, 3.68, 4.20,  4.92, 5.76, 6.60]
```

M=1 时正确，M=3 时失败。

## 根因

`build_kernargs` 写入 40 字节，但内核从 offset 40 读取 M。M 从未被写入。

### 内核期望的 kernarg 布局（tile_ir.rs:661-668）

| offset | 字段 | 类型 | 字节 |
|--------|------|------|------|
| 0 | X | u64 | 8 |
| 8 | WT | u64 | 8 |
| 16 | Y | u64 | 8 |
| 24 | K | u32 | 4 |
| 28 | N | u32 | 4 |
| 32 | split_k_shift | u32 | 4 |
| 36 | y_split_stride | u32 | 4 |
| **40** | **M** | **u32** | **4** |

总计 44 字节。

### build_kernargs 实际写入（gemm_gen.rs:414-430）

```rust
let mut ka = [0u8; 40];  // 只有 40 字节
// ... 写入 offset 0-39
// offset 40: 未写入，越界访问
```

### M 在内核中的 4 处使用

1. **Early exit**（tile_ir.rs:731）: `if tile_base_m >= M → 跳过整个 tile`
2. **Boundary 标记**（tile_ir.rs:746）: `if M < tile_end_m → 标记为边界 tile`
3. **X 加载 clamp**（tile_ir.rs:940）: `clamp(x_abs_row, 0, M-1)`
4. **存储 mask**（tile_ir.rs:2599）: `EXEC = (cur_row < M) && (col < N)`

当 M 为残留值 1 时，只有 cur_row=0 通过第 4 处检查。

## 修复

### 修复 1: `build_kernargs` 缺少 M 字段

`gemm_gen.rs`: `build_kernargs` 返回 40→44 字节，在 offset 40 写入 M。
同步修复 `build_kernargs_with_bias`（48→52 字节，M 在 40，bias 在 44）。
同步修复 `build_kernargs_backward_data`、`build_kernargs_backward_weight` 返回类型。
同步修复 `tile_ssa_lower.rs` 的 `build_kernargs` wrapper 返回类型。

### 修复 2: `f32_to_bf16_gpu_padded` 线性 padding 导致行数据错位

**现象**: 修复 M 问题后，M=3 的 GEMM 仍然只有 row 0 正确。

**根因**: `f32_to_bf16_gpu_padded` 将 f32 数据线性转换为 bf16 并在末尾补零。
但 GEMM 内核期望行主序 [rows_padded, cols_padded] 布局，每行独立 padding。

```
X[3,4] → bf16, pad 到 [16, 16]:
期望: row0=[X00..X03, 0×12], row1=[X10..X13, 0×12], row2=[X20..X23, 0×12]
实际: [X00,X01,X02,X03, X10,X11,X12,X13, X20,X21,X22,X23, 0×244]
       ↑ row 0 读到混杂数据    ↑ row 1 读到全是零
```

**修复**: 改函数签名为 `(rows, cols, rows_padded, cols_padded)`，按行 padding。

## 测试结果

- `test_gemm_m1_works`: ✅ M=1 GEMM 正确
- `test_gemm_m3_debug`: ✅ M=3 GEMM 所有行正确（bf16 精度内）
- `test_gemm_m3_raw_output`: ✅ 验证 raw padded 输出行数据正确
- 4 个 GPU attention 测试: ✅ 全部通过
- tile_ir 测试: 34 passed, 2 failed（pre-existing，非本次修改引入）
