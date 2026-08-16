# Flash Attention Decode 修复 — 根因分析与修复

## 日期
2026-07-06

## 目标
修复 `flash_attn_decode` kernel 的两个问题：推理时 GPU 硬 hang 和数值完全错误（max_rel=100%）。

## 症状
- 推理 decode 阶段 GPU 硬挂（`wait_read_ptr TIMEOUT 5s`）
- 单元测试 `test_gpu_flash_attn_decode_qwen_scale` 失败：`max_err=0.38, max_rel=100%`
- 修复前使用 `standard_attention` 兜底，decode 仅 1.9 tok/s

## 根因分析

### Bug 1: Grid Size 错误（`flash_attn_kernels.rs:89`）

```rust
// 错误: 启动了 n_heads * head_dim = 2048 个 workgroup
pub fn flash_attn_decode_grid(n_heads: u32, head_dim: u32) -> (u32, u32) {
    (n_heads * head_dim, 1)
}
```

Kernel 设计中每个 workgroup 处理 1 个 head（pid = head index），只需要 `n_heads = 16` 个 workgroup。旧的 `n_heads * head_dim` 导致额外 2032 个 workgroup 做 OOB 内存访问：pid 16~2047 读取 Q/K/V 越界数据，写入 output 越界地址。

**为什么有时能跑**: OOB 写踩到的 VRAM 区域随机——踩到未映射页则硬 hang，踩到已分配但无关内存则侥幸通过，踩到 KV cache 则输出错误。

### Bug 2: 缺少 Workgroup 边界检查

Kernel 未检查 `pid < n_heads`，无法防御多余的 workgroup。需要添加 `n_heads` kernarg 并用 mask 跳过无效 workgroup。

### Bug 3: T0 编译器标量比较 Bug（`tile_ssa_lower.rs:946`）

尝试将 grid 修正为 `(n_heads, 1) = (16, 1)` 时，kernel 输出完全错误（max_err=0.38）。经 ISA 级调试定位到 T0 编译器的标量比较 lowering bug：

```rust
// tile_ssa_lower.rs:945-947
// 标量比较结果存为 SReg（dummy），实际用 SCC
let dummy = k.alloc_sreg();
val_map.insert(*result, MachineVal::SReg(dummy));
```

当两个 SGPR 做 `cmp_lt` 时（如 `pid < n_heads`），编译器生成 `s_cmp_lt_u32` 设置 SCC，但结果映射到一个**从未被写入的 dummy SGPR**。后续 `and_bool` 读到该 dummy SGPR 中的垃圾值，导致 mask 随机为 0 或 1。

此 bug 在旧 kernel 中未被触发，因为原来没有 SGPR 间的 `lt` 操作——pid 只用于算术运算（乘、移位），不产生布尔结果。

## 修复方案

### 修复 1: 保留大 Grid + 添加边界检查

直接改小 grid 会触发 Bug 3，因此采用"保留旧 grid + 加边界检查"策略：

```rust
// flash_attn_kernels.rs
let n_heads = kb.arg_u32("n_heads");  // 新增 kernarg
let wg_mask = pid_v.lt(&mut kb, n_heads);  // pid < n_heads
let mask = d_mask.and_bool(&mut kb, wg_mask);  // 组合 mask
```

所有 `load`/`store` 操作使用 `mask`（而非旧的 `d_mask`），无效 workgroup 的内存操作被硬件跳过。

### 修复 2: Workaround T0 编译器 Bug

强制 pid 进入 VGPR 以走向量比较路径（VCC → v_cndmask，正确物化布尔值）：

```rust
// pid 是 SGPR (program_id)。(pid + tid - tid) 强制变 VGPR。
// tid 由 arange 产生，已在 VGPR 中。
let pid_v = pid.add(&mut kb, tid).sub(&mut kb, tid);
let wg_mask = pid_v.lt(&mut kb, n_heads);  // VGPR < SGPR → 正确路径
```

### 修复 3: Kernarg 同步更新

`flash_attention_decode()` 中 kernargs 增加 `n_heads => u32`，kernarg size 从 48 → 52 字节。

## 验证结果

| 测试 | 结果 |
|---|---|
| `test_flash_attn_decode_compiles` | `3424 bytes ELF, wg=128, lds=536` ✅ |
| `test_gpu_flash_attn_decode_small` | `max_err=0.000000` ✅ |
| `test_gpu_flash_attn_decode_qwen_scale` | `max_err=0.000000` ✅ |
| 推理 "Hello, who are you?" × 32 tokens (T=0.7) | 无 hang, 语义正确 ✅ |
| 推理 "The capital of France is" × 50 tokens (T=0.0) | "Paris. The capital of Italy..." ✅ |

**性能**: Decode 13.0 tok/s（修复前 1.9 tok/s，提升 6.8×）。

## 涉及文件

| 文件 | 操作 | 说明 |
|---|---|---|
| `src/t0/flash_attn_kernels.rs` | 修改 | 添加 n_heads 参数、边界 mask、pid→VGPR workaround |
| `src/ignis/ops/attention.rs` | 修改 | kernargs 增加 n_heads |
| `src/ignis/nn/transformer.rs` | 修改 | 恢复 flash_attention_decode 用于 decode |

## 遗留问题

- [ ] **T0 编译器 scalar cmp bug**（`tile_ssa_lower.rs:946`）: scalar `cmp_lt` 应生成 `s_cselect_b32` 或等效指令将 SCC 物化到目标 SGPR。当前 workaround 依赖 `(x + tid - tid)` 强制 VGPR，有微小性能代价（每个 workgroup 多 2 条 VALU 指令）。

## 相关报告
- [[tile_ir_GPU_Hang_RootCause]] — 地址偏移 double-count 根因
- [[split_k_GPU_Hang_根因分析]] — split-K buffer 不足根因
- [[循环归纳变量GPU硬挂_根因分析与修复]] — get_vreg 缓存覆写 SReg 根因
- [[k48_hang根因_coopload_bug]] — chunks_per_row 非 2 次幂根因
