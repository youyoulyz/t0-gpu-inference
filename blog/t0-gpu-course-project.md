---
title: 从零造一个 GPU 编译器：T0-GPU 作为课程项目的设计与实践
date: 2026-05-30 10:00:00
tags:
  - GPU
  - compiler
  - AMD RDNA3
  - Rust
  - inference
categories:
  - 系统编程
  - GPU 编译器
---

## 一句话

T0-GPU 是一个用纯 Rust 从零构建的 GPU 编译器和推理引擎，直接运行在 AMD RDNA3 硬件上，不依赖 HIP/ROCm/LLVM。58,000 行代码，从 ISA 编码到 LLM 推理，全栈可读可改。

<!-- more -->

## 为什么需要这样的项目

GPU 编程教学长期面临一个困境：

- **CUDA/Triton** 太高层 — 学生会调参，但不知道 kernel 最终变成了什么指令
- **手写汇编** 太底层 — 一个学期只够写一个矩阵乘法
- **编译器课程**太抽象 — SSA、寄存器分配都是纸上谈兵，没有真实硬件验证

T0-GPU 的定位是**中间层**：它是一个完整的 GPU 编译器，但每一层都足够薄，学生可以在一个学期内从头到尾读完、改完。

## 项目架构

```
┌─────────────────────────────────────────────────┐
│  Ignis (ML 框架层, ~17K LOC)                    │
│  Tensor, Autograd, NN modules, Qwen3 推理       │
├─────────────────────────────────────────────────┤
│  T0 Compiler (~38K LOC, 39 files)               │
│  ┌──────────────┐  ┌──────────┐  ┌───────────┐ │
│  │ BlockDSL /   │  │ 6-Pass   │  │ Register  │ │
│  │ TileSSA IR   │→ │ Optimizer│→ │ Allocator │ │
│  └──────────────┘  └──────────┘  └───────────┘ │
│                      ↓                          │
│  ┌──────────────────────────────────────────────┐│
│  │ AsmEmitter → RDNA3 汇编文本                  ││
│  │ rdna3_asm.rs → RDNA3 二进制编码 (3,114 行)   ││
│  └──────────────────────────────────────────────┘│
├─────────────────────────────────────────────────┤
│  KFD Runtime (~3,093 行)                         │
│  ioctl(/dev/kfd), VRAM mmap, AQL queue dispatch │
├─────────────────────────────────────────────────┤
│  AMD RDNA3 GPU (RX 7900 XTX, GFX1100)           │
└─────────────────────────────────────────────────┘
```

每一层都可以独立教学：

| 层 | 文件 | 行数 | 教学内容 |
|----|------|------|---------|
| ISA 编码 | `rdna3_asm.rs` | 3,114 | RDNA3 指令集、VOP/SOP/SMEM 编码、WMMA lane layout |
| ELF 生成 | `rdna3_code_object.rs` | 1,444 | AMD HSA ELF 格式、code object、kernarg layout |
| GPU 运行时 | `kfd/mod.rs` | 3,093 | KFD ioctl、VRAM 分配、AQL dispatch packet、doorbell |
| 编译器 IR | `tile_ssa.rs` | 2,669 | Tile 级 SSA IR、phi 节点、shape 推导 |
| 优化 pass | `opt_passes.rs` | ~2,000 | 死代码消除、常量折叠、指令调度、LICM |
| 寄存器分配 | `ssa_regalloc.rs` | ~800 | SSA-based 线性扫描、SGPR/VGPR 分离 |
| 汇编发射 | `asm_emitter.rs` | 1,085 | 虚拟寄存器→物理寄存器映射、waitcnt 插入 |
| GEMM 生成器 | `gemm_gen.rs` | 1,382 | 参数化 tiled GEMM、LDS 协作加载、double buffer |

## 学生能做什么

### 实验 1：手写一条 GPU 指令

在 `rdna3_asm.rs` 里，每条 RDNA3 指令都有对应的编码函数：

```rust
// v_add_f32 v0, v1, v2 的二进制编码
pub fn v_add_f32(vdst: u8, vsrc0: u8, vsrc1: u8) -> u32 {
    0x00000000 | ((vsrc1 as u32) << 9) | ((vsrc0 as u32) << 0) | ((vdst as u32) << 17)
}
```

学生可以：
- 写一个最小 kernel（两个向量相加），生成二进制，提交到 GPU 执行
- 用 `isa_probe` 工具验证生成的汇编
- 对比 LLVM 的编码输出，理解 VOP1/VOP2/VOP3 编码格式

### 实验 2：理解 WMMA 指令

RDNA3 的 `v_wmma_f32_16x16x16_bf16` 一条指令完成 16×16×16 的矩阵乘累加：

```
// 32 个 lane 协作完成一个 16×16 的输出 tile
// 每个 lane 持有 8 个 f32 累加器
// 输入: A[16×16] bf16, B[16×16] bf16, C[16×16] f32
// 输出: D[16×16] f32 = A @ B + C
```

学生可以在 `gemm_gen.rs` 的基础上：
- 修改 tile 大小（64×64, 128×64, 128×128），观察性能变化
- 开关 LDS double buffer，量化访存重叠的收益
- 修改 K 维 unroll factor，找到计算/访存平衡点

### 实验 3：写一个 Softmax 内核

`softmax_large.rs` 只有 197 行，实现了一个三遍（max → sum → normalize）的分块 softmax：

```rust
// Pass 1: 每个线程找自己的 max，然后 wg_reduce_max → global_max
// Pass 2: 每个线程算 exp(x - global_max) 的和，然后 wg_reduce_add → global_sum
// Pass 3: 每个线程写 output = exp(x - global_max) / global_sum
```

学生可以：
- 理解 online softmax 算法（避免数值溢出）
- 理解 workgroup 级 reduction（wave reduce → LDS broadcast）
- 修改 chunk 大小，分析对不同 vocab size 的性能影响

### 实验 4：优化 GEMM 到接近硬件极限

项目自带自动调优器（`auto_gemm.rs`），会搜索最优配置。当前 benchmark：

| 矩阵大小 | T0-GPU | rocBLAS | 比率 |
|----------|--------|---------|------|
| 1024³ | 34.5 TFLOPS | 27.9 TFLOPS | **124%** |
| 2048³ | 83.2 TFLOPS | 71.2 TFLOPS | **117%** |
| 4096³ | 105.2 TFLOPS | — | 63.8% 峰值 |

学生可以挑战：在特定矩阵形状下，手写配置能否超过自动调优的结果？

## 为什么是 AMD 而不是 NVIDIA

这个项目**只能存在于 AMD 生态**。原因是 AMD 的每一层都是开放的：

| | AMD (RDNA3) | NVIDIA |
|--|------------|--------|
| ISA 文档 | 公开 (RDNA3 ISA Guide) | SASS 是黑盒 |
| 内核驱动 | KFD 开源 (Linux 内核) | nvidia.ko 闭源 |
| 运行时 | 可绕过 ROCm，直接 ioctl | 必须走 CUDA driver API |
| 寄存器接口 | 公开 | 需要逆向 |

T0-GPU 直接通过 `/dev/kfd` 的 ioctl 创建 GPU 队列、分配 VRAM、提交 dispatch packet。整个运行时 3,093 行 Rust 代码，零外部依赖。这在 NVIDIA 平台上做不到。

## 课程设计建议

### 方案 A：GEMM 专精（16 周，本科高年级）

只做一件事：从零写一个高性能 GEMM kernel。

| 周 | 内容 | 产出 |
|----|------|------|
| 1-2 | GPU 架构基础，RDNA3 ISA | 手写向量加法汇编 |
| 3-4 | WMMA 指令，lane layout | 16×16 矩阵乘 |
| 5-6 | Tiling，LDS 协作加载 | 任意大小矩阵乘 |
| 7-8 | Double buffer，bank conflict 优化 | 带宽利用率提升 |
| 9-10 | 性能分析，roofline model | 找到瓶颈 |
| 11-12 | Epilogue fusion（bias + ReLU） | GEMM + 激活融合 |
| 13-14 | 优化挑战赛 | 性能排名 |
| 15-16 | 接入 PyTorch 自定义扩展 | 端到端验证 |

### 方案 B：推理引擎全栈（16 周，研究生）

从 softmax 到完整 transformer 推理。

| 周 | 内容 | 产出 |
|----|------|------|
| 1-3 | T0 编译器架构，TileSSA IR | 能读懂编译流水线 |
| 4-5 | Softmax kernel（三遍分块） | 通过 GPU 正确性测试 |
| 6-8 | GEMM kernel（WMMA + LDS） | 达到 60%+ 峰值利用率 |
| 9-10 | Flash Attention 思想（分块 attention） | fused attention kernel |
| 11-12 | Weight INT8 量化 | 权重带宽 2x |
| 13-14 | KV Cache + Prefill/Decode 分离 | 自回归生成 |
| 15-16 | 集成 Qwen3 推理，benchmark | 跑通 LLM 生成文本 |

### 方案 C：编译器优化专题（16 周，研究生）

深入 T0 编译器的优化 pass。

| 周 | 内容 |
|----|------|
| 1-4 | SSA IR 构建、phi 节点、支配树 |
| 5-8 | 寄存器分配（线性扫描 vs 图着色） |
| 9-12 | 指令调度（waitcnt、VLIW packing） |
| 13-16 | 对比 T0 和 LLVM 在同一 kernel 上的代码质量 |

## 技术亮点

**零依赖 GPU 运行时**。整个项目唯一的外部依赖是 `tokenizers`（HuggingFace 的 BPE tokenizer）。从 ISA 编码到 ELF 生成到 GPU dispatch，全部自己实现。

**GEMM 性能超过 rocBLAS**。在多个矩阵形状下，T0-GPU 的自动生成 GEMM kernel 比 AMD 的 rocBLAS 快 6-42%。这是通过参数化 tile 配置 + 自动调优实现的。

**完整的 LLM 推理管线**。从 safetensors 权重加载到 BPE tokenization 到 KV cache 管理到自回归生成，Qwen3-0.6B/4B 可以跑通。

**GPU 正确性测试**。39 种 GEMM 配置全部通过精度验证（max error ~1e-5）。softmax、attention、RMSNorm 等 kernel 都有 GPU 端到端测试。

## 快速开始

```bash
git clone <repo-url>
cd t0-gpu

# 编译测试（不需要 GPU）
cargo test --lib -- softmax_large

# GPU 测试（需要 AMD RDNA3 GPU + KFD 驱动）
cargo test --features rocm --lib -- test_gpu_softmax_large

# 运行 ISA 探测工具
cargo run --bin isa_probe
```

## 相关资源

- [RDNA3 ISA Reference Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023.pdf)
- [AMD KFD Documentation](https://docs.kernel.org/gpu/amdgpu/amdkfd.html)
- [Triton](https://github.com/triton-lang/triton) — 对比学习：同样的算法，Triton 如何编译
- [rocBLAS](https://github.com/ROCmSoftwarePlatform/rocBLAS) — 对比学习：AMD 官方 GEMM 库

## 许可证

MIT / Apache-2.0 双许可。
