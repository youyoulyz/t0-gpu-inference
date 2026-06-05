# 在裸金属 GPU 上调试大模型推理：从乱码到正确输出的漫长旅程

> 本文记录了在 AMD RX 7900 XTX 上通过直接 KFD ioctl（绕过 HIP/ROCm）运行 Qwen3-0.6B 推理引擎的调试过程。从模型输出乱码到最终产生正确文本，中间经历了多个隐蔽 bug 的定位与修复。

## 背景

这是一个极具挑战性的项目：在 AMD GPU 上，不依赖任何主流计算框架（无 ROCm、无 HIP、无 PyTorch），直接通过 `/dev/kfd` 的 ioctl 系统调用提交 GPU 计算任务。整个推理引擎从零构建，包括：

- **BlockDSL → SSA → 寄存器分配 → 机器码** 的 GPU kernel 编译管线
- BF16 WMMA GEMM 与 f32 CPU 回退双路径
- KV Cache 的连续 VRAM 零拷贝访问
- RoPE、RMSNorm、SiLU、Softmax、Causal Mask 等 GPU kernel

模型配置：Qwen3-0.6B，28 层，16 Q 头，8 KV 头，head_dim=128，hidden=1024，vocab=151936，GQA，RoPE theta=1,000,000。

## 症状：模型输出乱码

初始运行时，模型输出完全不可读——重复 prompt 中的 token，而不是生成有意义的续写：

```
Prompt: "What is the capital of France?"
Output: "What is the capital of France? What is the capital of France? What is"
```

模型只是在复读输入，完全没有"理解"。

## 调试策略：逐层对比 HuggingFace 参考

面对一个从零构建的推理引擎，哪里都可能有 bug。我采用了**二分法 + 逐层对比**的策略：

1. 用 HuggingFace 运行相同的 prompt，获取每层的隐藏状态作为参考
2. 在 GPU 推理中输出每层的中间值
3. 逐层对比，找到第一个发散的位置
4. 在发散的层内，逐步对比每个子操作

这个策略的关键洞察是：**如果 embedding 都不对，后面全白费；如果 embedding 对了但第一层输出不对，bug 就在第一层。**

## Bug #1：SiLU 精度损失与寄存器溢出

### 发现

在对比 FFN 子层时，发现 SiLU 门控（`silu(gate) * up`）的输出与 CPU 参考存在显著差异。追踪到根因：BlockDSL 将 `silu(x)` 展开为多个独立节点（sigmoid、乘法等），导致寄存器分配器无法在 256 个 VGPR 的限制内完成分配，产生 spill，进而导致精度损失。

### 修复

将 SiLU 融合为单个 `BNode::SiluF32` 节点，整个 `x * sigmoid(x) * y` 操作在一个节点内完成，避免中间结果的寄存器溢出：

```rust
// block_dsl.rs
pub fn silu(&mut self, x: BVal) -> BVal {
    self.push(BNode::SiluF32(x))
}
```

修复后 SiLU 精度从 max_diff > 0.1 改善到 max_diff ≈ 0.000002。

### 教训

在 GPU 编程中，寄存器压力不仅影响性能，还可能直接影响计算正确性。融合操作（fused ops）不只是优化，有时是正确性的必要条件。

## Bug #2：LM Head 权重维度交换

### 发现

SiLU 修复后，模型仍然输出乱码。我检查了最终的 logits 分布，发现 top-1 token 与 HuggingFace 完全不同。追踪到 LM Head 的权重维度：

Qwen3 的 `tie_word_embeddings=true`，意味着 LM Head 共享 embedding 的权重矩阵。Embedding 权重的形状是 `[vocab_size, hidden_dim] = [151936, 1024]`。

`Linear::from_weight()` 根据 shape 自动推断：
- `in_features = shape[0] = 151936`（错误！应该是 1024）
- `out_features = shape[1] = 1024`（错误！应该是 151936）

对于 LM Head，正确的维度应该是 `in_features=hidden_size=1024, out_features=vocab_size=151936`，因为我们要把隐藏状态映射到词表空间。

### 修复

在 `tie_lm_head()` 中显式设置正确的维度：

```rust
pub fn tie_lm_head(&mut self) {
    self.lm_head.weight = self.embedding.weight.clone();
    self.lm_head.in_features = self.config.hidden_size;  // 1024
    self.lm_head.out_features = self.config.vocab_size;  // 151936
}
```

### 教训

权重共享（weight tying）是一个看似简单但容易出错的模式。不同模块对同一权重矩阵的"视角"不同——Embedding 把它当作查找表（行索引=token ID），而 Linear 把它当作投影矩阵（需要转置）。自动推断维度时必须考虑语义。

## Bug #3：GpuBuffer::new_view 大小为零

### 发现

在添加 CPU 参考对比代码时，尝试读取 KV Cache 数据触发 panic：

```
read overflow: 28672 > 0
```

追踪到 `GpuBuffer::new_view()` 创建的 buffer 的 `size` 字段为 0，`host_ptr` 为 null。

### 修复

`new_view` 是为 KV Cache 的零拷贝访问设计的——GPU 内存通过 mmap 映射到相同的 CPU 地址，所以 `host_ptr = gpu_addr as *mut u8` 是有效的。修复方法：

```rust
pub fn new_view(gpu_addr: u64, size: usize, device: Arc<KfdDevice>) -> GpuBuffer {
    GpuBuffer {
        handle: 0,
        va_addr: gpu_addr,
        host_ptr: gpu_addr as *mut u8,  // GPU VRAM mmap'd to same CPU addr
        size,                            // 之前是 0
        device,
    }
}
```

### 教训

零拷贝 GPU 内存访问需要确保 buffer 元数据正确。`size=0` 的 buffer 在写入时不会报错（GPU 直接操作 VRAM 地址），但读取时会 panic。这种"写时不报错、读时才崩"的模式特别隐蔽。

## Bug #4（根因）：RoPE 旋转风格错误

### 发现

LM Head 修复后，模型仍然重复 prompt。我进行了系统性的逐层对比：

```
Embedding:  GPU vs HF → 完全匹配 ✅
RMSNorm:   GPU vs HF → 完全匹配 ✅
Q/K/V 投影: GPU vs HF → 匹配（BF16 精度差异）✅
QK-Norm:   GPU vs HF → 匹配（BF16 精度差异）✅
RoPE 之后:  GPU vs HF → 严重不匹配 ❌❌❌
```

具体数值对比：

| 位置 | HF Q_after_rope[:4] | GPU Q_after_rope[:4] |
|------|---------------------|----------------------|
| 0 | 0.634458 | 0.561809 |
| 1 | -0.427455 | -0.719186 |
| 2 | -0.026764 | 0.046676 |
| 3 | 0.133038 | -0.063416 |

完全不同的值！这意味着位置编码从根本上就错了。

### 根因分析

RoPE 有两种常见的元素配对风格：

**Interleaved 风格**（我们错误使用的）：
```
对每对 (x[2i], x[2i+1]) 应用旋转：
  x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
  x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
```

**Rotate-half 风格**（HuggingFace/Qwen3 使用的）：
```
对每对 (x[i], x[i + d/2]) 应用旋转：
  x'[i]       = x[i] * cos(θ) - x[i + d/2] * sin(θ)
  x'[i + d/2] = x[i] * sin(θ) + x[i + d/2] * cos(θ)
```

两者的区别在于哪些元素被配对旋转。Interleaved 风格将相邻元素配对（0和1、2和3...），而 rotate-half 将前半部分和后半部分的对应元素配对（0和d/2、1和d/2+1...）。

这个差异看起来很小，但后果是灾难性的——每个位置的旋转都应用在了错误的元素对上，导致位置编码完全无效。模型无法区分不同位置的信息，自然只能复读输入。

### 修复

修改 GPU kernel，从 interleaved 改为 rotate-half：

```rust
// 之前：interleaved 风格
let even_idx = tid.mul(&mut kb, two_u);
let odd_idx = even_idx.add(&mut kb, one);
let x_even = kb.load(x_ptr, row_base + even_idx, mask);
let x_odd  = kb.load(x_ptr, row_base + odd_idx, mask);

// 之后：rotate-half 风格
let half_d = d_model.shr(&mut kb, 1);
let first_off = row_base.add(&mut kb, tid);
let second_off = row_base.add(&mut kb, half_d).add(&mut kb, tid);
let x_first  = kb.load(x_ptr, first_off, mask);
let x_second = kb.load(x_ptr, second_off, mask);
```

同时更新 CPU 参考实现和 backward kernel。

### 教训

这是整个调试过程中最关键的 bug，也是最容易被忽视的。RoPE 的两种风格在数学上是等价的（都是旋转变换），但在实现上配对方式完全不同。**当你的实现与参考框架的约定不一致时，即使每个操作本身都是正确的，整体结果也会完全错误。**

在实现 Transformer 模型时，必须仔细确认每个操作的约定（convention），特别是：
- RoPE 的配对风格（interleaved vs rotate-half）
- 权重矩阵的存储布局（行优先 vs 列优先，是否转置）
- 注意力缩放因子的应用位置
- LayerNorm/RMSNorm 的 epsilon 值

## 修复后的结果

所有 bug 修复后，模型输出完全正确：

```
Prompt: "What is the capital of France?"
Output: "The answer to this question is the capital of France, which is Paris."

Prompt: "The meaning of life is"
Output: "a question that has been asked to the people in the world for a long time."
```

推理速度约 0.8 tokens/s（BF16 WMMA GEMM 路径，单 GPU）。

## 调试方法论总结

### 1. 逐层对比是最强大的工具

面对一个从零构建的系统，不要猜测 bug 在哪里。用参考实现（HuggingFace）逐层对比，让数据告诉你答案。

### 2. 二分法定位

先对比 embedding → 第一层输出 → 最后一层输出 → 最终 logits。找到第一个发散的位置，然后在该层内继续二分。

### 3. 中间值对比比范数对比更有效

范数（norm）对比只能告诉你"有差异"，但中间值的前几个元素可以直接告诉你差异的性质——是缩放错误、偏移错误、还是完全不同的计算。

### 4. CPU 参考实现是必要的

为每个 GPU kernel 编写 CPU 参考实现，不仅用于测试，更是调试时最重要的对比基准。

### 5. 注意"约定"而非"算法"

很多 bug 不是算法实现错误，而是约定不一致（RoPE 的配对风格、权重的布局约定、维度的语义含义）。在实现前，先确认参考框架的约定。

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `src/t0/rope_kernels.rs` | RoPE 从 interleaved 改为 rotate-half 风格 |
| `src/ignis/ops/rope.rs` | CPU 参考实现同步修改 |
| `src/ignis/nn/model.rs` | tie_lm_head 维度修复、generate() EOS 修复 |
| `src/t0/block_dsl.rs` | 添加 SiluF32 融合节点 |
| `src/t0/block_dsl_to_ssa.rs` | SiluF32 SSA 降级处理 |
| `src/kfd/mod.rs` | new_view 添加 size 参数 |
| `src/ignis/tensor.rs` | from_gpu_addr 传递 size |
| `src/ignis/nn/linear.rs` | 环境变量重命名 T0_F32_GEMM → T0_PRECISE |
| `src/ignis/nn/transformer.rs` | 调试输出简化 |
| `examples/qwen3_infer.rs` | token 文本解码输出 |

---

*在裸金属上跑通一个大语言模型，需要的不仅是对算法的理解，更需要对每一层抽象的精确把控——从 GPU 指令集到模型权重布局，任何一个环节的约定不一致都会导致静默的错误。调试的过程，就是逐一验证每个约定的过程。*
