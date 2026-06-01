# Lec 1: GPU 编程基础 — 从 CPU 思维到 GPU 思维

> **课程**: 人工智能硬件部署工程实践
> **目标**: 理解 GPU 编程模型，用 BlockDSL 写出第一个 GPU 内核

---

## 1. CPU vs GPU: 根本区别

### CPU 思维

```rust
// CPU: 一个线程，顺序处理每个元素
for i in 0..n {
    y[i] = a[i] + b[i];
}
```

- 1 个核心，做 n 次加法
- 顺序执行，延迟优先

### GPU 思维

```text
GPU: 1024 个线程，每个线程处理一个元素

Thread 0:   y[0] = a[0] + b[0]
Thread 1:   y[1] = a[1] + b[1]
Thread 2:   y[2] = a[2] + b[2]
...
Thread 1023: y[1023] = a[1023] + b[1023]

→ 1024 个加法同时执行（吞吐优先）
```

### 关键概念

| 概念 | CPU | GPU (RDNA3) |
|---|---|---|
| 执行单元 | Core (1-16 个) | CU (96 个) |
| 线程 | OS 线程，重量级 | Wave32，轻量级 |
| 并行粒度 | 任务级 | 数据级 |
| 内存 | DRAM (低延迟) | VRAM (高带宽) |
| 典型用途 | 控制流密集 | 计算密集 |

### GPU 线程层次

```text
Grid (整个内核)
 └── Workgroup (一个线程块，共享 LDS)
      └── Wave32 (32 个线程，SIMT 同步执行)
           └── Thread (单个线程)

RX 7900 XTX: 96 CU × 2 WGP/CU × 4 SIMD/WGP × 32 lanes = 24576 线程可同时执行
```

---

## 2. RDNA3 (GFX1100) 硬件模型

AMD RX 7900 XTX 的关键参数：

| 参数 | 值 | 含义 |
|---|---|---|
| CU 数量 | 96 | 计算单元 |
| Wave 大小 | 32 | 一组线程同步执行同一条指令 |
| VGPR/CU | 1536 | 向量通用寄存器（每线程用的寄存器） |
| SGPR/CU | 2048 | 标量通用寄存器（工作组共享） |
| LDS/CU | 64 KB | 本地数据共享（Workgroup 内共享） |
| VRAM | 24 GB GDDR6 | 全局显存 |
| 峰值算力 | 165 TFLOPS (bf16) | bf16 WMMA 指令 |

### 寄存器分类

```text
VGPR (Vector General Purpose Register):
  - 每个线程独立一份
  - 存储 per-thread 数据: x[i], y[i], 中间结果
  - 例: v_add_f32 v0, v1, v2  →  v0[tid] = v1[tid] + v2[tid]

SGPR (Scalar General Purpose Register):
  - 整个 workgroup 共享
  - 存储全局地址、循环计数、常量
  - 例: s_load_b32 s0, desc, offset  →  s0 = *desc[offset]

LDS (Local Data Share):
  - Workgroup 内线程共享的快速内存 (64 KB)
  - 用于 reduce、transpose、buffer 等
  - 比 VRAM 快 ~10x
```

---

## 3. T0 BlockDSL: GPU 内核的高级抽象

T0 的 BlockDSL 类似 Google Triton，让你用 Python/Rust 级别的抽象写 GPU 内核，编译器自动生成 ISA。

### 核心 API

```rust
use t0_gpu::t0::block_dsl::*;

// 创建内核：名字 + block_size（每工作组线程数）
let mut kb = BlockKernel::new("my_kernel", 256);

// 声明参数（GPU 端指针和标量）
let x_ptr = kb.arg_ptr("x");     // 输入指针
let y_ptr = kb.arg_ptr("y");     // 输出指针
let n     = kb.arg_u32("n");     // 元素数量

// 获取线程/工作组 ID
let tid = kb.thread_id();         // 0..block_size
let pid = kb.program_id(0);       // workgroup index

// 计算全局偏移
let offset = pid.mul(&mut kb, kb.const_u32(256)).add(&mut kb, tid);

// 边界检查
let mask = offset.lt(&mut kb, n);

// 加载（masked: 越界线程得到 0）
let val = kb.load(x_ptr, offset, mask);

// 计算
let result = val.mul(&mut kb, val);  // x²

// 存储（masked: 越界线程不写）
kb.store(y_ptr, offset, result, mask);

// 编译为 GFX1100 ELF
let elf = kb.compile(Target::GFX1100)?;
```

### BlockDSL 方法速查

| 方法 | 含义 | 示例 |
|---|---|---|
| `a.add(&mut kb, b)` | 加法 | `a + b` |
| `a.sub(&mut kb, b)` | 减法 | `a - b` |
| `a.mul(&mut kb, b)` | 乘法 | `a * b` |
| `a.div(&mut kb, b)` | 除法 | `a / b` |
| `a.exp(&mut kb)` | 自然指数 | `e^a` |
| `a.log(&mut kb)` | 自然对数 | `ln(a)` |
| `a.sqrt(&mut kb)` | 平方根 | `√a` |
| `a.relu(&mut kb)` | ReLU | `max(0, a)` |
| `a.sigmoid(&mut kb)` | Sigmoid | `1/(1+e^-a)` |
| `a.silu(&mut kb)` | SiLU | `x * sigmoid(x)` |
| `a.gelu(&mut kb)` | GELU | `x * Φ(x)` |
| `a.lt(&mut kb, b)` | 小于比较 | `a < b → bool` |
| `mask.select(&mut kb, t, f)` | 条件选择 | `if mask { t } else { f }` |
| `kb.wg_reduce_max(v)` | Workgroup 最大值归约 | 行内所有线程的最大值 |
| `kb.wg_reduce_sum(v)` | Workgroup 求和归约 | 行内所有线程的和 |
| `kb.wg_reduce_sum_sq(v)` | Workgroup 平方和归约 | 用于 RMSNorm |

---

## 4. 实战：Softmax 内核

Softmax 是 Transformer 的核心操作。算法：

```text
对每一行 x[0..n]:
  1. max_val  = max(x)                    // 数值稳定性
  2. exp_x[i] = exp(x[i] - max_val)       // 减去最大值防溢出
  3. sum_exp  = sum(exp_x)                 // 归一化因子
  4. y[i]     = exp_x[i] / sum_exp         // 概率分布
```

### BlockDSL 实现（来自 `src/t0/softmax_kernels.rs`）

```rust
pub fn build_softmax_forward() -> BlockKernel {
    let mut kb = BlockKernel::new("softmax_fwd", 256);

    let input_ptr  = kb.arg_ptr("input");
    let output_ptr = kb.arg_ptr("output");
    let cols        = kb.arg_u32("cols");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);  // 每个 workgroup 处理一行

    // 偏移 = 行号 * 列数 + 线程号
    let row_base = pid.mul(&mut kb, cols);
    let offset   = row_base.add(&mut kb, tid);
    let mask     = tid.lt(&mut kb, cols);  // 越界线程 mask

    // 加载（越界线程得到 0）
    let x = kb.load(input_ptr, offset, mask);

    // Phase 1: 行最大值（越界用 -inf 替代，不影响 max）
    let neg_inf   = kb.const_f32(f32::NEG_INFINITY);
    let x_for_max = mask.select(&mut kb, x, neg_inf);
    let row_max   = kb.wg_reduce_max(x_for_max);

    // Phase 2: exp(x - max)
    let shifted = x.sub(&mut kb, row_max);
    let exp_x   = shifted.exp(&mut kb);

    // Phase 3: 行求和（越界用 0 替代）
    let zero_f    = kb.const_f32(0.0);
    let exp_masked = mask.select(&mut kb, exp_x, zero_f);
    let exp_sum   = kb.wg_reduce_sum(exp_masked);
    let inv_sum   = exp_sum.rcp(&mut kb);  // 1/sum，乘法比除法快

    // Phase 4: output = exp(x - max) / sum
    let result = exp_x.mul(&mut kb, inv_sum);
    kb.store(output_ptr, offset, result, mask);

    kb
}
```

### 为什么这样写？

1. **每行一个 workgroup** — `pid` 就是行号，同一行的线程需要 reduce
2. **mask 机制** — 列数可能不是 256 的倍数，越界线程必须被 mask 掉
3. **减最大值** — `exp(x)` 在 x 很大时会溢出（f32 上限 ~88），减最大值保证所有指数 ≤ 0
4. **用 `rcp` 而非 `div`** — 倒数 + 乘法比除法快（硬件原生支持）
5. **`wg_reduce_max` / `wg_reduce_sum`** — 编译器自动生成 wave reduce + LDS 归约代码

---

## 5. 实战：从 BlockDSL 到 GPU 执行

完整流程（参考 `examples/hello_gemm.rs`）：

```rust
// Step 1: 编译内核
let kernel = build_softmax_forward();
let elf = kernel.compile(Target::GFX1100)?;
// → 生成 AMD HSA ELF 二进制（~200-500 字节）

// Step 2: 打开 GPU
let device = KfdDevice::open()?;
let queue = device.create_queue()?;

// Step 3: 加载内核到 GPU
let gpu_kernel = GpuKernel::load(&device, &elf, &KernelLoadConfig {
    workgroup_size: [256, 1, 1],
    lds_size: 0,  // softmax 不用 LDS
})?;

// Step 4: 分配 VRAM 并上传数据
let rows = 4;
let cols = 128;
let input_buf  = device.alloc_vram(rows * cols * 4)?;  // f32
let output_buf = device.alloc_vram(rows * cols * 4)?;

// 上传测试数据（省略填充代码）
input_buf.write(&input_data);

// Step 5: 构建 kernel arguments
// kernarg layout: [input_ptr:u64, output_ptr:u64, cols:u32]
let mut ka = [0u8; 20];
ka[0..8].copy_from_slice(&input_buf.gpu_addr().to_le_bytes());
ka[8..16].copy_from_slice(&output_buf.gpu_addr().to_le_bytes());
ka[16..20].copy_from_slice(&cols.to_le_bytes());

let pool = DispatchPool::new(&device, 4)?;
let ka_buf = pool.write_kernargs(0, &ka);

// Step 6: Dispatch
// Grid: rows 个工作组，每组 256 线程
let grid_x = rows * 256;
queue.submit(&gpu_kernel, [grid_x, 1, 1], ka_buf);
queue.wait_idle()?;  // 等 GPU 完成

// Step 7: 读回结果
output_buf.read(&mut output_data);
```

### 调度模型

```text
CPU 端                              GPU 端
────────                           ────────
queue.submit(kernel, grid, args)
   │                                ┌─────────────────┐
   │  ─── AQL packet (2μs) ───→    │ Workgroup 0     │
   │                                │  Wave 0: t0-t31 │
   │                                │  处理 row 0     │
   │                                ├─────────────────┤
   │                                │ Workgroup 1     │
   │                                │  处理 row 1     │
   │                                ├─────────────────┤
   │                                │ ...             │
   │                                └─────────────────┘
   │                                        │
   ← wait_idle()                            │
   │                                完成，写入 VRAM
```

---

## 6. Lab 1 作业

### 作业 1.1: 向量乘法 (30 分)

用 BlockDSL 实现 `y[i] = a[i] * b[i]`。

**要求**:
- 基于 `build_softmax_forward` 的模式改造
- 正确处理边界（n 不一定是 256 的倍数）
- 用 `cargo run --example your_example --features rocm --release` 验证

**Starter code**:

```rust
pub fn build_vec_mul() -> BlockKernel {
    let mut kb = BlockKernel::new("vec_mul", 256);

    let a_ptr = kb.arg_ptr("a");
    let b_ptr = kb.arg_ptr("b");
    let y_ptr = kb.arg_ptr("y");
    let n     = kb.arg_u32("n");

    let tid = kb.thread_id();
    let pid = kb.program_id(0);

    // TODO: 计算全局偏移
    // let offset = ...;

    // TODO: 边界检查
    // let mask = ...;

    // TODO: 加载 a 和 b
    // let a_val = ...;
    // let b_val = ...;

    // TODO: 乘法并存储
    // let result = ...;
    // kb.store(...);

    kb
}
```

### 作业 1.2: SiLU 激活函数 (30 分)

实现 `y[i] = x[i] * sigmoid(x[i])`。

**提示**: `BVal` 有 `.sigmoid()` 和 `.silu()` 方法，但本次作业请手动实现（用 `.exp()` 和基本运算）。

```rust
pub fn build_silu() -> BlockKernel {
    let mut kb = BlockKernel::new("silu", 256);

    let x_ptr = kb.arg_ptr("x");
    let y_ptr = kb.arg_ptr("y");
    let n     = kb.arg_u32("n");

    // TODO: 实现 y[i] = x[i] * sigmoid(x[i])
    // sigmoid(x) = 1 / (1 + exp(-x))
    // 提示: 用 exp(-x).rcp() 或 1/(1+exp(-x))

    kb
}
```

### 作业 1.3: 在线 Softmax (40 分)

参考上面的 softmax 实现，完成一个 **带温度缩放** 的 softmax：

```text
y[i] = softmax(x[i] / temperature)
```

**要求**:
- 温度 `temperature` 作为 kernel argument 传入
- 当 temperature = 1.0 时，行为与标准 softmax 一致
- 当 temperature < 1.0 时，分布更尖锐（更确定性）
- 当 temperature > 1.0 时，分布更平滑（更随机）

**提示**: `x[i] / temperature` 可以在加载后立即做，不影响后续逻辑。

---

## 7. 思考题

1. **为什么 Softmax 每行需要一个 workgroup 而不是一个 wave？** 提示：`wg_reduce_max` 需要什么级别的通信？

2. **如果 cols > 256（超过一个 workgroup 的线程数），这个 softmax 实现会怎样？** 提示：`wg_reduce_max` 只能在一个 workgroup 内归约。

3. **GPU 的 `exp(x)` 和 CPU 的 `f32::exp()` 结果完全一样吗？为什么这在 ML 中通常不是问题？**

---

## 参考资料

- BlockDSL 源码: `src/t0/block_dsl.rs`
- Softmax 内核: `src/t0/softmax_kernels.rs`
- 向量加法示例: `examples/hello_gemm.rs`
- GFX1100 ISA 手册: AMD RDNA3 Instruction Set Architecture
