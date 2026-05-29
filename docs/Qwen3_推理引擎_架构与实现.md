# Qwen3 裸金属推理引擎 — 架构与实现

## 概述

在 t0-gpu 的 T0 编译器 + KFD 运行时 + Ignis 自动微分框架之上，实现了 Qwen3-0.6B / 4B 的端到端推理。不依赖 HIP/ROCm，直接通过 `/dev/kfd` 与 RX 7900 XTX 通信。

**当前状态**: 端到端推理已跑通，Qwen3-0.6B 可在 RX 7900 XTX 上生成文本。

**运行方式**:
```bash
cargo run --release --features rocm --example qwen3_infer -- \
  --model-path /mnt/public/models/huggingface/Qwen3-0.6B \
  --prompt "Hello" \
  --max-tokens 32 \
  --temperature 0.7
```

## 涉及文件总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/ignis/nn/config.rs` | 修改 | 添加 `head_dim` 字段，修复 Qwen3 配置解析 |
| `src/ignis/nn/transformer.rs` | 修改 | 添加 QK-norm、RoPE、标准 attention、KV cache 集成 |
| `src/ignis/nn/model.rs` | 修改 | 添加 `forward_prefill`、`forward_decode`、`generate` |
| `src/ignis/nn/embedding.rs` | 修改 | 添加 CPU 侧权重表缓存 |
| `src/ignis/nn/linear.rs` | 修改 | 添加 bf16 权重缓存，避免重复 GPU↔CPU 转换 |
| `src/ignis/ops/rope.rs` | **新建** | RoPE 算子包装 + CPU 参考实现 |
| `src/ignis/ops/qk_norm.rs` | **新建** | Per-head RMSNorm 算子 + CPU 参考实现 |
| `src/ignis/ops/attention.rs` | **新建** | 标准 scaled dot-product attention (CPU 实现) |
| `src/ignis/ops/argmax.rs` | 修改 | 添加 `sample_token` (temperature + top-p) |
| `src/ignis/ops/bf16_matmul.rs` | 修改 | 添加 `precompute_wt_bf16` 缓存路径 |
| `src/ignis/ops/rmsnorm.rs` | 修改 | 动态 workgroup 大小，修复 SGPR 溢出 |
| `src/ignis/ops/mod.rs` | 修改 | 注册 rope、qk_norm、attention 模块 |
| `src/ignis/kv_cache.rs` | 修改 | 添加 `read_k_layer`、`read_v_layer` |
| `src/ignis/safetensors.rs` | 修改 | q_norm/k_norm 加载、权重转置、模型目录加载 |
| `src/ignis/tensor.rs` | 修改 | `to_f32_vec` 支持 bf16→f32 转换 |
| `src/ignis/tokenizer.rs` | 修改 | 添加 `HfTokenizer` 包装器 |
| `src/t0/rope_kernels.rs` | 修改 | 添加 `pos_base` 参数支持 decode |
| `examples/qwen3_infer.rs` | **新建** | 推理示例 binary |

## Qwen3 架构要点

### 与 Qwen2 的关键差异

| 参数 | Qwen3-0.6B | Qwen3-4B |
|------|-----------|----------|
| hidden_size | 1024 | 2560 |
| num_layers | 28 | 36 |
| num_attention_heads | 16 | 32 |
| num_key_value_heads | 8 | 8 |
| **head_dim** | **128** | **128** |
| intermediate_size | 3072 | 12288 |
| vocab_size | 151936 | 151936 |
| rope_theta | 1,000,000 | 1,000,000 |
| tie_word_embeddings | true | true |

**head_dim 独立于 hidden_size**: Qwen3 的 `head_dim=128` 是显式配置的，不等于 `hidden_size / num_attention_heads`。Qwen3-0.6B 的 `q_dim = 16 × 128 = 2048 ≠ hidden_size=1024`。

### 推理流程

```
输入 token_ids
  │
  ▼
Embedding: [seq_len] → [seq_len, hidden_size]
  │
  ▼
┌─ TransformerLayer (×28) ──────────────────────────────┐
│  RMSNorm(attn_norm)                                    │
│  Q = x @ Wq    → [seq, q_dim]     (q_dim = 16×128)   │
│  K = x @ Wk    → [seq, kv_dim]    (kv_dim = 8×128)   │
│  V = x @ Wv    → [seq, kv_dim]                        │
│  Q = QK_norm(Q, q_norm_gamma)     per-head RMSNorm    │
│  K = QK_norm(K, k_norm_gamma)     per-head RMSNorm    │
│  Q = RoPE(Q, pos)                                      │
│  K = RoPE(K, pos)                                      │
│  KV_cache.append(K, V)                                 │
│  K_all, V_all = KV_cache.read()                        │
│  attn = standard_attention(Q, K_all, V_all)  (GQA)    │
│  x = x + attn @ Wo                                     │
│  RMSNorm(ffn_norm)                                     │
│  gate = SiLU(x @ W_gate)                               │
│  up = x @ W_up                                         │
│  x = x + (gate * up) @ W_down                          │
└────────────────────────────────────────────────────────┘
  │
  ▼
RMSNorm(final_norm)
  │
  ▼
LM Head: [hidden_size] → [vocab_size]  (logits)
  │
  ▼
Sample: argmax (temperature=0) 或 top-p (temperature>0)
```

## 新增算子详解

### 1. RoPE (`src/ignis/ops/rope.rs`)

包装 `t0::rope_kernels::build_rope_forward()` BlockDSL 内核。

- 输入: `[n_tokens, dim]` f32，dim ≤ 256
- 每对元素 (2i, 2i+1) 应用旋转: `x'[2i] = x[2i]·cos - x[2i+1]·sin`
- `pos_base` 参数: prefill 时为 0，decode 时为当前序列位置
- **使用方式**: reshape `[seq, q_dim]` → `[seq·n_heads, head_dim]`，逐 head 应用 RoPE，再 reshape 回来

CPU 参考实现: `cpu_rope_forward()`、`cpu_rope_inverse()`

### 2. QK-norm (`src/ignis/ops/qk_norm.rs`)

对 Q 和 K 的每个 head 独立应用 RMSNorm。

- 输入: `[seq_len, n_heads·head_dim]` f32
- 实现: reshape 为 `[seq_len·n_heads, head_dim]`，复用 `t0::rmsnorm_kernels` 的 RMSNorm 内核
- gamma: `[head_dim]`，所有 head 共享同一组 scale 参数

### 3. 标准 Attention (`src/ignis/ops/attention.rs`)

Scaled dot-product attention with GQA 支持。当前为 **CPU 实现**。

- Q: `[seq_len, n_heads·head_dim]`，K/V: `[kv_len, n_kv_heads·head_dim]`
- GQA: 每个 query head 映射到 `kv_head = h / gqa_ratio`
- Causal mask: prefill 时 (seq_len > 1 && kv_len == seq_len) 应用上三角 mask
- 输出: `[seq_len, n_heads·head_dim]`

### 4. Sampling (`src/ignis/ops/argmax.rs`)

`sample_token(logits, temperature, top_p, runtime) -> u32`

- `temperature <= 0`: greedy argmax
- `temperature > 0`: temperature scaling → softmax → top-p 过滤 → 随机采样
- CPU 实现，从 GPU 读取 logits 后在 CPU 上采样

## 性能瓶颈与优化方向

### 当前瓶颈

1. **CPU-based Attention** (`standard_attention`): 每层每 token 从 GPU 读取 K/V cache 到 CPU，在 CPU 上计算 attention，再上传结果。时间随 kv_len 线性增长。

2. **冗余 GPU↔CPU 数据搬运**: `standard_attention` 中 `Tensor::from_f32` 上传 K/V 到 GPU，`to_f32_vec()` 又读回来。

3. **首次运行内核编译**: 8 个唯一内核 (RMSNorm、RoPE、QK-norm、GEMM 等) 首次编译约 200s。

### 已做的优化

1. **bf16 权重缓存** (`Linear::cached_wt_bf16`): 避免每次 forward 都从 GPU 读取权重并转换。
2. **Embedding 权重表缓存** (`Embedding::cached_table`): 避免每次 decode 都读取 155M 元素。
3. **RMSNorm 动态 workgroup**: `dim=1024` 时用 256 线程，避免 SGPR 溢出。
4. **bf16→f32 转换** (`Tensor::to_f32_vec`): 支持 bf16 张量的正确读回。

### 后续优化建议

1. **GPU Attention**: 将 attention 移到 GPU 上执行，消除 CPU 瓶颈。
2. **GEMM 输入缓存**: `f32_to_bf16_gpu_padded` 每次都从 GPU 读取激活值并转换，应缓存 bf16 版本。
3. **Flash Attention**: 实现 fused attention kernel，避免中间结果的内存读写。
4. **Paged KV Cache**: 支持更高效的内存管理，减少碎片。
5. **Batch Decode**: 支持多序列并行解码。

## 测试覆盖

### CPU 参考测试 (24 个)

| 模块 | 测试数 | 覆盖范围 |
|------|--------|----------|
| `ops/rope.rs` | 8 | pos_zero、pos_offset、inverse_roundtrip、norm_preserve、different_positions、multi_token、compile_fwd、compile_bwd |
| `ops/qk_norm.rs` | 5 | ones_gamma、custom_gamma、heads_independent、multiple_tokens、kernel_compiles |
| `ops/attention.rs` | 6 | identity_k、causal_mask、gqa、decode_single_token、softmax_correctness、two_heads |
| `ops/argmax.rs` | 12 | argmax_basic/tie/single、sample_greedy/temperature/top_p/uniform/probs_sum |
| `kv_cache.rs` | 3 | read_layers、read_layers_multi、read_layers_empty |

### 运行测试

```bash
# CPU 参考测试 (需要 --features rocm 因为 ignis 模块被 gate)
cargo test --release --features rocm --lib -- "cpu_qk_norm" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "cpu_attention" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "cpu_argmax" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "cpu_sample" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "test_rope" --nocapture --test-threads=1
cargo test --release --features rocm --lib -- "test_kv_cache_read" --nocapture --test-threads=1

# 内核编译测试
cargo test --release --lib -- "compiles" --nocapture

# Qwen3 配置测试
cargo test --release --features rocm --lib -- "qwen3" --nocapture --test-threads=1
```

## 模型下载

```bash
# 使用 hfd.sh 下载
cd /mnt/public/models/huggingface
REPO_ID=Qwen/Qwen3-0.6B HF_ENDPOINT=https://hf-mirror.com bash hfd.sh Qwen/Qwen3-0.6B

# 或使用 curl 手动下载
TOKEN="hf_xxx"
BASE="https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main"
DIR="/path/to/Qwen3-0.6B"
for f in config.json tokenizer.json model.safetensors; do
    curl -L "$BASE/$f" -H "Authorization: Bearer $TOKEN" -o "$DIR/$f"
done
```

## 已知问题

1. `head_dim` 上限 256: RoPE 和 RMSNorm 内核限制 dim ≤ 256。
2. Attention 为 CPU 实现: 性能受限，随 kv_len 线性增长。
3. 首次运行内核编译慢: 约 200s，后续运行内核已缓存。
4. GPU embedding kernel (`forward_ids`) 对 u32 索引有类型错误，暂用 `forward_cpu`。
