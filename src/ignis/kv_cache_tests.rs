//! KV Cache correctness and performance tests.
//!
//! Run: cargo test --release --features rocm -- kv_cache_integration --test-threads=1 --nocapture

#[cfg(all(test, feature = "rocm"))]
mod kv_cache_integration {
    use std::sync::{Arc, OnceLock};
    use crate::ignis::gpu_context::GpuRuntime;
    use crate::ignis::tensor::Tensor;
    use crate::ignis::kv_cache::{KvCache, KvCacheConfig};

    struct SyncRt(Arc<GpuRuntime>);
    unsafe impl Sync for SyncRt {}
    unsafe impl Send for SyncRt {}
    static GPU_RT: OnceLock<SyncRt> = OnceLock::new();

    fn rt() -> Arc<GpuRuntime> {
        GPU_RT.get_or_init(|| {
            SyncRt(GpuRuntime::new().expect("Failed to create GpuRuntime"))
        }).0.clone()
    }

    // ═══════════════════════════════════════════════════════════════
    // Correctness Tests
    // ═══════════════════════════════════════════════════════════════

    /// End-to-end correctness: fill KV cache with known pattern,
    /// read back every byte, verify exact match.
    #[test]
    fn test_kv_cache_roundtrip_correctness() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 8,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 256,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = cfg.num_kv_heads * cfg.head_dim; // 1024

        // Phase 1: Prefill with 64 tokens
        let prefill_seq = 64;
        for layer in 0..cfg.num_layers {
            let mut k_data = Vec::with_capacity(prefill_seq * head_elements);
            let mut v_data = Vec::with_capacity(prefill_seq * head_elements);
            for s in 0..prefill_seq {
                for i in 0..head_elements {
                    // Deterministic pattern
                    k_data.push(((layer * 10000 + s * 100 + i) % 10000) as f32 * 0.001);
                    v_data.push(((layer * 20000 + s * 100 + i) % 20000) as f32 * 0.001);
                }
            }
            let keys = Tensor::from_f32(&r, &k_data, &[prefill_seq, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let vals = Tensor::from_f32(&r, &v_data, &[prefill_seq, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
            cache.append_many(&r, layer, &keys, &vals).unwrap();
        }
        cache.advance_by(prefill_seq);

        // Verify each layer
        for layer in 0..cfg.num_layers {
            let k_data: Vec<f32> = (0..prefill_seq * head_elements)
                .map(|idx| ((layer * 10000 + (idx / head_elements) * 100 + (idx % head_elements)) % 10000) as f32 * 0.001)
                .collect();
            let v_data: Vec<f32> = (0..prefill_seq * head_elements)
                .map(|idx| ((layer * 20000 + (idx / head_elements) * 100 + (idx % head_elements)) % 20000) as f32 * 0.001)
                .collect();

            // Read back entire layer K slice
            let k_slice = cache.get_k(layer);
            assert_eq!(k_slice.seq_len, prefill_seq);

            // Read token-by-token for correctness
            for s in 0..prefill_seq {
                let k_read = cache.read_k_token(&r, layer, s);
                let v_read = cache.read_v_token(&r, layer, s);
                let k_start = s * head_elements;
                let v_start = s * head_elements;

                for i in 0..head_elements {
                    let expected_k = k_data[k_start + i];
                    let expected_v = v_data[v_start + i];
                    assert!((k_read[i] - expected_k).abs() < 1e-5,
                        "L{} pos{} K[{}]: got {:.6}, expected {:.6}",
                        layer, s, i, k_read[i], expected_k);
                    assert!((v_read[i] - expected_v).abs() < 1e-5,
                        "L{} pos{} V[{}]: got {:.6}, expected {:.6}",
                        layer, s, i, v_read[i], expected_v);
                }
            }
        }

        eprintln!("  ✓ Roundtrip correctness: {} layers × {} tokens × {} elements verified",
            cfg.num_layers, prefill_seq, head_elements);
    }

    /// Stress test: fill cache to max capacity, verify every position.
    #[test]
    fn test_kv_cache_max_capacity_correctness() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 4,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 128,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Fill all 128 positions one at a time (decode simulation)
        for pos in 0..cfg.max_seq_len {
            for layer in 0..cfg.num_layers {
                let k_data: Vec<f32> = (0..head_elements)
                    .map(|i| (pos * 1000 + layer * 100 + i) as f32)
                    .collect();
                let v_data: Vec<f32> = (0..head_elements)
                    .map(|i| (pos * 2000 + layer * 100 + i) as f32)
                    .collect();
                let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
                let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
                cache.append(&r, layer, &key, &val).unwrap();
            }
            cache.advance();
        }

        assert_eq!(cache.position(), cfg.max_seq_len);
        assert_eq!(cache.remaining(), 0);

        // Verify every position in every layer
        for layer in 0..cfg.num_layers {
            for pos in 0..cfg.max_seq_len {
                let k_read = cache.read_k_token(&r, layer, pos);
                let v_read = cache.read_v_token(&r, layer, pos);
                for i in 0..head_elements {
                    let expected_k = (pos * 1000 + layer * 100 + i) as f32;
                    let expected_v = (pos * 2000 + layer * 100 + i) as f32;
                    assert!((k_read[i] - expected_k).abs() < 1e-5,
                        "L{} pos{} K[{}]: got {}, expected {}",
                        layer, pos, i, k_read[i], expected_k);
                    assert!((v_read[i] - expected_v).abs() < 1e-5,
                        "L{} pos{} V[{}]: got {}, expected {}",
                        layer, pos, i, v_read[i], expected_v);
                }
            }
        }

        eprintln!("  ✓ Max capacity: {} layers × {} positions fully verified",
            cfg.num_layers, cfg.max_seq_len);
    }

    /// Verify get_k/get_v return correct GPU addresses that match actual data layout.
    #[test]
    fn test_kv_cache_slice_address_correctness() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 2,
            num_kv_heads: 2,
            head_dim: 8,
            max_seq_len: 32,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Fill 10 positions
        for pos in 0..10 {
            for layer in 0..cfg.num_layers {
                let k_data: Vec<f32> = (0..head_elements).map(|i| (pos * 100 + i) as f32).collect();
                let v_data: Vec<f32> = (0..head_elements).map(|i| (pos * 200 + i) as f32).collect();
                let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
                let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
                cache.append(&r, layer, &key, &val).unwrap();
            }
            cache.advance();
        }

        // Verify get_k returns the right slice
        let k_slice = cache.get_k(0);
        assert_eq!(k_slice.seq_len, 10);
        assert_eq!(k_slice.num_kv_heads, 2);
        assert_eq!(k_slice.head_dim, 8);

        // The GPU address should point to layer 0 K start
        let expected_k_addr = cache.gpu_addr() + cache.k_offset(0, 0) as u64;
        assert_eq!(k_slice.gpu_addr, expected_k_addr);

        let v_slice = cache.get_v(0);
        let expected_v_addr = cache.gpu_addr() + cache.v_offset(0, 0) as u64;
        assert_eq!(v_slice.gpu_addr, expected_v_addr);

        // K and V should be separated by exactly max_seq_len * head_elements * 4 bytes
        let expected_separation = (cfg.max_seq_len * head_elements * 4) as u64;
        assert_eq!(v_slice.gpu_addr - k_slice.gpu_addr, expected_separation);

        // Layer 1 should be offset by layer size
        let k_slice_l1 = cache.get_k(1);
        let expected_l1_separation = (2 * cfg.max_seq_len * head_elements * 4) as u64; // K+V per layer
        assert_eq!(k_slice_l1.gpu_addr - k_slice.gpu_addr, expected_l1_separation);

        eprintln!("  ✓ Slice addresses: correct separation K↔V={}, L0↔L1={}",
            expected_separation, expected_l1_separation);
    }

    // ═══════════════════════════════════════════════════════════════
    // Performance Benchmarks
    // ═══════════════════════════════════════════════════════════════

    /// Benchmark prefill throughput: bulk KV copy for all layers.
    #[test]
    fn test_kv_cache_prefill_throughput() {
        let r = rt();

        let configs = vec![
            ("Tiny",  KvCacheConfig { num_layers: 4,   num_kv_heads: 4,   head_dim: 64,  max_seq_len: 1024 }),
            ("Small", KvCacheConfig { num_layers: 8,   num_kv_heads: 8,   head_dim: 128, max_seq_len: 2048 }),
            ("Medium", KvCacheConfig { num_layers: 16,  num_kv_heads: 8,   head_dim: 128, max_seq_len: 4096 }),
            ("Qwen3-8B-like", KvCacheConfig { num_layers: 36, num_kv_heads: 8, head_dim: 128, max_seq_len: 8192 }),
        ];

        let seq_lens = vec![64, 256, 1024];

        for (name, cfg) in &configs {
            let cache = KvCache::new(&r, cfg.clone()).unwrap();
            let head_elements = cfg.num_kv_heads * cfg.head_dim;

            for &seq_len in &seq_lens {
                if seq_len > cfg.max_seq_len { continue; }

                let total_bytes = cfg.num_layers * 2 * seq_len * head_elements * 4;

                // Warmup
                let warmup_data = vec![0.0f32; seq_len * head_elements];
                for layer in 0..cfg.num_layers {
                    let keys = Tensor::from_f32(&r, &warmup_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "wk").unwrap();
                    let vals = Tensor::from_f32(&r, &warmup_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "wv").unwrap();
                    let _ = cache.append_many(&r, layer, &keys, &vals);
                }
                cache.reset();

                // Timed
                let n_iters = 10;
                let t0 = std::time::Instant::now();
                for _ in 0..n_iters {
                    for layer in 0..cfg.num_layers {
                        let keys = Tensor::from_f32(&r, &warmup_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
                        let vals = Tensor::from_f32(&r, &warmup_data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();
                        let _ = cache.append_many(&r, layer, &keys, &vals);
                    }
                    cache.advance_by(seq_len);
                    cache.reset();
                }
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let per_iter_ms = elapsed_ms / n_iters as f64;
                let throughput_gbps = if per_iter_ms > 0.0 {
                    (total_bytes as f64 * 2.0 / (1024.0 * 1024.0 * 1024.0)) / (per_iter_ms / 1000.0)
                } else { 0.0 };

                eprintln!("  {:<14} seq={:<5} {:.3} ms  ({:.1} GB/s)",
                    name, seq_len, per_iter_ms, throughput_gbps);
            }
        }
    }

    /// Benchmark decode latency: single-token KV append using sync dispatch (old path).
    /// This measures the baseline: N layers × 2 sync dispatches (K+V) per token.
    #[test]
    fn test_kv_cache_decode_latency_sync() {
        let r = rt();

        let configs = vec![
            ("Tiny",  KvCacheConfig { num_layers: 4,   num_kv_heads: 4,   head_dim: 64,  max_seq_len: 1024 }),
            ("Small", KvCacheConfig { num_layers: 8,   num_kv_heads: 8,   head_dim: 128, max_seq_len: 2048 }),
            ("Medium", KvCacheConfig { num_layers: 16,  num_kv_heads: 8,   head_dim: 128, max_seq_len: 4096 }),
            ("Qwen3-8B-like", KvCacheConfig { num_layers: 36, num_kv_heads: 8, head_dim: 128, max_seq_len: 8192 }),
        ];

        for (name, cfg) in &configs {
            let cache = KvCache::new(&r, cfg.clone()).unwrap();
            let head_elements = cfg.num_kv_heads * cfg.head_dim;

            // Pre-allocate tensors to eliminate alloc overhead
            let k_data = vec![0.5f32; head_elements];
            let v_data = vec![0.5f32; head_elements];
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

            // Warmup
            for _ in 0..3 {
                for layer in 0..cfg.num_layers {
                    let _ = cache.append(&r, layer, &key, &val);
                }
                cache.advance();
            }
            cache.reset();

            // Timed: measure per-token decode latency (sync dispatch per layer)
            let n_iters = 100;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                for layer in 0..cfg.num_layers {
                    let _ = cache.append(&r, layer, &key, &val);
                }
                cache.advance();
                cache.reset();
            }
            let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
            let per_token_us = elapsed_us / n_iters as f64;
            let per_layer_us = per_token_us / cfg.num_layers as f64;

            let tokens_per_sec = if per_token_us > 0.0 { 1e6 / per_token_us } else { 0.0 };

            eprintln!("  [SYNC] {:<14} {} layers  {:.1} μs/token  ({:.2} μs/layer)  {:.0} tok/s",
                name, cfg.num_layers, per_token_us, per_layer_us, tokens_per_sec);
        }
    }

    /// Benchmark decode latency: async batch path (append_batch).
    /// Submits all layer copies async, then syncs once.
    /// This is the optimal decode path: N async dispatches → 1 sync.
    #[test]
    fn test_kv_cache_decode_latency_async() {
        let r = rt();

        let configs = vec![
            ("Tiny",  KvCacheConfig { num_layers: 4,   num_kv_heads: 4,   head_dim: 64,  max_seq_len: 1024 }),
            ("Small", KvCacheConfig { num_layers: 8,   num_kv_heads: 8,   head_dim: 128, max_seq_len: 2048 }),
            ("Medium", KvCacheConfig { num_layers: 16,  num_kv_heads: 8,   head_dim: 128, max_seq_len: 4096 }),
            ("Qwen3-8B-like", KvCacheConfig { num_layers: 36, num_kv_heads: 8, head_dim: 128, max_seq_len: 8192 }),
        ];

        for (name, cfg) in &configs {
            let cache = KvCache::new(&r, cfg.clone()).unwrap();
            let head_elements = cfg.num_kv_heads * cfg.head_dim;

            // Pre-allocate tensors
            let k_data = vec![0.5f32; head_elements];
            let v_data = vec![0.5f32; head_elements];
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

            // Build layer_kvs array once
            let layer_kvs: Vec<(usize, &Tensor, &Tensor)> =
                (0..cfg.num_layers).map(|l| (l, &key, &val)).collect();

            // Warmup
            for _ in 0..3 {
                let _ = cache.append_batch(&r, &layer_kvs);
                cache.advance();
            }
            cache.reset();

            // Timed: async batch decode
            let n_iters = 100;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                let _ = cache.append_batch(&r, &layer_kvs);
                cache.advance();
                cache.reset();
            }
            let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
            let per_token_us = elapsed_us / n_iters as f64;
            let per_layer_us = per_token_us / cfg.num_layers as f64;

            let tokens_per_sec = if per_token_us > 0.0 { 1e6 / per_token_us } else { 0.0 };

            eprintln!("  [ASYNC] {:<13} {} layers  {:.1} μs/token  ({:.2} μs/layer)  {:.0} tok/s",
                name, cfg.num_layers, per_token_us, per_layer_us, tokens_per_sec);
        }
    }

    /// Benchmark decode latency: single-token KV append per layer.
    #[test]
    fn test_kv_cache_decode_latency() {
        let r = rt();

        let configs = vec![
            ("Tiny",  KvCacheConfig { num_layers: 4,   num_kv_heads: 4,   head_dim: 64,  max_seq_len: 1024 }),
            ("Small", KvCacheConfig { num_layers: 8,   num_kv_heads: 8,   head_dim: 128, max_seq_len: 2048 }),
            ("Medium", KvCacheConfig { num_layers: 16,  num_kv_heads: 8,   head_dim: 128, max_seq_len: 4096 }),
            ("Qwen3-8B-like", KvCacheConfig { num_layers: 36, num_kv_heads: 8, head_dim: 128, max_seq_len: 8192 }),
        ];

        for (name, cfg) in &configs {
            let cache = KvCache::new(&r, cfg.clone()).unwrap();
            let head_elements = cfg.num_kv_heads * cfg.head_dim;

            // Pre-allocate tensors to eliminate alloc overhead
            let k_data = vec![0.5f32; head_elements];
            let v_data = vec![0.5f32; head_elements];
            let key = Tensor::from_f32(&r, &k_data, &[cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
            let val = Tensor::from_f32(&r, &v_data, &[cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

            // Warmup
            for _ in 0..3 {
                for layer in 0..cfg.num_layers {
                    let _ = cache.append(&r, layer, &key, &val);
                }
                cache.advance();
            }
            cache.reset();

            // Timed: measure per-token decode latency
            let n_iters = 100;
            let t0 = std::time::Instant::now();
            for _ in 0..n_iters {
                for layer in 0..cfg.num_layers {
                    let _ = cache.append(&r, layer, &key, &val);
                }
                cache.advance();
                cache.reset();
            }
            let elapsed_us = t0.elapsed().as_secs_f64() * 1e6;
            let per_token_us = elapsed_us / n_iters as f64;
            let per_layer_us = per_token_us / cfg.num_layers as f64;

            // Decode tokens/sec
            let tokens_per_sec = if per_token_us > 0.0 { 1e6 / per_token_us } else { 0.0 };

            eprintln!("  {:<14} {} layers  {:.1} μs/token  ({:.2} μs/layer)  {:.0} tok/s",
                name, cfg.num_layers, per_token_us, per_layer_us, tokens_per_sec);
        }
    }

    /// Benchmark reset + prefill cycle (simulates multi-turn conversation) — sync path.
    #[test]
    fn test_kv_cache_multi_turn_sync() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 8,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        let seq_len = 256;
        let data = vec![0.5f32; seq_len * head_elements];
        let keys = Tensor::from_f32(&r, &data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let vals = Tensor::from_f32(&r, &data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

        // Warmup
        for _ in 0..3 {
            for layer in 0..cfg.num_layers {
                let _ = cache.append_many(&r, layer, &keys, &vals);
            }
            cache.advance_by(seq_len);
            cache.reset();
        }

        // Timed: simulate 20 conversation turns
        let n_turns = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_turns {
            for layer in 0..cfg.num_layers {
                let _ = cache.append_many(&r, layer, &keys, &vals);
            }
            cache.advance_by(seq_len);
            cache.reset();
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let per_turn_ms = elapsed_ms / n_turns as f64;

        let total_bytes = cfg.num_layers * 2 * seq_len * head_elements * 4;
        let bandwidth_gbs = (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (per_turn_ms / 1000.0);
        eprintln!("  [SYNC] Multi-turn: {} turns, {} tokens each, {} layers", n_turns, seq_len, cfg.num_layers);
        eprintln!("  Per turn: {:.2} ms ({:.1} GB/s)", per_turn_ms, bandwidth_gbs);
    }

    /// Benchmark reset + prefill cycle — async batch path (append_many_batch).
    /// Submits all layer copies async, then syncs once.
    #[test]
    fn test_kv_cache_multi_turn_async() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 8,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        let seq_len = 256;
        let data = vec![0.5f32; seq_len * head_elements];
        let keys = Tensor::from_f32(&r, &data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let vals = Tensor::from_f32(&r, &data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

        // Build layer_kvs array once
        let layer_kvs: Vec<(usize, &Tensor, &Tensor)> =
            (0..cfg.num_layers).map(|l| (l, &keys, &vals)).collect();

        // Warmup
        for _ in 0..3 {
            let _ = cache.append_many_batch(&r, &layer_kvs);
            cache.advance_by(seq_len);
            cache.reset();
        }

        // Timed: simulate 20 conversation turns
        let n_turns = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_turns {
            let _ = cache.append_many_batch(&r, &layer_kvs);
            cache.advance_by(seq_len);
            cache.reset();
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let per_turn_ms = elapsed_ms / n_turns as f64;

        let total_bytes = cfg.num_layers * 2 * seq_len * head_elements * 4;
        let bandwidth_gbs = (total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (per_turn_ms / 1000.0);
        eprintln!("  [ASYNC] Multi-turn: {} turns, {} tokens each, {} layers", n_turns, seq_len, cfg.num_layers);
        eprintln!("  Per turn: {:.2} ms ({:.1} GB/s)", per_turn_ms, bandwidth_gbs);
    }

    /// Benchmark reset + prefill cycle (simulates multi-turn conversation).
    #[test]
    fn test_kv_cache_multi_turn() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 8,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        let seq_len = 256;
        let data = vec![0.5f32; seq_len * head_elements];
        let keys = Tensor::from_f32(&r, &data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let vals = Tensor::from_f32(&r, &data, &[seq_len, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

        // Warmup
        for _ in 0..3 {
            for layer in 0..cfg.num_layers {
                let _ = cache.append_many(&r, layer, &keys, &vals);
            }
            cache.advance_by(seq_len);
            cache.reset();
        }

        // Timed: simulate 20 conversation turns
        let n_turns = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..n_turns {
            for layer in 0..cfg.num_layers {
                let _ = cache.append_many(&r, layer, &keys, &vals);
            }
            cache.advance_by(seq_len);
            cache.reset();
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let per_turn_ms = elapsed_ms / n_turns as f64;

        eprintln!("  Multi-turn: {} turns, {} tokens each, {} layers", n_turns, seq_len, cfg.num_layers);
        eprintln!("  Per turn: {:.2} ms", per_turn_ms);
    }

    /// Stress test: interleaved prefill + decode for 1000 tokens.
    #[test]
    fn test_kv_cache_long_sequence_stress() {
        let r = rt();
        let cfg = KvCacheConfig {
            num_layers: 8,
            num_kv_heads: 8,
            head_dim: 128,
            max_seq_len: 1024,
        };
        let cache = KvCache::new(&r, cfg.clone()).unwrap();
        let head_elements = cfg.num_kv_heads * cfg.head_dim;

        // Prefill 256 tokens
        let prefill_seq = 256;
        let k_data = vec![0.5f32; prefill_seq * head_elements];
        let v_data = vec![0.5f32; prefill_seq * head_elements];
        let keys = Tensor::from_f32(&r, &k_data, &[prefill_seq, cfg.num_kv_heads, cfg.head_dim], "k").unwrap();
        let vals = Tensor::from_f32(&r, &v_data, &[prefill_seq, cfg.num_kv_heads, cfg.head_dim], "v").unwrap();

        let t0 = std::time::Instant::now();
        for layer in 0..cfg.num_layers {
            let _ = cache.append_many(&r, layer, &keys, &vals);
        }
        cache.advance_by(prefill_seq);
        let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Decode 768 tokens one by one
        let single_k = vec![0.3f32; head_elements];
        let single_v = vec![0.3f32; head_elements];
        let key = Tensor::from_f32(&r, &single_k, &[cfg.num_kv_heads, cfg.head_dim], "dk").unwrap();
        let val = Tensor::from_f32(&r, &single_v, &[cfg.num_kv_heads, cfg.head_dim], "dv").unwrap();

        let t1 = std::time::Instant::now();
        for _ in 0..768 {
            for layer in 0..cfg.num_layers {
                let _ = cache.append(&r, layer, &key, &val);
            }
            cache.advance();
        }
        let decode_ms = t1.elapsed().as_secs_f64() * 1000.0;

        eprintln!("  Long sequence stress (8 layers, 128 dim, 8 heads):");
        eprintln!("    Prefill 256 tokens: {:.2} ms ({:.1} tokens/s)",
            prefill_ms, 256.0 / (prefill_ms / 1000.0));
        eprintln!("    Decode 768 tokens: {:.2} ms ({:.1} tokens/s)",
            decode_ms, 768.0 / (decode_ms / 1000.0));
        eprintln!("    Total 1024 tokens: {:.2} ms", prefill_ms + decode_ms);

        assert_eq!(cache.position(), 1024);
    }
}
