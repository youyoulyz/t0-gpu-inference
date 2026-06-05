use super::block_dsl::*;

pub fn build_flash_attn_decode(head_dim: u32, gqa_ratio_log2: u8) -> BlockKernel {
    let wg_size = head_dim;
    let mut kb = BlockKernel::new("flash_attn_decode", wg_size);

    let q_ptr = kb.arg_ptr("Q");
    let k_ptr = kb.arg_ptr("K");
    let v_ptr = kb.arg_ptr("V");
    let out_ptr = kb.arg_ptr("out");
    let scale = kb.arg_f32("scale");
    let head_dim_arg = kb.arg_u32("head_dim");
    let kv_len = kb.arg_u32("kv_len");
    let kv_stride = kb.arg_u32("kv_stride");

    let tid = kb.arange(0, wg_size);
    let pid = kb.program_id(0);

    let kv_h = pid.shr(&mut kb, gqa_ratio_log2);
    let kv_h_offset = kv_h.mul(&mut kb, head_dim_arg);
    let h_offset = pid.mul(&mut kb, head_dim_arg);

    let d_mask = tid.lt(&mut kb, head_dim_arg);

    let q_offset = h_offset.add(&mut kb, tid);
    let q_val = kb.load(q_ptr, q_offset, d_mask);

    let lds_base = kb.lds_alloc((wg_size + 2) * 4);

    let offset_max = kb.const_u32(wg_size);
    let offset_sum = kb.const_u32(wg_size + 1);
    let zero_u = kb.const_u32(0);
    let zero_f = kb.const_f32(0.0);
    let neg_inf = kb.const_f32(f32::NEG_INFINITY);
    let one_u = kb.const_u32(1);

    kb.lds_store(lds_base, tid, zero_f);
    kb.lds_store(lds_base, offset_max, neg_inf);
    kb.lds_store(lds_base, offset_sum, zero_f);
    kb.barrier();

    let iter = kb.for_range(zero_u, kv_len, 1);
    {
        let old_max = kb.lds_load(lds_base, offset_max);
        let old_sum = kb.lds_load(lds_base, offset_sum);
        let old_acc = kb.lds_load(lds_base, tid);

        let kv_row_base = iter.mul(&mut kb, kv_stride).add(&mut kb, kv_h_offset);
        let k_offset = kv_row_base.add(&mut kb, tid);
        let k_val = kb.load(k_ptr, k_offset, d_mask);

        let qk = q_val.mul(&mut kb, k_val);
        let dot = kb.wg_reduce_sum(qk);
        let score = dot.mul(&mut kb, scale);

        let v_offset = kv_row_base.add(&mut kb, tid);
        let v_val = kb.load(v_ptr, v_offset, d_mask);

        let new_max = old_max.max(&mut kb, score);
        let diff_old = old_max.sub(&mut kb, new_max);
        let correction = diff_old.exp(&mut kb);
        let diff_score = score.sub(&mut kb, new_max);
        let exp_score = diff_score.exp(&mut kb);

        let old_sum_scaled = old_sum.mul(&mut kb, correction);
        let new_sum = old_sum_scaled.add(&mut kb, exp_score);
        let weighted_v = exp_score.mul(&mut kb, v_val);
        let old_acc_scaled = old_acc.mul(&mut kb, correction);
        let new_acc = old_acc_scaled.add(&mut kb, weighted_v);

        kb.lds_store(lds_base, tid, new_acc);
        kb.lds_store(lds_base, offset_max, new_max);
        kb.lds_store(lds_base, offset_sum, new_sum);
        kb.barrier();
    }
    kb.end_for(iter);

    let final_sum = kb.lds_load(lds_base, offset_sum);
    let final_acc = kb.lds_load(lds_base, tid);
    let inv_sum = final_sum.rcp(&mut kb);
    let result = final_acc.mul(&mut kb, inv_sum);

    let out_offset = h_offset.add(&mut kb, tid);
    kb.store(out_ptr, out_offset, result, d_mask);

    kb
}

pub fn flash_attn_decode_grid(n_heads: u32, head_dim: u32) -> (u32, u32) {
    (n_heads * head_dim, 1)
}

pub fn cpu_flash_attn_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_len: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let gqa_ratio = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_heads * head_dim];

    for h in 0..n_heads {
        let kv_h = h / gqa_ratio;
        let q_base = h * head_dim;
        let kv_base = kv_h * head_dim;
        let kv_stride = n_kv_heads * head_dim;

        let mut max_val = f32::NEG_INFINITY;
        let mut sum_exp = 0.0f32;
        let mut acc = vec![0.0f32; head_dim];

        for kv in 0..kv_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[q_base + d] * k[kv * kv_stride + kv_base + d];
            }
            let score = dot * scale;

            let new_max = max_val.max(score);
            let correction = (max_val - new_max).exp();
            let exp_score = (score - new_max).exp();

            sum_exp = sum_exp * correction + exp_score;
            for d in 0..head_dim {
                acc[d] = acc[d] * correction + exp_score * v[kv * kv_stride + kv_base + d];
            }
            max_val = new_max;
        }

        for d in 0..head_dim {
            out[h * head_dim + d] = acc[d] / sum_exp;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ir::Target;

    #[test]
    fn test_flash_attn_decode_compiles() {
        let kb = build_flash_attn_decode(128, 1);
        let ck = kb.compile_via_ssa(Target::GFX1100).expect("flash_attn_decode compile");
        assert!(!ck.elf.is_empty());
        eprintln!("flash_attn_decode: {} bytes ELF, wg={:?}, lds={}",
            ck.elf.len(), ck.workgroup_size, ck.lds_size);
    }

    #[test]
    fn test_cpu_flash_attn_matches_standard() {
        let head_dim = 8;
        let n_heads = 4;
        let n_kv_heads = 2;
        let kv_len = 5;

        let mut rng_state = 42u64;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u64 << 31) as f32 - 1.0) * 0.5
        };

        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| rand()).collect();
        let k: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|_| rand()).collect();
        let v: Vec<f32> = (0..kv_len * n_kv_heads * head_dim).map(|_| rand()).collect();

        let flash_out = cpu_flash_attn_decode(&q, &k, &v, n_heads, n_kv_heads, head_dim, kv_len);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let gqa_ratio = n_heads / n_kv_heads;
        let mut std_out = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv_h = h / gqa_ratio;
            let mut scores = vec![0.0f32; kv_len];
            for kv in 0..kv_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[h * head_dim + d] * k[kv * n_kv_heads * head_dim + kv_h * head_dim + d];
                }
                scores[kv] = dot * scale;
            }
            let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut weights = vec![0.0f32; kv_len];
            let mut sum = 0.0f32;
            for kv in 0..kv_len {
                weights[kv] = (scores[kv] - max_val).exp();
                sum += weights[kv];
            }
            for kv in 0..kv_len {
                weights[kv] /= sum;
            }
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for kv in 0..kv_len {
                    val += weights[kv] * v[kv * n_kv_heads * head_dim + kv_h * head_dim + d];
                }
                std_out[h * head_dim + d] = val;
            }
        }

        let mut max_err = 0.0f32;
        for i in 0..n_heads * head_dim {
            let err = (flash_out[i] - std_out[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(max_err < 1e-5, "CPU flash vs standard: max_err={}", max_err);
    }
}
