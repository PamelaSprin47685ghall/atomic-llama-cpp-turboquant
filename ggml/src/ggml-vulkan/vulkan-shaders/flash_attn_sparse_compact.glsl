#if !defined(FLASH_ATTN_SPARSE_COMPACT_GLSL)
#    define FLASH_ATTN_SPARSE_COMPACT_GLSL

struct CompactionParams {
    uint KV;
    uint N;
    uint nem2;
    uint nem3;
    uint nbm1;
    uint nbm2;
    uint nbm3;
    uint n_kv_max;
    uint gqa_ratio;
};

shared uint s_active_count;
shared uint s_overflow;
shared uint s_subgroup_counts[512];

void run_sparse_compact(uint q, uint iq2, uint iq3, CompactionParams cp) {
    uint tid = gl_LocalInvocationIndex;
    if (tid == 0u) {
        s_active_count = 0u;
        s_overflow     = 0u;
    }
    barrier();

    if (q >= cp.N || iq2 >= cp.nem2 || iq3 >= cp.nem3) {
        return;
    }

    uint m_row_offset = q * cp.nbm1 + iq2 * cp.nbm2 + iq3 * cp.nbm3;
    uint num_rows     = cp.N * cp.nem2 * cp.nem3;
    uint row_idx      = (iq3 * cp.nem2 + iq2) * cp.N + q;

    // Metadata ABI (local://tp5-sparse-contract.txt):
    // word 0 = n_kv_max
    // words 1..R = per-row active counts (-1 for overflow/fallback)
    // H = align_up(1 + R, 4)
    // pairs for row r at H + 2 * n_kv_max * r contain (original KV index, floatBitsToInt(mask value))
    uint header_size    = ((1u + num_rows + 3u) / 4u) * 4u;
    uint q_entry_offset = header_size + row_idx * (cp.n_kv_max * 2u);

    if (q == 0u && iq2 == 0u && iq3 == 0u && tid == 0u) {
        data_sm[0] = int(cp.n_kv_max);
    }

    for (uint tile = 0u; tile < cp.KV; tile += gl_WorkGroupSize.x) {
        uint  k         = tile + tid;
        float m         = k < cp.KV ? float(data_m[m_row_offset + k]) : -1.0 / 0.0;
        // Select ALL finite mask values including -65504, skip negative infinity only.
        // NaN or positive infinity must trigger correct dense fallback rather than disappear.
        bool  is_finite = !isinf(m) && !isnan(m);
        if (isnan(m) || (isinf(m) && m > 0.0f)) {
            atomicOr(s_overflow, 1u);
        }

        uvec4 b     = subgroupBallot(is_finite);
        uint  count = subgroupBallotBitCount(b);
        uint  ex    = subgroupBallotExclusiveBitCount(b);

        if (subgroupElect()) {
            s_subgroup_counts[gl_SubgroupID] = count;
        }
        barrier();

        // Canonical KV order must not depend on which subgroup wins an atomic.
        // Different mask allocations otherwise change attention's sum order
        // once more than one subgroup has visible keys.
        uint sg_base    = s_active_count;
        uint tile_count = 0u;
        for (uint sg = 0u; sg < gl_NumSubgroups; ++sg) {
            uint n = s_subgroup_counts[sg];
            if (sg < gl_SubgroupID) {
                sg_base += n;
            }
            tile_count += n;
        }

        if (is_finite) {
            uint slot = sg_base + ex;
            if (slot < cp.n_kv_max) {
                data_sm[q_entry_offset + 2u * slot + 0u] = int(k);
                data_sm[q_entry_offset + 2u * slot + 1u] = floatBitsToInt(m);
            } else {
                atomicOr(s_overflow, 1u);
            }
        }
        // All old subgroup counts and the tile base have been consumed before
        // the next iteration overwrites them. Its first barrier publishes the
        // updated base; the final barrier below covers the last tile.
        barrier();
        if (tid == 0u) {
            s_active_count += tile_count;
        }
    }

    barrier();

    if (tid == 0u) {
        uint total = s_active_count;
        if (s_overflow != 0u || total > cp.n_kv_max) {
            // Signal overflow so downstream FA falls back to dense execution safely
            data_sm[1u + row_idx] = -1;
        } else {
            data_sm[1u + row_idx] = int(total);
        }
    }
}

#endif  // FLASH_ATTN_SPARSE_COMPACT_GLSL
