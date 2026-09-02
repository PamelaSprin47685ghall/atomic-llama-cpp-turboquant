#ifdef MUL_MAT_ID
shared u16vec2 row_ids[BN];
uint _ne1;

uint expert_count(uint expert_idx) {
    return uint(data_expert_count[1 + p.active_capacity + expert_idx]);
}

void load_row_ids(uint expert_idx, bool /*nei0_is_pow2*/, uint ic) {
    _ne1 = expert_count(expert_idx);

    const uint row_begin = ic * BN;
    const uint n_rows = min(BN, _ne1 - row_begin);
    const uint expert_stride = p.nei0 * p.nei1;

    for (uint i = gl_LocalInvocationIndex; i < n_rows; i += gl_WorkGroupSize.x) {
        const uint packed = uint(data_ids[expert_idx*expert_stride + row_begin + i]);
        row_ids[i] = u16vec2(packed & 0xffffu, packed >> 16);
    }
    barrier();
}
#endif // MUL_MAT_ID
