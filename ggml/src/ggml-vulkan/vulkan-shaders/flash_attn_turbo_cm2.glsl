// Cooperative-matrix decoding must return the same stored WHT domain as
// flash_attn_dequant.glsl. The graph rotates Q and inverse-rotates the result.
// Buffer-reference alignment is 2, not 16: Turbo3 blocks are 50 bytes.
layout(buffer_reference, std430, buffer_reference_align = 2) buffer faTurbo2Buf {
    block_turbo2_0 block;
};
layout(buffer_reference, std430, buffer_reference_align = 2) buffer faTurbo3Buf {
    block_turbo3_0 block;
};
layout(buffer_reference, std430, buffer_reference_align = 2) buffer faTurbo4Buf {
    block_turbo4_0 block;
};

float16_t faTurbo2(faTurbo2Buf b, uint i) {
    const float c[4] = float[4](-0.133462, -0.039994, 0.039994, 0.133462);
    uint q = (uint(b.block.qs[i / 4u]) >> (2u * (i % 4u))) & 3u;
    return float16_t(float(b.block.norm) * c[q]);
}

float16_t faTurbo3(faTurbo3Buf b, uint i) {
    const float c[8] = float[8](-0.190685, -0.117832, -0.065717, -0.021460,
                               0.021460,  0.065717,  0.117832,  0.190685);
    uint low = (uint(b.block.qs[i / 4u]) >> (2u * (i % 4u))) & 3u;
    uint high = (uint(b.block.signs[i / 8u]) >> (i % 8u)) & 1u;
    return float16_t(float(b.block.norm) * c[low | (high << 2u)]);
}

float16_t faTurbo4(faTurbo4Buf b, uint i) {
    const float c[16] = float[16](
        -0.173926, -0.117195, -0.089527, -0.068756,
        -0.051262, -0.035597, -0.020989, -0.006938,
         0.006938,  0.020989,  0.035597,  0.051262,
         0.068756,  0.089527,  0.117195,  0.173926);
    uint q = (uint(b.block.qs[i / 2u]) >> (4u * (i % 2u))) & 15u;
    return float16_t(float(b.block.norm) * c[q]);
}

f16vec4 faTurbo2Vector(faTurbo2Buf b, uint i) {
    return f16vec4(faTurbo2(b, i), faTurbo2(b, i + 1u), faTurbo2(b, i + 2u), faTurbo2(b, i + 3u));
}
f16vec4 faTurbo3Vector(faTurbo3Buf b, uint i) {
    return f16vec4(faTurbo3(b, i), faTurbo3(b, i + 1u), faTurbo3(b, i + 2u), faTurbo3(b, i + 3u));
}
f16vec4 faTurbo4Vector(faTurbo4Buf b, uint i) {
    return f16vec4(faTurbo4(b, i), faTurbo4(b, i + 1u), faTurbo4(b, i + 2u), faTurbo4(b, i + 3u));
}
