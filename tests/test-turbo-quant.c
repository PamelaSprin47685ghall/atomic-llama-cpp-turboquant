#include "ggml-quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

extern int turbo3_cpu_wht_group_size;
extern void turbo_cpu_fwht_inverse(float * x, int group_size);

// These are storage-domain tests. Dequantization intentionally leaves the
// output WHT-rotated; invert once per head before comparing with model data.
static int check_case(enum ggml_type type, int group, int n, int kind, float scale) {
    float input[512], output[512];
    uint64_t storage[80], repeat[80]; // aligned buffers with untouched canaries
    unsigned char * packed = (unsigned char *) storage + 16;
    unsigned char * again = (unsigned char *) repeat + 16;
    const size_t bytes = ggml_row_size(type, n);
    memset(storage, 0xa5, sizeof(storage));
    memset(repeat, 0xa5, sizeof(repeat));
    uint32_t rng = 1234567;
    for (int i = 0; i < n; ++i) {
        // Sum uniforms for a deterministic approximately Gaussian population.
        float x = 0;
        for (int j = 0; j < 12; ++j) {
            rng = 1664525u * rng + 1013904223u;
            x += (float) (rng >> 8) / 16777216.0f - 0.5f;
        }
        input[i] = kind == 0 ? 0.0f : kind == 1 ? (i % group == 0 ? scale : 0.0f) :
            x * scale * (float) (1 + i / group);
    }
    for (int pass = 0; pass < 2; ++pass) {
        // A stale legacy override must not silently skip quantization now
        // that the storage block size is fixed at 128 (64/128 used to do so).
        turbo3_cpu_wht_group_size = pass == 0 ? 0 : 64;
        void * dst = pass == 0 ? packed : again;
        switch (type) {
            case GGML_TYPE_TURBO2_0: quantize_row_turbo2_0_ref(input, dst, n); break;
            case GGML_TYPE_TURBO3_0: quantize_row_turbo3_0_ref(input, dst, n); break;
            case GGML_TYPE_TURBO4_0: quantize_row_turbo4_0_ref(input, dst, n); break;
            default: return 1;
        }
    }
    if (memcmp(packed, again, bytes) != 0) {
        fprintf(stderr, "%s: non-deterministic encoding\n", ggml_type_name(type));
        return 1;
    }
    for (size_t i = 0; i < sizeof(storage); ++i) {
        if ((i < 16 || i >= 16 + bytes) && ((unsigned char *) storage)[i] != 0xa5) {
            fprintf(stderr, "%s: encoder wrote outside the destination\n", ggml_type_name(type));
            return 1;
        }
    }
    switch (type) {
        case GGML_TYPE_TURBO2_0: dequantize_row_turbo2_0((const void *) packed, output, n); break;
        case GGML_TYPE_TURBO3_0: dequantize_row_turbo3_0((const void *) packed, output, n); break;
        case GGML_TYPE_TURBO4_0: dequantize_row_turbo4_0((const void *) packed, output, n); break;
        default: return 1;
    }
    for (int off = 0; off < n; off += group) {
        turbo_cpu_fwht_inverse(output + off, group);
        double ni = 0, no = 0, dot = 0;
        for (int i = off; i < off + group; ++i) {
            if (!isfinite(output[i])) {
                fprintf(stderr, "%s: non-finite decoded value\n", ggml_type_name(type));
                return 1;
            }
            ni += (double) input[i] * (double) input[i];
            no += (double) output[i] * (double) output[i];
            dot += (double) input[i] * (double) output[i];
        }
        if (ni == 0) {
            if (no != 0) return 1;
            continue;
        }
        const double cosine = dot / sqrt(ni * no);
        const double floor = type == GGML_TYPE_TURBO2_0 ? 0.85 :
                             type == GGML_TYPE_TURBO3_0 ? 0.95 : 0.97;
        if (!isfinite(cosine) || cosine < floor || fabs(sqrt(no / ni) - 1.0) > 0.002) {
            fprintf(stderr, "%s group=%d n=%d kind=%d scale=%g: cosine=%g norm-ratio=%g\n",
                    ggml_type_name(type), group, n, kind, (double) scale, cosine, sqrt(no / ni));
            return 1;
        }
    }
#include "ggml.h"

#define CHECK_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        return 1; \
    } \
} while (0)

// Codec math: normalize -> orthogonal WHT (Parseval: MSE preserved across
// domains) -> Lloyd-Max scalar quant of ~N(0,var) rotated coefficients ->
// norm correction restores exact input norm (up to fp16). Expected per-element
// MSE ~= distortion * signal_variance with Lloyd-Max distortions
//   2-bit (4 levels): ~0.1175, 3-bit (8 levels): ~0.0345, 4-bit: ~0.0095.
// Bounds below are ~3x expected: they pass a correct codec with wide margin
// but fail a broken inverse-WHT / wrong-seed rotation (cosine ~ 0,
// MSE ~ 2*var) or a wrong-norm / wrong-stride layout (norm ratio far from 1).
static void stats_row(const float * ref, const float * got, int n,
                      float * mse_out, float * cos_out, float * norm_ratio_out) {
    double se = 0.0, dot = 0.0, nr = 0.0, ng = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = (double)ref[i] - (double)got[i];
        se += d * d;
        dot += (double)ref[i] * (double)got[i];
        nr += (double)ref[i] * (double)ref[i];
        ng += (double)got[i] * (double)got[i];
    }
    *mse_out = (float)(se / n);
    double denom = sqrt(nr * ng);
    *cos_out = denom > 1e-12 ? (float)(dot / denom) : 0.0f;
    *norm_ratio_out = nr > 1e-12 ? (float)sqrt(ng / nr) : 0.0f;
}

// Per-bit-width aggregate bounds for a given signal variance.
// Caller passes var so thresholds scale with signal energy; factors are ~3x
// the Lloyd-Max expectation, tight enough to catch rotation/norm breakage.
static void bounds_for(enum ggml_type type, float var,
                       float * mse_max, float * cos_min) {
    float dist = 0.1175f;
    if (type == GGML_TYPE_TURBO3_0) dist = 0.0345f;
    if (type == GGML_TYPE_TURBO4_0) dist = 0.0095f;
    // 3x margin on MSE; cosine floor derived from relative error rel=MSE/var:
    // cos ~= 1 - rel/2, minus slack for tails/fp16 norm.
    *mse_max = 3.0f * dist * var + 1e-6f;
    float rel = *mse_max / (var > 1e-12f ? var : 1.0f);
    *cos_min = 1.0f - rel * 0.5f - 0.06f;
    // Clamp floors so the bound never goes vacuous for large var.
    float floor = 0.80f;
    if (type == GGML_TYPE_TURBO3_0) floor = 0.90f;
    if (type == GGML_TYPE_TURBO4_0) floor = 0.93f;
    if (*cos_min < floor) *cos_min = floor;
}

int main(void) {
    const int d = 128;
    float input[128];
    float canon_out[128];
    float rot_out[128];
    enum ggml_type types[3] = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };

    // 1. Invalid argument, malformed length, and malformed group rejection
    {
        char buf[512];
        CHECK_TRUE(!ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, NULL, buf, d, 128), "null src rejected");
        CHECK_TRUE(!ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, input, NULL, d, 128), "null dst rejected");
        CHECK_TRUE(!ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, input, buf, 64, 128), "non-128 n_elements rejected");
        CHECK_TRUE(!ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, input, buf, 0, 128), "zero n_elements rejected");
        CHECK_TRUE(!ggml_quantize_turbo_row(GGML_TYPE_TURBO4_0, input, buf, 96, 128), "96 n_elements rejected");
        CHECK_TRUE(!ggml_quantize_turbo_row(GGML_TYPE_F32, input, buf, d, 128), "non-turbo type rejected");
        CHECK_TRUE(!ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, NULL, canon_out, d, 128, GGML_TURBO_DECODE_CANONICAL), "null src rejected");
        CHECK_TRUE(!ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, buf, NULL, d, 128, GGML_TURBO_DECODE_CANONICAL), "null dst rejected");
        CHECK_TRUE(!ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, buf, canon_out, 64, 128, GGML_TURBO_DECODE_CANONICAL), "non-128 n_elements rejected");
        CHECK_TRUE(!ggml_dequantize_turbo_row(GGML_TYPE_F32, buf, canon_out, d, 128, GGML_TURBO_DECODE_CANONICAL), "non-turbo type rejected");
        CHECK_TRUE(!ggml_dequantize_turbo_row(GGML_TYPE_TURBO4_0, buf, canon_out, d, 128, (enum ggml_turbo_decode_domain)99), "bad domain rejected");

        // Malformed group sizes rejected for every Turbo type on both APIs
        const int bad_groups[] = { 0, 32, 64, 96, 256 };
        for (int t = 0; t < 3; ++t) {
            for (unsigned gi = 0; gi < sizeof(bad_groups)/sizeof(bad_groups[0]); ++gi) {
                int g = bad_groups[gi];
                CHECK_TRUE(!ggml_quantize_turbo_row(types[t], input, buf, d, g), "bad quant group rejected");
                CHECK_TRUE(!ggml_dequantize_turbo_row(types[t], buf, canon_out, d, g, GGML_TURBO_DECODE_CANONICAL), "bad dequant group rejected");
                CHECK_TRUE(!ggml_dequantize_turbo_row(types[t], buf, canon_out, d, g, GGML_TURBO_DECODE_ROTATED), "bad dequant group rejected");
            }
            // 256/384-element rows also reject non-128 groups
            CHECK_TRUE(!ggml_quantize_turbo_row(types[t], input, buf, d, 64), "bad group rejected at 128");
        }
        float big_in[384];
        CHECK_TRUE(!ggml_quantize_turbo_row(GGML_TYPE_TURBO2_0, big_in, buf, 384, 64), "bad group rejected at 384");
        CHECK_TRUE(!ggml_dequantize_turbo_row(GGML_TYPE_TURBO2_0, buf, canon_out, 384, 64, GGML_TURBO_DECODE_CANONICAL), "bad dequant group at 384");
    }

    // Compiled layout fingerprint must be nonzero for supported Turbo types
    CHECK_TRUE(ggml_turbo_layout_fingerprint(GGML_TYPE_TURBO2_0) != 0, "turbo2 fingerprint valid");
    CHECK_TRUE(ggml_turbo_layout_fingerprint(GGML_TYPE_TURBO3_0) != 0, "turbo3 fingerprint valid");
    CHECK_TRUE(ggml_turbo_layout_fingerprint(GGML_TYPE_TURBO4_0) != 0, "turbo4 fingerprint valid");
    CHECK_TRUE(ggml_turbo_layout_fingerprint(GGML_TYPE_F32) == 0, "non-turbo fingerprint 0");

    // 2. Deterministic bytes, WHT-domain relation, finite output, bit-width
    // accuracy + ordering on a sinusoidal signal (var ~= 50 for ampl 10).
    {
        float mse_of[3], cos_of[3];
        for (int t = 0; t < 3; ++t) {
            enum ggml_type type = types[t];
            size_t row_bytes = ggml_row_size(type, d);
            CHECK_TRUE(row_bytes > 0, "ggml_row_size > 0");
            uint8_t * enc_a = (uint8_t *)malloc(row_bytes);
            uint8_t * enc_b = (uint8_t *)malloc(row_bytes);
            CHECK_TRUE(enc_a && enc_b, "malloc enc");

            // Pattern 1: unit basis vector (WHT-domain checks + finiteness)
            memset(input, 0, sizeof(input));
            input[0] = 1.0f;
            CHECK_TRUE(ggml_quantize_turbo_row(type, input, enc_a, d, 128), "quantize basis");
            CHECK_TRUE(ggml_quantize_turbo_row(type, input, enc_b, d, 128), "quantize basis again");
            CHECK_TRUE(memcmp(enc_a, enc_b, row_bytes) == 0, "deterministic bytes");
            CHECK_TRUE(ggml_dequantize_turbo_row(type, enc_a, canon_out, d, 128, GGML_TURBO_DECODE_CANONICAL), "dequant canon");
            CHECK_TRUE(ggml_dequantize_turbo_row(type, enc_a, rot_out, d, 128, GGML_TURBO_DECODE_ROTATED), "dequant rot");
            for (int i = 0; i < d; ++i) {
                CHECK_TRUE(isfinite(canon_out[i]), "canon_out is finite");
                CHECK_TRUE(isfinite(rot_out[i]), "rot_out is finite");
            }
            // ROTATED vs CANONICAL: forward WHT on CANONICAL must equal ROTATED
            float check_rot[128];
            memcpy(check_rot, canon_out, sizeof(check_rot));
            ggml_turbo_wht_row(check_rot, 128);
            for (int i = 0; i < d; ++i) {
                CHECK_TRUE(fabsf(check_rot[i] - rot_out[i]) < 1e-4f, "wht_row(canon) == rot");
            }
            // Inverse WHT on ROTATED must equal CANONICAL
            float check_canon[128];
            memcpy(check_canon, rot_out, sizeof(check_canon));
            ggml_turbo_wht_inverse_row(check_canon, 128);
            for (int i = 0; i < d; ++i) {
                CHECK_TRUE(fabsf(check_canon[i] - canon_out[i]) < 1e-4f, "wht_inverse_row(rot) == canon");
            }

            // Pattern 2: sinusoidal signal, aggregate (never per-element) bounds
            float norm_in_sq = 0.0f;
            for (int i = 0; i < d; ++i) {
                input[i] = sinf(i * 0.1f + 0.5f) * 10.0f;
                norm_in_sq += input[i] * input[i];
            }
            float sig_var = norm_in_sq / d;
            CHECK_TRUE(ggml_quantize_turbo_row(type, input, enc_a, d, 128), "quantize sin");
            CHECK_TRUE(ggml_dequantize_turbo_row(type, enc_a, canon_out, d, 128, GGML_TURBO_DECODE_CANONICAL), "dequant canon sin");
            for (int i = 0; i < d; ++i) CHECK_TRUE(isfinite(canon_out[i]), "sin decoded finite");

            float mse, cos, nratio;
            stats_row(input, canon_out, d, &mse, &cos, &nratio);
            float mse_max, cos_min;
            bounds_for(type, sig_var, &mse_max, &cos_min);
            CHECK_TRUE(mse < mse_max, "sin mse within Lloyd-Max bound");
            CHECK_TRUE(cos > cos_min, "sin cosine within Lloyd-Max bound");
            CHECK_TRUE(nratio > 0.90f && nratio < 1.10f, "norm correction restores energy");
            mse_of[t] = mse;
            cos_of[t] = cos;

            free(enc_a);
            free(enc_b);
        }
        // Meaningful bit-width ordering on the shared input: more bits win.
        CHECK_TRUE(mse_of[2] <= mse_of[1] + 1e-6f, "t4 mse <= t3 mse");
        CHECK_TRUE(mse_of[1] <= mse_of[0] + 1e-6f, "t3 mse <= t2 mse");
        CHECK_TRUE(cos_of[2] + 1e-6f >= cos_of[1], "t4 cosine >= t3 cosine");
        CHECK_TRUE(cos_of[1] + 1e-6f >= cos_of[0], "t3 cosine >= t2 cosine");
    }

    // 3. Misaligned (buf+1) roundtrip at 128/256/384 with aggregate bounds.
    // Per-element |err|<1 thresholds are invalid here: WHT spreads energy so
    // Gaussian tails legitimately exceed 1.0 even for a correct codec.
    {
        const int lengths[] = { 128, 256, 384 };
        for (unsigned li = 0; li < 3; ++li) {
            int n = lengths[li];
            float * in = (float *)malloc(n * sizeof(float));
            float * out = (float *)malloc(n * sizeof(float));
            CHECK_TRUE(in && out, "malloc misaligned io");
            float var_acc = 0.0f;
            for (int i = 0; i < n; ++i) {
                in[i] = (float)(i % 7) - 3.0f; // uniform [-3,3], var = 4
                var_acc += in[i] * in[i];
            }
            float sig_var = var_acc / n;
            for (int t = 0; t < 3; ++t) {
                enum ggml_type type = types[t];
                size_t row_bytes = ggml_row_size(type, n);
                uint8_t * aligned = (uint8_t *)malloc(row_bytes);
                uint8_t * raw = (uint8_t *)malloc(row_bytes + 1);
                CHECK_TRUE(aligned && raw, "malloc misaligned bufs");
                uint8_t * mis = raw + 1; // deliberately odd/misaligned
                CHECK_TRUE(ggml_quantize_turbo_row(type, in, aligned, n, 128), "quantize aligned");
                CHECK_TRUE(ggml_quantize_turbo_row(type, in, mis, n, 128), "quantize misaligned buf+1");
                CHECK_TRUE(memcmp(aligned, mis, row_bytes) == 0, "misaligned bytes identical");
                CHECK_TRUE(ggml_dequantize_turbo_row(type, mis, out, n, 128, GGML_TURBO_DECODE_CANONICAL), "dequantize misaligned buf+1");
                for (int i = 0; i < n; ++i) CHECK_TRUE(isfinite(out[i]), "misaligned decoded finite");
                float mse, cos, nratio;
                stats_row(in, out, n, &mse, &cos, &nratio);
                float mse_max, cos_min;
                bounds_for(type, sig_var, &mse_max, &cos_min);
                CHECK_TRUE(mse < mse_max, "misaligned mse bounded");
                CHECK_TRUE(cos > cos_min, "misaligned cosine bounded");
                CHECK_TRUE(nratio > 0.85f && nratio < 1.15f, "misaligned norm restored");
                free(aligned);
                free(raw);
            }
            free(in);
            free(out);
        }
    }

    // 4. Exact ggml_row_size strides with pre/post guards at 128/256/384, plus
    // proof that dequant reads exactly row_bytes (canary independence).
    {
        const int test_lengths[] = { 128, 256, 384 };
        const uint8_t GUARD_PRE = 0xAA;
        const uint8_t GUARD_POST = 0x55;
        const size_t PRE = 16, POST = 16;
        for (unsigned l = 0; l < 3; ++l) {
            int n_elem = test_lengths[l];
            float * long_input = (float *)malloc(n_elem * sizeof(float));
            float * out_a = (float *)malloc(n_elem * sizeof(float));
            float * out_b = (float *)malloc(n_elem * sizeof(float));
            CHECK_TRUE(long_input && out_a && out_b, "malloc inputs");
            for (int i = 0; i < n_elem; ++i) long_input[i] = sinf(i * 0.05f) * 2.0f;

            for (int t = 0; t < 3; ++t) {
                enum ggml_type type = types[t];
                size_t row_bytes = ggml_row_size(type, n_elem);
                CHECK_TRUE(row_bytes == (size_t)(n_elem / 128) * ggml_row_size(type, 128), "stride is row_size-exact");

                // +1 keeps the target at an odd (misaligned) address while both
                // guards stay fully outside [target, target+row_bytes).
                size_t total = PRE + 1 + row_bytes + POST;
                uint8_t * base = (uint8_t *)malloc(total);
                CHECK_TRUE(base != NULL, "malloc guard_buf");
                memset(base, GUARD_PRE, PRE + 1);
                memset(base + PRE + 1 + row_bytes, GUARD_POST, POST);
                uint8_t * target = base + PRE + 1;

                CHECK_TRUE(ggml_quantize_turbo_row(type, long_input, target, n_elem, 128), "quantize guarded row");
                for (size_t b = 0; b < PRE + 1; ++b) CHECK_TRUE(base[b] == GUARD_PRE, "pre-guard untouched by quantize");
                for (size_t b = PRE + 1 + row_bytes; b < total; ++b) CHECK_TRUE(base[b] == GUARD_POST, "post-guard untouched by quantize");

                // Deterministic bytes: a second quantize into a fresh guarded
                // buffer must match exactly.
                uint8_t * base2 = (uint8_t *)malloc(total);
                CHECK_TRUE(base2 != NULL, "malloc guard_buf2");
                memset(base2, GUARD_PRE, PRE + 1);
                memset(base2 + PRE + 1 + row_bytes, GUARD_POST, POST);
                CHECK_TRUE(ggml_quantize_turbo_row(type, long_input, base2 + PRE + 1, n_elem, 128), "quantize guarded again");
                CHECK_TRUE(memcmp(target, base2 + PRE + 1, row_bytes) == 0, "guarded bytes deterministic");
                free(base2);

                CHECK_TRUE(ggml_dequantize_turbo_row(type, target, out_a, n_elem, 128, GGML_TURBO_DECODE_CANONICAL), "dequantize guarded row");
                for (int i = 0; i < n_elem; ++i) CHECK_TRUE(isfinite(out_a[i]), "guarded decoded finite");
                for (size_t b = 0; b < PRE + 1; ++b) CHECK_TRUE(base[b] == GUARD_PRE, "pre-guard untouched by dequant");
                for (size_t b = PRE + 1 + row_bytes; b < total; ++b) CHECK_TRUE(base[b] == GUARD_POST, "post-guard untouched by dequant");

                // Read-independence: mutating every guard byte must not change
                // the decoded floats, proving no read past row_bytes.
                memset(base, 0xE5, PRE + 1);
                memset(base + PRE + 1 + row_bytes, 0x1E, POST);
                CHECK_TRUE(ggml_dequantize_turbo_row(type, target, out_b, n_elem, 128, GGML_TURBO_DECODE_CANONICAL), "dequantize with flipped canaries");
                CHECK_TRUE(memcmp(out_a, out_b, n_elem * sizeof(float)) == 0, "decode independent of guard bytes");

                free(base);
            }
            free(long_input);
            free(out_a);
            free(out_b);
        }
    }

    printf("PASS: test-turbo-quant verified public row APIs, ROTATED/CANONICAL domains, and bounds.\n");
    return 0;
}

int main(void) {
    // Initialize the FP16 lookup tables used by the reference dequantizers.
    struct ggml_context * ctx = ggml_init((struct ggml_init_params) { 1024 * 1024, NULL, true });
    if (!ctx) return 1;
    const enum ggml_type types[] = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0 };
    const float scales[] = { 0.001f, 1.0f, 10.0f };
    int failures = 0, cases = 0;
    for (int t = 0; t < 3; ++t) {
        {
            const int group = 128;
            for (int groups = 1; groups <= 4; groups *= 2) {
                for (int kind = 0; kind < 3; ++kind) {
                    for (int s = 0; s < 3; ++s) {
                        failures += check_case(types[t], group, group * groups, kind, scales[s]);
                        ++cases;
                    }
                }
            }
        }
    }
    turbo3_cpu_wht_group_size = 0;
    ggml_free(ctx);
    printf("TurboQuant round-trip: %d cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
