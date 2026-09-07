#include "ggml-quants.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
