#include "types.glsl"

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8
#define MAT_VEC_FUSION_FLAGS_TP5_ELIDE_LOCAL 0x100

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPEV4)
layout (binding = 0) readonly buffer AV4 {A_TYPEV4 data_a_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif

layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
#ifdef B_TYPEV2
layout (binding = 1) readonly buffer BV2 {B_TYPEV2 data_b_v2[];};
#endif
#ifdef B_TYPEV4
layout (binding = 1) readonly buffer BV4 {B_TYPEV4 data_b_v4[];};
#endif

layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (binding = 3) readonly buffer Fuse0 {D_TYPE data_fuse0[];};
#ifdef MOE_SHARED_UP_SWIGLU
layout(binding = 3) readonly buffer Gate {
    A_TYPE data_gate[];
};

layout(binding = 3) readonly buffer GatePacked {
    A_TYPE_PACKED16 data_gate_packed16[];
};
#endif
layout (binding = 4) readonly buffer Fuse1 {D_TYPE data_fuse1[];};

#ifdef TP5_RELAY_OUTPUT
#    ifdef TP5_RELAY_OUTPUT_F32
    #define TP5_RELAY_PAYLOAD_TYPE float
#    else
    #define TP5_RELAY_PAYLOAD_TYPE float16_t
#    endif
#endif

#ifdef MUL_MAT_ID
#    if defined(TP5_WIRE_OUTPUT) || defined(TP5_RELAY_OUTPUT)
#        ifdef MOE_DOWN_FOLD
#            ifdef MOE_FUSE_SHARED_DOWN
#                ifdef TP5_RELAY_OUTPUT
layout(std430, binding = 9) coherent readonly buffer TP5RelayRoute {
    uint bank;
    uint ready;
    uint epoch;
    uint reserved;
} tp5_relay_route;
layout(std430, binding = 10) coherent writeonly buffer TP5RelayPayload0 {
    TP5_RELAY_PAYLOAD_TYPE data_payload0[];
};
layout(std430, binding = 11) coherent writeonly buffer TP5RelayPayload1 {
    TP5_RELAY_PAYLOAD_TYPE data_payload1[];
};
#                else
layout(binding = 9) writeonly buffer WireOut {
    float16_t data_wire[];
};
#                endif
#            else
#                ifdef TP5_RELAY_OUTPUT
layout(std430, binding = 8) coherent readonly buffer TP5RelayRoute {
    uint bank;
    uint ready;
    uint epoch;
    uint reserved;
} tp5_relay_route;
layout(std430, binding = 9) coherent writeonly buffer TP5RelayPayload0 {
    TP5_RELAY_PAYLOAD_TYPE data_payload0[];
};
layout(std430, binding = 10) coherent writeonly buffer TP5RelayPayload1 {
    TP5_RELAY_PAYLOAD_TYPE data_payload1[];
};
#                else
layout(binding = 8) writeonly buffer WireOut {
    float16_t data_wire[];
};
#                endif
#            endif
#        endif
#    endif
#elif defined(TP5_RELAY_OUTPUT)
layout(std430, binding = 5) coherent readonly buffer TP5RelayRoute {
    uint bank;
    uint ready;
    uint epoch;
    uint reserved;
} tp5_relay_route;
layout(std430, binding = 6) coherent writeonly buffer TP5RelayPayload0 {
    TP5_RELAY_PAYLOAD_TYPE data_payload0[];
};
layout(std430, binding = 7) coherent writeonly buffer TP5RelayPayload1 {
    TP5_RELAY_PAYLOAD_TYPE data_payload1[];
};
#elif defined(TP5_WIRE_OUTPUT)
layout(binding = 5) writeonly buffer WireOut {
    float16_t data_wire[];
};
#endif

void tp5_write_wire(const uint index, const float value) {
#ifdef TP5_RELAY_OUTPUT
#    ifdef TP5_RELAY_OUTPUT_F32
    if (tp5_relay_route.bank == 0u)
        data_payload0[index] = value;
    else
        data_payload1[index] = value;
#    else
    if (tp5_relay_route.bank == 0u)
        data_payload0[index] = float16_t(value);
    else
        data_payload1[index] = float16_t(value);
#    endif
#elif defined(TP5_WIRE_OUTPUT)
    data_wire[index] = float16_t(value);
#endif
}

#ifdef TP5_RELAY_OUTPUT
// The scalar aliases above retain tail support. Full row tiles can store a
// vector without adding a packing dispatch or cross-workgroup synchronization.
#if defined(MUL_MAT_ID) && defined(MOE_FUSE_SHARED_DOWN)
#define TP5_PAYLOAD0_BINDING 10
#define TP5_PAYLOAD1_BINDING 11
#elif defined(MUL_MAT_ID)
#define TP5_PAYLOAD0_BINDING 9
#define TP5_PAYLOAD1_BINDING 10
#else
#define TP5_PAYLOAD0_BINDING 6
#define TP5_PAYLOAD1_BINDING 7
#endif
#ifdef TP5_RELAY_OUTPUT_F32
#define TP5_PAYLOAD_VEC4 vec4
#define TP5_PAYLOAD_VEC2 vec2
#else
#define TP5_PAYLOAD_VEC4 f16vec4
#define TP5_PAYLOAD_VEC2 f16vec2
#endif
layout(std430, binding = TP5_PAYLOAD0_BINDING) coherent writeonly buffer TP5Payload0V4 {
    TP5_PAYLOAD_VEC4 data_payload0_v4[];
};
layout(std430, binding = TP5_PAYLOAD1_BINDING) coherent writeonly buffer TP5Payload1V4 {
    TP5_PAYLOAD_VEC4 data_payload1_v4[];
};
layout(std430, binding = TP5_PAYLOAD0_BINDING) coherent writeonly buffer TP5Payload0V2 {
    TP5_PAYLOAD_VEC2 data_payload0_v2[];
};
layout(std430, binding = TP5_PAYLOAD1_BINDING) coherent writeonly buffer TP5Payload1V2 {
    TP5_PAYLOAD_VEC2 data_payload1_v2[];
};

void tp5_write_wire2(const uint index, const vec2 value) {
    if ((index & 1u) != 0u) {
        tp5_write_wire(index, value.x);
        tp5_write_wire(index + 1u, value.y);
    } else if (tp5_relay_route.bank == 0u) {
        data_payload0_v2[index / 2u] = TP5_PAYLOAD_VEC2(value);
    } else {
        data_payload1_v2[index / 2u] = TP5_PAYLOAD_VEC2(value);
    }
}

void tp5_write_wire4(const uint index, const vec4 value) {
    if ((index & 3u) != 0u) {
        for (uint j = 0u; j < 4u; ++j) tp5_write_wire(index + j, value[j]);
    } else if (tp5_relay_route.bank == 0u) {
        data_payload0_v4[index / 4u] = TP5_PAYLOAD_VEC4(value);
    } else {
        data_payload1_v4[index / 4u] = TP5_PAYLOAD_VEC4(value);
    }
}
#elif defined(TP5_WIRE_OUTPUT)
// Producer-local F16 companions use the same packetized epilogue as direct
// output, without inheriting host-coherence semantics or dropping local F32.
#if defined(MUL_MAT_ID) && defined(MOE_FUSE_SHARED_DOWN)
#define TP5_LOCAL_WIRE_BINDING 9
#elif defined(MUL_MAT_ID)
#define TP5_LOCAL_WIRE_BINDING 8
#else
#define TP5_LOCAL_WIRE_BINDING 5
#endif
layout(std430, binding = TP5_LOCAL_WIRE_BINDING) writeonly buffer TP5LocalWireV4 {
    f16vec4 data_wire_v4[];
};
layout(std430, binding = TP5_LOCAL_WIRE_BINDING) writeonly buffer TP5LocalWireV2 {
    f16vec2 data_wire_v2[];
};

void tp5_write_wire2(const uint index, const vec2 value) {
    if ((index & 1u) != 0u) {
        tp5_write_wire(index, value.x);
        tp5_write_wire(index + 1u, value.y);
    } else {
        data_wire_v2[index / 2u] = f16vec2(value);
    }
}

void tp5_write_wire4(const uint index, const vec4 value) {
    if ((index & 3u) != 0u) {
        tp5_write_wire2(index, value.xy);
        tp5_write_wire2(index + 2u, value.zw);
    } else {
        data_wire_v4[index / 4u] = f16vec4(value);
    }
}
#endif

bool tp5_keep_local_output(const uint fusion_flags) {
#ifdef TP5_RELAY_OUTPUT
    // Permission is proved for the actual graph and recorded in this dispatch.
    // A dynamic route may require z_p (LateBind or fallback), never authorize
    // discarding a value that another fused/unfused graph node still reads.
    return (fusion_flags & MAT_VEC_FUSION_FLAGS_TP5_ELIDE_LOCAL) == 0u ||
           (tp5_relay_route.reserved & 1u) != 0u;
#else
    return true;
#endif
}

#ifdef MUL_MAT_ID
layout (binding = 5) readonly buffer IDS {int data_ids[];};
#endif
#ifdef MUL_MAT_ID_GROUPED
layout (binding = 6) readonly buffer EXPERT_COUNTS {uint data_expert_count[];};
#endif
#ifdef MOE_DOWN_FOLD
#    ifdef MOE_FUSE_SHARED_DOWN
layout(binding = 6) readonly buffer SharedDownWeights {
    block_q8_0 data_shared_down[];
};

layout(binding = 6) readonly buffer SharedDownWeightsPacked {
    block_q8_0_packed16 data_shared_down_packed16[];
};

layout(binding = 7) readonly buffer SharedGate {
    float shared_gate[];
};

layout(binding = 8) readonly buffer SharedSwiglu {
    float shared_swiglu[];
};
#    else
layout(binding = 6) readonly buffer SharedDown {
    float shared_down[];
};

layout(binding = 7) readonly buffer SharedGate {
    float shared_gate[];
};
#    endif
#endif
