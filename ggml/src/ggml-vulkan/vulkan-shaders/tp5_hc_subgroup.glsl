#ifndef TP5_HC_SUBGROUP_GLSL
#define TP5_HC_SUBGROUP_GLSL
// A 32-thread workgroup can still contain several subgroups on another
// device. Keep the one-subgroup path identical to native HC, but do not lose
// partial sums when subgroupSize is smaller than the workgroup size.
shared float tp5_hc_subgroup_partials[32];
float tp5_hc_sum32(float value) {
    float sum = subgroupAdd(value);
    if (gl_NumSubgroups == 1u) return sum;
    if (gl_SubgroupInvocationID == 0u)
        tp5_hc_subgroup_partials[gl_SubgroupID] = sum;
    barrier();
    float result = 0.0;
    if (gl_SubgroupID == 0u) {
        // A very small subgroup may have fewer lanes than the number of
        // partials. Walk all of them, then reduce in that first subgroup.
        for (uint i = gl_SubgroupInvocationID; i < gl_NumSubgroups; i += gl_SubgroupSize)
            result += tp5_hc_subgroup_partials[i];
        result = subgroupAdd(result);
    }
    barrier(); // partial storage can be reused for the next stream only now
    return result; // only workgroup lane zero commits the result
}
#endif
