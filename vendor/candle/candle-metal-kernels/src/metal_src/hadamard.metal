// Fused sign flip + blockwise normalized natural-order (Sylvester) fast
// Walsh-Hadamard transform over contiguous f32 rows, used by the
// prism_hadamard_qwen35 activation rotation in bonsai-candle.
//
// One threadgroup transforms one `block`-sized chunk in threadgroup memory:
// load (optionally * signs), log2(block) butterfly stages with strides
// 1, 2, 4, ..., then store (* scale, optionally * signs).
//   signs_first = true:  y = ((x * s) @ H) * scale   (forward)
//   signs_first = false: y = (x @ H) * scale * s     (inverse)
// `signs` has `blocks_per_row * block` entries and is indexed by the chunk's
// position within its row, so chunk `c` uses signs[(c % blocks_per_row) * block ..].
#include <metal_stdlib>
using namespace metal;

kernel void hadamard_block_fwht_f32(
    device const float *src [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device float *dst [[buffer(2)]],
    constant uint &block [[buffer(3)]],
    constant uint &blocks_per_row [[buffer(4)]],
    constant bool &signs_first [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    threadgroup float *buf [[threadgroup(0)]],
    uint chunk [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint ntg [[threads_per_threadgroup]]) {
    const ulong base = ulong(chunk) * block;
    device const float *s = signs + (chunk % blocks_per_row) * block;

    for (uint i = tid; i < block; i += ntg) {
        const float v = src[base + i];
        buf[i] = signs_first ? v * s[i] : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each stage touches every element exactly once through block/2 disjoint
    // pairs (i, i + h), so no pair is shared between threads within a stage.
    const uint half_block = block / 2;
    for (uint h = 1; h < block; h <<= 1) {
        for (uint p = tid; p < half_block; p += ntg) {
            const uint i = ((p & ~(h - 1)) << 1) | (p & (h - 1));
            const float a = buf[i];
            const float b = buf[i + h];
            buf[i] = a + b;
            buf[i + h] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < block; i += ntg) {
        const float v = buf[i] * scale;
        dst[base + i] = signs_first ? v : v * s[i];
    }
}
