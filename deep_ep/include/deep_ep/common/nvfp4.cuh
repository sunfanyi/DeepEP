#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic::nvfp4 {

constexpr int kNumElementsPerGroup = 16;
constexpr int kNumElementsPerThread = 8;

struct PackedVec {
    __nv_bfloat162 values[kNumElementsPerThread / 2];
};

__device__ __forceinline__ float reciprocal_approximate_ftz(const float value) {
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ uint8_t float_to_e2m1(const float value) {
    const auto sign = static_cast<uint8_t>(value < 0);
    const auto absolute_value = fabsf(value);
    const auto log_value = log2f(absolute_value);

    uint8_t exponent, mantissa;
    if (log_value < 0) {
        exponent = 0;
        mantissa = log_value < -1 ? 0 : 1;
    } else {
        exponent = static_cast<uint8_t>(fminf(floorf(log_value + 1), 3.0f));
        mantissa = log_value + 1 - exponent > 0.5f ? 1 : 0;
    }
    return static_cast<uint8_t>((sign << 3) | (exponent << 1) | mantissa);
}

__device__ __forceinline__ uint32_t fp32_vec_to_e2m1(float2 (&values)[4]) {
#if CUDA_VERSION >= 12080
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && __CUDA_ARCH_HAS_FEATURE__(SM100_ALL)
    uint32_t ptx_result;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(ptx_result)
        : "f"(values[0].x), "f"(values[0].y),
          "f"(values[1].x), "f"(values[1].y),
          "f"(values[2].x), "f"(values[2].y),
          "f"(values[3].x), "f"(values[3].y));
    return ptx_result;
#endif
#endif
    uint32_t result = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
        result |= (float_to_e2m1(values[i].x) & 0xf) << (8 * i);
        result |= (float_to_e2m1(values[i].y) & 0xf) << (8 * i + 4);
    }
    return result;
}

template <int kGroupSize = kNumElementsPerGroup>
__device__ __forceinline__ uint32_t quantize_group(
        const PackedVec& input, const float global_scale, uint8_t* scale_out) {
    auto local_max = __habs2(input.values[0]);
    #pragma unroll
    for (int i = 1; i < kNumElementsPerThread / 2; ++ i)
        local_max = __hmax2(local_max, __habs2(input.values[i]));

    constexpr int kNumThreadsPerGroup = kGroupSize / kNumElementsPerThread;
    EP_STATIC_ASSERT(kNumThreadsPerGroup == 2 or kNumThreadsPerGroup == 4,
                     "Invalid NVFP4 group size");
    local_max = __hmax2(__shfl_xor_sync(0xffffffff, local_max, 1), local_max);
    if constexpr (kNumThreadsPerGroup == 4)
        local_max = __hmax2(__shfl_xor_sync(0xffffffff, local_max, 2), local_max);

    const auto group_max = static_cast<float>(__hmax(local_max.x, local_max.y));
    auto scale_value = global_scale * (group_max * reciprocal_approximate_ftz(6.0f));
    const auto fp8_scale = __nv_fp8_e4m3(scale_value);
    *scale_out = fp8_scale.__x;
    scale_value = static_cast<float>(fp8_scale);
    const auto output_scale = scale_value != 0
        ? reciprocal_approximate_ftz(scale_value * reciprocal_approximate_ftz(global_scale))
        : 0.0f;

    float2 scaled_values[kNumElementsPerThread / 2];
    #pragma unroll
    for (int i = 0; i < kNumElementsPerThread / 2; ++ i) {
        scaled_values[i] = __bfloat1622float2(input.values[i]);
        scaled_values[i].x *= output_scale;
        scaled_values[i].y *= output_scale;
    }
    return fp32_vec_to_e2m1(scaled_values);
}

template <int kHidden>
__device__ __forceinline__ void quantize_token(
        const void* x, void* packed_x, sf_pack_t* packed_scales,
        const float global_scale, const int token_idx, const int lane_idx) {
    constexpr int kNumInputChunks = kHidden / kNumElementsPerThread;
    EP_STATIC_ASSERT(kHidden % (32 * kNumElementsPerThread) == 0,
                     "NVFP4 hidden size must support warp vectorization");

    const auto src = static_cast<const int4*>(x) + token_idx * kNumInputChunks;
    auto dst = static_cast<uint32_t*>(packed_x);
    auto dst_scales = reinterpret_cast<uint8_t*>(packed_scales);
    for (int i = lane_idx; i < kNumInputChunks; i += 32) {
        const auto value = __ldg(src + i);
        const auto packed_value = *reinterpret_cast<const PackedVec*>(&value);
        uint8_t scale;
        dst[i] = quantize_group(packed_value, global_scale, &scale);
        if (i % 2 == 0)
            dst_scales[i / 2] = scale;
    }
    __syncwarp();
}

template <int kNumSFPacks, int kExpertCapacity>
__device__ __forceinline__ void store_expert_scales(
        const sf_pack_t* packed_scales, sf_pack_t* recv_sf,
        const int dst_tensor_idx, const int lane_idx) {
    constexpr int kScaleTokenBlock = 128;
    constexpr int kNumTokenBlocks = math::constexpr_ceil_div(kExpertCapacity, kScaleTokenBlock);
    // Each group of four scales is contiguous and aligned in both layouts.
    const auto src_scales = reinterpret_cast<const uint32_t*>(packed_scales);

    auto mask = ptx::gather(dst_tensor_idx >= 0);
    while (mask) {
        const int valid_lane_idx = __ffs(mask) - 1;
        const auto flat_idx = ptx::exchange(dst_tensor_idx, valid_lane_idx);
        const auto expert_idx = flat_idx / kExpertCapacity;
        const auto token_idx = flat_idx % kExpertCapacity;
        auto dst_scales = reinterpret_cast<uint32_t*>(recv_sf) +
            expert_idx * kNumTokenBlocks * kNumSFPacks * kScaleTokenBlock;

        const auto token_block_idx = token_idx / kScaleTokenBlock;
        const auto token_lane_idx = token_idx % 32;
        const auto token_quad_idx = token_idx % kScaleTokenBlock / 32;
        #pragma unroll
        for (int i = lane_idx; i < kNumSFPacks; i += 32) {
            const auto offset =
                token_block_idx * (kNumSFPacks * kScaleTokenBlock) +
                i * kScaleTokenBlock +
                token_lane_idx * (kScaleTokenBlock / 32) + token_quad_idx;
            dst_scales[offset] = src_scales[i];
        }
        mask ^= 1 << valid_lane_idx;
    }
}

}  // namespace deep_ep::elastic::nvfp4
