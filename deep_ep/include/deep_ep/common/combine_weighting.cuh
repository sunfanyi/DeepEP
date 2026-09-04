#pragma once

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/ptx.cuh>


namespace deep_ep::elastic {

template <typename vec_t>
__device__ __forceinline__
void scale_combine_vector(vec_t& value, const float weight, const float scaling_factor) {
    auto pairs = reinterpret_cast<nv_bfloat162*>(&value);
    #pragma unroll
    for (int i = 0; i < sizeof(vec_t) / sizeof(nv_bfloat162); ++ i) {
        auto pair = __bfloat1622float2(pairs[i]);
        // Match the standalone FP32 multiplications followed by BF16 rounding.
        pair.x = __fmul_rn(__fmul_rn(pair.x, weight), scaling_factor);
        pair.y = __fmul_rn(__fmul_rn(pair.y, weight), scaling_factor);
        pairs[i] = __float22bfloat162_rn(pair);
    }
}

template <int kHiddenVec, typename vec_t>
__device__ __forceinline__
void scale_combine_buffer(const int lane_idx, vec_t* buffer,
                          const float weight, const float scaling_factor) {
    // The elected lane has waited for TMA; synchronize before the warp reads.
    __syncwarp();
    #pragma unroll
    for (int i = lane_idx; i < kHiddenVec; i += 32) {
        auto value = buffer[i];
        scale_combine_vector(value, weight, scaling_factor);
        buffer[i] = value;
    }
    ptx::tma_store_fence();
    __syncwarp();
}

}  // namespace deep_ep::elastic
