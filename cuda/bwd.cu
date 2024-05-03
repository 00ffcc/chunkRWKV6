#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

template <typename F>
__global__ void bwd_w_inner_kernel(const int B, const int T, const int C, const int H, float *__restrict__ _state,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _dy,
                               F *__restrict__ const _sa, F *__restrict__ const _sb, F *__restrict__ const _sc)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int i = threadIdx.x;
    
}