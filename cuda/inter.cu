#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

template <typename F>
__global__ void inter_fwd_kernel(const int B, const int T, const int C, const int H, const int CK, float *__restrict__ _state,
                               const F *__restrict__ const _r, const float *__restrict__ _w, F *__restrict__ const _y)
                               // CK : chunk nums
{
    const int b = blockIdx.x;
    const int ck= blockIdx.y+1;
    const int h = blockIdx.z;
    const int i = threadIdx.x;
    _state+=b*CK*H*_N_*_N_ + (ck-1)*H*_N_*_N_ + h*_N_*_N_ + i*_N_; // state_(ck-1)
    __shared__ float r[_N_], cw[_N_];

    __syncthreads();
    cw[i]=1.0f;
    
    // process y
    for (int t = b*CK*T*C + ck*T*C + h*_N_ + i; t < b*CK*T*C + (ck+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        __syncthreads();
        float y = float(_y[t]);
        #pragma unroll
        for (int j = 0; j < _N_; j += 4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& w_ = (float4&)(cw[j]);
            const float4& state_ = (float4&)(_state[j]);
            y += r_.x*w_.x*state_.x;
            y += r_.y*w_.y*state_.y;
            y += r_.z*w_.z*state_.z;
            y += r_.w*w_.w*state_.w;
        }
        _y[t] = F(y);
        __syncthreads();
        cw[i] *= _w[t];
        __syncthreads();
    }
}
void inter_fwd_bf16(int B, int T, int C, int H, int CK, float *state, bf16 *r, float *w, bf16 *y)
{
    assert(H*_N_ == C);
    inter_fwd_kernel<<<dim3(B, CK, H), dim3(_N_)>>>(B, T, C, H, CK, state, r, w, y);
}
void inter_fwd_fp16(int B, int T, int C, int H, int CK, float *state, fp16 *r, float *w, fp16 *y)
{
    assert(H*_N_ == C);
    inter_fwd_kernel<<<dim3(B, CK, H), dim3(_N_)>>>(B, T, C, H, CK, state, r, w, y);
}
void inter_fwd_fp32(int B, int T, int C, int H, int CK, float *state, fp32 *r, float *w, fp32 *y)
{
    assert(H*_N_ == C);
    inter_fwd_kernel<<<dim3(B, CK-1, H), dim3(_N_)>>>(B, T, C, H, CK, state, r, w, y);
}