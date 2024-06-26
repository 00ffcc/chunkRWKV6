#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H, const int CT, 
                                     float *__restrict__ _state    , 
                               const   int *__restrict__ _state_idx,
                               const     F *__restrict__ const _r  , 
                               const     F *__restrict__ const _k  , 
                               const     F *__restrict__ const _v  , 
                               const float *__restrict__ _w        , 
                                     float *__restrict__ _w_orig   , 
                               const float *__restrict__ _u        ,
                                     float *__restrict__ const _y  )
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _state_idx += b*T;
    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    
    int current_state_idx = -1, state_idx0 = _state_idx[0];
    float state[_N_];
    float w_orig = 0;
    u[i] = _u[h*_N_ + i];
    __syncthreads();
    int t_end = b*T*C + h*_N_ + i + ((b+1)*T <= CT ? T : CT - b*T)*C;
    for (int t = b*T*C + h*_N_ + i, ti = 0; t < t_end; t += C, ti++)
    {
        __syncthreads();
        if (_state_idx[ti] != current_state_idx)
        {
            // store
            if (current_state_idx != -1)
            {
                int t0 = current_state_idx*H*_N_*_N_ + h*_N_*_N_ + i*_N_;
                #pragma unroll
                for (int j = 0; j < _N_; j++)
                    _state[t0 + j] = state[j];
            }
            if (current_state_idx == state_idx0)
                _w_orig[b*C + h*_N_ + i] = exp(w_orig);
            current_state_idx = _state_idx[ti];

            // load
            int t0 = current_state_idx*H*_N_*_N_ + h*_N_*_N_ + i*_N_;
            #pragma unroll
            for (int j = 0; j < _N_; j++)
                state[j] = _state[t0 + j];
        }
        const float ww = -exp(_w[t]);
        w_orig += ww;
        w[i] = exp(ww);
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = y;
    }
    int t0 = current_state_idx*H*_N_*_N_ + h*_N_*_N_ + i*_N_;
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[t0 + j] = state[j];
    if (current_state_idx == state_idx0)
        _w_orig[b*C + h*_N_ + i] = exp(w_orig);
}

void cuda_forward_bf16(int B, int T, int C, int H, int CT, float *state, int *_state_idx, bf16 *r, bf16 *k, bf16 *v, float *w, float *w_orig, float *u, float *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, CT, state, _state_idx, r, k, v, w, w_orig, u, y);
}
void cuda_forward_fp16(int B, int T, int C, int H, int CT, float *state, int *_state_idx, fp16 *r, fp16 *k, fp16 *v, float *w, float *w_orig, float *u, float *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, CT, state, _state_idx, r, k, v, w, w_orig, u, y);
}
void cuda_forward_fp32(int B, int T, int C, int H, int CT, float *state, int *_state_idx, fp32 *r, fp32 *k, fp32 *v, float *w, float *w_orig, float *u, float *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, CT, state, _state_idx, r, k, v, w, w_orig, u, y);
}
