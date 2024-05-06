#include <torch/extension.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAGuard.h>
typedef at::BFloat16 bf16;
typedef at::Half fp16;
typedef float fp32;

void cuda_forward_bf16(int B, int T, int C, int H, int CT, float *state, int *_state_idx, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, float *y);
void cuda_forward_fp16(int B, int T, int C, int H, int CT, float *state, int *_state_idx, fp16 *r, fp16 *k, fp16 *v, float *w, fp16 *u, float *y);
void cuda_forward_fp32(int B, int T, int C, int H, int CT, float *state, int *_state_idx, fp32 *r, fp32 *k, fp32 *v, float *w, fp32 *u, float *y);

void forward_bf16(int64_t B, int64_t T, int64_t C, int64_t H, int64_t CT, torch::Tensor &state, torch::Tensor &state_idx, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    cuda_forward_bf16(B, T, C, H, CT, state.data_ptr<float>(), state_idx.data_ptr<int>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), w.data_ptr<float>(), u.data_ptr<bf16>(), y.data_ptr<float>());
}
void forward_fp16(int64_t B, int64_t T, int64_t C, int64_t H, int64_t CT, torch::Tensor &state, torch::Tensor &state_idx, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    cuda_forward_fp16(B, T, C, H, CT, state.data_ptr<float>(), state_idx.data_ptr<int>(), r.data_ptr<fp16>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), w.data_ptr<float>(), u.data_ptr<fp16>(), y.data_ptr<float>());
}
void forward_fp32(int64_t B, int64_t T, int64_t C, int64_t H, int64_t CT, torch::Tensor &state, torch::Tensor &state_idx, torch::Tensor &r, torch::Tensor &k, torch::Tensor &v, torch::Tensor &w, torch::Tensor &u, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    cuda_forward_fp32(B, T, C, H, CT, state.data_ptr<float>(), state_idx.data_ptr<int>(), r.data_ptr<fp32>(), k.data_ptr<fp32>(), v.data_ptr<fp32>(), w.data_ptr<float>(), u.data_ptr<fp32>(), y.data_ptr<float>());
}


void inter_fwd_bf16(int B, int T, int C, int H, int CK, float *state, int *_state_idx, int *_length, bf16 *r, float *w, float *y);
void inter_fwd_fp16(int B, int T, int C, int H, int CK, float *state, int *_state_idx, int *_length, fp16 *r, float *w, float *y);
void inter_fwd_fp32(int B, int T, int C, int H, int CK, float *state, int *_state_idx, int *_length, fp32 *r, float *w, float *y);
void Inter_fwd_bf16(int64_t B, int64_t T, int64_t C, int64_t H, int64_t CK, torch::Tensor &state, torch::Tensor &state_idx, torch::Tensor &length, torch::Tensor &r, torch::Tensor &w, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    inter_fwd_bf16(B, T, C, H, CK, state.data_ptr<float>(), state_idx.data_ptr<int>(), length.data_ptr<int>(), r.data_ptr<bf16>(), w.data_ptr<float>(), y.data_ptr<float>());
}
void Inter_fwd_fp16(int64_t B, int64_t T, int64_t C, int64_t H, int64_t CK, torch::Tensor &state, torch::Tensor &state_idx, torch::Tensor &length, torch::Tensor &r, torch::Tensor &w, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    inter_fwd_fp16(B, T, C, H, CK, state.data_ptr<float>(), state_idx.data_ptr<int>(), length.data_ptr<int>(), r.data_ptr<fp16>(), w.data_ptr<float>(), y.data_ptr<float>());
}
void Inter_fwd_fp32(int64_t B, int64_t T, int64_t C, int64_t H, int64_t CK, torch::Tensor &state, torch::Tensor &state_idx, torch::Tensor &length, torch::Tensor &r, torch::Tensor &w, torch::Tensor &y) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(state));
    inter_fwd_fp32(B, T, C, H, CK, state.data_ptr<float>(), state_idx.data_ptr<int>(), length.data_ptr<int>(), r.data_ptr<fp32>(), w.data_ptr<float>(), y.data_ptr<float>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_bf16", &forward_bf16, "rwkv6 forward_bf16");
    m.def("forward_fp16", &forward_fp16, "rwkv6 forward_fp16");
    m.def("forward_fp32", &forward_fp32, "rwkv6 forward_fp32");

    m.def("Inter_fwd_bf16", &Inter_fwd_bf16, "rwkv6 inter forward_bf16");
    m.def("Inter_fwd_fp16", &Inter_fwd_fp16, "rwkv6 inter forward_fp16");
    m.def("Inter_fwd_fp32", &Inter_fwd_fp32, "rwkv6 inter forward_fp32");
}
TORCH_LIBRARY(continous_chunk_rwkv6, m) {
    m.def("forward_bf16", forward_bf16);
    m.def("forward_fp16", forward_fp16);
    m.def("forward_fp32", forward_fp32);

    m.def("Inter_fwd_bf16", Inter_fwd_bf16);
    m.def("Inter_fwd_fp16", Inter_fwd_fp16);
    m.def("Inter_fwd_fp32", Inter_fwd_fp32);
}
