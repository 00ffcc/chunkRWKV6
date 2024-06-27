import os
import torch
from torch.utils.cpp_extension import load
from einops import rearrange, reduce, repeat

dir_path = os.path.dirname(os.path.abspath(__file__))
HEAD_SIZE=64 # 每个head的大小
if torch.version.cuda is not None:
    rwkv6 = load(name="rwkv6", sources=[f"{dir_path}/cuda/rwkv6_op.cpp", f"{dir_path}/cuda/rwkv6.cu", f"{dir_path}/cuda/inter.cu"], # cuda
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3",  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"]) 
elif torch.version.hip is not None:
    rwkv6 = load(name="rwkv6", sources=[f"{dir_path}/cuda/rwkv6_op.cpp", f"{dir_path}/cuda/rwkv6.cu", f"{dir_path}/cuda/inter.cu"], # rocm
                verbose=True, extra_cuda_cflags=["-O3", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
else:
    raise NotImplementedError("Only support CUDA and ROCm")

class vanillaRWKV6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, state, r, k, v, w, u):
        with torch.no_grad():
            assert state.dtype == torch.float32
            assert w.dtype == torch.float32
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert state.is_contiguous()
            eew = torch.exp(-torch.exp(w.float())).contiguous()

            y = torch.empty((B, T, C), device=w.device, dtype=torch.float32, memory_format=torch.contiguous_format)
            if r.dtype == torch.bfloat16:
                rwkv6.forward_bf16(B, T, C, H, state, r, k, v, eew, u, y)
            elif r.dtype == torch.float16:
                rwkv6.forward_fp16(B, T, C, H, state, r, k, v, eew, u, y)
            elif r.dtype == torch.float32:
                rwkv6.forward_fp32(B, T, C, H, state, r, k, v, eew, u, y)
            return y

class chunkRWKV6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, state, r, k, v, w, u, chunk_size=32):
        with torch.no_grad():
            # r,k,v,w (B, T, C)
            assert state.dtype == torch.float32
            assert w.dtype == torch.float32
            assert T % chunk_size == 0, "T must be divisible by chunk_size"
            assert C == H*HEAD_SIZE, "C must be equal to H*HEAD_SIZE"
            nc=T//chunk_size # num_chunks
            cs=chunk_size # chunk_size

            w = -torch.exp(w)
            w_orig = w.clone()
            w_orig = rearrange(w_orig, 'b (nc cs) (h hs) -> b nc cs h hs', nc=nc, cs=cs, h=H, hs=HEAD_SIZE)
            w_orig = w_orig.cumsum(dim=2)

            w = torch.exp(w) # time_decay TODO 优化

            r = rearrange(r, 'b (nc cs) c -> b nc cs c', nc=nc, cs=cs)
            k = rearrange(k, 'b (nc cs) c -> b nc cs c', nc=nc, cs=cs)
            v = rearrange(v, 'b (nc cs) c -> b nc cs c', nc=nc, cs=cs)
            w = rearrange(w, 'b (nc cs) c -> b nc cs c', nc=nc, cs=cs)


            state = torch.stack([torch.cat([state[i:i+1], 
                                 torch.zeros((nc-1, H, HEAD_SIZE, HEAD_SIZE), device=state.device, dtype=state.dtype)]).contiguous() 
                                for i in range(B)], dim=0) # (B*nc, H, HEAD_SIZE, HEAD_SIZE)
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert state.is_contiguous()

            # 块内计算
            y = torch.empty((B, nc, cs, H, HEAD_SIZE), device=w.device, dtype=torch.float32, memory_format=torch.contiguous_format) # result
            if r.dtype == torch.bfloat16:
                rwkv6.forward_bf16(B*nc, cs, C, H, state, r, k, v, w, u, y)
            elif r.dtype == torch.float16:
                rwkv6.forward_fp16(B*nc, cs, C, H, state, r, k, v, w, u, y)
            elif r.dtype == torch.float32:
                rwkv6.forward_fp32(B*nc, cs, C, H, state, r, k, v, w, u, y)
            if nc > 1:
                # 计算块间的贡献
                r = rearrange(r, 'b nc cs (h hs) -> b nc cs h hs', h=H, hs=HEAD_SIZE)

                for j in range(1, nc): # TODO 优化
                    state[:, j, :, :, :] += torch.einsum('b h i j, b h j -> b h i j', 
                                                            state[:, j-1, :, :, :], 
                                                            torch.exp(w_orig[:, j, -1, :, :]))
                if r.dtype == torch.bfloat16:
                    rwkv6.Inter_fwd_bf16(B, cs, C, H, nc, state, r, w, y)
                elif r.dtype == torch.float16:
                    rwkv6.Inter_fwd_fp16(B, cs, C, H, nc, state, r, w, y)
                elif r.dtype == torch.float32:
                    rwkv6.Inter_fwd_fp32(B, cs, C, H, nc, state, r, w, y)
                
            state = state[:, -1, :, :, :] # 取最后一个块的状态

            # 输出
            y = rearrange(y, 'b nc cs h hs -> b (nc cs) (h hs)')
            return y, state

            
if __name__ == '__main__':
    B, T, H = 2, 64, 1
    C = H*HEAD_SIZE
    state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device='cuda', dtype=torch.float32)
    r = torch.randn(B, T, C, device='cuda', dtype=torch.float16)
    k = torch.randn(B, T, C, device='cuda', dtype=torch.float16)
    v = torch.randn(B, T, C, device='cuda', dtype=torch.float16)
    w = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    u = torch.randn(C, device='cuda', dtype=torch.float16)

    state1 = state.clone()
    y1 = vanillaRWKV6.apply(B, T, C, H, state1, r, k, v, w, u)

    y2, state2 = chunkRWKV6.apply(B, T, C, H, state, r, k, v, w, u)

    print(torch.max((y1-y2).abs()))
    print(y1-y2)
    print((state1-state2).abs().max())



            
