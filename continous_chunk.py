import os
import torch
from torch.utils.cpp_extension import load
from einops import rearrange, reduce, repeat

dir_path = os.path.dirname(os.path.abspath(__file__))
HEAD_SIZE=64 # 每个head的大小
if torch.version.cuda is not None:
    continous_chunk_rwkv6 = load(name="continous_chunk_rwkv6", sources=[f"{dir_path}/cuda/continous_rwkv6_op.cpp", f"{dir_path}/cuda/continous_rwkv6.cu", f"{dir_path}/cuda/continous_inter.cu"], # cuda
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3",  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"]) 
elif torch.version.hip is not None:
    continous_chunk_rwkv6 = load(name="continous_chunk_rwkv6", sources=[f"{dir_path}/cuda/continous_rwkv6_op.cpp", f"{dir_path}/cuda/continous_rwkv6.cu", f"{dir_path}/cuda/continous_inter.cu"], # rocm
                verbose=True, extra_cuda_cflags=["-O3", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
else:
    raise NotImplementedError("Only support CUDA and ROCm")

class continousChunkRWKV6(torch.autograd.Function):
    # 将所有序列拼在一起,batch_size=1
    # seq_idx: 每个位置是属于哪个序列的
    @staticmethod
    def forward(ctx, B, T, C, H, state, r, k, v, w, u, seq_idx, chunk_size=32):
        with torch.no_grad():
            # r,k,v,w (B, T, C)
            assert B == 1, "batch_size must be 1 for continous_chunk_rwkv6"
            assert state.dtype == torch.float32
            assert w.dtype == torch.float32
            assert T % chunk_size == 0, "T must be divisible by chunk_size"
            assert C == H*HEAD_SIZE, "C must be equal to H*HEAD_SIZE"
            nc=T//chunk_size # num_chunks
            cs=chunk_size # chunk_size

            cnt = seq_idx[0, -1].item() + 1 # 总的state的个数 [0,cnt)
            end_state_idx = torch.tensor([i for i in range(cnt)], dtype=torch.int32, device=state.device)

            seq_idx = rearrange(seq_idx, 'b (nc cs) -> b nc cs', nc=nc, cs=cs)
            state_idx = seq_idx.clone() # 对应的state的编号
            
            for i in range(1, nc):
                if seq_idx[0, i, 0] == seq_idx[0, i-1, -1]:
                    state_idx[0, i] = torch.where(state_idx[0, i] == state_idx[0, i, 0], cnt, state_idx[0, i])
                    end_state_idx[seq_idx[0, i, 0].item()] = cnt
                    cnt += 1
            if cnt-state.shape[0] > 0:
                state = torch.cat([state, torch.zeros((cnt-state.shape[0], H, HEAD_SIZE, HEAD_SIZE), device=state.device, dtype=state.dtype)], dim=0).contiguous() # (cnt, H, HEAD_SIZE, HEAD_SIZE)

            w = -torch.exp(w)
            w_orig = w.clone()
            w_orig = rearrange(w_orig, 'b (nc cs) (h hs) -> b nc cs h hs', nc=nc, cs=cs, h=H, hs=HEAD_SIZE)
            w_orig = w_orig.cumsum(dim=2)

            w = torch.exp(w) # time_decay TODO 优化

            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert state.is_contiguous()

            # print("shape", state_idx.shape, state.shape, w_orig.shape)

            # 块内计算
            y = torch.empty((B, nc, cs, H, HEAD_SIZE), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format) # result
            if r.dtype == torch.bfloat16:
                continous_chunk_rwkv6.forward_bf16(B*nc, cs, C, H, state, state_idx, r, k, v, w, u, y)
            elif r.dtype == torch.float16:
                continous_chunk_rwkv6.forward_fp16(B*nc, cs, C, H, state, state_idx, r, k, v, w, u, y)
            elif r.dtype == torch.float32:
                continous_chunk_rwkv6.forward_fp32(B*nc, cs, C, H, state, state_idx, r, k, v, w, u, y)

            # 计算块间的贡献

            if nc > 1:
                lengths = torch.zeros(B, nc, device=w.device, dtype=torch.int32)
                for j in range(1, nc): # TODO 优化
                    if seq_idx[0, j, 0] == seq_idx[0, j-1, -1]:
                        lengths[0, j] = torch.sum(seq_idx[0, j, :]==seq_idx[0, j-1, -1], dtype=torch.int32).item()
                        state[state_idx[0, j, 0]] += torch.einsum('h i j, h j -> h i j', 
                                                                    state[state_idx[0, j-1, -1]], 
                                                                    torch.exp(w_orig[0, j, lengths[0, j]-1, :, :]))

                if r.dtype == torch.bfloat16:
                    continous_chunk_rwkv6.Inter_fwd_bf16(B, cs, C, H, nc, state, state_idx, lengths, r, w, y)
                elif r.dtype == torch.float16:
                    continous_chunk_rwkv6.Inter_fwd_fp16(B, cs, C, H, nc, state, state_idx, lengths, r, w, y)
                elif r.dtype == torch.float32:
                    continous_chunk_rwkv6.Inter_fwd_fp32(B, cs, C, H, nc, state, state_idx, lengths, r, w, y)
            
            state = state[end_state_idx, :, :, :]

            # 输出
            y = rearrange(y, 'b nc cs h hs -> b (nc cs) (h hs)')
            return y, state

            
if __name__ == '__main__':
    B, T, H = 1, 64, 1
    C = H*HEAD_SIZE
    state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device='cuda', dtype=torch.float32)
    r = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    k = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    v = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    w = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    u = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    seq_idx = torch.zeros((B, T), device='cuda', dtype=torch.int32)
    state1 = state.clone()
    import chunk
    y1 = chunk.vanillaRWKV6.apply(B, T, C, H, state1, r, k, v, w, u)

    y2, state2 = continousChunkRWKV6.apply(B, T, C, H, state, r, k, v, w, u, seq_idx)

    print(torch.max((y1-y2).abs()))
    print(y1-y2)
    print((state1-state2).abs().max())



            