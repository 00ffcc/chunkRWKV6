from chunk import chunkRWKV6,vanillaRWKV6,HEAD_SIZE

from einops import rearrange
from continous_chunk import continousChunkRWKV6
import time
import torch
import matplotlib.pyplot as plt
from fla.ops.rwkv6 import chunk_rwkv6
from torch.utils.cpp_extension import load
DEVICE="cuda:4"
def benchmark(T, chunk_size, dtype=torch.float32):
    B, H = 1, 32
    C = H*HEAD_SIZE
    if chunk_size==-1:
        r = torch.randn(B, T, C, device=DEVICE, dtype=dtype)
        k = torch.randn(B, T, C, device=DEVICE, dtype=dtype)
        v = torch.randn(B, T, C, device=DEVICE, dtype=dtype)
        w = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
        u = torch.randn(C, device=DEVICE, dtype=torch.float16)
    if chunk_size==-2:
        r = torch.randn(B, H, T, HEAD_SIZE, device=DEVICE, dtype=dtype)
        k = torch.randn(B, H, T, HEAD_SIZE, device=DEVICE, dtype=dtype)
        v = torch.randn(B, H, T, HEAD_SIZE, device=DEVICE, dtype=dtype)
        w = torch.randn(B, H, T, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
        u = torch.randn(H, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
    if chunk_size>0:
        r = torch.randn(1, B*T, C, device=DEVICE, dtype=dtype)
        k = torch.randn(1, B*T, C, device=DEVICE, dtype=dtype)
        v = torch.randn(1, B*T, C, device=DEVICE, dtype=dtype)
        w = torch.randn(1, B*T, C, device=DEVICE, dtype=torch.float32)
        u = torch.randn(C, device=DEVICE, dtype=torch.float16) # 记得改
        seq_idx = torch.tensor([[i]*T for i in range(B)], device=DEVICE, dtype=torch.int32)
        seq_idx = rearrange(seq_idx, "B T -> 1 (B T)")
    state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
    st=0
    num=20
    for i in range(num):
        torch.cuda.synchronize()
        start = time.time()
        if chunk_size==-2:
            chunk_rwkv6(r,k,v,w,u)
        elif chunk_size==-1:
            y1 = vanillaRWKV6.apply(B, T, C, H, state, r, k, v, w, u)
        else:
            # y2, state2 = continousChunkRWKV6.apply(B, T, C, H, state, r, k, v, w, u, seq_idx, HEAD_SIZE, chunk_size)
            y2, state2 = chunkRWKV6.apply(B, T, C, H, state, r, k, v, w, u, chunk_size) # TODO: ?为什么continous这么慢
        torch.cuda.synchronize()
        end = time.time()
        if i!=0:
            st+=end-start
        # print(f"chunkRWKV6: {end - start}s")
    return st/(num-1)
if __name__ == '__main__':
    import os
    dir_path = os.path.dirname(os.path.abspath(__file__))
    if torch.version.cuda is not None:
        continous_chunk_rwkv6 = load(name="continous_chunk_rwkv6", sources=[f"{dir_path}/cuda/continous_rwkv6_op.cpp", f"{dir_path}/cuda/continous_rwkv6.cu", f"{dir_path}/cuda/continous_inter.cu"], # cuda
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3",  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"]) 
    elif torch.version.hip is not None:
        continous_chunk_rwkv6 = load(name="continous_chunk_rwkv6", sources=[f"{dir_path}/cuda/continous_rwkv6_op.cpp", f"{dir_path}/cuda/continous_rwkv6.cu", f"{dir_path}/cuda/continous_inter.cu"], # rocm
                    verbose=True, extra_cuda_cflags=["-O3", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
    else:
        raise NotImplementedError("Only support CUDA and ROCm")
    torch.cuda.set_device(DEVICE)
    t=[ 32768, 65536, 98304, 131072, 163840]
    for chunk_size in [2048, 4096, 8192, 16384, -1,-2]:
        ti=[]
        for T in t:
            tt=benchmark(T, chunk_size, torch.float16)
            ti.append(tt)
            print(f"chunk_size={chunk_size}, T={T}, time={tt}s")
        if chunk_size == -2:
            name = "fla chunk rwkv6(use triton)"
        elif chunk_size == -1:
            name = "vanilla"
        else:
            name = f"chunk_size={chunk_size}"
        plt.plot(t, ti, label= name)
    plt.xlabel("tokens")
    plt.ylabel("operation time (s)")
    plt.legend()
    plt.show()
    plt.savefig("1.png")