from chunk import chunkRWKV6,vanillaRWKV6,HEAD_SIZE
import time
import torch
import matplotlib.pyplot as plt
from fla.ops.rwkv6 import chunk_rwkv6
DEVICE="cuda:0"
def benchmark(T, chunk_size):
    B, H = 1, 32
    C = H*HEAD_SIZE
    if chunk_size!=-2:
        r = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
        w = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
        u = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
    else:
        r = torch.randn(B,H, T, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B,H, T, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
        v = torch.randn(B,H, T, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
        w = torch.randn(B,H, T, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
        u = torch.randn(B,H, T, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
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
            y2, state2 = chunkRWKV6.apply(B, T, C, H, state, r, k, v, w, u, chunk_size)
        torch.cuda.synchronize()
        end = time.time()
        if i!=0:
            st+=end-start
        # print(f"chunkRWKV6: {end - start}s")
    return st/(num-1)
if __name__ == '__main__':
    torch.cuda.set_device(DEVICE)
    t=[ 32768, 65536, 98304, 131072, 163840]
    for chunk_size in [2048, 4096, 8192, 16384, -1,-2]:
        ti=[]
        for T in t:
            tt=benchmark(T, chunk_size)
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