from chunk import chunkRWKV6,vanillaRWKV6,HEAD_SIZE
import time
import torch
import matplotlib.pyplot as plt

def benchmark(T, chunk_size):
    B, H = 1, 32
    C = H*HEAD_SIZE
    r = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    k = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    v = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    w = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    u = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
    state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device='cuda', dtype=torch.float32)
    st=0
    num=20
    for i in range(num):
        start = time.time()
        if chunk_size==-1:
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
    t=[4096, 8192, 16384, 32768, 65536]

    for chunk_size in [512, 1024, 2048, 4096, -1]:
        ti=[]
        for T in t:
            tt=benchmark(T, chunk_size)
            ti.append(tt)
            print(f"chunk_size={chunk_size}, T={T}, time={tt}s")
        plt.plot(t, ti, label=f"chunk_size={chunk_size}" if chunk_size!=-1 else "vanilla")
    plt.xlabel("tokens")
    plt.ylabel("operation time (s)")
    plt.legend()
    plt.show()
        