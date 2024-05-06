from chunk import chunkRWKV6,vanillaRWKV6,HEAD_SIZE
import time
import torch
import matplotlib.pyplot as plt
from fla.ops.rwkv6 import chunk_rwkv6
from einops import rearrange
B, H = 1, 32
C = H*HEAD_SIZE
T = 1024
r = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
k = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
v = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
w = torch.randn(B, T, C, device=DEVICE, dtype=torch.float32)
u = torch.randn(C, device=DEVICE, dtype=torch.float32)
state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device=DEVICE, dtype=torch.float32)
state1 = state.clone()

y1 = vanillaRWKV6.apply(B, T, C, H, state1, r, k, v, w, u)

r = rearrange(r, 'b t (h c) -> b h t c', h=H)
k = rearrange(k, 'b t (h c) -> b h t c', h=H)
v = rearrange(v, 'b t (h c) -> b h t c', h=H)
w = rearrange(w, 'b t (h c) -> b h t c', h=H)
u = rearrange(u, '(h c) -> c', h=H)

y2 = chunk_rwkv6(B, T, C, H, state, r, k, v, w, u)

print((y1-y2).abs().max())