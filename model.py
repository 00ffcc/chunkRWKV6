########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# only for inference, no training code
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import types
from typing import List, Optional, Tuple, Union

os.environ["RWKV_JIT_ON"]='0'
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load
continous_chunk_rwkv6 = None
HEAD_SIZE = None
from continous_chunk import continousChunkRWKV6

class RWKV_Tmix_x060(MyModule):
    def __init__(self, config, layer_id, dtype=torch.float32):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.head_size = config.head_size_a
        self.n_head = config.dim_att // self.head_size
        assert config.dim_att % self.n_head == 0

        with torch.no_grad():
            self.time_maa_x = nn.Parameter(torch.zeros(1, 1, config.n_embd, dtype=dtype))
            self.time_maa_w = nn.Parameter(torch.zeros(1, 1, config.n_embd, dtype=dtype))
            self.time_maa_k = nn.Parameter(torch.zeros(1, 1, config.n_embd, dtype=dtype))
            self.time_maa_v = nn.Parameter(torch.zeros(1, 1, config.n_embd, dtype=dtype))
            self.time_maa_r = nn.Parameter(torch.zeros(1, 1, config.n_embd, dtype=dtype))
            self.time_maa_g = nn.Parameter(torch.zeros(1, 1, config.n_embd, dtype=dtype))

            TIME_MIX_EXTRA_DIM = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(config.n_embd, TIME_MIX_EXTRA_DIM*5, dtype=dtype))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, config.n_embd, dtype=dtype))


            self.time_decay = nn.Parameter(torch.zeros(1, 1, config.dim_att, dtype=torch.float32))

            TIME_DECAY_EXTRA_DIM = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(config.n_embd, TIME_DECAY_EXTRA_DIM, dtype=dtype))
            self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, config.dim_att, dtype=dtype))

            self.time_faaaa = nn.Parameter(torch.zeros(self.n_head, self.head_size, dtype=torch.float32))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)
        self.key = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)

        self.value = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)
        self.output = nn.Linear(config.dim_att, config.n_embd, bias=False, dtype=dtype)
        self.gate = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(self.n_head, config.dim_att, eps=(1e-5)*(config.head_size_divisor**2), dtype=torch.float32)

    @MyFunction
    def jit_func(self, x, start_pos, state):
        B, T, C = x.size()
        shifted = self.time_shift(x)
        for ids,i in enumerate(start_pos): # TODO 优化
            shifted[0, i, :] = state[0][ids, :, self.layer_id]
        xx = shifted - x
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        for ids in range(len(start_pos)): # TODO 优化
            if ids == len(start_pos)-1:
                state[0][ids, :, self.layer_id] = x[0, -1, :]
            else:
                state[0][ids, :, self.layer_id] = x[0, start_pos[ids+1]-1, :]
        return r, k, v, g, w

    def forward(self, x, seq_idx, start_pos, state):
        B, T, C = x.size()
        H = self.n_head
        dtype = x.dtype
        r, k, v, g, w = self.jit_func(x, start_pos, state)
        layer_state = state[1][:, :, :, :,self.layer_id].contiguous()
        x, state[1][:, :, :, :,self.layer_id] = continousChunkRWKV6.apply(B, T, C, H, layer_state,r, k, v, w, self.time_faaaa, seq_idx, HEAD_SIZE, continous_chunk_rwkv6)
        x = x.view(B * T, C)
        x = self.ln_x(x).view(B, T, C).to(dtype)
        x = self.output(x * g)
        return x

########################################################################################################

class RWKV_CMix_x060(MyModule):
    def __init__(self, config, layer_id, dtype=torch.float32):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix

            self.time_maa_k = nn.Parameter(torch.zeros(1, 1, config.n_embd,dtype=dtype))
            self.time_maa_r = nn.Parameter(torch.zeros(1, 1, config.n_embd,dtype=dtype))

        self.key = nn.Linear(config.n_embd, config.dim_ffn, bias=False,dtype=dtype)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False,dtype=dtype)
        self.value = nn.Linear(config.dim_ffn, config.n_embd, bias=False,dtype=dtype)

    @MyFunction
    def forward(self, x, start_pos, state):
        shifted = self.time_shift(x)
        for ids,i in enumerate(start_pos): # TODO 优化
            shifted[0, i, :] = state[2][ids, :, self.layer_id]
        
        xx = shifted - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)

        for ids in range(len(start_pos)): # TODO 优化
            if ids == len(start_pos)-1:
                state[2][ids, :, self.layer_id] = x[0, -1, :]
            else:
                state[2][ids, :, self.layer_id] = x[0, start_pos[ids+1]-1, :]

        return torch.sigmoid(self.receptance(xr)) * kv

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, config, layer_id,dtype=torch.float32):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd, dtype=dtype)
        self.ln2 = nn.LayerNorm(config.n_embd, dtype=dtype)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd, dtype=dtype)

        self.att = RWKV_Tmix_x060(config, layer_id, dtype=dtype)
        self.ffn = RWKV_CMix_x060(config, layer_id, dtype=dtype)

        
    def forward(self, x, seq_idx, start_pos, states):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.att(self.ln1(x), seq_idx, start_pos, states)
        x = x + self.ffn(self.ln2(x),          start_pos, states)
        return x



class RWKV6(nn.Module):
    def __init__(self, config,device,dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.dtype=dtype
        self.device=device
        assert config.n_embd % 32 == 0
        assert config.dim_att % 32 == 0
        assert config.dim_ffn % 32 == 0

        self.emb = nn.Embedding(config.vocab_size, config.n_embd, dtype=dtype)

        self.blocks = nn.ModuleList([Block(config, i, dtype=dtype) for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd, dtype=dtype)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False, dtype=dtype)
        global HEAD_SIZE
        HEAD_SIZE = config.head_size_a
        global continous_chunk_rwkv6
        dir_path = os.path.dirname(os.path.abspath(__file__))
        if torch.version.cuda is not None:
            continous_chunk_rwkv6 = load(name="continous_chunk_rwkv6", sources=[f"{dir_path}/cuda/continous_rwkv6_op.cpp", f"{dir_path}/cuda/continous_rwkv6.cu", f"{dir_path}/cuda/continous_inter.cu"], # cuda
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3",  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"]) 
        elif torch.version.hip is not None:
            continous_chunk_rwkv6 = load(name="continous_chunk_rwkv6", sources=[f"{dir_path}/cuda/continous_rwkv6_op.cpp", f"{dir_path}/cuda/continous_rwkv6.cu", f"{dir_path}/cuda/continous_inter.cu"], # rocm
                        verbose=True, extra_cuda_cflags=["-O3", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
        else:
            raise NotImplementedError("Only support CUDA and ROCm")       


    def empty_states(self,B=1):
        state = []
        state.append(
            torch.zeros(
                (B, self.config.n_embd, self.config.n_layer),
                dtype=self.dtype,
                requires_grad=False,
                device=self.device,
            ).contiguous()
        )
        state.append(
            torch.zeros(
                (
                    B,
                    self.config.dim_att // self.config.head_size_a,
                    self.config.head_size_a,
                    self.config.head_size_a,
                    self.config.n_layer,# 不连续会有问题
                ),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            ).contiguous()
        )
        state.append(
            torch.zeros(
                (B, self.config.n_embd, self.config.n_layer),
                dtype=self.dtype,
                requires_grad=False,
                device=self.device,
            ).contiguous()
        )
        return state
    def forward(self, input_ids, seq_idx, state):
        # seq_idx: 每个位置属于哪个序列
        # state定义与hf的rwkv5一致
        # 训练时不用cache
        B, T = input_ids.size()
        assert B == 1, "Batch size should be 1 in continous batching mode"
        x = self.emb(input_ids)

        start_pos = []
        for ids in range(seq_idx.shape[1]):
            if ids == 0:
                start_pos.append(0)
            elif seq_idx[0,ids] != seq_idx[0,ids-1]:
                start_pos.append(ids)
        for i,block in enumerate(self.blocks):
            x = block(x, seq_idx, start_pos, state)
            if self.config.RESCALE_LAYER>0 and (i+1)%self.config.RESCALE_LAYER==0:
                x=x/2
        x = self.ln_out(x)
        x = self.head(x)
        out_x = []
        for ids in range(len(start_pos)):
            if ids == len(start_pos)-1:
                out_x.append(x[0, -1, :])
            else:
                out_x.append(x[0, start_pos[ids+1]-1, :])
        out_x = torch.stack(out_x,dim=0)
        re = types.SimpleNamespace()
        re.logits = out_x
        re.state = state
        return re
    @staticmethod
    def from_pretrained(model_path, device="cuda",dtype=torch.bfloat16):
        w = torch.load(model_path, map_location="cpu")
        config = types.SimpleNamespace()
        config.vocab_size = w['emb.weight'].shape[0]
        config.n_embd = w['emb.weight'].shape[1]
        config.dim_ffn = w['blocks.0.ffn.key.weight'].shape[0]
        config.head_size_a = w['blocks.0.att.time_faaaa'].shape[1]
        config.n_layer = 0
        config.dim_att = w['blocks.0.att.receptance.weight'].shape[1]
        config.head_size_divisor = 8 # default value in https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/train.py
        config.RESCALE_LAYER = 6 if dtype==torch.float16 else 0 
        # Rescale for fp16 mode: set x = x/2 every X layer (to avoid fp16 overflow)
        # TODO:转换
        assert '_strategy' not in w,"应使用未经转化的模型"
        for x in list(w.keys()):
            w[x].requires_grad = False
            layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
            config.n_layer = max(config.n_layer,layer_id+1)
            if config.RESCALE_LAYER > 0:
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(layer_id // config.RESCALE_LAYER))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(layer_id // config.RESCALE_LAYER))
            
        model = RWKV6(config,device=device,dtype=dtype)
        model.load_state_dict(w)
        model.to(model.device)
        return model
if __name__ == '__main__':
    prompt_parts = [
        [
            "何为指令集调度",
            "通过调整重排指令集的执行(顺序)，提升指令的并行性",
            "避免非法或者模糊语义的操作，保证正确性（如相关等）",
        ],
        [
            "指令集调度实现",
            "静态调度-编译器优化（如分支）",
            "动态调度-硬件实现（记分牌、Tomasulo）",
        ],
        [
            "Superscalar: 每个时钟周期发射2条指令，1条FP指令和1条其他指令",
            "每个时钟周期取64位; 左边为Int , 右边为FP  只有第一条指令发射了，才能发射第二条 需要更多的寄存器端口，因为如果两条指令中第一条指令是对FP的",
            "load操作（通过整数部件完成），另一条指令为浮点操作指令，则都会有对浮点寄存器文件的操作",
        ],
    ]
    prompts = []
    for parts in prompt_parts:
        prompt = ""
        for part in parts:
            prompt += part
        prompts.append(prompt)
    from tokenizer.tokenization_rwkv_world import RWKVWorldTokenizer
    tokenizer = RWKVWorldTokenizer(vocab_file=r"D:\rwkv_input\tokenizer\rwkv_vocab_v20230424.txt")
    
    model = RWKV6.from_pretrained(r"D:\rwkv_input\model\rwkv6\RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth",dtype=torch.float16)
    states = model.empty_states(B=len(prompts))
    with torch.no_grad():
        for i in range(len(prompt_parts[0])):
            prompt = [j[i] for j in prompt_parts]
            input_ids, seq_idx = tokenizer.continous_encode(prompt)
            re = model(input_ids.to(model.device), seq_idx.to(model.device), states)
    log1 = re.logits
    model = None
    torch.cuda.empty_cache()

    from rwkv.model import RWKV
    model = RWKV(model=r"D:\rwkv_input\model\rwkv6\RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth", strategy='cuda fp16')
    log2 = []
    with torch.no_grad():
        for i,prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt)
            out, state = model.forward(input_ids, None)
            print((out-log1[i]).abs().max())

