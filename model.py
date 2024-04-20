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
rwkv6=None
class RWKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, state, r, k, v, w, u):
        with torch.no_grad():
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert state.dtype == torch.float32
            # w=w.to(torch.float32) #?
            assert w.dtype == torch.float32
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert state.is_contiguous()
            eew = torch.exp(-torch.exp(w.float())).contiguous()

            y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
            if r.dtype == torch.bfloat16:
                rwkv6.forward_bf16(B, T, C, H, state, r, k, v, eew, u, y)
            elif r.dtype == torch.float16:
                rwkv6.forward_fp16(B, T, C, H, state, r, k, v, eew, u, y)
            elif r.dtype == torch.float32:
                rwkv6.forward_fp32(B, T, C, H, state, r, k, v, eew, u, y)
            return y
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

            self.time_faaaa = nn.Parameter(torch.zeros(self.n_head, self.head_size, dtype=dtype))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)
        self.key = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)

        self.value = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)
        self.output = nn.Linear(config.dim_att, config.n_embd, bias=False, dtype=dtype)
        self.gate = nn.Linear(config.n_embd, config.dim_att, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(self.n_head, config.dim_att, eps=(1e-5)*(config.head_size_divisor**2), dtype=dtype)

    @MyFunction
    def jit_func(self, x,state):
        # print('jit_func', x.size(), state[0].size())
        B, T, C = x.size()
        if x.size(1) == 1:
            shifted = state[0][:, :, self.layer_id].unsqueeze(1)
        else:
            shifted = self.time_shift(x)
            shifted[:, 0] = state[0][:, :, self.layer_id]
        xx = shifted - x
        # print("xx:",xx.size())
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        # print(xxx.size(),self.time_maa_w2.size())
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

        state[0][:, :, self.layer_id] = x[:, -1]

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x,state):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x,state)
        layer_state = state[1][:, :, :, :,self.layer_id].contiguous()
        x = RWKV_6.apply(B, T, C, H, layer_state,r, k, v, w, self.time_faaaa)
        state[1][:, :, :, :,self.layer_id]=layer_state
        return self.jit_func_2(x, g)

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
    def forward(self, x,state):
        if x.size(1) == 1:
            shifted = state[2][:, :, self.layer_id].unsqueeze(1)
        else:
            shifted = self.time_shift(x)
            shifted[:, 0] = state[2][:, :, self.layer_id]
        xx = shifted - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)

        state[2][:, :, self.layer_id] = x[:, -1]

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

        
    def forward(self, x, states):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.att(self.ln1(x), state=states)
        x = x + self.ffn(self.ln2(x), state=states)
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
        HEAD_SIZE = config.head_size_a
        global rwkv6
        dir_path = os.path.dirname(os.path.abspath(__file__))
        if torch.version.hip is not None:
            rwkv6=load(name="rwkv6", sources=[f"{dir_path}/cuda/rwkv6_op.cpp", f"{dir_path}cuda/rwkv6.cu"], # TODO:路径
                        verbose=True, extra_cuda_cflags=["-O3", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
        else:
            rwkv6 = load(name="rwkv6", sources=[f"{dir_path}/cuda/rwkv6_op.cpp", f"{dir_path}/cuda/rwkv6.cu"], # TODO:路径
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3",  "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])            


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
    def forward(self, input_ids,state=None,use_cache=True):
        # state定义与hf的rwkv5一致
        # 训练时不用cache
        B, T = input_ids.size()
        if state is None:
            state=self.empty_states(B) 
        x = self.emb(input_ids)
        for i,block in enumerate(self.blocks):
            x = block(x,states=state)
            if self.config.RESCALE_LAYER>0 and (i+1)%self.config.RESCALE_LAYER==0:
                x=x/2
        x = self.ln_out(x)
        x = self.head(x)
        # return x,state
        re=types.SimpleNamespace()
        re.logits=x
        re.state=state
        return re
    @staticmethod
    def from_pretrained(model_path, device="cuda",dtype=torch.bfloat16):
        w = torch.load(model_path, map_location="cpu")
        config=types.SimpleNamespace()
        config.vocab_size=w['emb.weight'].shape[0]
        config.n_embd = w['emb.weight'].shape[1]
        config.dim_ffn=w['blocks.0.ffn.key.weight'].shape[0]
        config.head_size_a=w['blocks.0.att.time_faaaa'].shape[1]
        config.n_layer=0
        config.dim_att=w['blocks.0.att.receptance.weight'].shape[1]
        config.head_size_divisor=8 # default value in https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/train.py
        config.RESCALE_LAYER = 6 if dtype==torch.float16 else 0 
        # Rescale for fp16 mode: set x = x/2 every X layer (to avoid fp16 overflow)
        # TODO:转换
        assert '_strategy' not in w,"应使用未经转化的模型"
        for x in list(w.keys()):
            w[x].requires_grad = False
            layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
            config.n_layer=max(config.n_layer,layer_id+1)
            if config.RESCALE_LAYER > 0:
                if 'att.output.weight' in x:
                    w[x] = w[x] / (2 ** int(layer_id // config.RESCALE_LAYER))
                if 'ffn.value.weight' in x:
                    w[x] = w[x] / (2 ** int(layer_id // config.RESCALE_LAYER))
            
        model=RWKV6(config,device=device,dtype=dtype)
        model.load_state_dict(w)
        model.to(model.device)
        return model
if __name__ == '__main__':
    import sys
    sys.path.append(r"/home/aistudio/rwkvinput")
    from tokenizer.tokenization_rwkv_world import RWKVWorldTokenizer
    tokenizer=RWKVWorldTokenizer(vocab_file=r"/home/aistudio/rwkvinput/tokenizer/rwkv_vocab_v20230424.txt")
    model = RWKV6.from_pretrained('RWKV-x060-World-1B6-v2-20240208-ctx4096.pth',dtype=torch.float16)
    input_ids=tokenizer(["目标文件（Object File）指的是编译器对源代码进行编译后生成的文件。例如编译生成的未链接的中间文件 hello.o ，以及最终经过链接生成的不带文件扩展名的可执行文件 hello 都属于目标文件。目标文件包含编译后的机器指令、数据（全局变量、字符串等），以及链接和运行时需要的符号表、调试信息、字符串等。目前主流的目标文件格式是 Windows 系统采用的 PE ( Portable Executable，包括未链接的 .obj 文件和可执行的 .exe 文件 )和 Linux 系统中采用的 ELF ( Executable Linkable Format，包括未链接的 .o 文件和可执行文件 )。\
6.1 ELF 文件格式解析\
       ELF 文件是用在 Linux 系统下的一种目标文件存储格式。典型的目标文件有以下 3 类：\
       可重定向文件（Relocatable File）:还未经过链接的目标文件。其内容包含经过编译器编译的汇编代码和数据，用于和其他可重定向文件一起链接形成一个可执行文件或者动态库。通常文件扩展名为 .o 。\
       可执行文件（Executable File）：经过链接器链接，可被 Linux 系统直接执行的目标文件。其内容包含可以运行的机器指令和数据。通常此文件无扩展名。\
       动态库文件（Shared Object）：动态库文件是共享程序代码的一种方式，其内容和可重定向文件类似，包含可用于链接的代码和程序，可看作多个可重定向文件、动态库一起链接形成一个可执行文件。程序运行时，动态链接器负责在需要的时候动态加载动态库文件到内存\
"], return_tensors="pt").input_ids.to("cuda")
    import time
    for i in range(10):
        ti=time.time()
        re=model(input_ids)
        print(time.time()-ti)
        time.sleep(1)
