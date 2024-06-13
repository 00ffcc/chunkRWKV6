import torch
from typing import Dict, List, Optional, Union,Tuple
import time
from tokenizer.tokenization_rwkv_world import RWKVWorldTokenizer
from model import RWKV6
from queue import Queue 
from einops import rearrange
from functools import partial
import asyncio
import os
from sample.sample import sample
def make_async(func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """
    from https://github.com/vllm-project/vllm/blob/main/vllm/utils.py
    Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.

    """

    def _async_wrapper(*args, **kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=p_func)

    return _async_wrapper

class request_manager:
    def __init__(self, device):
        self.device = device
        self.tokenizer = RWKVWorldTokenizer(vocab_file=r"D:\rwkv_input\tokenizer\rwkv_vocab_v20230424.txt")
        self.model = RWKV6.from_pretrained(r"D:\rwkv_input\model\rwkv6\RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth",dtype=torch.float16)
        self.queue = Queue()
    async def generate(self, prompt, sampling_params):
        # prefill
        state = self.model.empty_states(1, device=torch.cpu)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        loop=asyncio.get_running_loop()
        fut=loop.create_future()
        self.queue.put((fut,(input_ids,state)))
        words = [[]]
        await fut
        # generate
        while True:
            logprobs, states = fut.result()
            words, states = sample(logprobs, states, words, sampling_params)
    def shedule(self, maxSeqLen = -1):
        input_idss = torch.zeros((1,0), dtype=torch.long, device=torch.cpu)
        states = []
        seq_idx = []
        seq_idx_top = 0
        fut_map = {}
        while self.queue.qsize():
            fut, (input_ids, state) = self.queue.get()
            if maxSeqLen != -1 and input_idss.shape[0]+input_ids.shape[0]*input_ids.shape[1] > maxSeqLen:
                break
            self.queue.pop()
            for i in range(input_ids.shape[0]):
                seq_idx += [seq_idx_top]*input_ids.shape[1]
                fut_map[fut] = fut_map.get(fut, []) + [seq_idx_top]
                seq_idx_top += 1
            states.append(state)
            input_ids = rearrange(input_ids, 'b s -> (b s)')
            input_idss = torch.cat([input_idss, input_ids], dim=0)
        states=[torch.cat([j[i] for j in states], dim=0, device=self.device) for i in range(3)]
        seq_idx = torch.tensor(seq_idx, device=self.device).unsqueeze(0)
        input_idss = input_idss.to(self.device).unsqueeze(0)
        return input_idss, states, fut_map, seq_idx

    async def run(self):
        input_idss, states, fut_map, seq_idx = self.shedule()
        with torch.no_grad():
            logits, states = make_async(self.model)(input_idss, seq_idx, states)
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        for fut, idxs in fut_map.items():
            fut.set_result(logprobs[idxs], states[idxs])
    async def run_loop(self):
        if self.seq_queue.qsize():
            self.run()
        await asyncio.sleep(0)
