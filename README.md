# chunk_RWKV6

使用分块并行优化RWKV的prefill和训练速度

# benchmark

![](img/1.png)

在3090上测试，batch_size=1, head_num=32, head_size=64, channel=2048, 与RWKV6-1.6b设置相同



# Todolist

- 优化速度
- 增加反向传播
- 与continous batching结合
- 支持continous batching的后端推理引擎

# 参考



[flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention/tree/main)

[ChatRWKV](https://github.com/BlinkDL/ChatRWKV/tree/main)

