# `model.py` 中 Transformer 全流程与数据形状变化

本文档根据当前项目的 [`model.py`](model.py) 讲解这个 Seq2Seq Transformer 的完整前向流程，以及每一步张量形状如何变化。

这个实现用于中译英任务。数据进入 `model.py` 前，已经在 `data.py` 中完成了文本清洗、SentencePiece 编码、batch padding，以及 `tgt_input` / `tgt_output` 的错位构造。因此 `model.py` 只处理整数 token id、embedding、attention mask 和 logits。

## 1. 符号约定

下文使用这些符号表示张量维度：

| 符号 | 含义 |
| --- | --- |
| `B` | batch size，一个 batch 中的句子数量 |
| `S` | source length，源语言序列长度，也就是中文 token 数 |
| `T` | target length，目标语言输入序列长度，也就是英文 decoder 输入 token 数 |
| `E` | `embeding_size`，embedding 维度 |
| `H` | `num_heads`，多头注意力头数 |
| `D` | 每个 head 的维度，`D = E / H` |
| `F` | `Mlp_hidden_size`，前馈网络隐藏层维度 |
| `V_src` | 源语言词表大小 |
| `V_tgt` | 目标语言词表大小 |

默认情况下，`model.py` 里的 `E=512`，`H=8`，所以每个注意力头的维度是：

```text
D = E / H = 512 / 8 = 64
```

注意：代码里使用的是变量名 `embeding_size`，少了一个 `d`，本文档沿用代码里的拼写。

## 2. 进入模型前的数据形状

训练时，`data.py` 的 collator 会把目标句子错开一位：

```text
tgt_ids:     <bos> I like NLP . <eos>
tgt_input:   <bos> I like NLP .
tgt_output:        I like NLP . <eos>
```

送入 `Transformer.forward()` 的主要张量是：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `src` | `[B, S]` | 中文源句 token id |
| `tgt` | `[B, T]` | 英文目标输入 token id，即 `tgt_input` |
| `src_mask` | `[B, 1, 1, S]` | 源句 padding mask |
| `tgt_mask` | `[B, 1, T, T]` | 目标句 padding mask + causal mask |
| `memory_mask` | `[B, 1, 1, S]` | decoder cross-attention 读取 encoder memory 时使用的源句 padding mask |

最终模型输出：

| 名称 | 形状 | 含义 |
| --- | --- | --- |
| `logits` | `[B, T, V_tgt]` | 每个目标位置对整个目标词表的未归一化预测分数 |

`logits` 不是概率。训练时通常直接交给 `CrossEntropyLoss`，推理时可以取 `argmax`，如果要得到概率，需要额外做 `softmax(logits, dim=-1)`。

## 3. 总体调用链

`Transformer.forward()` 的核心流程是：

```text
src [B, S]
    -> Encoder
    -> memory [B, S, E]

tgt [B, T] + memory [B, S, E]
    -> Decoder
    -> logits [B, T, V_tgt]
```

对应代码逻辑：

```python
memory = self.encoder(src, src_mask)
logits = self.decoder(tgt, memory, tgt_mask, memory_mask)
```

也就是说：

1. Encoder 先把中文源句编码成上下文表示 `memory`。
2. Decoder 再根据英文目标输入和 `memory` 预测下一个英文 token。
3. 训练时，`logits` 会和 `tgt_output` 计算交叉熵。
4. 推理时，decoder 从 `<bos>` 开始逐步生成，每一步取最后一个位置的预测 token。

## 4. Mask 语义

当前实现统一使用 PyTorch bool mask：

```text
True  = 这个位置需要被屏蔽
False = 这个位置可以被 attention 看到
```

在 `SelfAttention.forward()` 里，mask 的使用方式是：

```python
scores = scores.masked_fill(mask, -1e10)
```

也就是把需要屏蔽的位置变成一个极小值。随后对 `scores` 做 softmax 时，这些位置的注意力权重会接近 0。

## 5. `generate_mask(size)`：生成 causal mask

`generate_mask(size)` 用于 decoder self-attention，防止当前位置看到未来 token。

输入：

```text
size = T
```

输出：

```text
causal_mask: [T, T]
```

例如 `T=4` 时，mask 逻辑类似：

```text
[
  [False, True,  True,  True ],
  [False, False, True,  True ],
  [False, False, False, True ],
  [False, False, False, False],
]
```

含义是：

- 第 0 个位置只能看自己，不能看 1、2、3。
- 第 1 个位置可以看 0、1，不能看 2、3。
- 第 2 个位置可以看 0、1、2，不能看 3。
- 第 3 个位置可以看 0、1、2、3。

## 6. `make_src_mask(src, src_pad_id)`：源句 padding mask

输入：

```text
src: [B, S]
```

代码：

```python
return src.eq(src_pad_id).unsqueeze(1).unsqueeze(2)
```

形状变化：

```text
src.eq(src_pad_id):         [B, S]
unsqueeze(1):               [B, 1, S]
unsqueeze(2):               [B, 1, 1, S]
```

最终：

```text
src_mask: [B, 1, 1, S]
```

这个 mask 会用于两个地方：

1. Encoder self-attention：防止源句 token 关注 `<pad>`。
2. Decoder cross-attention：防止目标句 token 从源句的 `<pad>` 里读取信息。

## 7. `make_tgt_mask(tgt, tgt_pad_id)`：目标句 self-attention mask

输入：

```text
tgt: [B, T]
```

第一部分是 padding mask：

```python
tgt_pad_mask = tgt.eq(tgt_pad_id).unsqueeze(1).unsqueeze(2)
```

形状变化：

```text
tgt.eq(tgt_pad_id):         [B, T]
unsqueeze(1):               [B, 1, T]
unsqueeze(2):               [B, 1, 1, T]
```

第二部分是 causal mask：

```python
causal_mask = generate_mask(T).unsqueeze(0).unsqueeze(0)
```

形状变化：

```text
generate_mask(T):           [T, T]
unsqueeze(0):               [1, T, T]
unsqueeze(0):               [1, 1, T, T]
```

两者合并：

```python
return tgt_pad_mask | causal_mask
```

广播后最终形状：

```text
tgt_mask: [B, 1, T, T]
```

它同时完成两件事：

1. 屏蔽目标句中的 `<pad>`。
2. 屏蔽当前位置之后的未来 token。

## 8. Encoder 全流程

Encoder 接收：

```text
src:      [B, S]
src_mask: [B, 1, 1, S]
```

### 8.1 Token embedding

代码：

```python
out = self.embeding(src) * math.sqrt(self.embeding.embedding_dim)
```

形状变化：

```text
src:                     [B, S]
self.embeding(src):      [B, S, E]
乘 sqrt(E):              [B, S, E]
```

乘 `sqrt(E)` 是 Transformer 里的常见缩放操作，用来调整 embedding 的数值尺度。

### 8.2 Position encoding

`PositionEncoding` 里保存的 `pe` 形状是：

```text
pe: [1, max_len, E]
```

forward 时根据当前序列长度取前 `S` 个位置：

```python
x + self.pe[:, :seq_len, :]
```

形状变化：

```text
out:             [B, S, E]
pe[:, :S, :]:    [1, S, E]
相加后:          [B, S, E]
```

这里 `[1, S, E]` 会广播到 `[B, S, E]`。

### 8.3 多层 EncoderLayer

Encoder 中有 `num_encoder_layers` 层，每一层结构是：

```text
src self-attention -> MLP
```

每一层的输入输出形状保持不变：

```text
[B, S, E] -> [B, S, E]
```

最终 Encoder 输出：

```text
memory: [B, S, E]
```

这个 `memory` 会传给 Decoder 做 cross-attention。

## 9. EncoderLayer 内部形状

EncoderLayer 的输入：

```text
src:      [B, S, E]
src_mask: [B, 1, 1, S]
```

### 9.1 Encoder self-attention

调用：

```python
out, _ = self.mutihead_attention(src, src, src, src_mask)
```

这里：

```text
q = src: [B, S, E]
k = src: [B, S, E]
v = src: [B, S, E]
```

因为是 self-attention，所以 Q、K、V 都来自同一个源句表示。

输出：

```text
out: [B, S, E]
```

### 9.2 MLP

调用：

```python
out = self.mlp(out)
```

形状变化：

```text
输入:        [B, S, E]
fc1:         [B, S, F]
ReLU:        [B, S, F]
Dropout:     [B, S, F]
fc2:         [B, S, E]
残差 + LN:   [B, S, E]
```

MLP 是逐 token 独立处理的，不在不同 token 之间交换信息。token 之间的信息交换主要发生在 attention 中。

## 10. MultiHeadAttention 内部形状

`MultiHeadAttention.forward(q, k, v, mask)` 是整个模型中最关键的形状变化位置。

输入一般是：

```text
q: [B, Lq, E]
k: [B, Lk, E]
v: [B, Lv, E]
```

在 self-attention 中，通常 `Lq = Lk = Lv`。

在 decoder cross-attention 中：

```text
Lq = T
Lk = Lv = S
```

### 10.1 线性投影

代码：

```python
Q = self.W_q(q)
K = self.W_k(k)
V = self.W_v(v)
```

形状：

```text
q: [B, Lq, E] -> Q: [B, Lq, E]
k: [B, Lk, E] -> K: [B, Lk, E]
v: [B, Lv, E] -> V: [B, Lv, E]
```

### 10.2 拆成多个 head

代码：

```python
Q = self.W_q(q).view(B, -1, H, D).transpose(1, 2)
K = self.W_k(k).view(B, -1, H, D).transpose(1, 2)
V = self.W_v(v).view(B, -1, H, D).transpose(1, 2)
```

形状变化：

```text
Q: [B, Lq, E] -> [B, Lq, H, D] -> [B, H, Lq, D]
K: [B, Lk, E] -> [B, Lk, H, D] -> [B, H, Lk, D]
V: [B, Lv, E] -> [B, Lv, H, D] -> [B, H, Lv, D]
```

注意：在标准 attention 中，`Lk` 和 `Lv` 应该相同。

### 10.3 进入 SelfAttention

调用：

```python
output, attn = self.attention(Q, K, V, mask)
```

输入形状：

```text
Q: [B, H, Lq, D]
K: [B, H, Lk, D]
V: [B, H, Lk, D]
```

输出形状：

```text
output: [B, H, Lq, D]
attn:   [B, H, Lq, Lk]
```

### 10.4 合并多个 head

SelfAttention 输出后，需要把多个 head 拼回 embedding 维度：

```python
output = output.transpose(1, 2).contiguous().view(B, -1, H * D)
```

形状变化：

```text
output:              [B, H, Lq, D]
transpose(1, 2):     [B, Lq, H, D]
view(B, -1, H * D):  [B, Lq, E]
```

然后经过输出投影：

```python
output = self.fc(output)
```

形状不变：

```text
[B, Lq, E] -> [B, Lq, E]
```

最后做残差连接和 LayerNorm：

```python
return self.layer_norm(output + q), attn
```

输出：

```text
[B, Lq, E]
```

## 11. SelfAttention 内部形状

SelfAttention 接收已经拆好 head 的 Q、K、V：

```text
Q: [B, H, Lq, D]
K: [B, H, Lk, D]
V: [B, H, Lk, D]
```

### 11.1 计算 attention scores

代码：

```python
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
```

其中：

```text
K:                    [B, H, Lk, D]
K.transpose(-2, -1):  [B, H, D, Lk]
```

矩阵乘法：

```text
Q @ K^T:
[B, H, Lq, D] @ [B, H, D, Lk] -> [B, H, Lq, Lk]
```

所以：

```text
scores: [B, H, Lq, Lk]
```

`scores[b, h, i, j]` 表示第 `b` 个样本、第 `h` 个注意力头中，第 `i` 个 query token 对第 `j` 个 key token 的注意力打分。

### 11.2 应用 mask

代码：

```python
scores = scores.masked_fill(mask, -1e10)
```

mask 会广播到 `scores` 的形状：

```text
scores: [B, H, Lq, Lk]
mask:   可广播到 [B, H, Lq, Lk]
```

不同场景下 mask 的形状：

| 场景 | scores 形状 | mask 形状 |
| --- | --- | --- |
| Encoder self-attention | `[B, H, S, S]` | `[B, 1, 1, S]` |
| Decoder self-attention | `[B, H, T, T]` | `[B, 1, T, T]` |
| Decoder cross-attention | `[B, H, T, S]` | `[B, 1, 1, S]` |

### 11.3 Softmax

代码：

```python
attn = self.softmax(scores)
```

`self.softmax = nn.Softmax(dim=-1)`，所以 softmax 对最后一维做归一化。

由于 `scores` 形状是：

```text
[B, H, Lq, Lk]
```

最后一维是 `Lk`，也就是 key token 维度。

含义是：对每个 query token，在所有 key token 上得到一个注意力分布。

形状不变：

```text
scores: [B, H, Lq, Lk]
attn:   [B, H, Lq, Lk]
```

对任意固定的 `b, h, i`：

```text
sum(attn[b, h, i, :]) = 1
```

### 11.4 加权求和 V

代码：

```python
output = torch.matmul(attn, V)
```

形状：

```text
attn: [B, H, Lq, Lk]
V:    [B, H, Lk, D]
```

矩阵乘法：

```text
[B, H, Lq, Lk] @ [B, H, Lk, D] -> [B, H, Lq, D]
```

输出：

```text
output: [B, H, Lq, D]
```

## 12. Decoder 全流程

Decoder 接收：

```text
tgt:         [B, T]
memory:      [B, S, E]
tgt_mask:    [B, 1, T, T]
memory_mask: [B, 1, 1, S]
```

### 12.1 Target embedding

代码：

```python
out = self.embeding(tgt) * math.sqrt(self.embeding.embedding_dim)
```

形状变化：

```text
tgt:                    [B, T]
self.embeding(tgt):     [B, T, E]
乘 sqrt(E):             [B, T, E]
```

### 12.2 Position encoding

形状变化：

```text
out:             [B, T, E]
pe[:, :T, :]:    [1, T, E]
相加后:          [B, T, E]
```

### 12.3 多层 DecoderLayer

Decoder 中有 `num_decoder_layers` 层，每一层结构是：

```text
masked tgt self-attention -> cross-attention -> MLP
```

每层输入输出形状保持：

```text
[B, T, E] -> [B, T, E]
```

### 12.4 输出投影

代码：

```python
return self.fc_out(out)
```

形状变化：

```text
out:             [B, T, E]
fc_out(out):     [B, T, V_tgt]
```

最终输出是 logits：

```text
logits: [B, T, V_tgt]
```

## 13. DecoderLayer 内部形状

DecoderLayer 输入：

```text
tgt:         [B, T, E]
memory:      [B, S, E]
tgt_mask:    [B, 1, T, T]
memory_mask: [B, 1, 1, S]
```

### 13.1 Masked target self-attention

调用：

```python
out, _ = self.mutihead_attention(tgt, tgt, tgt, tgt_mask)
```

这里：

```text
q = tgt: [B, T, E]
k = tgt: [B, T, E]
v = tgt: [B, T, E]
```

拆 head 后：

```text
Q: [B, H, T, D]
K: [B, H, T, D]
V: [B, H, T, D]
```

attention scores：

```text
scores: [B, H, T, T]
```

mask：

```text
tgt_mask: [B, 1, T, T]
```

输出：

```text
out: [B, T, E]
```

这一步让目标端 token 只能看当前位置及之前的 token，不能看未来答案。

### 13.2 Cross-attention

调用：

```python
out, _ = self.cross_attention(out, memory, memory, memory_mask)
```

这里：

```text
q = out:    [B, T, E]
k = memory: [B, S, E]
v = memory: [B, S, E]
```

拆 head 后：

```text
Q: [B, H, T, D]
K: [B, H, S, D]
V: [B, H, S, D]
```

attention scores：

```text
scores: [B, H, T, S]
```

mask：

```text
memory_mask: [B, 1, 1, S]
```

输出：

```text
out: [B, T, E]
```

这一步让每个目标端 token 根据 encoder 输出的 `memory` 读取源语言信息。

### 13.3 MLP

调用：

```python
out = self.mlp(out)
```

形状变化：

```text
输入:        [B, T, E]
fc1:         [B, T, F]
ReLU:        [B, T, F]
Dropout:     [B, T, F]
fc2:         [B, T, E]
残差 + LN:   [B, T, E]
```

## 14. 一次完整 forward 的形状总表

假设：

```text
B = 32
S = 10
T = 20
E = 512
H = 8
D = 64
F = 2048
V_src = 10000
V_tgt = 10000
```

整体形状变化如下：

| 步骤 | 张量 | 形状 |
| --- | --- | --- |
| 输入源句 | `src` | `[32, 10]` |
| 输入目标句 | `tgt` | `[32, 20]` |
| 源句 mask | `src_mask` | `[32, 1, 1, 10]` |
| 目标句 mask | `tgt_mask` | `[32, 1, 20, 20]` |
| Encoder embedding | `src_emb` | `[32, 10, 512]` |
| Encoder position encoding 后 | `out` | `[32, 10, 512]` |
| Encoder self-attention Q | `Q` | `[32, 8, 10, 64]` |
| Encoder self-attention scores | `scores` | `[32, 8, 10, 10]` |
| Encoder self-attention attn | `attn` | `[32, 8, 10, 10]` |
| Encoder self-attention 输出 | `out` | `[32, 10, 512]` |
| Encoder MLP 输出 | `out` | `[32, 10, 512]` |
| Encoder 最终输出 | `memory` | `[32, 10, 512]` |
| Decoder embedding | `tgt_emb` | `[32, 20, 512]` |
| Decoder self-attention Q | `Q` | `[32, 8, 20, 64]` |
| Decoder self-attention scores | `scores` | `[32, 8, 20, 20]` |
| Decoder self-attention 输出 | `out` | `[32, 20, 512]` |
| Decoder cross-attention Q | `Q` | `[32, 8, 20, 64]` |
| Decoder cross-attention K/V | `K/V` | `[32, 8, 10, 64]` |
| Decoder cross-attention scores | `scores` | `[32, 8, 20, 10]` |
| Decoder cross-attention 输出 | `out` | `[32, 20, 512]` |
| Decoder MLP 输出 | `out` | `[32, 20, 512]` |
| 词表投影 | `logits` | `[32, 20, 10000]` |

## 15. 训练和推理时的区别

### 15.1 训练阶段

训练时使用 teacher forcing：

```text
tgt_input  -> 送进 Decoder
tgt_output -> 作为监督标签
```

模型输出：

```text
logits: [B, T, V_tgt]
```

`train.py` 会把它展平后计算 loss：

```text
logits.reshape(-1, V_tgt): [B * T, V_tgt]
tgt_output.reshape(-1):    [B * T]
```

然后用 `CrossEntropyLoss(ignore_index=tgt_pad_id)` 忽略 padding 位置。

### 15.2 推理阶段

推理时没有完整的目标句输入，decoder 从 `<bos>` 开始逐步生成：

```text
step 1: tgt = [<bos>]
step 2: tgt = [<bos>, token_1]
step 3: tgt = [<bos>, token_1, token_2]
...
```

每一步都会重新构造：

```text
tgt_tensor: [1, 当前已生成长度]
tgt_mask:   [1, 1, 当前已生成长度, 当前已生成长度]
```

模型输出：

```text
logits: [1, 当前已生成长度, V_tgt]
```

推理只取最后一个时间步：

```python
next_token_id = argmax(logits[0, -1])
```

如果生成 `<eos>`，或者达到最大生成长度，就停止。

## 16. 最关键的三条理解线

第一，Encoder 的作用是把源句：

```text
src [B, S] -> memory [B, S, E]
```

第二，Decoder 的作用是根据目标端历史 token 和源句 memory 预测下一个 token：

```text
tgt [B, T] + memory [B, S, E] -> logits [B, T, V_tgt]
```

第三，attention 的核心形状永远可以理解成：

```text
Q:      [B, H, Lq, D]
K:      [B, H, Lk, D]
V:      [B, H, Lk, D]
scores: [B, H, Lq, Lk]
attn:   [B, H, Lq, Lk]
output: [B, H, Lq, D] -> [B, Lq, E]
```

只要分清楚 `Lq` 是 query 序列长度，`Lk` 是 key 序列长度，就能推导出 Encoder self-attention、Decoder self-attention 和 Decoder cross-attention 的所有形状。
