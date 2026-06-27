"""
从零实现的 Seq2Seq Transformer 模型。

这个文件只定义模型结构和注意力 mask 工具函数，不负责数据读取、训练循环或推理交互。
项目中的调用关系是：
    train.py     -> 使用 Transformer 训练中译英模型；
    inference.py -> 加载 checkpoint 后使用 Transformer 做逐词解码；
    demo.py      -> 复用 inference.py 的推理函数做终端演示。

本实现采用 batch_first 风格：
    输入 token id 形状为 [batch_size, seq_len]；
    embedding 后形状为 [batch_size, seq_len, embeding_size]。

mask 语义统一为：
    True  = 该位置需要被屏蔽；
    False = 该位置可以被注意力看到。
"""

import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    """
    缩放点积注意力 Scaled Dot-Product Attention。

    输入：
        Q: [B, heads, Lq, d_k]
        K: [B, heads, Lk, d_k]
        V: [B, heads, Lv, d_v]

    输出：
        output: [B, heads, Lq, d_v]
        attn:   [B, heads, Lq, Lk]

    这里是多头注意力中的“单个注意力计算核心”，
    Q/K/V 的线性投影和多头拆分由 MultiHeadAttention 负责。
    """

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 对10%的神经元做随机失活, 防止过拟合
        self.softmax = nn.Softmax(dim=-1)  # 对最后的一维进行归一化，转换成概率分布

    def forward(self, Q, K, V, mask=None):
        # Q,K,V的维度都是[batch_size, seq_len, embeding_size]
        # batch_size:一次送到模型的句子个数,seq_len:句子的token数量,embeding_size:词向量的维度default = 512
        # Q:输入的query向量 维度 batch_size,heads,seq_len_q,d_k
        # K:输入的key向量 维度 batch_size,heads,seq_len_k,d_k
        # V:输入的value向量 维度 batch_size,heads,seq_len_v,d_v
        d_k = Q.size(-1)  # d_k = d_v = embeding_size 就是词向量的维度 default = 512
        # 计算Q*K.T/sqrt(d_k)得到注意力分数 维度 batch_size,heads,seq_len_q,seq_len_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # 对scores进行遮罩
        if mask is not None:
            scores = scores.masked_fill(mask, -1e10)
        # batch_size,heads,seq_len_q,seq_len_k ,得到注意力的权重 概率
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        # batch_size,heads,seq_len_q,d_v
        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadAttention(nn.Module):  # 多头注意力
    """
    多头注意力模块。

    它完成三件事：
    1. 用 W_q / W_k / W_v 把输入映射成 Q/K/V。
    2. 把 embedding 维度拆成多个 head 并行计算注意力。
    3. 把多个 head 的结果拼回 embeding_size，并做残差连接和 LayerNorm。

    在 EncoderLayer 中，它用于 encoder self-attention。
    在 DecoderLayer 中，它分别用于 decoder self-attention 和 cross-attention。
    """

    def __init__(
        self,
        embeding_size,  # 词向量的维度 default = 512
        num_heads,  # 多头注意力的头数 default = 8
        dropout=0.1,  # 对10%的神经元做随机失活, 防止过拟合
    ):
        super().__init__()
        assert embeding_size % num_heads == 0, "embeding_size必须能被num_heads整除"
        self.d_k = (
            embeding_size // num_heads
        )  # 每个头的维度 default = 64 例如 512/8 = 64
        self.num_heads = num_heads  # 多头注意力的头数 default = 8
        # 将输入映射到Q,K,V的三个向量
        self.W_q = nn.Linear(embeding_size, embeding_size)
        self.W_k = nn.Linear(embeding_size, embeding_size)
        self.W_v = nn.Linear(embeding_size, embeding_size)
        self.fc = nn.Linear(embeding_size, embeding_size)  # 多头拼接后的映射
        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embeding_size)  # 用于残差后续归一化

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # Q:输入的query向量 维度 batch_size,seq_len,embeding_size  -> batch_size,seq_len,num_heads,d_k
        # -> batch_size,num_heads,seq_len,d_k  为了让每个注意力头独立处理整个序列,方便后续计算注意力权重
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 计算注意力权重
        output, attn = self.attention(Q, K, V, mask)
        # batch_size,heads,seq_len_q,d_v -> batch_size,seq_len_q,heads,d_v  -> batch_size,seq_len_q,num_heads*d_v
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_k)
        )
        output = self.fc(output)  # 让输入和输出维度相同 ,方便残差链接
        output = self.dropout(output)
        return self.layer_norm(output + q), attn


class Mlp(nn.Module):
    """
    Transformer 中的前馈网络 Feed Forward Network。

    每个 token 位置独立经过两层全连接：
        embeding_size -> Mlp_hidden_size -> embeding_size

    注意：
        这里的 MLP 不在不同 token 之间交换信息；
        token 之间的信息交互主要由 attention 完成。
    """

    def __init__(
        self,
        embeding_size=512,  # 词向量的维度 default = 512
        output_size=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embeding_size, output_size)
        self.fc2 = nn.Linear(output_size, embeding_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embeding_size)

    def forward(self, x):  # x: batch_size,seq_len,embeding_size
        # batch_size, seq_len, embeding_size -> batch_size, seq_len, output_size -> batch_size, seq_len, embeding_size
        out = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return self.layer_norm(x + out)


class EncoderLayer(nn.Module):
    """
    单层 Transformer Encoder。

    结构：
        src self-attention -> feed-forward MLP

    输入和输出形状保持一致：
        [batch_size, src_len, embeding_size]
    """

    def __init__(
        self, embeding_size=512, num_heads=8, Mlp_hidden_size=2048, dropout=0.1
    ):
        super().__init__()
        # 多头注意力机制
        # 输入为原始输入序列 实现序列内部的信息交互,输入序列和输出序列的维度相同
        self.mutihead_attention = MultiHeadAttention(embeding_size, num_heads, dropout)
        # MLP 独立进行非线性变换
        self.mlp = Mlp(embeding_size, Mlp_hidden_size, dropout)

    def forward(self, src, src_mask=None):
        # src : batch_size, seq_len, embeding_size 输入序列张量
        # src_mask : batch_size, seq_len, seq_len 屏蔽padding位置，避免关注无效区域
        # Q K V　=  src
        out, _ = self.mutihead_attention(src, src, src, src_mask)
        out = self.mlp(out)

        return out


class DecoderLayer(nn.Module):
    """
    单层 Transformer Decoder。

    结构：
        1. masked tgt self-attention：
           目标端只能看已经生成的 token，不能看未来 token。
        2. cross-attention：
           目标端 token 根据 encoder 输出的 memory 读取源语言信息。
        3. feed-forward MLP。
    """

    def __init__(
        self, embeding_size=512, num_heads=8, Mlp_hidden_size=2048, dropout=0.1
    ):
        super().__init__()
        # mask 多头注意力机制
        # 输入tgt(目标序列）在翻译任务中是 已经生成的前几个翻译词
        # 计算目标序列内部的自注意力，通过mask遮挡未来的token
        self.mutihead_attention = MultiHeadAttention(embeding_size, num_heads, dropout)
        # cross 多头注意力机制 和encoder做交互
        # 输入为Q= 当前解码器的输出,k=v=来自编码器的memory(原序列的上下文信息)
        self.cross_attention = MultiHeadAttention(embeding_size, num_heads, dropout)
        self.mlp = Mlp(embeding_size, Mlp_hidden_size, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt : batch_size, seq_len, embeding_size 输入序列张量
        # memory : batch_size, seq_len, embeding_size 编码器的输出张量
        out, _ = self.mutihead_attention(tgt, tgt, tgt, tgt_mask)
        out, _ = self.cross_attention(out, memory, memory, memory_mask)
        out = self.mlp(out)
        return out


class PositionEncoding(nn.Module):
    """
    正弦/余弦位置编码。

    Transformer 本身没有 RNN 的顺序递推结构，
    因此需要把 token 的位置信息加到 embedding 上。

    pe 的形状是 [1, max_len, embeding_size]，
    forward 时根据当前序列长度截取前 seq_len 个位置。

    注意：
        当前实现要求 embeding_size 是偶数，因为偶数维使用 sin，奇数维使用 cos。
    """

    def __init__(self, embeding_size=512, max_len=5000):
        super().__init__()
        # embeding_size :每个词向量的维度 , max_len : 最大序列长度
        # 初始化位置编码向量，形状为(max_len, embeding_size)
        pe = torch.zeros(max_len, embeding_size)
        # 定义记录每个token位置的索引，0-max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # torch.arange(0, embeding_size, 2) = 2i
        # (-math.log(10000.0) / embeding_size) = -log(10000.0) / embeding_size
        div_term = torch.exp(
            torch.arange(0, embeding_size, 2).float()
            * (-math.log(10000.0) / embeding_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe : 1,max_len, embeding_size
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: 输入的embeding 形状 batch_size, seq_len, embeding_size
        seq_len = x.size(1)
        # 取前seq_len的位置,形状为(1, seq_len, embeding_size)
        return x + self.pe[:, :seq_len, :]  # 广播相加


class Encoder(nn.Module):
    """
    Transformer Encoder 堆叠。

    处理流程：
        token id -> embedding -> position encoding -> 多层 EncoderLayer

    输出 memory 会传给 Decoder 做 cross-attention。
    """

    def __init__(
        self,
        vocab_size,
        embeding_size=512,
        num_heads=8,
        num_layers=6,
        Mlp_hidden_size=2048,
        dropout=0.1,
        max_len=5000,
        pad_id=0,
    ):
        super().__init__()
        self.embeding = nn.Embedding(
            vocab_size, embeding_size, padding_idx=pad_id
        )  # [vocab_size, embeding_size] vocab_size个词向量，每个词向量的维度为embeding_size
        self.position_encoding = PositionEncoding(embeding_size, max_len)
        # 构建编码器的堆叠结构 堆叠的层数num_layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embeding_size, num_heads, Mlp_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, src_mask=None):
        # 将输入的token ID转换成embedding向量
        # 输出 shape batch_size, seq_len, embeding_size
        # 乘上sqrt(embeding_size) 进行缩放,让后续注意力计算更稳定
        out = self.embeding(src) * math.sqrt(self.embeding.embedding_dim)
        out = self.position_encoding(out)
        # 逐层进入encoderlayer
        for encoder_layer in self.encoder_layers:
            out = encoder_layer(out, src_mask)
        return out  # batch_size, seq_len, embeding_size


class Decoder(nn.Module):
    """
    Transformer Decoder 堆叠。

    处理流程：
        target token id -> embedding -> position encoding -> 多层 DecoderLayer -> 词表投影

    最终输出 logits，形状是：
        [batch_size, tgt_len, tgt_vocab_size]
    """

    def __init__(
        self,
        vocab_size,
        embeding_size,
        num_heads=8,
        num_layers=6,
        Mlp_hidden_size=2048,
        dropout=0.1,
        max_len=5000,
        pad_id=0,
    ):
        super().__init__()
        self.embeding = nn.Embedding(
            vocab_size, embeding_size, padding_idx=pad_id
        )  # 将目标序列的token ID转换成embedding向量 维度为 embeding_size
        self.position_encoding = PositionEncoding(embeding_size, max_len)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embeding_size, num_heads, Mlp_hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        # 输出投影层 将decoder的输出元词汇表的大小,从而得到每个token的预测概率分布
        self.fc_out = nn.Linear(embeding_size, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # tgt 目标序列 解码器的输入 ,memory 编码器的输出 也叫上下文信息
        # tgt_mask 目标序列的mask 用来屏蔽未来的位置,memory_mask:用来屏蔽pad
        out = self.embeding(tgt) * math.sqrt(self.embeding.embedding_dim)
        out = self.position_encoding(out)
        for decoder_layer in self.decoder_layers:
            out = decoder_layer(out, memory, tgt_mask, memory_mask)
        return self.fc_out(out)


class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer。

    forward 输入：
        src:         中文源句 token id，[B, src_len]
        tgt:         英文目标输入 token id，[B, tgt_len]
        src_mask:    encoder self-attention 的 padding mask
        tgt_mask:    decoder self-attention 的 padding + causal mask
        memory_mask: decoder cross-attention 对源句 padding 的 mask

    forward 输出：
        logits: [B, tgt_len, tgt_vocab_size]
    """

    def __init__(
        self,
        src_vocab_size,  # 源语言词表大小
        tgt_vocab_size,  # 目标语言词表大小
        embeding_size=512,  # embeding的向量维度
        num_heads=8,  # 多头注意力的头数
        num_encoder_layers=6,  # 编码器的层数
        num_decoder_layers=6,  # 解码器的层数
        Mlp_hidden_size=2048,  # mlp 隐藏层维度
        dropout=0.1,  # dropout概率
        max_len=5000,  # 最大序列长度
    ):
        super().__init__()
        # 编码器 将源语言token编码为上下文表示
        self.encoder = Encoder(
            src_vocab_size,
            embeding_size,
            num_heads,
            num_encoder_layers,
            Mlp_hidden_size,
            dropout,
            max_len,
        )
        # 解码器 将根据编码器的输出和目标语言的输入生成预测
        self.decoder = Decoder(
            tgt_vocab_size,
            embeding_size,
            num_heads,
            num_decoder_layers,
            Mlp_hidden_size,
            dropout,
            max_len,
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        return self.decoder(
            tgt, memory, tgt_mask, memory_mask
        )  # batch_size, seq_len_tgt, vocab_size


def generate_mask(size, device=None):
    """
    生成 decoder self-attention 使用的 causal mask。

    返回形状是 [size, size]。
    True 表示未来位置，需要屏蔽；False 表示当前位置或历史位置，可以看到。
    """
    # 防止解码器看到未来的token , size为序列长度
    # torch.triu:生成一个上三角，不含对角线
    mask = torch.triu(
        torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
    ).bool()
    # 上三角部分对应“当前位置之后的未来 token”，这些位置需要被屏蔽。
    return mask  # True 表示需要屏蔽


def make_src_mask(src, src_pad_id):
    # src 的形状是 [batch_size, src_len]，里面存的是源语言句子的 token id。
    # src.eq(src_pad_id) 会得到一个 bool mask，形状仍然是 [batch_size, src_len]。
    # 其中 True 表示当前位置是 <pad>，也就是注意力里需要被屏蔽的位置。
    #
    # SelfAttention 里的 scores 形状是 [batch_size, num_heads, query_len, key_len]。
    # 因此这里通过两次 unsqueeze 把 mask 变成 [batch_size, 1, 1, src_len]。
    # 这样它可以自动广播到每个 attention head、每个 query 位置上，
    # 让 encoder self-attention 和 decoder cross-attention 都不会关注源句子的 padding。
    return src.eq(src_pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, tgt_pad_id):
    # tgt 的形状是 [batch_size, tgt_len]，通常对应 decoder 的输入 tgt_input。
    # tgt_pad_mask 用来屏蔽目标句子里的 <pad>，True 表示该位置需要被屏蔽。
    # 两次 unsqueeze 后形状变成 [batch_size, 1, 1, tgt_len]，
    # 可以广播到 attention scores 的 [batch_size, num_heads, tgt_len, tgt_len]。
    tgt_pad_mask = tgt.eq(tgt_pad_id).unsqueeze(1).unsqueeze(2)

    # causal_mask 用来屏蔽未来 token，防止 decoder 在训练时偷看当前位置之后的答案。
    # generate_mask(tgt_len) 返回 [tgt_len, tgt_len]：
    # - True 表示未来位置，需要屏蔽；
    # - False 表示当前位置和历史位置，可以被看到。
    # unsqueeze 后变成 [1, 1, tgt_len, tgt_len]，
    # 可以广播到 batch 维度和多头注意力维度。
    tgt_len = tgt.size(1)
    causal_mask = generate_mask(tgt_len, device=tgt.device).unsqueeze(0).unsqueeze(0)

    # 目标端 self-attention 需要同时满足两个约束：
    # 1. 不能看 <pad>；
    # 2. 不能看未来 token。
    # 两个 mask 都采用 True = 需要屏蔽 的语义，所以用按位或 | 合并。
    # 返回形状是 [batch_size, 1, tgt_len, tgt_len]。
    return tgt_pad_mask | causal_mask


if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    # 初始化
    model = Transformer(src_vocab_size, tgt_vocab_size)
    src = torch.randint(
        0, src_vocab_size, (32, 10)
    )  # 原序列batch=32 src_len=10 每个元素是token ID
    tgt = torch.randint(
        0, tgt_vocab_size, (32, 20)
    )  # 目标序列batch=32 tgt_len=20 每个元素是token ID
    tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)
    out = model(src, tgt, tgt_mask=tgt_mask)
    # 每个目标token对应此表中每个词的预测概率
    print(out.shape)  # batch_size, seq_len_tgt, vocab_size_tgt
