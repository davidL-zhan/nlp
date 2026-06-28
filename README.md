# WMT19 Chinese-English Transformer

这是一个从零实现的中译英 Transformer 学习项目。项目使用 HuggingFace `wmt/wmt19` 的 `zh-en` 数据，先训练共享 SentencePiece tokenizer，再用 PyTorch 实现的 encoder-decoder Transformer 进行训练，最后通过 checkpoint 做单句或交互式翻译。

当前项目主流程如下：

```text
WMT19 zh-en
    -> shared SentencePiece tokenizer
    -> PyTorch Dataset / DataLoader
    -> Transformer encoder-decoder
    -> checkpoints/best.pt
    -> infer.py greedy decoding
```

## Environment

建议使用 Python 3.11 的 conda 环境。环境名可以自定义，例如：

```powershell
conda create -n <env-name> python=3.11 -y
```

运行项目前先激活你自己的环境：

```powershell
conda activate <env-name>
```

激活后，下面所有命令都可以直接使用 `python`：

```powershell
python train.py --help
```

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

当前依赖主要包括：

```text
torch==2.5.1+cu121
datasets==5.0.0
sentencepiece
```

## Files

```text
model.py            Transformer 模型结构、位置编码、多头注意力和 mask 工具
spm.py              SentencePiece tokenizer 训练、加载、校验、编码和解码
data.py             WMT19 Dataset 包装、动态 padding collator 和 DataLoader 构建
download.py         下载并缓存 HuggingFace WMT19 zh-en 数据集
train.py            训练入口，负责 loss、optimizer、验证和 checkpoint 保存
infer.py            推理入口，加载 checkpoint 后执行 greedy decoding
tokenizer_demo.py   tokenizer 演示脚本，打印 pieces、ids 和 decode 结果
requirements.txt    Python 依赖
README.md           项目说明
```

默认生成或依赖的本地文件：

```text
data/spm/wmt19_zh_en_unigram_32k.model   SentencePiece 模型
data/spm/wmt19_zh_en_unigram_32k.vocab   SentencePiece 词表
checkpoints/last.pt                      最近一次 epoch 的 checkpoint
checkpoints/best.pt                      验证集 loss 最低的 checkpoint
```

`data/` 和 `checkpoints/` 通常较大，已经在 `.gitignore` 中忽略。

## Quick Start

完整流程一般按下面顺序执行。

### 1. 下载数据

下载并缓存 WMT19 zh-en：

```powershell
python download.py
```

下载后打印一条样本：

```powershell
python download.py --show_sample
```

如果 HuggingFace 本地 cache 已经存在，`datasets` 会优先复用缓存。强制重新下载：

```powershell
python download.py --force
```

### 2. 训练 SentencePiece tokenizer

默认训练 32k 共享词表：

```powershell
python spm.py train
```

默认输出：

```text
data/spm/wmt19_zh_en_unigram_32k.model
data/spm/wmt19_zh_en_unigram_32k.vocab
```

使用更多句对训练 tokenizer：

```powershell
python spm.py train --max_pairs 5000000
```

检查 tokenizer 特殊 token id 和基本编码效果：

```powershell
python spm.py check
```

更详细地查看分词结果：

```powershell
python tokenizer_demo.py
```

自定义输入：

```powershell
python tokenizer_demo.py --text "我喜欢自然语言处理。"
```

### 3. 检查 DataLoader

`data.py` 会读取 HuggingFace 数据集，包装成 PyTorch Dataset，并在 collator 中动态 padding。

```powershell
python data.py check
```

检查内容包括：

```text
tokenizer 路径
词表大小
训练/验证样本数
src_ids / tgt_input / tgt_output shape
样本文本和 SentencePiece pieces
```

### 4. 训练模型

先做一个极小 smoke run，确认数据、tokenizer、模型和 loss 都能跑通：

```powershell
python train.py --epochs 1 --batch_size 2 --hf_train_samples 8 --hf_valid_samples 4 --hf_test_samples 4 --max_train_batches 1 --max_valid_batches 1 --checkpoint_dir checkpoints/smoke --device cuda
```

正常训练示例：

```powershell
python train.py --epochs 10 --batch_size 32 --device cuda
```

默认训练配置：

```text
train split:        wmt/wmt19 zh-en train
valid split:        wmt/wmt19 zh-en validation
训练样本数:          100000
验证样本数:          3981
max_src_tokens:     128
max_tgt_tokens:     128
embedding size:     512
heads:              8
encoder layers:     6
decoder layers:     6
FFN hidden size:    2048
dropout:            0.1
optimizer:          AdamW
loss:               CrossEntropyLoss(ignore_index=tgt_pad_id)
```

使用全量训练集：

```powershell
python train.py --epochs 10 --batch_size 32 --hf_train_samples 0 --device cuda
```

在 CPU 上只检查流程：

```powershell
python train.py --epochs 1 --batch_size 2 --hf_train_samples 8 --hf_valid_samples 4 --max_train_batches 1 --max_valid_batches 1 --device cpu
```

训练输出：

```text
checkpoints/last.pt
checkpoints/best.pt
```

`last.pt` 是最近一次 epoch 的模型，`best.pt` 是验证集 loss 最低的模型。

### 5. 推理

单句推理：

```powershell
python infer.py --text "我喜欢自然语言处理。" --device cuda
```

指定 checkpoint：

```powershell
python infer.py --checkpoint checkpoints/best.pt --text "我喜欢自然语言处理。" --device cuda
```

显示分词和生成 token：

```powershell
python infer.py --text "我喜欢自然语言处理。" --show_tokens --device cuda
```

交互模式：

```powershell
python infer.py --device cuda
```

进入交互模式后，输入中文句子并回车。输入 `q`、`quit`、`exit` 或空行退出。

## Dataset

本项目使用 HuggingFace Datasets 中的 WMT19 中英翻译数据：

```text
dataset name:   wmt/wmt19
config:         zh-en
task:           Chinese -> English machine translation
source side:    zh
target side:    en
default train:  train split
default valid:  validation split
```

每条样本的核心字段是 `translation`，里面包含一对中英文句子：

```python
{
    "translation": {
        "zh": "中文句子",
        "en": "English sentence"
    }
}
```

在这个项目里，中文 `zh` 是 encoder 输入，英文 `en` 是 decoder 的训练目标。`download.py` 会把数据集下载到 HuggingFace 默认缓存目录，后续 `spm.py`、`data.py` 和 `train.py` 再调用 `load_dataset("wmt/wmt19", "zh-en")` 时会优先复用本地缓存。

项目没有把原始 WMT19 数据复制进仓库，原因是数据集和缓存文件都比较大。仓库里只保留数据处理代码，实际数据通过 HuggingFace Datasets 自动下载和缓存。

默认训练时没有直接使用完整 train split，而是为了方便学习和调试，先取前一部分样本：

```text
训练样本: 默认前 100000 条
验证样本: 默认前 3981 条
测试样本: 默认从 validation 子集再取 3981 条
```

这些数量可以通过 `train.py` 参数修改：

```powershell
python train.py --hf_train_samples 100000 --hf_valid_samples 3981 --hf_test_samples 3981
```

如果要使用完整训练集，把 `--hf_train_samples` 设为 `0`：

```powershell
python train.py --hf_train_samples 0
```

数据进入模型前会做三步处理：

```text
1. clean_text: 去掉首尾空白，把句子内部换行替换为空格
2. SentencePiece: 把中文和英文都切成共享子词 token id
3. dynamic padding: 在每个 batch 内按最长句子动态补齐
```

这样做的好处是：原始数据仍然保持 HuggingFace Dataset 格式，tokenizer 和 batch padding 在训练时动态完成，避免提前保存大量中间 `.pt` 文件。

## Data Pipeline

`download.py` 只负责把 HuggingFace 数据缓存到本机。真正训练时，`train.py` 会通过 `data.py -> build_loaders()` 重新读取数据集。

`data.py` 的核心流程：

```text
load_dataset("wmt/wmt19", "zh-en")
    -> select_samples()
    -> WMT19Dataset.__getitem__()
    -> clean_text()
    -> SentencePiece encode
    -> TranslationCollator dynamic padding
    -> src_ids / tgt_input / tgt_output
```

其中 decoder 训练目标采用 teacher forcing：

```text
tgt_ids:     <bos> I like NLP . <eos>
tgt_input:   <bos> I like NLP .
tgt_output:        I like NLP . <eos>
```

这样模型在训练时看到前面的目标 token，预测下一个 token。

## Tokenizer

项目使用中英共享 SentencePiece Unigram tokenizer。

特殊 token 固定为：

```text
<pad> = 0
<unk> = 1
<bos> = 2
<eos> = 3
```

这些 id 会影响：

```text
padding mask
loss ignore_index
decoder 起始 token
推理停止条件
checkpoint meta
```

因此 `spm.py -> load_spm()` 会在加载 tokenizer 后检查特殊 token id。如果 id 不匹配，脚本会直接报错，避免训练和推理使用不一致的 tokenizer。

## Model

`model.py` 是手写 Transformer，采用 batch-first 输入：

```text
token ids:    [batch_size, seq_len]
embedding:    [batch_size, seq_len, embeding_size]
logits:       [batch_size, tgt_len, tgt_vocab_size]
```

主要模块：

```text
SelfAttention        scaled dot-product attention
MultiHeadAttention   Q/K/V 线性投影、多头拆分、拼接和残差 LayerNorm
Mlp                  Transformer feed-forward network
EncoderLayer         encoder self-attention + FFN
DecoderLayer         masked self-attention + cross-attention + FFN
PositionEncoding     sinusoidal position encoding
Transformer          encoder-decoder 总模型
```

mask 语义统一为：

```text
True  = 需要屏蔽
False = 可以被 attention 看到
```

`make_src_mask()` 屏蔽源句中的 `<pad>`，`make_tgt_mask()` 同时屏蔽目标句中的 `<pad>` 和未来 token。

## Checkpoint

`train.py` 保存的 checkpoint 包含：

```text
epoch
model_state_dict
optimizer_state_dict
train_loss
valid_loss
meta
args
```

`infer.py` 不会随便使用默认模型结构，而是从 checkpoint 中读取：

```text
词表大小
SentencePiece model 路径
pad/bos/eos id
embedding size
attention heads
encoder/decoder 层数
FFN hidden size
dropout
max_len
```

这样可以保证推理模型结构和训练时一致。

## Common Commands

查看训练参数：

```powershell
python train.py --help
```

查看推理参数：

```powershell
python infer.py --help
```

查看 tokenizer 参数：

```powershell
python spm.py --help
```

检查当前环境依赖：

```powershell
python -m pip show torch datasets sentencepiece
```

## Troubleshooting

### VS Code / Pylance 爆红

先确认 VS Code 选择的是当前 conda 环境里的 Python 解释器。

如果解释器选错，`torch`、`datasets`、`sentencepiece` 都可能显示无法解析导入。

### SentencePiece model 不存在

如果看到：

```text
SentencePiece model 不存在: data/spm/wmt19_zh_en_unigram_32k.model
```

先运行：

```powershell
python spm.py train
```

### checkpoint 不存在

如果看到：

```text
checkpoint 不存在: checkpoints/best.pt
```

先运行训练，或者用 `--checkpoint` 指向已有 checkpoint。

### HuggingFace 下载失败

`download.py`、`spm.py train` 和 `data.py check` 都依赖 HuggingFace `wmt/wmt19`。如果网络无法访问 HuggingFace，需要先处理网络或镜像配置，再重新运行下载/训练命令。

## Notes

这个项目更偏学习和调试用途，重点是把机器翻译流程拆开看清楚：

```text
文本清洗 -> 子词切分 -> batch padding -> attention mask -> teacher forcing -> checkpoint -> greedy decoding
```

当前推理使用 greedy decoding，没有 beam search，也没有增量 KV cache。这样速度不是最优，但逻辑清楚，适合检查 Transformer 训练和推理的完整链路。
