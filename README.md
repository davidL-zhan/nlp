# Chinese-English Transformer Translation

这是一个从零实现的中译英 Transformer 学习项目。项目包含完整的数据处理、词表构建、样本编码、DataLoader、Transformer 模型、训练循环和推理脚本，适合用来理解 Seq2Seq Transformer 的基本训练流程。

## 项目功能

- 使用 `jieba` 对中文句子分词。
- 使用简单空格分词处理英文句子。
- 从训练集构建中英文词表。
- 将平行语料编码成 token id。
- 使用 PyTorch 实现 Encoder-Decoder Transformer。
- 支持训练、验证、保存 checkpoint。
- 支持命令行单句推理和交互式翻译。

## 目录结构

```text
.
├── Transformer.py             # Transformer 模型、注意力、mask 工具
├── tokenizer.py               # 分词、JSON 读写、文本编码工具
├── download.py                # 将 data/cmn.txt 切分为 train/valid/test
├── build_vocab.py             # 根据训练集构建中英文词表
├── encode_dataset.py          # 将文本数据编码成 .pt 训练数据
├── dataset_dataloader.py      # Dataset、DataLoader 和动态 padding
├── train.py                   # 模型训练入口
├── inference.py               # 加载 checkpoint 进行推理
├── demo.py                    # 终端交互式翻译 demo
├── utils.py                   # 训练辅助函数
├── requirements.txt           # Python 依赖
├── data/                      # 原始数据、切分数据、词表和编码数据
├── checkpoints/               # 训练保存的模型参数
└── test/                      # 一些学习和测试脚本
```

## 环境安装

建议使用 Python 虚拟环境或 Conda 环境。

```powershell
pip install -r requirements.txt
```

`requirements.txt` 默认安装 CUDA 12.1 对应的 PyTorch 版本。如果本机没有 NVIDIA GPU，或者 CUDA 版本不同，需要根据 PyTorch 官方安装命令调整 `torch` 安装方式。

## 数据准备

项目默认使用 `data/cmn.txt` 作为原始中英平行语料。原始文件格式通常为：

```text
English sentence<TAB>中文句子<TAB>来源信息
```

本项目做的是中译英：

- source：中文，保存为 `.zh`
- target：英文，保存为 `.en`

运行数据切分：

```powershell
python download.py
```

生成文件：

```text
data/train.zh
data/train.en
data/valid.zh
data/valid.en
data/test.zh
data/test.en
```

## 构建词表

```powershell
python build_vocab.py
```

输出：

```text
data/vocab/src_zh_jieba_vocab.json
data/vocab/src_zh_jieba_id2token.json
data/vocab/tgt_en_vocab.json
data/vocab/tgt_en_id2token.json
```

特殊 token 固定为：

```text
<pad> = 0
<unk> = 1
<bos> = 2
<eos> = 3
```

注意：训练集、验证集、测试集和推理阶段必须使用同一套词表，不能分别建词表。

## 编码数据

```powershell
python encode_dataset.py
```

输出：

```text
data/cmn_eng_encoded/train.pt
data/cmn_eng_encoded/valid.pt
data/cmn_eng_encoded/test.pt
data/cmn_eng_encoded/meta.json
```

每条样本会保存中文原文、英文原文、tokens 和对应的 token ids。

## 检查 DataLoader

```powershell
python dataset_dataloader.py
```

该脚本会打印一个 batch 的 `src_ids`、`tgt_input`、`tgt_output` 形状，用来确认动态 padding 和 decoder 输入输出错位是否正常。

## 训练模型

```powershell
python train.py
```

默认会自动选择 CUDA 或 CPU，并保存：

```text
checkpoints/last.pt
checkpoints/best.pt
```

常用调试命令：

```powershell
python train.py --epochs 1 --max_train_batches 2 --max_valid_batches 2
```

常用参数：

```text
--epochs                 训练轮数
--batch_size             batch 大小
--lr                     学习率
--embeding_size          token embedding 维度
--num_heads              多头注意力头数
--num_encoder_layers     Encoder 层数
--num_decoder_layers     Decoder 层数
--mlp_hidden_size        前馈网络隐藏层维度
--device                 指定 cpu 或 cuda
```

## 推理翻译

单句翻译：

```powershell
python inference.py --text "你好。"
```

显示分词和生成 token：

```powershell
python inference.py --text "你好。" --show_tokens
```

进入交互模式：

```powershell
python inference.py
```

也可以使用简化版 demo：

```powershell
python demo.py
```

## 模型输入输出形状

项目使用 `batch_first` 格式：

```text
src_ids: [batch_size, src_len]
tgt_input: [batch_size, tgt_len]
embedding output: [batch_size, seq_len, embeding_size]
model logits: [batch_size, tgt_len, tgt_vocab_size]
```

训练时 decoder 使用错位输入：

```text
tgt:        <bos> I like cats . <eos>
tgt_input:  <bos> I like cats .
tgt_output:       I like cats . <eos>
```

## Git 注意事项

`data/`、`checkpoints/` 和 `paper/` 通常不建议提交到普通 Git 仓库，尤其是 `.pt` checkpoint 文件可能很大。如果需要管理模型文件，建议使用 Git LFS 或单独的模型存储方式。

当前 `.gitignore` 已经忽略：

```text
data/
checkpoints/
paper/
```

如果这些文件曾经被提交过，需要先从 Git 追踪中移除后，`.gitignore` 才会生效。

## 推荐运行顺序

```powershell
python download.py
python build_vocab.py
python encode_dataset.py
python dataset_dataloader.py
python train.py
python inference.py --text "你好。"
```
