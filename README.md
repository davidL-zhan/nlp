# WMT19 Chinese-English Transformer

从零实现的中译英 Transformer。当前项目只保留一条主流程：

```text
WMT19 zh-en -> shared SentencePiece -> DataLoader -> Transformer -> checkpoint -> inference
```

## Files

```text
model.py          Transformer 模型和 attention mask
data.py           SentencePiece 训练、编码/解码、WMT19 DataLoader
train.py          训练入口
infer.py          单句推理和交互推理入口
requirements.txt 依赖
```

## Install

```powershell
E:\miniconda\envs\NLP311\python.exe -m pip install -r requirements.txt
```

## Train Tokenizer

如果已有下面文件，可以跳过：

```text
data/spm/wmt19_zh_en_unigram_32k.model
data/spm/wmt19_zh_en_unigram_32k.vocab
```

重新训练 SentencePiece：

```powershell
E:\miniconda\envs\NLP311\python.exe data.py spm
```

检查 tokenizer 和 DataLoader：

```powershell
E:\miniconda\envs\NLP311\python.exe data.py check
```

## Train Model

极小 smoke run：

```powershell
E:\miniconda\envs\NLP311\python.exe train.py --epochs 1 --batch_size 2 --hf_train_samples 8 --hf_valid_samples 4 --hf_test_samples 4 --max_train_batches 1 --max_valid_batches 1 --checkpoint_dir checkpoints/smoke --device cuda
```

正常训练：

```powershell
E:\miniconda\envs\NLP311\python.exe train.py --epochs 10 --batch_size 32 --device cuda
```

默认使用前 `100000` 条 WMT19 训练样本。全量训练：

```powershell
E:\miniconda\envs\NLP311\python.exe train.py --epochs 10 --batch_size 32 --hf_train_samples 0 --device cuda
```

## Inference

单句推理：

```powershell
E:\miniconda\envs\NLP311\python.exe infer.py --text "我喜欢自然语言处理。" --device cuda
```

交互模式：

```powershell
E:\miniconda\envs\NLP311\python.exe infer.py --device cuda
```
