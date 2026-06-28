"""
WMT19 Dataset 和 DataLoader 构建工具。

这个文件只负责数据集读取和 batch 构造：
1. 从 HuggingFace Datasets 读取 WMT19 zh-en；
2. 用 spm.py 中的 SentencePiece 工具把文本编码成 token id；
3. 在 collator 中动态 padding；
4. 返回 train/valid/test DataLoader 和 checkpoint 需要保存的 meta。

SentencePiece 的训练、加载、编码/解码实现都在 spm.py 中。
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from spm import (
    BOS_TOKEN,
    DEFAULT_CONFIG,
    DEFAULT_DATASET,
    DEFAULT_SPM_MODEL,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    clean_text,
    encode_text,
    load_spm,
)


def pad_sequences(sequences, pad_id: int):
    """
    把同一个 batch 内不同长度的序列 pad 到相同长度。

    输入是若干一维 LongTensor，输出是二维 LongTensor：
        [batch_size, 当前 batch 最大序列长度]

    这里做的是动态 padding，只 pad 到当前 batch 的最大长度，
    比固定 pad 到全局 max_len 更省显存和计算。
    """
    batch_size = len(sequences)
    max_len = max(seq.size(0) for seq in sequences)

    batch = torch.full(
        size=(batch_size, max_len),
        fill_value=pad_id,
        dtype=torch.long,
    )

    for i, seq in enumerate(sequences):
        batch[i, : seq.size(0)] = seq

    return batch


def build_meta(
    spm_model: str | Path,
    max_src_len: int,
    max_tgt_len: int,
    dataset_name: str,
    dataset_config: str,
):
    """
    根据 tokenizer 和数据设置构造训练/推理共用的 meta 字典。

    meta 会被保存到 checkpoint 中。推理时不需要重新猜 tokenizer 路径、
    词表大小或特殊 token id，而是直接读取 checkpoint["meta"]。
    """
    processor = load_spm(spm_model)
    vocab_size = processor.get_piece_size()

    return {
        "tokenizer_type": "sentencepiece",
        "spm_model_path": str(Path(spm_model)),
        "hf_dataset_name": dataset_name,
        "hf_dataset_config": dataset_config,
        "src_vocab_size": vocab_size,
        "tgt_vocab_size": vocab_size,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "src_pad_id": processor.pad_id(),
        "src_unk_id": processor.unk_id(),
        "src_bos_id": processor.bos_id(),
        "src_eos_id": processor.eos_id(),
        "tgt_pad_id": processor.pad_id(),
        "tgt_unk_id": processor.unk_id(),
        "tgt_bos_id": processor.bos_id(),
        "tgt_eos_id": processor.eos_id(),
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
    }


class WMT19Dataset(Dataset):
    """
    HuggingFace WMT19 zh-en 的 PyTorch Dataset 包装。

    Dataset 的职责：
    1. 根据 index 取出一条 HuggingFace 样本；
    2. 读取 translation["zh"] 和 translation["en"]；
    3. 用 SentencePiece 编码成 src_ids/tgt_ids；
    4. 返回单条样本，不做 padding。

    padding 放到 TranslationCollator 中做，因为同一个 batch 的最大长度是动态的。
    """

    def __init__(
        self,
        hf_dataset,
        spm_model: str | Path = DEFAULT_SPM_MODEL,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        """
        初始化数据集包装器。

        self._sp 故意延迟加载：
        - num_workers=0 时，第一次 __getitem__ 再加载；
        - num_workers>0 时，每个 worker 进程单独加载，避免 SentencePiece 对象 pickle 问题。
        """
        self.hf_dataset = hf_dataset
        self.spm_model = str(spm_model)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self._sp = None

    def __getstate__(self):
        """
        DataLoader 多进程复制 Dataset 时会调用 pickle。

        SentencePieceProcessor 不适合直接跨进程 pickle，
        所以复制状态时把 _sp 清空，让 worker 进程自己重新加载。
        """
        state = self.__dict__.copy()
        state["_sp"] = None
        return state

    def _processor(self):
        """懒加载 SentencePieceProcessor。"""
        if self._sp is None:
            self._sp = load_spm(self.spm_model)
        return self._sp

    def __len__(self):
        """返回当前 split 的样本数。"""
        return len(self.hf_dataset)

    def __getitem__(self, index):
        """
        返回一条已经编码但尚未 padding 的训练样本。

        src_ids/tgt_ids 都是一维 LongTensor，长度可能不同。
        """
        item = self.hf_dataset[index]
        translation = item["translation"]
        processor = self._processor()

        src_text = clean_text(translation.get("zh", ""))
        tgt_text = clean_text(translation.get("en", ""))
        src_ids, src_pieces = encode_text(src_text, processor, max_len=self.max_src_len)
        tgt_ids, tgt_pieces = encode_text(tgt_text, processor, max_len=self.max_tgt_len)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src_pieces": src_pieces,
            "tgt_pieces": tgt_pieces,
        }


class TranslationCollator:
    """
    DataLoader 的 batch 组装器。

    它负责把多条样本合成一个 batch：
    - 对 src_ids 和 tgt_ids 做动态 padding；
    - 构造 decoder 训练用的 tgt_input/tgt_output；
    - 额外保留原文和 SentencePiece pieces 方便调试。
    """

    def __init__(self, src_pad_id: int, tgt_pad_id: int):
        """保存源端和目标端的 padding id。共享词表时两者相同。"""
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def __call__(self, batch):
        """把 list[dict] 样本转换成模型训练需要的 batch dict。"""
        src_ids = pad_sequences([item["src_ids"] for item in batch], self.src_pad_id)
        tgt_ids = pad_sequences([item["tgt_ids"] for item in batch], self.tgt_pad_id)

        # decoder 训练采用 teacher forcing：
        # tgt_ids:     <bos> I like NLP . <eos>
        # tgt_input:   <bos> I like NLP .
        # tgt_output:        I like NLP . <eos>
        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
            "src_key_padding_mask": src_ids.eq(self.src_pad_id),
            "tgt_key_padding_mask": tgt_input.eq(self.tgt_pad_id),
            "src_texts": [item["src_text"] for item in batch],
            "tgt_texts": [item["tgt_text"] for item in batch],
            "src_pieces": [item["src_pieces"] for item in batch],
            "tgt_pieces": [item["tgt_pieces"] for item in batch],
        }


def select_samples(hf_dataset, max_samples: int):
    """
    从 HuggingFace Dataset 前部截取固定数量样本。

    max_samples <= 0 表示不截取，直接使用完整 split。
    """
    if max_samples is None or max_samples <= 0:
        return hf_dataset

    return hf_dataset.select(range(min(max_samples, len(hf_dataset))))


def build_loaders(
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_CONFIG,
    train_split: str = "train",
    valid_split: str = "validation",
    spm_model: str | Path = DEFAULT_SPM_MODEL,
    batch_size: int = 64,
    num_workers: int = 0,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
    max_train_samples: int = 100000,
    max_valid_samples: int = 3981,
    max_test_samples: int = 3981,
    seed: int = 42,
    cache_dir: str | None = None,
):
    """
    构建 train/valid/test 三个 DataLoader 和 meta 信息。

    训练默认只取前 100000 条 WMT19 train 样本，便于快速验证。
    如果需要全量训练，命令行传 --hf_train_samples 0。
    """
    meta = build_meta(
        spm_model=spm_model,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )

    # 这里会优先使用本地 HuggingFace cache；如果缓存不存在则联网下载。
    train_hf = load_dataset(
        dataset_name,
        dataset_config,
        split=train_split,
        cache_dir=cache_dir,
    )
    valid_hf = load_dataset(
        dataset_name,
        dataset_config,
        split=valid_split,
        cache_dir=cache_dir,
    )

    # 为了让学习和调试更快，默认只取子集；正式训练时可以设为 0 使用全量。
    train_hf = select_samples(train_hf, max_train_samples)
    valid_hf = select_samples(valid_hf, max_valid_samples)
    test_hf = select_samples(valid_hf, max_test_samples)

    train_dataset = WMT19Dataset(train_hf, spm_model, max_src_len, max_tgt_len)
    valid_dataset = WMT19Dataset(valid_hf, spm_model, max_src_len, max_tgt_len)
    test_dataset = WMT19Dataset(test_hf, spm_model, max_src_len, max_tgt_len)

    collate_fn = TranslationCollator(meta["src_pad_id"], meta["tgt_pad_id"])

    # 给 DataLoader shuffle 一个固定随机种子，保证调试时样本顺序可复现。
    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffle_train = max_train_samples is not None and max_train_samples > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator if shuffle_train else None,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader, meta


def check_loader(spm_model: str | Path = DEFAULT_SPM_MODEL):
    """打印一个小 batch，检查 WMT19 -> SentencePiece -> DataLoader 是否正常。"""
    train_loader, valid_loader, _, meta = build_loaders(
        spm_model=spm_model,
        batch_size=4,
        max_train_samples=16,
        max_valid_samples=8,
        max_test_samples=8,
    )

    print("tokenizer:", meta["tokenizer_type"])
    print("SentencePiece:", meta["spm_model_path"])
    print("词表大小:", meta["src_vocab_size"])
    print("训练样本数:", len(train_loader.dataset))
    print("验证样本数:", len(valid_loader.dataset))

    batch = next(iter(train_loader))
    print("src_ids shape:", batch["src_ids"].shape)
    print("tgt_input shape:", batch["tgt_input"].shape)
    print("tgt_output shape:", batch["tgt_output"].shape)
    print("中文:", batch["src_texts"][0])
    print("英文:", batch["tgt_texts"][0])
    print("中文 pieces:", batch["src_pieces"][0][:30])
    print("英文 pieces:", batch["tgt_pieces"][0][:30])


def parse_args():
    """解析 data.py 的子命令。"""
    parser = argparse.ArgumentParser(description="Dataset/DataLoader utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="check DataLoader")
    check_parser.add_argument("--spm_model", type=str, default=DEFAULT_SPM_MODEL)

    return parser.parse_args()


def main():
    """命令行入口：检查数据管线。"""
    args = parse_args()

    if args.command == "check":
        check_loader(args.spm_model)
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
