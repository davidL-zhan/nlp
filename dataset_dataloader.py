"""
把 encode_dataset.py 生成的 .pt 文件封装成 PyTorch Dataset / DataLoader。

这个文件处在数据流水线的最后一步：
    原始文本 -> build_vocab.py 建词表 -> encode_dataset.py 转 token id -> 本文件组 batch

主要职责：
1. 读取 train.pt / valid.pt / test.pt。
2. 把每条样本中的 src_ids / tgt_ids 转成 torch.LongTensor。
3. 在 collate_fn 中对同一个 batch 内的不同长度句子做动态 padding。
4. 自动生成 decoder 训练需要的 tgt_input 和 tgt_output。
5. 生成简单的 key padding mask，方便调试；真正训练时 train.py 会重新调用
   Transformer.py 里的 make_src_mask / make_tgt_mask 构造注意力 mask。
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import load_json


class TranslationEncodedDataset(Dataset):
    """
    已编码翻译数据集。

    .pt 文件里的每条样本来自 encode_dataset.py，结构大致是：
        {
            "src_text": 原始中文句子,
            "tgt_text": 原始英文句子,
            "src_ids": 中文 token id 列表,
            "tgt_ids": 英文 token id 列表
        }

    Dataset 的职责只是一条一条取样本，不做 padding。
    padding 必须放到 collate_fn 里做，因为同一个 batch 中的最大长度每次可能不同。
    """

    def __init__(self, pt_path: str | Path):
        # 保存数据文件路径，方便报错时定位具体缺失的文件。
        self.pt_path = Path(pt_path)

        if not self.pt_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.pt_path}")

        # torch.load 得到的是 encode_dataset.py 保存的 list[dict]。
        # weights_only=False 是为了兼容当前 .pt 文件里保存的普通 Python 对象。
        self.data = torch.load(self.pt_path, weights_only=False)

    def __len__(self):
        # DataLoader 会调用 __len__ 来知道数据集总样本数。
        return len(self.data)

    def __getitem__(self, index):
        # DataLoader 根据 index 取一条样本。
        item = self.data[index]

        # token id 必须是 long 类型，因为 nn.Embedding 只能接收整数索引。
        src_ids = torch.tensor(item["src_ids"], dtype=torch.long)
        tgt_ids = torch.tensor(item["tgt_ids"], dtype=torch.long)

        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_text": item["src_text"],
            "tgt_text": item["tgt_text"],
        }


def pad_sequences(sequences, pad_id: int):
    """
    把不同长度的 id 序列 padding 成同一长度。

    输入：
        [
            tensor([2, 10, 20, 3]),
            tensor([2, 15, 16, 17, 18, 3])
        ]

    输出：
        tensor([
            [2, 10, 20,  3, 0, 0],
            [2, 15, 16, 17, 18, 3]
        ])
    """
    batch_size = len(sequences)
    max_len = max(seq.size(0) for seq in sequences)

    batch = torch.full(
        size=(batch_size, max_len),
        fill_value=pad_id,
        dtype=torch.long,
    )

    for i, seq in enumerate(sequences):
        length = seq.size(0)
        batch[i, :length] = seq

    return batch


def build_collate_fn(src_pad_id: int, tgt_pad_id: int):
    """
    根据源语言和目标语言的 pad id 构造 collate_fn。

    collate_fn 是 DataLoader 在“多条样本组成一个 batch”时调用的函数。
    这里使用闭包保存 src_pad_id / tgt_pad_id，这样外层不用每次手动传 pad id。

    Windows 注意：
        如果 DataLoader 使用 num_workers > 0，嵌套函数可能无法 pickle。
        当前 train.py 默认 num_workers=0，因此不会触发这个问题。
        如果以后想使用多进程 DataLoader，需要把 collate_fn 改成顶层 class。
    """

    def collate_fn(batch):
        # batch 是一个 list，里面的每个元素都是 TranslationEncodedDataset.__getitem__
        # 返回的 dict。
        src_ids_list = [item["src_ids"] for item in batch]
        tgt_ids_list = [item["tgt_ids"] for item in batch]

        # 原始文本不参与训练计算，但保留下来方便调试和打印样本。
        src_texts = [item["src_text"] for item in batch]
        tgt_texts = [item["tgt_text"] for item in batch]

        # 动态 padding：只 pad 到当前 batch 的最大长度，而不是全数据集最大长度。
        # 这样比固定 pad 到 128 更省显存和计算量。
        src_ids = pad_sequences(src_ids_list, src_pad_id)
        tgt_ids = pad_sequences(tgt_ids_list, tgt_pad_id)

        # Transformer decoder 训练时要错开一位
        # tgt:       <bos> I like you . <eos>
        # tgt_input: <bos> I like you .
        # tgt_output:      I like you . <eos>
        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]

        # padding mask，后面 Transformer 里会用
        # True 表示该位置是 pad，需要被 mask 掉
        src_key_padding_mask = src_ids.eq(src_pad_id)
        tgt_key_padding_mask = tgt_input.eq(tgt_pad_id)

        return {
            "src_ids": src_ids,  # [B, src_len]
            "tgt_ids": tgt_ids,  # [B, tgt_len]
            "tgt_input": tgt_input,  # [B, tgt_len - 1]
            "tgt_output": tgt_output,  # [B, tgt_len - 1]
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask,
            "src_texts": src_texts,
            "tgt_texts": tgt_texts,
        }

    return collate_fn


def load_meta(meta_path: str | Path):
    """
    读取 encode_dataset.py 保存的 meta.json。

    meta.json 保存了训练和推理必须共享的关键信息：
        词表大小、pad/unk/bos/eos 对应的 id、最大句长等。
    """
    meta_path = Path(meta_path)

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_path}")

    return load_json(meta_path)


def build_dataloaders(
    encoded_dir: str | Path = r"data/cmn_eng_encoded",
    batch_size: int = 64,
    num_workers: int = 0,
):
    """
    构建 train / valid / test 三个 DataLoader。

    参数：
        encoded_dir:
            encode_dataset.py 输出目录，里面应该包含 train.pt、valid.pt、test.pt、meta.json。
        batch_size:
            每个 batch 的样本数。
        num_workers:
            DataLoader 子进程数量。Windows 下建议保持 0，避免 collate_fn pickle 问题。

    返回：
        train_loader, valid_loader, test_loader, meta
    """
    encoded_dir = Path(encoded_dir)

    meta = load_meta(encoded_dir / "meta.json")

    src_pad_id = meta["src_pad_id"]
    tgt_pad_id = meta["tgt_pad_id"]

    train_dataset = TranslationEncodedDataset(encoded_dir / "train.pt")
    valid_dataset = TranslationEncodedDataset(encoded_dir / "valid.pt")
    test_dataset = TranslationEncodedDataset(encoded_dir / "test.pt")

    collate_fn = build_collate_fn(
        src_pad_id=src_pad_id,
        tgt_pad_id=tgt_pad_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
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


def main():
    """
    DataLoader 自检入口。

    直接运行：
        python dataset_dataloader.py

    可以打印一个 batch 的 tensor 形状、mask 形状和原始文本，
    用来确认数据编码和动态 padding 是否正常。
    """
    train_loader, valid_loader, test_loader, meta = build_dataloaders(
        encoded_dir=r"data/cmn_eng_encoded",
        batch_size=4,
        num_workers=0,
    )

    print("中文词表大小:", meta["src_vocab_size"])
    print("英文词表大小:", meta["tgt_vocab_size"])
    print("src_pad_id:", meta["src_pad_id"])
    print("tgt_pad_id:", meta["tgt_pad_id"])

    batch = next(iter(train_loader))

    print()
    print("src_ids shape:", batch["src_ids"].shape)
    print("tgt_ids shape:", batch["tgt_ids"].shape)
    print("tgt_input shape:", batch["tgt_input"].shape)
    print("tgt_output shape:", batch["tgt_output"].shape)

    print()
    print("src_ids:")
    print(batch["src_ids"])

    print()
    print("tgt_input:")
    print(batch["tgt_input"])

    print()
    print("tgt_output:")
    print(batch["tgt_output"])

    print()
    print("src_key_padding_mask shape:", batch["src_key_padding_mask"].shape)
    print("tgt_key_padding_mask shape:", batch["tgt_key_padding_mask"].shape)

    print()
    print("中文原文:")
    for text in batch["src_texts"]:
        print(text)

    print()
    print("英文原文:")
    for text in batch["tgt_texts"]:
        print(text)


if __name__ == "__main__":
    main()
