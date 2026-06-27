"""
把中英文平行文本编码成模型训练需要的 token id 数据。

输入：
    data/train.zh / data/train.en
    data/valid.zh / data/valid.en
    data/test.zh  / data/test.en
    data/vocab/src_zh_jieba_vocab.json
    data/vocab/tgt_en_vocab.json

输出：
    data/cmn_eng_encoded/train.pt
    data/cmn_eng_encoded/valid.pt
    data/cmn_eng_encoded/test.pt
    data/cmn_eng_encoded/meta.json

每条 .pt 样本保存：
    原始中文、原始英文、中文 tokens、英文 tokens、中文 ids、英文 ids。

注意：
    训练、验证、测试集都必须使用训练集建立出来的同一套词表，
    不能给 valid/test 单独建词表，否则 token id 含义会不一致。
"""

from pathlib import Path
import torch

from tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    encode_text,
    load_json,
    save_json,
    tokenize_en,
    tokenize_zh,
)


# =========================
# 1. 编码一个 split
# =========================
def encode_split(
    split_name: str,
    src_path: Path,
    tgt_path: Path,
    src_vocab: dict,
    tgt_vocab: dict,
    max_src_len: int = 128,
    max_tgt_len: int = 128,
):
    """
    编码一个数据 split，例如 train / valid / test。

    参数：
        split_name:
            当前 split 名字，只用于打印日志。
        src_path / tgt_path:
            中文源文件和英文目标文件路径。
        src_vocab / tgt_vocab:
            已经由 build_vocab.py 生成的词表。
        max_src_len / max_tgt_len:
            最大长度，包含 <bos> 和 <eos>。

    返回：
        encoded_data:
            list[dict]，每个 dict 保存一条平行语料的原文、tokens 和 ids。

    这里会跳过空行和过长样本，避免训练时出现无效样本或显存压力过大。
    """
    if not src_path.exists():
        raise FileNotFoundError(f"源语言文件不存在: {src_path}")

    if not tgt_path.exists():
        raise FileNotFoundError(f"目标语言文件不存在: {tgt_path}")

    encoded_data = []

    total_lines = 0
    kept_lines = 0
    skipped_empty = 0
    skipped_too_long = 0

    with src_path.open("r", encoding="utf-8") as f_src, tgt_path.open("r", encoding="utf-8") as f_tgt:
        for src_text, tgt_text in zip(f_src, f_tgt):
            total_lines += 1

            src_text = src_text.strip()
            tgt_text = tgt_text.strip()

            if not src_text or not tgt_text:
                skipped_empty += 1
                continue

            src_ids, src_tokens = encode_text(
                text=src_text,
                vocab=src_vocab,
                tokenizer=tokenize_zh,
                add_bos=True,
                add_eos=True,
            )

            tgt_ids, tgt_tokens = encode_text(
                text=tgt_text,
                vocab=tgt_vocab,
                tokenizer=tokenize_en,
                add_bos=True,
                add_eos=True,
            )

            # 太长的句子先丢掉，避免训练 Transformer 时显存压力过大
            if len(src_ids) > max_src_len or len(tgt_ids) > max_tgt_len:
                skipped_too_long += 1
                continue

            item = {
                "src_text": src_text,
                "tgt_text": tgt_text,
                "src_tokens": src_tokens,
                "tgt_tokens": tgt_tokens,
                "src_ids": src_ids,
                "tgt_ids": tgt_ids,
            }

            encoded_data.append(item)
            kept_lines += 1

    print(f"\n[{split_name}]")
    print("原始行数:", total_lines)
    print("保留样本:", kept_lines)
    print("空行跳过:", skipped_empty)
    print("过长跳过:", skipped_too_long)

    return encoded_data


# =========================
# 2. 主函数
# =========================
def main():
    """
    数据编码主入口。

    运行：
        python encode_dataset.py

    会把 data 目录下的 train / valid / test 平行文本全部编码成 .pt 文件，
    并额外保存 meta.json，供 train.py 和 inference.py 使用。
    """
    data_dir = Path(r"data")
    vocab_dir = Path(r"data/vocab")
    save_dir = Path(r"data\cmn_eng_encoded")
    save_dir.mkdir(parents=True, exist_ok=True)

    src_vocab_path = vocab_dir / "src_zh_jieba_vocab.json"
    tgt_vocab_path = vocab_dir / "tgt_en_vocab.json"

    print("读取中文词表:", src_vocab_path)
    src_vocab = load_json(src_vocab_path)

    print("读取英文词表:", tgt_vocab_path)
    tgt_vocab = load_json(tgt_vocab_path)

    print("中文词表大小:", len(src_vocab))
    print("英文词表大小:", len(tgt_vocab))

    # 句子最大长度，包含 <bos> 和 <eos>
    max_src_len = 128
    max_tgt_len = 128

    split_files = {
        "train": {
            "src": data_dir / "train.zh",
            "tgt": data_dir / "train.en",
        },
        "valid": {
            "src": data_dir / "valid.zh",
            "tgt": data_dir / "valid.en",
        },
        "test": {
            "src": data_dir / "test.zh",
            "tgt": data_dir / "test.en",
        },
    }

    for split_name, paths in split_files.items():
        encoded_data = encode_split(
            split_name=split_name,
            src_path=paths["src"],
            tgt_path=paths["tgt"],
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
        )

        save_path = save_dir / f"{split_name}.pt"
        torch.save(encoded_data, save_path)

        print("已保存:", save_path)

    # 保存一些元信息，后面训练时会用到
    meta = {
        "src_vocab_path": str(src_vocab_path),
        "tgt_vocab_path": str(tgt_vocab_path),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "src_pad_id": src_vocab[PAD_TOKEN],
        "src_unk_id": src_vocab[UNK_TOKEN],
        "src_bos_id": src_vocab[BOS_TOKEN],
        "src_eos_id": src_vocab[EOS_TOKEN],
        "tgt_pad_id": tgt_vocab[PAD_TOKEN],
        "tgt_unk_id": tgt_vocab[UNK_TOKEN],
        "tgt_bos_id": tgt_vocab[BOS_TOKEN],
        "tgt_eos_id": tgt_vocab[EOS_TOKEN],
        "max_src_len": max_src_len,
        "max_tgt_len": max_tgt_len,
    }

    meta_path = save_dir / "meta.json"
    save_json(meta, meta_path)

    print("\n已保存元信息:", meta_path)

    # 打印一个样本检查
    train_data = torch.load(save_dir / "train.pt", weights_only=False)

    print("\n样本检查:")
    sample = train_data[0]

    print("中文原文:", sample["src_text"])
    print("英文原文:", sample["tgt_text"])
    print("中文 tokens:", sample["src_tokens"])
    print("英文 tokens:", sample["tgt_tokens"])
    print("中文 ids:", sample["src_ids"])
    print("英文 ids:", sample["tgt_ids"])

    print("\n编码完成。")


if __name__ == "__main__":
    main()
