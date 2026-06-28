"""
SentencePiece tokenizer 工具。

这个文件只负责 SentencePiece 相关能力：
1. 从 WMT19 zh-en 中抽样导出 tokenizer 训练语料；
2. 训练共享 SentencePiece Unigram tokenizer；
3. 加载并校验 tokenizer 的特殊 token id；
4. 编码/解码文本，供 data.py 和 infer.py 复用。
"""

import argparse
from pathlib import Path

import sentencepiece as spm
from datasets import load_dataset

# 这四个特殊 token 的 id 必须和模型训练、mask、loss、推理停止条件保持一致。
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

SPECIAL_IDS = {PAD_ID, UNK_ID, BOS_ID, EOS_ID}

DEFAULT_SPM_MODEL = "data/spm/wmt19_zh_en_unigram_32k.model"
DEFAULT_DATASET = "wmt/wmt19"
DEFAULT_CONFIG = "zh-en"


def clean_text(text: str):
    """
    清理单句文本。

    WMT19 的样本一般是一句对应一句，但实际数据里可能含有换行。
    训练 tokenizer 和模型时，把内部换行替换成空格可以避免“一条样本被拆成多行”。
    """
    return text.replace("\n", " ").strip()


def validate_spm_ids(processor: spm.SentencePieceProcessor):
    """
    校验 SentencePiece 模型里的特殊 token id。

    这里必须严格检查，因为：
    - <pad> id 错了，attention mask 和 loss ignore_index 会错；
    - <bos>/<eos> id 错了，decoder 训练和推理停止条件会错；
    - <unk> id 错了，未知 token 处理会和训练期不一致。
    """
    expected = {
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
    }
    actual = {
        "pad_id": processor.pad_id(),
        "unk_id": processor.unk_id(),
        "bos_id": processor.bos_id(),
        "eos_id": processor.eos_id(),
    }

    for key, expected_id in expected.items():
        if actual[key] != expected_id:
            raise ValueError(
                f"SentencePiece {key}={actual[key]}, "
                f"但项目要求 {key}={expected_id}。"
            )


def load_spm(model_path: str | Path = DEFAULT_SPM_MODEL):
    """
    加载并校验 SentencePiece model。

    所有训练、验证和推理入口都应该通过这个函数加载 tokenizer，
    避免某个入口绕过特殊 token id 校验。
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece model 不存在: {model_path}")

    processor = spm.SentencePieceProcessor()
    processor.load(str(model_path))
    validate_spm_ids(processor)
    return processor


def encode_text(
    text: str,
    processor: spm.SentencePieceProcessor,
    add_bos: bool = True,
    add_eos: bool = True,
    max_len: int | None = None,
):
    """
    把一段文本编码成 Transformer 需要的 token id。

    max_len 包含 <bos>/<eos>。返回的 pieces 不包含特殊 token，只用于调试显示。
    """
    piece_ids = processor.encode(clean_text(text), out_type=int)

    if max_len is not None:
        reserved = int(add_bos) + int(add_eos)
        piece_ids = piece_ids[: max(max_len - reserved, 0)]

    ids = []
    if add_bos:
        ids.append(processor.bos_id())

    ids.extend(piece_ids)

    if add_eos:
        ids.append(processor.eos_id())

    pieces = [processor.id_to_piece(piece_id) for piece_id in piece_ids]
    return ids, pieces


def decode_ids(
    ids: list[int],
    processor: spm.SentencePieceProcessor,
    skip_special: bool = True,
):
    """把模型生成的 token id 还原成文本。"""
    if skip_special:
        ids = [piece_id for piece_id in ids if piece_id not in SPECIAL_IDS]

    return processor.decode(ids)


def train_spm(
    out_dir: str | Path = "data/spm",
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_CONFIG,
    split: str = "train",
    max_pairs: int = 1_000_000,
    vocab_size: int = 32000,
):
    """
    从 WMT19 抽样训练共享 SentencePiece Unigram tokenizer。

    当前训练策略：
    - 从 WMT19 zh-en 的 train split 中读取平行句对；
    - 每个句对写两行：中文一行、英文一行；
    - 中文和英文共用一个 SentencePiece 词表；
    - 默认最多使用 100 万个句对，也就是最多 200 万行文本。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = out_dir / "wmt19_zh_en_spm_corpus.txt"
    model_prefix = out_dir / f"wmt19_zh_en_unigram_{vocab_size // 1000}k"
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    print("开始导出 SentencePiece 训练语料...")
    print("最多使用句对数:", max_pairs)

    written_lines = 0
    used_pairs = 0

    with corpus_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            translation = item["translation"]
            zh = clean_text(translation["zh"])
            en = clean_text(translation["en"])

            if not zh or not en:
                continue

            if len(zh) > 300 or len(en.split()) > 120:
                continue

            f.write(zh + "\n")
            f.write(en + "\n")

            written_lines += 2
            used_pairs += 1

            if used_pairs >= max_pairs:
                break

            if used_pairs % 100000 == 0:
                print("已导出句对:", used_pairs)

    print("语料文件:", corpus_path)
    print("使用句对数:", used_pairs)
    print("写入行数:", written_lines)
    print("开始训练 SentencePiece Unigram...")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=0.9995,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece=PAD_TOKEN,
        unk_piece=UNK_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_TOKEN,
        input_sentence_size=max_pairs * 2,
        shuffle_input_sentence=True,
        hard_vocab_limit=False,
    )

    print("训练完成:")
    print(str(model_prefix) + ".model")
    print(str(model_prefix) + ".vocab")


def check_tokenizer(spm_model: str | Path = DEFAULT_SPM_MODEL):
    """打印 tokenizer 的基本编码/解码结果和特殊 token id。"""
    processor = load_spm(spm_model)
    sample = "我喜欢自然语言处理。"
    ids, pieces = encode_text(sample, processor)

    print(pieces)
    print(ids)
    print(decode_ids(ids, processor))
    print("pad:", processor.pad_id())
    print("unk:", processor.unk_id())
    print("bos:", processor.bos_id())
    print("eos:", processor.eos_id())


def parse_args():
    """解析 spm.py 的子命令。"""
    parser = argparse.ArgumentParser(description="SentencePiece utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train SentencePiece")
    train_parser.add_argument("--out_dir", type=str, default="data/spm")
    train_parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    train_parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    train_parser.add_argument("--split", type=str, default="train")
    train_parser.add_argument("--max_pairs", type=int, default=1_000_000)
    train_parser.add_argument("--vocab_size", type=int, default=32000)

    check_parser = subparsers.add_parser("check", help="check tokenizer")
    check_parser.add_argument("--spm_model", type=str, default=DEFAULT_SPM_MODEL)

    return parser.parse_args()


def main():
    """命令行入口：训练或检查 SentencePiece tokenizer。"""
    args = parse_args()

    if args.command == "train":
        train_spm(
            out_dir=args.out_dir,
            dataset_name=args.dataset,
            dataset_config=args.config,
            split=args.split,
            max_pairs=args.max_pairs,
            vocab_size=args.vocab_size,
        )
        return

    if args.command == "check":
        check_tokenizer(args.spm_model)
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
