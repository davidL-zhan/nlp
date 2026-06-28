"""
SentencePiece and WMT19 data utilities.

This file owns all reusable data code:
- train the shared SentencePiece tokenizer;
- encode/decode text with that tokenizer;
- build WMT19 zh-en DataLoaders for Transformer training.
"""

from pathlib import Path
import argparse
import json

import sentencepiece as spm
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

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
    return text.replace("\n", " ").strip()


def load_json(path: str | Path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def validate_spm_ids(processor: spm.SentencePieceProcessor):
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
    if skip_special:
        ids = [piece_id for piece_id in ids if piece_id not in SPECIAL_IDS]

    return processor.decode(ids)


def pad_sequences(sequences, pad_id: int):
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


def train_spm(
    out_dir: str | Path = "data/spm",
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_CONFIG,
    split: str = "train",
    max_pairs: int = 1_000_000,
    vocab_size: int = 32000,
):
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


def build_meta(
    spm_model: str | Path,
    max_src_len: int,
    max_tgt_len: int,
    dataset_name: str,
    dataset_config: str,
):
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
    def __init__(
        self,
        hf_dataset,
        spm_model: str | Path = DEFAULT_SPM_MODEL,
        max_src_len: int = 128,
        max_tgt_len: int = 128,
    ):
        self.hf_dataset = hf_dataset
        self.spm_model = str(spm_model)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self._sp = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sp"] = None
        return state

    def _processor(self):
        if self._sp is None:
            self._sp = load_spm(self.spm_model)
        return self._sp

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
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
    def __init__(self, src_pad_id: int, tgt_pad_id: int):
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def __call__(self, batch):
        src_ids = pad_sequences([item["src_ids"] for item in batch], self.src_pad_id)
        tgt_ids = pad_sequences([item["tgt_ids"] for item in batch], self.tgt_pad_id)

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
    meta = build_meta(
        spm_model=spm_model,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
    )

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

    train_hf = select_samples(train_hf, max_train_samples)
    valid_hf = select_samples(valid_hf, max_valid_samples)
    test_hf = select_samples(valid_hf, max_test_samples)

    train_dataset = WMT19Dataset(train_hf, spm_model, max_src_len, max_tgt_len)
    valid_dataset = WMT19Dataset(valid_hf, spm_model, max_src_len, max_tgt_len)
    test_dataset = WMT19Dataset(test_hf, spm_model, max_src_len, max_tgt_len)

    collate_fn = TranslationCollator(meta["src_pad_id"], meta["tgt_pad_id"])

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


def check_tokenizer(spm_model: str | Path = DEFAULT_SPM_MODEL):
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


def check_loader(spm_model: str | Path = DEFAULT_SPM_MODEL):
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
    parser = argparse.ArgumentParser(description="Data utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spm_parser = subparsers.add_parser("spm", help="train SentencePiece")
    spm_parser.add_argument("--out_dir", type=str, default="data/spm")
    spm_parser.add_argument("--max_pairs", type=int, default=1_000_000)
    spm_parser.add_argument("--vocab_size", type=int, default=32000)

    check_parser = subparsers.add_parser("check", help="check tokenizer and loader")
    check_parser.add_argument("--spm_model", type=str, default=DEFAULT_SPM_MODEL)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "spm":
        train_spm(
            out_dir=args.out_dir,
            max_pairs=args.max_pairs,
            vocab_size=args.vocab_size,
        )
        return

    if args.command == "check":
        check_tokenizer(args.spm_model)
        print()
        check_loader(args.spm_model)
        return

    raise ValueError(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
