"""
SentencePiece tokenizer demo.

This script loads the trained WMT19 zh-en SentencePiece model and prints:
1. tokenizer metadata;
2. subword pieces;
3. token ids with optional <bos>/<eos>;
4. decoded text after ids -> text conversion.
"""

from __future__ import annotations

import argparse
import sys

from spm import DEFAULT_SPM_MODEL, decode_ids, encode_text, load_spm


DEFAULT_SAMPLES = [
    "我喜欢自然语言处理，也在训练Transformer翻译模型。",
    "机器学习模型需要高质量的数据。",
    "This tokenizer can split unseen words like microarchitecture and quantization.",
    "中英混合: I love Beijing and neural machine translation.",
    "2026年6月28日，模型 loss 降到了 3.14。",
]


def configure_stdout():
    """Keep Chinese output readable in Windows terminals when possible."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_model_info(processor, model_path: str):
    """Print tokenizer metadata and special token ids."""
    print("SentencePiece model:", model_path)
    print("vocab_size:", processor.get_piece_size())
    print(
        "special_ids:",
        {
            "pad": processor.pad_id(),
            "unk": processor.unk_id(),
            "bos": processor.bos_id(),
            "eos": processor.eos_id(),
        },
    )
    print()


def run_demo(samples: list[str], spm_model: str, add_special: bool, max_len: int | None):
    """Encode and decode each sample with the trained SentencePiece model."""
    processor = load_spm(spm_model)
    print_model_info(processor, spm_model)

    for index, text in enumerate(samples, start=1):
        ids, pieces = encode_text(
            text,
            processor,
            add_bos=add_special,
            add_eos=add_special,
            max_len=max_len,
        )
        piece_ids = processor.Encode(text, out_type=int)
        if max_len is not None:
            reserved = 2 if add_special else 0
            piece_ids = piece_ids[: max(max_len - reserved, 0)]

        print(f"[{index}] text:")
        print(text)
        print("pieces:")
        print(" | ".join(pieces))
        print("piece_ids_without_special:")
        print(piece_ids)
        print("ids_with_special:" if add_special else "ids:")
        print(ids)
        print("decoded:")
        print(decode_ids(ids, processor))
        print("piece_count_without_special:", len(pieces))
        print("-" * 80)


def parse_args():
    """Parse command line options for the tokenizer demo."""
    parser = argparse.ArgumentParser(
        description="Demo the trained WMT19 zh-en SentencePiece tokenizer."
    )
    parser.add_argument(
        "--spm_model",
        type=str,
        default=DEFAULT_SPM_MODEL,
        help="Path to the trained SentencePiece .model file.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=None,
        help="Custom text to tokenize. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Optional max token length, including <bos>/<eos> when enabled.",
    )
    parser.add_argument(
        "--no_special",
        action="store_true",
        help="Do not add <bos>/<eos> around encoded token ids.",
    )
    return parser.parse_args()


def main():
    """Run the tokenizer demo."""
    configure_stdout()
    args = parse_args()

    samples = args.text if args.text else DEFAULT_SAMPLES
    run_demo(
        samples=samples,
        spm_model=args.spm_model,
        add_special=not args.no_special,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
