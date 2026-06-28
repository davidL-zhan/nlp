"""
Run Chinese-to-English inference with a trained Transformer checkpoint.

The checkpoint must come from train.py's WMT19 + SentencePiece pipeline.
"""

from pathlib import Path
import argparse

import torch

from data import decode_ids, encode_text, load_spm
from model import Transformer, make_src_mask, make_tgt_mask


def resolve_spm_model_path(checkpoint: dict, override_path: str | None = None):
    if override_path:
        return override_path

    meta = checkpoint["meta"]
    spm_model_path = meta.get("spm_model_path")

    if not spm_model_path:
        raise ValueError(
            "checkpoint meta 中没有 spm_model_path。"
            "请使用当前 SentencePiece 训练流程重新训练 checkpoint。"
        )

    return spm_model_path


def encode_source_text(text: str, processor, meta: dict):
    max_src_len = meta.get("max_src_len", 128)
    return encode_text(
        text=text,
        processor=processor,
        add_bos=True,
        add_eos=True,
        max_len=max_src_len,
    )


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    meta = checkpoint["meta"]
    train_args = checkpoint.get("args", {})

    model = Transformer(
        src_vocab_size=meta["src_vocab_size"],
        tgt_vocab_size=meta["tgt_vocab_size"],
        embeding_size=train_args.get("embeding_size", 256),
        num_heads=train_args.get("num_heads", 4),
        num_encoder_layers=train_args.get("num_encoder_layers", 3),
        num_decoder_layers=train_args.get("num_decoder_layers", 3),
        Mlp_hidden_size=train_args.get("mlp_hidden_size", 1024),
        dropout=train_args.get("dropout", 0.1),
        max_len=train_args.get("max_len", 5000),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def greedy_decode(
    model,
    text: str,
    processor,
    meta: dict,
    device: torch.device,
    max_decode_len: int,
):
    src_ids, src_pieces = encode_source_text(text, processor, meta)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = make_src_mask(src_tensor, meta["src_pad_id"])

    generated_ids = [meta["tgt_bos_id"]]

    for _ in range(max_decode_len):
        tgt_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        tgt_mask = make_tgt_mask(tgt_tensor, meta["tgt_pad_id"])

        logits = model(
            src=src_tensor,
            tgt=tgt_tensor,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )

        next_token_id = int(torch.argmax(logits[0, -1]).item())
        generated_ids.append(next_token_id)

        if next_token_id == meta["tgt_eos_id"]:
            break

    output_ids = [
        token_id
        for token_id in generated_ids
        if token_id
        not in {
            meta["tgt_pad_id"],
            meta["tgt_bos_id"],
            meta["tgt_eos_id"],
        }
    ]
    output_pieces = [processor.id_to_piece(token_id) for token_id in output_ids]
    translation = decode_ids(generated_ids, processor, skip_special=True)

    return {
        "src_pieces": src_pieces,
        "generated_ids": generated_ids,
        "output_pieces": output_pieces,
        "translation": translation,
    }


def load_checkpoint(path: str | Path, device: torch.device):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在: {path}\n"
            f"请先运行 train.py，或用 --checkpoint 指向已有 checkpoint。"
        )

    return torch.load(path, map_location=device, weights_only=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a WMT19 SentencePiece Transformer."
    )

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--spm_model", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--max_decode_len", type=int, default=60)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show_tokens", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("使用设备:", device)
    print("加载 checkpoint:", args.checkpoint)

    checkpoint = load_checkpoint(args.checkpoint, device)
    meta = checkpoint["meta"]

    spm_model_path = resolve_spm_model_path(checkpoint, args.spm_model)
    processor = load_spm(spm_model_path)
    model = build_model_from_checkpoint(checkpoint, device)

    print("SentencePiece:", spm_model_path)
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint valid loss:", checkpoint.get("valid_loss"))

    def translate_and_print(text: str):
        result = greedy_decode(
            model=model,
            text=text,
            processor=processor,
            meta=meta,
            device=device,
            max_decode_len=args.max_decode_len,
        )

        print("中文:", text)

        if args.show_tokens:
            print("中文 pieces:", result["src_pieces"])
            print("英文 pieces:", result["output_pieces"])
            print("生成 ids:", result["generated_ids"])

        print("英文:", result["translation"])

    if args.text is not None:
        translate_and_print(args.text)
        return

    print("\n进入交互模式。输入中文句子后回车；输入 q 或空行退出。")

    while True:
        text = input("\n中文> ").strip()

        if not text or text.lower() in {"q", "quit", "exit"}:
            break

        translate_and_print(text)


if __name__ == "__main__":
    main()
