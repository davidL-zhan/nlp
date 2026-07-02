"""
中译英推理入口。

这个文件负责加载 train.py 保存的 checkpoint，并用同一个 SentencePiece
tokenizer 把输入中文编码成 token id，再通过 Transformer greedy decoding
逐步生成英文 token。

checkpoint 必须来自当前 WMT19 + SentencePiece 训练流程，因为推理需要读取：
- checkpoint["meta"]["spm_model_path"]
- 词表大小
- <pad>/<bos>/<eos> 等特殊 token id
- 训练时的模型结构参数
"""

from pathlib import Path
import argparse

import torch

from model import Transformer, make_src_mask, make_tgt_mask
from spm import decode_ids, encode_text, load_spm


def resolve_spm_model_path(checkpoint: dict, override_path: str | None = None):
    """
    决定推理时使用哪个 SentencePiece model。

    优先级：
    1. 如果命令行传入 --spm_model，则使用用户指定路径；
    2. 否则使用 checkpoint meta 中保存的 spm_model_path。

    正常情况下建议使用 checkpoint 自带路径，保证训练和推理 tokenizer 一致。
    """
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
    """
    把输入中文编码成 encoder 可以接收的 src_ids。

    max_src_len 从 checkpoint meta 读取，确保推理阶段和训练阶段的最大长度一致。
    """
    max_src_len = meta.get("max_src_len", 128)
    return encode_text(
        text=text,
        processor=processor,
        add_bos=True,
        add_eos=True,
        max_len=max_src_len,
    )


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """
    根据 checkpoint 恢复 Transformer 结构和参数。

    这里不能随意使用默认模型配置，必须读取 checkpoint["args"]：
    - embeding_size
    - num_heads
    - encoder/decoder 层数
    - FFN hidden size

    否则模型结构不同，load_state_dict 会因为参数形状不匹配而失败。
    """
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
    """
    使用 greedy decoding 逐步生成英文翻译。

    greedy decoding 的策略很简单：
    每一步只取当前 logits 最大的 token 作为下一个 token。

    解码流程：
    1. 中文输入编码成 src_ids；
    2. decoder 从 <bos> 开始；
    3. 每一步重新把已生成序列送入 decoder；
    4. 取最后一个位置的预测 token；
    5. 如果生成 <eos> 或达到 max_decode_len，就停止。

    这种实现没有缓存 encoder/decoder 的中间状态，速度不是最优，
    但逻辑直观，适合当前学习项目。
    """
    src_ids, src_pieces = encode_source_text(text, processor, meta)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    # 单句推理通常没有 padding，但仍然构造 src_mask，
    # 保持和训练阶段的模型调用方式一致。
    src_mask = make_src_mask(src_tensor, meta["src_pad_id"])

    # decoder 输入从目标语言 <bos> 开始。
    generated_ids = [meta["tgt_bos_id"]]

    for _ in range(max_decode_len):
        tgt_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

        # 推理阶段也必须使用 causal mask，保证 decoder 只能看到已生成 token。
        tgt_mask = make_tgt_mask(tgt_tensor, meta["tgt_pad_id"])

        logits = model(
            src=src_tensor,
            tgt=tgt_tensor,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )

        # 只取最后一个时间步的 logits，因为它对应“下一个 token”的预测。
        next_token_id = int(torch.argmax(logits[0, -1]).item())
        generated_ids.append(next_token_id)

        # 生成 <eos> 表示句子结束。
        if next_token_id == meta["tgt_eos_id"]:
            break

    # pieces 用于 --show_tokens 调试展示；最终文本由 SentencePiece decode 负责。
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
    """加载训练保存的 checkpoint。"""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在: {path}\n"
            f"请先运行 train.py，或用 --checkpoint 指向已有 checkpoint。"
        )

    return torch.load(path, map_location=device, weights_only=False)


def parse_args():
    """解析推理命令行参数。"""
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
    """推理主入口，支持单句模式和交互模式。"""
    args = parse_args()

    # 默认自动选择 CUDA；如果只是检查流程，可以传 --device cpu。
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("使用设备:", device)
    print("加载 checkpoint:", args.checkpoint)

    checkpoint = load_checkpoint(args.checkpoint, device)
    meta = checkpoint["meta"]

    # tokenizer 和模型结构都从 checkpoint 恢复，避免训练/推理不一致。
    spm_model_path = resolve_spm_model_path(checkpoint, args.spm_model)
    processor = load_spm(spm_model_path)
    model = build_model_from_checkpoint(checkpoint, device)

    print("SentencePiece:", spm_model_path)
    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint valid loss:", checkpoint.get("valid_loss"))

    def translate_and_print(text: str):
        """翻译一句文本并打印结果。"""
        result = greedy_decode(
            model=model,
            text=text,
            processor=processor,
            meta=meta,
            device=device,
            max_decode_len=args.max_decode_len,
        )

        # print("中文:", text)

        if args.show_tokens:
            print("中文 pieces:", result["src_pieces"])
            print("英文 pieces:", result["output_pieces"])
            print("生成 ids:", result["generated_ids"])

        print("英文:", result["translation"])

    if args.text is not None:
        # 命令行单句模式：传入 --text 时只翻译这一句。
        translate_and_print(args.text)
        return

    # 未传 --text 时进入交互模式。
    print("\n进入交互模式。输入中文句子后回车；输入 q 或空行退出。")

    while True:
        text = input("\n中文> ").strip()

        if not text or text.lower() in {"q", "quit", "exit"}:
            break

        translate_and_print(text)


if __name__ == "__main__":
    main()
