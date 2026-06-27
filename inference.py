"""
加载训练好的 Transformer checkpoint，执行中译英推理。

这个文件提供两种用法：
1. 命令行单句翻译：
       python inference.py --text "你好。"

2. 不传 --text 时进入交互模式：
       python inference.py

推理流程：
    中文句子 -> jieba 分词 -> 中文 token id -> Transformer greedy decoding -> 英文 token

注意：
    推理时使用的中文分词、词表、bos/eos/pad id 必须和训练阶段保持一致。
"""

from pathlib import Path
import argparse

import torch

from tokenizer import load_json, tokenize_zh
from Transformer import Transformer, make_src_mask, make_tgt_mask


def encode_source_text(text: str, src_vocab: dict, meta: dict):
    """
    把输入中文句子编码成模型可以接收的 src_ids。

    处理流程：
    1. 使用 jieba 对中文句子分词。
    2. 在句首添加 <bos>。
    3. 把每个中文 token 转成词表 id。
    4. 如果 token 不在词表中，使用 <unk> 的 id。
    5. 在句尾添加 <eos>。

    返回：
        ids:    List[int]，例如 [2, 45, 87, 3]
        tokens: List[str]，方便调试时查看 jieba 分词结果
    """
    tokens = tokenize_zh(text)

    max_src_len = meta.get("max_src_len", 128)

    # max_src_len 包含 <bos> 和 <eos>。
    # 如果输入句子太长，就先截断中文 token，避免超过训练时的最大长度。
    max_token_len = max_src_len - 2
    if len(tokens) > max_token_len:
        tokens = tokens[:max_token_len]

    ids = [meta["src_bos_id"]]

    unk_id = meta["src_unk_id"]
    for token in tokens:
        ids.append(src_vocab.get(token, unk_id))

    ids.append(meta["src_eos_id"])

    return ids, tokens


def build_model_from_checkpoint(checkpoint: dict, device: torch.device):
    """
    根据 checkpoint 中保存的 meta 和 args 重建 Transformer。

    训练时 train.py 保存了：
        checkpoint["meta"]：词表大小、pad/bos/eos id 等信息；
        checkpoint["args"]：模型超参数，例如 embeding_size、层数、head 数。

    推理时必须用和训练时完全一致的模型结构。
    如果结构不一致，load_state_dict 会因为参数形状不同而报错。
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
    src_vocab: dict,
    tgt_id2token: dict,
    meta: dict,
    device: torch.device,
    max_decode_len: int,
):
    """
    使用 greedy decoding 做中译英。

    greedy decoding 的含义：
    每一步都选择当前概率最大的英文 token，作为下一个输出 token。

    解码流程：
    1. 先把中文句子编码成 src_ids。
    2. decoder 输入从英文 <bos> 开始。
    3. 每次把当前已生成的英文 token 序列喂给 decoder。
    4. 取最后一个位置的 logits，选择概率最大的 token。
    5. 把新 token 追加到 decoder 输入中。
    6. 如果生成 <eos>，或者达到 max_decode_len，就停止。

    注意：
    这里为了代码清晰，每一步都会重新调用完整 model。
    这比缓存 encoder memory 的写法慢一些，但更容易理解，也更适合当前学习项目。
    """
    src_ids, src_tokens = encode_source_text(text, src_vocab, meta)

    # 模型输入必须是二维：[batch_size, src_len]。
    # 单句推理时 batch_size = 1。
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

    # 源端 mask 用于：
    # 1. encoder self-attention 屏蔽中文 <pad>；
    # 2. decoder cross-attention 屏蔽 encoder 里对应中文 <pad> 的位置。
    # 单句推理通常没有 padding，但为了和训练逻辑完全一致，这里仍然生成 src_mask。
    src_mask = make_src_mask(src_tensor, meta["src_pad_id"])

    generated_ids = [meta["tgt_bos_id"]]

    for _ in range(max_decode_len):
        tgt_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

        # decoder self-attention 仍然需要 causal mask，
        # 保证当前位置不能看到未来 token。
        tgt_mask = make_tgt_mask(tgt_tensor, meta["tgt_pad_id"])

        logits = model(
            src=src_tensor,
            tgt=tgt_tensor,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )

        # logits 形状是 [1, 当前英文长度, 英文词表大小]。
        # 只取最后一个位置，因为我们只需要预测下一个 token。
        next_token_logits = logits[0, -1]
        next_token_id = int(torch.argmax(next_token_logits).item())

        generated_ids.append(next_token_id)

        if next_token_id == meta["tgt_eos_id"]:
            break

    output_tokens = []
    special_ids = {
        meta["tgt_pad_id"],
        meta["tgt_bos_id"],
        meta["tgt_eos_id"],
    }

    for token_id in generated_ids:
        if token_id in special_ids:
            continue

        # tgt_id2token 的 key 是 json 字符串，所以这里要把 int 转成 str。
        output_tokens.append(tgt_id2token.get(str(token_id), "<unk>"))

    translation = " ".join(output_tokens)

    return {
        "src_tokens": src_tokens,
        "generated_ids": generated_ids,
        "output_tokens": output_tokens,
        "translation": translation,
    }


def load_checkpoint(path: str | Path, device: torch.device):
    """
    加载 train.py 保存的 checkpoint。

    如果还没有训练正式模型，默认路径 checkpoints/best.pt 不存在。
    这种情况下需要先运行：
        python train.py
    或者用 --checkpoint 指向已经存在的 checkpoint。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint 不存在: {path}\n"
            f"请先训练模型，或使用 --checkpoint 指向已有的 best.pt / last.pt。"
        )

    return torch.load(path, map_location=device, weights_only=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained Chinese-to-English Transformer."
    )

    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--src_vocab", type=str, default="data/vocab/src_zh_jieba_vocab.json")
    parser.add_argument("--tgt_id2token", type=str, default="data/vocab/tgt_en_id2token.json")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--max_decode_len", type=int, default=60)
    parser.add_argument("--device", type=str, default=None)

    # 打开后会额外打印中文分词结果和英文 token 序列，方便排查推理效果。
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

    src_vocab = load_json(args.src_vocab)
    tgt_id2token = load_json(args.tgt_id2token)

    model = build_model_from_checkpoint(checkpoint, device)

    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("checkpoint valid loss:", checkpoint.get("valid_loss"))

    def translate_and_print(text: str):
        result = greedy_decode(
            model=model,
            text=text,
            src_vocab=src_vocab,
            tgt_id2token=tgt_id2token,
            meta=meta,
            device=device,
            max_decode_len=args.max_decode_len,
        )

        print("中文:", text)

        if args.show_tokens:
            print("中文 tokens:", result["src_tokens"])
            print("英文 tokens:", result["output_tokens"])
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
