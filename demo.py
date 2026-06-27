"""
终端版中译英演示程序。

这个脚本是 inference.py 的轻量封装：
    1. 启动时加载 checkpoints/best.pt；
    2. 在终端里循环读取中文输入；
    3. 每输入一句中文就调用 greedy_decode 翻译；
    4. 输入 !bye 时退出。

适合用于课堂展示或快速体验模型效果。
"""

import argparse
import torch

from inference import (
    build_model_from_checkpoint,
    greedy_decode,
    load_checkpoint,
)
from tokenizer import load_json


def repair_windows_pipe_mojibake(text: str):
    """
    修复一种常见的 Windows PowerShell 管道输入乱码。

    在某些 Windows 终端里，PowerShell 管道传给 Python 的中文是 UTF-8 字节，
    但 Python 可能按 GBK 去解码 stdin，于是 "你好。" 会变成类似 "浣犲ソ銆..."。

    正常键盘输入如果已经是正确中文，这个函数通常不会改动。
    只有当它能把疑似乱码重新还原成有效 UTF-8 中文时，才返回修复后的文本。
    """
    try:
        repaired = text.encode("gbk", errors="surrogateescape").decode("utf-8")
    except UnicodeError:
        return text

    has_chinese = any("\u4e00" <= char <= "\u9fff" for char in repaired)
    if repaired != text and has_chinese:
        return repaired

    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Terminal demo for Chinese-to-English translation."
    )

    # 默认使用 train.py 保存的最佳模型。
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")

    # 中文 token -> id 词表，用来把终端输入的中文句子转成模型输入。
    parser.add_argument("--src_vocab", type=str, default="data/vocab/src_zh_jieba_vocab.json")

    # 英文 id -> token 词表，用来把模型生成的 token id 转回英文单词。
    parser.add_argument("--tgt_id2token", type=str, default="data/vocab/tgt_en_id2token.json")

    # 最多生成多少个英文 token。太小会截断翻译，太大可能生成很长的重复句子。
    parser.add_argument("--max_decode_len", type=int, default=60)

    # 默认自动选择 cuda/cpu；也可以手动指定 --device cpu 或 --device cuda。
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("中英翻译 Demo")
    print("输入中文句子后按回车，输入 !bye 结束。")
    print("使用设备:", device)
    print("加载模型:", args.checkpoint)

    checkpoint = load_checkpoint(args.checkpoint, device)
    meta = checkpoint["meta"]

    src_vocab = load_json(args.src_vocab)
    tgt_id2token = load_json(args.tgt_id2token)
    model = build_model_from_checkpoint(checkpoint, device)

    print("模型加载完成。")

    while True:
        text = input("\n中文> ").strip()
        text = repair_windows_pipe_mojibake(text)

        if text == "!bye":
            print("已退出。")
            break

        if not text:
            print("请输入一句中文，或输入 !bye 结束。")
            continue

        result = greedy_decode(
            model=model,
            text=text,
            src_vocab=src_vocab,
            tgt_id2token=tgt_id2token,
            meta=meta,
            device=device,
            max_decode_len=args.max_decode_len,
        )

        print("英文>", result["translation"])


if __name__ == "__main__":
    main()
