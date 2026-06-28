"""
项目通用的分词、JSON 读写和 token 编码工具。

这个文件集中放置训练、编码、推理都会复用的基础函数，避免多个脚本各自复制一份。

当前统一规则：
    中文：使用 jieba.cut(text, cut_all=False)；
    英文：转小写后按空格切分；
    JSON：统一使用 UTF-8 读写；
    特殊 token：统一使用 <pad> / <unk> / <bos> / <eos>。

如果以后要更换中文分词方式，应该优先修改这里，
再让 build_vocab.py、encode_dataset.py、inference.py 自动复用同一套逻辑。
"""

from pathlib import Path
import json

import jieba

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]


def tokenize_zh(text: str):
    """
    中文分词函数。

    参数：
        text:
            原始中文句子。

    返回：
        list[str]，例如：
            "我喜欢自然语言处理" -> ["我", "喜欢", "自然语言", "处理"]

    注意：
        训练建词表、数据编码、推理输入都必须使用同一个 tokenize_zh。
        如果这三个阶段分词不一致，同一个中文词可能映射到不同 token id，
        模型效果会明显变差。
    """
    text = text.strip()
    tokens = jieba.cut(text, cut_all=False)
    return [tok.strip() for tok in tokens if tok.strip()]


def tokenize_en(text: str):
    """
    英文分词函数。

    当前使用最简单的空格分词，并统一转小写。

    例如：
        "I Like Cats." -> ["i", "like", "cats."]

    注意：
        这个实现不会单独拆分标点，所以 "cat." 和 "cat" 会被视为不同 token。
        对学习版 Transformer 项目来说足够直观；如果后续追求效果，可以再换更强的 tokenizer。
    """
    text = text.strip().lower()
    return [tok.strip() for tok in text.split() if tok.strip()]


def load_json(path: str | Path):
    """
    使用 UTF-8 读取 JSON 文件。

    参数：
        path:
            字符串路径或 pathlib.Path 对象。

    返回：
        json.load 解析后的 Python 对象，通常是 dict。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path):
    """
    使用 UTF-8 保存 JSON 文件。

    ensure_ascii=False 可以让中文 token 直接以中文保存，
    而不是变成 \\uXXXX，方便人工检查词表。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def encode_text(
    text: str,
    vocab: dict,
    tokenizer,
    add_bos: bool = True,
    add_eos: bool = True,
):
    """
    把一段文本编码成 token id。

    参数：
        text:
            原始文本。
        vocab:
            token -> id 词表。
        tokenizer:
            分词函数，例如 tokenize_zh 或 tokenize_en。
        add_bos / add_eos:
            是否在句首/句尾添加 <bos> 和 <eos>。

    返回：
        ids:
            token id 列表。
        tokens:
            分词后的 token 列表，便于调试。
    """
    tokens = tokenizer(text)  # ['我', '喜欢', '自然语言', '处理']

    ids = []

    if add_bos: #　ids :[2,]
        ids.append(vocab[BOS_TOKEN])

    unk_id = vocab[UNK_TOKEN]

    for token in tokens:  # ['我', '喜欢', '自然语言', '处理']
        token_id = vocab.get(token, unk_id)
        ids.append(token_id) 

    if add_eos:
        ids.append(vocab[EOS_TOKEN])

    return ids, tokens


if __name__ == "__main__":
    text = "我喜欢自然语言处理"
    src_vocab = load_json("data/vocab/src_zh_jieba_vocab.json")
    print(tokenize_zh(text))
    print(encode_text(text, src_vocab, tokenize_zh))
