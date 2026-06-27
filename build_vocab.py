"""
用训练集建立中英文词表。

输入文件：
    data/train.zh
    data/train.en

输出文件：
    data/vocab/src_zh_jieba_vocab.json       中文 token -> id
    data/vocab/src_zh_jieba_id2token.json   中文 id -> token
    data/vocab/tgt_en_vocab.json            英文 token -> id
    data/vocab/tgt_en_id2token.json         英文 id -> token

分词策略：
    中文：jieba 分词；
    英文：简单空格分词并转小写。

特殊 token 固定放在词表最前面：
    <pad>=0, <unk>=1, <bos>=2, <eos>=3
"""

from pathlib import Path
from collections import Counter

from tokenizer import SPECIAL_TOKENS, save_json, tokenize_en, tokenize_zh


# =========================
# 1. 从文件统计 token 频率
# =========================
def count_tokens(file_path: Path, tokenizer):
    """
    统计一个文本文件中的 token 频率。

    参数：
        file_path:
            待统计的文本文件路径，例如 data/train.zh。
        tokenizer:
            分词函数。中文传 tokenize_zh，英文传 tokenize_en。

    返回：
        Counter，其中 key 是 token，value 是该 token 在文件中出现的次数。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    counter = Counter()

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            tokens = tokenizer(line)
            counter.update(tokens)

    return counter


# =========================
# 2. 根据频率建立词表
# =========================
def build_vocab(counter: Counter, min_freq: int = 1, max_size: int | None = None):
    """
    counter: token 频率统计
    min_freq: 低于该频率的 token 不加入词表
    max_size: 最大词表大小，包含特殊 token
    """
    vocab = {}

    # 先加入特殊 token，保证 id 固定
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    # 再加入普通 token
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue

        if token in vocab:
            continue

        if max_size is not None and len(vocab) >= max_size:
            break

        vocab[token] = len(vocab)

    return vocab


# =========================
# 3. 主函数
# =========================
def main():
    """
    建词表主入口。

    运行：
        python build_vocab.py

    会读取 data/train.zh 和 data/train.en，
    分别建立中文源语言词表和英文目标语言词表。
    """
    data_dir = Path(r"data")

    train_zh_path = data_dir / "train.zh"
    train_en_path = data_dir / "train.en"

    vocab_dir = Path(r"data/vocab")
    vocab_dir.mkdir(parents=True, exist_ok=True)

    # 你可以先用 min_freq=1，保证词尽量完整
    # 后面如果词表太大，再改成 min_freq=2
    zh_min_freq = 1
    en_min_freq = 1

    zh_max_size = None
    en_max_size = None

    print("正在统计中文 token 频率...")
    zh_counter = count_tokens(train_zh_path, tokenize_zh)

    print("正在统计英文 token 频率...")
    en_counter = count_tokens(train_en_path, tokenize_en)

    print("正在建立中文词表...")
    zh_vocab = build_vocab(
        counter=zh_counter,
        min_freq=zh_min_freq,
        max_size=zh_max_size,
    )

    print("正在建立英文词表...")
    en_vocab = build_vocab(
        counter=en_counter,
        min_freq=en_min_freq,
        max_size=en_max_size,
    )

    print()
    print("中文 token 总数:", sum(zh_counter.values()))
    print("中文不同 token 数:", len(zh_counter))
    print("中文词表大小:", len(zh_vocab))

    print()
    print("英文 token 总数:", sum(en_counter.values()))
    print("英文不同 token 数:", len(en_counter))
    print("英文词表大小:", len(en_vocab))

    print()
    print("中文高频 token 前 30 个:")
    for token, freq in zh_counter.most_common(30):
        print(f"{token}\t{freq}")

    print()
    print("英文高频 token 前 30 个:")
    for token, freq in en_counter.most_common(30):
        print(f"{token}\t{freq}")

    # 保存 token -> id
    zh_vocab_path = vocab_dir / "src_zh_jieba_vocab.json"
    en_vocab_path = vocab_dir / "tgt_en_vocab.json"
    save_json(zh_vocab, zh_vocab_path)
    save_json(en_vocab, en_vocab_path)
    print(f"已保存: {zh_vocab_path}")
    print(f"已保存: {en_vocab_path}")

    # 额外保存 id -> token，方便后面把模型输出 id 转回文字
    zh_id2token = {str(idx): token for token, idx in zh_vocab.items()}
    en_id2token = {str(idx): token for token, idx in en_vocab.items()}

    zh_id2token_path = vocab_dir / "src_zh_jieba_id2token.json"
    en_id2token_path = vocab_dir / "tgt_en_id2token.json"
    save_json(zh_id2token, zh_id2token_path)
    save_json(en_id2token, en_id2token_path)
    print(f"已保存: {zh_id2token_path}")
    print(f"已保存: {en_id2token_path}")

    print()
    print("词表建立完成。")
    print("中文词表:", vocab_dir / "src_zh_jieba_vocab.json")
    print("英文词表:", vocab_dir / "tgt_en_vocab.json")


if __name__ == "__main__":
    main()
