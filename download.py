"""
把原始 cmn.txt 平行语料切分成本项目训练需要的 train / valid / test 文件。

原始 cmn.txt 的常见格式是：
    English sentence<TAB>中文句子<TAB>其他来源信息

本项目做的是“中译英”，所以：
    source = 中文，保存到 .zh 文件；
    target = 英文，保存到 .en 文件。

脚本运行后会在 data 目录下生成：
    train.zh / train.en
    valid.zh / valid.en
    test.zh  / test.en

注意：
    这个脚本只负责切分原始文本，不负责分词、建词表或转 token id。
    后续流程是 build_vocab.py -> encode_dataset.py -> train.py。
"""

from pathlib import Path
import random

# 原始中英平行语料路径。
# 当前项目把 cmn.txt 放在 data 目录下，所以这里使用相对路径。
input_path = Path(r"data\cmn.txt")

# 切分后的 train / valid / test 文件也保存到 data 目录。
out_dir = Path(r"data")
out_dir.mkdir(parents=True, exist_ok=True)

# pairs 中的每个元素都是一个二元组：
#     (中文句子, 英文句子)
# 这样后面写文件时可以直接把中文写到 .zh，把英文写到 .en。
pairs = []

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        # cmn.txt 通常使用 tab 分隔。
        # 这里只需要前两列：英文和中文；后面的来源信息直接忽略。
        parts = line.rstrip("\n").split("\t")

        if len(parts) < 2:
            continue

        en = parts[0].strip()
        zh = parts[1].strip()

        if not en or not zh:
            continue

        # 中译英：source = zh, target = en
        pairs.append((zh, en))

print("Total pairs:", len(pairs))

# 固定随机种子，保证每次切分得到的 train / valid / test 都一致。
random.seed(42)
random.shuffle(pairs)

# 按 90% / 5% / 5% 切分训练集、验证集、测试集。
n = len(pairs)
n_train = int(n * 0.9)
n_valid = int(n * 0.05)

train_pairs = pairs[:n_train]
valid_pairs = pairs[n_train : n_train + n_valid]
test_pairs = pairs[n_train + n_valid :]

splits = {
    "train": train_pairs,
    "valid": valid_pairs,
    "test": test_pairs,
}

for split, data in splits.items():
    # 每个 split 保存成两个平行文件：
    #     train.zh 和 train.en 的第 N 行互相对应；
    #     valid / test 同理。
    zh_path = out_dir / f"{split}.zh"
    en_path = out_dir / f"{split}.en"

    with zh_path.open("w", encoding="utf-8") as f_zh, en_path.open(
        "w", encoding="utf-8"
    ) as f_en:
        for zh, en in data:
            f_zh.write(zh + "\n")
            f_en.write(en + "\n")

    print(split, len(data), zh_path, en_path)
