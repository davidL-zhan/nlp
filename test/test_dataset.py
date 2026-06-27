"""
查看原始平行语料样本的小工具。

这个文件只用于人工检查 train.zh / train.en 是否一一对应，
不参与正式训练。它会打印前 20 条中文和英文句子，方便确认：
1. 中文和英文是否对齐；
2. 文件编码是否正常；
3. download.py 切分后的 data 目录是否符合预期。
"""

from pathlib import Path

# 当前项目的 train.zh / train.en 保存在 data 目录下。
data_dir = Path(r"data")

# train.zh 和 train.en 的第 N 行应该互为翻译。
zh_path = data_dir / "train.zh"
en_path = data_dir / "train.en"

with zh_path.open("r", encoding="utf-8") as f_zh, en_path.open(
    "r", encoding="utf-8"
) as f_en:
    # zip 会同时读取两个文件；如果两个文件行数不同，只会遍历到较短文件结束。
    for i, (zh, en) in enumerate(zip(f_zh, f_en)):
        print("ZH:", zh.strip())
        print("EN:", en.strip())
        print("-" * 50)

        # 只打印前 20 条，避免终端输出过长。
        if i >= 19:
            break
