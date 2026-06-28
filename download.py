"""
下载并缓存 WMT19 zh-en 数据集。

这个脚本只负责把 HuggingFace 上的 `wmt/wmt19` 数据集下载到本机缓存，
不负责训练 tokenizer，也不负责训练 Transformer。

默认行为：
    python.exe download.py

执行后 HuggingFace Datasets 会把数据保存到本机 cache。后续 data.py、train.py
再次调用 load_dataset("wmt/wmt19", "zh-en") 时，会优先复用这份本地缓存。
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from spm import DEFAULT_CONFIG, DEFAULT_DATASET, clean_text


def format_size(num_bytes: int):
    """把字节数格式化成便于阅读的 MB/GB。"""
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"

    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MB"

    return f"{num_bytes} B"


def file_size(path: str | Path):
    """读取单个缓存文件大小；文件不存在时返回 0。"""
    path = Path(path)
    if not path.exists():
        return 0

    return path.stat().st_size


def print_cache_files(dataset_dict):
    """
    打印每个 split 对应的 Arrow cache 文件。

    这些文件就是 HuggingFace Datasets 处理后的本地数据缓存。
    """
    total_size = 0

    for split_name, split_data in dataset_dict.items():
        print(f"\n[{split_name}]")
        print("样本数:", split_data.num_rows)
        print("字段:", list(split_data.features.keys()))

        cache_files = split_data.cache_files
        print("cache 文件数:", len(cache_files))

        for cache_file in cache_files:
            filename = cache_file["filename"]
            size = file_size(filename)
            total_size += size
            print(f"- {filename} ({format_size(size)})")

    print("\nArrow cache 总大小:", format_size(total_size))


def print_sample(dataset_dict, index: int):
    """打印 train split 中的一条样本，确认字段和中英文本正常。"""
    if "train" not in dataset_dict:
        print("\n没有 train split，跳过样本打印。")
        return

    train_data = dataset_dict["train"]
    if len(train_data) == 0:
        print("\ntrain split 为空，跳过样本打印。")
        return

    index = min(index, len(train_data) - 1)
    sample = train_data[index]
    translation = sample["translation"]

    print(f"\ntrain[{index}]")
    print("zh:", clean_text(translation.get("zh", "")))
    print("en:", clean_text(translation.get("en", "")))


def parse_args():
    """解析下载脚本参数。"""
    parser = argparse.ArgumentParser(
        description="Download/cache HuggingFace WMT19 zh-en dataset."
    )

    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="强制重新下载并重建缓存")
    parser.add_argument("--show_sample", action="store_true", help="下载后打印一条样本")
    parser.add_argument("--sample_index", type=int, default=10)

    return parser.parse_args()


def main():
    """下载入口。"""
    args = parse_args()

    print("数据集:", args.dataset)
    print("配置:", args.config)
    print("cache_dir:", args.cache_dir or "HuggingFace 默认缓存目录")

    load_kwargs = {
        "cache_dir": args.cache_dir,
    }

    if args.force:
        # force_redownload 会忽略已有下载缓存，重新从 HuggingFace 拉取数据。
        load_kwargs["download_mode"] = "force_redownload"

    dataset_dict = load_dataset(args.dataset, args.config, **load_kwargs)

    print("\n下载/缓存完成。DatasetDict:")
    print(dataset_dict)

    print_cache_files(dataset_dict)

    if args.show_sample:
        print_sample(dataset_dict, args.sample_index)


if __name__ == "__main__":
    main()
