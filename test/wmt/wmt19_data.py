from datasets import load_dataset

dataset = load_dataset("wmt/wmt19", "zh-en")
print(dataset)
print(dataset["train"][10])
