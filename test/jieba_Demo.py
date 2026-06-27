"""
jieba 分词最小演示脚本。

这个脚本不参与训练流程，只用于观察 jieba 会如何切分中文句子。
项目里的 build_vocab.py、encode_dataset.py、inference.py 都依赖 jieba，
所以在调试中文分词效果时可以先运行这个文件。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import jieba

from tokenizer import load_json, tokenize_zh

# jieba.cut 返回的是一个可迭代对象，不是普通 list。
# 直接 print 时只能看到对象信息，不能看到具体分词结果。
print(jieba.cut("我来到北京清华大学"))

# 下面两行是查看中文词表大小和内容的临时调试代码。
# 注意：正式的数据读取逻辑在 build_vocab.py / encode_dataset.py 中；
# 训练和推理不会调用这个测试脚本。
vocab2id = load_json("data/vocab/src_zh_jieba_vocab.json")

unk_id = vocab2id["<unk>"]
id2vocab = load_json(r"data\vocab\src_zh_jieba_id2token.json")
# print(id2vocab)
# 逐个遍历 tokenize_zh 的结果，就可以看到当前项目真实使用的中文 token。
for word in tokenize_zh("我来到北京清华大学"):
    token_id = vocab2id.get(word, unk_id)
    token = id2vocab.get(str(token_id), "UNK")
    print(word, token_id, token)
