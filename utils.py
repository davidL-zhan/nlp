"""
训练和评估阶段通用的小工具函数。

这个文件只放和文本分词无关、和训练流程相关的通用逻辑。
文本分词、JSON 读写和 token 编码放在 tokenizer.py。
"""

import random

import torch


def set_seed(seed: int):
    """
    固定随机种子，让初始化、shuffle 等行为尽量可复现。

    注意：
        GPU 上部分算子仍可能存在轻微非确定性。
        这里主要用于让训练调试过程更稳定。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device):
    """
    把 batch 中的 tensor 移动到指定设备。

    DataLoader 返回的 batch 里既有 tensor，也有原始文本列表。
    原始文本列表不能调用 .to(device)，所以这里只移动 tensor。
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_loss(logits: torch.Tensor, tgt_output: torch.Tensor, criterion):
    """
    计算序列到序列任务的 token-level 交叉熵 loss。

    logits:
        模型输出，形状是 [batch_size, tgt_len, tgt_vocab_size]。

    tgt_output:
        真实答案，形状是 [batch_size, tgt_len]。

    CrossEntropyLoss 需要：
        logits: [N, num_classes]
        target: [N]

    因此这里把 batch_size 和 tgt_len 两个维度展平。
    """
    vocab_size = logits.size(-1)
    return criterion(
        logits.reshape(-1, vocab_size),
        tgt_output.reshape(-1),
    )
