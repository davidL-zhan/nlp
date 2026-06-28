"""
训练中译英 Transformer 模型。

数据入口是 HuggingFace WMT19 + SentencePiece：
    1. spm.py train 训练 data/spm/*.model；
    2. train.py     读取 HuggingFace datasets 本地缓存；
    3. DataLoader   动态用 SentencePiece 编码 token id。

本脚本负责：
    1. 读取 data.py 构建的 DataLoader；
    2. 初始化 Transformer；
    3. 构造注意力 mask；
    4. 计算交叉熵 loss；
    5. 训练、验证并保存 checkpoint。

默认 checkpoint 输出：
    checkpoints/last.pt  最后一个 epoch 的模型；
    checkpoints/best.pt  验证集 loss 最低的模型。
"""

from pathlib import Path
import argparse
import random
import time

import torch
import torch.nn as nn

from data import build_loaders
from model import Transformer, make_src_mask, make_tgt_mask
from spm import DEFAULT_SPM_MODEL


def set_seed(seed: int):
    """
    固定随机种子，尽量保证调试时可复现。

    注意：GPU 上某些底层算子仍可能存在非确定性，
    这里主要用于固定 Python 随机数、PyTorch 初始化和 DataLoader shuffle。
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict, device: torch.device):
    """
    把 batch 中的 tensor 移动到指定设备。

    batch 里既有模型输入 tensor，也有调试用的原始文本和 pieces。
    原始文本是 list[str]，不能调用 .to(device)，所以这里只移动 tensor。
    """
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def compute_loss(logits: torch.Tensor, tgt_output: torch.Tensor, criterion):
    """
    计算 token-level 交叉熵 loss。

    模型输出 logits 的形状是：
        [batch_size, tgt_len, tgt_vocab_size]

    CrossEntropyLoss 需要：
        input:  [N, num_classes]
        target: [N]

    因此这里把 batch_size 和 tgt_len 两个维度展平。
    """
    vocab_size = logits.size(-1)
    return criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))


def build_model(meta: dict, args: argparse.Namespace, device: torch.device):
    """
    根据 SentencePiece meta 里的词表大小创建 Transformer。

    当前使用共享 SentencePiece 词表，所以 src_vocab_size 和 tgt_vocab_size
    通常相同，但仍保留两个字段，方便模型接口保持清晰。
    """
    model = Transformer(
        src_vocab_size=meta["src_vocab_size"],
        tgt_vocab_size=meta["tgt_vocab_size"],
        embeding_size=args.embeding_size,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        Mlp_hidden_size=args.mlp_hidden_size,
        dropout=args.dropout,
        max_len=args.max_len,
    )

    return model.to(device)


def build_training_dataloaders(args: argparse.Namespace):
    """
    根据命令行参数构建训练/验证/测试 DataLoader。

    具体数据读取和 SentencePiece 编码逻辑都在 data.py 中，
    train.py 只关心返回的 batch 是否包含 src_ids、tgt_input、tgt_output。
    """
    return build_loaders(
        dataset_name=args.hf_dataset,
        dataset_config=args.hf_config,
        train_split=args.hf_train_split,
        valid_split=args.hf_valid_split,
        spm_model=args.spm_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_src_len=args.max_src_tokens,
        max_tgt_len=args.max_tgt_tokens,
        max_train_samples=args.hf_train_samples,
        max_valid_samples=args.hf_valid_samples,
        max_test_samples=args.hf_test_samples,
        seed=args.seed,
        cache_dir=args.hf_cache_dir,
    )


def train_epoch(
    model,
    train_loader,
    meta,
    criterion,
    optimizer,
    device,
    grad_clip: float,
    log_interval: int,
    max_batches: int,
):
    """
    训练一个 epoch。

    每个 batch 的训练流程是：
    1. 取出 src_ids、tgt_input、tgt_output。
    2. 根据 src_ids 生成 encoder 和 cross-attention 使用的 src_mask。
    3. 根据 tgt_input 生成 decoder self-attention 使用的 tgt_mask。
    4. 前向传播得到 logits。
    5. 用 logits 和 tgt_output 计算 loss。
    6. 反向传播，更新参数。

    注意 tgt_input 和 tgt_output 已经在 DataLoader collator 里错开一位：
        tgt_input : <bos> I like cats .
        tgt_output:       I like cats . <eos>
    """
    model.train()

    # total_loss 累计的是“每个非 pad token 的 loss 总和”，
    # total_tokens 累计的是非 pad token 数，最终返回 token 平均 loss。
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for step, batch in enumerate(train_loader, start=1):
        if max_batches > 0 and step > max_batches:
            break

        batch = move_batch_to_device(batch, device)

        # src_ids: 中文源句 token id，形状 [B, src_len]
        # tgt_input: decoder 输入，形状 [B, tgt_len - 1]
        # tgt_output: decoder 监督目标，形状 [B, tgt_len - 1]
        src_ids = batch["src_ids"]
        tgt_input = batch["tgt_input"]
        tgt_output = batch["tgt_output"]

        # src_mask 形状是 [B, 1, 1, src_len]。
        # 它会在 encoder self-attention 和 decoder cross-attention 中使用。
        # 作用是屏蔽中文源句子里的 <pad>。
        src_mask = make_src_mask(src_ids, meta["src_pad_id"])

        # tgt_mask 形状是 [B, 1, tgt_len, tgt_len]。
        # 它同时屏蔽英文目标句子里的 <pad> 和当前位置之后的未来 token。
        tgt_mask = make_tgt_mask(tgt_input, meta["tgt_pad_id"])

        optimizer.zero_grad(set_to_none=True)

        # 前向传播：
        # encoder 读取 src_ids；
        # decoder 在 tgt_input 的 teacher forcing 条件下预测每个下一个 token。
        logits = model(
            src=src_ids,
            tgt=tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )

        loss = compute_loss(logits, tgt_output, criterion)
        loss.backward()

        # 梯度裁剪可以避免训练早期梯度突然过大导致 loss 变成 nan。
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # 只统计非 padding 的目标 token，这样平均 loss 不会被 <pad> 数量干扰。
        non_pad_tokens = tgt_output.ne(meta["tgt_pad_id"]).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

        if log_interval > 0 and step % log_interval == 0:
            avg_loss = total_loss / max(total_tokens, 1)
            elapsed = time.time() - start_time
            print(
                f"  step {step:5d} | "
                f"train loss {avg_loss:.4f} | "
                f"tokens {total_tokens} | "
                f"time {elapsed:.1f}s"
            )

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def valid_epoch(model, valid_loader, meta, criterion, device, max_batches: int):
    """
    在验证集上评估一个 epoch。

    这里使用 torch.no_grad()，不会保存反向传播需要的中间梯度，
    所以验证阶段显存占用更小，速度也更快。
    """
    model.eval()

    # 验证阶段只统计 loss，不更新参数。
    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(valid_loader, start=1):
        if max_batches > 0 and step > max_batches:
            break

        batch = move_batch_to_device(batch, device)

        # 验证阶段的 mask 构造和训练阶段完全一致，保证评估口径一致。
        src_ids = batch["src_ids"]
        tgt_input = batch["tgt_input"]
        tgt_output = batch["tgt_output"]

        src_mask = make_src_mask(src_ids, meta["src_pad_id"])
        tgt_mask = make_tgt_mask(tgt_input, meta["tgt_pad_id"])

        logits = model(
            src=src_ids,
            tgt=tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=src_mask,
        )

        loss = compute_loss(logits, tgt_output, criterion)

        non_pad_tokens = tgt_output.ne(meta["tgt_pad_id"]).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    return total_loss / max(total_tokens, 1)


def save_checkpoint(
    checkpoint_path: Path,
    model,
    optimizer,
    epoch: int,
    train_loss: float,
    valid_loss: float,
    meta: dict,
    args: argparse.Namespace,
):
    """
    保存训练断点。

    checkpoint 里保存：
    1. model_state_dict: 模型参数。
    2. optimizer_state_dict: 优化器状态，之后如果要继续训练会用到。
    3. meta: 词表大小、pad id、bos/eos id 等信息。
    4. args: 本次训练使用的超参数。
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # checkpoint 中保存 meta 和 args，是为了让 infer.py 能恢复完全一致的模型结构、
    # tokenizer 路径、词表大小和特殊 token id。
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "meta": meta,
            "args": vars(args),
        },
        checkpoint_path,
    )


def parse_args():
    """
    定义训练命令行参数。

    参数大致分三类：
    1. 数据/tokenizer/checkpoint 路径；
    2. 训练超参数；
    3. Transformer 模型结构参数。
    """
    parser = argparse.ArgumentParser(
        description="Train a Transformer model for Chinese-to-English translation."
    )

    # 数据和保存路径
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--spm_model",
        type=str,
        default=DEFAULT_SPM_MODEL,
    )
    parser.add_argument("--hf_dataset", type=str, default="wmt/wmt19")
    parser.add_argument("--hf_config", type=str, default="zh-en")
    parser.add_argument("--hf_train_split", type=str, default="train")
    parser.add_argument("--hf_valid_split", type=str, default="validation")
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--hf_train_samples", type=int, default=100000)
    parser.add_argument("--hf_valid_samples", type=int, default=3981)
    parser.add_argument("--hf_test_samples", type=int, default=3981)
    parser.add_argument("--max_src_tokens", type=int, default=128)
    parser.add_argument("--max_tgt_tokens", type=int, default=128)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    # 模型参数
    parser.add_argument("--embeding_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--mlp_hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=5000)

    # 调试参数：
    # 如果设置为 0，表示使用完整 train/valid。
    # 如果设置为正数，例如 2，就只跑前 2 个 batch，方便快速检查代码是否能跑通。
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_valid_batches", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)

    # device 默认自动选择 cuda 或 cpu。
    # 也可以手动指定 --device cpu 或 --device cuda。
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def main():
    """训练主入口。"""
    args = parse_args()
    set_seed(args.seed)

    # 默认自动使用 CUDA；如果需要 CPU 调试，可以传 --device cpu。
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("使用设备:", device)

    train_loader, valid_loader, _, meta = build_training_dataloaders(args)

    # 打印关键数据配置，方便确认当前训练到底使用了多少样本和多大词表。
    print("数据后端: wmt19_spm")
    print("训练样本数:", len(train_loader.dataset))
    print("验证样本数:", len(valid_loader.dataset))
    print("源端词表大小:", meta["src_vocab_size"])
    print("目标端词表大小:", meta["tgt_vocab_size"])
    print("src_pad_id:", meta["src_pad_id"])
    print("tgt_pad_id:", meta["tgt_pad_id"])

    model = build_model(meta, args, device)

    # ignore_index=tgt_pad_id 表示计算 loss 时忽略目标端 padding。
    # 否则模型会被迫学习预测大量 <pad>，影响真正词语的学习。
    criterion = nn.CrossEntropyLoss(ignore_index=meta["tgt_pad_id"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    best_checkpoint_path = checkpoint_dir / "best.pt"
    last_checkpoint_path = checkpoint_dir / "last.pt"

    best_valid_loss = float("inf")

    # 每个 epoch 后保存 last.pt；如果验证集 loss 更低，则额外保存 best.pt。
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            meta=meta,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            max_batches=args.max_train_batches,
        )

        valid_loss = valid_epoch(
            model=model,
            valid_loader=valid_loader,
            meta=meta,
            criterion=criterion,
            device=device,
            max_batches=args.max_valid_batches,
        )

        print(
            f"Epoch {epoch} finished | train loss {train_loss:.4f} | valid loss {valid_loss:.4f}"
        )

        save_checkpoint(
            checkpoint_path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            valid_loss=valid_loss,
            meta=meta,
            args=args,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                meta=meta,
                args=args,
            )
            print("已保存新的最佳模型:", best_checkpoint_path)

    print("\n训练完成。")
    print("最后一次 checkpoint:", last_checkpoint_path)
    print("最佳 checkpoint:", best_checkpoint_path)
    print("最佳 valid loss:", f"{best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
