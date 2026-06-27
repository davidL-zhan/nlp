"""
训练中译英 Transformer 模型。

运行前需要先完成：
    1. download.py        把 data/cmn.txt 切分成 train/valid/test 文本；
    2. build_vocab.py     用训练集建立中英文词表；
    3. encode_dataset.py  把文本转成 token id，并保存为 .pt 文件。

本脚本负责：
    1. 读取 dataset_dataloader.py 构建的 DataLoader；
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
import time

import torch
import torch.nn as nn

from dataset_dataloader import build_dataloaders
from Transformer import Transformer, make_src_mask, make_tgt_mask
from utils import compute_loss, move_batch_to_device, set_seed


def build_model(meta: dict, args: argparse.Namespace, device: torch.device):
    """
    根据 meta.json 里的词表大小创建 Transformer。

    meta["src_vocab_size"] 是中文源语言词表大小。
    meta["tgt_vocab_size"] 是英文目标语言词表大小。

    这里没有直接使用 Transformer 默认的 512 维、6 层配置，
    因为当前数据集只有两万多条，先用较小模型更容易快速验证训练链路。
    等确认 loss 能正常下降后，再把层数和隐藏维度调大。
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

    注意 tgt_input 和 tgt_output 已经在 dataset_dataloader.py 里错开一位：
        tgt_input : <bos> I like cats .
        tgt_output:       I like cats . <eos>
    """
    model.train()

    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for step, batch in enumerate(train_loader, start=1):
        if max_batches > 0 and step > max_batches:
            break

        batch = move_batch_to_device(batch, device)

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

    total_loss = 0.0
    total_tokens = 0

    for step, batch in enumerate(valid_loader, start=1):
        if max_batches > 0 and step > max_batches:
            break

        batch = move_batch_to_device(batch, device)

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
    parser = argparse.ArgumentParser(
        description="Train a Transformer model for Chinese-to-English translation."
    )

    # 数据和保存路径
    parser.add_argument("--encoded_dir", type=str, default="data/cmn_eng_encoded")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=500)
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
    args = parse_args()
    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("使用设备:", device)

    train_loader, valid_loader, _, meta = build_dataloaders(
        encoded_dir=args.encoded_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("中文词表大小:", meta["src_vocab_size"])
    print("英文词表大小:", meta["tgt_vocab_size"])
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
