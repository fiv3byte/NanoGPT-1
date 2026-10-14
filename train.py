"""
Train NanoGPT-1M on any CUDA or CPU.
"""
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import NanoGPT
from src.dataset import ByteDataset
from src.utils import get_device, save_quantized


def parse_args():
    p = argparse.ArgumentParser(description="Train NanoGPT ~1M params")
    p.add_argument("--block_size", type=int, default=128, help="context length")
    p.add_argument("--batch_size", type=int, default=32, help="batch size (P3000 can do 48–64)")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=5)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--data", type=str, default="data/input.txt")
    p.add_argument("--out_dir", type=str, default="checkpoints_utf8")
    p.add_argument("--compile", action="store_true", help="torch.compile (PyTorch 2+)")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataset: external UTF-8 file, or built-in fairy-tale if absent/short
    dataset = ByteDataset(args.data, block_size=args.block_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,          # safer on Windows / old drivers
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = NanoGPT(
        vocab_size=256,
        block_size=args.block_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model…")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    # cosine schedule with warm-up
    total_steps = args.epochs * len(loader)
    warmup = min(max(1, total_steps // 20), max(1, total_steps // 2))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = (device == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Steps per epoch: {len(loader)} | total steps: {total_steps}")
    print(f"AMP: {use_amp} | batch={args.batch_size} block={args.block_size}")

    best_loss = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        running = 0.0
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(xb, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

        avg = running / len(loader)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}  avg_loss={avg:.4f}  time={elapsed:.0f}s")

        # save every epoch
        ckpt = {
            "model": model.state_dict(),
            "config": {
                "vocab_size": 256,
                "block_size": args.block_size,
                "dim": args.dim,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "dropout": args.dropout,
            },
            "tokenizer": "utf8_bytes_v1",
            "epoch": epoch,
            "loss": avg,
        }
        path = os.path.join(args.out_dir, f"nanogpt_epoch{epoch}.pt")
        torch.save(ckpt, path)

        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, os.path.join(args.out_dir, "nanogpt_best.pt"))
            print(f"  → new best ({avg:.4f})")

    # final quantized model (CPU-friendly)
    model.cpu()
    save_quantized(model, os.path.join(args.out_dir, "nanogpt_int8.pt"))
    print(f"Done in {time.time()-t0:.0f}s. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
