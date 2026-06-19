import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Subset, random_split
from tqdm.auto import tqdm

from dataset import TacticsDataset
from model import GPT, GPTConfig
from utils import ensure_dirs, get_device, latest_checkpoint, save_checkpoint, set_seed
from wandb_utils import add_wandb_args, init_wandb, wandb_finish, wandb_log


def make_loaders(dataset, batch_size: int, val_fraction: float, num_workers: int, seed: int, device: str):
    val_size = max(1, int(len(dataset) * val_fraction)) if val_fraction > 0 and len(dataset) > 20 else 0
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    if val_size:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
        )
    return train_loader, val_loader, train_size, val_size


def learning_rate(step: int, base_lr: float, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int = 50) -> float | None:
    if loader is None:
        return None
    model.eval()
    losses = []
    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/tactics_corpus.txt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer/tokenizer.json")
    parser.add_argument("--out_dir", default="checkpoints/pretrain")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--ffn_hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means train for all epochs")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--no_resume", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dirs(args.out_dir)
    metrics_path = Path(args.out_dir) / "metrics.jsonl"
    device = get_device()
    use_amp = device == "cuda"
    print("Device:", device)
    print("AMP:", use_amp)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = GPTConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=args.context_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        ffn_hidden=args.ffn_hidden,
        dropout=args.dropout,
    )

    dataset = TacticsDataset(args.corpus, args.tokenizer, args.context_length, args.stride)
    train_loader, val_loader, train_size, val_size = make_loaders(
        dataset, args.batch_size, args.val_fraction, args.num_workers, args.seed, device
    )
    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum_steps))
    planned_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs

    print(f"Dataset training documents/sections: {dataset.num_documents:,}")
    print(f"Dataset tokens: {dataset.num_tokens:,}")
    print(f"Dataset windows: {len(dataset):,}")
    print(f"Train windows: {train_size:,}")
    print(f"Validation windows: {val_size:,}")
    print(f"Optimization steps planned: {planned_steps:,}")
    print(f"Effective batch size: {args.batch_size * args.grad_accum_steps:,}")

    wandb_run = init_wandb(
        args,
        "pretrain",
        {
            "stage": "pretrain",
            "args": vars(args),
            "model": asdict(config),
            "dataset_documents": dataset.num_documents,
            "dataset_tokens": dataset.num_tokens,
            "dataset_windows": len(dataset),
            "train_windows": train_size,
            "val_windows": val_size,
            "planned_steps": planned_steps,
        },
    )

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    ckpt = None if args.no_resume else latest_checkpoint(args.out_dir)
    if ckpt:
        print("Resuming from:", ckpt)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = int(state.get("step", 0))
        start_epoch = int(state.get("epoch", 0))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for micro_step, (x, y) in enumerate(progress, start=1):
            lr = learning_rate(global_step, args.lr, args.warmup_steps, planned_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(x, y)
                loss_to_backprop = loss / args.grad_accum_steps

            scaler.scale(loss_to_backprop).backward()

            if micro_step % args.grad_accum_steps != 0:
                progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step, lr=f"{lr:.2e}")
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", step=global_step, lr=f"{lr:.2e}")
            if args.log_every > 0 and global_step % args.log_every == 0:
                append_jsonl(metrics_path, {
                    "stage": "pretrain",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": float(loss.item()),
                    "lr": float(lr),
                })
                wandb_log(
                    wandb_run,
                    {
                        "pretrain/train_loss": float(loss.item()),
                        "pretrain/lr": float(lr),
                        "pretrain/epoch": epoch + 1,
                    },
                    step=global_step,
                )

            should_eval = val_loader is not None and args.eval_every > 0 and global_step % args.eval_every == 0
            if should_eval:
                val_loss = evaluate(model, val_loader, device, args.eval_batches)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"\\nValidation loss at step {global_step}: {val_loss:.4f}")
                append_jsonl(metrics_path, {
                    "stage": "pretrain",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "val_loss": float(val_loss),
                    "val_perplexity": float(val_ppl),
                    "best_val_loss": float(min(best_val_loss, val_loss)),
                })
                wandb_log(
                    wandb_run,
                    {
                        "pretrain/val_loss": float(val_loss),
                        "pretrain/val_perplexity": float(val_ppl),
                        "pretrain/best_val_loss": float(min(best_val_loss, val_loss)),
                    },
                    step=global_step,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = Path(args.out_dir) / "checkpoint_best.pt"
                    save_checkpoint(str(best_path), model, optimizer, global_step, epoch, args.tokenizer, asdict(config), best_val_loss)
                    print("Saved new best checkpoint:", best_path)

            if global_step % args.save_every == 0:
                path = Path(args.out_dir) / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(str(path), model, optimizer, global_step, epoch, args.tokenizer, asdict(config), best_val_loss)
                print("\\nSaved", path)

            if args.max_steps and global_step >= args.max_steps:
                if val_loader is not None and (args.eval_every <= 0 or global_step % args.eval_every != 0):
                    val_loss = evaluate(model, val_loader, device, args.eval_batches)
                    val_ppl = math.exp(min(val_loss, 20))
                    append_jsonl(metrics_path, {
                        "stage": "pretrain",
                        "step": global_step,
                        "epoch": epoch + 1,
                        "val_loss": float(val_loss),
                        "val_perplexity": float(val_ppl),
                        "best_val_loss": float(min(best_val_loss, val_loss)),
                        "final": True,
                    })
                    wandb_log(
                        wandb_run,
                        {
                            "pretrain/val_loss": float(val_loss),
                            "pretrain/val_perplexity": float(val_ppl),
                            "pretrain/best_val_loss": float(min(best_val_loss, val_loss)),
                        },
                        step=global_step,
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = Path(args.out_dir) / "checkpoint_best.pt"
                        save_checkpoint(str(best_path), model, optimizer, global_step, epoch, args.tokenizer, asdict(config), best_val_loss)
                        print("Saved new best checkpoint:", best_path)
                final_path = Path(args.out_dir) / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(str(final_path), model, optimizer, global_step, epoch, args.tokenizer, asdict(config), best_val_loss)
                print("\\nReached max_steps. Saved", final_path)
                wandb_finish(wandb_run)
                return

        # Save at the end of every epoch so progress survives Colab runtime resets.
        epoch_path = Path(args.out_dir) / f"checkpoint_step_{global_step}.pt"
        save_checkpoint(str(epoch_path), model, optimizer, global_step, epoch + 1, args.tokenizer, asdict(config), best_val_loss)
        print("\\nSaved end-of-epoch checkpoint:", epoch_path)

    if val_loader is not None and (args.eval_every <= 0 or global_step % args.eval_every != 0):
        val_loss = evaluate(model, val_loader, device, args.eval_batches)
        val_ppl = math.exp(min(val_loss, 20))
        append_jsonl(metrics_path, {
            "stage": "pretrain",
            "step": global_step,
            "epoch": args.epochs,
            "val_loss": float(val_loss),
            "val_perplexity": float(val_ppl),
            "best_val_loss": float(min(best_val_loss, val_loss)),
            "final": True,
        })
        wandb_log(
            wandb_run,
            {
                "pretrain/val_loss": float(val_loss),
                "pretrain/val_perplexity": float(val_ppl),
                "pretrain/best_val_loss": float(min(best_val_loss, val_loss)),
            },
            step=global_step,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(args.out_dir) / "checkpoint_best.pt"
            save_checkpoint(str(best_path), model, optimizer, global_step, args.epochs, args.tokenizer, asdict(config), best_val_loss)
            print("Saved new best checkpoint:", best_path)

    final_path = Path(args.out_dir) / f"checkpoint_step_{global_step}.pt"
    save_checkpoint(str(final_path), model, optimizer, global_step, args.epochs, args.tokenizer, asdict(config), best_val_loss)
    print("Training complete. Saved", final_path)
    if best_val_loss < float("inf"):
        print(f"Best validation loss: {best_val_loss:.4f}")
    wandb_finish(wandb_run)


if __name__ == "__main__":
    main()
