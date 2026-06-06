import argparse, json, glob, os, re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from model import GPT, GPTConfig


def latest_step_checkpoint(folder, pattern):
    files = glob.glob(str(Path(folder) / pattern))
    if not files:
        return None
    def step(p):
        m = re.search(r"(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return max(files, key=step)


def find_base_checkpoint():
    best = Path("checkpoints/pretrain/checkpoint_best.pt")
    if best.exists():
        return str(best)
    ckpt = latest_step_checkpoint("checkpoints/pretrain", "checkpoint_step_*.pt")
    if ckpt:
        return ckpt
    raise FileNotFoundError("No Phase 1 checkpoint found in checkpoints/pretrain")


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora(model, r=8, alpha=16, dropout=0.05):
    for block in model.blocks:
        block.attn.qkv = LoRALinear(block.attn.qkv, r, alpha, dropout)
        block.attn.out = LoRALinear(block.attn.out, r, alpha, dropout)
        block.ffn.net[0] = LoRALinear(block.ffn.net[0], r, alpha, dropout)
        block.ffn.net[2] = LoRALinear(block.ffn.net[2], r, alpha, dropout)

    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name

    return model


def lora_state_dict(model):
    return {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer_path, context_length=256):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.context_length = context_length
        self.pad_id = self.tokenizer.token_to_id("<pad>") or 0
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                instruction = obj["instruction"].strip()
                response = obj["response"].strip()

                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
                prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
                response_ids = self.tokenizer.encode(response, add_special_tokens=False).ids

                ids = [self.bos_id] + prompt_ids + response_ids + [self.eos_id]
                labels = [-100] * (1 + len(prompt_ids)) + response_ids + [self.eos_id]

                ids = ids[:context_length]
                labels = labels[:context_length]

                # Causal LM shift:
                # x sees tokens up to t, y asks for token t+1.
                x_ids = ids[:-1]
                y_ids = labels[1:]

                if any(x != -100 for x in y_ids):
                    self.samples.append((x_ids, y_ids))

        if not self.samples:
            raise ValueError("No usable SFT samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate(self, batch):
        max_len = max(len(x[0]) for x in batch)
        xs, ys = [], []
        for ids, labels in batch:
            pad = max_len - len(ids)
            xs.append(ids + [self.pad_id] * pad)
            ys.append(labels + [-100] * pad)
        return torch.tensor(xs), torch.tensor(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_data", default="data/sft_dataset.jsonl")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer/tokenizer.json")
    ap.add_argument("--base_checkpoint", default="")
    ap.add_argument("--out_dir", default="checkpoints/sft")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_ckpt = args.base_checkpoint or find_base_checkpoint()
    print("Base checkpoint:", base_ckpt)

    state = torch.load(base_ckpt, map_location=device)
    config = GPTConfig(**state["config"])

    model = GPT(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model = apply_lora(model, args.r, args.alpha, args.dropout).to(device)

    dataset = SFTDataset(args.sft_data, args.tokenizer, args.context_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step = 0
    start_epoch = 0
    resume = latest_step_checkpoint(args.out_dir, "sft_step_*.pt")
    if resume:
        print("Resuming SFT:", resume)
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt["step"]
        start_epoch = ckpt["epoch"]

    model.train()
    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(loader, desc=f"SFT epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", step=step)

            if step % args.save_every == 0:
                path = Path(args.out_dir) / f"sft_step_{step}.pt"
                torch.save({
                    "lora_state_dict": lora_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "base_checkpoint": base_ckpt,
                    "config": state["config"],
                }, path)
                print("Saved", path)

    final = Path(args.out_dir) / f"sft_step_{step}.pt"
    torch.save({
        "lora_state_dict": lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": args.epochs,
        "base_checkpoint": base_ckpt,
        "config": state["config"],
    }, final)
    print("Saved final SFT:", final)


if __name__ == "__main__":
    main()
