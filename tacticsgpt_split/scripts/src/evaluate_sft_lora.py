import argparse, glob, os, re, json, math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from train_sft_lora import apply_lora, SFTDataset


def latest_step_checkpoint(folder, pattern):
    files = glob.glob(str(Path(folder) / pattern))
    if not files:
        return None

    def step(p):
        m = re.search(r"(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(files, key=step)


@torch.no_grad()
def eval_loss(model, dataset, batch_size, device, max_batches):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate)
    model.eval()

    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        losses.append(loss.item())

    avg = sum(losses) / max(1, len(losses))
    return avg, math.exp(min(avg, 20))


@torch.no_grad()
def generate_answer(model, tokenizer, instruction, device, max_new_tokens=160, temperature=0.75, top_k=40):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")

    if bos_id is not None:
        ids = [bos_id] + ids

    x = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        greedy=False,
    )

    text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)

    if "### Response:" in text:
        text = text.split("### Response:", 1)[1].strip()

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_data", default="data/sft_dataset.jsonl")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer/tokenizer.json")
    ap.add_argument("--sft_dir", default="checkpoints/sft")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=50)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sft_ckpt = args.checkpoint or latest_step_checkpoint(args.sft_dir, "sft_step_*.pt")
    if not sft_ckpt:
        raise FileNotFoundError("No SFT checkpoint found.")

    ckpt = torch.load(sft_ckpt, map_location=device)
    base_ckpt_path = ckpt["base_checkpoint"]

    print("SFT checkpoint:", sft_ckpt)
    print("Base checkpoint:", base_ckpt_path)

    base_state = torch.load(base_ckpt_path, map_location=device)
    config = GPTConfig(**base_state["config"])

    model = GPT(config).to(device)
    model.load_state_dict(base_state["model_state_dict"])

    model = apply_lora(model).to(device)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)

    tokenizer = Tokenizer.from_file(args.tokenizer)

    dataset = SFTDataset(args.sft_data, args.tokenizer, config.context_length)
    loss, ppl = eval_loss(model, dataset, args.batch_size, device, args.max_batches)

    print(f"\nSFT eval loss: {loss:.4f}")
    print(f"SFT perplexity: {ppl:.2f}")

    prompts = [
        "How should a 4-4-2 defend centrally against a 4-2-3-1?",
        "How can a team attack a compact 5-3-2 low block?",
        "What pressing cues should a 4-3-3 use against a back three?",
        "How should full-backs behave during defensive transitions?",
        "How should a team defend crosses in a low block?",
    ]

    for p in prompts:
        print("\n" + "=" * 80)
        print("INSTRUCTION:")
        print(p)
        print("\nMODEL RESPONSE:")
        print(generate_answer(model, tokenizer, p, device))


if __name__ == "__main__":
    main()
