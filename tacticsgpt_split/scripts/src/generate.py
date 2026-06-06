import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig
from utils import get_device, latest_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/pretrain/checkpoint_best.pt")
    parser.add_argument("--checkpoint_dir", default="checkpoints/pretrain")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer/tokenizer.json")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if not checkpoint or not Path(checkpoint).exists():
        checkpoint = latest_checkpoint(args.checkpoint_dir)
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint found in {args.checkpoint_dir}")

    device = get_device()
    state = torch.load(checkpoint, map_location=device)
    config = GPTConfig(**state["config"])

    tokenizer_path = args.tokenizer or state.get("tokenizer_path")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    model = GPT(config).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    input_ids = tokenizer.encode(args.prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    output = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )
    text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

    print("Checkpoint:", checkpoint)
    if "best_val_loss" in state:
        print("Best validation loss:", state["best_val_loss"])
    print("\nGenerated text:\n")
    print(text)


if __name__ == "__main__":
    main()
