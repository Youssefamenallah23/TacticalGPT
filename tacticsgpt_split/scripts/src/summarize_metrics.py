import argparse
import json
import math
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def last_with(rows, *keys):
    for row in reversed(rows):
        if all(key in row and row[key] is not None for key in keys):
            return row
    return None


def reward_improvement_pct(current_reward, baseline_reward):
    if baseline_reward is None or not math.isfinite(baseline_reward) or abs(baseline_reward) < 1e-8:
        return None
    return 100.0 * (current_reward - baseline_reward) / abs(baseline_reward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain_metrics", default="checkpoints/pretrain/metrics.jsonl")
    ap.add_argument("--sft_metrics", default="checkpoints/sft/metrics.jsonl")
    ap.add_argument("--rl_metrics", default="checkpoints/rl/metrics.jsonl")
    args = ap.parse_args()

    pretrain_rows = read_jsonl(args.pretrain_metrics)
    sft_rows = read_jsonl(args.sft_metrics)
    rl_rows = read_jsonl(args.rl_metrics)

    pretrain_eval = last_with(pretrain_rows, "val_loss", "val_perplexity")
    sft_eval = last_with(sft_rows, "eval_loss", "eval_perplexity")

    rl_eval_rows = [row for row in rl_rows if "eval_reward" in row and row["eval_reward"] is not None]
    rl_best = max(rl_eval_rows, key=lambda row: row["eval_reward"]) if rl_eval_rows else None
    rl_first = rl_eval_rows[0] if rl_eval_rows else None

    print("TacticsGPT metrics summary")
    print("=" * 28)

    if pretrain_eval:
        print(f"Pretrain final val loss: {pretrain_eval['val_loss']:.4f}")
        print(f"Pretrain final perplexity: {pretrain_eval['val_perplexity']:.2f}")
    else:
        print("Pretrain eval metrics: not found")

    if sft_eval:
        print(f"SFT final eval loss: {sft_eval['eval_loss']:.4f}")
        print(f"SFT final perplexity: {sft_eval['eval_perplexity']:.2f}")
    else:
        print("SFT eval metrics: not found")

    improvement = None
    if rl_best and rl_first:
        improvement = rl_best.get("reward_improvement_pct")
        if improvement is None:
            improvement = reward_improvement_pct(float(rl_best["eval_reward"]), float(rl_first["eval_reward"]))
        print(f"GRPO first eval reward: {float(rl_first['eval_reward']):.4f}")
        print(f"GRPO best eval reward: {float(rl_best['eval_reward']):.4f}")
        if improvement is not None:
            print(f"GRPO reward improvement: {improvement:.2f}%")
        else:
            print("GRPO reward improvement: unavailable because the first eval reward is zero")
    else:
        print("GRPO reward metrics: not found")

    if sft_eval and improvement is not None:
        print()
        print("Resume bullet:")
        print(
            "- Built TacticsGPT, a domain-specific football tactics LLM, with integrated W&B "
            "experiment tracking across pretraining, SFT, and GRPO; logged training loss, "
            f"perplexity, and reward curves, reaching {sft_eval['eval_loss']:.4f} eval loss, "
            f"{sft_eval['eval_perplexity']:.2f} perplexity, and {improvement:.1f}% GRPO reward improvement."
        )


if __name__ == "__main__":
    main()
