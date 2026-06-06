import glob
import os
import random
import re
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def checkpoint_step(path: str) -> int:
    match = re.search(r"checkpoint_step_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def latest_checkpoint(out_dir: str) -> str | None:
    checkpoints = glob.glob(os.path.join(out_dir, "checkpoint_step_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def save_checkpoint(
    path: str,
    model,
    optimizer,
    step: int,
    epoch: int,
    tokenizer_path: str,
    config: dict,
    best_val_loss: float | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "tokenizer_path": tokenizer_path,
        "config": config,
    }
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    torch.save(payload, path)
