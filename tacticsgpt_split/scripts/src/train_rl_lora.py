import argparse
import glob
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from model import GPT, GPTConfig
from train_sft_lora import apply_lora, lora_state_dict


ALLOWED_FORMATIONS = {
    "5-4-1",
    "5-2-1-2",
    "5-2-2-1",
    "5-2-3",
    "5-3-2",
    "3-1-4-2",
    "3-2-3-2",
    "3-2-4-1",
    "3-4-1-2",
    "3-4-2-1",
    "3-4-3",
    "3-5-1-1",
    "3-5-2",
    "4-1-2-1-2",
    "4-1-2-3",
    "4-1-3-2",
    "4-1-4-1",
    "4-2-1-3",
    "4-2-2-2",
    "4-2-3-1",
    "4-2-4",
    "4-3-1-2",
    "4-3-2-1",
    "4-3-3",
    "4-4-1-1",
    "4-4-2",
    "4-5-1",
}

DEFAULT_TACTICAL_TERMS = [
    "compact", "press", "pressing", "trigger", "block", "low block", "mid block",
    "high press", "transition", "counter", "half-space", "wide", "central",
    "overload", "underload", "pivot", "full-back", "wing-back", "centre-back",
    "center-back", "back line", "midfield", "cover shadow", "passing lane",
    "second ball", "rest defence", "rest defense", "numerical superiority",
    "depth", "width", "line", "back four", "back three", "third man",
]

TACTICAL_TERMS = DEFAULT_TACTICAL_TERMS

DEFAULT_ROLE_TERMS = [
    "full-back", "wing-back", "winger", "pivot", "centre-back", "center-back",
    "midfielder", "striker", "number 10", "holding midfielder", "back line",
    "front line", "wide player", "defensive midfielder",
]

ROLE_TERMS = DEFAULT_ROLE_TERMS

DEFAULT_ACTION_VERBS = [
    "screen", "delay", "press", "drop", "shift", "cover", "track", "force",
    "block", "protect", "hold", "mark", "trap", "recover", "switch", "stretch",
    "jump", "support", "occupy", "scan", "deny", "step", "pin", "rotate",
]

ACTION_VERBS = DEFAULT_ACTION_VERBS

PROMPT_GROUPS = {
    "press": ["press", "pressing", "trigger", "cover shadow", "jump", "force wide", "trap"],
    "transition": ["transition", "counter", "rest defence", "rest defense", "recover", "delay", "track runners"],
    "low_block": ["low block", "compact", "cross", "second ball", "box", "wide", "switch"],
    "build_up": ["build", "pivot", "centre-back", "center-back", "third man", "passing lane"],
    "cross": ["cross", "box", "near post", "far post", "second ball", "full-back", "winger"],
    "midfield": ["midfield", "pivot", "screen", "number 10", "central", "passing lane"],
}

STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "because", "before",
    "being", "between", "could", "does", "doing", "down", "during", "each",
    "from", "have", "into", "more", "must", "need", "only", "other", "over",
    "same", "should", "than", "that", "their", "then", "there", "these",
    "they", "this", "those", "through", "under", "using", "very", "when",
    "where", "which", "while", "with", "would", "your", "team", "player",
    "players", "opponent", "opponents", "ball", "space", "spaces",
}

FORMATION_ANY_RE = re.compile(r"\b(?:[1-5]-){1,5}[1-5]\b")
BAD_REPETITION_RE = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)
SYMBOL_SPAM_RE = re.compile(r"([.*_\-=])\1{6,}")
SPECIAL_TOKEN_RE = re.compile(r"(###|<bos>|<eos>|<pad>|instruction:|response:)", re.IGNORECASE)
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")

GENERIC_BAD_PHRASES = [
    "you must be fluid",
    "specific zones of the pitch",
    "act as the pivot point of the ball carrier",
    "the ball is central",
    "to prevent the opponent from behind",
    "maintain the anchor",
    "players should work together",
    "keep good shape",
]


def normalize_prompt(text):
    return " ".join(text.lower().strip().split())


def words(text):
    return WORD_RE.findall(text.lower())


def content_words(text):
    return [w for w in words(text) if len(w) >= 4 and w not in STOPWORDS]


def sentence_count(text):
    return len([s for s in re.split(r"[.!?]+", text) if len(s.strip()) > 12])


def unique_word_ratio(text):
    w = words(text)
    return len(set(w)) / max(1, len(w))


def tactical_terms():
    return globals().get("TACTICAL_TERMS", DEFAULT_TACTICAL_TERMS)


def role_terms():
    return globals().get("ROLE_TERMS", DEFAULT_ROLE_TERMS)


def action_verbs():
    return globals().get("ACTION_VERBS", DEFAULT_ACTION_VERBS)


def extract_key_terms(text, max_terms=32):
    lower = text.lower()
    terms = set()

    for term in tactical_terms() + role_terms() + action_verbs():
        if term in lower:
            terms.add(term)

    counts = Counter(content_words(text))
    for word, _ in counts.most_common(max_terms):
        terms.add(word)

    cw = content_words(text)
    phrase_counts = Counter()
    for n in (2, 3):
        for i in range(0, max(0, len(cw) - n + 1)):
            phrase = " ".join(cw[i:i + n])
            if len(phrase) >= 9:
                phrase_counts[phrase] += 1

    for phrase, _ in phrase_counts.most_common(10):
        terms.add(phrase)

    return terms


def parse_formations(text):
    formations = []
    for match in FORMATION_ANY_RE.finditer(text):
        raw = match.group(0)
        parts = [int(x) for x in raw.split("-")]
        formations.append((raw, parts, sum(parts)))
    return formations


def formation_score(response):
    formations = parse_formations(response)
    if not formations:
        return 0.0, ["no_formation"]

    invalid = []
    valid = []
    for raw, parts, total in formations:
        if raw in ALLOWED_FORMATIONS and total == 10:
            valid.append(raw)
        else:
            invalid.append(raw)

    if invalid:
        return -1.8, [f"invalid_formation:{','.join(invalid[:3])}"]
    return 0.35, [f"valid_formation:{valid[0]}"]


def prompt_focus_groups(prompt):
    lower = prompt.lower()
    return [group for group, keys in PROMPT_GROUPS.items() if any(k in lower for k in keys)]


def prompt_relevance_score(prompt, response):
    lower = response.lower()
    prompt_terms = set(content_words(prompt))
    if not prompt_terms:
        prompt_overlap = 0.0
    else:
        prompt_overlap = len([w for w in prompt_terms if w in lower]) / max(2, min(6, len(prompt_terms)))
        prompt_overlap = min(1.0, prompt_overlap)

    groups = prompt_focus_groups(prompt)
    if not groups:
        return prompt_overlap

    group_scores = []
    for group in groups:
        keys = PROMPT_GROUPS[group]
        hits = sum(1 for key in keys if key in lower)
        group_scores.append(min(1.0, hits / 3))

    group_score = sum(group_scores) / max(1, len(group_scores))
    return 0.55 * group_score + 0.45 * prompt_overlap


def load_reference_index(path):
    ref_path = Path(path)
    if not ref_path.exists():
        print(f"Reference note: {ref_path} not found, SFT answer matching disabled.")
        return {}

    index = {}
    for line in ref_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        prompt = obj.get("instruction") or obj.get("prompt") or obj.get("question")
        answer = obj.get("response") or obj.get("answer") or obj.get("completion")
        if not prompt or not answer:
            continue

        key = normalize_prompt(prompt)
        index.setdefault(key, []).append({
            "answer": answer.strip(),
            "terms": extract_key_terms(answer),
        })

    print(f"Loaded reference answers: {len(index)} prompts from {ref_path}")
    return index


def reference_alignment_score(prompt, response, reference_index):
    refs = reference_index.get(normalize_prompt(prompt), [])
    if not refs:
        return 0.0, ["no_reference"]

    response_terms = extract_key_terms(response)
    if not response_terms:
        return 0.0, ["empty_response_terms"]

    best = 0.0
    best_shared = []
    for ref in refs:
        ref_terms = set(ref["terms"])
        if not ref_terms:
            continue
        shared = ref_terms & response_terms
        coverage = len(shared) / max(1, len(ref_terms))
        precision = len(shared) / max(1, len(response_terms))
        score = 0.75 * coverage + 0.25 * precision
        if score > best:
            best = score
            best_shared = sorted(shared)[:8]

    return min(1.0, best), [f"reference_overlap:{','.join(best_shared)}"] if best_shared else ["no_reference_overlap"]


def formatting_score(response):
    text = response.strip()
    lower = text.lower()
    score = 0.0
    reasons = []

    if not text:
        return -1.5, ["empty_format"]

    if SPECIAL_TOKEN_RE.search(text):
        score -= 1.0
        reasons.append("leaked_prompt_tokens")

    if text[-1] in ".!?":
        score += 0.15
        reasons.append("clean_ending")
    else:
        score -= 0.35
        reasons.append("unfinished_sentence")

    sentences = sentence_count(text)
    if 2 <= sentences <= 6:
        score += 0.25
        reasons.append("clear_sentence_count")
    elif sentences == 0:
        score -= 0.5
        reasons.append("no_clear_sentences")

    if re.search(r"(^|\n)\s*(-|\d+\.)\s+\w+", text):
        score += 0.15
        reasons.append("structured_list")

    if any(connector in lower for connector in ["because", "so that", "which forces", "when", "if", "therefore"]):
        score += 0.2
        reasons.append("clear_cause_effect")

    if re.search(r"([!?.,])\1{3,}", text):
        score -= 0.4
        reasons.append("punctuation_spam")

    return score, reasons


def tactical_logic_score(response):
    lower = response.lower()
    role_hit = any(role in lower for role in role_terms())
    action_hit = any(action in lower for action in action_verbs())
    outcome_hit = any(x in lower for x in [
        "so that", "because", "which forces", "to prevent", "to deny",
        "to protect", "when", "if", "therefore", "as a result",
    ])

    if role_hit and action_hit and outcome_hit:
        return 0.9, ["role_action_outcome"]
    if action_hit and outcome_hit:
        return 0.35, ["action_outcome"]
    return -0.55, ["weak_tactical_logic"]


def repetition_penalty(response):
    lower = response.lower()
    penalty = 0.0
    reasons = []

    if BAD_REPETITION_RE.search(lower):
        penalty -= 1.2
        reasons.append("word_loop")

    if SYMBOL_SPAM_RE.search(response):
        penalty -= 1.2
        reasons.append("symbol_spam")

    w = words(response)
    if len(w) > 20:
        trigrams = list(zip(w, w[1:], w[2:]))
        if trigrams:
            repeated_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
            if repeated_ratio > 0.18:
                penalty -= min(1.2, repeated_ratio * 3.0)
                reasons.append("repeated_trigrams")

        lower_counts = Counter(w)
        for word, count in lower_counts.items():
            if len(word) > 4 and count >= 7:
                penalty -= 0.2 * (count - 6)
                reasons.append(f"repeated_word:{word}")
                break

    return penalty, reasons


def generic_answer_penalty(response):
    lower = response.lower()
    penalty = 0.0
    reasons = []

    for phrase in GENERIC_BAD_PHRASES:
        if phrase in lower:
            penalty -= 0.6
            reasons.append(f"generic_phrase:{phrase[:20]}")

    if unique_word_ratio(response) < 0.42 and len(words(response)) > 45:
        penalty -= 0.7
        reasons.append("low_diversity")

    return penalty, reasons


def reward_response(prompt, response, reference_index=None):
    reference_index = reference_index or {}
    text = response.strip()
    w = words(text)
    n_words = len(w)
    reward = 0.0
    reasons = []

    if not text:
        return -4.0, ["empty_response"]

    rep_penalty, rep_reasons = repetition_penalty(text)
    if rep_penalty <= -2.0:
        return -4.0, ["collapse_repetition"] + rep_reasons

    if n_words < 35:
        reward -= 1.2
        reasons.append("too_short")
    elif 55 <= n_words <= 200:
        reward += 0.45
        reasons.append("good_length")
    elif 35 <= n_words < 55 or 200 < n_words <= 280:
        reward += 0.05
        reasons.append("acceptable_length")
    else:
        reward -= 0.6
        reasons.append("bad_length")

    rel = prompt_relevance_score(prompt, text)
    if rel >= 0.62:
        reward += 1.1
        reasons.append("prompt_relevant")
    elif rel >= 0.35:
        reward += 0.35
        reasons.append("partly_relevant")
    else:
        reward -= 1.1
        reasons.append("off_prompt")

    ref_score, ref_reasons = reference_alignment_score(prompt, text, reference_index)
    if reference_index:
        if ref_score >= 0.42:
            reward += 1.25
            reasons.append("strong_reference_alignment")
        elif ref_score >= 0.22:
            reward += 0.5
            reasons.append("some_reference_alignment")
        elif ref_score > 0.0:
            reward += 0.1
            reasons.append("thin_reference_alignment")
        else:
            reward -= 0.7
            reasons.append("missing_reference_concepts")
        reasons.extend(ref_reasons)

    f_score, f_reasons = formation_score(text)
    reward += f_score
    reasons.extend(f_reasons)

    fmt_score, fmt_reasons = formatting_score(text)
    reward += fmt_score
    reasons.extend(fmt_reasons)

    logic_score, logic_reasons = tactical_logic_score(text)
    reward += logic_score
    reasons.extend(logic_reasons)

    term_hits = [term for term in tactical_terms() if term in text.lower()]
    if 3 <= len(term_hits) <= 10:
        reward += 0.35
        reasons.append("balanced_tactical_terms")
    elif len(term_hits) > 13:
        reward -= 0.65
        reasons.append("keyword_stuffing")
    elif len(term_hits) < 2:
        reward -= 0.25
        reasons.append("thin_tactical_terms")

    reward += rep_penalty
    reasons.extend(rep_reasons)

    generic_penalty, generic_reasons = generic_answer_penalty(text)
    reward += generic_penalty
    reasons.extend(generic_reasons)

    return max(-4.0, min(5.0, reward)), reasons


class RLPromptDataset(Dataset):
    def __init__(self, rl_path, fallback_sft_path=None):
        self.prompts = []
        path = Path(rl_path)

        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt") or obj.get("instruction") or obj.get("question")
                if prompt:
                    self.prompts.append(prompt.strip())

        if not self.prompts and fallback_sft_path and Path(fallback_sft_path).exists():
            for line in Path(fallback_sft_path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj.get("instruction") or obj.get("prompt") or obj.get("question")
                if prompt:
                    self.prompts.append(prompt.strip())

        self.prompts = sorted(set(self.prompts))
        if not self.prompts:
            raise ValueError(f"No RL prompts found in {rl_path} or fallback SFT data.")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def latest_step_checkpoint(folder, pattern):
    files = glob.glob(str(Path(folder) / pattern))
    if not files:
        return None

    def step(path):
        match = re.search(r"(\d+)\.pt$", os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(files, key=step)


def find_base_checkpoint():
    best = Path("checkpoints/pretrain/checkpoint_best.pt")
    if best.exists():
        return str(best)
    ckpt = latest_step_checkpoint("checkpoints/pretrain", "checkpoint_step_*.pt")
    if ckpt:
        return ckpt
    raise FileNotFoundError("No Phase 1 checkpoint found.")


def find_sft_checkpoint():
    ckpt = latest_step_checkpoint("checkpoints/sft", "sft_step_*.pt")
    if ckpt:
        return ckpt
    raise FileNotFoundError("No SFT checkpoint found. Train SFT before RL.")


def sample_with_logprobs(model, tokenizer, instruction, device, max_new_tokens=120, temperature=0.8, top_k=40):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if bos_id is not None:
        ids = [bos_id] + ids

    x = torch.tensor([ids], dtype=torch.long, device=device)
    log_probs = []
    generated = []

    model.train()
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.context_length:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        next_id = dist.sample()
        log_probs.append(dist.log_prob(next_id))

        token_id = int(next_id.item())
        generated.append(token_id)
        x = torch.cat([x, next_id.view(1, 1)], dim=1)

        if eos_id is not None and token_id == eos_id:
            break

    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    if not log_probs:
        return response, torch.tensor(0.0, device=device), 1

    return response, torch.stack(log_probs).sum(), max(1, len(generated))


@torch.no_grad()
def generate_response(model, tokenizer, instruction, device, max_new_tokens=140, temperature=0.7, top_k=40):
    was_training = model.training
    model.eval()

    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    if bos_id is not None:
        ids = [bos_id] + ids

    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.context_length:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k and top_k > 0:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_id = int(next_id.item())
        generated.append(token_id)
        x = torch.cat([x, next_id.view(1, 1)], dim=1)

        if eos_id is not None and token_id == eos_id:
            break

    if was_training:
        model.train()

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_policy(model, tokenizer, prompts, reference_index, device, args):
    sample_size = min(args.eval_samples, len(prompts))
    eval_prompts = random.sample(prompts, sample_size) if sample_size < len(prompts) else list(prompts)

    rewards = []
    reason_counts = Counter()
    examples = []

    for prompt in eval_prompts:
        response = generate_response(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.eval_temperature,
            top_k=args.eval_top_k,
        )
        reward, reasons = reward_response(prompt, response, reference_index)
        rewards.append(float(reward))
        reason_counts.update(reasons)
        if len(examples) < 3:
            examples.append({
                "prompt": prompt,
                "response": response[:600],
                "reward": float(reward),
                "reasons": reasons[:12],
            })

    avg_reward = sum(rewards) / max(1, len(rewards))
    return {
        "avg_reward": avg_reward,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "samples": len(rewards),
        "top_reasons": reason_counts.most_common(12),
        "examples": examples,
    }


def checkpoint_payload(model, optimizer, step, base_path, sft_path, base_config, best_eval_reward,
                       reward_history, eval_history, top_k, args):
    return {
        "lora_state_dict": lora_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_eval_reward": best_eval_reward,
        "reward_history": reward_history[-1000:],
        "eval_history": eval_history[-200:],
        "top_k": top_k,
        "base_checkpoint": base_path,
        "sft_checkpoint": sft_path,
        "config": base_config,
        "args": vars(args),
    }


def save_training_checkpoint(path, model, optimizer, step, base_path, sft_path, base_config,
                             best_eval_reward, reward_history, eval_history, top_k, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            model, optimizer, step, base_path, sft_path, base_config,
            best_eval_reward, reward_history, eval_history, top_k, args,
        ),
        path,
    )
    return str(path)


def update_top_k(top_k, score, path, k):
    if k <= 0:
        return []

    path = str(path)
    top_k = [item for item in top_k if item.get("path") != path]
    top_k.append({"score": float(score), "path": path})
    top_k = sorted(top_k, key=lambda item: item["score"], reverse=True)

    removed = top_k[k:]
    kept = top_k[:k]

    for item in removed:
        old_path = Path(item["path"])
        if old_path.exists() and old_path.name != "checkpoint_best.pt":
            try:
                old_path.unlink()
            except OSError:
                pass

    return kept


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl_data", default="data/rl_prompts.jsonl")
    ap.add_argument("--sft_data", default="data/sft_dataset.jsonl")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer/tokenizer.json")
    ap.add_argument("--base_checkpoint", default="")
    ap.add_argument("--sft_checkpoint", default="")
    ap.add_argument("--resume_checkpoint", default="")
    ap.add_argument("--out_dir", default="checkpoints/rl")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--group_size", type=int, default=2)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_samples", type=int, default=16)
    ap.add_argument("--top_k_best", type=int, default=1)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--eval_temperature", type=float, default=0.65)
    ap.add_argument("--eval_top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_path = args.base_checkpoint or find_base_checkpoint()
    sft_path = args.sft_checkpoint or find_sft_checkpoint()

    print("Device:", device)
    print("Base:", base_path)
    print("SFT:", sft_path)
    print("Reward mode: GRPO-style group-relative advantage, SFT-reference keyword matching, strict optional formations.")

    base_state = torch.load(base_path, map_location=device)
    sft_state = torch.load(sft_path, map_location=device)

    config = GPTConfig(**base_state["config"])
    model = GPT(config).to(device)
    model.load_state_dict(base_state["model_state_dict"])

    model = apply_lora(model).to(device)
    model.load_state_dict(sft_state["lora_state_dict"], strict=False)

    for _, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    dataset = RLPromptDataset(args.rl_data, fallback_sft_path=args.sft_data)
    reference_index = load_reference_index(args.sft_data)

    step = 0
    baseline = 0.0
    best_eval_reward = -float("inf")
    reward_history = []
    eval_history = []
    top_k = []

    resume = args.resume_checkpoint or latest_step_checkpoint(args.out_dir, "rl_step_*.pt")
    if resume:
        print("Resuming RL:", resume)
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt.get("step", 0)
        baseline = ckpt.get("baseline", 0.0)
        best_eval_reward = ckpt.get("best_eval_reward", ckpt.get("best_avg_reward", best_eval_reward))
        reward_history = ckpt.get("reward_history", [])
        eval_history = ckpt.get("eval_history", [])
        top_k = ckpt.get("top_k", [])
        print("Resume state:", {"step": step, "best_eval_reward": best_eval_reward, "top_k": top_k})

    pbar = tqdm(total=args.steps, initial=step, desc="RL GRPO-style")

    while step < args.steps:
        batch_losses = []
        batch_rewards = []
        batch_records = []

        for _ in range(args.batch_size):
            prompt = random.choice(dataset.prompts)
            group = []

            for _ in range(max(1, args.group_size)):
                response, log_prob_sum, token_count = sample_with_logprobs(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )
                reward, reasons = reward_response(prompt, response, reference_index)
                group.append({
                    "prompt": prompt,
                    "response": response,
                    "log_prob_sum": log_prob_sum,
                    "token_count": token_count,
                    "reward": float(reward),
                    "reasons": reasons,
                })

            rewards_tensor = torch.tensor([item["reward"] for item in group], dtype=torch.float32, device=device)
            if len(group) > 1:
                mean = rewards_tensor.mean()
                std = rewards_tensor.std(unbiased=False)
                advantages = (rewards_tensor - mean) / (std + 1e-6)
            else:
                reward_value = float(rewards_tensor.item())
                baseline = 0.95 * baseline + 0.05 * reward_value
                advantages = torch.tensor([reward_value - baseline], dtype=torch.float32, device=device)

            for item, advantage in zip(group, advantages):
                normalized_logprob = item["log_prob_sum"] / max(1, item["token_count"])
                batch_losses.append(-normalized_logprob * advantage.detach())
                batch_rewards.append(item["reward"])
                batch_records.append(item)

        loss = torch.stack(batch_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        step += 1
        current_reward = float(batch_rewards[-1])
        avg_train_reward = sum(batch_rewards) / max(1, len(batch_rewards))
        reward_history.append(avg_train_reward)
        reward_history = reward_history[-1000:]
        recent_avg = sum(reward_history[-25:]) / max(1, len(reward_history[-25:]))

        checkpoint_path = ""
        best_checkpoint_path = ""
        eval_result = None

        if step % args.save_every == 0:
            checkpoint_path = save_training_checkpoint(
                out_dir / f"rl_step_{step}.pt",
                model,
                optimizer,
                step,
                base_path,
                sft_path,
                base_state["config"],
                best_eval_reward,
                reward_history,
                eval_history,
                top_k,
                args,
            )
            print(f"\nSaved recovery checkpoint: {checkpoint_path}")

        if args.eval_every > 0 and step % args.eval_every == 0:
            eval_result = evaluate_policy(model, tokenizer, dataset.prompts, reference_index, device, args)
            eval_reward = float(eval_result["avg_reward"])
            eval_history.append({"step": step, **eval_result})

            if eval_reward > best_eval_reward + args.min_delta:
                best_eval_reward = eval_reward
                best_checkpoint_path = save_training_checkpoint(
                    out_dir / "best_model" / "checkpoint_best.pt",
                    model,
                    optimizer,
                    step,
                    base_path,
                    sft_path,
                    base_state["config"],
                    best_eval_reward,
                    reward_history,
                    eval_history,
                    top_k,
                    args,
                )

                if args.top_k_best > 1:
                    top_k_path = out_dir / "best_model" / "top_k" / f"best_step_{step}_reward_{eval_reward:.4f}.pt"
                    saved_top_k_path = save_training_checkpoint(
                        top_k_path,
                        model,
                        optimizer,
                        step,
                        base_path,
                        sft_path,
                        base_state["config"],
                        best_eval_reward,
                        reward_history,
                        eval_history,
                        top_k,
                        args,
                    )
                    top_k = update_top_k(top_k, eval_reward, saved_top_k_path, args.top_k_best)

                print(f"\nSaved new best model: {best_checkpoint_path} eval_reward={eval_reward:.4f}")

        if step % args.log_every == 0 or checkpoint_path or best_checkpoint_path:
            record = {
                "step": step,
                "current_reward": current_reward,
                "avg_train_reward": avg_train_reward,
                "recent_avg_reward_25": recent_avg,
                "best_eval_reward": best_eval_reward,
                "loss": float(loss.item()),
                "checkpoint_path": checkpoint_path,
                "best_checkpoint_path": best_checkpoint_path,
                "sample_prompt": batch_records[-1]["prompt"] if batch_records else "",
                "sample_response": batch_records[-1]["response"][:500] if batch_records else "",
                "sample_reasons": batch_records[-1]["reasons"][:12] if batch_records else [],
            }
            if eval_result:
                record["eval_reward"] = eval_result["avg_reward"]
                record["eval_top_reasons"] = eval_result["top_reasons"]
            append_jsonl(metrics_path, record)

            print(
                "\nLOG",
                json.dumps({
                    "step": step,
                    "current_reward": round(current_reward, 3),
                    "avg_train_reward": round(avg_train_reward, 3),
                    "recent_avg25": round(recent_avg, 3),
                    "best_eval_reward": round(best_eval_reward, 3) if math.isfinite(best_eval_reward) else None,
                    "checkpoint_path": checkpoint_path,
                    "best_checkpoint_path": best_checkpoint_path,
                }),
            )

        pbar.update(1)
        pbar.set_postfix(
            reward=f"{current_reward:.2f}",
            avg=f"{avg_train_reward:.2f}",
            avg25=f"{recent_avg:.2f}",
            best=f"{best_eval_reward:.2f}" if math.isfinite(best_eval_reward) else "none",
            loss=f"{loss.item():.4f}",
        )

    final_path = save_training_checkpoint(
        out_dir / f"rl_step_{step}.pt",
        model,
        optimizer,
        step,
        base_path,
        sft_path,
        base_state["config"],
        best_eval_reward,
        reward_history,
        eval_history,
        top_k,
        args,
    )
    print("Saved final RL:", final_path)
    print("Metrics:", metrics_path)
    if math.isfinite(best_eval_reward):
        print("Best eval reward:", best_eval_reward)
        print("Best model:", out_dir / "best_model" / "checkpoint_best.pt")


if __name__ == "__main__":
    main()
