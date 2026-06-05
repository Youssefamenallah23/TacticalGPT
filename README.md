# TacticsGPT

A football tactics language model trained from scratch — decoder-only GPT with supervised fine-tuning via LoRA and reinforcement learning with group-relative policy optimization (GRPO-style rewards).

Built entirely in PyTorch without HuggingFace model wrappers. Every component — tokenizer, architecture, training loop, RL reward pipeline — is implemented from the ground up.

---

## What's inside

```
TacticsGPT/
├── src/
│   ├── model.py               # Decoder-only GPT (causal attention, pre-LayerNorm, weight tying)
│   ├── dataset.py             # Article-aware sliding-window dataset
│   ├── train_pretrain.py      # Base pretraining loop (AMP, grad accum, cosine LR, resume)
│   ├── train_sft_lora.py      # LoRA supervised fine-tuning on instruction-response pairs
│   ├── train_rl_lora.py       # GRPO-style RL with multi-signal tactical reward function
│   ├── evaluate_sft_lora.py   # SFT evaluation: loss, perplexity, generation samples
│   ├── generate.py            # Base model generation utility
│   ├── build_corpus.py        # Raw text cleaning and document extraction
│   ├── tokenizer_train.py     # BPE tokenizer training on the tactics corpus
│   └── utils.py               # Shared helpers: device, seed, checkpointing
├── data/
│   ├── tactics_corpus.txt     # Cleaned pretraining corpus
│   ├── sft_dataset.jsonl      # Instruction-response pairs for SFT
│   └── rl_prompts.jsonl       # Prompts for RL rollouts
├── checkpoints/
│   ├── tokenizer/             # tokenizer.json, vocab.json, merges.txt
│   ├── pretrain/              # Pretrain step checkpoints + best
│   ├── sft/                   # SFT LoRA adapter checkpoints
│   └── rl/                    # RL LoRA adapter checkpoints + best model
├── TacticsGPT_Phase_1_Colab.ipynb
└── README_01_PRETRAIN.md
```

---

## Architecture

| Component | Details |
|---|---|
| Model type | Decoder-only GPT |
| Layers | 4 transformer blocks |
| Attention heads | 4 |
| Model dimension | 256 |
| FFN hidden | 1024 |
| Context length | 256 tokens |
| Vocab size | 8,000 (domain BPE) |
| Positional encoding | Learned embeddings |
| Normalization | Pre-LayerNorm |
| Weight tying | Token embedding ↔ LM head |
| Parameters | ~7M |

The attention block uses a fused QKV projection with a registered causal mask buffer. Residual connections wrap both attention and FFN sub-layers.

---

## Training pipeline

### Stage 1 — Pretraining

Next-token prediction on cleaned football tactical text. Trained with:

- AdamW (β₁=0.9, β₂=0.95, weight decay=0.1)
- Cosine LR schedule with linear warmup
- Mixed precision (AMP) on CUDA
- Gradient accumulation for larger effective batch size
- Automatic checkpoint resuming

```bash
python src/train_pretrain.py \
  --corpus data/tactics_corpus.txt \
  --tokenizer checkpoints/tokenizer/tokenizer.json \
  --epochs 30 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --lr 3e-4 \
  --warmup_steps 500
```

### Stage 2 — Supervised Fine-tuning (LoRA)

LoRA adapters applied to all attention projections (QKV, output) and FFN layers. Base weights are frozen; only `lora_A` and `lora_B` parameters are trained. The SFT dataset uses an instruction-response format with prompt masking so loss is computed only on the response tokens.

```bash
python src/train_sft_lora.py \
  --sft_data data/sft_dataset.jsonl \
  --tokenizer checkpoints/tokenizer/tokenizer.json \
  --epochs 20 \
  --lr 1e-4
```

LoRA configuration: rank r=8, alpha=16, dropout=0.05.

### Stage 3 — Reinforcement Learning (GRPO-style)

The SFT-adapted model is further fine-tuned using a policy gradient approach inspired by Group Relative Policy Optimization. For each prompt, multiple responses are sampled, scored, and their relative advantage is used as the learning signal — no separate value network required.

```bash
python src/train_rl_lora.py \
  --rl_data data/rl_prompts.jsonl \
  --sft_data data/sft_dataset.jsonl \
  --tokenizer checkpoints/tokenizer/tokenizer.json \
  --steps 1000 \
  --group_size 2 \
  --lr 5e-5
```

---

## Reward function

The RL reward is a composite multi-signal score designed around tactical coherence. Each component is computed independently and summed:

| Signal | What it measures | Range |
|---|---|---|
| Formation validity | Detects and validates formation strings (e.g. 4-3-3) against a known-valid set | −1.8 to +0.35 |
| Prompt relevance | Term overlap between prompt focus group and response | 0 to +1.1 |
| Reference alignment | Keyword coverage vs SFT reference answers | 0 to +1.25 |
| Tactical logic | Role + action verb + outcome reasoning chain present | −0.55 to +0.9 |
| Formatting | Sentence completeness, structure, no leaked prompt tokens | −1.5 to +0.75 |
| Repetition penalty | Trigram repetition ratio, word loop, symbol spam | up to −4.0 |
| Generic phrase penalty | Penalizes low-information filler outputs | up to −1.2 |

The advantage for each sample in a group is normalized: `A = (R − mean(group)) / (std(group) + ε)`. When group size is 1, a moving exponential baseline is used instead.

---

## Data

The pretraining corpus is built from football tactical text sources: match analysis articles, coaching methodology pieces, and positional role descriptions. Raw text is cleaned through:

- Unicode normalization (NFKC)
- Mojibake detection and repair
- Document boundary parsing (`ARTICLE #N` delimiters)
- Paragraph chunking with configurable target and max character limits
- Minimum document length filtering

The BPE tokenizer is trained from scratch on the cleaned corpus with a vocabulary of 8,000 tokens and special tokens `<pad>`, `<bos>`, `<eos>`, `<unk>`.

To rebuild the corpus from a raw dump:

```bash
python src/build_corpus.py \
  --raw data/raw_tactical_match_analysis.txt \
  --out data/tactics_corpus.txt

python src/tokenizer_train.py \
  --corpus data/tactics_corpus.txt \
  --out_dir checkpoints/tokenizer \
  --vocab_size 8000
```

---

## Setup

```bash
pip install torch tokenizers tqdm numpy
```

For the Colab notebook, all dependencies are installed in the first cell. Checkpoints are saved to Google Drive so training survives runtime resets.

---

## Generation

```bash
# From the base pretrained model
python src/generate.py \
  --prompt "How should a 4-4-2 defend centrally?" \
  --temperature 0.8 \
  --top_k 50 \
  --max_new_tokens 120

# From the SFT or RL adapter (interactive)
# Run cell 12 in the Colab notebook
```

Sample output after RL fine-tuning:

```
Prompt: How should a 4-4-2 defend centrally against a 4-2-3-1?

The two banks of four must remain compact and narrow, denying the opponent's
number 10 any space between the lines. The defensive midfielder screens the
central channel while the two centre-backs hold their shape and delay the
striker. Full-backs tuck in rather than push wide, forcing play outside where
the winger can press with a cover shadow blocking the inside pass. The trigger
to press is a back pass to the goalkeeper or a slow lateral ball to a centre-back.
```

---

## Checkpointing and resuming

All three training stages resume automatically from the latest checkpoint in their respective output directories. The pretrain script tracks the best validation loss and saves a separate `checkpoint_best.pt`. The RL script tracks best evaluation reward and maintains a configurable top-k checkpoint pool with automatic cleanup of older files.

To force a fresh run at any stage, add `--no_resume` (pretrain) or delete the relevant checkpoint directory.

---

## What's next

- [ ] Port the core training step to JAX/Flax with `jit` and `optax`
- [ ] FAISS retrieval layer — embed the tactics corpus and retrieve top-k chunks at inference time
- [ ] Replace rule-based reward with a small learned reward model trained on preference pairs
- [ ] Scale up: 6 layers, d_model=384, context_length=512
- [ ] Replace REINFORCE/GRPO with PPO for more stable RL training

---

## Motivation

This project is a warmup toward building a full Transformer + Mixture-of-Experts architecture from scratch. The goal was to get fluent in every component of the modern LLM training stack — tokenization, architecture, SFT, and RL alignment — before scaling up to more complex designs.

---

## References

- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
- Shao et al., *DeepSeekMath* (GRPO)
- Schulman et al., *Proximal Policy Optimization Algorithms*
