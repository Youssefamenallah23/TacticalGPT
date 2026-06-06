# TacticsGPT Phase 1: Pretraining

This project trains a small decoder-only GPT model on football tactical text using next-token prediction.

## Data

The provided Gemini article dump format is supported. Put the raw file at:

```text
data/raw_tactical_match_analysis.txt
```

Then build the cleaned corpus:

```bash
python src/build_corpus.py --raw data/raw_tactical_match_analysis.txt --out data/tactics_corpus.txt
```

This removes `ARTICLE #n` wrappers and separator bars while keeping match analysis, coach speeches, explanations, and tactical notes as training documents.

The cleaned text is written at:

```text
data/tactics_corpus.txt
```

Do not use JSON, CSV, labels, or question-answer pairs for Phase 1.

## Tokenizer

```bash
python src/tokenizer_train.py --corpus data/tactics_corpus.txt --out_dir checkpoints/tokenizer --vocab_size 8000
```

Outputs:

- `checkpoints/tokenizer/tokenizer.json`
- `checkpoints/tokenizer/vocab.json`
- `checkpoints/tokenizer/merges.txt`

## Pretraining

```bash
python src/train_pretrain.py --corpus data/tactics_corpus.txt --tokenizer checkpoints/tokenizer/tokenizer.json --epochs 30 --grad_accum_steps 2
```

The trainer automatically resumes from the latest `checkpoints/pretrain/checkpoint_step_*.pt` unless `--no_resume` is passed.

## Generation

```bash
python src/generate.py --prompt "How should a 4-4-2 defend centrally?"
```

This phase only trains tactical language modeling. It is not a RAG assistant and does not aim for factual perfection yet.
