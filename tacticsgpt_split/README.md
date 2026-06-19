# TacticsGPT Split Colab Pipeline

This folder splits the original all-in-one notebook into three runnable Colab notebooks:

1. `notebooks/01_pretrain_and_model.ipynb`
   - cleans the pretraining corpus
   - writes the tokenizer/model/pretraining scripts
   - trains or resumes the base GPT
   - tests base generation

2. `notebooks/02_sft_lora_and_testing.ipynb`
   - uploads or uses `data/sft_dataset.jsonl`
   - writes the LoRA SFT trainer
   - trains/resumes SFT
   - evaluates and tests the SFT adapter

3. `notebooks/03_rl_lora_and_testing.ipynb`
   - builds `data/rl_prompts.jsonl`
   - writes the RL trainer
   - trains/resumes RL
   - includes fixed-prompt and interactive testing cells

The `scripts/` folder contains the `.py` files extracted from the notebooks.

Optional experiment tracking is available in all three training stages. Install W&B with:

```bash
pip install wandb
```

Then add `--wandb_project TacticsGPT` plus a stage-specific `--wandb_run_name` to the pretrain, SFT, and RL commands. The scripts log loss/LR curves, SFT/pretrain perplexity, and GRPO reward curves. They also write local metrics JSONL files under `checkpoints/<stage>/metrics.jsonl`.

After training, summarize the resume numbers with:

```bash
python src/summarize_metrics.py
```

The `data_generation/` folder contains sanitized data-generation notebooks/scripts:

- `01_pretrain_data_generator_nvidia_sanitized.*`
- `02_sft_dataset_generator_gemini.*`
- `03_rl_prompt_generator_gemini.*`

No API keys are stored in these files. Use environment variables or paste keys at runtime.

Recommended Colab order:

- For a new project: run notebook 01, then generate SFT data, then notebook 02, then generate RL prompts or derive them from SFT, then notebook 03.
- After closing Colab: install dependencies, mount Drive, run the relevant `%%writefile` source cells, then rerun the train cell. Training scripts resume from Drive checkpoints.
