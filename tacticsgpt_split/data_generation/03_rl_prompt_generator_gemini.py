# RL prompt generator. Set GOOGLE_API_KEY or paste it at runtime.

# # Generate RL Prompt JSONL With Gemini
# 
# Creates `data/rl_prompts.jsonl` containing prompts only. RL does not use target answers.

# Colab magic: !pip install -q -U google-genai

from google.colab import drive
from google import genai
from google.genai import types
from pathlib import Path
from getpass import getpass
import os, json, re, random, time

drive.mount("/content/drive", force_remount=True)
PROJECT_DIR = Path("/content/drive/MyDrive/TacticsGPT_Phase1_Full_Pretrain")
os.chdir(PROJECT_DIR)

api_key = os.environ.get("GOOGLE_API_KEY") or getpass("Paste Google AI Studio API key: ")
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

CORPUS_PATH = Path("data/tactics_corpus.txt")
OUT_PATH = Path("data/rl_prompts.jsonl")
text = CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")
docs = [x.strip() for x in re.split(r"\n{3,}", text) if len(x.strip()) > 500]

TARGET_PROMPTS = 500
PROMPTS_PER_CALL = 20
SLEEP_SECONDS = 6

existing = set()
if OUT_PATH.exists():
    for line in OUT_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            existing.add(json.loads(line)["prompt"].strip())
        except Exception:
            pass

def clean_json_array(raw):
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)

def valid_prompt(prompt):
    return isinstance(prompt, str) and 20 <= len(prompt.strip()) <= 260 and "```" not in prompt

def generate_rl_prompts(source_text, n=20):
    prompt = f'''
Create {n} high-quality RL training prompts for a football tactics assistant.

Return prompts/questions/tasks only. Do NOT include answers.
Return ONLY a valid JSON array of strings.

Source inspiration:
[BEGIN SOURCE]
{source_text[:5000]}
[END SOURCE]
'''
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.8,
            top_p=0.9,
            max_output_tokens=3000,
            response_mime_type="application/json",
        ),
    )
    return clean_json_array(response.text)

print("Existing prompts:", len(existing))
while len(existing) < TARGET_PROMPTS:
    try:
        prompts = generate_rl_prompts(random.choice(docs), PROMPTS_PER_CALL)
    except Exception as exc:
        print("API/parse error:", exc)
        time.sleep(30)
        continue

    added = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for prompt in prompts:
            prompt = prompt.strip()
            if not valid_prompt(prompt) or prompt in existing:
                continue
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            existing.add(prompt)
            added += 1
            if len(existing) >= TARGET_PROMPTS:
                break
    print(f"Added {added}. Total: {len(existing)} / {TARGET_PROMPTS}")
    time.sleep(SLEEP_SECONDS)

print("Saved:", OUT_PATH)

rows = [json.loads(line) for line in OUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
print("RL prompts:", len(rows))
for obj in random.sample(rows, min(10, len(rows))):
    print("-", obj["prompt"])
