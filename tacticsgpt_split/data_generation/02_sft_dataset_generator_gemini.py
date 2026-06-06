# SFT dataset generator. Set GOOGLE_API_KEY or paste it at runtime.

# # Generate SFT JSONL With Gemini
# 
# Creates `data/sft_dataset.jsonl` with `{instruction, response}` examples for Phase 2 SFT.
# 
# No API key is stored in this notebook. Paste it at runtime or set `GOOGLE_API_KEY`.

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

print("Current folder:", Path.cwd())
print("Corpus exists:", Path("data/tactics_corpus.txt").exists())

CORPUS_PATH = Path("data/tactics_corpus.txt")
OUT_PATH = Path("data/sft_dataset.jsonl")

if not CORPUS_PATH.exists():
    raise FileNotFoundError("Missing data/tactics_corpus.txt. Run Phase 1 corpus cleaning first.")

text = CORPUS_PATH.read_text(encoding="utf-8", errors="ignore")
docs = [x.strip() for x in re.split(r"\n{3,}", text) if len(x.strip()) > 500]

print("Documents/sections:", len(docs))
print("Characters:", len(text))
print(docs[0][:800])

TARGET_EXAMPLES = 1000
EXAMPLES_PER_CALL = 8
SLEEP_SECONDS = 6

instruction_types = [
    "answer a tactical question",
    "explain a formation adjustment",
    "coach a team through a defensive problem",
    "analyze a pressing structure",
    "explain how to attack a low block",
    "explain how to defend transitions",
    "give a concise match-analysis answer",
    "give a practical training-ground coaching answer",
]

def clean_json_array(raw):
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return json.loads(raw)

def load_existing(path):
    rows, seen = [], set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
                seen.add(obj["instruction"].strip().lower())
            except Exception:
                pass
    return rows, seen

def valid_example(obj):
    if not isinstance(obj, dict):
        return False
    inst = str(obj.get("instruction", "")).strip()
    resp = str(obj.get("response", "")).strip()
    if len(inst) < 20 or len(inst) > 260:
        return False
    if len(resp) < 80 or len(resp) > 1500:
        return False
    if "###" in inst or "###" in resp:
        return False
    return True

def generate_sft_batch(source_text, n=8):
    focus = random.choice(instruction_types)
    prompt = f'''
Generate {n} supervised fine-tuning examples for a football tactics assistant.

Use the source only as inspiration. Do not copy long passages.
Each answer should be clear, practical, tactically precise, and grounded in football coaching logic.

Focus: {focus}

Return ONLY a valid JSON array. Each object must have exactly:
- "instruction"
- "response"

Source material:
[BEGIN SOURCE]
{source_text[:6000]}
[END SOURCE]
'''
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.8,
            top_p=0.9,
            max_output_tokens=6000,
            response_mime_type="application/json",
        ),
    )
    return clean_json_array(response.text)

# Preview one batch before generating the full file.
test_batch = generate_sft_batch(random.choice(docs), n=3)
for i, obj in enumerate(test_batch, 1):
    print("\n" + "=" * 80)
    print("EXAMPLE", i)
    print("INSTRUCTION:", obj.get("instruction", ""))
    print("RESPONSE:", obj.get("response", ""))
    print("VALID:", valid_example(obj))

rows, seen = load_existing(OUT_PATH)
print("Existing examples:", len(rows))

while len(rows) < TARGET_EXAMPLES:
    try:
        batch = generate_sft_batch(random.choice(docs), EXAMPLES_PER_CALL)
    except Exception as exc:
        print("Generation/parsing error:", exc)
        time.sleep(30)
        continue

    added = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for obj in batch:
            if not valid_example(obj):
                continue
            inst = obj["instruction"].strip()
            resp = obj["response"].strip()
            key = inst.lower()
            if key in seen:
                continue
            f.write(json.dumps({"instruction": inst, "response": resp}, ensure_ascii=False) + "\n")
            rows.append(obj)
            seen.add(key)
            added += 1
            if len(rows) >= TARGET_EXAMPLES:
                break

    print(f"Added {added}. Total: {len(rows)} / {TARGET_EXAMPLES}")
    time.sleep(SLEEP_SECONDS)

print("Saved:", OUT_PATH)
