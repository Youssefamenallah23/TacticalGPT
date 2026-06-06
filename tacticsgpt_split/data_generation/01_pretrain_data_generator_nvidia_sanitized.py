# Sanitized pretraining-data generator converted from notebook.
# Set NVIDIA_API_KEY or paste it at runtime.

# # Generate a 5 MB File of Tactical Match-Analysis Articles
# This notebook uses the **free NVIDIA NIM API** (Llama 3.1 8B Instruct) to produce long-form tactical breakdowns—
# like studio analysis pieces—about formations, positional adjustments, and in-game tactical changes.
# 
# **Required:**
# - Python packages: `openai` (install via `!pip install openai`)
# - A free NVIDIA API key from [build.nvidia.com](https://build.nvidia.com) (NIM APIs section)
# 
# **What you'll get:**
# A single `tactical_match_analysis_5mb.txt` file (≥5 MB) full of articles such as:
# - *4‑3‑3 vs 5‑3‑2 – Breaking Down a Compact Defense*
# - *Counter‑Attack Masterclass – 5‑3‑2 Springing into a 3‑5‑2*
# - *Formation Fluidity – How a 4‑2‑3‑1 Became a 3‑4‑3 in Possession*
# - … and many more.
# 
# The notebook cycles through different tactical scenarios, slightly varying each prompt so every article is unique. It stops automatically when the output file reaches 5 MB.
# 
# ---
# 1. Set your API key in the cell below.
# 2. Run all cells.
# 3. Wait (it may take 15‑30 minutes depending on rate limits).
# 4. Retrieve the file `tactical_match_analysis_5mb.txt` from the notebook’s working directory.

# Install the openai package if you haven't already
# Colab magic: !pip install -q openai

import openai
import time
from pathlib import Path

# ========== CONFIGURE YOUR NVIDIA API KEY ==========
# Replace the string below with your actual key, or set the environment variable NVIDIA_API_KEY.
import os
from getpass import getpass
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or getpass("Paste NVIDIA API key: ")
# ==================================================

client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)
MODEL = "google/gemma-3-4b-it"  # free model, fast enough for our purpose

def generate_article(prompt, max_tokens=4096, temperature=0.85):
    """Call the NVIDIA API and return the generated article text. Retries once on error."""
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API error: {e}")
        time.sleep(15)
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e2:
            print(f"Retry failed: {e2}")
            return ""

print("Setup complete.")

# ## Prompt Templates
# Each template describes a different tactical scenario. The notebook will cycle through them and add a uniqueness phrase to keep the content fresh.

base_prompts = [
    {
        "title": "4-3-3 vs 5-3-2 – Breaking Down a Compact Defense",
        "prompt": """Write a detailed tactical analysis article (2000+ words) of a match where Team A set up in a 4-3-3 and faced Team B in a deep 5-3-2 block.
Discuss the initial positioning, how Team A tried to break the low block, the role of the full‑backs and the midfield pivot, and why the 5-3-2 was so difficult to penetrate.
Then describe a tactical change: Team A switched to a 3-4-3 or pushed a midfielder higher, and explain what the opponent should have done to counter that.
Include specific moments, player movements, and what both managers did right/wrong. Write in narrative, analytical prose like a professional tactics blog – no bullet points."""
    },
    {
        "title": "Counter-Attack Masterclass – 5-3-2 Springing into a 3-5-2",
        "prompt": """Write a 2000+ word match analysis article about a side playing a 5-3-2 that transitions into a 3-5-2 in attack, executed a devastating counter‑attack strategy against a possession-heavy 4-3-3 opponent.
Explain how they set traps in midfield, the cues for the wing‑backs to bomb forward, and the front two’s synchronised runs.
Detail a key second‑half formation switch (e.g., the opponent moving to a 4-2-3-1) and what the counter‑attacking team should have adjusted, but didn’t.
Narrate the tactical chess match, the positional rotations, and the decisive moments. Pure prose, no lists."""
    },
    {
        "title": "High‑Press Failure – When the 4-3-3 Press Got Bypassed",
        "prompt": """Analyse a match where Team A tried to impose a high press from a 4-3-3 shape but was repeatedly bypassed by a clever opponent.
Describe the pressing triggers, the covering shadows, and why the press failed (e.g., poor communication, wrong angle, opponent’s use of a double pivot).
Show how the opponent switched from a 4-4-2 to a 3-5-2 at half‑time to overload the first line, completely neutralising the press.
Explain what the high‑pressing team should have done instead: maybe dropping into a mid‑block or changing marking responsibilities.
Write like a detailed studio analysis piece, full of tactical insight and “if they had only…” moments."""
    },
    {
        "title": "Formation Fluidity – How a 4-2-3-1 Became a 3-4-3 in Possession",
        "prompt": """Write a 2000+ word article dissecting a game where one team’s formation on paper was a 4-2-3-1, but in possession it morphed into a 3-4-3 diamond, causing the opponent’s 5-3-2 to unravel.
Focus on the movement of the holding midfielder dropping between the centre‑backs, the full‑backs pushing high, and the winger coming inside to create overloads.
Describe a tactical counter‑measure the opponent attempted (e.g., switching to a 4-4-2 mid‑block) and why it didn’t work.
Discuss the pressing schemes, the positional rotations, and the critical errors. Continue the narrative style throughout."""
    },
    {
        "title": "The Second‑Half Overhaul – From 4-4-2 to 3-5-2 to Protect a Lead",
        "prompt": """Compose an in‑depth match analysis (2000+ words) about a team that started in a 4-4-2, took the lead, and then switched to a 3-5-2 in the second half to defend deeper.
Explain the reasoning behind the change, how the wingers became wing‑backs, the midfield three’s shielding, and the transition to a counter‑attacking style.
Describe how the opposition, initially in a 4-3-3, struggled to create chances against the new low block.
Add a “what should they have done” section: should the opposition have moved to a 4-2-4? Brought on a target man?
Write like a feature for a coaching website, full of diagrams‑like descriptions in words."""
    },
    {
        "title": "Man‑Marking vs Zonal – A Tactical Breakdown of a 4-3-3 vs 3-5-2 Battle",
        "prompt": """Analyse a high‑intensity match where Team A played a zonal 4-3-3 press while Team B used a man‑marking 3-5-2 system in midfield.
Detail the chaos it caused: the false nine dragging markers, the overloads on the flanks, and how the man‑marking eventually collapsed after a tactical tweak.
Include a moment where Team B’s coach abandoned the man‑marking, dropping into a 5-4-1, and discuss if that was the right call.
Weave in positional analysis, individual battles, and the final outcome. Keep it purely narrative, no bullet points."""
    },
    {
        "title": "Set‑Piece Chess – How a 5-3-2 Side Used Corners to Flip a Game",
        "prompt": """Write a 2000+ word analysis focusing on set‑pieces in a match between a 5-3-2 team and a 4-3-3 side.
Explain the zonal vs man‑marking setup on corners, the blocking schemes, and how a late tactical adjustment (switching from near‑post to far‑post deliveries) resulted in two goals.
Discuss the opponent’s failure to adapt and what they should have done (e.g., changing the marking system, adding a player on the line).
This should read like a specialist set‑piece analysis article, full of detail about runs, feints, and positioning."""
    },
    {
        "title": "Exploiting the Space Behind – A 4-3-3’s Direct Counter Against a High Backline",
        "prompt": """Narrate a match where a team in a 4-3-3 deliberately sat deep, absorbed pressure from a 4-2-3-1 side that pushed its full‑backs extremely high, and then repeatedly launched long diagonals behind the defence.
Describe the striker’s angled runs, the inside forwards cutting in, and the moment the opponent changed formation (to a 3-4-3) to try and fix the issue – but it was too late.
Analyse the tactical naivety of the high‑line team and what they should have instructed their holding midfielder to do.
Write in a flowing, article style like a TV analysis segment."""
    },
    {
        "title": "Pressing Trap – How a 4-4-2 Diamond Turned into a 4-2-4 Press",
        "prompt": """Analyse a game where a side set up in a 4-4-2 diamond that, when pressing, transformed into a 4-2-4 with the attacking midfielder joining the front line.
Dissect the specific pressing trap used to force the opponent (in a 4-3-3) to play into wide areas where they were immediately swarmed.
Discuss the moment the opponent adjusted by dropping a midfielder into a back three in build‑up (creating a 3-4-3 shape), and how the pressing team failed to react.
Provide a detailed breakdown of the positional rotations and the “if they did X instead of Y” analysis."""
    }
]

print(f"Loaded {len(base_prompts)} base prompts.")

# ## The Main Generation Loop
# This cell will repeatedly generate articles and append them to the output file until it reaches 5 MB.  
# You can stop and restart the notebook at any time; the file will continue to grow.

output_path = "tactical_match_analysis_5mb.txt"
target_size = 5 * 1024 * 1024  # 5 MB

# Initialise the file with a header
with open(output_path, "w", encoding="utf-8") as f:
    f.write("TACTICAL MATCH ANALYSIS ARTICLES\n")
    f.write("Generated by NVIDIA Llama 3.1 8B Instruct\n")
    f.write("A compilation of in‑depth tactical breakdowns, formation adjustments, and positional analysis.\n\n")

print(f"Target file size: {target_size / (1024*1024):.1f} MB")
iteration = 0

while True:
    current_size = Path(output_path).stat().st_size
    if current_size >= target_size:
        print(f"\nReached {current_size / (1024*1024):.2f} MB. File saved as '{output_path}'.")
        break

    # Cycle through the base prompts
    base = base_prompts[iteration % len(base_prompts)]
    # Add a uniqueness phrase so each article is different
    variation = f"\n\n(Part {iteration + 1}: make this analysis completely unique with new team names, match details, and player descriptions. Ensure the article is at least 2000 words and reads like a freshly written tactical column. Do not repeat previous narratives.)"

    full_prompt = base["prompt"] + variation

    print(f"\n[{current_size / (1024*1024):.2f} MB] Article: {base['title']} (iteration {iteration+1})")
    article_text = generate_article(full_prompt, max_tokens=4096)
    if not article_text:
        print("Empty response. Waiting 30 seconds and retrying...")
        time.sleep(30)
        continue

    # Append article with clear heading
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"ARTICLE: {base['title']} (Analysis #{iteration+1})\n")
        f.write(f"{'='*80}\n\n")
        f.write(article_text)

    iteration += 1
    # Respect the free API rate limit – wait 5 seconds between calls
    time.sleep(5)

print(f"Final file size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
print("Done! You can download the file from the current working directory.")
