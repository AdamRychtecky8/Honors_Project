# %% [markdown]
# 
# # SmokeTest — Environment & Mini-Run Validator
# 
# Use this notebook to quickly verify:
# 1. API keys & model IDs are set and clients initialize.
# 2. `trials_participants.csv` exists and has basic integrity.
# 3. Each provider (OpenAI & Gemini) can return a clean **1–10** on a tiny image.
# 4. A **mini end-to-end run** over a small number of trials (configurable) succeeds.
# 
# > Run this **before** a full experiment to save time and money.
# 

# %%

# ==== Quick knobs ====
from pathlib import Path
import os

# How many TrialIDs to run in the mini live run (Phase 1 + Phase 2)?
N_TRIALS = 5

# Set to True to simulate responses without hitting APIs
DRY_RUN = False

# Model IDs (override via env if needed)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Paths
TRIALS_PARTICIPANTS_PATH = Path("trials_participants.csv")
LLMRUNNER2_NOTEBOOK = Path("LLMRunner2.ipynb")

# Propagate DRY_RUN to runner
os.environ["DRY_RUN"] = "1" if DRY_RUN else "0"

print(f"[cfg] N_TRIALS={N_TRIALS} DRY_RUN={DRY_RUN}")
print(f"[cfg] OPENAI_MODEL={OPENAI_MODEL}  GEMINI_MODEL={GEMINI_MODEL}")
print(f"[cfg] trials_participants={TRIALS_PARTICIPANTS_PATH}  runner_nb={LLMRUNNER2_NOTEBOOK}")


# %%

# ==== Environment & client sanity ====
import sys

if not DRY_RUN:
    assert os.getenv("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"
    assert os.getenv("GEMINI_API_KEY"), "Missing GEMINI_API_KEY"

# Imports
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("OpenAI SDK missing. Try: pip install openai>=1.30") from e

try:
    import google.generativeai as genai
except Exception as e:
    raise RuntimeError("Google GenAI SDK missing. Try: pip install google-generativeai") from e

print("[ok] SDK imports succeeded")

# Initialize (no requests yet)
if not DRY_RUN:
    _oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    _gem = genai.GenerativeModel(GEMINI_MODEL, generation_config={"temperature": 0})
    print("[ok] Clients constructed")
else:
    print("[info] DRY_RUN=1 — skipping client construction checks")


# %%

# ==== Schedule integrity (fast) ====
import pandas as pd

assert TRIALS_PARTICIPANTS_PATH.exists(), "trials_participants.csv not found. Run ImageGen2 export first."
df = pd.read_csv(TRIALS_PARTICIPANTS_PATH)

need = {"TrialID","Block","ParticipantID","Truth","AssignedCondition","Image"}
assert need.issubset(df.columns), f"Missing required columns. Need {need}, got {set(df.columns)}"

triplets = df.groupby("TrialID").size()
assert (triplets == 3).all(), "Each TrialID must have exactly 3 participant rows"

# Basic mixture-present composition check on a small sample
mix_pre = df[(df["Block"]=="mixture") & (df["Truth"]==1)]
if not mix_pre.empty:
    ct = (mix_pre.groupby(["TrialID","AssignedCondition"]).size()
          .unstack(fill_value=0))
    sample = ct.head(10)
    display(sample)
    assert ((ct.get("strong",0)==1) & (ct.get("weak",0)==2)).all(), "Mixture-present must be 1 strong + 2 weak"
else:
    print("[warn] No mixture-present rows found (small dataset?)")

print("[ok] trials_participants.csv basic integrity looks good")


# %%

# ==== Path existence check (sample 30 rows) ====
from pathlib import Path

sample = df.sample(min(30, len(df)), random_state=0)
missing = []
for _, r in sample.iterrows():
    p = Path(str(r["Image"]))
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        missing.append(str(p))
if missing:
    print("Missing files (first 10):", missing[:10])
    raise FileNotFoundError(f"{len(missing)} missing scheduled image files")
print("[ok] Sampled image paths exist")


# %%

# ==== Single-call provider smoke tests (tiny image) ====
import numpy as np
from PIL import Image

def parse_confidence(text: str) -> int:
    s = (text or "").strip()
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif num:
            break
    try:
        val = int(num)
        return val if 1 <= val <= 10 else -1
    except:
        return -1

# Build tiny image
tmp = Path("smoke_present.png")
arr = np.full((64,64), 128, np.uint8)
yy, xx = np.ogrid[:64, :64]
mask = (yy-32)**2 + (xx-32)**2 <= 7**2
arr[mask] = 160
Image.fromarray(arr).save(tmp)

SYSTEM_PROMPT = "You are a perceptual decision-maker. Return ONLY one integer 1–10."
PHASE1_INSTR = "Is a faint luminance signal present? 1–5=absent, 6–10=present. Return ONLY the integer."

if not DRY_RUN:
    # OpenAI (data URL)
    import base64, time
    b64 = base64.b64encode(tmp.read_bytes()).decode("utf-8")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    t0 = time.time()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":[
                {"type":"text","text":PHASE1_INSTR},
                {"type":"image_url","image_url":{"url": f"data:image/png;base64,{b64}"}}
            ]}
        ],
        temperature=0, max_tokens=6,
    )
    oai_text = (resp.choices[0].message.content or "").strip()
    oai_conf = parse_confidence(oai_text)
    print(f"[OpenAI] raw='{oai_text}' parsed={oai_conf}")

    # Gemini (inline PIL)
    import google.generativeai as genai, time
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL, generation_config={"temperature": 0})
    img = Image.open(tmp).convert("RGB")
    t0 = time.time()
    resp = model.generate_content([SYSTEM_PROMPT + "\n\n" + PHASE1_INSTR, img])
    g_text = (resp.text or "").strip()
    g_conf = parse_confidence(g_text)
    print(f"[Gemini] raw='{g_text}' parsed={g_conf}")

    assert 1 <= oai_conf <= 10, "OpenAI smoke test did not return 1–10"
    assert 1 <= g_conf <= 10, "Gemini smoke test did not return 1–10"
    print("[ok] Both providers returned clean 1–10 integers")
else:
    print("[info] DRY_RUN=1 — skipping provider smoke tests")


# %% [markdown]
# 
# ## Mini End-to-End Run (Phase 1 + Phase 2)
# 
# Executes the participant-scheduled run using **LLMRunner2** over `N_TRIALS` TrialIDs.
# - Honors `DRY_RUN` (set at the top).
# - Writes results into `API_files/`.
# 

# %%

import nbformat

assert LLMRUNNER2_NOTEBOOK.exists(), "LLMRunner2.ipynb not found. Generate it first."

# Execute code cells from LLMRunner2 in this kernel
nb_runner = nbformat.read(LLMRUNNER2_NOTEBOOK.open("r", encoding="utf-8"), as_version=4)
for cell in nb_runner.cells:
    if cell.cell_type == "code" and cell.source.strip():
        exec(compile(cell.source, "<LLMRunner2_cell>", "exec"), globals())

# Now run a limited set
from pathlib import Path as _Path
dfp = load_trials_participants(_Path(TRIALS_PARTICIPANTS_PATH))
p1 = run_phase1(dfp, limit_trials=int(N_TRIALS))
p2 = run_phase2(p1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
p1.to_csv(PHASE1_CSV, index=False)
p2.to_csv(PHASE2_CSV, index=False)
consensus = consensus_from_phase2(p2)
consensus.to_csv(CONSENSUS_CSV, index=False)

print(f"[ok] Wrote: {PHASE1_CSV}, {PHASE2_CSV}, {CONSENSUS_CSV}")
display(p1.head(3)); display(p2.head(3)); display(consensus.head(3))


# %%

# Optional: quick summaries (works in DRY_RUN too)
summarize_phase(p1, "Phase 1 — Independent (mini)")
summarize_phase(p2, "Phase 2 — Discussion (mini)")
summarize_delta(p1, p2)



