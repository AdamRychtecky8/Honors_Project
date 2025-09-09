# %%
# ## Imports & Configuration

# %%
from __future__ import annotations

import os, base64, time, json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# APIs
import google.generativeai as genai
from openai import OpenAI

# --- Paths ---
TRIALS_PATH = Path(os.getenv("TRIALS_PATH", "trials.csv"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "API_files"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PHASE1_CSV = RESULTS_DIR / "phase1_results.csv"
PHASE2_CSV = RESULTS_DIR / "phase2_results.csv"
CONSENSUS_CSV = RESULTS_DIR / "phase2_consensus.csv"

# --- Models & keys ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-thinking-exp-01-21")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Safety toggle (no network when True) ---
DRY_RUN = bool(int(os.getenv("DRY_RUN", "0")))

print(f"[cfg] trials={TRIALS_PATH}  results_dir={RESULTS_DIR}")
print(f"[cfg] providers=['OpenAI','Gemini'] models={[OPENAI_MODEL, GEMINI_MODEL]}  dry_run={DRY_RUN}")


# %%
def load_trials(trials_path: Path) -> pd.DataFrame:
    if not trials_path.exists():
        raise FileNotFoundError(
            f"Missing {trials_path}. Build it from your generator's allimgs/index.csv first."
        )
    df = pd.read_csv(trials_path)
    need = {"TrialID","Condition","Truth","Image"}
    if not need.issubset(df.columns):
        raise ValueError(f"trials.csv must contain {need}. Got: {list(df.columns)}")

    df["TrialID"] = pd.to_numeric(df["TrialID"], errors="coerce").astype(int)
    df["Condition"] = df["Condition"].astype(str).str.lower().str.strip()
    df["Truth"] = pd.to_numeric(df["Truth"], errors="coerce").astype(int)
    # make image paths absolute if not already
    root = Path.cwd()
    def _to_abs(p: str) -> str:
        pth = Path(str(p))
        if pth.is_absolute():
            return str(pth)
        return str((root / pth).resolve())
    df["Image"] = df["Image"].astype(str).map(_to_abs)
    return df

df_trials = load_trials(TRIALS_PATH)
print(f"[data] loaded {len(df_trials)} trials")
display(df_trials.head(3))

# %%
if not DRY_RUN:
    assert OPENAI_API_KEY, "Missing OPENAI_API_KEY"
    assert GEMINI_API_KEY, "Missing GEMINI_API_KEY"

# Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY) if not DRY_RUN else None
if not DRY_RUN:
    genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL) if not DRY_RUN else None

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_confidence(text: str) -> int:
    """
    Extract the first integer 1..10 from the model output.
    Returns -1 if parsing fails or out of range.
    """
    s = str(text).strip()
    # grab first integer token
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

def conf_to_pred(c: int) -> int:
    """Map 1–5 -> 0 (absent), 6–10 -> 1 (present)."""
    return 1 if c >= 6 else 0

# %%
# ## Prompts
# - Phase 1: independent rating with 50% prior.
# - Phase 2: re-prompt with own initial rating + others' ratings.

# %%
SYSTEM_PROMPT = (
    "You are a perceptual decision-maker. You will receive an 8-bit grayscale image (mid-gray ≈ 128).\n"
    "A luminance signal may or may not be present. The prior probability of signal presence is 50%."
)

PHASE1_INSTR = (
    "Decide if a faint luminance signal is present.\n"
    "Return ONLY one integer 1–10 (no other text):\n"
    "1 = very confident ABSENT ... 5 = low confidence ABSENT; 6 = low confidence PRESENT ... 10 = very confident PRESENT."
)

def phase2_instr(self_rating: int, others: Dict[str, int]) -> str:
    other_txt = ", ".join([f"{k}: {v}" for k, v in others.items()]) if others else "none"
    return (
        "You already responded independently. Now reconsider after seeing the others' initial ratings.\n"
        f"Your initial rating: {self_rating}. Others: {other_txt}.\n"
        "Return ONLY one integer 1–10."
    )

# %%
# ## Provider Call Functions (Phase 1 & Phase 2)

# %%
def call_openai_phase1(image_path: str, model: str = OPENAI_MODEL) -> Tuple[str, int]:
    if DRY_RUN:
        c = int(np.random.choice(range(1,11)))
        return str(c), c
    img_b64 = encode_image_b64(image_path)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": PHASE1_INSTR},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}" }},
            ]}
        ],
        max_tokens=6, temperature=0
    )
    out = (resp.choices[0].message.content or "").strip()
    return out, parse_confidence(out)

def call_openai_phase2(image_path: str, self_rating: int, others: Dict[str,int], model: str = OPENAI_MODEL) -> Tuple[str, int]:
    if DRY_RUN:
        peer = int(round(np.mean(list(others.values())))) if others else self_rating
        c = int(np.clip(round((self_rating + peer)/2), 1, 10))
        return str(c), c
    img_b64 = encode_image_b64(image_path)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": phase2_instr(self_rating, others)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}" }},
            ]}
        ],
        max_tokens=6, temperature=0
    )
    out = (resp.choices[0].message.content or "").strip()
    return out, parse_confidence(out)

def call_gemini_phase1(image_path: str) -> Tuple[str, int]:
    if DRY_RUN:
        c = int(np.random.choice(range(1,11)))
        return str(c), c
    img = Image.open(image_path).convert("RGB")   # model expects RGB; content is grayscale-looking
    prompt = SYSTEM_PROMPT + "\n\n" + PHASE1_INSTR
    resp = gemini_model.generate_content([prompt, img])
    out = (resp.text or "").strip()
    return out, parse_confidence(out)

def call_gemini_phase2(image_path: str, self_rating: int, others: Dict[str,int]) -> Tuple[str, int]:
    if DRY_RUN:
        peer = int(round(np.mean(list(others.values())))) if others else self_rating
        c = int(np.clip(round((self_rating + peer)/2), 1, 10))
        return str(c), c
    img = Image.open(image_path).convert("RGB")
    prompt = SYSTEM_PROMPT + "\n\n" + phase2_instr(self_rating, others)
    resp = gemini_model.generate_content([prompt, img])
    out = (resp.text or "").strip()
    return out, parse_confidence(out)


# %%
# ## Runner — Phase 1 (Independent)

# %%
def run_phase1(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    it = df if limit is None else df.head(limit)
    rows = []
    for _, r in tqdm(it.iterrows(), total=len(it), desc="Phase 1 — independent"):
        trial_id  = int(r["TrialID"])
        cond      = str(r["Condition"])
        truth     = int(r["Truth"])
        img       = str(r["Image"])

        # OpenAI
        t0 = time.time()
        oai_raw, oai_conf = call_openai_phase1(img)
        t1 = time.time()

        # Gemini
        gmi_raw, gmi_conf = call_gemini_phase1(img)
        t2 = time.time()

        for prov, model, raw, conf, lat in [
            ("OpenAI", OPENAI_MODEL, oai_raw, oai_conf, round(t1-t0,3)),
            ("Gemini", GEMINI_MODEL, gmi_raw, gmi_conf, round(t2-t1,3)),
        ]:
            pred = conf_to_pred(conf) if 1 <= conf <= 10 else -1
            correct = int(pred == truth) if pred in (0,1) else 0
            rows.append({
                "phase": 1, "TrialID": trial_id, "Condition": cond, "Truth": truth, "Image": img,
                "provider": prov, "model": model, "raw": raw, "confidence": conf,
                "pred": pred, "correct": correct, "latency_s": lat
            })
    return pd.DataFrame(rows)

# %%
def run_phase2(p1: pd.DataFrame) -> pd.DataFrame:
    if p1 is None or p1.empty:
        raise ValueError("Phase 1 results are required for Phase 2.")
    rows = []

    # group by trial
    for trial_id, grp in tqdm(p1.groupby("TrialID"), total=p1["TrialID"].nunique(), desc="Phase 2 — discussion"):
        cond  = grp["Condition"].iloc[0]
        truth = int(grp["Truth"].iloc[0])
        img   = grp["Image"].iloc[0]

        # initial ratings per provider
        initial = {
            prov: int(conf) for prov, conf in zip(grp["provider"], grp["confidence"])
            if pd.notna(conf) and 1 <= int(conf) <= 10
        }

        # OpenAI revises
        oai_self = initial.get("OpenAI", -1)
        others_oai = {k:v for k,v in initial.items() if k != "OpenAI"}
        t0 = time.time()
        oai_raw, oai_conf = call_openai_phase2(img, oai_self, others_oai)
        t1 = time.time()

        # Gemini revises
        gmi_self = initial.get("Gemini", -1)
        others_gmi = {k:v for k,v in initial.items() if k != "Gemini"}
        gmi_raw, gmi_conf = call_gemini_phase2(img, gmi_self, others_gmi)
        t2 = time.time()

        for prov, model, raw, conf, lat in [
            ("OpenAI", OPENAI_MODEL, oai_raw, oai_conf, round(t1-t0,3)),
            ("Gemini", GEMINI_MODEL, gmi_raw, gmi_conf, round(t2-t1,3)),
        ]:
            pred = conf_to_pred(conf) if 1 <= conf <= 10 else -1
            correct = int(pred == truth) if pred in (0,1) else 0
            rows.append({
                "phase": 2, "TrialID": int(trial_id), "Condition": cond, "Truth": truth, "Image": img,
                "provider": prov, "model": model, "raw": raw, "confidence": conf,
                "pred": pred, "correct": correct, "latency_s": lat,
                "initial_confidence": initial.get(prov, -1)
            })
    return pd.DataFrame(rows)


# %%
# ## Consensus (Phase 2 majority) & Summaries

# %%
def majority_from_confidences(conf_list):
    """majority of binary predictions derived from confidences; ties -> -1"""
    bins = [conf_to_pred(c) for c in conf_list if 1 <= int(c) <= 10]
    if not bins:
        return -1
    ones = sum(bins)
    zeros = len(bins) - ones
    if ones > zeros: return 1
    if zeros > ones: return 0
    return -1  # tie

def consensus_from_phase2(p2: pd.DataFrame) -> pd.DataFrame:
    if p2 is None or p2.empty:
        return pd.DataFrame()
    agg = (
        p2.groupby("TrialID")
          .agg({
              "Condition":"first","Truth":"first","Image":"first",
              "confidence": list
          })
          .reset_index()
    )
    agg["consensus_pred"] = agg["confidence"].apply(majority_from_confidences)
    agg["consensus_correct"] = (agg["consensus_pred"] == agg["Truth"]).astype(int)
    return agg[["TrialID","Condition","Truth","Image","consensus_pred","consensus_correct"]]

def summarize_phase(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        print(f"[warn] no results for {title}"); return
    d = df.copy()
    d = d[(d["confidence"]>=1) & (d["confidence"]<=10)]
    d["pred"] = (d["confidence"]>=6).astype(int)
    d["correct"] = (d["pred"]==d["Truth"]).astype(int)

    overall = d.groupby("provider")["correct"].mean().rename("accuracy").to_frame().reset_index()
    by_cond = d.groupby(["provider","Condition"])["correct"].mean().rename("accuracy").to_frame().reset_index()
    mean_conf = d.groupby(["provider","Truth"])["confidence"].mean().rename("mean_conf").to_frame().reset_index().replace({"Truth":{0:"absent",1:"present"}})

    print(f"=== {title} ===")
    display(overall); display(by_cond); display(mean_conf)

    fig, axes = plt.subplots(1,3, figsize=(14,4))
    axes[0].set_title("Overall accuracy")
    axes[0].bar(overall["provider"], overall["accuracy"]); axes[0].set_ylim(0,1)

    pv = by_cond.pivot(index="Condition", columns="provider", values="accuracy").fillna(0)
    pv.plot(kind="bar", ax=axes[1], title="Accuracy by condition", ylim=(0,1))

    pv2 = mean_conf.pivot(index="Truth", columns="provider", values="mean_conf").reindex(["absent","present"])
    pv2.plot(kind="bar", ax=axes[2], title="Mean confidence by truth", ylim=(1,10))

    plt.tight_layout(); plt.show()

def summarize_delta(p1: pd.DataFrame, p2: pd.DataFrame):
    if p1 is None or p2 is None or p1.empty or p2.empty:
        print("[warn] need both phases for delta"); return
    a = p1[["TrialID","provider","confidence"]].rename(columns={"confidence":"conf1"})
    b = p2[["TrialID","provider","confidence"]].rename(columns={"confidence":"conf2"})
    m = a.merge(b, on=["TrialID","provider"], how="inner")
    m["delta"] = m["conf2"] - m["conf1"]
    print("Change in confidence (phase2 - phase1):")
    display(m.groupby("provider")["delta"].describe().round(2))

# %%
# ## One-Click Run (commented by default)
# Uncomment to run end-to-end. Use `DRY_RUN=1` to simulate first.

# %%
# # Phase 1
# p1 = run_phase1(df_trials, limit=None)  # set an int for a quick smoke test
# p1.to_csv(PHASE1_CSV, index=False)
# summarize_phase(p1, "Phase 1 — Independent")

# # Phase 2
# p2 = run_phase2(p1)
# p2.to_csv(PHASE2_CSV, index=False)
# summarize_phase(p2, "Phase 2 — Discussion")
# summarize_delta(p1, p2)

# # Consensus (from Phase 2)
# consensus = consensus_from_phase2(p2)
# consensus.to_csv(CONSENSUS_CSV, index=False)
# print(f"[ok] wrote consensus to {CONSENSUS_CSV}  (rows={len(consensus)})")
# display(consensus.head())


