# %% [markdown]
# 
# # LLMRunner2 — Participant-Scheduled Two‑Phase Perceptual Task
# 
# This runner consumes the generator-built schedule **`trials_participants.csv`** (from `export_trials_with_participants()` in your ImageGen),
# and executes **Phase 1 (independent)** and **Phase 2 (group revision)** for each **(TrialID, ParticipantID)** pairing.
# 
# **Key differences vs. the legacy runner:**
# - No internal scheduling; **generator decides** trial presence/absence and mixture assignment.
# - One noise background per trial; mixture-present trials have **one strong** and **two weak** participants.
# - Minimal, lean loops that directly call providers based on `ParticipantID → provider/model` mapping.
# 

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
TRIALS_PARTICIPANTS_PATH = Path(os.getenv("TRIALS_PARTICIPANTS_PATH", "trials_participants.csv"))
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

# --- Participant mapping ---
# Map ParticipantID -> (provider_kind, model)
# Adjust as needed (e.g., use two distinct OpenAI models)
PARTICIPANT_TO_PROVIDER = {
    1: ("OpenAI", OPENAI_MODEL),
    2: ("OpenAI", OPENAI_MODEL),
    3: ("Gemini", GEMINI_MODEL),
}

print(f"[cfg] participants_csv={TRIALS_PARTICIPANTS_PATH}")
print(f"[cfg] results_dir={RESULTS_DIR}")
print(f"[cfg] providers={{pid: prov for pid, prov in PARTICIPANT_TO_PROVIDER.items()}}")
print(f"[cfg] models={{'openai': OPENAI_MODEL, 'gemini': GEMINI_MODEL}} dry_run={DRY_RUN}")


# %%

# Initialize clients
openai_client = None
gemini_model = None

if not DRY_RUN:
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        raise RuntimeError("Missing OPENAI_API_KEY (or set DRY_RUN=1)")

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    else:
        raise RuntimeError("Missing GEMINI_API_KEY (or set DRY_RUN=1)")
else:
    print("[info] DRY_RUN=1 — API calls will be simulated.")


# %%

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def parse_confidence(text: str) -> int:
    """Extract the first integer 1..10 from the model output; -1 if parsing fails."""
    s = str(text).strip()
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

def call_openai_phase1(image_path: str, model: str = OPENAI_MODEL):
    if DRY_RUN:
        c = int(np.random.choice(range(1, 11)))
        return str(c), c
    img_b64 = encode_image_b64(image_path)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": PHASE1_INSTR},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ],
        max_tokens=6, temperature=0
    )
    out = (resp.choices[0].message.content or "").strip()
    return out, parse_confidence(out)

def call_openai_phase2(image_path: str, self_rating: int, others: Dict[str, int], model: str = OPENAI_MODEL):
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
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}
        ],
        max_tokens=6, temperature=0
    )
    out = (resp.choices[0].message.content or "").strip()
    return out, parse_confidence(out)

def call_gemini_phase1(image_path: str):
    if DRY_RUN:
        c = int(np.random.choice(range(1, 11)))
        return str(c), c
    img = Image.open(image_path).convert("RGB")
    prompt = SYSTEM_PROMPT + "\n\n" + PHASE1_INSTR
    resp = gemini_model.generate_content([prompt, img])
    out = (resp.text or "").strip()
    return out, parse_confidence(out)

def call_gemini_phase2(image_path: str, self_rating: int, others: Dict[str, int]):
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

def load_trials_participants(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run export_trials_with_participants() in ImageGen.py first.")
    df = pd.read_csv(path)
    need = {"TrialID","Block","ParticipantID","Truth","AssignedCondition","Image"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain {need}. Got: {list(df.columns)}")
    root = Path.cwd()
    def _to_abs(p: str) -> str:
        pth = Path(str(p))
        return str(pth if pth.is_absolute() else (root / pth).resolve())
    df["Image"] = df["Image"].astype(str).map(_to_abs)
    for c in ["TrialID","ParticipantID","Truth"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    df["AssignedCondition"] = df["AssignedCondition"].astype(str)
    df["Block"] = df["Block"].astype(str)
    print(f"[data] loaded {len(df)} participant-rows across {df['TrialID'].nunique()} trials")
    display(df.head(3))
    return df


# %%

def run_phase1(dfp: pd.DataFrame, limit_trials: int | None = None) -> pd.DataFrame:
    if limit_trials is not None:
        keep_tids = dfp["TrialID"].drop_duplicates().head(limit_trials).tolist()
        dfp = dfp[dfp["TrialID"].isin(keep_tids)]

    rows = []
    it = dfp.itertuples(index=False)
    for r in tqdm(it, total=len(dfp), desc="Phase 1 — independent (participants)"):
        tid   = int(r.TrialID)
        pid   = int(r.ParticipantID)
        truth = int(r.Truth)
        img   = str(r.Image)
        assigned = str(r.AssignedCondition)  # equal/weak/strong
        prov_kind, model = PARTICIPANT_TO_PROVIDER[pid]

        t0 = time.time()
        if prov_kind == "OpenAI":
            raw, conf = call_openai_phase1(img, model=model)
        else:
            raw, conf = call_gemini_phase1(img)
        lat = round(time.time() - t0, 3)

        pred = conf_to_pred(conf) if 1 <= conf <= 10 else -1
        correct = int(pred == truth) if pred in (0,1) else 0

        rows.append({
            "phase": 1,
            "TrialID": tid,
            "ParticipantID": pid,
            "Truth": truth,
            "Image": img,
            "provider": f"P{pid}",
            "provider_kind": prov_kind,
            "model": model,
            "assigned_condition": assigned,
            "raw": raw,
            "confidence": conf,
            "pred": pred,
            "correct": correct,
            "latency_s": lat,
        })
    return pd.DataFrame(rows)


# %%

def run_phase2(p1: pd.DataFrame) -> pd.DataFrame:
    if p1 is None or p1.empty:
        raise ValueError("Phase 1 results are required for Phase 2.")

    rows = []
    for tid, grp in tqdm(p1.groupby("TrialID"), total=p1["TrialID"].nunique(),
                         desc="Phase 2 — discussion (participants)"):
        truth = int(grp["Truth"].iloc[0])
        # initial ratings per participant (P1/P2/P3)
        initial = {}
        for _, rr in grp.iterrows():
            pid = int(rr["ParticipantID"])
            conf = int(rr["confidence"]) if pd.notna(rr["confidence"]) else -1
            if 1 <= conf <= 10:
                initial[f"P{pid}"] = conf

        # each participant revises
        for _, rr in grp.iterrows():
            pid = int(rr["ParticipantID"])
            img = str(rr["Image"])
            prov_kind = rr["provider_kind"]
            model = rr["model"]

            self_key = f"P{pid}"
            self_conf = initial.get(self_key, -1)
            others = {k:v for k,v in initial.items() if k != self_key}

            t0 = time.time()
            if prov_kind == "OpenAI":
                raw, conf = call_openai_phase2(img, self_conf, others, model=model)
            else:
                raw, conf = call_gemini_phase2(img, self_conf, others)
            lat = round(time.time() - t0, 3)

            pred = conf_to_pred(conf) if 1 <= conf <= 10 else -1
            correct = int(pred == truth) if pred in (0,1) else 0

            rows.append({
                "phase": 2,
                "TrialID": int(tid),
                "ParticipantID": pid,
                "Truth": truth,
                "Image": img,
                "provider": f"P{pid}",
                "provider_kind": prov_kind,
                "model": model,
                "raw": raw,
                "confidence": conf,
                "pred": pred,
                "correct": correct,
                "latency_s": lat,
                "initial_confidence": self_conf,
            })
    return pd.DataFrame(rows)


# %%

def majority_from_confidences(conf_list):
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
              "Truth":"first","Image":"first",
              "confidence": list
          })
          .reset_index()
    )
    agg["consensus_pred"] = agg["confidence"].apply(majority_from_confidences)
    agg["consensus_correct"] = (agg["consensus_pred"] == agg["Truth"]).astype(int)
    return agg[["TrialID","Truth","Image","consensus_pred","consensus_correct"]]

def summarize_phase(df: pd.DataFrame, title: str):
    if df is None or df.empty:
        print(f"[warn] no results for {title}"); return
    d = df.copy()
    d = d[(d["confidence"]>=1) & (d["confidence"]<=10)]
    d["pred"] = (d["confidence"]>=6).astype(int)
    d["correct"] = (d["pred"]==d["Truth"]).astype(int)

    overall = d.groupby("provider")["correct"].mean().rename("accuracy").to_frame().reset_index()
    by_assigned = d.groupby(["provider","assigned_condition"])["correct"].mean().rename("accuracy").to_frame().reset_index()
    by_kind = d.groupby("provider_kind")["correct"].mean().rename("accuracy").to_frame().reset_index()

    print(f"=== {title} ===")
    display(overall); display(by_assigned); display(by_kind)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].set_title("Overall accuracy")
    axes[0].bar(overall["provider"], overall["accuracy"]); axes[0].set_ylim(0, 1)

    pv = by_assigned.pivot(index="assigned_condition", columns="provider", values="accuracy").fillna(0)
    pv.plot(kind="bar", ax=axes[1], title="Accuracy by assigned condition", ylim=(0, 1))

    pv2 = d.groupby(["provider","Truth"])["confidence"].mean().rename("mean_conf").to_frame().reset_index().replace({"Truth":{0:"absent",1:"present"}})
    pv2 = pv2.pivot(index="Truth", columns="provider", values="mean_conf").reindex(["absent","present"])
    pv2.plot(kind="bar", ax=axes[2], title="Mean confidence by truth", ylim=(1, 10))

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

# %% One-Click Run (uncomment to execute end-to-end)
# dfp = load_trials_participants(TRIALS_PARTICIPANTS_PATH)
# p1 = run_phase1(dfp, limit_trials=None)  # set an int to smoke-test
# p1.to_csv(PHASE1_CSV, index=False)
# summarize_phase(p1, "Phase 1 — Independent")
#
# p2 = run_phase2(p1)
# p2.to_csv(PHASE2_CSV, index=False)
# summarize_phase(p2, "Phase 2 — Discussion")
# summarize_delta(p1, p2)
#
# consensus = consensus_from_phase2(p2)
# consensus.to_csv(CONSENSUS_CSV, index=False)
# print(f"[ok] wrote consensus to {CONSENSUS_CSV}  (rows={len(consensus)})")
# display(consensus.head())


# %% [markdown]
# 
# ### Optional sanity checks (manual)
# - Confirm each *mixture-present* TrialID in `trials_participants.csv` has exactly **one** `AssignedCondition == "strong"` and **two** `"weak"` rows (same `TrialID`, different `ParticipantID`).
# - Confirm all three participant rows for a `TrialID` reference images derived from the **same noise background** (guaranteed by the generator).
# 


