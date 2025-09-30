# %% [markdown]
# 
# # ImageGen2 — Calibrated Stimuli + Participant Schedule (Equal & Mixture)
# 
# This notebook:
# 1. Generates calibrated Gaussian-blob stimuli with **one shared noise background per TrialID**.
# 2. Builds **participant-level schedule** `trials_participants.csv`:
#    - **Equal block**: 3 participants see the same image (present/absent, 50% prior).
#    - **Mixture block**: present/absent (50% prior); if present, **one random participant** gets **strong** and the other two **weak**, **same noise** for all three.
# 3. Saves PNG/NPY under `perceptual_dataset_calibrated/<cond>/<label>/...` (same structure as before).
# 4. Provides **one-click cells** at the end to:
#    - Run **LLMRunner2** logic in **DRY_RUN** mode (no API costs).
#    - Execute **Preflight_Tests.ipynb** assertions.
# 
# > Set small trial counts for fast testing, then scale up.
# 

# %%

from __future__ import annotations
import numpy as np, math, json, csv, random, os, re, shutil
from pathlib import Path
from PIL import Image
import pandas as pd

# ====== Experiment constants ======
MEAN_LUMINANCE = 28.0          # cd/m^2
NOISE_SD = 4.375               # cd/m^2 (RMS)
FIELD_DEG = 15.0               # degrees
PIXELS_PER_DEG = 1.0 / 0.037   # ≈27.027 px/deg
IMG_SIZE = int(round(FIELD_DEG * PIXELS_PER_DEG))  # ≈405 px
SIGMA_DEG = 0.5
SIGMA_PX  = SIGMA_DEG * PIXELS_PER_DEG             # ≈13.5 px

# Peak contrasts → amplitudes = contrast * MEAN_LUMINANCE
PEAK_CONTRAST_EQUAL  = 0.07
PEAK_CONTRAST_WEAK   = 0.02
PEAK_CONTRAST_STRONG = 0.15
TARGET_SNRS = {"equal": 3.07, "weak": 0.77, "strong": 13.8}

# ====== Display calibration (fixed linear mapping) ======
DISPLAY_MEAN_8BIT = 128.0
DISPLAY_GAIN      = DISPLAY_MEAN_8BIT / MEAN_LUMINANCE   # ≈ 128/28 ≈ 4.57

# ====== Trial scheduling config ======
N_EQUAL_TRIALS   = 24     # adjust for production (e.g., 200)
N_MIX_TRIALS     = 24     # adjust for production (e.g., 200)
P_SIGNAL_PRESENT = 0.5
PARTICIPANTS     = [1, 2, 3]

# RNG (fixed by default for reproducibility; set to None for random)
RANDOM_SEED = 1
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

# Output roots
OUTPUT_DIR = Path("perceptual_dataset_calibrated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%

def gaussian_blob(size, sigma_px, amplitude, center=None):
    h = w = size
    if center is None:
        cy, cx = (h-1)/2.0, (w-1)/2.0
    else:
        cy, cx = center
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    g = np.exp(-(((y - cy)**2 + (x - cx)**2) / (2*sigma_px**2)))
    return amplitude * g

def white_noise(size, mean_lum=MEAN_LUMINANCE, sigma=NOISE_SD):
    return np.random.normal(mean_lum, sigma, (size, size))

def make_stimulus(present=True, peak_contrast=PEAK_CONTRAST_EQUAL, noise=None):
    '''
    Returns float cd/m^2 arrays: combined, noise, signal.
    If `noise` is provided, reuse it so multiple variants share the exact same background.
    '''
    if noise is None:
        noise = white_noise(IMG_SIZE)
    else:
        assert noise.shape == (IMG_SIZE, IMG_SIZE), "noise shape mismatch"

    if present:
        amp = peak_contrast * MEAN_LUMINANCE
        signal = gaussian_blob(IMG_SIZE, SIGMA_PX, amp)
        combined = noise + signal
    else:
        signal = np.zeros((IMG_SIZE, IMG_SIZE), dtype=float)
        combined = noise
    return combined, noise, signal

def to_uint8_visible(lum_float):
    counts = lum_float * DISPLAY_GAIN
    return np.clip(counts, 0, 255).astype(np.uint8)

def to_uint8_visible_signal(signal_float):
    return to_uint8_visible(MEAN_LUMINANCE + signal_float)


# %%

def ensure_dirs(base: Path, cond: str):
    for label in ("absent", "present"):
        (base / cond / label / "combined").mkdir(parents=True, exist_ok=True)
        (base / cond / label / "noise").mkdir(parents=True, exist_ok=True)
        (base / cond / label / "signal").mkdir(parents=True, exist_ok=True)

def save_trial(out_dir: Path, cond: str, label: str, idx: int, combined, noise, signal,
               save_npy=True, save_signal_png=True):
    stem = f"{cond}_{label}_{idx:03d}"
    # PNGs
    Image.fromarray(to_uint8_visible(combined)).save(out_dir/cond/label/"combined"/f"{stem}_combined.png")
    Image.fromarray(to_uint8_visible(noise)).save(out_dir/cond/label/"noise"/f"{stem}_noise.png")
    if save_signal_png:
        Image.fromarray(to_uint8_visible_signal(signal)).save(out_dir/cond/label/"signal"/f"{stem}_signal.png")
    # NPYs
    if save_npy:
        np.save(out_dir/cond/label/"combined"/f"{stem}_combined.npy", combined)
        np.save(out_dir/cond/label/"noise"/f"{stem}_noise.npy", noise)
        np.save(out_dir/cond/label/"signal"/f"{stem}_signal.npy", signal)


# %%

def export_trials_with_participants():
    '''
    Build stimuli and a participant-level schedule so the runner can just look up (TrialID, ParticipantID).
    - Equal block: 3 participants see the same image (present/absent with 0.5 prior).
    - Mixture block: present/absent with 0.5 prior; if present, one random participant gets STRONG, others WEAK.
      All three share the same noise background for that trial.
    Outputs:
      - calibrated PNG/NPY under perceptual_dataset_calibrated/<cond>/<label>/...
      - trials_participants.csv at project root with one row per (TrialID, ParticipantID)
    '''
    # Ensure folder tree
    for cond in ("equal","weak","strong"):
        ensure_dirs(OUTPUT_DIR, cond)

    rng = np.random.RandomState(RANDOM_SEED if RANDOM_SEED is not None else None)

    # Template norm once (for metadata)
    tpl = gaussian_blob(IMG_SIZE, SIGMA_PX, amplitude=1.0)
    K = math.sqrt(np.sum(tpl**2))

    cond_table = {
        "equal":  {"contrast": PEAK_CONTRAST_EQUAL,  "target_snr": TARGET_SNRS["equal"]},
        "weak":   {"contrast": PEAK_CONTRAST_WEAK,   "target_snr": TARGET_SNRS["weak"]},
        "strong": {"contrast": PEAK_CONTRAST_STRONG, "target_snr": TARGET_SNRS["strong"]},
    }

    # metadata header
    meta_path = OUTPUT_DIR/"metadata.csv"
    if not meta_path.exists():
        with meta_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "trial_id","condition","label","contrast","target_snr",
                "mean_noise","sd_noise","rms_contrast","empirical_snr_template",
                "combined_png","noise_png","signal_png",
                "combined_npy","noise_npy","signal_npy"
            ])

    # participant schedule
    TRIALS_PARTICIPANTS = Path("trials_participants.csv")
    with TRIALS_PARTICIPANTS.open("w", newline="") as fparts:
        w = csv.writer(fparts)
        w.writerow(["TrialID","Block","ParticipantID","Truth","AssignedCondition","Image"])

        # ======= EQUAL BLOCK =======
        for i in range(1, N_EQUAL_TRIALS+1):
            noise_bg = white_noise(IMG_SIZE)
            is_present = bool(rng.rand() < P_SIGNAL_PRESENT)
            label = "present" if is_present else "absent"

            contrast = cond_table["equal"]["contrast"]; target_snr = cond_table["equal"]["target_snr"]
            emp_snr_template = (contrast * MEAN_LUMINANCE * K) / NOISE_SD
            comb, noi, sig = make_stimulus(present=is_present, peak_contrast=contrast, noise=noise_bg)
            save_trial(OUTPUT_DIR, "equal", label, i, comb, noi, sig)

            mean_n, sd_n = float(np.mean(noi)), float(np.std(noi, ddof=1))
            rms = sd_n / max(mean_n, 1e-9)
            with meta_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    i,"equal",label,contrast,target_snr,mean_n,sd_n,rms,emp_snr_template,
                    str(OUTPUT_DIR/"equal"/label/"combined"/f"equal_{label}_{i:03d}_combined.png"),
                    str(OUTPUT_DIR/"equal"/label/"noise"/f"equal_{label}_{i:03d}_noise.png"),
                    str(OUTPUT_DIR/"equal"/label/"signal"/f"equal_{label}_{i:03d}_signal.png"),
                    str(OUTPUT_DIR/"equal"/label/"combined"/f"equal_{label}_{i:03d}_combined.npy"),
                    str(OUTPUT_DIR/"equal"/label/"noise"/f"equal_{label}_{i:03d}_noise.npy"),
                    str(OUTPUT_DIR/"equal"/label/"signal"/f"equal_{label}_{i:03d}_signal.npy"),
                ])

            img_path = str(OUTPUT_DIR/"equal"/label/"combined"/f"equal_{label}_{i:03d}_combined.png")
            truth = 1 if is_present else 0
            for pid in PARTICIPANTS:
                w.writerow([i, "equal", pid, truth, "equal", img_path])

        # ======= MIXTURE BLOCK =======
        offset = N_EQUAL_TRIALS
        for j in range(1, N_MIX_TRIALS+1):
            trial_id = offset + j
            noise_bg = white_noise(IMG_SIZE)
            is_present = bool(rng.rand() < P_SIGNAL_PRESENT)
            label = "present" if is_present else "absent"

            if not is_present:
                # Save absent under both weak/strong for completeness (same noise)
                for cond_name in ("weak","strong"):
                    contrast = cond_table[cond_name]["contrast"]; target_snr = cond_table[cond_name]["target_snr"]
                    emp_snr_template = (contrast * MEAN_LUMINANCE * K) / NOISE_SD
                    comb, noi, sig = make_stimulus(present=False, peak_contrast=contrast, noise=noise_bg)
                    save_trial(OUTPUT_DIR, cond_name, "absent", trial_id, comb, noi, sig)

                    mean_n, sd_n = float(np.mean(noi)), float(np.std(noi, ddof=1))
                    rms = sd_n / max(mean_n, 1e-9)
                    with meta_path.open("a", newline="") as f:
                        csv.writer(f).writerow([
                            trial_id,cond_name,"absent",contrast,target_snr,mean_n,sd_n,rms,emp_snr_template,
                            str(OUTPUT_DIR/cond_name/"absent"/"combined"/f"{cond_name}_absent_{trial_id:03d}_combined.png"),
                            str(OUTPUT_DIR/cond_name/"absent"/"noise"/f"{cond_name}_absent_{trial_id:03d}_noise.png"),
                            str(OUTPUT_DIR/cond_name/"absent"/"signal"/f"{cond_name}_absent_{trial_id:03d}_signal.png"),
                            str(OUTPUT_DIR/cond_name/"absent"/"combined"/f"{cond_name}_absent_{trial_id:03d}_combined.npy"),
                            str(OUTPUT_DIR/cond_name/"absent"/"noise"/f"{cond_name}_absent_{trial_id:03d}_noise.npy"),
                            str(OUTPUT_DIR/cond_name/"absent"/"signal"/f"{cond_name}_absent_{trial_id:03d}_signal.npy"),
                        ])
                # Participants: all see the same absent image (use weak/absent path)
                img_path = str(OUTPUT_DIR/"weak"/"absent"/"combined"/f"weak_absent_{trial_id:03d}_combined.png")
                for pid in PARTICIPANTS:
                    w.writerow([trial_id, "mixture", pid, 0, "weak", img_path])

            else:
                # Present: random strong participant, others weak (same noise)
                strong_pid = int(rng.choice(PARTICIPANTS))
                weak_pids = [p for p in PARTICIPANTS if p != strong_pid]

                for cond_name in ("weak","strong"):
                    contrast = cond_table[cond_name]["contrast"]; target_snr = cond_table[cond_name]["target_snr"]
                    emp_snr_template = (contrast * MEAN_LUMINANCE * K) / NOISE_SD
                    comb, noi, sig = make_stimulus(present=True, peak_contrast=contrast, noise=noise_bg)
                    save_trial(OUTPUT_DIR, cond_name, "present", trial_id, comb, noi, sig)

                    mean_n, sd_n = float(np.mean(noi)), float(np.std(noi, ddof=1))
                    rms = sd_n / max(mean_n, 1e-9)
                    with meta_path.open("a", newline="") as f:
                        csv.writer(f).writerow([
                            trial_id,cond_name,"present",contrast,target_snr,mean_n,sd_n,rms,emp_snr_template,
                            str(OUTPUT_DIR/cond_name/"present"/"combined"/f"{cond_name}_present_{trial_id:03d}_combined.png"),
                            str(OUTPUT_DIR/cond_name/"present"/"noise"/f"{cond_name}_present_{trial_id:03d}_noise.png"),
                            str(OUTPUT_DIR/cond_name/"present"/"signal"/f"{cond_name}_present_{trial_id:03d}_signal.png"),
                            str(OUTPUT_DIR/cond_name/"present"/"combined"/f"{cond_name}_present_{trial_id:03d}_combined.npy"),
                            str(OUTPUT_DIR/cond_name/"present"/"noise"/f"{cond_name}_present_{trial_id:03d}_noise.npy"),
                            str(OUTPUT_DIR/cond_name/"present"/"signal"/f"{cond_name}_present_{trial_id:03d}_signal.npy"),
                        ])

                strong_img = str(OUTPUT_DIR/"strong"/"present"/"combined"/f"strong_present_{trial_id:03d}_combined.png")
                weak_img   = str(OUTPUT_DIR/"weak"/"present"/"combined"/f"weak_present_{trial_id:03d}_combined.png")
                # write participant rows
                w.writerow([trial_id, "mixture", strong_pid, 1, "strong", strong_img])
                for pid in weak_pids:
                    w.writerow([trial_id, "mixture", pid, 1, "weak", weak_img])

    print(f"[OK] Wrote participant schedule at {TRIALS_PARTICIPANTS.resolve()}")
    print(f"[OK] Images & metadata under {OUTPUT_DIR.resolve()}")


# %%

# === Run export ===
_ = export_trials_with_participants()


# %%

# Quick peek at the schedule
dfp = pd.read_csv("trials_participants.csv")
print(dfp.shape)
display(dfp.head(6))

# Count composition sanity
mix_pre = dfp[(dfp["Block"]=="mixture") & (dfp["Truth"]==1)]
if not mix_pre.empty:
    ct = (mix_pre.groupby(["TrialID","AssignedCondition"]).size().unstack(fill_value=0))
    print("First few mixture-present counts per TrialID (expect 1 strong, 2 weak):")
    display(ct.head())


# %% [markdown]
# 
# ## Run LLMRunner2 (DRY RUN) — define & execute locally
# 

# %%

import os, nbformat

# Ensure DRY_RUN
os.environ["DRY_RUN"] = "1"

runner_path = Path("LLMRunner2.ipynb")
assert runner_path.exists(), "LLMRunner2.ipynb not found — generate it first."

# Execute code cells from LLMRunner2 in *this* kernel
nb_runner = nbformat.read(runner_path.open("r", encoding="utf-8"), as_version=4)
for cell in nb_runner.cells:
    if cell.cell_type == "code":
        code = cell.source
        if code.strip():
            exec(compile(code, "<LLMRunner2_cell>", "exec"), globals())

# Now use its API
from pathlib import Path as _Path

dfp2 = load_trials_participants(_Path("trials_participants.csv"))
p1 = run_phase1(dfp2, limit_trials=5)  # small smoke test
p2 = run_phase2(p1)

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "API_files"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PHASE1_CSV = RESULTS_DIR / "phase1_results.csv"
PHASE2_CSV = RESULTS_DIR / "phase2_results.csv"
CONSENSUS_CSV = RESULTS_DIR / "phase2_consensus.csv"

p1.to_csv(PHASE1_CSV, index=False)
p2.to_csv(PHASE2_CSV, index=False)
consensus = consensus_from_phase2(p2)
consensus.to_csv(CONSENSUS_CSV, index=False)
print(f"[OK] DRY_RUN wrote: {PHASE1_CSV}, {PHASE2_CSV}, {CONSENSUS_CSV}")
consensus.head()


# %% [markdown]
# 
# ## Execute Preflight Tests (from Preflight_Tests.ipynb)
# 

# %%

import nbformat

pre_path = Path("Preflight_Tests.ipynb")
assert pre_path.exists(), "Preflight_Tests.ipynb not found — generate it first."

nb_pre = nbformat.read(pre_path.open("r", encoding="utf-8"), as_version=4)
for cell in nb_pre.cells:
    if cell.cell_type == "code" and cell.source.strip():
        exec(compile(cell.source, "<Preflight_Tests_cell>", "exec"), globals())

print("[OK] Preflight tests executed.")



