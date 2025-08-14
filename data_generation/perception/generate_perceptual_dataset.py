"""
Perceptual-task dataset generator (Experiment 1 replica, image-only)

- Outputs 8-bit grayscale PNG images + separate noise fields (.npy).
- Implements equal-strength and mixture conditions.
- Uses paper geometry: 0.037°/px, 15° noise patch, Gaussian signal (σ=0.5°).
- Noise RMS contrast ~0.1562 of mean luminance (scaled to 8-bit base µ=128).
- Identical noise across the three observers within a trial.

Folder layout:
  out_root/
    equal/
      trial_0001/
        noise.npy                  # (H, W) float32 noise field
        obs1.png                   # final image (mean + noise + signal_1)
        obs2.png
        obs3.png
    mixture/
      trial_0001/
        noise.npy
        obs1.png
        obs2.png
        obs3.png
    metadata.csv

Usage:
  python generate_perceptual_dataset.py --out ./perceptual_out \
      --trials_equal 200 --trials_mixture 200 --seed 42
"""

import argparse
import csv
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image

# --------------------------
# Defaults grounded in paper
# --------------------------
PIX_PER_DEG = 1.0 / 0.037          # ≈ 27.027 px/deg (paper)
PATCH_DEG = 15.0                    # 15° x 15° patch (paper)
IMG_SIZE = int(round(PATCH_DEG * PIX_PER_DEG))  # ≈ 405 px

SIGMA_DEG = 0.5                     # signal Gaussian σ in degrees (paper)
SIGMA_PX = SIGMA_DEG * PIX_PER_DEG  # ≈ 13.5 px

MEAN_LUM = 128.0                    # 8-bit midpoint as "mean luminance"
NOISE_RMS_CONTRAST = 0.1562         # paper RMS contrast
NOISE_STD = MEAN_LUM * NOISE_RMS_CONTRAST  # apply in 8-bit space (~20)

# Peak signal contrasts (exposed so you can adjust if needed)
CONTRAST_EQUAL = 0.02               # 2%
CONTRAST_WEAK  = 0.005              # 0.5%
CONTRAST_STRONG= 0.09               # 9%

# Observers per trial
N_OBS = 3


def make_centered_gaussian(h, w, sigma_px, peak_delta):
    """
    Create a centered 2D Gaussian with peak value = peak_delta (additive luminance).
    """
    y = np.arange(h, dtype=np.float32)
    x = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    g = np.exp(-0.5 * r2 / (sigma_px ** 2))
    g *= peak_delta  # scale so center pixel has 'peak_delta'
    return g.astype(np.float32)


def clamp_to_uint8(arr):
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_png(path, arr):
    img = Image.fromarray(arr, mode='L')
    img.save(path, compress_level=0)


def generate_trial_noise(h, w, rng):
    # zero-mean Gaussian noise with std=NOISE_STD, later added to MEAN_LUM
    noise = rng.normal(loc=0.0, scale=NOISE_STD, size=(h, w)).astype(np.float32)
    return noise


def gen_equal_condition_images(rng, signal_present):
    noise = generate_trial_noise(IMG_SIZE, IMG_SIZE, rng)
    if signal_present:
        peak_delta = MEAN_LUM * CONTRAST_EQUAL
        signal_field = make_centered_gaussian(IMG_SIZE, IMG_SIZE, SIGMA_PX, peak_delta)
        contrasts = [CONTRAST_EQUAL] * N_OBS
    else:
        signal_field = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        contrasts = [0.0] * N_OBS

    images = []
    for _ in range(N_OBS):
        img = MEAN_LUM + noise + signal_field
        images.append(clamp_to_uint8(img))
    return images, noise, contrasts, -1


def gen_mixture_condition_images(rng, signal_present):
    """
    Returns list of 3 observer images for mixture condition + the (shared) noise.
    If signal is present: one random observer gets STRONG, two get WEAK.
    If absent: everyone gets no signal.
    """
    noise = generate_trial_noise(IMG_SIZE, IMG_SIZE, rng)

    if not signal_present:
        signal_fields = [np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32) for _ in range(N_OBS)]
        contrasts = [0.0] * N_OBS
        strong_idx = -1
    else:
        strong_idx = rng.integers(low=0, high=N_OBS)
        contrasts = []
        signal_fields = []
        for i in range(N_OBS):
            c = CONTRAST_STRONG if i == strong_idx else CONTRAST_WEAK
            contrasts.append(c)
            peak_delta = MEAN_LUM * c
            signal_fields.append(make_centered_gaussian(IMG_SIZE, IMG_SIZE, SIGMA_PX, peak_delta))

    images = []
    for i in range(N_OBS):
        img = MEAN_LUM + noise + signal_fields[i]
        images.append(clamp_to_uint8(img))

    return images, noise, contrasts, strong_idx


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output root directory")
    ap.add_argument("--trials_equal", type=int, default=200, help="# trials in equal-strength block")
    ap.add_argument("--trials_mixture", type=int, default=200, help="# trials in mixture block")
    ap.add_argument("--p_signal", type=float, default=0.5, help="P(signal present)")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    ap.add_argument("--prefix_equal", type=str, default="equal")
    ap.add_argument("--prefix_mixture", type=str, default="mixture")
    args = ap.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)

    # Metadata CSV
    meta_path = out_root / "metadata.csv"
    fields = [
        "condition", "trial_index", "signal_present",
        "strong_observer_index", "contrast_obs1", "contrast_obs2", "contrast_obs3",
        "noise_npy", "obs1_png", "obs2_png", "obs3_png",
        "seed"
    ]
    fmeta = open(meta_path, "w", newline="")
    writer = csv.DictWriter(fmeta, fieldnames=fields)
    writer.writeheader()

    # Global RNG for reproducibility
    rng_global = np.random.default_rng(args.seed)

    # ------------------------
    # Equal-strength condition
    # ------------------------
    equal_dir = out_root / args.prefix_equal
    ensure_dir(equal_dir)
    for t in range(1, args.trials_equal + 1):
        trial_dir = equal_dir / f"trial_{t:04d}"
        ensure_dir(trial_dir)

        # per-trial RNG seed (recorded)
        trial_seed = int(rng_global.integers(0, 2**31 - 1))
        rng = np.random.default_rng(trial_seed)

        signal_present = rng.random() < args.p_signal
        imgs, noise, contrasts, _ = gen_equal_condition_images(rng, signal_present)

        # Save noise separately
        noise_path = trial_dir / "noise.npy"
        np.save(noise_path, noise.astype(np.float32))

        # Save observer images
        obs_paths = []
        for i, img in enumerate(imgs, start=1):
            p = trial_dir / f"obs{i}.png"
            save_png(p, img)
            obs_paths.append(p)

        writer.writerow({
            "condition": "equal",
            "trial_index": t,
            "signal_present": int(signal_present),
            "strong_observer_index": -1,
            "contrast_obs1": contrasts[0],
            "contrast_obs2": contrasts[1],
            "contrast_obs3": contrasts[2],
            "noise_npy": str(noise_path.relative_to(out_root)),
            "obs1_png": str(obs_paths[0].relative_to(out_root)),
            "obs2_png": str(obs_paths[1].relative_to(out_root)),
            "obs3_png": str(obs_paths[2].relative_to(out_root)),
            "seed": trial_seed
        })

    # -------------
    # Mixture block
    # -------------
    mix_dir = out_root / args.prefix_mixture
    ensure_dir(mix_dir)
    for t in range(1, args.trials_mixture + 1):
        trial_dir = mix_dir / f"trial_{t:04d}"
        ensure_dir(trial_dir)

        # per-trial RNG seed (recorded)
        trial_seed = int(rng_global.integers(0, 2**31 - 1))
        rng = np.random.default_rng(trial_seed)

        signal_present = rng.random() < args.p_signal
        imgs, noise, contrasts, strong_idx = gen_mixture_condition_images(rng, signal_present)

        # Save noise separately
        noise_path = trial_dir / "noise.npy"
        np.save(noise_path, noise.astype(np.float32))

        # Save observer images
        obs_paths = []
        for i, img in enumerate(imgs, start=1):
            p = trial_dir / f"obs{i}.png"
            save_png(p, img)
            obs_paths.append(p)

        writer.writerow({
            "condition": "mixture",
            "trial_index": t,
            "signal_present": int(signal_present),
            "strong_observer_index": int(strong_idx),
            "contrast_obs1": contrasts[0],
            "contrast_obs2": contrasts[1],
            "contrast_obs3": contrasts[2],
            "noise_npy": str(noise_path.relative_to(out_root)),
            "obs1_png": str(obs_paths[0].relative_to(out_root)),
            "obs2_png": str(obs_paths[1].relative_to(out_root)),
            "obs3_png": str(obs_paths[2].relative_to(out_root)),
            "seed": trial_seed
        })

    fmeta.close()
    print(f"Done. Wrote dataset to: {out_root}")


if __name__ == "__main__":
    main()
