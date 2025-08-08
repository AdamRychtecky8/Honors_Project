"""
Generate sandstorm-task datasets (equal & mixture) with the exact CSV format
expected by the modeling pipeline:
- label at column 0
- condition, strong_comp, trial_id at columns 1–3
- Participant 1 features at columns 4–6
- Participant 2 features at columns 7–9
- Participant 3 features at columns 10–12

Outputs:
- participant_data_equal_condition.csv
- participant_data_mixture_condition.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# --- choose where you want files to go ---
# Option B (script’s folder): Path(__file__).parent / "data" / "generated"
# OUTPUT_DIR = Path(__file__).parent / "data" / "generated"

# Fixed output directory
OUTPUT_DIR = Path("/workspaces/Honors_Project/data_generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Make sure the folder exists

# File paths
equal_path   = OUTPUT_DIR / "participant_data_equal_condition.csv"
mixture_path = OUTPUT_DIR / "participant_data_mixture_condition.csv"

# =========================
# CONFIGURATION
# =========================
N_TRIALS_PER_CONDITION = 5000  # total trials per condition (half noise, half signal)
N_MEASUREMENTS = 3             # triplet measurements per stimulus
RHO_EXPERIMENT = 0.2           # across-computer correlation during experimental phase

# Signal-to-noise ratios (to set mean shifts relative to noise)
SNRs = {
    "weak": 0.29,
    "medium": 0.78,
    "strong": 3.46
}

# Reproducibility
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# =========================
# PARTICIPANT PARAMETER GENERATION
# =========================
def generate_participant_params(rng_obj):
    """
    For each 'computer' (Comp1..Comp3), sample mu_noise and derive
    mu_weak/medium/strong from SNR definitions and a shared sigma per computer.
    """
    mu_noise = {
        'Comp1': rng_obj.uniform(260, 290),
        'Comp2': rng_obj.uniform(350, 380),
        'Comp3': rng_obj.uniform(440, 470)
    }

    params = {}
    for comp, noise_mu in mu_noise.items():
        # Define medium as 15% above noise, then back out sigma from SNR = (mu_signal - mu_noise)/sigma
        mu_medium = noise_mu * 1.15
        sigma = (mu_medium - noise_mu) / SNRs["medium"]

        params[comp] = {
            "sigma": float(sigma),
            "mu_noise": float(noise_mu),
            "mu_weak": float(noise_mu + SNRs["weak"] * sigma),
            "mu_medium": float(mu_medium),
            "mu_strong": float(noise_mu + SNRs["strong"] * sigma),
        }
    return params

PARTICIPANT_PARAMS = generate_participant_params(rng)

# =========================
# CORRELATED GAUSSIAN SAMPLER
# =========================
def generate_correlated_gaussian(mu_vec, sigma_vec, rho, n_measurements, rng_obj):
    """
    Draw n_measurements samples from a 3D Gaussian whose dimensions correspond
    to Comp1/Comp2/Comp3. The covariance couples computers within a single
    measurement index; samples are independent across indices.
    Returns array of shape (3, n_measurements).
    """
    mu_vec = np.asarray(mu_vec, dtype=float)           # length 3
    sigma_vec = np.asarray(sigma_vec, dtype=float)     # length 3

    # Correlation matrix across computers
    corr = rho * np.ones((3, 3)) + (1 - rho) * np.eye(3)
    # Covariance = D_sigma * Corr * D_sigma
    D = np.diag(sigma_vec)
    cov = D @ corr @ D

    samples = rng_obj.multivariate_normal(mean=mu_vec, cov=cov, size=n_measurements)
    # transpose to shape (3, n_measurements): rows -> Comp1/2/3, columns -> feat1/2/3
    return samples.T

# =========================
# TRIAL GENERATION
# =========================
def generate_trials(condition: str, label: int, n_trials: int, rng_obj, rho=RHO_EXPERIMENT):
    """
    Generate n_trials for a given condition ('equal' or 'mixture') and label (0 noise, 1 signal).
    Returns a list of row dicts ready for DataFrame construction.
    """
    trials = []
    comps = ['Comp1', 'Comp2', 'Comp3']

    for t in range(n_trials):
        mus, sigmas = [], []
        strong_comp = None

        if condition == 'equal':
            # All three computers have medium mean in signal, noise mean otherwise.
            for comp in comps:
                mu = PARTICIPANT_PARAMS[comp]["mu_medium"] if label == 1 else PARTICIPANT_PARAMS[comp]["mu_noise"]
                sigma = PARTICIPANT_PARAMS[comp]["sigma"]
                mus.append(mu)
                sigmas.append(sigma)

        elif condition == 'mixture':
            if label == 1:
                # One computer strong, two weak
                strong_idx = rng_obj.integers(0, 3)
                strong_comp = comps[strong_idx]
                for i, comp in enumerate(comps):
                    key = "mu_strong" if i == strong_idx else "mu_weak"
                    mu = PARTICIPANT_PARAMS[comp][key]
                    sigma = PARTICIPANT_PARAMS[comp]["sigma"]
                    mus.append(mu)
                    sigmas.append(sigma)
            else:
                # Noise trial
                for comp in comps:
                    mu = PARTICIPANT_PARAMS[comp]["mu_noise"]
                    sigma = PARTICIPANT_PARAMS[comp]["sigma"]
                    mus.append(mu)
                    sigmas.append(sigma)
        else:
            raise ValueError("condition must be 'equal' or 'mixture'")

        # Draw triplet (feat1..feat3) per computer with across-computer correlation
        measurements = generate_correlated_gaussian(
            mu_vec=mus,
            sigma_vec=sigmas,
            rho=rho,
            n_measurements=N_MEASUREMENTS,
            rng_obj=rng_obj
        )
        # Round to ints to mirror original task format
        measurements = np.clip(measurements, 100, 999)  # or (117, 947) to mimic realized bounds
        measurements = np.round(measurements).astype(int)

        # Build row with the exact column layout expected by your modeling code
        row = {
            # --- non-feature columns ---
            "label": int(label),                 # col 0
            "condition": condition,              # col 1
            "strong_comp": strong_comp,          # col 2 (None for non-signal or equal)
            "trial_id": t,                       # col 3 (padding so features begin at col 4)
            # --- Participant 1 features (cols 4–6) ---
            "comp1_feat1": int(measurements[0, 0]),
            "comp1_feat2": int(measurements[0, 1]),
            "comp1_feat3": int(measurements[0, 2]),
            # --- Participant 2 features (cols 7–9) ---
            "comp2_feat1": int(measurements[1, 0]),
            "comp2_feat2": int(measurements[1, 1]),
            "comp2_feat3": int(measurements[1, 2]),
            # --- Participant 3 features (cols 10–12) ---
            "comp3_feat1": int(measurements[2, 0]),
            "comp3_feat2": int(measurements[2, 1]),
            "comp3_feat3": int(measurements[2, 2]),
        }

        trials.append(row)

    return trials

# =========================
# DATASET BUILDERS
# =========================
def build_condition_df(condition: str, n_total_trials: int, rng_obj) -> pd.DataFrame:
    """
    Build DataFrame for a condition with 50/50 class balance.
    """
    n_each = n_total_trials // 2
    rows = []
    rows += generate_trials(condition, label=0, n_trials=n_each, rng_obj=rng_obj)
    rows += generate_trials(condition, label=1, n_trials=n_each, rng_obj=rng_obj)

    df = pd.DataFrame(rows)

    # Ensure the exact column order your modeling code expects
    ordered_cols = [
        "label", "condition", "strong_comp", "trial_id",
        "comp1_feat1", "comp1_feat2", "comp1_feat3",
        "comp2_feat1", "comp2_feat2", "comp2_feat3",
        "comp3_feat1", "comp3_feat2", "comp3_feat3",
    ]
    df = df[ordered_cols].reset_index(drop=True)
    return df

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    equal_df   = build_condition_df("equal",   N_TRIALS_PER_CONDITION, rng)
    mixture_df = build_condition_df("mixture", N_TRIALS_PER_CONDITION, rng)

    # Save CSVs to /workspaces/Honors_Project/data_generation
    equal_df.to_csv(equal_path, index=False)
    mixture_df.to_csv(mixture_path, index=False)

    # Quick sanity checks
    print("Equal condition shape:", equal_df.shape)
    print("Mixture condition shape:", mixture_df.shape)
    print("\nSaved equal condition data to:", equal_path)
    print("Saved mixture condition data to:", mixture_path)

