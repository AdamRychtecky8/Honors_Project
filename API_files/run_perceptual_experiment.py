import argparse, os, sys, csv, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from perceptual_stimulus_pixels import make_stimulus, MEAN_LUMINANCE, PEAK_CONTRAST_WEAK, PEAK_CONTRAST_EQUAL, PEAK_CONTRAST_STRONG

def run_experiment(condition="equal", n_trials=20, out_dir="results", save_images=False):
    contrasts = {
        "weak": PEAK_CONTRAST_WEAK,
        "equal": PEAK_CONTRAST_EQUAL,
        "strong": PEAK_CONTRAST_STRONG
    }
    contrast = contrasts[condition]

    os.makedirs(out_dir, exist_ok=True)
    results_file = os.path.join(out_dir, "responses.csv")
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial","condition","ground_truth","response","correct"])

        print(f"Starting experiment: {condition} condition, {n_trials} trials")
        print("Press 'j' for PRESENT, 'f' for ABSENT. Close the window to move on.")

        for t in range(1, n_trials+1):
            present = bool(np.random.randint(0,2))
            combined, noise, signal = make_stimulus(present=present, peak_contrast=contrast)
            img8 = np.clip(combined, 0, 255).astype(np.uint8)

            plt.imshow(img8, cmap="gray", vmin=0, vmax=255)
            plt.axis("off")
            plt.title(f"Trial {t}/{n_trials}")
            plt.show(block=False)

            # Wait for user key press
            response = None
            while response not in ["j","f"]:
                response = input("Your response (j=PRESENT, f=ABSENT): ").strip().lower()

            correct = ((response=="j" and present) or (response=="f" and not present))
            writer.writerow([t, condition, "present" if present else "absent",
                             response, int(correct)])

            if save_images:
                from PIL import Image
                fname = os.path.join(out_dir, f"{condition}_trial{t:03d}_{'present' if present else 'absent'}.png")
                Image.fromarray(img8).save(fname)

            print(f"Trial {t} complete. Correct: {correct}\n")

    print(f"Experiment finished. Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run perceptual detection experiment (pixel-based).")
    parser.add_argument("--condition", choices=["weak","equal","strong"], default="equal")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--save_images", action="store_true")
    args = parser.parse_args()

    run_experiment(condition=args.condition, n_trials=args.n_trials,
                   out_dir=args.out_dir, save_images=args.save_images)
