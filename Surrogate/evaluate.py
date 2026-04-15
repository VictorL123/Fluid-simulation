"""
surrogate/evaluate.py
======================
Evaluates the trained surrogate model against the ground truth C++ solver.

Produces:
    1. Accuracy metrics — RMS error per field (u, v, p) on validation set
    2. Speed comparison — surrogate inference vs solver runtime
    3. Visual comparison — side-by-side plots of predicted vs true fields

Usage:
    python evaluate.py
    python evaluate.py --show        # show plots interactively
    python evaluate.py --nu 0.01     # evaluate a specific nu value

Requirements:
    pip install torch numpy matplotlib pandas
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import UNet, CavityDataset, count_parameters

# ============================================================
#  Paths
# ============================================================
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "eval_output")


# ============================================================
#  Load trained model
# ============================================================
def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']
    model = UNet(in_channels=1, out_channels=3,
                 base_features=args.get('base_feat', 32)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[eval] Loaded model from epoch {checkpoint['epoch']}")
    print(f"[eval] Validation loss at checkpoint: {checkpoint['val_loss']:.5f}")
    return model


# ============================================================
#  Compute RMS error on validation set
# ============================================================
def compute_accuracy(model, val_dataset, device):
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    u_errors, v_errors, p_errors = [], [], []

    with torch.no_grad():
        for x, y_true in loader:
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)

            # Denormalise both pred and true back to physical units
            y_pred_phys = val_dataset.denormalise_output(y_pred)
            y_true_phys = val_dataset.denormalise_output(y_true)

            # RMS error per field
            u_err = torch.sqrt(torch.mean((y_pred_phys[:, 0] - y_true_phys[:, 0])**2))
            v_err = torch.sqrt(torch.mean((y_pred_phys[:, 1] - y_true_phys[:, 1])**2))
            p_err = torch.sqrt(torch.mean((y_pred_phys[:, 2] - y_true_phys[:, 2])**2))

            u_errors.append(u_err.item())
            v_errors.append(v_err.item())
            p_errors.append(p_err.item())

    print(f"\n[eval] === Accuracy on validation set ===")
    print(f"[eval] u-field RMS error: {np.mean(u_errors):.5f} ± {np.std(u_errors):.5f}")
    print(f"[eval] v-field RMS error: {np.mean(v_errors):.5f} ± {np.std(v_errors):.5f}")
    print(f"[eval] p-field RMS error: {np.mean(p_errors):.5f} ± {np.std(p_errors):.5f}")

    return np.mean(u_errors), np.mean(v_errors), np.mean(p_errors)


# ============================================================
#  Speed benchmark — surrogate vs solver
# ============================================================
def benchmark_speed(model, device, n_runs=100, grid_size=41):
    """
    Compare surrogate inference time vs approximate solver time.
    Solver time is estimated from generate_data.py runtime data.
    """
    x = torch.randn(1, 1, grid_size, grid_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # Measure surrogate inference
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    surrogate_time = (time.perf_counter() - t0) / n_runs * 1000  # ms

    print(f"\n[eval] === Speed Comparison ===")
    print(f"[eval] Surrogate inference:  {surrogate_time:.3f} ms per sample")
    print(f"[eval] C++ solver (typical): ~24,000 ms (15000 steps at Re=100)")
    speedup = 24000 / surrogate_time
    print(f"[eval] Speedup:              ~{speedup:.0f}x")
    print(f"[eval] Note: solver time varies with Re and grid size")

    return surrogate_time


# ============================================================
#  Visual comparison — predicted vs true fields
# ============================================================
def plot_comparison(model, val_dataset, device, n_examples=3, save=True):
    loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    field_names = ['u-velocity', 'v-velocity', 'pressure']
    cmaps       = ['RdBu_r', 'RdBu_r', 'viridis']

    for example_idx, (x, y_true) in enumerate(loader):
        if example_idx >= n_examples:
            break

        with torch.no_grad():
            x_dev    = x.to(device)
            y_pred   = model(x_dev)
            y_pred_p = val_dataset.denormalise_output(y_pred).cpu()
            y_true_p = val_dataset.denormalise_output(y_true.to(device)).cpu()

        # Get nu value for this sample
        nu_norm = x[0, 0, 0, 0].item()
        nu_phys = nu_norm * val_dataset.nu_std + val_dataset.nu_mean

        fig, axes = plt.subplots(3, 3, figsize=(13, 10))
        fig.suptitle(f"Surrogate vs Solver  —  nu={nu_phys:.4f}  Re={1/nu_phys:.1f}",
                     fontsize=12)

        for field_idx in range(3):
            true_field = y_true_p[0, field_idx].numpy()
            pred_field = y_pred_p[0, field_idx].numpy()
            err_field  = np.abs(pred_field - true_field)

            vmin = min(true_field.min(), pred_field.min())
            vmax = max(true_field.max(), pred_field.max())

            # True
            im = axes[field_idx, 0].imshow(true_field, cmap=cmaps[field_idx],
                                            vmin=vmin, vmax=vmax, origin='lower')
            axes[field_idx, 0].set_title(f"Solver: {field_names[field_idx]}")
            plt.colorbar(im, ax=axes[field_idx, 0])

            # Predicted
            im = axes[field_idx, 1].imshow(pred_field, cmap=cmaps[field_idx],
                                            vmin=vmin, vmax=vmax, origin='lower')
            axes[field_idx, 1].set_title(f"Surrogate: {field_names[field_idx]}")
            plt.colorbar(im, ax=axes[field_idx, 1])

            # Error
            im = axes[field_idx, 2].imshow(err_field, cmap='hot', origin='lower')
            axes[field_idx, 2].set_title(f"Abs error (RMS={err_field.mean():.4f})")
            plt.colorbar(im, ax=axes[field_idx, 2])

        plt.tight_layout()

        if save:
            path = os.path.join(OUTPUT_DIR, f"comparison_{example_idx:02d}.png")
            plt.savefig(path, dpi=120, bbox_inches='tight')
            print(f"[eval] Saved: {path}")
        else:
            plt.show()

        plt.close()


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained surrogate model")
    parser.add_argument("--show",       action="store_true", help="Show plots interactively")
    parser.add_argument("--n_examples", type=int, default=3, help="Number of visual comparisons")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(CHECKPOINT_DIR, "best_model.pt"),
                        help="Path to model checkpoint")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"[eval] ERROR: No checkpoint found at {args.checkpoint}")
        print("[eval] Run train.py first.")
        exit(1)

    model = load_model(args.checkpoint, device)

    # Load validation dataset
    val_dataset = CavityDataset(DATA_DIR, split="val")

    # Run evaluations
    compute_accuracy(model, val_dataset, device)
    benchmark_speed(model, device)
    plot_comparison(model, val_dataset, device,
                    n_examples=args.n_examples,
                    save=not args.show)

    print(f"\n[eval] All outputs saved to {OUTPUT_DIR}")
