"""
validate/plot.py
================
Visualise and validate the Lid-Driven Cavity simulation output.

Usage:
    python plot.py                  # plot the last CSV in ../solver/output/
    python plot.py --step 5000      # plot a specific step
    python plot.py --all            # animate all steps (saves anim.gif)
    python plot.py --validate       # compare against Ghia et al. benchmark

Requirements:
    pip install numpy matplotlib pandas
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ============================================================
#  Ghia et al. (1982) benchmark data  —  Re = 100
#  u-velocity along vertical centreline (x = 0.5)
#  y values from 0 (bottom) to 1 (top)
# ============================================================
GHIA_RE100_Y = [
    0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813,
    0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609,
    0.9688, 0.9766, 1.0000
]
GHIA_RE100_U = [
    0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
    -0.21090, -0.20581, -0.13641,  0.00332,  0.23151,  0.68717,  0.73722,
     0.78871,  0.84123,  1.00000
]


# ============================================================
#  Load a CSV output file from the solver
# ============================================================
def load_csv(filepath):
    df = pd.read_csv(filepath)
    N = int(df['i'].max()) + 1
    u = df['u'].values.reshape(N, N)
    v = df['v'].values.reshape(N, N)
    p = df['p'].values.reshape(N, N)
    speed = df['speed'].values.reshape(N, N)
    return N, u, v, p, speed


def get_latest_csv(output_dir):
    files = sorted(glob.glob(os.path.join(output_dir, "step_*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {output_dir}")
    return files[-1]


def get_csv_at_step(output_dir, step):
    path = os.path.join(output_dir, f"step_{step:05d}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No file found: {path}")
    return path


# ============================================================
#  Plot: velocity field + pressure + speed heatmap
# ============================================================
def plot_fields(filepath, save=True):
    N, u, v, p, speed = load_csv(filepath)
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    step_label = os.path.basename(filepath).replace(".csv", "")
    fig.suptitle(f"Lid-Driven Cavity  —  {step_label}", fontsize=13)

    # --- Speed heatmap with streamlines ---
    ax = axes[0]
    im = ax.contourf(X, Y, speed, levels=50, cmap='plasma')
    ax.streamplot(x, y, u, v, color='white', linewidth=0.6, density=1.2, arrowsize=0.8)
    plt.colorbar(im, ax=ax, label='Speed |u|')
    ax.set_title("Velocity field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

    # --- Pressure field ---
    ax = axes[1]
    im = ax.contourf(X, Y, p, levels=50, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='Pressure')
    ax.set_title("Pressure")
    ax.set_xlabel("x")
    ax.set_aspect('equal')

    # --- U-velocity vertical profile at x=0.5 ---
    ax = axes[2]
    mid_j = N // 2
    u_profile = u[:, mid_j]
    y_vals = np.linspace(0, 1, N)
    ax.plot(u_profile, y_vals, 'b-o', markersize=3, label='Simulation')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_title("u-velocity at x=0.5")
    ax.set_xlabel("u")
    ax.set_ylabel("y")
    ax.legend()

    plt.tight_layout()

    if save:
        out = filepath.replace(".csv", "_fields.png")
        plt.savefig(out, dpi=120, bbox_inches='tight')
        print(f"[plot] Saved: {out}")
    else:
        plt.show()


# ============================================================
#  Validate against Ghia et al. (1982) Re=100 benchmark
# ============================================================
def validate_ghia(filepath):
    N, u, v, p, speed = load_csv(filepath)
    y_vals = np.linspace(0, 1, N)

    mid_j = N // 2
    u_profile = u[:, mid_j]

    fig, ax = plt.subplots(figsize=(6, 7))

    ax.plot(u_profile, y_vals, 'b-', linewidth=2, label='Simulation')
    ax.scatter(GHIA_RE100_U, GHIA_RE100_Y,
               color='red', s=40, zorder=5, label='Ghia et al. (1982) Re=100')

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0.0, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlabel("u-velocity")
    ax.set_ylabel("y")
    ax.set_title("Validation: u-velocity along vertical centreline (x=0.5)\nRe=100")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = os.path.join(os.path.dirname(filepath), "validation_ghia.png")
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f"[validate] Saved: {out}")

    # Simple RMS error against interpolated Ghia data
    ghia_u_interp = np.interp(y_vals, GHIA_RE100_Y, GHIA_RE100_U)
    rms_error = np.sqrt(np.mean((u_profile - ghia_u_interp)**2))
    print(f"[validate] RMS error vs Ghia benchmark: {rms_error:.6f}")
    if rms_error < 0.02:
        print("[validate] PASS — good agreement with benchmark")
    else:
        print("[validate] WARN — error is high. Run more steps or reduce dt.")


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../solver/output")

    parser = argparse.ArgumentParser(description="Plot and validate cavity solver output")
    parser.add_argument("--step",     type=int,  default=None,  help="Plot specific step number")
    parser.add_argument("--validate", action="store_true",      help="Compare against Ghia benchmark")
    parser.add_argument("--show",     action="store_true",      help="Show plots interactively instead of saving")
    args = parser.parse_args()

    # Resolve which file to use
    if args.step is not None:
        filepath = get_csv_at_step(OUTPUT_DIR, args.step)
    else:
        filepath = get_latest_csv(OUTPUT_DIR)

    print(f"[plot] Using: {filepath}")

    if args.validate:
        validate_ghia(filepath)
    else:
        plot_fields(filepath, save=not args.show)
