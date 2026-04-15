"""
surrogate/generate_data.py
===========================
Generates training data for the neural surrogate model by running
the C++ solver across a sweep of viscosity (nu) values.

FAST VERSION: builds the solver ONCE then passes nu as a command
line argument to each run. No recompilation between samples.

Each run produces one training sample:
    Input:  nu (viscosity scalar)
    Output: u, v, p fields at steady state (3 x N x N array)

Usage:
    python generate_data.py
    python generate_data.py --n_samples 150 --steps 15000
    python generate_data.py --dry_run     # preview without running

Output:
    surrogate/data/sample_XXXX.npz   <- one file per run
    surrogate/data/dataset_info.json <- metadata

Requirements:
    pip install numpy pandas
"""

import argparse
import os
import json
import subprocess
import glob
import time
import numpy as np
import pandas as pd

# ============================================================
#  Paths
# ============================================================
SOLVER_DIR = os.path.join(os.path.dirname(__file__), "../solver")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

# Find solver binary — try OMP version first, fall back to serial
def find_binary():
    candidates = [
        os.path.join(SOLVER_DIR, "fluid_sim_omp.exe"),  # Windows OMP
        os.path.join(SOLVER_DIR, "fluid_sim_omp"),      # Linux OMP
        os.path.join(SOLVER_DIR, "fluid_sim.exe"),      # Windows serial
        os.path.join(SOLVER_DIR, "fluid_sim"),          # Linux serial
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ============================================================
#  Build solver once
# ============================================================
def build_solver():
    print("[generate] Building solver (once only)...")
    result = subprocess.run(
        ["make", "omp"],
        capture_output=True, cwd=SOLVER_DIR
    )
    if result.returncode != 0:
        # Try serial build if OMP fails
        result = subprocess.run(
            ["make"],
            capture_output=True, cwd=SOLVER_DIR
        )
    if result.returncode != 0:
        print(f"[generate] ERROR: Build failed:\n{result.stderr.decode()}")
        exit(1)
    print("[generate] Build successful.\n")


# ============================================================
#  Run solver with a specific nu value
#  No recompilation — just passes nu as argv[1]
# ============================================================
def run_solver(binary, nu, n_steps, output_every, sample_idx):
    result = subprocess.run(
        [binary, str(nu), str(n_steps), str(output_every)],
        capture_output=True,
        cwd=SOLVER_DIR
    )
    if result.returncode != 0:
        print(f"  [SKIP] Solver failed for nu={nu:.5f} — likely unstable")
        return None

    # Find the final output CSV
    output_files = sorted(glob.glob(os.path.join(SOLVER_DIR, "output", "step_*.csv")))
    if not output_files:
        print(f"  [SKIP] No output files found for nu={nu:.5f}")
        return None

    final_csv = output_files[-1]
    try:
        df = pd.read_csv(final_csv)
        N  = int(df['i'].max()) + 1
        u  = df['u'].values.reshape(N, N).astype(np.float32)
        v  = df['v'].values.reshape(N, N).astype(np.float32)
        p  = df['p'].values.reshape(N, N).astype(np.float32)
        return u, v, p
    except Exception as e:
        print(f"  [SKIP] Failed to read CSV: {e}")
        return None


# ============================================================
#  Save a single sample
# ============================================================
def save_sample(sample_idx, nu, u, v, p):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"sample_{sample_idx:04d}.npz")
    np.savez_compressed(path,
                        nu=np.float32(nu),
                        re=np.float32(1.0 / nu),
                        u=u, v=v, p=p)
    return path


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate surrogate training data")
    parser.add_argument("--n_samples",    type=int,   default=150)
    parser.add_argument("--steps",        type=int,   default=15000)
    parser.add_argument("--output_every", type=int,   default=15000,
                        help="Write CSV every N steps (default: only final step)")
    parser.add_argument("--nu_min",       type=float, default=0.005)
    parser.add_argument("--nu_max",       type=float, default=0.1)
    parser.add_argument("--dry_run",      action="store_true")
    parser.add_argument("--skip_build",   action="store_true",
                        help="Skip build step if solver already compiled")
    args = parser.parse_args()

    # Sample nu values log-uniformly
    nu_values = np.exp(np.linspace(np.log(args.nu_min),
                                   np.log(args.nu_max),
                                   args.n_samples))

    print(f"[generate] {args.n_samples} samples")
    print(f"[generate] Nu:    {args.nu_min:.4f} → {args.nu_max:.4f}")
    print(f"[generate] Re:    {1/args.nu_max:.1f} → {1/args.nu_min:.1f}")
    print(f"[generate] Steps: {args.steps}")
    print(f"[generate] Output dir: {DATA_DIR}\n")

    if args.dry_run:
        for i, nu in enumerate(nu_values):
            print(f"  sample {i:04d}  nu={nu:.5f}  Re={1/nu:.1f}")
        exit(0)

    # --- Build once ---
    if not args.skip_build:
        build_solver()

    binary = find_binary()
    if binary is None:
        print("[generate] ERROR: Could not find solver binary. Run 'make omp' in solver/")
        exit(1)
    print(f"[generate] Using binary: {binary}\n")

    # --- Run sweep ---
    saved    = 0
    skipped  = 0
    metadata = []
    t_start  = time.time()

    for i, nu in enumerate(nu_values):
        t0 = time.time()
        print(f"  [{i+1:3d}/{args.n_samples}] nu={nu:.5f}  Re={1/nu:.1f} ...", end=" ", flush=True)

        result = run_solver(binary, nu, args.steps, args.output_every, i)

        if result is None:
            skipped += 1
            print("SKIPPED")
            continue

        u, v, p = result
        path = save_sample(saved, nu, u, v, p)
        elapsed = time.time() - t0

        # Estimate time remaining
        done = i + 1
        avg_time = (time.time() - t_start) / done
        remaining = avg_time * (args.n_samples - done)
        mins_left = int(remaining // 60)
        secs_left = int(remaining % 60)

        print(f"done ({elapsed:.1f}s)  ETA: {mins_left}m {secs_left}s")
        metadata.append({"sample": saved, "nu": float(nu), "re": float(1/nu)})
        saved += 1

    # --- Save metadata ---
    info = {
        "n_samples": saved,
        "n_skipped": skipped,
        "nu_min":    args.nu_min,
        "nu_max":    args.nu_max,
        "steps":     args.steps,
        "samples":   metadata
    }
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    total_mins = int((time.time() - t_start) // 60)
    print(f"\n[generate] Done in {total_mins} minutes.")
    print(f"[generate] {saved} samples saved, {skipped} skipped.")
    print(f"[generate] Metadata: {info_path}")
