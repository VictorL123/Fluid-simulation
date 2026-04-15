"""
validate/benchmark.py
======================
Benchmarks the serial vs OpenMP solver and plots speedup curves.

Runs the solver multiple times with different thread counts,
measures wall-clock time, and produces a speedup chart.

Usage:
    python benchmark.py
    python benchmark.py --steps 1000 --runs 3

Requirements:
    pip install numpy matplotlib
"""

import argparse
import subprocess
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SOLVER_DIR  = os.path.join(os.path.dirname(__file__), "../solver")
SERIAL_BIN  = os.path.join(SOLVER_DIR, "fluid_sim.exe")
OMP_BIN     = os.path.join(SOLVER_DIR, "fluid_sim_omp.exe")
OUTPUT_DIR  = os.path.join(SOLVER_DIR, "output")
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "benchmark_results.json")


# ============================================================
#  Check binaries exist
# ============================================================
def check_binaries():
    missing = []
    if not os.path.exists(SERIAL_BIN):
        missing.append("fluid_sim  (run: make)")
    if not os.path.exists(OMP_BIN):
        missing.append("fluid_sim_omp  (run: make omp)")
    if missing:
        print("[benchmark] ERROR — missing binaries:")
        for m in missing:
            print(f"  {m}")
        print("\nBuild both from the solver/ directory first.")
        exit(1)


# ============================================================
#  Run solver and measure wall-clock time
# ============================================================
def time_solver(binary, n_threads, n_steps, n_runs):
    """
    Runs the solver binary n_runs times with the given thread count.
    Returns the median wall-clock time in seconds.
    The solver reads N and TOTAL_STEPS from its compiled constants,
    so we patch a temporary main just for benchmarking.
    """
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(n_threads)

    times = []
    for run in range(n_runs):
        start = time.perf_counter()
        result = subprocess.run(
            [binary],
            env=env,
            capture_output=True,
            cwd=SOLVER_DIR
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            print(f"[benchmark] ERROR: solver returned non-zero exit code")
            print(result.stderr.decode())
            exit(1)

        times.append(elapsed)
        print(f"  threads={n_threads}  run={run+1}/{n_runs}  time={elapsed:.3f}s")

    return float(np.median(times))


# ============================================================
#  Plot speedup curve
# ============================================================
def plot_speedup(results, save=True):
    thread_counts = [r["threads"] for r in results]
    times         = [r["time_s"]  for r in results]
    serial_time   = results[0]["time_s"]  # 1-thread time is baseline

    speedups      = [serial_time / t for t in times]
    ideal_speedup = thread_counts  # perfect linear scaling

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("OpenMP Parallelisation Benchmark — Lid-Driven Cavity Solver", fontsize=13)

    # --- Plot 1: Wall-clock time ---
    ax = axes[0]
    ax.plot(thread_counts, times, 'b-o', markersize=7, linewidth=2, label='Measured time')
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Simulation time vs thread count")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate each point with its time
    for tc, t in zip(thread_counts, times):
        ax.annotate(f"{t:.2f}s", xy=(tc, t), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    # --- Plot 2: Speedup curve ---
    ax = axes[1]
    ax.plot(thread_counts, speedups, 'b-o', markersize=7, linewidth=2, label='Measured speedup')
    ax.plot(thread_counts, ideal_speedup, 'r--', linewidth=1.5, label='Ideal linear speedup')
    ax.set_xlabel("Number of threads")
    ax.set_ylabel("Speedup (vs serial)")
    ax.set_title("Speedup vs thread count")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate each point with its speedup
    for tc, sp in zip(thread_counts, speedups):
        ax.annotate(f"{sp:.2f}x", xy=(tc, sp), textcoords="offset points",
                    xytext=(5, -12), fontsize=8)

    # Add efficiency annotation
    max_threads = thread_counts[-1]
    max_speedup = speedups[-1]
    efficiency  = max_speedup / max_threads * 100
    ax.text(0.05, 0.95,
            f"Parallel efficiency at {max_threads} threads: {efficiency:.0f}%",
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save:
        out = os.path.join(OUTPUT_DIR, "benchmark_speedup.png")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(out, dpi=120, bbox_inches='tight')
        print(f"\n[benchmark] Saved: {out}")
    else:
        plt.show()


# ============================================================
#  Print summary table
# ============================================================
def print_summary(results):
    serial_time = results[0]["time_s"]
    print("\n[benchmark] === Results ===")
    print(f"{'Threads':<10} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 45)
    for r in results:
        speedup    = serial_time / r["time_s"]
        efficiency = speedup / r["threads"] * 100
        print(f"{r['threads']:<10} {r['time_s']:<12.3f} {speedup:<10.2f} {efficiency:<10.1f}%")

    print("\n[benchmark] Amdahl's Law note:")
    print("  Efficiency below 100% is normal — some parts of the code")
    print("  are serial (boundary conditions, file I/O) and cannot be parallelised.")
    print("  This is expected and worth explaining in an interview.")


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark serial vs OpenMP solver")
    parser.add_argument("--runs",    type=int, default=3,
                        help="Number of runs per thread count (median is taken)")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Thread counts to benchmark (e.g. --threads 1 2 4 8)")
    parser.add_argument("--show",    action="store_true",
                        help="Show plot interactively instead of saving")
    args = parser.parse_args()

    check_binaries()

    print(f"[benchmark] Benchmarking with thread counts: {args.threads}")
    print(f"[benchmark] Runs per configuration: {args.runs}")
    print(f"[benchmark] Using solver binary: {OMP_BIN}\n")

    results = []
    for n_threads in args.threads:
        print(f"[benchmark] Testing {n_threads} thread(s)...")
        # Always use the OMP binary — at 1 thread it matches serial performance
        t = time_solver(OMP_BIN, n_threads, n_runs=args.runs, n_steps=None)
        results.append({"threads": n_threads, "time_s": t})

    # Save raw results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[benchmark] Raw results saved to {RESULTS_FILE}")

    print_summary(results)
    plot_speedup(results, save=not args.show)
