"""
validate/signal_analysis.py
============================
Signal processing analysis of the lid-driven cavity simulation.

Analyses the velocity time series recorded at the probe point (cavity centre)
and produces three plots:

  1. Time series   — u(t) and v(t) over time
  2. FFT spectrum  — frequency content of the velocity signal
  3. Spectrogram   — how frequency content evolves over time (flow settling)

Usage:
    python signal_analysis.py                  # analyse probe at cavity centre
    python signal_analysis.py --re 1000        # label plots with Reynolds number
    python signal_analysis.py --show           # show interactively

Requirements:
    pip install numpy matplotlib pandas scipy
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from scipy.fft import fft, fftfreq


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../solver/output")
PROBE_FILE = os.path.join(OUTPUT_DIR, "probe_timeseries.csv")


# ============================================================
#  Load probe data
# ============================================================
def load_probe(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Probe file not found: {filepath}\n"
            "Make sure you have run the solver first (make run)"
        )
    df = pd.read_csv(filepath)
    print(f"[signal] Loaded {len(df)} timesteps from probe file")
    return df


# ============================================================
#  Plot 1: Time series
#  Shows how u and v at the probe point evolve over time.
#
#  What to expect:
#    Re=100:  signal settles to a steady value (flat line) — steady flow
#    Re=1000: signal oscillates periodically — vortex shedding
# ============================================================
def plot_timeseries(df, ax, re=100):
    ax.plot(df['t'], df['u'], label='u (x-velocity)', linewidth=0.8, color='royalblue')
    ax.plot(df['t'], df['v'], label='v (y-velocity)', linewidth=0.8, color='tomato')
    ax.set_xlabel("Time")
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity time series at cavity centre  (Re={re})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate steady-state region (last 20% of simulation)
    t_settle = df['t'].max() * 0.8
    ax.axvline(t_settle, color='green', linestyle='--', linewidth=0.8,
               label='Steady-state region')
    ax.legend()


# ============================================================
#  Plot 2: FFT Power Spectrum
#  Shows which frequencies are present in the velocity signal.
#
#  What to expect:
#    Re=100:  flat/noisy spectrum — no dominant frequency (steady flow)
#    Re=1000: sharp spike at the vortex shedding frequency
#
#  Uses Welch's method (averaged periodogram) for a cleaner spectrum
#  than a raw FFT — standard practice in signal processing.
# ============================================================
def plot_fft(df, ax, dt, re=100):
    # Use only the steady-state portion (last 50% of signal)
    # to avoid transient startup effects contaminating the spectrum
    n_half = len(df) // 2
    u_steady = df['u'].values[n_half:]

    # Welch's method: splits signal into overlapping segments,
    # FFTs each one, then averages — much cleaner than a single FFT
    freqs, power = welch(u_steady, fs=1.0/dt, nperseg=min(256, len(u_steady)//4))

    # Only plot positive frequencies up to a sensible limit
    mask = freqs > 0
    ax.semilogy(freqs[mask], power[mask], color='royalblue', linewidth=1.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(f"FFT Power Spectrum of u-velocity  (Re={re})")
    ax.grid(True, alpha=0.3, which='both')

    # Find and annotate dominant frequency
    peak_idx = np.argmax(power[mask])
    peak_freq = freqs[mask][peak_idx]
    peak_power = power[mask][peak_idx]
    if peak_power > 1e-10:  # only annotate if there's a meaningful peak
        ax.annotate(f"f = {peak_freq:.4f} Hz",
                    xy=(peak_freq, peak_power),
                    xytext=(peak_freq * 2, peak_power * 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    color='red', fontsize=9)

    print(f"[signal] Dominant frequency: {peak_freq:.6f} Hz")
    print(f"[signal] Corresponding period: {1.0/peak_freq:.2f} time units" if peak_freq > 0 else "")


# ============================================================
#  Plot 3: Spectrogram
#  Shows how the frequency content CHANGES over time.
#  This lets you see the flow transitioning from startup
#  transient to settled steady state.
#
#  What to expect:
#    Early time: broad frequency content (flow is developing)
#    Late time:  content concentrates or dies away (flow settles)
# ============================================================
def plot_spectrogram(df, ax, dt, re=100):
    u_signal = df['u'].values

    f, t_seg, Sxx = spectrogram(u_signal, fs=1.0/dt,
                                 nperseg=min(512, len(u_signal)//8),
                                 noverlap=None,
                                 scaling = 'density')

    # Plot in dB scale for better dynamic range
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    im = ax.pcolormesh(t_seg, f, Sxx_db, shading='gouraud', cmap='inferno')
    plt.colorbar(im, ax=ax, label='Power (dB)')
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram of u-velocity  (Re={re})")

    # Cap y-axis at a sensible frequency range
    ax.set_ylim(0, min(f.max(), 5.0))


# ============================================================
#  Steady-state statistics
#  Simple summary of the settled flow behaviour
# ============================================================
def print_stats(df):
    # Use last 20% of simulation as steady state
    n_steady = int(len(df) * 0.8)
    u_ss = df['u'].values[n_steady:]
    v_ss = df['v'].values[n_steady:]

    print("\n[signal] === Steady-state statistics (last 20% of simulation) ===")
    print(f"[signal] u mean:    {u_ss.mean():.6f}")
    print(f"[signal] u std dev: {u_ss.std():.6f}  (near zero = steady flow)")
    print(f"[signal] v mean:    {v_ss.mean():.6f}")
    print(f"[signal] v std dev: {v_ss.std():.6f}  (near zero = steady flow)")

    if u_ss.std() < 1e-4:
        print("[signal] Flow is STEADY — no oscillation detected")
    else:
        print("[signal] Flow is UNSTEADY — periodic or chaotic behaviour detected")


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal analysis of cavity flow probe data")
    parser.add_argument("--re",   type=float, default=100,  help="Reynolds number (for plot labels)")
    parser.add_argument("--dt",   type=float, default=0.001, help="Timestep used in simulation")
    parser.add_argument("--show", action="store_true",       help="Show plots interactively")
    args = parser.parse_args()

    df = load_probe(PROBE_FILE)
    print_stats(df)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f"Signal Analysis — Lid-Driven Cavity  Re={args.re}", fontsize=13)

    plot_timeseries(df, axes[0], re=args.re)
    plot_fft(df, axes[1], dt=args.dt, re=args.re)
    plot_spectrogram(df, axes[2], dt=args.dt, re=args.re)

    plt.tight_layout()

    if args.show:
        plt.show()
    else:
        out = os.path.join(OUTPUT_DIR, "signal_analysis.png")
        plt.savefig(out, dpi=120, bbox_inches='tight')
        print(f"\n[signal] Saved: {out}")
        print("[signal] Open output/signal_analysis.png to view results")
