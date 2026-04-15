# Phase 3 — Neural Surrogate Model

A UNet neural network trained to predict steady-state fluid fields directly
from viscosity, replacing thousands of C++ solver timesteps with a single
forward pass.

## Files

```
surrogate/
├── generate_data.py   <- runs C++ solver across nu sweep, saves .npz files
├── model.py           <- CavityDataset class + UNet architecture
├── train.py           <- training loop, checkpointing, loss curves
├── evaluate.py        <- accuracy metrics, speed benchmark, visual comparison
├── data/              <- generated training samples (created by generate_data.py)
└── checkpoints/       <- saved model weights (created by train.py)
```

## Quick Start

### Step 1 — Install dependencies
```bash
pip install torch numpy matplotlib pandas
```

### Step 2 — Generate training data
```bash
python generate_data.py --n_samples 150 --steps 15000
```
This runs the C++ solver 150 times across a range of Reynolds numbers
and saves the steady-state fields. Takes 30-60 minutes.

To preview what will be run without executing:
```bash
python generate_data.py --dry_run
```

### Step 3 — Verify model architecture
```bash
python model.py
```
Should print input/output shapes and parameter count (~500k).

### Step 4 — Train
```bash
python train.py --epochs 150 --batch_size 8
```
Saves best checkpoint to `checkpoints/best_model.pt`.
Saves training curve to `checkpoints/training_curve.png`.

### Step 5 — Evaluate
```bash
python evaluate.py
```
Prints RMS error per field and speedup vs solver.
Saves side-by-side comparison plots to `eval_output/`.

## What to Expect

After training on 150 samples:
- u, v field RMS error: ~0.01-0.03 (comparable to solver's Ghia error)
- Speedup over solver: ~1000-10000x (ms vs tens of seconds)
- Training time: ~5-15 minutes on CPU, ~1-2 minutes on GPU

## Architecture

```
Input: (1, 41, 41)  — nu broadcast spatially
    ↓
Encoder: 32 → 64 → 128 → 256 channels  (with MaxPool downsampling)
    ↓
Bottleneck: 512 channels
    ↓
Decoder: 256 → 128 → 64 → 32 channels  (with ConvTranspose upsampling)
         + skip connections from encoder at each level
    ↓
Output: (3, 41, 41)  — u, v, p fields
```

Skip connections pass fine spatial detail (boundary layers, corner vortices)
from encoder to decoder, which is critical for accurate field reconstruction.

## CV Talking Points

- "Trained a UNet neural surrogate model in PyTorch to predict steady-state
  fluid velocity and pressure fields from viscosity, achieving RMS error
  comparable to the numerical solver"
- "Demonstrated Xx inference speedup over the C++ solver by replacing
  iterative timestepping with a single neural network forward pass"
- "Generated a 150-sample training dataset via automated parameter sweep
  of the C++ solver across Reynolds numbers 10-200"
