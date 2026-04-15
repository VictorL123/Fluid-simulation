"""
surrogate/train.py
===================
Trains the UNet surrogate model on the generated simulation dataset.

Produces:
    surrogate/checkpoints/best_model.pt   <- best validation loss checkpoint
    surrogate/checkpoints/final_model.pt  <- model after all epochs
    surrogate/training_curve.png          <- loss curve plot

Usage:
    python train.py
    python train.py --epochs 100 --batch_size 16 --lr 1e-3

Requirements:
    pip install torch numpy matplotlib
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import UNet, CavityDataset, count_parameters

# ============================================================
#  Paths
# ============================================================
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


# ============================================================
#  Training loop — one epoch
# ============================================================
def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimiser.zero_grad()
        y_pred = model(x)
        loss   = criterion(y_pred, y)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
#  Validation loop — one epoch
# ============================================================
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss   = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# ============================================================
#  Plot training curve
# ============================================================
def plot_training_curve(train_losses, val_losses, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-',  linewidth=1.5, label='Training loss')
    ax.plot(epochs, val_losses,   'r--', linewidth=1.5, label='Validation loss')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalised)")
    ax.set_title("UNet Surrogate Model — Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Mark best validation epoch
    best_epoch = int(np.argmin(val_losses)) + 1
    best_loss  = min(val_losses)
    ax.axvline(best_epoch, color='green', linestyle=':', linewidth=1,
               label=f'Best val epoch ({best_epoch})')
    ax.annotate(f"best: {best_loss:.4f}",
                xy=(best_epoch, best_loss),
                xytext=(best_epoch + 1, best_loss * 2),
                fontsize=8, color='green')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"[train] Saved training curve: {save_path}")


# ============================================================
#  Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet surrogate model")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--base_feat",  type=int,   default=32,
                        help="Base feature count for UNet (default 32)")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # --- Datasets ---
    train_dataset = CavityDataset(DATA_DIR, split="train")
    val_dataset   = CavityDataset(DATA_DIR, split="val")

    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0)

    # --- Model ---
    model = UNet(in_channels=1, out_channels=3, base_features=args.base_feat).to(device)
    print(f"[train] Model parameters: {count_parameters(model):,}")

    # --- Optimiser and loss ---
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Learning rate scheduler — reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=10
    )

    # --- Training loop ---
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_epoch    = 0

    print(f"\n[train] Starting training for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss   = val_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - t0
        print(f"  epoch {epoch:4d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"lr={optimiser.param_groups[0]['lr']:.2e}  "
              f"time={elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                'epoch':      epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':   val_loss,
                'args':       vars(args)
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    # Save final model
    torch.save({
        'epoch':      args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss':   val_losses[-1],
        'args':       vars(args)
    }, os.path.join(CHECKPOINT_DIR, "final_model.pt"))

    print(f"\n[train] Training complete.")
    print(f"[train] Best val loss: {best_val_loss:.5f} at epoch {best_epoch}")
    print(f"[train] Checkpoints saved to {CHECKPOINT_DIR}")

    # Plot training curve
    curve_path = os.path.join(CHECKPOINT_DIR, "training_curve.png")
    plot_training_curve(train_losses, val_losses, curve_path)
