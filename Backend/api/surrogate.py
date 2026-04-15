"""
api/surrogate.py
=================
Surrogate model interface.

Currently uses a MOCK surrogate that generates a realistic-looking
vortex field analytically. This means the UI works fully without
a trained PyTorch model.

TO REPLACE WITH REAL MODEL LATER:
    1. Train the PyTorch model (surrogate/train.py)
    2. Set SURROGATE_READY = True
    3. Set CHECKPOINT_PATH to your best_model.pt
    The rest of the code will automatically use the real model.
"""

import numpy as np
import os

# ============================================================
#  Toggle this to True once your PyTorch model is trained
# ============================================================
SURROGATE_READY = False

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), '../../surrogate/checkpoints/best_model.pt'
)

# ============================================================
#  Mock surrogate — generates realistic vortex field
#  analytically using a stream function approach.
#  Looks convincing in the UI without any ML involved.
# ============================================================
def mock_surrogate(nu, grid_size=41):
    """
    Generate a realistic-looking lid-driven cavity velocity field
    without running the actual solver or ML model.

    Uses a superposition of vortex stream functions that
    qualitatively matches the real flow at low Re.
    """
    N = grid_size
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    re = 1.0 / nu

    # Primary vortex centre moves with Re
    # At low Re: centre is near middle
    # At higher Re: centre shifts toward top-right
    cx = 0.5 + 0.15 * (re / 400.0)
    cy = 0.5 + 0.10 * (re / 400.0)
    cx = min(cx, 0.75)
    cy = min(cy, 0.75)

    # Vortex strength scales with Re
    strength = 0.3 + 0.2 * (re / 200.0)

    # Primary vortex stream function
    r2 = (X - cx)**2 + (Y - cy)**2
    sigma = 0.25
    psi = -strength * np.exp(-r2 / (2 * sigma**2))

    # u = d(psi)/dy,  v = -d(psi)/dx
    dy = 1.0 / (N - 1)
    dx = 1.0 / (N - 1)

    u = np.gradient(psi, dy, axis=0)
    v = -np.gradient(psi, dx, axis=1)

    # Add lid velocity (top row moves right at u=1)
    # Blend it into the field so it looks natural
    lid_influence = Y ** 3
    u = u + lid_influence * (1.0 - np.abs(u))

    # Pressure: roughly proportional to vortex core
    p = -0.5 * (u**2 + v**2)
    p = p - p.mean()

    # Apply boundary conditions
    u[0,  :] = 0.0   # bottom
    u[-1, :] = 1.0   # top lid
    u[:,  0] = 0.0   # left
    u[:, -1] = 0.0   # right
    v[0,  :] = 0.0
    v[-1, :] = 0.0
    v[:,  0] = 0.0
    v[:, -1] = 0.0

    return u.astype(np.float32), v.astype(np.float32), p.astype(np.float32)


# ============================================================
#  Real surrogate — loaded once and cached
# ============================================================
_model_cache = None

def real_surrogate(nu, grid_size=41):
    """Load and run the trained PyTorch UNet surrogate model."""
    global _model_cache

    import sys
    surrogate_dir = os.path.join(os.path.dirname(__file__), '../../surrogate')
    if surrogate_dir not in sys.path:
        sys.path.insert(0, surrogate_dir)

    import torch
    from model import UNet, CavityDataset

    if _model_cache is None:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        args = checkpoint['args']
        model = UNet(in_channels=1, out_channels=3,
                     base_features=args.get('base_feat', 32))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _model_cache = model
        print("[surrogate] Real model loaded from checkpoint")

    # Load dataset just for normalisation stats
    data_dir = os.path.join(os.path.dirname(__file__), '../../surrogate/data')
    dataset = CavityDataset(data_dir, split='train')

    nu_norm = (nu - dataset.nu_mean) / dataset.nu_std
    x = torch.full((1, 1, grid_size, grid_size), nu_norm, dtype=torch.float32)

    with torch.no_grad():
        y_pred = _model_cache(x)
        y_phys = dataset.denormalise_output(y_pred)

    u = y_phys[0, 0].numpy()
    v = y_phys[0, 1].numpy()
    p = y_phys[0, 2].numpy()
    return u, v, p


# ============================================================
#  Public interface — call this from views.py
# ============================================================
def run_surrogate(nu, grid_size=41):
    """
    Run the surrogate model.
    Uses real model if SURROGATE_READY=True and checkpoint exists,
    otherwise falls back to mock surrogate.
    """
    if SURROGATE_READY and os.path.exists(CHECKPOINT_PATH):
        try:
            return real_surrogate(nu, grid_size), True  # (fields, is_real)
        except Exception as e:
            print(f"[surrogate] Real model failed: {e}, falling back to mock")

    return mock_surrogate(nu, grid_size), False  # (fields, is_real)
