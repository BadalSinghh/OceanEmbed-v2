# -*- coding: utf-8 -*-
"""
backend/app/loaders.py
======================
Production data and model loaders for the FastAPI backend.
Directly interfaces with c:\oceanEmbed models, checkpoints, and processed datasets.
"""

import functools
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import netCDF4 as nc
import os
import numpy as np
import pandas as pd
import torch

WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", Path(__file__).resolve().parent.parent.parent))
SCRIPTS_DIR = WORKSPACE_ROOT / "OceanEmbed_Data" / "scripts"
CHECKPOINT_DIR = WORKSPACE_ROOT / "checkpoints"
REPORTS_DIR = WORKSPACE_ROOT / "outputs" / "reports"

try:
    from backend.app.models.model_fno_unet import OceanFNO_UNet
    from backend.app.models.model_fno_vit import OceanFNO_ViT
    from backend.app.models.model_vit import OceanViT
    from backend.app.models.model_unet import OceanUNet
    from backend.app.models.model_fno import OceanFNO
except ImportError:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    from model_fno_unet import OceanFNO_UNet
    from model_fno_vit import OceanFNO_ViT
    from model_vit import OceanViT
    from model_unet import OceanUNet
    from model_fno import OceanFNO

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ---------------------------------------------------------------------------
# Coordinate / mask / metadata loaders — cached once per process
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_coordinates() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (lats, lons, depths) for the canonical 0.25° North Indian Ocean grid."""
    lats = np.linspace(5.0, 30.0, 101, dtype=np.float32)
    lons = np.linspace(45.0, 105.0, 241, dtype=np.float32)
    depths = np.array(TARGET_DEPTHS, dtype=np.float32)
    return lats, lons, depths


@functools.lru_cache(maxsize=1)
def load_land_mask() -> np.ndarray:
    """Returns 2D surface land mask [101, 241] where True = land/masked."""
    mask_file = WORKSPACE_ROOT / "ocean_mask.nc"
    if mask_file.exists():
        with nc.Dataset(mask_file, "r") as ds:
            # ocean_mask: 1 = ocean, 0 = land
            mask = ds.variables["ocean_mask"][:]
            return (mask == 0)
    # Default fallback
    return np.zeros((101, 241), dtype=bool)


@functools.lru_cache(maxsize=1)
def load_3d_depth_mask() -> np.ndarray:
    """Returns 3D bathymetry mask [15, 101, 241] where True = ocean."""
    mask_file = WORKSPACE_ROOT / "ocean_mask.nc"
    if mask_file.exists():
        with nc.Dataset(mask_file, "r") as ds:
            if "depth_mask" in ds.variables:
                return ds.variables["depth_mask"][:].astype(bool)
    return np.ones((15, 101, 241), dtype=bool)


@functools.lru_cache(maxsize=1)
def load_test_metadata() -> Tuple[List[str], int]:
    """Returns dates list for the sealed 2021 test set (365 days)."""
    test_csv = WORKSPACE_ROOT / "test_sequences.csv"
    if test_csv.exists():
        df = pd.read_csv(test_csv)
        dates = df["target_date"].tolist()
        return dates, len(dates)
    return [], 0


@functools.lru_cache(maxsize=1)
def load_normalization_stats() -> Dict:
    stats_file = WORKSPACE_ROOT / "backend" / "app" / "data" / "normalization_stats.json"
    if not stats_file.exists():
        stats_file = WORKSPACE_ROOT / "OceanEmbed_Data" / "harmonized" / "normalization_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)
    return {}


@functools.lru_cache(maxsize=1)
def get_test_dataset():
    """Cached dataset handle for test sequence queries."""
    return OceanEmbedDataset(split="test", cache_in_memory=True)


# ---------------------------------------------------------------------------
# Sample loader — loads from pre-packaged test_samples.npz or dataset
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _load_packaged_test_samples():
    npz_path = WORKSPACE_ROOT / "backend" / "app" / "data" / "test_samples.npz"
    if npz_path.exists():
        return np.load(npz_path, allow_pickle=True)
    return None

def load_test_sample(idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Returns (x, y, mask, date) for test sample index.
    x: [7, 7, 101, 241]
    y: [15, 101, 241]
    mask: [15, 101, 241]
    """
    pkg = _load_packaged_test_samples()
    if pkg is not None:
        indices = pkg["indices"]
        # Find exact or closest available sample in package
        if idx in indices:
            target_idx = idx
        else:
            target_idx = int(indices[np.argmin(np.abs(indices - idx))])
        
        idx_pos = list(indices).index(target_idx)
        date_str = str(pkg["dates"][idx_pos])
        x = pkg[f"x_{target_idx}"]
        y = pkg[f"y_{target_idx}"]
        mask = pkg[f"mask_{target_idx}"]
        return x, y, mask, date_str

    ds = get_test_dataset()
    item = ds[idx]
    date_str = ds.df.iloc[idx]["target_date"]
    return item["x"].numpy(), item["y"].numpy(), item["mask"].numpy(), date_str


# ---------------------------------------------------------------------------
# Model loader — cached per model_id
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=6)
def load_pytorch_model(model_id: str):
    """
    Load and cache PyTorch checkpoints:
    - model_b: Dual-Branch FNO + ViT (Grand Winner)
    - model_a: Dual-Branch FNO + U-Net (Top Physical In-Situ Model)
    - vit: Vision Transformer Baseline
    - unet: Deep Residual U-Net Baseline
    - fno: Fourier Neural Operator Baseline
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_id == "model_b":
        model = OceanFNO_ViT(in_vars=7, num_days=7, num_depths=15, width=64, embed_dim=256, patch_size=4, num_vit_layers=6, num_heads=8, modes1=16, modes2=24)
        ckpt_path = CHECKPOINT_DIR / "best_fno_vit_model_b.pth"
    elif model_id == "model_a":
        model = OceanFNO_UNet(in_vars=7, num_days=7, num_depths=15, width=64, modes1=16, modes2=24)
        ckpt_path = CHECKPOINT_DIR / "best_fno_unet_model_a.pth"
    elif model_id == "vit":
        model = OceanViT(in_vars=7, num_days=7, num_depths=15, in_channels=64, embed_dim=256, patch_size=4, num_layers=6, num_heads=8)
        ckpt_path = CHECKPOINT_DIR / "best_vit_baseline.pth"
    elif model_id == "unet":
        model = OceanUNet(in_vars=7, num_days=7, num_depths=15)
        ckpt_path = CHECKPOINT_DIR / "best_unet_baseline.pth"
    elif model_id == "fno":
        model = OceanFNO(in_vars=7, num_days=7, num_depths=15, width=64, modes1=16, modes2=24)
        ckpt_path = CHECKPOINT_DIR / "best_fno_baseline.pth"
    else:
        return None

    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_temperature_field(model, x: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    """
    Runs model inference on input sequence x [7, 7, 101, 241].
    Returns reconstructed 3D temperature [15, 101, 241] with land/bathymetry masked as NaN.
    """
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_tensor, _ = model(x_tensor)
        pred = pred_tensor[0].float().cpu().numpy()

    # Mask land and seabed
    mask_bool = mask_3d.astype(bool)
    pred_masked = np.where(mask_bool, pred, np.nan)
    return pred_masked
