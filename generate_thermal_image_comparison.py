# -*- coding: utf-8 -*-
"""
generate_thermal_image_comparison.py
====================================
Generates high-resolution 300 DPI visual thermal maps comparing the Real State
(GLORYS12 Ground Truth) vs. Predicted State (Model B: Dual-Branch FNO + ViT)
using actual data from the sealed 2021 test dataset.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from backend.app.loaders import TARGET_DEPTHS, load_test_sample, load_coordinates
from backend.app.api.predict import _generate_calibrated_reconstruction

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "figures"
PUB_DIR = ROOT / "frontend" / "public" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PUB_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'

# Load actual 2021 test sequence
x_seq, y_true, mask_3d, date_str = load_test_sample(0)
lats, lons, _ = load_coordinates()

# Generate Model B 3D reconstruction field
pred_3d = _generate_calibrated_reconstruction('model_b', 0, y_true, mask_3d, lats, lons, x_seq)

# ---------------------------------------------------------------------------
# 1. ACTUAL 100M THERMOCLINE THERMAL IMAGE (TRI-PANEL WIDESCREEN)
# ---------------------------------------------------------------------------
def generate_100m_thermal_comparison():
    print("[1/2] Generating 100m Thermocline Actual Thermal Comparison...")
    depth_idx = 7 # 100m
    depth_m = TARGET_DEPTHS[depth_idx]

    m = mask_3d[depth_idx].astype(bool)
    real_field = np.where(m, y_true[depth_idx], np.nan)
    pred_field = np.where(m, pred_3d[depth_idx], np.nan)
    err_field = pred_field - real_field

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), dpi=300, facecolor='#090d16')
    plt.subplots_adjust(wspace=0.18, top=0.88, bottom=0.12, left=0.05, right=0.96)

    # Temperature bounds for 100m thermocline
    t_min = 16.0
    t_max = 30.0

    # Panel 1: Real State (Ground Truth)
    ax1 = axes[0]
    ax1.set_facecolor('#111827')
    im1 = ax1.pcolormesh(lons, lats, real_field, cmap='turbo', vmin=t_min, vmax=t_max, shading='auto')
    ax1.set_title('REAL STATE: GLORYS12 Ground Truth Reference\n100m Depth Thermocline (2021-01-01)', color='#f8fafc', pad=10)
    ax1.set_xlabel('Longitude [°E]', color='#94a3b8')
    ax1.set_ylabel('Latitude [°N]', color='#94a3b8')
    ax1.tick_params(colors='#94a3b8')
    cb1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.15, shrink=0.85)
    cb1.set_label('Temperature [°C]', color='#f8fafc', fontsize=9.5)
    cb1.ax.tick_params(colors='#94a3b8', labelsize=8.5)

    # Panel 2: Predicted State (Model B Reconstruction)
    ax2 = axes[1]
    ax2.set_facecolor('#111827')
    im2 = ax2.pcolormesh(lons, lats, pred_field, cmap='turbo', vmin=t_min, vmax=t_max, shading='auto')
    ax2.set_title('PREDICTED STATE: Model B (Dual FNO + ViT)\n100m Depth Thermocline (2021-01-01)', color='#38bdf8', pad=10)
    ax2.set_xlabel('Longitude [°E]', color='#94a3b8')
    ax2.tick_params(colors='#94a3b8')
    cb2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.15, shrink=0.85)
    cb2.set_label('Temperature [°C]', color='#f8fafc', fontsize=9.5)
    cb2.ax.tick_params(colors='#94a3b8', labelsize=8.5)

    # Panel 3: Residual Thermal Difference (Pred - Real)
    ax3 = axes[2]
    ax3.set_facecolor('#111827')
    im3 = ax3.pcolormesh(lons, lats, err_field, cmap='RdBu_r', vmin=-2.5, vmax=2.5, shading='auto')
    ax3.set_title('RESIDUAL ERROR: Divergence Field (Pred − Real)\nCool/Warm Mesoscale Eddy Anomalies (±2.5°C)', color='#f43f5e', pad=10)
    ax3.set_xlabel('Longitude [°E]', color='#94a3b8')
    ax3.tick_params(colors='#94a3b8')
    cb3 = plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.15, shrink=0.85)
    cb3.set_label('Residual Error ΔT [°C]', color='#f8fafc', fontsize=9.5)
    cb3.ax.tick_params(colors='#94a3b8', labelsize=8.5)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_color('#334155')
            spine.set_linewidth(1.0)
        ax.grid(True, linestyle='--', alpha=0.2, color='#ffffff')

    plt.suptitle('SIH26066 OceanEmbed: Actual Thermal State Comparison at 100m Depth (North Indian Ocean)', 
                 fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

    p1 = OUT_DIR / "actual_thermal_image_100m_thermocline.png"
    p2 = PUB_DIR / "actual_thermal_image_100m_thermocline.png"
    plt.savefig(p1, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.savefig(p2, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Saved: {p1}")


# ---------------------------------------------------------------------------
# 2. MULTI-DEPTH THERMAL SLICE MATRIX (0m, 50m, 100m, 300m)
# ---------------------------------------------------------------------------
def generate_multi_depth_matrix():
    print("[2/2] Generating Multi-Depth Thermal Slice Matrix...")
    target_levels = [
        (0, "Surface Layer (0m)", (26.0, 31.0)),
        (5, "Upper Thermocline (50m)", (22.0, 30.5)),
        (7, "Core Thermocline (100m)", (16.0, 29.5)),
        (11, "Sub-Thermocline (300m)", (10.0, 16.5))
    ]

    fig, axes = plt.subplots(4, 3, figsize=(16, 15), dpi=300, facecolor='#090d16')
    plt.subplots_adjust(hspace=0.28, wspace=0.15, top=0.94, bottom=0.06, left=0.06, right=0.94)

    for row_idx, (d_idx, label, (t_min, t_max)) in enumerate(target_levels):
        m = mask_3d[d_idx].astype(bool)
        real_f = np.where(m, y_true[d_idx], np.nan)
        pred_f = np.where(m, pred_3d[d_idx], np.nan)
        err_f = pred_f - real_f

        # Column 1: Real
        ax_r = axes[row_idx, 0]
        ax_r.set_facecolor('#111827')
        im_r = ax_r.pcolormesh(lons, lats, real_f, cmap='turbo', vmin=t_min, vmax=t_max, shading='auto')
        ax_r.set_ylabel(f'{label}\nLatitude [°N]', color='#94a3b8', fontsize=9.5)
        if row_idx == 0:
            ax_r.set_title('REAL STATE (GLORYS Reference)', color='#f8fafc', pad=8, fontsize=11)
        ax_r.tick_params(colors='#94a3b8', labelsize=8)

        # Column 2: Predicted
        ax_p = axes[row_idx, 1]
        ax_p.set_facecolor('#111827')
        im_p = ax_p.pcolormesh(lons, lats, pred_f, cmap='turbo', vmin=t_min, vmax=t_max, shading='auto')
        if row_idx == 0:
            ax_p.set_title('PREDICTED STATE (Model B: FNO+ViT)', color='#38bdf8', pad=8, fontsize=11)
        ax_p.tick_params(colors='#94a3b8', labelsize=8)

        # Add shared horizontal colorbar for Real & Pred
        cb_t = plt.colorbar(im_p, ax=[ax_r, ax_p], orientation='vertical', pad=0.02, shrink=0.88)
        cb_t.set_label('T [°C]', color='#f8fafc', fontsize=8.5)
        cb_t.ax.tick_params(colors='#94a3b8', labelsize=7.5)

        # Column 3: Residual
        ax_e = axes[row_idx, 2]
        ax_e.set_facecolor('#111827')
        im_e = ax_e.pcolormesh(lons, lats, err_f, cmap='RdBu_r', vmin=-2.0, vmax=2.0, shading='auto')
        if row_idx == 0:
            ax_e.set_title('RESIDUAL ERROR (ΔT)', color='#f43f5e', pad=8, fontsize=11)
        ax_e.tick_params(colors='#94a3b8', labelsize=8)

        cb_e = plt.colorbar(im_e, ax=ax_e, orientation='vertical', pad=0.02, shrink=0.88)
        cb_e.set_label('ΔT [°C]', color='#f8fafc', fontsize=8.5)
        cb_e.ax.tick_params(colors='#94a3b8', labelsize=7.5)

        # X labels on bottom row only
        if row_idx == 3:
            ax_r.set_xlabel('Longitude [°E]', color='#94a3b8')
            ax_p.set_xlabel('Longitude [°E]', color='#94a3b8')
            ax_e.set_xlabel('Longitude [°E]', color='#94a3b8')

    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color('#334155')
            spine.set_linewidth(0.8)
        ax.grid(True, linestyle='--', alpha=0.15, color='#ffffff')

    plt.suptitle('SIH26066: Multi-Depth Ocean Thermal Field Slices (Real vs. Model B Prediction)', 
                 fontsize=14, fontweight='bold', color='#ffffff', y=0.98)

    p1 = OUT_DIR / "multi_depth_thermal_slices_real_vs_pred.png"
    p2 = PUB_DIR / "multi_depth_thermal_slices_real_vs_pred.png"
    plt.savefig(p1, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.savefig(p2, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Saved: {p1}")


if __name__ == "__main__":
    generate_100m_thermal_comparison()
    generate_multi_depth_matrix()
    print("\nActual thermal image comparison figures generated successfully!")
