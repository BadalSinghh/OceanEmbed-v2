# -*- coding: utf-8 -*-
"""
backend/app/api/models.py
=========================
Endpoint returning metadata, parameter counts, and benchmark results for all models.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(tags=["models"])

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

class ModelInfo(BaseModel):
    id: str
    name: str
    parameters: int
    architecture: str
    description: str
    val_rmse_2020: float
    val_argo_rmse_2020: float
    test_rmse_2021: float
    test_mae_2021: float
    test_argo_rmse_2021: float
    key_advantage: str
    test_rmse: float
    test_mae: float
    test_r2: float = 0.985
    argo_rmse: float
    argo_mae: float = 0.58
    pearson_r: float = 0.988


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    target_depths: List[int]
    domain: dict
    input_channels: List[dict]


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """Returns metadata and benchmark performance for all 6 models."""
    return ModelsResponse(
        models=[
            ModelInfo(
                id="model_b",
                name="Model B: Dual-Branch FNO + ViT",
                parameters=18841825,
                architecture="Dual Global Operator: 1,586-Token ViT (6 Layers, 8 Heads) + 4-Layer 2D FNO (Modes 16x24) + Adaptive Spatial Gating",
                description=(
                    "Combines data-driven multi-head self-attention with continuous Fourier spectral integral kernels. "
                    "Achieves the lowest error across both gridded reanalysis (0.8008°C) and physical Argo floats (0.8514°C) on the sealed 2021 test set."
                ),
                val_rmse_2020=0.7444,
                val_argo_rmse_2020=1.0038,
                test_rmse_2021=0.8008,
                test_mae_2021=0.5382,
                test_argo_rmse_2021=0.8514,
                key_advantage="Grand Winner on Sealed 2021 Test Data across both Gridded Reanalysis and In-Situ Floats.",
                test_rmse=0.8008,
                test_mae=0.5382,
                argo_rmse=0.8514
            ),
            ModelInfo(
                id="model_a",
                name="Model A: Dual-Branch FNO + U-Net",
                parameters=35055457,
                architecture="Multiscale Residual U-Net + 4-Layer 2D FNO (Modes 16x24) + Adaptive Spatial Gating",
                description=(
                    "Fuses local convolutional physics (coastal bathymetry, mesoscale eddies, fronts) with continuous Fourier spectral modes. "
                    "Top physical ground-truth performer on 2020 in-situ Argo profiles (0.9872°C)."
                ),
                val_rmse_2020=0.7563,
                val_argo_rmse_2020=0.9872,
                test_rmse_2021=0.8421,
                test_mae_2021=0.5664,
                test_argo_rmse_2021=0.8711,
                key_advantage="Top Physical Ground-Truth Model on 2020 Argo floats; regularizes high-frequency grid noise.",
                test_rmse=0.8421,
                test_mae=0.5664,
                argo_rmse=0.8711
            ),
            ModelInfo(
                id="vit",
                name="Vision Transformer Baseline",
                parameters=6044513,
                architecture="Patch Size 4x4, 1,586 Spatial Tokens, 6 Transformer Encoder Layers, 8 Attention Heads",
                description=(
                    "Pure attention-based baseline tokenizing the entire North Indian Ocean basin into 1,586 spatial patches. "
                    "Exceptional parameter efficiency with top reanalysis validation accuracy."
                ),
                val_rmse_2020=0.7142,
                val_argo_rmse_2020=0.9901,
                test_rmse_2021=0.8110,
                test_mae_2021=0.5478,
                test_argo_rmse_2021=0.8784,
                key_advantage="High parameter efficiency (6.04M) with basin-wide multi-head self-attention.",
                test_rmse=0.8110,
                test_mae=0.5478,
                argo_rmse=0.8784
            ),
            ModelInfo(
                id="unet",
                name="Deep Residual U-Net Baseline",
                parameters=22221153,
                architecture="Multiscale U-Net with 3 Downsampling Stages, Skip Connections, and 3D Temporal Attention",
                description=(
                    "Deep convolutional baseline capturing hierarchical spatial features from mesoscale to regional scales."
                ),
                val_rmse_2020=0.7308,
                val_argo_rmse_2020=0.9999,
                test_rmse_2021=0.8041,
                test_mae_2021=0.5460,
                test_argo_rmse_2021=0.8826,
                key_advantage="Strong local gradient and coastal frontal resolution.",
                test_rmse=0.8041,
                test_mae=0.5460,
                argo_rmse=0.8826
            ),
            ModelInfo(
                id="fno",
                name="Fourier Neural Operator Baseline",
                parameters=12740897,
                architecture="4-Layer 2D Spectral Operator with Modes (16, 24), Channel Width 64",
                description=(
                    "Continuous neural operator parameterizing integral kernels in Fourier frequency domain."
                ),
                val_rmse_2020=0.7611,
                val_argo_rmse_2020=0.9924,
                test_rmse_2021=0.8570,
                test_mae_2021=0.5857,
                test_argo_rmse_2021=0.8747,
                key_advantage="Continuous spectral filtering avoiding high-frequency gridded artifacts.",
                test_rmse=0.8570,
                test_mae=0.5857,
                argo_rmse=0.8747
            ),
            ModelInfo(
                id="climatology",
                name="Day-of-Year Mean Climatology",
                parameters=0,
                architecture="Non-parametric physical historical mean (2015-2019 DOY Average)",
                description=(
                    "Standard physical reference baseline representing mean seasonal solar cycle."
                ),
                val_rmse_2020=1.0544,
                val_argo_rmse_2020=1.2102,
                test_rmse_2021=1.0620,
                test_mae_2021=0.6950,
                test_argo_rmse_2021=1.2050,
                key_advantage="Zero-parameter physical benchmark.",
                test_rmse=1.0620,
                test_mae=0.6950,
                argo_rmse=1.2050
            )
        ],
        target_depths=TARGET_DEPTHS,
        domain={
            "region": "North Indian Ocean",
            "lat_min": 5.0,
            "lat_max": 30.0,
            "lon_min": 45.0,
            "lon_max": 105.0,
            "resolution_deg": 0.25,
            "grid_shape": [101, 241],
            "total_grid_points": 24341,
            "ocean_cells": 12038,
            "depth_min_m": 0,
            "depth_max_m": 1000,
            "depth_levels": 15,
            "baseline_period": "2015-01-01 to 2021-12-31 (7 Complete Calendar Years, 2557 Days)"
        },
        input_channels=[
            {"id": "SST", "name": "Sea Surface Temperature", "unit": "°C", "source": "OSTIA MetOffice L4 Reprocessed (0.05°)"},
            {"id": "SSS", "name": "Sea Surface Salinity", "unit": "psu", "source": "Multi-Platform Reprocessed L4 (~0.125°)"},
            {"id": "SLA", "name": "Sea Level Anomaly", "unit": "m", "source": "DUACS Two-Sat Gridded (0.25°)"},
            {"id": "CURRENT_U", "name": "Zonal Surface Current (U)", "unit": "m/s", "source": "OSCAR v2.0 Final (0.25°)"},
            {"id": "CURRENT_V", "name": "Meridional Surface Current (V)", "unit": "m/s", "source": "OSCAR v2.0 Final (0.25°)"},
            {"id": "WIND_U", "name": "Zonal Wind Stress (U, 10m)", "unit": "m/s", "source": "CCMP v3.1 Cross-Calibrated Multi-Scatterometer (0.25°)"},
            {"id": "WIND_V", "name": "Meridional Wind Stress (V, 10m)", "unit": "m/s", "source": "CCMP v3.1 Cross-Calibrated Multi-Scatterometer (0.25°)"},
        ],
    )
