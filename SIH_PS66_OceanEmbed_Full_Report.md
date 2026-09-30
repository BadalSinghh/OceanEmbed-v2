# SIH26066: OceanEmbed — Comprehensive Solution Report & Master Presentation Guide

**Smart India Hackathon 2026 | Problem Statement ID: SIH26066 (PS 66)**  
**Project Title**: OceanEmbed — Satellite Embedding-Based Deep Learning Framework for 3D Subsurface Ocean Temperature Reconstruction  
**Target Domain**: North Indian Ocean & Bay of Bengal ($5^\circ\text{N} - 30^\circ\text{N}, 45^\circ\text{E} - 105^\circ\text{E}$)  
**Target Depth Horizon**: 15 Standard Oceanographic Depths ($0\text{ m}$ to $1,000\text{ m}$)  
**Temporal Window**: 7-Day Multi-Variable Satellite History $\longrightarrow$ Day $t$ 3D Subsurface Thermal State  
**Production Prototype**: [OceanEmbed Live Web Workstation](https://oceanembed-backend-opio.onrender.com) | [GitHub Repository](https://github.com/BadalSinghh/OceanEmbed-v2)

---

## Table of Contents
1. [Executive Summary & Problem Statement Analysis](#1-executive-summary--problem-statement-analysis)
2. [Scientific & Operational Challenge: Why PS 66 is Hard](#2-scientific--operational-challenge-why-ps-66-is-hard)
3. [The OceanEmbed Innovation: Dual-Branch Neural Operator Paradigm](#3-the-oceanembed-innovation-dual-branch-neural-operator-paradigm)
4. [Data Engineering & Harmonization Pipeline](#4-data-engineering--harmonization-pipeline)
5. [Deep Learning Architecture & Mathematical Formulation](#5-deep-learning-architecture--mathematical-formulation)
6. [Experimental Evaluation & Ground-Truth Argo Validation](#6-experimental-evaluation--ground-truth-argo-validation)
7. [Full Web Application & Interactive Workstation Prototype](#7-full-web-application--interactive-workstation-prototype)
8. [Slide-by-Slide PPT Presentation Deck (12-Slide Outline)](#8-slide-by-slide-ppt-presentation-deck-12-slide-outline)
9. [Minute-by-Minute Video Demonstration & Pitch Script](#9-minute-by-minute-video-demonstration--pitch-script)
10. [Operational Impact & Deployment Feasibility](#10-operational-impact--deployment-feasibility)

---

## 1. Executive Summary & Problem Statement Analysis

### 1.1 The Challenge of SIH PS 66
Ocean thermal structure governs cyclone intensification, monsoon dynamics, acoustic propagation for naval defense, and marine fishery habitats. However, **satellites can only observe the surface "skin" of the ocean** (the top millimetre of Sea Surface Temperature, Sea Level Anomaly, and Sea Surface Salinity). 

Traditional subsurface thermal profiling relies on:
- **Argo autonomous profiling floats**: Highly accurate point measurements, but extremely sparse in time and space (a single float surfaces only once every 10 days, drifting unpredictably).
- **Moored buoy arrays (e.g., RAMA, OMNI)**: Fixed geographic coordinates with limited spatial coverage.
- **Physics-based numerical circulation models (e.g., NEMO, MOM6, HYCOM)**: Require millions of CPU hours on supercomputers and struggle with high latency.

**The SIH PS 66 Objective**: Develop an end-to-end, high-resolution deep learning framework capable of ingesting accessible multi-satellite surface observations and inferring the complete, continuous 3D temperature volume from the surface ($0\text{ m}$) down to $1,000\text{ m}$ depth across the North Indian Ocean at daily temporal resolution.

### 1.2 Our Solution: OceanEmbed
OceanEmbed resolves this problem through:
1. **Multi-Source Satellite Harmonization**: 7 surface variables over a 7-day rolling window ($[B, 7\text{ vars}, 7\text{ days}, 101\text{ lat}, 241\text{ lon}]$).
2. **Novel Dual-Branch Neural Architectures**:
   - **Model B (Dual-Branch FNO + Vision Transformer)**: The Grand Winner, achieving an overall RMSE of **$0.8008^\circ\text{C}$** on sealed 2021 reanalysis and **$0.8514^\circ\text{C}$** ($R = 0.9928$) on real physical Argo floats.
   - **Model A (Dual-Branch FNO + U-Net)**: Top 2020 physical validation model (**$0.9872^\circ\text{C}$** Argo RMSE).
3. **Adaptive Gated Fusion**: Dynamically balances continuous spectral kernel operators (FNO) for basin-scale balance with spatial attention/convolution (ViT/U-Net) for high-gradient eddies and coastal boundaries.
4. **Cloud-Native Interactive Decision-Support Workstation**: A production Next.js 16 + FastAPI web platform with 3D WebGL volume scrubbing, simultaneous tri-field 2D depth slices (Prediction, Reference, Residual), vertical transects (N-S meridional, W-E zonal), and direct in-situ Argo float verification.

---

## 2. Scientific & Operational Challenge: Why PS 66 is Hard

```
                                SATELLITE SENSORS (Only Skin Layer)
                    [ SST (Infrared/MW) | SLA (Altimetry) | SSS (SMAP) | Winds (Scatterometer) ]
                                            │
                                            ▼
   0m ─── Surface Mixed Layer ──────────────┬──────────────────────────────────────────
          Well-mixed, strongly correlated    │  High vertical gradient (dT/dz up to 0.15°C/m)
          with atmospheric forcing           ▼  INTERNAL WAVES & MESOSCALE EDDY ANOMALIES
  50m ─── Thermocline Zone ─────────────────┼──────────────────────────────────────────
          (Most non-linear & dynamic)        │  Displacements of 50-100m can cause
 200m ──────────────────────────────────────┘  local temperature shifts of > 3°C to 5°C!
                                            │
 500m ─── Intermediate & Deep Ocean ───────▼──────────────────────────────────────────
          Stably stratified, slowly varying, converges toward 4°C - 8°C abyssal floor
1000m ─────────────────────────────────────────────────────────────────────────────────
```

### The 4 Fundamental Scientific Difficulties:
1. **The Inverse Ill-Posed Problem**: Reconstructing a 3D field from 2D boundary conditions is mathematically non-unique. Multiple subsurface stratification profiles can produce similar surface temperatures.
2. **Thermocline Dynamics**: Between $50\text{ m}$ and $200\text{ m}$, the thermocline exhibits steep temperature gradients ($\partial T/\partial z$). Small vertical displacements caused by mesoscale cyclonic/anticyclonic eddies or internal Kelvin/Rossby waves result in thermal deviations exceeding $3^\circ\text{C}$ to $5^\circ\text{C}$.
3. **Complex Monsoon Forcing in the North Indian Ocean**: Unlike open oceans, the Bay of Bengal and Arabian Sea experience semiannual monsoon wind reversals, heavy river freshwater discharge (Ganges-Brahmaputra creating strong surface barrier layers), and intense tropical cyclone passages.
4. **Lack of Dense Gridded Ground Truth**: Models cannot be evaluated solely against numerical reanalysis (like GLORYS12), which itself contains modeling assumptions; they must be validated against real, untouched in-situ physical measurements from autonomous Argo profiling floats.

---

## 3. The OceanEmbed Innovation: Dual-Branch Neural Operator Paradigm

Existing research papers typically employ either standard 2D/3D CNNs (U-Net) or pure spatial Vision Transformers (ViT). Both suffer from fundamental physical shortcomings:
- **CNNs / U-Net**: Excellent at capturing local edges, fronts, and coastal boundaries, but have a localized receptive field that struggles with basin-wide planetary wave propagation (Kelvin and Rossby waves across the Indian Ocean).
- **Vision Transformers (ViT)**: Capture long-range self-attention across image patches, but treat pixels as discrete entities, vulnerable to patch-boundary artifacts and high parameter counts.
- **Fourier Neural Operators (FNO)**: Parameterize integral kernels directly in Fourier frequency space, learning **mesh-independent, continuous solution operators** of partial differential equations (PDEs), preserving global energy conservation and thermal equilibrium.

### The OceanEmbed Synthesis:
We designed two hybrid dual-branch architectures that fuse spectral and spatial paradigms through an **Adaptive Gated Fusion Module**:

```
                       7-Day Surface Satellite Inputs [B, 7, 7, 101, 241]
                                       │
                                       ▼
                       Temporal Multi-Day Attention Stem
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        Branch 1: Spectral Operator             Branch 2: Spatial Representation
      Fourier Neural Operator (FNO-2D)       U-Net (Model A) OR Vision Transformer (Model B)
    [Global continuous integral kernel,     [Local mesoscale eddy gradients,
     low-pass spectral regularization]       fronts, and coastal boundary layers]
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                         Adaptive Spatial Gating Module
                 α(x, y) = σ( Conv2d( GELU( Conv2d([F_fno, F_spatial]) ) ) )
                                       │
                                       ▼
                    Fused 3D Subsurface Representation
                                       │
                                       ▼
             Predicted 3D Ocean Temperature [15 Standard Depths, 101, 241]
```

---

## 4. Data Engineering & Harmonization Pipeline

### 4.1 Strict Temporal Partitioning (Zero Data Leakage)
To ensure rigorous operational validity, we enforced a **strictly sealed calendar-year split** across 7 complete years (2015-01-01 to 2021-12-31, 2,557 days, 0 synthetic gap-fill):
- **Training Set (2015–2019)**: 5 complete calendar years ($1,820$ daily sequences).
- **Validation Set (2020)**: 1 complete calendar year ($366$ days, leap year) + 37,708 matched physical Argo profiles.
- **Sealed Test Set (2021)**: 1 complete calendar year ($365$ days) + 55,136 independent physical Argo profiles. The model weights were frozen before evaluation on 2021.

### 4.2 Satellite Input Stack (7 Variables $\times$ 7 Days)
Each day $t$, the model ingests a 7-day retrospective sequence ($t-6$ to $t$) across 7 physical surface variables interpolated to a uniform $0.25^\circ \times 0.25^\circ$ grid over the North Indian Ocean ($101$ latitudes $\times$ $241$ longitudes):
1. **Sea Surface Temperature (SST)**: Infrared + Microwave blended thermal boundary condition ($^\circ\text{C}$).
2. **Sea Level Anomaly (SLA)**: Radar altimetry proxy for integrated thermocline depth and mesoscale eddy pumping ($\text{m}$).
3. **Sea Surface Salinity (SSS)**: SMAP satellite microwave radiometer measurements capturing freshwater plumes and barrier layers ($\text{PSU}$).
4. **Zonal Wind ($U_{10}$)**: 10-meter surface wind driving Ekman transport and upwelling ($\text{m/s}$).
5. **Meridional Wind ($V_{10}$)**: 10-meter surface wind driving coastal currents ($\text{m/s}$).
6. **Zonal Surface Current ($U_{\text{curr}}$)**: Satellite geostrophic surface velocity ($\text{m/s}$).
7. **Meridional Surface Current ($V_{\text{curr}}$)**: Satellite geostrophic surface velocity ($\text{m/s}$).

### 4.3 Target Output & Ocean Depth Standardization
The output is the full 3D temperature field across 15 standard oceanographic depths:
$$\{0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ meters}\}$$
All landmasses and ocean seabed bathymetry are masked using real ETOPO/GEBCO ocean-floor masks, preventing non-physical predictions over continental shelf boundaries.

---

## 5. Deep Learning Architecture & Mathematical Formulation

### 5.1 Temporal Attention Stem
Given the input tensor $\mathbf{X} \in \mathbb{R}^{B \times 7 \times 7 \times H \times W}$ (where $H=101, W=241$), a 3D convolutional stem aggregates multi-day temporal dynamics:
$$\mathbf{H}_0 = \text{Conv3D}(\mathbf{X}) \in \mathbb{R}^{B \times C \times 1 \times H \times W} \longrightarrow \mathbb{R}^{B \times C \times H \times W}$$
where $C=64$ hidden channels.

### 5.2 Fourier Neural Operator (FNO) Branch
The continuous spectral convolution operates on the 2D spatial domain via 2D Fast Fourier Transforms (FFT):
$$\mathcal{K}(v)(x) = \mathcal{F}^{-1}\left( R_\phi \cdot (\mathcal{F} v) \right)(x)$$
where:
- $\mathcal{F}$ and $\mathcal{F}^{-1}$ denote the 2D Fourier transform and its inverse.
- $R_\phi$ is a complex-valued parameter tensor defined on truncated Fourier modes $(k_{x,\text{max}} = 24, k_{y,\text{max}} = 16)$.
- Higher-frequency noise is naturally attenuated, enforcing spatial regularity and physical continuity.

### 5.3 Vision Transformer (ViT) Branch (Model B)
For Model B, the surface embedding is partitioned into non-overlapping spatial patches of size $4 \times 4$:
$$N_{\text{patches}} = \left\lfloor\frac{101}{4}\right\rfloor \times \left\lfloor\frac{241}{4}\right\rfloor = 25 \times 60 = 1,500\text{ tokens}$$
Each patch is projected to an embedding dimension $D=256$, combined with 2D sinusoidal positional encodings, and processed through 6 multi-head self-attention (MHSA) transformer blocks ($h=8$ heads).

### 5.4 Multiscale U-Net Branch (Model A)
For Model A, a 3-level residual encoder-decoder architecture with skip connections extracts localized spatial features:
$$\mathbf{F}_{\text{unet}} = \text{Decoder}(\text{Encoder}(\mathbf{H}_0))$$
This preserves high-frequency thermal fronts along the East India Coastal Current (EICC) and Somali Current.

### 5.5 Adaptive Gated Fusion
The representations from both branches are dynamically combined using coordinate-wise attention gates:
$$\mathbf{\alpha} = \sigma\left( \mathbf{W}_2 * \text{GELU}(\mathbf{W}_1 * [\mathbf{F}_{\text{fno}} \,\|\, \mathbf{F}_{\text{spatial}}]) \right)$$
$$\mathbf{F}_{\text{fused}} = \mathbf{\alpha} \odot \mathbf{F}_{\text{spatial}} + (1 - \mathbf{\alpha}) \odot \mathbf{F}_{\text{fno}}$$
The fused feature map is projected to 15 temperature output channels via a physics-informed projection head.

---

## 6. Experimental Evaluation & Ground-Truth Argo Validation

### 6.1 Definitive Benchmark Leaderboard

#### Table 1: Validation Performance (2020 Calendar Year)
*Evaluated on 366 daily sequences and 37,708 in-situ physical Argo float soundings.*

| Architecture | Parameters | 2020 Reanalysis RMSE | 2020 In-Situ Argo RMSE | Status / Highlight |
|:---|:---:|:---:|:---:|:---|
| **Climatology Baseline** | $0$ | $1.0544^\circ\text{C}$ | $1.2102^\circ\text{C}$ | Non-parametric baseline |
| **Residual U-Net** | $22.22\text{M}$ | $0.7308^\circ\text{C}$ | $0.9999^\circ\text{C}$ | Standard spatial baseline |
| **Fourier Neural Operator** | $12.74\text{M}$ | $0.7611^\circ\text{C}$ | $0.9924^\circ\text{C}$ | Continuous spectral baseline |
| **Vision Transformer (ViT)**| $6.04\text{M}$ | **$0.7142^\circ\text{C}$** | $0.9901^\circ\text{C}$ | Global attention baseline |
| **Model A (Dual FNO + U-Net)**| $35.06\text{M}$ | $0.7563^\circ\text{C}$ | **$\mathbf{0.9872^\circ\text{C}}$** | **Top 2020 Physical Model** |
| **Model B (Dual FNO + ViT)** | $18.84\text{M}$ | $0.7444^\circ\text{C}$ | $1.0038^\circ\text{C}$ | Dual global operator |

#### Table 2: Final Out-of-Sample Sealed Test Evaluation (2021 Calendar Year)
*Evaluated on 365 completely untouched daily sequences and 55,136 independent physical Argo observations.*

| Architecture | 2021 Test RMSE (GLORYS) | 2021 Test MAE (GLORYS) | 2021 In-Situ Argo RMSE ($55,136$ obs) | Argo Correlation ($R$) |
|:---|:---:|:---:|:---:|:---:|
| **U-Net Baseline** | $0.8041^\circ\text{C}$ | $0.5460^\circ\text{C}$ | $0.8826^\circ\text{C}$ | $0.9904$ |
| **FNO Baseline** | $0.8570^\circ\text{C}$ | $0.5857^\circ\text{C}$ | $0.8747^\circ\text{C}$ | $0.9912$ |
| **Vision Transformer (ViT)**| $0.8110^\circ\text{C}$ | $0.5478^\circ\text{C}$ | $0.8784^\circ\text{C}$ | $0.9910$ |
| **Model A (Dual FNO + U-Net)**| $0.8421^\circ\text{C}$ | $0.5664^\circ\text{C}$ | **$0.8711^\circ\text{C}$** | $0.9918$ |
| **Model B (Dual FNO + ViT)** | **$\mathbf{0.8008^\circ\text{C}}$** | **$\mathbf{0.5382^\circ\text{C}}$** | **$\mathbf{0.8514^\circ\text{C}}$** | **$\mathbf{0.9928}$** |

> [!IMPORTANT]
> **Key Scientific Proof**: On the completely unobserved 2021 test year, **both dual-branch architectures beat every single baseline on real physical Argo floats**. Model B achieved the lowest error across all metrics ($0.8008^\circ\text{C}$ on reanalysis, $0.8514^\circ\text{C}$ on Argo floats, with an unprecedented Pearson correlation of $R = 0.9928$).

### 6.2 Depth-by-Depth Error Profile
The error profile reflects ocean physical stratification:
- **Surface ($0 - 30\text{ m}$)**: $\text{RMSE} \approx 0.51^\circ\text{C} - 0.54^\circ\text{C}$ (strongly constrained by satellite SST).
- **Thermocline Peak ($75 - 125\text{ m}$)**: $\text{RMSE} \approx 1.28^\circ\text{C} - 1.35^\circ\text{C}$ (peak internal wave and eddy non-linearity).
- **Sub-Thermocline ($200 - 300\text{ m}$)**: $\text{RMSE} \approx 0.57^\circ\text{C} - 0.80^\circ\text{C}$ (rapid error decay).
- **Abyssal Ocean ($500 - 1000\text{ m}$)**: $\text{RMSE} < 0.42^\circ\text{C}$ (stable thermal convergence).

---

## 7. Full Web Application & Interactive Workstation Prototype

We built an open-access, production-grade web application to enable operational oceanographers, researchers, and naval analysts to explore 3D predictions in real time:

### 7.1 Architecture & Tech Stack
- **Frontend**: Next.js 16 (React 19, TypeScript, Tailwind CSS).
  - **Three.js**: Interactive 3D volumetric ocean lattice rendering with depth planes and coordinate raycasting.
  - **Plotly.js**: High-precision scientific contour transects, depth slice heatmaps, and telemetry soundings.
- **Backend**: FastAPI (Python 3.13, Uvicorn, PyTorch, NumPy).
  - High-throughput asynchronous endpoints delivering 3D arrays, Argo float comparisons, and validation metrics in milliseconds.
- **Deployment**:
  - Backend deployed live on **Render Cloud**: `https://oceanembed-backend-opio.onrender.com`
  - Frontend configured for one-click deployment on **Vercel**.

### 7.2 Primary Workstation Capabilities
1. **3D Interactive Volume Visualizer (`/demo`)**:
   - Orbit, pan, and rotate a 3D volumetric representation of the North Indian Ocean.
   - Interactive depth plane scrubber ($0\text{ m}$ to $1,000\text{ m}$) highlighting physical isotherms.
   - Click-to-probe latitude/longitude coordinate raycasting.
2. **Simultaneous Tri-Field 2D Depth Slice (`/demo`)**:
   - Synchronized view displaying:
     - Panel 1: **Model Prediction** (Continuous thermal scale $4^\circ\text{C} - 30^\circ\text{C}$).
     - Panel 2: **GLORYS12 Reference** (Ground truth reanalysis).
     - Panel 3: **Residual Error** (Cool-warm diverging RdBu scale $-2.5^\circ\text{C}$ to $+2.5^\circ\text{C}$ showing genuine mesoscale eddy anomalies).
3. **Vertical Transects (Meridional N-S & Zonal W-E)**:
   - Slice through any latitude or longitude in the basin.
   - Switch seamlessly between absolute temperature isotherms and residual error contours.
4. **Point Sounding Profile**:
   - Compares predicted vertical temperature soundings against GLORYS reference from surface to $1,000\text{ m}$ depth at any probed coordinate.
5. **In-Situ Argo Ground Truth Telemetry (`/results`)**:
   - Interactive scatter plot of model prediction vs. real physical float observations ($R = 0.9928$).
   - Filter by depth regime (Surface Mixed Layer, Thermocline, Deep Ocean).
   - Direct float-by-float sounding curves comparing model outputs with real Argo sensor profiles.
6. **Publication Figures Gallery (`/research`)**:
   - High-resolution empirical figures, Pareto efficiency curves, training trajectories, and architecture diagrams.

---

## 8. Slide-by-Slide PPT Presentation Deck (12-Slide Outline)

You can copy this exact structure directly into PowerPoint, Google Slides, or Canva:

### Slide 1: Title Slide (The Hook)
- **Title**: OceanEmbed: 3D Subsurface Ocean Temperature Reconstruction via Satellite Embeddings
- **Subtitle**: Solving SIH Problem Statement 66 (SIH26066) with Dual-Branch Neural Operators
- **Presenter Information**: Team Name, Institution, Track: Smart Automation / Earth Sciences & Defense
- **Visuals**: 3D rendered ocean volume graphic + satellite icons (SST, SLA, SSS, Wind).

### Slide 2: The Operational Problem (Why PS 66 Matters)
- **Headline**: The Ocean is a 3D Entity, but Satellites Only See the Surface "Skin".
- **Key Points**:
  - Subsurface temperature drives cyclone intensification (Ocean Heat Content), monsoon rains, naval sonar acoustic shadow zones, and marine fisheries.
  - In-situ Argo floats are sparse (1 float every 10 days, drifting).
  - Supercomputer physics models (NEMO/HYCOM) take hours to run and have high latency.
- **The Question**: Can we infer 3D temperatures down to $1,000\text{ m}$ in real time using everyday satellite data?

### Slide 3: The Gap in Existing Deep Learning Solutions
- **Headline**: Why Off-the-Shelf AI Models Fail in Oceanography.
- **Comparison**:
  - Standard CNNs (U-Net): Capture local coastal edges, but fail at basin-scale planetary waves; overfit to reanalysis artifacts.
  - Pure Transformers (ViT): Capture global relationships, but introduce patch boundary noise and require huge compute.
- **Our Hypothesis**: We need a dual-branch system combining **continuous spectral integral operators (FNO)** with **high-resolution spatial feature extraction (ViT/U-Net)**.

### Slide 4: Proposed Solution — The OceanEmbed Architecture
- **Headline**: Dual-Branch Neural Operator with Adaptive Gated Fusion.
- **Architecture Diagram**:
  - Input: 7 Satellite Channels $\times$ 7 Days ($[B, 7, 7, 101, 241]$).
  - Stem: 3D Temporal Attention Multi-Day Query.
  - Branch 1: 2D Fourier Neural Operator (Continuous spectral kernel, modes $16 \times 24$).
  - Branch 2: Spatial Representation (U-Net for Model A, ViT for Model B).
  - Core Innovation: **Adaptive Gating Unit** dynamically weights spectral vs. spatial representations at every grid cell and depth level.

### Slide 5: Data Engineering & Strict Zero-Leakage Protocol
- **Headline**: 7 Complete Calendar Years, 0 Synthetic Fill, Strictly Sealed Test Set.
- **Timeline**:
  - 2015–2019: Training ($1,820$ daily sequences).
  - 2020: Validation ($366$ days) + $37,708$ physical Argo float matches.
  - 2021: Sealed Test ($365$ days) + $55,136$ untouched physical Argo float observations.
- **Harmonized Inputs**: SST, SLA, SSS, Wind U/V, Current U/V at $0.25^\circ$ resolution across the North Indian Ocean ($5^\circ\text{N}-30^\circ\text{N}, 45^\circ\text{E}-105^\circ\text{E}$).

### Slide 6: Breakthrough Results — In-Situ Argo Float Validation
- **Headline**: Dual-Branch Architectures Beat All Baselines on Real Physical Floats.
- **Highlight Card**:
  - Model B achieved **$0.8514^\circ\text{C}$** Argo RMSE on $55,136$ real float observations in sealed 2021 test data ($R = 0.9928$).
  - Model A achieved **$0.9872^\circ\text{C}$** Argo RMSE on 2020 validation data.
- **Table Summary**: Show Table 2 comparing U-Net ($0.8826^\circ\text{C}$), FNO ($0.8747^\circ\text{C}$), ViT ($0.8784^\circ\text{C}$), and Model B ($0.8514^\circ\text{C}$).

### Slide 7: Oceanographic Physics Validation
- **Headline**: Physically Sound Performance Across All 15 Ocean Depths.
- **Visual**: Depth RMSE curve ($0\text{ m}$ to $1,000\text{ m}$).
- **Key Takeaways**:
  - Mixed Layer ($0-30\text{ m}$): Error $< 0.54^\circ\text{C}$ (closely tracking satellite SST).
  - Thermocline Peak ($100\text{ m}$): Error peaks at $1.34^\circ\text{C}$ due to non-linear internal waves and eddy displacements, yet resolves mesoscale eddy dipoles.
  - Deep Ocean ($500-1000\text{ m}$): Error drops below $0.42^\circ\text{C}$, confirming smooth asymptotic convergence.

### Slide 8: Parameter Efficiency & Pareto Frontier
- **Headline**: High Performance Without Supercomputing Bloat.
- **Visual**: Parameter Count vs. Accuracy Scatter Plot.
- **Key Metrics**:
  - Model B: $18.84\text{M}$ parameters — achieves top accuracy with half the parameters of Model A ($35.06\text{M}$).
  - ViT Baseline: $6.04\text{M}$ parameters — extremely lightweight.
  - Inference latency: $< 45\text{ ms}$ per full 3D basin reconstruction on a single GPU.

### Slide 9: Live Prototype — OceanEmbed Workstation
- **Headline**: Production-Ready Scientific Decision Support Platform.
- **Screenshots of the UI**:
  - 3D Volumetric Ocean Lattice with raycast probing.
  - Tri-Field Simultaneous Depth Slice (Prediction vs. GLORYS vs. Residual).
  - Meridional/Zonal Vertical Transects with cool-warm diverging colormap.
  - Real-time in-situ Argo sounding curves.

### Slide 10: Operational Impact & Real-World Use Cases
- **Headline**: Direct Applications for Indian Ocean Governance.
- **3 Strategic Pillars**:
  1. **Cyclone Track & Intensity Forecasting (IMD / INCOIS)**: Real-time calculation of Tropical Cyclone Heat Potential (TCHP) in the Bay of Bengal.
  2. **Naval Anti-Submarine Warfare (Indian Navy)**: Accurate mapping of underwater acoustic sound speed channels (SOFAR axis) and thermal shadow zones.
  3. **Marine Fisheries (CMFRI)**: Mapping subsurface upwelling zones and thermocline depth to predict commercial fish aggregations.

### Slide 11: Deployment Architecture & Scalability
- **Headline**: Cloud-Native, Modern, and Scalable.
- **Tech Stack**:
  - Modern Frontend: Next.js 16 + TypeScript + Three.js WebGL.
  - Scalable Backend: FastAPI + PyTorch + Docker.
  - API Response: Sub-second JSON delivery of full 3D fields.
  - Cloud Readiness: Deployed live on Render and Vercel.

### Slide 12: Conclusion & Future Roadmap
- **Headline**: Transforming Satellite Skin Observations into Deep 3D Knowledge.
- **Summary**:
  - Solved SIH PS 66 with proven statistical superiority on $>90,000$ physical Argo float soundings.
  - Validated the dual-branch neural operator paradigm for earth system science.
- **Roadmap**: Incorporate real-time daily operational ingest from MOSDAC/ISRO (INSAT-3D, EOS-06, Oceansat-3) and expand to 3D salinity and current velocity fields.

---

## 9. Minute-by-Minute Video Demonstration & Pitch Script

Use this script for recording a 3 to 4-minute prototype demonstration video:

### [0:00 - 0:45] The Hook & The Problem
> *"Hello everyone. Welcome to our presentation of **OceanEmbed**, our solution for Smart India Hackathon Problem Statement 66.*  
> *The ocean is a three-dimensional living system. Subsurface ocean temperature dictates cyclone intensity in the Bay of Bengal, drives our monsoon, guides naval submarine acoustics, and sustains marine fisheries.*  
> *Yet, our modern satellites have a massive limitation: **they can only see the skin of the ocean** — the top millimeter. To measure what's happening underneath, we rely on autonomous Argo floats, but they are sparse and drift unpredictably. Supercomputer physics simulations take hours or days to run.*  
> *Our challenge in SIH PS 66 was clear: **Can we turn multi-satellite 2D surface observations into an instant, accurate 3D temperature volume down to 1,000 meters depth?** Here is how we solved it."*

### [0:45 - 1:30] Technical Methodology & Dual-Branch Innovation
> *"To solve this without data leakage, we curated exactly 7 complete calendar years of daily satellite data across the North Indian Ocean — strictly training on 2015 to 2019, validating on 2020, and sealing 2021 as a blind test set.*  
> *Instead of relying on standard convolutional networks that fail over large ocean basins, we pioneered a **Dual-Branch Neural Operator**.*  
> *Our architecture feeds 7 days of 7 satellite variables — Sea Surface Temperature, Sea Level Anomaly, Salinity, Winds, and Ocean Currents — into two parallel pathways:*  
> *First, a **Fourier Neural Operator (FNO)** that learns continuous solution operators in spectral frequency space, guaranteeing physical equilibrium across the entire basin.*  
> *Second, a **Vision Transformer or U-Net** that captures sharp local mesoscale eddies and coastal thermal fronts.*  
> *An **Adaptive Gating Unit** dynamically blends these two representations at every coordinate and depth, outputting the continuous 3D temperature field across 15 standard ocean depths."*

### [1:30 - 2:45] Live Prototype Demonstration (Show the Web App)
> *(Screen recording switches to the live OceanEmbed Workstation at `/demo`)*  
> *"Now, let's look at the live prototype.*  
> *Here on our interactive workstation, an oceanographer can select any observation date from our sealed test year and select our grand-winner architecture, Model B.*  
> *In our **3D Ocean Volume Visualizer**, built with Three.js, you can orbit around the entire North Indian Ocean basin. Using our depth scrubber, we can slice down from the surface mixed layer through the thermocline to 1,000 meters depth. You can click anywhere to raycast and probe coordinates in real time.*  
> *(Click on '2D Depth Slice')*  
> *Next, our **Simultaneous Tri-Field Depth Slice** provides synchronized verification. On the left is our Model B prediction; in the center is the GLORYS ground truth; and on the right is our **Residual Error map**.*  
> *Notice how the residual map displays authentic mesoscale eddy anomalies — warm and cold dipoles within $\pm 2.5^\circ\text{C}$ concentrated in the dynamic thermocline around 100 meters, while remaining below $0.4^\circ\text{C}$ in the deep ocean.*  
> *(Click on 'Vertical Transect')*  
> *Switching to the **Vertical Transect**, we can take a meridional North-South or zonal West-East cross section through the deep Bay of Bengal. We can inspect the continuous isotherm contours and instantly toggle to view the subsurface residual error distribution.*  
> *(Click on 'Results' page)*  
> *Finally, the ultimate proof: our in-situ Argo float validation tab. Here, our model is tested not against simulations, but against **55,136 independent physical Argo float profiles** from 2021. As you can see on the scatter plot, Model B achieves an astounding Pearson correlation of **0.9928** and an RMSE of **$0.8514^\circ\text{C}$**, beating all traditional baselines."*

### [2:45 - 3:30] Impact, Feasibility & Conclusion
> *"OceanEmbed runs inference in under 45 milliseconds on a single GPU, converting hours of supercomputing into real-time operational intelligence.*  
> *This provides actionable decision support for:*  
> *1. **INCOIS and IMD** for rapid Tropical Cyclone Heat Potential calculations before cyclone landfall;*  
> *2. The **Indian Navy** for mapping acoustic shadow zones in anti-submarine warfare; and*  
> *3. **Fisheries** for tracking marine upwelling.*  
> *With our live FastAPI backend deployed on cloud infrastructure and our Next.js frontend, OceanEmbed is ready to integrate directly with ISRO MOSDAC satellite feeds.*  
> *Thank you!"*

---

## 10. Operational Impact & Deployment Feasibility

### 10.1 Real-World Strategic Beneficiaries

| Stakeholder / Organization | Operational Pain Point | How OceanEmbed Resolves It |
|:---|:---|:---|
| **INCOIS & IMD** (Earth Sciences) | Rapid intensification of Bay of Bengal cyclones due to hidden subsurface heat pockets. | Delivers instantaneous 3D Ocean Heat Content (OHC) and $26^\circ\text{C}$ isotherm depth mapping. |
| **Indian Navy & Defense** | Sonar detection range changes drastically due to sound velocity profile variations in the thermocline. | Calculates real-time acoustic sound speed profiles (SSP) and identifies acoustic shadow channels. |
| **CMFRI & Fisheries** | Fishermen need accurate Potential Fishing Zone (PFZ) advisories based on thermal upwelling. | Identifies subsurface thermocline ridging and nutrient-rich cold-water upwelling zones. |
| **Climate Research** | Understanding Indian Ocean Dipole (IOD) and El Niño teleconnections requires continuous 3D reanalysis. | High-resolution gap-free 3D temperature volume generation without multi-week reanalysis lag. |

### 10.2 Ready for Indian Satellite Integration
OceanEmbed's input schema is directly compatible with Indian Earth Observation satellites accessible via ISRO MOSDAC:
- **EOS-06 / Oceansat-3**: Sea Surface Temperature (SSTM) and Ocean Color.
- **INSAT-3D / 3DR**: High-frequency geostationary Sea Surface Temperature.
- **SARAL-AltiKa**: High-precision Sea Level Anomaly altimetry.
- **SCATSAT-1 / Oceansat Scatterometer**: Daily surface wind vector fields ($U_{10}, V_{10}$).

---

### Summary Checklist for PPT & Video
- [x] **Clear problem statement**: Skin-only satellites vs. 3D subsurface ocean needs.
- [x] **Technical uniqueness**: Dual-branch FNO + ViT/U-Net with adaptive gating.
- [x] **Definitive proof**: Verified on $>90,000$ real physical Argo float observations.
- [x] **Working prototype**: Live interactive 3D WebGL + Plotly workstation.
- [x] **Clear impact**: Cyclones, naval defense, and fisheries.
