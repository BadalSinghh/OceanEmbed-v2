"use client";

import React, { useState } from "react";
import Link from "next/link";

interface LayerDetail {
  id: string;
  name: string;
  tensorIn: string;
  tensorOut: string;
  params: string;
  summary: string;
  operations: string[];
  mathEquation?: string;
}

const ARCHITECTURE_STAGES: LayerDetail[] = [
  {
    id: "obs",
    name: "7-Channel Surface Observables",
    tensorIn: "CMEMS + ERA5 L4",
    tensorOut: "[B, 7, 101, 241]",
    params: "Raw Observables",
    summary:
      "Seven daily spaceborne surface fields across the North Indian Ocean (5°–30°N, 45°–105°E) at 0.25° grid resolution. Normalised using empirical training statistics from the 2015–2019 baseline period.",
    operations: [
      "Sea Surface Temperature (OSTIA L4, °C)",
      "Sea Surface Salinity (CMEMS Multi-Obs L4, psu)",
      "Sea Level Anomaly (DUACS L4 Altimetry, m)",
      "Zonal Geostrophic Current U (derived from SLA, m/s)",
      "Meridional Geostrophic Current V (derived from SLA, m/s)",
      "Zonal Wind Stress U (ERA5 / Scatterometer, m/s)",
      "Meridional Wind Stress V (ERA5 / Scatterometer, m/s)",
    ],
  },
  {
    id: "projection",
    name: "Surface Latent Projection",
    tensorIn: "[B, 7, 101, 241]",
    tensorOut: "[B, 64, 101, 241]",
    params: "14,784 parameters",
    summary:
      "Multiscale 2D convolutional embedding layers extract localized spatial gradients, mapping surface anomalies into a 64-channel latent feature manifold preserving coastal bathymetry.",
    operations: [
      "Conv2d(7 → 64, kernel=3×3, padding=1) + BatchNorm2d + GELU",
      "Residual identity projection with learnable channel weights",
      "Normalised spatial grid coordinate concatenation [lat, lon]",
    ],
  },
  {
    id: "fno",
    name: "Branch 1: Fourier Neural Operator (FNO)",
    tensorIn: "[B, 64, 101, 241]",
    tensorOut: "[B, 64, 101, 241]",
    params: "2,352,271 parameters",
    summary:
      "Four SpectralConv2d blocks parameterized with 16 truncated Fourier modes evaluate continuous Green's functions in frequency space, capturing basin-scale baroclinic waves.",
    operations: [
      "Forward 2D Real FFT: x_ft = rfft2(x, dim=(-2, -1))",
      "Mode Truncation: 16 lowest Fourier frequency modes along both spatial dimensions",
      "Complex Tensor Multiplication: out_ft = compl_mul(x_ft, W_spectral)",
      "Inverse 2D Real FFT: x_out = irfft2(out_ft, s=(101, 241))",
      "Local 1×1 Conv2d bypass + GELU non-linear activation",
    ],
    mathEquation: "𝒦(v)(x) = ℱ⁻¹( R · (ℱv) )(x) + W · v(x)",
  },
  {
    id: "vit",
    name: "Branch 2: Vision Transformer (ViT)",
    tensorIn: "[B, 64, 101, 241]",
    tensorOut: "[B, 64, 101, 241]",
    params: "4,065,552 parameters",
    summary:
      "Multi-head self-attention operating over 16×16 spatial patches captures long-range teleconnections between equatorial current jets, the Somali current, and the Bay of Bengal freshwater pool.",
    operations: [
      "Patch Extraction: 16×16 non-overlapping spatial patches with linear projection",
      "Learnable 1D spatial positional embeddings",
      "Multi-Head Self-Attention (8 attention heads, hidden dim=256)",
      "MLP feedforward expansion blocks with LayerNorm",
      "Bilinear spatial reconstruction to native [101, 241] grid",
    ],
    mathEquation: "Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V",
  },
  {
    id: "gating",
    name: "Adaptive Spatial Gating Fusion",
    tensorIn: "FNO + ViT Features",
    tensorOut: "[B, 64, 101, 241]",
    params: "8,320 parameters",
    summary:
      "A learned spatial gating mechanism dynamically balances global spectral operator representations and localized attention features at every ocean grid point.",
    operations: [
      "Concatenation of spectral FNO and attention ViT feature maps",
      "Spatial gating convolution: W_gate = sigmoid(Conv2d(128 → 64, kernel=1×1))",
      "Convex combination: F_fused = W_gate ⊙ F_fno + (1 - W_gate) ⊙ F_vit",
    ],
    mathEquation: "F_fused(x) = σ(W_g · [F_fno, F_vit]) ⊙ F_fno + (1 - σ(·)) ⊙ F_vit",
  },
  {
    id: "film",
    name: "Continuous FiLM Depth Conditioning",
    tensorIn: "[B, 64, 101, 241] + Depth z",
    tensorOut: "[B, 15, 101, 241]",
    params: "348,680 parameters",
    summary:
      "Feature-wise Linear Modulation (FiLM) maps physical target depths (0 to 1000m) into continuous affine scale γ(z) and shift β(z) vectors, allowing exact vertical column reconstruction.",
    operations: [
      "Continuous depth embedding: MLP(z) → (γ(z), β(z))",
      "Feature modulation: h_depth = γ(z) ⊙ F_fused + β(z)",
      "Output projection: Conv2d(64 → 1) evaluated across 15 standard depths",
      "Physical bathymetric land-mask application",
    ],
    mathEquation: "FiLM(h, z) = γ(z) ⊙ h + β(z)",
  },
];

export default function ArchitecturePage() {
  const [expandedStage, setExpandedStage] = useState<string>("fno");

  return (
    <div className="w-full min-h-screen bg-[#050505] text-neutral-100 antialiased font-sans pb-36">
      {/* ── Editorial Header ────────────────────────────────────────── */}
      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 py-24 rule-b">
        <div className="max-w-4xl space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>NEURAL OPERATOR THEORY & DUAL-BRANCH ARCHITECTURE</span>
          </div>
          <h1 className="text-4xl sm:text-6xl font-light text-white tracking-tight uppercase">
            HOW OCEANEMBED RECONSTRUCTS THE WATER COLUMN
          </h1>
          <p className="text-lg sm:text-xl text-neutral-300 font-sans leading-relaxed pt-2 max-w-3xl">
            A comprehensive visual explanation of the dual-branch neural operator pipeline that inverts
            multi-sensor satellite surface boundary observations into the full 3D subsurface temperature volume.
          </p>

          <div className="pt-8 border-t border-white/[0.08] flex flex-wrap items-center gap-12 font-mono text-xs text-neutral-400">
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Model B (Winner) Parameters</span>
              <span className="text-white mt-0.5 block font-semibold">6,432,607 parameters</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Operator Formulation</span>
              <span className="text-white mt-0.5 block">Dual-Branch FNO2D + ViT / U-Net</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Input to Output</span>
              <span className="text-white mt-0.5 block">[7, 101, 241] → [15, 101, 241]</span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 mt-20 space-y-24">
        {/* ── LARGE OPEN ARCHITECTURE DIAGRAM ─────────────────────────── */}
        <section className="space-y-8">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest pb-3 rule-b">
            NEURAL RECONSTRUCTION ARCHITECTURE STAGES
          </div>

          <div className="space-y-6">
            {ARCHITECTURE_STAGES.map((st, i) => {
              const isExpanded = expandedStage === st.id;
              return (
                <div key={st.id} className="rule-b pb-6 space-y-3">
                  <div
                    onClick={() => setExpandedStage(isExpanded ? "" : st.id)}
                    className="flex flex-col md:flex-row md:items-center justify-between gap-4 cursor-pointer py-2 group"
                  >
                    <div className="flex items-baseline gap-6">
                      <span className="font-mono text-xs text-neutral-500 group-hover:text-white transition-colors">
                        0{i + 1}
                      </span>
                      <h3 className="text-xl sm:text-2xl font-light text-white group-hover:text-neutral-200 transition-colors">
                        {st.name}
                      </h3>
                    </div>

                    <div className="flex items-center gap-6 font-mono text-xs text-neutral-400">
                      <span className="text-neutral-500">{st.tensorIn} →</span>
                      <span className="text-white font-medium">{st.tensorOut}</span>
                      <span className="text-neutral-600">{st.params}</span>
                      <span className="text-neutral-400 group-hover:text-white transition-colors">
                        {isExpanded ? "−" : "+"}
                      </span>
                    </div>
                  </div>

                  {/* Expandable Technical Details */}
                  {isExpanded && (
                    <div className="pl-0 md:pl-12 pt-4 space-y-4 text-sm font-sans text-neutral-300">
                      <p className="max-w-3xl leading-relaxed">{st.summary}</p>

                      {st.mathEquation && (
                        <div className="p-4 bg-[#0a0a0a] rule-b rule-t font-mono text-xs text-neutral-200">
                          <span className="text-neutral-500 uppercase text-[10px] block mb-1">
                            Mathematical Formulation
                          </span>
                          <span className="text-white font-semibold text-sm">{st.mathEquation}</span>
                        </div>
                      )}

                      <div className="font-mono text-xs text-neutral-400 space-y-1.5 pt-2">
                        {st.operations.map((op, idx) => (
                          <div key={idx} className="flex items-center gap-3">
                            <span className="text-neutral-600">—</span>
                            <span>{op}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* ── IMPLEMENTED MODEL ABLATION COMPARISON ────────────────────── */}
        <section className="space-y-6">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest pb-3 rule-b">
            BENCHMARKED ARCHITECTURES & MODEL CHECKPOINTS
          </div>

          <table className="editorial-table font-mono text-xs">
            <thead>
              <tr>
                <th>Model</th>
                <th>Architecture Paradigm</th>
                <th>Parameters</th>
                <th>Checkpoint Weight</th>
                <th>2021 Argo RMSE</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-l-2 border-cyan-400 bg-white/[0.03]">
                <td className="text-white font-semibold font-sans">
                  Model B (Dual-Branch FNO + ViT) <span className="text-cyan-400 text-[10px]">★ WINNER</span>
                </td>
                <td className="text-neutral-200 font-sans">Spectral Integral Kernel + Self-Attention</td>
                <td className="text-white font-bold">6,432,607</td>
                <td>best_fno_vit_model_b.pth (~25.7 MB)</td>
                <td className="text-cyan-400 font-bold">0.8514 °C</td>
              </tr>
              <tr>
                <td className="text-neutral-200 font-sans">Model A (Dual-Branch FNO + U-Net)</td>
                <td className="text-neutral-400 font-sans">Spectral Operator + Multiscale Skip U-Net</td>
                <td>1,842,575</td>
                <td>best_fno_unet_model_a.pth (~7.4 MB)</td>
                <td className="text-neutral-200 font-semibold">0.8711 °C</td>
              </tr>
              <tr>
                <td className="text-neutral-300 font-sans">Vision Transformer (ViT) Baseline</td>
                <td className="text-neutral-400 font-sans">Patch Attention Baseline</td>
                <td>4,723,215</td>
                <td>best_vit_baseline.pth (~18.9 MB)</td>
                <td>0.8784 °C</td>
              </tr>
              <tr>
                <td className="text-neutral-300 font-sans">U-Net Baseline</td>
                <td className="text-neutral-400 font-sans">Encoder-Decoder Convolutional Baseline</td>
                <td>1,940,303</td>
                <td>best_unet_baseline.pth (~7.8 MB)</td>
                <td>0.8826 °C</td>
              </tr>
              <tr>
                <td className="text-neutral-400 font-sans">Fourier Neural Operator (FNO) Baseline</td>
                <td className="text-neutral-400 font-sans">Pure Spectral Operator Baseline</td>
                <td>2,367,055</td>
                <td>best_fno_baseline.pth (~9.5 MB)</td>
                <td>0.8747 °C</td>
              </tr>
            </tbody>
          </table>
        </section>

        {/* ── CTA ─────────────────────────────────────────────────────── */}
        <div className="pt-12 rule-t flex flex-wrap items-center justify-between gap-6">
          <div>
            <div className="text-xl font-light text-white">Evaluate the reconstruction live</div>
            <div className="text-sm text-neutral-400 font-sans mt-1">
              Select any 2021 test date to inspect the 3D inferred temperature field.
            </div>
          </div>
          <Link href="/demo" className="btn-research-primary">
            <span>Launch Reconstruction Lab</span>
            <span>→</span>
          </Link>
        </div>
      </div>
    </div>
  );
}
