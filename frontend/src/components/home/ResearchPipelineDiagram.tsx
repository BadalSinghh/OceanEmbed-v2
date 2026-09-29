"use client";

import React, { useState } from "react";

interface PipelineStage {
  id: string;
  step: string;
  title: string;
  tensor: string;
  subtitle: string;
  description: string;
}

const STAGES: PipelineStage[] = [
  {
    id: "obs",
    step: "01",
    title: "Satellite Observations",
    tensor: "[7, 101, 241]",
    subtitle: "Multi-Sensor Surface Products",
    description: "Daily L4 gridded observations of SST, SSS, SLA, geostrophic current vectors (U, V), and ERA5 wind stress fields across the North Indian Ocean at 0.25° resolution.",
  },
  {
    id: "repr",
    step: "02",
    title: "Surface Representation",
    tensor: "[64, 101, 241]",
    subtitle: "Dual-Branch Projection",
    description: "Spatial convolutional projection transforms 7 multi-sensor surface channels into a 64-dimensional latent embedding space preserving coastal boundary topology.",
  },
  {
    id: "fno",
    step: "03",
    title: "Spectral Operator Branch",
    tensor: "FNO2D (16 Modes)",
    subtitle: "Fourier Neural Operator",
    description: "Four SpectralConv2d blocks with 16 truncated Fourier modes evaluate parameterised continuous integral kernels in frequency space, resolving mesh-invariant flow physics.",
  },
  {
    id: "vit",
    step: "04",
    title: "Self-Attention Branch",
    tensor: "ViT (Patch 16×16)",
    subtitle: "Vision Transformer",
    description: "Multi-head self-attention captures long-range teleconnections between equatorial Kelvin wave propagation, Somali current jets, and Bay of Bengal freshwater lenses.",
  },
  {
    id: "field",
    step: "05",
    title: "3D Temperature Field",
    tensor: "[15, 101, 241]",
    subtitle: "Continuous Water Column (0–1000 m)",
    description: "Reconstructed full 3D temperature tensor across 15 standard depth horizons conditioned via continuous FiLM depth embeddings, resolving the mixed layer and sharp thermocline.",
  },
  {
    id: "argo",
    step: "06",
    title: "Argo In-Situ Validation",
    tensor: "37,708 Profiles",
    subtitle: "CORA / INCOIS Delayed-Mode Floats",
    description: "Rigorous empirical validation against 37,708 independent autonomous profiling CTD floats, achieving 0.8514 °C RMSE on 55,136 held-out 2021 test soundings.",
  },
];

export default function ResearchPipelineDiagram() {
  const [activeStageId, setActiveStageId] = useState<string>("recon");
  const active = STAGES.find((s) => s.id === activeStageId) || STAGES[3];

  return (
    <div className="w-full space-y-10">
      {/* 6-Stage Open Horizontal Grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 divide-y md:divide-y-0 md:divide-x divide-white/[0.08] rule-b rule-t">
        {STAGES.map((st) => {
          const isSelected = st.id === activeStageId;
          return (
            <button
              key={st.id}
              onClick={() => setActiveStageId(st.id)}
              className={`p-6 text-left transition-colors cursor-pointer ${
                isSelected ? "bg-white/[0.05]" : "hover:bg-white/[0.02]"
              }`}
            >
              <div className="font-mono text-[11px] text-neutral-500 mb-2">
                STAGE {st.step}
              </div>
              <div
                className={`text-sm font-medium leading-snug ${
                  isSelected ? "text-white" : "text-neutral-300"
                }`}
              >
                {st.title}
              </div>
              <div className="font-mono text-xs text-neutral-500 mt-4 truncate">
                {st.tensor}
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Stage Detail Inspector */}
      <div className="py-6 px-8 bg-[#0a0a0a] rule-b rule-t flex flex-col md:flex-row items-start justify-between gap-8">
        <div className="max-w-2xl space-y-2">
          <div className="font-mono text-xs text-cyan-400 uppercase tracking-wider">
            Stage {active.step}: {active.title} — {active.subtitle}
          </div>
          <p className="text-neutral-300 text-sm font-sans leading-relaxed">
            {active.description}
          </p>
        </div>
        <div className="font-mono text-xs text-neutral-400 shrink-0 border-t md:border-t-0 md:border-l border-white/[0.08] pt-4 md:pt-0 md:pl-8">
          <div className="text-neutral-500 uppercase text-[9px]">Active Tensor Dimension</div>
          <div className="text-neutral-100 text-sm font-semibold mt-1">{active.tensor}</div>
        </div>
      </div>
    </div>
  );
}
