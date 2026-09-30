"use client";

import React, { useState } from "react";

const PUBLICATION_FIGURES = [
  {
    id: "master_dashboard",
    title: "Master Benchmark Dashboard (6 Panels)",
    subtitle: "Complete multi-dimensional evaluation summary across all models, depths, and in-situ Argo observations",
    src: "/figures/sih_ps66_master_evaluation_dashboard.png",
    caption:
      "Comprehensive 6-panel benchmark summary for SIH PS66: (A) Sealed 2021 GLORYS vs In-Situ Argo RMSE; (B) Vertical error trajectories from surface to 1000m depth; (C) Physical Argo float correlation scatter (R = 0.9928); (D) Parameter efficiency Pareto frontier; (E) Thermocline 100m error probability density; (F) Vertical oceanographic regime comparison.",
  },
  {
    id: "spatial_showcase",
    title: "Surface-to-Subsurface 3D Showcase",
    subtitle: "Multi-satellite input stack, 100m thermocline reconstruction, and vertical transect cross-section",
    src: "/figures/sih_ps66_spatial_reconstruction_showcase.png",
    caption:
      "Physical reconstruction showcase across the North Indian Ocean basin: Panels 1-2 show satellite Sea Surface Temperature (SST) and Sea Level Anomaly (SLA); Panel 3 displays the 88°E vertical meridional transect; Panels 4-6 compare Model B predicted 100m thermal field against GLORYS reference and the resulting cool-warm mesoscale eddy residual map.",
  },
  {
    id: "combined_ab_correlation",
    title: "Combined Model A & B Correlation",
    subtitle: "Direct side-by-side predicted temperature vs in-situ Argo float measurement",
    src: "/figures/combined_model_a_b_argo_correlation.png",
    caption:
      "Combined head-to-head correlation graph comparing Model A (Dual FNO + U-Net, R = 0.9918) and Model B (Dual FNO + ViT, R = 0.9928) against 55,136 independent in-situ Argo float observations. Inset panels show the tighter residual error variance of Model B and vertical depth-by-depth correlation trajectories.",
  },
  {
    id: "depth",
    title: "Vertical Depth RMSE Comparison",
    subtitle: "Error trajectories across 15 standard depths (0 to 1000m) for all 6 models",
    src: "/figures/depth_rmse_comparison.png",
    caption:
      "Depth-dependent root-mean-square error (RMSE) evaluated over the North Indian Ocean basin. In the thermocline layer (75–125m), baroclinic variability causes error peaks across all models. Dual-Branch Model B maintains the lowest error profile throughout the column.",
  },
  {
    id: "argo",
    title: "In-Situ Argo Sounding Validation",
    subtitle: "Scatter correlation and residual distribution vs autonomous profiling floats",
    src: "/figures/argo_insitu_comparison.png",
    caption:
      "Physical ground-truth validation against 37,708 in-situ autonomous profiling Argo CTD floats. Model B achieves an in-situ RMSE of 0.8514 °C with high Pearson correlation (R = 0.9928) across 55,136 depth soundings.",
  },
  {
    id: "test",
    title: "2021 Sealed Out-of-Sample Benchmark",
    subtitle: "Overall GLORYS reanalysis vs Argo in-situ test metrics",
    src: "/figures/test_2021_final_comparison.png",
    caption:
      "Final out-of-sample benchmark on the sealed 2021 test set (365 unseen consecutive days). Model B outperforms the standard climatology baseline by 29.3% and surpasses single-branch operators in deep-ocean generalization.",
  },
  {
    id: "pareto",
    title: "Parameter-Accuracy Pareto Frontier",
    subtitle: "Model parameter efficiency vs physical in-situ accuracy",
    src: "/figures/parameter_accuracy_pareto.png",
    caption:
      "Pareto frontier evaluating architectural parameter efficiency against reconstruction error. Model B establishes the state-of-the-art accuracy frontier at 18.84M trainable parameters.",
  },
];

export default function PublicationFiguresGallery() {
  const [activeTab, setActiveTab] = useState<string>("master_dashboard");
  const activeFig = PUBLICATION_FIGURES.find((f) => f.id === activeTab) || PUBLICATION_FIGURES[0];

  return (
    <div className="space-y-6">
      {/* Interactive Tabs */}
      <div className="flex flex-wrap gap-2 pb-4 rule-b font-mono text-xs">
        {PUBLICATION_FIGURES.map((fig) => {
          const isSelected = fig.id === activeTab;
          return (
            <button
              key={fig.id}
              onClick={() => setActiveTab(fig.id)}
              className={`px-4 py-2 transition-colors cursor-pointer text-left ${
                isSelected
                  ? "bg-white text-black font-semibold shadow-sm"
                  : "bg-white/[0.04] text-neutral-400 hover:text-white hover:bg-white/[0.08]"
              }`}
            >
              {fig.title}
            </button>
          );
        })}
      </div>

      {/* Figure Frame */}
      <div className="w-full bg-[#080808] border border-white/10 p-6 md:p-8 space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 font-mono text-xs">
          <span className="text-white font-medium">{activeFig.title}</span>
          <span className="text-neutral-500">{activeFig.subtitle}</span>
        </div>

        <div className="w-full relative flex items-center justify-center bg-black/60 rounded border border-white/[0.06] overflow-hidden p-2 md:p-4">
          <img
            src={activeFig.src}
            alt={activeFig.title}
            className="w-full max-h-[720px] object-contain rounded"
          />
        </div>

        <p className="text-neutral-400 text-xs font-sans leading-relaxed pt-2">
          {activeFig.caption}
        </p>
      </div>
    </div>
  );
}
