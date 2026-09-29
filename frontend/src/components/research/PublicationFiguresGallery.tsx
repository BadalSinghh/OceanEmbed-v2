"use client";

import React, { useState } from "react";

const PUBLICATION_FIGURES = [
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
      "Pareto frontier evaluating architectural parameter efficiency against reconstruction error. Model B establishes the state-of-the-art accuracy frontier at 6.43M trainable parameters.",
  },
];

export default function PublicationFiguresGallery() {
  const [activeTab, setActiveTab] = useState<string>("depth");
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
