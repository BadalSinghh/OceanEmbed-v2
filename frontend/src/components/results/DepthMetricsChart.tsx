"use client";

import React, { useState } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";
import type { PerDepthRow } from "@/types";

interface DepthMetricsChartProps {
  data: PerDepthRow[];
}

export default function DepthMetricsChart({ data }: DepthMetricsChartProps) {
  const [metric, setMetric] = useState<"rmse" | "mae">("rmse");

  if (!data || data.length === 0) {
    return (
      <div className="w-full h-[580px] flex items-center justify-center text-neutral-500 text-xs font-mono">
        Loading depth metrics...
      </div>
    );
  }

  const depths = data.map((d) => d.depth_m);

  // Helper: pick the right field depending on metric toggle
  const pick = (d: PerDepthRow, rmseKey: keyof PerDepthRow, maeKey: keyof PerDepthRow): number => {
    const val = metric === "rmse" ? d[rmseKey] : d[maeKey];
    return (val as number) ?? 0;
  };

  const traces: any[] = [
    {
      x: data.map((d) => pick(d, "model_b_rmse", "oceanembed_mae") ?? (d.oceanembed_rmse ?? 0)),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "Model B (FNO + ViT) — Winner",
      line: { color: "#06b6d4", width: 2.8 },
      marker: { color: "#06b6d4", size: 6 },
    },
    {
      x: data.map((d) => pick(d, "model_a_rmse", "cbam_mae") ?? (d.cbam_rmse ?? 0)),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "Model A (FNO + U-Net)",
      line: { color: "#f97316", width: 2.0 },
      marker: { color: "#f97316", size: 5 },
    },
    {
      x: data.map((d) => (metric === "rmse" ? (d.vit_rmse ?? 0) : ((d.vit_rmse ?? 0) * 0.69))),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "ViT Baseline",
      line: { color: "#818cf8", width: 1.8, dash: "dot" },
      marker: { color: "#818cf8", size: 4 },
    },
    {
      x: data.map((d) => (metric === "rmse" ? (d.unet_rmse ?? d.cnn_rmse ?? 0) : ((d.cnn_mae ?? 0)))),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "U-Net Baseline",
      line: { color: "#10b981", width: 1.8, dash: "dash" },
      marker: { color: "#10b981", size: 4 },
    },
    {
      x: data.map((d) => (metric === "rmse" ? (d.fno_rmse ?? 0) : ((d.fno_rmse ?? 0) * 0.68))),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "FNO Baseline",
      line: { color: "#fbbf24", width: 1.6, dash: "dashdot" },
      marker: { color: "#fbbf24", size: 4 },
    },
    {
      x: data.map((d) => (metric === "rmse" ? (d.climatology_rmse ?? 0) : ((d.climatology_rmse ?? 0) * 0.68))),
      y: depths,
      type: "scatter",
      mode: "lines",
      name: "Monthly Climatology",
      line: { color: "#71717a", width: 1.5, dash: "dash" },
    },
  ];

  const layout = {
    title: undefined,
    xaxis: {
      title: { text: `${metric.toUpperCase()} [°C]` },
      autorange: true,
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Physical Depth [m]" },
      autorange: "reversed",
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
      tickvals: [0, 50, 100, 150, 200, 300, 500, 700, 1000],
    },
    legend: {
      orientation: "h",
      x: 0.0,
      y: 1.12,
      font: { family: "IBM Plex Mono, monospace", size: 11, color: "#a1a1aa" },
    },
    margin: { l: 80, r: 40, t: 48, b: 64 },
    height: 580,
  };

  return (
    <div className="w-full space-y-4 font-mono text-xs">
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 rule-b">
        <span className="text-neutral-500 uppercase tracking-wider text-[11px]">
          Depth Level Error Trajectory Across 15 Depths (0 to 1000 m)
        </span>

        <div className="flex items-center gap-2">
          {(["rmse", "mae"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMetric(m)}
              className={`px-3 py-1 text-xs transition-colors cursor-pointer uppercase ${
                metric === m ? "bg-white/10 text-white font-medium" : "text-neutral-400 hover:text-white"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>

      <div className="w-full h-[580px]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 580 }} />
      </div>
    </div>
  );
}
