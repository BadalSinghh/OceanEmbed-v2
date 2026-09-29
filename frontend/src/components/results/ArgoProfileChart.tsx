"use client";

import React from "react";
import PlotlyChart from "@/components/common/PlotlyChart";
import type { ArgoObservation } from "@/types";

interface ArgoProfileChartProps {
  observations: ArgoObservation[];
}

export default function ArgoProfileChart({ observations }: ArgoProfileChartProps) {
  if (!observations || observations.length === 0) {
    return (
      <div className="w-full h-[580px] flex items-center justify-center text-neutral-500 text-xs font-mono">
        Select a float to view its vertical sounding profile.
      </div>
    );
  }

  const sorted = [...observations].sort((a, b) => (a.depth_m ?? 0) - (b.depth_m ?? 0));
  const depths = sorted.map((o) => o.depth_m);

  const traces: any[] = [
    {
      x: sorted.map((o) => o.obs_temp),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "In-Situ Argo Sounding (Ground Truth)",
      line: { color: "#ffffff", width: 2.8 },
      marker: { color: "#ffffff", size: 6 },
    },
    {
      x: sorted.map((o) => o.oe_temp),
      y: depths,
      type: "scatter",
      mode: "lines+markers",
      name: "Model B (FNO + ViT) — Winner",
      line: { color: "#06b6d4", width: 2.6 },
      marker: { color: "#06b6d4", size: 5 },
    },
    {
      x: sorted.map((o) => o.cbam_temp),
      y: depths,
      type: "scatter",
      mode: "lines",
      name: "Model A (FNO + U-Net)",
      line: { color: "#f97316", width: 1.8, dash: "dot" },
    },
    {
      x: sorted.map((o) => o.glorys_temp),
      y: depths,
      type: "scatter",
      mode: "lines",
      name: "GLORYS12 Reanalysis",
      line: { color: "#fb923c", width: 1.8, dash: "dash" },
    },
  ];

  const layout = {
    title: undefined,
    xaxis: {
      title: { text: "Temperature [°C]" },
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Sounding Depth [m]" },
      autorange: "reversed",
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
      tickvals: [0, 50, 100, 150, 200, 300, 500, 700, 1000],
    },
    legend: {
      orientation: "h",
      x: 0.0,
      y: 1.10,
      font: { family: "IBM Plex Mono, monospace", size: 11, color: "#a1a1aa" },
    },
    margin: { l: 80, r: 40, t: 48, b: 64 },
    height: 580,
  };

  return (
    <div className="w-full space-y-4 font-mono text-xs">
      <div className="w-full h-[580px]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 580 }} />
      </div>
    </div>
  );
}
