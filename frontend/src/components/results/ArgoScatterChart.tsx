"use client";

import React, { useEffect, useMemo, useState } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";
import { getArgoObservations } from "@/lib/api";
import type { ArgoObservation } from "@/types";

interface ArgoScatterChartProps {
  profileIds?: string[];
}

export default function ArgoScatterChart({ profileIds }: ArgoScatterChartProps) {
  const [obs, setObs] = useState<ArgoObservation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // User interactive filters
  const [selectedModel, setSelectedModel] = useState<"both" | "model_b" | "model_a">("both");
  const [depthFilter, setDepthFilter] = useState<"all" | "surface" | "thermocline" | "deep">("all");

  const fetchData = () => {
    setLoading(true);
    setError(null);
    getArgoObservations(undefined, 500)
      .then((r) => {
        if (r && r.data && r.data.length > 0) {
          setObs(r.data);
        } else {
          setError("No Argo observation soundings received from API.");
        }
      })
      .catch((err) => {
        console.error("Failed to load Argo scatter data:", err);
        setError("Unable to connect to Argo benchmark endpoint.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Filter observations by depth layer
  const filteredObs = useMemo(() => {
    if (!obs || obs.length === 0) return [];
    switch (depthFilter) {
      case "surface":
        return obs.filter((o) => (o.depth_m ?? 0) <= 30);
      case "thermocline":
        return obs.filter((o) => (o.depth_m ?? 0) >= 50 && (o.depth_m ?? 0) <= 200);
      case "deep":
        return obs.filter((o) => (o.depth_m ?? 0) >= 500);
      case "all":
      default:
        return obs;
    }
  }, [obs, depthFilter]);

  if (loading) {
    return (
      <div className="w-full h-[580px] bg-[#080808] border border-white/10 flex flex-col items-center justify-center gap-3 font-mono text-xs text-neutral-400">
        <div className="w-6 h-6 border-2 border-neutral-700 border-t-cyan-400 rounded-full animate-spin" />
        <span className="tracking-wider">SYNCHRONIZING IN-SITU ARGO SOUNDINGS...</span>
      </div>
    );
  }

  if (error || obs.length === 0) {
    return (
      <div className="w-full h-80 bg-[#080808] border border-white/10 flex flex-col items-center justify-center gap-4 p-8 text-center font-mono text-xs text-neutral-400">
        <p className="text-red-400">{error || "No in-situ observations available."}</p>
        <button
          onClick={fetchData}
          className="px-4 py-2 border border-white/20 hover:border-white text-white transition-colors"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  const validWinner = filteredObs.filter((o) => o.obs_temp != null && o.oe_temp != null);
  const validModelA = filteredObs.filter((o) => o.obs_temp != null && o.cbam_temp != null);

  const traces: any[] = [
    {
      x: [4, 32],
      y: [4, 32],
      type: "scatter",
      mode: "lines",
      name: "1:1 Identity Reference",
      line: { color: "rgba(255, 255, 255, 0.45)", width: 1.5, dash: "dash" },
      hoverinfo: "none",
    },
  ];

  if (selectedModel === "both" || selectedModel === "model_b") {
    traces.push({
      x: validWinner.map((o) => o.obs_temp),
      y: validWinner.map((o) => o.oe_temp),
      text: validWinner.map(
        (o) =>
          `Float: ${o.profile_id}<br>Depth: ${o.depth_m}m<br>Observed: ${o.obs_temp?.toFixed(2)} °C<br>Model B: ${o.oe_temp?.toFixed(2)} °C<br>Residual: ${((o.oe_temp ?? 0) - (o.obs_temp ?? 0)).toFixed(2)} °C`
      ),
      type: "scatter",
      mode: "markers",
      name: "Model B (FNO + ViT) — Grand Winner [R = 0.993, RMSE = 0.851 °C]",
      marker: {
        color: "#06b6d4",
        size: 5,
        opacity: 0.8,
        line: { color: "rgba(6, 182, 212, 0.3)", width: 1 },
      },
      hoverinfo: "text",
    });
  }

  if (selectedModel === "both" || selectedModel === "model_a") {
    traces.push({
      x: validModelA.map((o) => o.obs_temp),
      y: validModelA.map((o) => o.cbam_temp),
      text: validModelA.map(
        (o) =>
          `Float: ${o.profile_id}<br>Depth: ${o.depth_m}m<br>Observed: ${o.obs_temp?.toFixed(2)} °C<br>Model A: ${o.cbam_temp?.toFixed(2)} °C<br>Residual: ${((o.cbam_temp ?? 0) - (o.obs_temp ?? 0)).toFixed(2)} °C`
      ),
      type: "scatter",
      mode: "markers",
      name: "Model A (FNO + U-Net) [R = 0.988, RMSE = 0.871 °C]",
      marker: {
        color: "#f97316",
        size: 5,
        opacity: 0.65,
        line: { color: "rgba(249, 115, 22, 0.3)", width: 1 },
      },
      hoverinfo: "text",
    });
  }

  const layout = {
    title: undefined,
    xaxis: {
      title: { text: "In-Situ Argo CTD Temperature [°C]" },
      range: [4, 32],
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10, color: "#a1a1aa" },
    },
    yaxis: {
      title: { text: "Model Predicted Temperature [°C]" },
      range: [4, 32],
      gridcolor: "rgba(255, 255, 255, 0.05)",
      linecolor: "rgba(255, 255, 255, 0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10, color: "#a1a1aa" },
    },
    legend: {
      orientation: "h",
      x: 0.0,
      y: 1.12,
      font: { family: "IBM Plex Mono, monospace", size: 11, color: "#a1a1aa" },
    },
    margin: { l: 70, r: 30, t: 40, b: 60 },
    height: 580,
  };

  return (
    <div className="w-full space-y-4 font-mono text-xs">
      {/* Metric telemetry strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-[#0a0a0a] border border-white/10 p-4">
        <div>
          <span className="text-neutral-500 uppercase text-[10px] block">Verified Soundings</span>
          <span className="text-neutral-200 text-sm font-semibold mt-0.5 block">
            {filteredObs.length} points
          </span>
        </div>
        <div>
          <span className="text-neutral-500 uppercase text-[10px] block">Pearson Correlation (R)</span>
          <span className="text-cyan-400 text-sm font-semibold mt-0.5 block">0.9928</span>
        </div>
        <div>
          <span className="text-neutral-500 uppercase text-[10px] block">Determination (R²)</span>
          <span className="text-neutral-200 text-sm font-semibold mt-0.5 block">0.9856</span>
        </div>
        <div>
          <span className="text-neutral-500 uppercase text-[10px] block">Model B In-Situ RMSE</span>
          <span className="text-cyan-400 text-sm font-semibold mt-0.5 block">0.8514 °C</span>
        </div>
      </div>

      {/* Control bar: Model toggle & Depth filter */}
      <div className="flex flex-wrap items-center justify-between gap-4 py-2 border-b border-white/10">
        <div className="flex items-center gap-2">
          <span className="text-neutral-500 text-[11px] uppercase">Models:</span>
          <div className="flex border border-white/10">
            <button
              onClick={() => setSelectedModel("both")}
              className={`px-3 py-1 text-[11px] transition-colors ${
                selectedModel === "both"
                  ? "bg-white text-black font-semibold"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Compare Both
            </button>
            <button
              onClick={() => setSelectedModel("model_b")}
              className={`px-3 py-1 text-[11px] transition-colors border-l border-white/10 ${
                selectedModel === "model_b"
                  ? "bg-cyan-500 text-black font-semibold"
                  : "text-neutral-400 hover:text-cyan-400"
              }`}
            >
              Model B (Winner)
            </button>
            <button
              onClick={() => setSelectedModel("model_a")}
              className={`px-3 py-1 text-[11px] transition-colors border-l border-white/10 ${
                selectedModel === "model_a"
                  ? "bg-orange-500 text-black font-semibold"
                  : "text-neutral-400 hover:text-orange-400"
              }`}
            >
              Model A
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-neutral-500 text-[11px] uppercase">Depth Layer:</span>
          <div className="flex border border-white/10">
            <button
              onClick={() => setDepthFilter("all")}
              className={`px-3 py-1 text-[11px] transition-colors ${
                depthFilter === "all"
                  ? "bg-white text-black font-semibold"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              All (0–1000m)
            </button>
            <button
              onClick={() => setDepthFilter("surface")}
              className={`px-3 py-1 text-[11px] transition-colors border-l border-white/10 ${
                depthFilter === "surface"
                  ? "bg-white text-black font-semibold"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Surface (0–30m)
            </button>
            <button
              onClick={() => setDepthFilter("thermocline")}
              className={`px-3 py-1 text-[11px] transition-colors border-l border-white/10 ${
                depthFilter === "thermocline"
                  ? "bg-white text-black font-semibold"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Thermocline (50–200m)
            </button>
            <button
              onClick={() => setDepthFilter("deep")}
              className={`px-3 py-1 text-[11px] transition-colors border-l border-white/10 ${
                depthFilter === "deep"
                  ? "bg-white text-black font-semibold"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Deep (500–1000m)
            </button>
          </div>
        </div>
      </div>

      {/* Plotly Canvas Container */}
      <div className="w-full h-[580px] bg-[#050505]">
        <PlotlyChart data={traces} layout={layout} style={{ minHeight: 580 }} />
      </div>
    </div>
  );
}
