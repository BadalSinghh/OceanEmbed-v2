"use client";

import React, { useState } from "react";
import PlotlyChart from "@/components/common/PlotlyChart";

interface VerticalTransectProps {
  prediction?: (number | null)[][][]; // [15, 101, 241]
  fieldData?: (number | null)[][][];   // [15, 101, 241]
  fieldMode?: "prediction" | "groundTruth" | "error";
  depths: number[];                  // 15 depths
  lats: number[];                    // 101 lats
  lons: number[];                    // 241 lons
  model_name: string;
}

export default function VerticalTransectPlotly({
  prediction,
  fieldData,
  fieldMode = "prediction",
  depths,
  lats,
  lons,
  model_name,
}: VerticalTransectProps) {
  const [transectType, setTransectType] = useState<"meridional" | "zonal">("meridional");

  // Default to central Bay of Bengal (~88°E, ~14°N)
  const defaultLonIdx = Math.max(0, lons.findIndex((l) => l >= 88.0));
  const defaultLatIdx = Math.max(0, lats.findIndex((l) => l >= 14.0));

  const [lonIndex, setLonIndex] = useState<number>(defaultLonIdx >= 0 ? defaultLonIdx : 172);
  const [latIndex, setLatIndex] = useState<number>(defaultLatIdx >= 0 ? defaultLatIdx : 36);

  const activeTensor = fieldData || prediction || [];
  const isError = fieldMode === "error";

  const zMatrix: (number | null)[][] = [];
  const xValues = transectType === "meridional" ? lats : lons;

  for (let d = 0; d < depths.length; d++) {
    const row: (number | null)[] = [];
    if (transectType === "meridional") {
      for (let i = 0; i < lats.length; i++) {
        row.push(activeTensor[d]?.[i]?.[lonIndex] ?? null);
      }
    } else {
      for (let j = 0; j < lons.length; j++) {
        row.push(activeTensor[d]?.[latIndex]?.[j] ?? null);
      }
    }
    zMatrix.push(row);
  }

  // Thermal vs Divergent Residual Colormaps
  const tempColorscale = [
    [0, "#030712"],
    [0.12, "#1e3a8a"],
    [0.28, "#0284c7"],
    [0.48, "#06b6d4"],
    [0.65, "#059669"],
    [0.82, "#d97706"],
    [1, "#dc2626"],
  ];

  const residualColorscale = [
    [0, "#2563eb"],     // -2.5 °C (Underestimation - Blue)
    [0.32, "#93c5fd"],  // -0.9 °C
    [0.5, "#18181b"],   //  0.0 °C (Zero Error - Neutral Dark)
    [0.68, "#fca5a5"],  // +0.9 °C
    [1.0, "#dc2626"],   // +2.5 °C (Overestimation - Red)
  ];

  const trace = {
    z: zMatrix,
    x: xValues,
    y: depths,
    type: "contour",
    colorscale: isError ? residualColorscale : tempColorscale,
    zmin: isError ? -2.5 : 4,
    zmax: isError ? 2.5 : 30,
    contours: {
      coloring: "heatmap",
      showlines: !isError,
    },
    line: { smoothing: 1.3, width: 0.7, color: "rgba(255,255,255,0.25)" },
    colorbar: {
      title: {
        text: isError ? "ΔT [°C]" : "Temp [°C]",
        font: { family: "IBM Plex Mono, monospace", size: 11, color: "#d4d4d8" },
      },
      tickfont: { family: "IBM Plex Mono, monospace", size: 10, color: "#a1a1aa" },
      thickness: 14,
      len: 0.9,
    },
    hoverongaps: false,
    hovertemplate: isError
      ? "%{x:.2f}° / Depth: %{y}m<br>Residual Error: %{z:+.2f} °C<extra></extra>"
      : "%{x:.2f}° / Depth: %{y}m<br>Temp: %{z:.2f} °C<extra></extra>",
  };

  const layout = {
    title: undefined,
    xaxis: {
      title: { text: transectType === "meridional" ? "Latitude [°N]" : "Longitude [°E]" },
      gridcolor: "rgba(255,255,255,0.05)",
      linecolor: "rgba(255,255,255,0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
    },
    yaxis: {
      title: { text: "Depth [m]" },
      autorange: "reversed",
      gridcolor: "rgba(255,255,255,0.05)",
      linecolor: "rgba(255,255,255,0.12)",
      tickfont: { family: "IBM Plex Mono, monospace", size: 10 },
      tickvals: [0, 30, 75, 150, 300, 500, 700, 1000],
    },
    margin: { l: 80, r: 40, t: 20, b: 64 },
    height: 600,
  };

  return (
    <div className="w-full bg-[#050505] font-mono text-xs space-y-4">
      {/* Transect Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-3 rule-b">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 bg-white/[0.04] p-0.5 border border-white/10">
            <button
              onClick={() => setTransectType("meridional")}
              className={`px-3 py-1 text-xs transition-colors cursor-pointer ${
                transectType === "meridional"
                  ? "bg-white/15 text-white font-medium"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Meridional Transect (N–S)
            </button>
            <button
              onClick={() => setTransectType("zonal")}
              className={`px-3 py-1 text-xs transition-colors cursor-pointer ${
                transectType === "zonal"
                  ? "bg-white/15 text-white font-medium"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Zonal Transect (W–E)
            </button>
          </div>

          <span className={`px-2 py-0.5 text-[10px] uppercase font-mono tracking-wider ${
            isError
              ? "bg-rose-500/15 text-rose-300 border border-rose-500/30"
              : fieldMode === "groundTruth"
              ? "bg-amber-500/15 text-amber-300 border border-amber-500/30"
              : "bg-cyan-500/15 text-cyan-300 border border-cyan-500/30"
          }`}>
            {isError ? "Residual Error Field" : fieldMode === "groundTruth" ? "GLORYS12 Ground Truth" : "Prediction Field"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          {transectType === "meridional" ? (
            <>
              <span className="text-neutral-500 text-[11px]">Longitude Slice:</span>
              <input
                type="range"
                min={0}
                max={lons.length - 1}
                value={lonIndex}
                onChange={(e) => setLonIndex(Number(e.target.value))}
                className="w-36 accent-white"
              />
              <span className="text-neutral-200 text-xs w-16 text-right font-medium">
                {lons[lonIndex]?.toFixed(1)}°E
              </span>
            </>
          ) : (
            <>
              <span className="text-neutral-500 text-[11px]">Latitude Slice:</span>
              <input
                type="range"
                min={0}
                max={lats.length - 1}
                value={latIndex}
                onChange={(e) => setLatIndex(Number(e.target.value))}
                className="w-36 accent-white"
              />
              <span className="text-neutral-200 text-xs w-16 text-right font-medium">
                {lats[latIndex]?.toFixed(1)}°N
              </span>
            </>
          )}
        </div>
      </div>

      <div className="w-full h-[600px]">
        <PlotlyChart data={[trace]} layout={layout} style={{ minHeight: 600 }} />
      </div>
    </div>
  );
}
