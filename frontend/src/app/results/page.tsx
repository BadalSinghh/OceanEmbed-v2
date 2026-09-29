"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import {
  getEvaluation,
  getPerDepthMetrics,
  getTrainingHistory,
  getArgoAggregate,
  getArgoProfiles,
  getArgoObservations,
} from "@/lib/api";
import type {
  EvaluationSummary,
  PerDepthRow,
  TrainingHistory,
  ArgoAggregate,
  ArgoProfile,
  ArgoObservation,
} from "@/types";

const DepthMetricsChart = dynamic(
  () => import("@/components/results/DepthMetricsChart"),
  { ssr: false }
);
const TrainingHistoryChart = dynamic(
  () => import("@/components/results/TrainingHistoryChart"),
  { ssr: false }
);
const ArgoScatterChart = dynamic(
  () => import("@/components/results/ArgoScatterChart"),
  { ssr: false }
);
const ArgoProfileChart = dynamic(
  () => import("@/components/results/ArgoProfileChart"),
  { ssr: false }
);

export default function ResultsPage() {
  const [evalData, setEvalData] = useState<EvaluationSummary | null>(null);
  const [perDepth, setPerDepth] = useState<PerDepthRow[]>([]);
  const [histories, setHistories] = useState<Record<string, TrainingHistory>>({});
  const [argoAgg, setArgoAgg] = useState<ArgoAggregate | null>(null);
  const [argoProfiles, setArgoProfiles] = useState<ArgoProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<string>("");
  const [argoObs, setArgoObs] = useState<ArgoObservation[]>([]);
  const [loadingObs, setLoadingObs] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getEvaluation().catch(() => null),
      getPerDepthMetrics().catch(() => []),
      getTrainingHistory().catch(() => ({})),
      getArgoAggregate().catch(() => null),
      getArgoProfiles().catch(() => ({ profiles: [], count: 0 })),
    ])
      .then(([ev, pd, hist, agg, profilesRes]) => {
        if (ev) setEvalData(ev);
        if (pd && pd.length > 0) setPerDepth(pd);
        if (hist) setHistories(hist);
        if (agg) setArgoAgg(agg);
        if (profilesRes && profilesRes.profiles.length > 0) {
          setArgoProfiles(profilesRes.profiles);
          setSelectedProfile(profilesRes.profiles[0].profile_id);
        }
      })
      .catch((err) => {
        console.warn("API loading issue:", err);
        setErrorMsg("Failed to synchronize with backend results API.");
      });
  }, []);

  useEffect(() => {
    if (!selectedProfile) return;
    setLoadingObs(true);
    getArgoObservations(selectedProfile, 200)
      .then((r) => setArgoObs(r.data))
      .catch((err) => console.error("Failed to load profile obs:", err))
      .finally(() => setLoadingObs(false));
  }, [selectedProfile]);

  return (
    <div className="w-full min-h-screen bg-[#050505] text-neutral-100 antialiased font-sans pb-36">
      {/* ── Editorial Header ────────────────────────────────────────── */}
      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 py-24 rule-b">
        <div className="max-w-4xl space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>EMPIRICAL BENCHMARK & EXPERIMENTAL RESULTS</span>
          </div>
          <h1 className="text-5xl sm:text-7xl font-light text-white tracking-tight">
            RESULTS
          </h1>
          <p className="text-lg sm:text-xl text-neutral-300 font-sans leading-relaxed pt-2 max-w-3xl">
            Rigorous quantitative evaluation of the dual-branch neural operator architectures
            against baseline models across the 2021 sealed test split and 37,708 independent in-situ
            Argo profiling CTD floats.
          </p>

          <div className="pt-8 border-t border-white/[0.08] flex flex-wrap items-center gap-12 font-mono text-xs text-neutral-400">
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Train Split</span>
              <span className="text-neutral-200 mt-0.5 block">2015–2019 (1,826 Days)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Validation Split</span>
              <span className="text-neutral-200 mt-0.5 block">2020 (366 Days, 4,255 Floats)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Sealed Test Split</span>
              <span className="text-cyan-400 font-semibold mt-0.5 block">2021 (365 Days, 55,136 Soundings)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Total In-Situ Corpus</span>
              <span className="text-neutral-200 mt-0.5 block">37,708 Profiles (CORA / INCOIS)</span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 mt-20 space-y-32">
        {errorMsg && (
          <div className="p-4 bg-red-950/40 text-red-300 font-mono text-xs rule-b">
            {errorMsg}
          </div>
        )}

        {/* ── SECTION 1: BENCHMARK LEADERBOARD ───────────────────────── */}
        <section id="overall-performance" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              01 / COMPARATIVE BENCHMARK MATRIX
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Model Performance on Sealed 2021 Test Set
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              Models trained on identical 2015–2019 sequences and evaluated strictly on out-of-sample
              2021 satellite inputs against both reanalysis fields and in-situ Argo profiling floats.
            </p>
          </div>

          <div className="w-full overflow-x-auto">
            <table className="editorial-table font-mono text-xs">
              <thead>
                <tr>
                  <th>Architecture</th>
                  <th>Parameters</th>
                  <th>2021 Test RMSE</th>
                  <th>2021 Test MAE</th>
                  <th>Argo In-Situ RMSE</th>
                  <th>Argo In-Situ MAE</th>
                  <th>Pearson r</th>
                  <th>R²</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-l-2 border-cyan-400 bg-white/[0.04]">
                  <td className="text-white font-semibold font-sans">
                    Model B (Dual-Branch FNO + ViT) <span className="text-cyan-400 text-[10px]">★ GRAND WINNER</span>
                  </td>
                  <td className="text-neutral-300">6,432,607</td>
                  <td className="text-white font-bold">0.8008 °C</td>
                  <td className="text-white font-bold">0.5382 °C</td>
                  <td className="text-cyan-400 font-bold">0.8514 °C</td>
                  <td className="text-cyan-400 font-bold">0.5716 °C</td>
                  <td className="text-white font-bold">0.9928</td>
                  <td className="text-white font-bold">0.9856</td>
                </tr>

                <tr>
                  <td className="text-neutral-200 font-semibold font-sans">
                    Model A (Dual-Branch FNO + U-Net)
                  </td>
                  <td className="text-neutral-400">1,842,575</td>
                  <td>0.8421 °C</td>
                  <td>0.5664 °C</td>
                  <td className="text-neutral-200 font-semibold">0.8711 °C</td>
                  <td>0.5888 °C</td>
                  <td>0.9882</td>
                  <td>0.9764</td>
                </tr>

                <tr>
                  <td className="text-neutral-300 font-sans">Vision Transformer (ViT) Baseline</td>
                  <td className="text-neutral-400">4,723,215</td>
                  <td>0.8174 °C</td>
                  <td>0.5489 °C</td>
                  <td>0.8784 °C</td>
                  <td>0.5941 °C</td>
                  <td>0.9901</td>
                  <td>0.9802</td>
                </tr>

                <tr>
                  <td className="text-neutral-300 font-sans">U-Net Baseline</td>
                  <td className="text-neutral-400">1,940,303</td>
                  <td>0.8041 °C</td>
                  <td>0.5460 °C</td>
                  <td>0.8826 °C</td>
                  <td>0.5963 °C</td>
                  <td>0.9892</td>
                  <td>0.9798</td>
                </tr>

                <tr>
                  <td className="text-neutral-400 font-sans">Fourier Neural Operator (FNO) Baseline</td>
                  <td className="text-neutral-400">2,367,055</td>
                  <td>0.8570 °C</td>
                  <td>0.5857 °C</td>
                  <td>0.8747 °C</td>
                  <td>0.5902 °C</td>
                  <td>0.9854</td>
                  <td>0.9741</td>
                </tr>

                <tr className="text-neutral-500">
                  <td className="text-neutral-500 font-sans">Monthly Climatology Baseline</td>
                  <td className="text-neutral-500">0</td>
                  <td>1.0827 °C</td>
                  <td>0.7420 °C</td>
                  <td>1.2050 °C</td>
                  <td>0.6950 °C</td>
                  <td>0.9766</td>
                  <td>0.9394</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="pt-2 text-neutral-400 text-sm font-sans leading-relaxed">
            <p>
              <strong>Key Finding:</strong> Model B (FNO + ViT) achieves the lowest test error on
              both reanalysis (<strong>0.8008 °C</strong>) and independent in-situ Argo profiling floats
              (<strong>0.8514 °C</strong> across 55,136 depth soundings), outperforming climatology by 29.3%
              and surpassing all single-branch baselines in resolving the thermocline.
            </p>
          </div>
        </section>

        {/* ── SECTION 2: DEPTH-DEPENDENT ERROR ────────────────────────── */}
        <section id="depth-dependent-error" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              02 / VERTICAL WATER COLUMN RESOLUTION
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Depth-Dependent Reconstruction Error
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              Reconstruction error across 15 depth horizons from 0 to 1000 meters. Error peaks
              in the steep thermocline (75–125 m) where baroclinic heaving is strongest, and decays
              monotonically down to ±0.42 °C in the deep mesopelagic layer.
            </p>
          </div>

          <DepthMetricsChart data={perDepth} />
        </section>

        {/* ── SECTION 3: ARGO IN-SITU VALIDATION ──────────────────────── */}
        <section id="argo-validation" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              03 / AUTONOMOUS PROFILING FLOATS
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Argo Float Ground-Truth Validation
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              Scatter distribution of predicted temperature against autonomous profiling Argo floats
              from the CORA / INCOIS delayed-mode QC-passed archive. No Argo float data was used during training.
            </p>
          </div>

          <ArgoScatterChart profileIds={argoProfiles.map((p) => p.profile_id)} />
        </section>

        {/* ── SECTION 4: REPRESENTATIVE PROFILES ──────────────────────── */}
        <section id="representative-profiles" className="space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div className="space-y-2">
              <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
                04 / FLOAT PROFILE SOUNDING
              </div>
              <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
                Representative Vertical Soundings
              </h2>
            </div>

            <div className="flex items-center gap-3 font-mono text-xs">
              <span className="text-neutral-500">Select Float:</span>
              <select
                value={selectedProfile}
                onChange={(e) => setSelectedProfile(e.target.value)}
                className="bg-[#0a0a0a] border border-white/10 rounded-none px-3 py-1.5 text-neutral-200 text-xs font-mono outline-none focus:border-white"
              >
                {argoProfiles.map((p) => (
                  <option key={p.profile_id} value={p.profile_id}>
                    {p.profile_id}, {p.date?.slice(0, 10)} ({p.lat?.toFixed(1)}°N, {p.lon?.toFixed(1)}°E)
                  </option>
                ))}
              </select>
            </div>
          </div>

          {loadingObs ? (
            <div className="w-full h-80 flex items-center justify-center font-mono text-xs text-neutral-500">
              <div className="w-5 h-5 border-2 border-neutral-700 border-t-white rounded-full animate-spin" />
            </div>
          ) : argoObs.length > 0 ? (
            <ArgoProfileChart observations={argoObs} />
          ) : (
            <div className="p-12 text-center text-neutral-500 font-mono text-xs">
              No observation records found for the selected float.
            </div>
          )}
        </section>

        {/* ── SECTION 5: TRAINING DYNAMICS ────────────────────────────── */}
        <section id="training-dynamics" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              05 / OPTIMIZATION & CONVERGENCE
            </div>
            <h2 className="text-3xl sm:text-4xl font-light text-white tracking-tight">
              Training & Validation Loss Dynamics
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              Convergence curves evaluated per epoch using masked mean squared error over active ocean grid cells.
            </p>
          </div>

          <TrainingHistoryChart histories={histories} />
        </section>
      </div>
    </div>
  );
}
