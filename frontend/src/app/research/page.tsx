import type { Metadata } from "next";
import Link from "next/link";
import PublicationFiguresGallery from "@/components/research/PublicationFiguresGallery";

export const metadata: Metadata = {
  title: "Research Methodology",
  description:
    "Scientific methodology, satellite data provenance, and in-situ validation protocols behind the OceanEmbed subsurface ocean temperature reconstruction project.",
};

export default function ResearchPage() {
  return (
    <div className="w-full min-h-screen bg-[#050505] text-neutral-100 antialiased font-sans pb-36">
      {/* ── Editorial Header ────────────────────────────────────────── */}
      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 py-24 rule-b">
        <div className="max-w-4xl space-y-4">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
            <span>SCIENTIFIC REPORT & EXPERIMENTAL PROTOCOL</span>
          </div>
          <h1 className="text-4xl sm:text-6xl font-light text-white tracking-tight">
            RESEARCH & PROTOCOL
          </h1>
          <p className="text-lg sm:text-xl text-neutral-300 font-sans leading-relaxed pt-2 max-w-3xl">
            Theoretical foundations, multi-sensor observational provenance, and rigorous 7-year
            empirical validation protocols for deep operator learning in computational oceanography.
          </p>

          <div className="pt-8 border-t border-white/[0.08] flex flex-wrap items-center gap-12 font-mono text-xs text-neutral-400">
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Target Domain</span>
              <span className="text-neutral-200 mt-0.5 block">North Indian Ocean (5°–30°N, 45°–105°E)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Spatial Grid</span>
              <span className="text-neutral-200 mt-0.5 block">101 × 241 at 0.25° (24,341 points/lvl)</span>
            </div>
            <div>
              <span className="text-neutral-500 uppercase text-[10px] block">Validation Corpus</span>
              <span className="text-cyan-400 font-semibold mt-0.5 block">37,708 In-Situ Argo Floats (2015–2021)</span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-full px-6 md:px-16 lg:px-24 xl:px-32 mt-20 space-y-24">
        {/* Section 1: Problem Formulation */}
        <section className="space-y-4 max-w-4xl">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            01 / PROBLEM FORMULATION
          </div>
          <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
            The Inverse Surface-to-Subsurface Operator Mapping
          </h2>
          <div className="space-y-4 text-neutral-300 text-base leading-relaxed font-sans pt-2">
            <p>
              Spaceborne remote sensing satellites observe the continuous two-dimensional surface
              boundary skin of the global ocean. However, critical oceanic phenomena—such as tropical
              cyclone intensification, heat uptake, internal wave generation, and baroclinic transport—are
              governed by the three-dimensional vertical structure of the water column.
            </p>
            <p>
              Autonomous profiling floats (e.g. Argo) provide high-precision in-situ vertical soundings
              but remain fundamentally sparse in space and time (typically one sounding every 10 days
              over ~300 km). OceanEmbed addresses this fundamental observation gap by learning a parameterised
              continuous operator mapping 2D multi-satellite surface observables to the full 3D temperature
              column (0 to 1000 m).
            </p>
          </div>
        </section>

        {/* Section 2: Study Domain */}
        <section className="space-y-4 max-w-4xl">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            02 / STUDY DOMAIN & DYNAMICS
          </div>
          <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
            The North Indian Ocean Basin
          </h2>
          <div className="space-y-4 text-neutral-300 text-base leading-relaxed font-sans pt-2">
            <p>
              The target geographic domain is bounded by <strong>5.0°N–30.0°N</strong> and{" "}
              <strong>45.0°E–105.0°E</strong> at a uniform <strong>0.25° grid resolution</strong>,
              yielding an active ocean matrix of 101 × 241 cells across 15 standard depth horizons:
              0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, and 1000 meters.
            </p>
            <p>
              The North Indian Ocean comprises the Arabian Sea, the Bay of Bengal, and the equatorial
              boundary corridor. It is governed by semi-annual monsoon reversals, intense coastal
              upwelling along Somalia and Oman, and massive freshwater river discharge into the northern
              Bay of Bengal that produces an acute salinity barrier layer. These coupled baroclinic
              processes decouple the surface mixed layer from the thermocline, necessitating deep
              operator learning capable of handling non-local Green&apos;s function kernels.
            </p>
          </div>
        </section>

        {/* Section 3: Observational Provenance */}
        <section className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              03 / OBSERVATIONAL PROVENANCE
            </div>
            <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
              Multi-Satellite Surface Channels (7 Observables)
            </h2>
          </div>

          <div className="w-full overflow-x-auto">
            <table className="editorial-table font-mono text-xs">
              <thead>
                <tr>
                  <th>Channel</th>
                  <th>Observable</th>
                  <th>Unit</th>
                  <th>Dataset Identifier</th>
                  <th>Satellite Sensor / Provenance</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td className="text-white font-medium">SST</td>
                  <td className="text-neutral-200 font-sans">Sea Surface Temperature</td>
                  <td>°C</td>
                  <td className="text-neutral-400">METOFFICE-GLO-SST-L4-REP-OBS-SST</td>
                  <td className="text-neutral-400 font-sans">Infrared + Microwave radiometry (OSTIA L4)</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">SSS</td>
                  <td className="text-neutral-200 font-sans">Sea Surface Salinity</td>
                  <td>psu</td>
                  <td className="text-neutral-400">cmems_obs-mob_glo_phy-sss_my_multi_P1D</td>
                  <td className="text-neutral-400 font-sans">SMOS / SMAP L-band microwave</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">SLA</td>
                  <td className="text-neutral-200 font-sans">Sea Level Anomaly</td>
                  <td>m</td>
                  <td className="text-neutral-400">cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs</td>
                  <td className="text-neutral-400 font-sans">Merged multi-mission radar altimetry</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Current U</td>
                  <td className="text-neutral-200 font-sans">Zonal Geostrophic Current</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">DUACS Geostrophic Component</td>
                  <td className="text-neutral-400 font-sans">Derived via geostrophic balance from SLA</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Current V</td>
                  <td className="text-neutral-200 font-sans">Meridional Geostrophic Current</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">DUACS Geostrophic Component</td>
                  <td className="text-neutral-400 font-sans">Derived via geostrophic balance from SLA</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Wind U</td>
                  <td className="text-neutral-200 font-sans">Zonal 10m Wind Stress</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">ERA5 / ASCAT Scatterometer</td>
                  <td className="text-neutral-400 font-sans">MetOp ASCAT scatterometer constellation</td>
                </tr>
                <tr>
                  <td className="text-white font-medium">Wind V</td>
                  <td className="text-neutral-200 font-sans">Meridional 10m Wind Stress</td>
                  <td>m/s</td>
                  <td className="text-neutral-400">ERA5 / ASCAT Scatterometer</td>
                  <td className="text-neutral-400 font-sans">MetOp ASCAT scatterometer constellation</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Section 4: 7-Year Temporal Protocol */}
        <section className="space-y-4 max-w-4xl">
          <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
            04 / 7-YEAR TEMPORAL PROTOCOL
          </div>
          <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
            Strict Temporal Segregation & In-Situ Benchmark
          </h2>
          <div className="space-y-4 text-neutral-300 text-base leading-relaxed font-sans pt-2">
            <p>
              To completely prevent temporal auto-correlation leakage, OceanEmbed adopts a strict
              multi-year chronological partition across seven complete calendar years (2015–2021):
            </p>
            <ol className="list-decimal list-inside space-y-3 text-neutral-300">
              <li>
                <strong>Training Split (2015-01-01 to 2019-12-31):</strong> 1,826 consecutive days
                used exclusively for model parameter optimization.
              </li>
              <li>
                <strong>Validation Split (2020-01-01 to 2020-12-31):</strong> 366 consecutive days (leap year)
                used for hyperparameter selection and early stopping checkpointing. Validated against 4,255
                in-situ Argo profiles.
              </li>
              <li>
                <strong>Sealed Test Split (2021-01-01 to 2021-12-31):</strong> 365 days held strictly
                unseen until final evaluation. Validated across 5,302 independent Argo profiles
                yielding 55,136 depth-matched observations.
              </li>
            </ol>
            <p className="pt-2">
              The grand winner, <strong>Model B (Dual-Branch FNO + ViT)</strong>, achieves an overall
              test RMSE of <strong>0.8008 °C</strong> on GLORYS reanalysis and <strong>0.8514 °C</strong> on
              in-situ Argo floats, establishing the state of the art in satellite subsurface ocean reconstruction.
            </p>
          </div>
        </section>

        {/* Section 5: Scientific Analysis & Empirical Figures */}
        <section id="publication-figures" className="space-y-6">
          <div className="space-y-2">
            <div className="font-mono text-xs text-neutral-400 uppercase tracking-widest">
              05 / HIGH-RESOLUTION EMPIRICAL FIGURES
            </div>
            <h2 className="text-2xl sm:text-3xl font-light text-white tracking-tight">
              Scientific Analysis & Empirical Figures
            </h2>
            <p className="text-neutral-400 text-sm font-sans max-w-3xl leading-relaxed">
              Explore the publication-grade figures generated from the final 2021 sealed test
              and 37,708 in-situ Argo float comparative benchmark.
            </p>
          </div>

          <PublicationFiguresGallery />
        </section>

        {/* CTA */}
        <div className="pt-12 rule-t flex flex-wrap items-center justify-between gap-6">
          <div>
            <div className="text-xl font-light text-white">Experience the interactive reconstruction</div>
            <div className="text-sm text-neutral-400 font-sans mt-1">
              Run real-time neural inference on any observation date across the basin.
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
