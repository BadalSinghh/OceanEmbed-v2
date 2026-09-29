# OceanEmbed — Web Application & Scientific Platform

[![Next.js 16](https://img.shields.io/badge/Next.js-16.3-black.svg)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178c6.svg)](https://www.typescriptlang.org/)

**A satellite-embedding deep learning web platform for reconstructing the 3D vertical ocean temperature structure from multi-satellite surface observations over the North Indian Ocean ($5^\circ\text{–}30^\circ\text{N}, 45^\circ\text{–}105^\circ\text{E}$).**

---

## 1. Web Application Features

The OceanEmbed web application provides an interactive research environment:

* **Interactive 3D Ocean Volume**: Interactive Three.js/WebGL volume reconstruction across 15 standard ocean depths ($0\text{ to }1000\text{ meters}$).
* **Dynamic 2D Depth Slice Explorer**: Basin-wide horizontal thermal field slices across user-selected depth horizons with responsive thermal color scaling.
* **Vertical Transect & Sounding Lab**: Real-time cross-sectional slices and vertical CTD profile extractions at any coordinate across the North Indian Ocean.
* **Argo Float Ground-Truth Validation**: Physical validation against 37,708 in-situ autonomous profiling Argo CTD floats with interactive depth layer filtering (Surface, Thermocline, Deep) and model comparisons.
* **Dual-Branch Architecture Explorer**: Interactive layer-by-layer breakdown of the FNO and ViT/U-Net dual-branch neural operator pipelines.
* **Scientific Research Protocol**: Comprehensive observational provenance, 7-channel satellite documentation, and high-resolution publication figures gallery.

---

## 2. Architecture & Tech Stack

* **Frontend**: Next.js 16 (App Router), React 19, TypeScript, Vanilla CSS design system, Plotly.js, Three.js (`@react-three/fiber`, `@react-three/drei`).
* **Backend**: FastAPI, Uvicorn, PyTorch (CPU-optimized for cloud deployment), NumPy, SciPy, NetCDF4, Pandas.
* **Pre-trained Models**:
  * **Model B (Dual-Branch FNO + ViT)**: Grand Winner ($0.8008^\circ\text{C}$ reanalysis RMSE, $0.8514^\circ\text{C}$ in-situ Argo RMSE).
  * **Model A (Dual-Branch FNO + U-Net)**: ($0.8421^\circ\text{C}$ reanalysis RMSE, $0.8711^\circ\text{C}$ in-situ Argo RMSE).
  * **Baselines**: Vision Transformer (ViT), ResU-Net, Fourier Neural Operator (FNO), and Monthly Climatology.

---

## 3. Running Locally

### Prerequisites
* Node.js 18+ & npm
* Python 3.10+

### Step 1: Start the FastAPI Backend
```bash
# From the project root
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

# Run the backend (port 8000)
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Backend API will be running at `http://localhost:8000` with interactive Swagger docs at `http://localhost:8000/api/docs`.

### Step 2: Start the Next.js Frontend
```bash
# In a new terminal window
cd frontend
npm install

# Run the development server (port 3000)
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 4. Production Deployment

### Frontend (Vercel)
1. Import the repository into **[Vercel](https://vercel.com/)**.
2. Set the **Root Directory** to `frontend`.
3. In Environment Variables, set:
   * `NEXT_PUBLIC_API_URL`: Your deployed FastAPI backend URL (e.g. `https://your-backend.onrender.com`).
4. Click **Deploy**.

### Backend (Render / Railway / Docker)
* **Render**: Use the included [`render.yaml`](./render.yaml) or [`Procfile`](./Procfile). Set build command to `pip install -r backend/requirements.txt` and start command to `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`.
* **Docker**: Build and run using the included [`Dockerfile`](./Dockerfile):
  ```bash
  docker build -t oceanembed-backend .
  docker run -p 8000:8000 oceanembed-backend
  ```

---

## 5. Repository Structure

```
c:\oceanEmbed\
├── backend/                  # FastAPI production backend
│   ├── app/
│   │   ├── api/              # API routes: health, models, predict, results
│   │   ├── loaders.py        # Model checkpoints and test sequence loaders
│   │   └── main.py           # Application entrypoint & CORS configuration
│   └── requirements.txt      # Production Python dependencies
├── checkpoints/              # Best pre-trained PyTorch model weights (.pth)
├── frontend/                 # Next.js 16 Web Application
│   ├── public/figures/       # High-resolution publication benchmark figures
│   └── src/
│       ├── app/              # Routes: /, /demo, /results, /architecture, /research
│       ├── components/       # UI & scientific visualization components
│       ├── lib/api.ts        # Typed API client
│       └── types/            # Shared TypeScript type definitions
├── OceanEmbed_Data/          # Test sequence tensors, Argo ground-truth, & models
├── outputs/reports/          # Quantitative benchmark evaluation JSON reports
├── ocean_mask.nc             # Canonical North Indian Ocean land/depth mask
├── test_sequences.csv        # 2021 daily test sequence index
├── experiments.csv           # Leaderboard benchmark matrix
├── Dockerfile                # Production container deployment specification
├── Procfile                  # Process command for cloud deployment
├── render.yaml               # 1-click Render deployment blueprint
└── README.md                 # Project documentation
```
