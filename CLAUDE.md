# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project combines a project-owned incremental SfM pipeline (`src/sfm/`) with thin adapters for VGGT and 3D Gaussian Splatting backends. The goal is an auditable, paper-aligned camera-pose-and-sparse-point-cloud pipeline that exports to 3DGS for comparison training.

**Core principle:** All project-owned code lives in `src/`. Third-party code stays in vendored directories, accessed through thin adapters in `src/adapters/`. The data handoff boundary is COLMAP-format `images/ + sparse/0/` directories.

**Vendored copies:** Each third-party backend has two copies — a clean upstream reference under `third_party/` (never modified) and a working copy at the repo root (`vggt/`, `gaussian-splatting/`) with built CUDA extensions and experiment outputs. The VGGT adapter points at `third_party/vggt/`; experiment tools may additionally reference the top-level workspace.

## Environment Setup

The project uses **separate conda environments** for different components:

```bash
# Main SfM environment (CPU-friendly)
conda create -n mm26 python=3.10 -y
conda activate mm26
pip install -r requirements.txt

# 3DGS environment (requires CUDA)
conda env create -f third_party/gaussian-splatting/environment.yml
conda activate gaussian_splatting

# VGGT environment (requires CUDA)
conda create -n vggt python=3.10 -y
conda activate vggt
pip install -r third_party/vggt/requirements.txt
pip install -e third_party/vggt
```

Key dependencies for the main SfM environment: `numpy`, `scipy`, `opencv-contrib-python`, `pycolmap`, `open3d`, `networkx`, `PyYAML`.

## Architecture

### Core SfM Library (`src/sfm/`)

The geometry pipeline is built from pure Python with NumPy/SciPy/CV2 — no CUDA, no deep-learning dependencies. Key modules:

- **`camera.py`** — `PinholeCamera` dataclass (frozen), intrinsics estimation (EXIF fallback → focal_scale heuristic), projection/depth/reprojection utilities. Cameras are persistently stored per-image in `camera_intrinsics.json` and can override scene-wide defaults.
- **`features.py`** — Image listing, grayscale loading, resizing for SfM, RootSIFT extraction/loading.
- **`matching.py`** — `ImageFeatures` and `PairMatchResult` dataclasses, exhaustive/sequential/retrieval match pair generation, ratio test matching.
- **`geometry.py`** — Fundamental/Essential/Homography estimation, cheirality/degeneracy checks, verified pair scoring.
- **`initialization.py`** — `InitializationResult` dataclass, two-view initialization with DLT triangulation, pair scoring by inlier count and triangulation quality.
- **`reconstruction.py`** — Core data structures: `RegisteredImage`, `ReconstructionState`, `Candidate2D3D`, `NextBestViewScore`, `RegistrationResult`. Functions for loading/saving state, PnP registration with strict/soft accept, next-best-view scoring, retry logic.
- **`triangulation.py`** — DLT triangulation, cheirality and angle gating, post-BA triangulation with residual filtering, track merge/split.
- **`bundle_adjustment.py`** — SciPy-based local/global BA with: Cauchy/Huber/SoftL1/Arctan robust loss, optional shared-focal optimization, capped observations for memory, `BundleAdjustmentProblem` / `BundleAdjustmentResult` dataclasses.
- **`filtering.py`** — Observation filtering (reprojection error, track length), point filtering (median/max reprojection error), degenerate camera detection.
- **`config.py`** — Merges `configs/default.yaml` with per-scene YAML, path resolution relative to project root.
- **`runtime.py`** — `SfMRuntimeContext` — aggregates config, image paths, keypoints, cameras into a single frozen object passed through the SfM pipeline.
- **`residuals.py`** — In-memory residual evaluation and report/NPZ export.
- **`track_cache.py`** — Track construction and caching for triangulation.

**Dataclass convention:** Most data structures use `@dataclass(frozen=True)` — they are value objects that should not be mutated. Reconstruction state uses mutable containers (lists, dicts) to accumulate incremental results.

### Tools CLI Layer (`src/tools/`)

Each tool is a runnable `python -m src.tools.<name>` entry point.

**SfM Pipeline (canonical workflow):**

```
extract_features → match_features → verify_matches → initialize_reconstruction → run_paper_aligned_sfm
```

- **`extract_features.py`** — RootSIFT feature extraction
- **`match_features.py`** — Supports `--strategy exhaustive|sequential|retrieval`
- **`verify_matches.py`** — Geometric verification, scene graph construction
- **`initialize_reconstruction.py`** — Two-view initialization with pair scoring
- **`run_paper_aligned_sfm.py`** — **Main controller.** Orchestrates the incremental loop: PnP registration → triangulation → local BA → filtering → growth-triggered global BA → final refinement. This is the single recommended entry point for SfM reconstruction.

**SfM Diagnostics & Utilities:**

- **`register_images.py`** — Incremental image registration (sub-step; usually called via the controller)
- **`triangulate_registered_tracks.py`** — Triangulate tracks after registration or BA
- **`run_bundle_adjustment.py`** — Standalone local/global BA entry point
- **`evaluate_degenerate_cameras.py`** — Diagnose degenerate camera geometry; removal requires `--remove-degenerate-cameras`
- **`evaluate_registered_residuals.py`** — Evaluate registered-image residuals
- **`manage_reconstruction_checkpoint.py`** — Inspect/restore reconstruction checkpoints
- **`export_colmap_model.py`** — Write current sparse model as COLMAP text files
- **`validate_features.py`** — Validate extracted feature files
- **`prepare_scenes.py`** — Create scene working directories from raw datasets

**3DGS Handoff:**

- **`prepare_3dgs_from_sfm.py`** — Export SfM reconstruction to 3DGS-compatible COLMAP text format
- **`prepare_3dgs_from_vggt.py`** — Export VGGT predictions to 3DGS-compatible COLMAP text format
- **`check_3dgs_scene.py`** — CPU-side validation of 3DGS source directories (camera model, image consistency, point cloud, optional PLY)
- **`train_vggt_gsplat.py`** — Minimal gsplat training initialized from VGGT outputs

**VGGT / DTU Benchmark & Evaluation:**

- **`run_vggt_inference.py`** — Lightweight VGGT inference on a single image folder
- **`run_vggt_dtu_benchmark.py`** — Batch VGGT depth prediction for a single DTU scan (GPU required)
- **`align_to_dtu_gt.py`** — Align predicted point cloud to DTU GT via camera-center Umeyama Sim(3)
- **`evaluate_dtu.py`** — Evaluate pose-aligned VGGT point clouds with the DTU point protocol
- **`evaluate_pointcloud.py`** — Generic point cloud evaluation (Accuracy, Completeness, Overall) against GT
- **`dtu_camera_utils.py`** — DTU camera center loading from extrinsics/projection matrices
- **`umeyama.py`** — Umeyama Sim(3) alignment implementation

See `src/tools/README.md` for the full tool index.

### Adapters (`src/adapters/`)

- **`vggt_adapter.py`** — One function: `ensure_vggt_importable()` adds `third_party/vggt` to `sys.path`. Does not embed VGGT logic — strictly a path helper.

### Config System

```
configs/
  default.yaml                  # Global SfM parameters (feature type, RANSAC, BA loss, etc.)
  third_party.yaml              # Third-party paths: VGGT, 3DGS, VGGT-omega locations and entry points
  scenes/<scene>.yaml           # Per-scene paths and subset config
```

Scene configs only specify paths and scene identity — all algorithmic parameters live in `default.yaml`. The merge is done by `src/sfm/config.py:load_scene_config()`. Third-party tool paths are resolved from `third_party.yaml`.

### Viewer (`viewer/`)

A FastAPI + Gradio web viewer for comparing camera poses from different sources (SfM, VGGT, COLMAP). Uses Three.js for 3D rendering and `gaussian-splats-3d` for point cloud visualization. Dependencies: `gradio>=6.0`, `fastapi>=0.115`, `uvicorn>=0.30` (has its own `requirements.txt`).

Launch:
```bash
python -m viewer.app --host 0.0.0.0 --port 7860
```

Preset scenes are configured in `viewer/examples/<example>/` (each with `scene.json`, `cameras.json`). Large PLY assets are gitignored; prepare them with:
```bash
python -m viewer.prepare_examples --mode symlink
```

### Key Directories (gitignored)

Everything under `data/` and `outputs/` is excluded from Git except `.gitkeep` files:
- `data/scenes/<scene>/images/` — input images
- `data/scenes/<scene>/features/`, `matches/`, `verified/`, `tracks/`, `sparse/0/` — pipeline caches
- `data/3dgs_inputs/<scene>_sfm/`, `data/3dgs_inputs/<scene>_vggt/` — 3DGS source directories
- `outputs/<scene>/reports/` — diagnostic reports
- `outputs/3dgs/<scene>_{sfm,vggt}/` — 3DGS training outputs
- `outputs/vggt_dtu_benchmark/` — VGGT DTU benchmark results

### Other Top-Level Directories

- **`docs/`** — Design documents: paper-aligned pipeline spec, reproduction manual, 3DGS pipeline, artifact policy, third-party compliance
- **`report/`** — LaTeX final report source (uses `elegantpaper.cls`)
- **`scripts/`** — Pipeline entry scripts: `run_sfm_full.ps1` (Windows PowerShell), `run_vggt_dtu_benchmark.sh` (multi-GPU tmux batch)
- **`datasets/`** — Dataset metadata or small reference files
- **`checkpoints/`** — Saved model checkpoints (gitignored)
- **`tmp/`** — Debug scripts and experimental scratch work
- **`licenses/`** — Third-party license notices
- **`dtu/`** — DTU dataset reference (expected layout: `dtu/DTU/scan<N>/image/`, `dtu/Points/`)

## Common Commands

### Run full SfM pipeline for a DTU scene (Windows PowerShell)

```powershell
.\scripts\run_sfm_full.ps1 -Scene dtu_scan55
```

### Run full SfM pipeline for a DTU scene (Linux/macOS, manual)

```bash
conda activate mm26
python -m src.tools.extract_features --scene configs/scenes/dtu_scan55.yaml
python -m src.tools.match_features --scene configs/scenes/dtu_scan55.yaml --strategy exhaustive
python -m src.tools.verify_matches --scene configs/scenes/dtu_scan55.yaml
python -m src.tools.initialize_reconstruction --scene configs/scenes/dtu_scan55.yaml
python -m src.tools.run_paper_aligned_sfm \
  --scene configs/scenes/dtu_scan55.yaml \
  --max-register 70 \
  --strict-pnp-median-error 4.0 --max-pnp-median-error 6.0 \
  --strict-pnp-mean-error 6.0 --max-pnp-mean-error 8 \
  --local-ba-after-registration --run-final-refinement \
  --global-ba-growth-ratio 1.3 --global-ba-min-interval 4 \
  --global-ba-max-iterations 40 --global-ba-max-points 0 --global-ba-max-observations 0 \
  --diagnose-degenerate-cameras \
  --report-name dtu_scan55_paper_aligned_sfm_report.json
```

### Prepare 3DGS input from SfM and validate

```bash
python -m src.tools.prepare_3dgs_from_sfm \
  --scene configs/scenes/dtu_scan55.yaml \
  --output-root data/3dgs_inputs --output-name dtu_scan55_sfm

python -m src.tools.check_3dgs_scene \
  --source-path data/3dgs_inputs/dtu_scan55_sfm --write-ply
```

### Train and render with 3DGS

```bash
conda activate gaussian_splatting
python third_party/gaussian-splatting/train.py -s data/3dgs_inputs/dtu_scan55_sfm -m outputs/3dgs/dtu_scan55_sfm
python third_party/gaussian-splatting/render.py -m outputs/3dgs/dtu_scan55_sfm
python third_party/gaussian-splatting/metrics.py -m outputs/3dgs/dtu_scan55_sfm
```

### Run VGGT inference

```bash
conda activate vggt
python -m src.tools.run_vggt_inference \
  --image_folder data/scenes/dtu_scan55/images \
  --output_dir outputs/dtu_scan55/vggt
```

### Run VGGT DTU benchmark (single scan)

```bash
conda activate vggt
python -m src.tools.run_vggt_dtu_benchmark \
  --scan-dir dtu/DTU/scan55 \
  --output-dir outputs/vggt_dtu_benchmark/scan55 \
  --ckpt vggt/hf_cache/hub/models--facebook--VGGT-1B/snapshots/*/model.pt
```

### Run VGGT DTU benchmark (all 15 scans, multi-GPU tmux)

```bash
bash scripts/run_vggt_dtu_benchmark.sh launch
tmux attach -t vggt_dtu_benchmark
```

This spawns 8 GPU workers across a tmux session, runs inference + point cloud evaluation per scan, then aggregates results into `outputs/vggt_dtu_benchmark/summary.json` and `metrics.csv`.

### Evaluate point clouds against GT

```bash
# Generic point cloud evaluation (ICP alignment)
python -m src.tools.evaluate_pointcloud \
  --pred outputs/vggt_dtu_benchmark/scan55/points_depth.ply \
  --gt dtu/Points/stl/stl055_total.ply \
  --alignment icp --num-samples 200000

# Align to DTU GT first, then evaluate
python -m src.tools.align_to_dtu_gt \
  --predictions-npz outputs/scan55/predictions.npz \
  --gt-cameras dtu/DTU/scan55/cameras.npz \
  --input-ply outputs/scan55/points_depth.ply \
  --output-ply outputs/scan55/points_depth_aligned.ply

python -m src.tools.evaluate_pointcloud \
  --pred outputs/scan55/points_depth_aligned.ply \
  --gt dtu/Points/stl/stl055_total.ply \
  --alignment none
```

### Environment check

```bash
python -m src.tools.check_env
```

### Inspect any tool's full CLI

```bash
python -m src.tools.run_paper_aligned_sfm --help
```

## Design Constraints

- **No CUDA in main SfM environment** — VGGT and 3DGS run in separate conda environments. The SfM pipeline uses CPU-only SciPy/NumPy.
- **Third-party code is vendored, not patched** — Never modify files under `third_party/`. If a workaround is needed, do it in `src/adapters/` or `src/tools/`.
- **Frozen dataclasses for value types** — Most geometry/state dataclasses are `frozen=True`. Mutation goes into the reconstruction container (lists of images, dicts of points), not individual records.
- **SciPy LM replaces Ceres** — The BA implementation uses `scipy.optimize.least_squares` with capped observations; it is not a Schur-complement/Ceres solver. This is a known simplification.
- **Fixed intrinsics model** — Only `SIMPLE_PINHOLE` (shared fx=fy, fixed principal point, no distortion). Radial/tangential distortion is not modeled in the current pipeline.
- **DTU is the primary benchmark** — Scene configs target DTU scans; other datasets (South Building, Gerrard Hall, etc.) are supplementary.
- **Reconstruction states are checkpointed** — The controller saves `reconstruction_state.json` after each major step; use `manage_reconstruction_checkpoint.py` to inspect/restore.
- **Filtering is measurement-first** — Residual reports and filtering are separate stages. BA → residual evaluation → filtering → post-filter evaluation is the canonical sequence.
- **Two VGGT copies exist** — `third_party/vggt/` is the clean upstream; `vggt/vggt/` is the experiment workspace with built extensions and cached checkpoints. The adapter (`src/adapters/vggt_adapter.py`) points at `third_party/vggt/`. Do not mix code between them.
- **DTU evaluation has two alignment paths** — `evaluate_pointcloud.py` supports ICP or camera-center Sim(3) alignment. For VGGT DTU benchmarks, the canonical path is: inference → point cloud → ICP evaluation. Alternatively, use `align_to_dtu_gt.py` to pre-compute camera-center Umeyama, then evaluate with `--alignment none`.
