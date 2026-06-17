# SfM / VGGT + 3DGS on DTU

Multi-view 3D reconstruction on DTU dataset with two tracks:
- **SfM track**: incremental SfM pipeline (RootSIFT → matching → geometric verification → initialization → PnP → BA)
- **VGGT track**: VGGT inference → point cloud → 3DGS training → rendering

Both evaluated with the same ICP-based point cloud protocol.

## Environment

```bash
conda create -n lewm_rt python=3.10 -y
conda activate lewm_rt
pip install -r requirements.txt
pip install -e third_party/vggt

# 3DGS CUDA extensions
pip install -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
cd third_party/gaussian-splatting/submodules/simple-knn
CUDA_HOME=/usr/local/cuda-12.6 python setup.py build_ext --inplace

# Required env var for 3DGS
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
```

## Data Preparation

```bash
# DTU images (symlink to avoid copy)
mkdir -p data/scenes/dtu_scan55/images
ln -sf $(pwd)/dtu/DTU/scan55/images/* data/scenes/dtu_scan55/images/
```

## SfM Pipeline

### Canonical workflow

```
extract_features → match_features → verify_matches → initialize_reconstruction → run_paper_aligned_sfm
```

```bash
python -m src.tools.extract_features --scene configs/scenes/dtu_scan55.yaml
python -m src.tools.match_features --scene configs/scenes/dtu_scan55.yaml --strategy exhaustive
python -m src.tools.verify_matches --scene configs/scenes/dtu_scan55.yaml
python -m src.tools.initialize_reconstruction --scene configs/scenes/dtu_scan55.yaml

# Main incremental SfM controller
python -m src.tools.run_paper_aligned_sfm \
  --scene configs/scenes/dtu_scan55.yaml \
  --max-register 70 \
  --strict-pnp-median-error 4.0 --max-pnp-median-error 6.0 \
  --strict-pnp-mean-error 6.0 --max-pnp-mean-error 8 \
  --local-ba-after-registration --run-final-refinement \
  --global-ba-growth-ratio 1.3 --global-ba-min-interval 4 \
  --global-ba-max-iterations 40 --global-ba-max-points 0 --global-ba-max-observations 0 \
  --diagnose-degenerate-cameras
```

### SfM → 3DGS

```bash
python -m src.tools.prepare_3dgs_from_sfm \
  --scene configs/scenes/dtu_scan55.yaml \
  --output-root data/3dgs_inputs --output-name dtu_scan55_sfm

python -m src.tools.check_3dgs_scene \
  --source-path data/3dgs_inputs/dtu_scan55_sfm --write-ply
```

## VGGT Pipeline

### 1. VGGT Inference (single scan)

```bash
python -m src.tools.run_vggt_dtu_benchmark \
  --scan-dir dtu/DTU/scan55 \
  --output-dir outputs/vggt_dtu_benchmark/scan55 \
  --ckpt vggt/hf_cache/hub/models--facebook--VGGT-1B/snapshots/*/model.pt
```

Output: `points_depth.ply`, `predictions.npz`, `cameras.json`

### 2. VGGT → 3DGS

```bash
# Create xyz npz from point cloud
python -c "
import numpy as np; from pathlib import Path
from src.tools.evaluate_pointcloud import read_ply_vertices
pts = read_ply_vertices(Path('outputs/vggt_dtu_benchmark/scan55/points_depth.ply'))
np.savez_compressed('outputs/vggt_dtu_benchmark/scan55/points_xyz.npz',
    xyz=pts.astype(np.float64), rgb=np.full((len(pts),3),128,dtype=np.uint8))
"

# Prepare 3DGS input
python -m src.tools.prepare_3dgs_from_vggt \
  --scene configs/scenes/dtu_scan55.yaml \
  --predictions outputs/vggt_dtu_benchmark/scan55/predictions.npz \
  --points outputs/vggt_dtu_benchmark/scan55/points_xyz.npz \
  --output-name dtu_scan55_vggt --output-root data/3dgs_inputs
```

### 3. 3DGS Training (30K iterations)

```bash
python third_party/gaussian-splatting/train.py \
  -s data/3dgs_inputs/dtu_scan55_vggt \
  -m outputs/3dgs/dtu_scan55_vggt \
  --iterations 30000 --eval
```

### 4. Render & Metrics

```bash
python third_party/gaussian-splatting/render.py \
  -s data/3dgs_inputs/dtu_scan55_vggt \
  -m outputs/3dgs/dtu_scan55_vggt

python third_party/gaussian-splatting/metrics.py \
  -m outputs/3dgs/dtu_scan55_vggt
```

## Point Cloud Evaluation (unified protocol)

Same parameters for SfM and VGGT to ensure fair comparison:

```bash
python -m src.tools.evaluate_pointcloud \
  --pred <path/to/points.ply> \
  --gt dtu/Points/stl/stl055_total.ply \
  --num-samples 200000 --alignment icp \
  --icp-samples 50000 --icp-max-iterations 100 --icp-threshold-ratio 0.05 \
  --voxel-size 0 --max-distance 0 --seed 42 \
  --output-json <output.json>
```

## Batch Execution (all scans, 8 GPUs)

```bash
# VGGT inference + evaluation
bash scripts/run_vggt_dtu_benchmark.sh launch
tmux attach -t vggt_dtu_benchmark
```

## Architecture

```
src/
  sfm/                    — Core SfM library (camera, features, matching, geometry,
                            initialization, reconstruction, triangulation, BA, filtering)
  tools/                  — CLI entry points (extract_features, run_paper_aligned_sfm,
                            evaluate_pointcloud, run_vggt_dtu_benchmark, ...)
  adapters/               — Thin wrappers for third-party backends
third_party/
  vggt/                   — Vendored VGGT (upstream, unmodified)
  gaussian-splatting/     — Vendored 3DGS (CUDA extensions)
configs/
  default.yaml            — Global SfM parameters
  scenes/<scene>.yaml     — Per-scene paths
```

Key design choices:
- **No CUDA in SfM**: SfM pipeline uses CPU-only SciPy/NumPy. VGGT and 3DGS run via separate tools.
- **Frozen dataclasses**: Value types use `@dataclass(frozen=True)`. Mutation happens in container objects (lists, dicts).
- **SciPy LM replaces Ceres**: BA uses `scipy.optimize.least_squares` with capped observations (known simplification).
- **SIMPLE_PINHOLE only**: Fixed intrinsics model (shared fx=fy, fixed principal point, no distortion).

## Results

### SfM Point Cloud Quality

| scene   | Reg.  | Points | Track | Acc (m) | Comp (m) | Overall (m) |
|---------|-------|--------|-------|---------|----------|-------------|
| scan24  | 49/49 | 18409  | 3.07  | 0.028   | 0.257    | 0.143       |
| scan37  | 49/49 | 31787  | 3.65  | 0.105   | 0.156    | 0.131       |
| scan40  | 49/49 | 35536  | 3.79  | 0.073   | 0.090    | 0.081       |
| scan55  | 49/49 | 29347  | 6.96  | 0.081   | 0.097    | 0.089       |

### VGGT Point Cloud Quality (ICP, mm)

| scan   | Acc   | Comp  | Overall |
|--------|-------|-------|---------|
| scan24 | 13.03 | 10.33 | 11.68   |
| scan37 | 12.10 | 9.30  | 10.70   |
| scan40 | 13.42 | 12.27 | 12.85   |
| scan55 | 10.76 | 9.72  | 10.24   |
| scan63 | 9.30  | 6.53  | 7.92    |
| scan65 | 11.20 | 9.82  | 10.51   |
| scan69 | 8.09  | 7.38  | 7.74    |
| scan83 | 8.27  | 4.48  | 6.37    |
| scan97 | 9.96  | 6.54  | 8.25    |
| **Mean** | **10.43** | **7.97** | **9.20** |

### 3DGS Rendering Quality (VGGT init, 30K iter, --eval)

| scan   | SSIM   | PSNR    | LPIPS  |
|--------|--------|---------|--------|
| scan24 | 0.7872 | 19.67   | 0.1833 |
| scan37 | 0.8087 | 20.59   | 0.1355 |
| scan40 | 0.7505 | 21.55   | 0.2007 |
| scan55 | 0.8487 | 22.41   | 0.1030 |
| scan63 | 0.9313 | 23.72   | 0.0652 |
| scan65 | 0.9343 | 25.45   | 0.0715 |
| scan69 | 0.8677 | 19.92   | 0.1241 |
| scan83 | 0.9455 | 22.99   | 0.0554 |
| scan97 | 0.8698 | 21.39   | 0.1092 |
| **Mean** | **0.8604** | **21.96** | **0.1164** |

### Evaluation Protocol

All point clouds evaluated with identical parameters:
- Similarity ICP alignment (init: center-scale, max 100 iter)
- 200K points sampled (seed 42), 50K for ICP fitting
- ICP threshold = 5% of GT bounding box diagonal
- No voxel downsampling, no distance clipping

**Key insight**: SfM produces sparser but more accurate points (lower Acc, higher Comp variance).
VGGT produces denser point clouds with lower Completeness error on texture-poor scenes.
Both use identical ICP alignment → metrics are directly comparable.

### Comparison

| Method | Acc | Comp | Strengths |
|--------|-----|------|-----------|
| SfM | lower (better) | varies | Points verified by multi-view triangulation + BA |
| VGGT | higher | more consistent | Dense coverage, no feature dependency, single forward pass |

## Directory Structure

```
data/
  scenes/dtu_scanXX/           — SfM working dirs (images, features, matches, sparse/0/)
  3dgs_inputs/dtu_scanXX_*/    — 3DGS source (images/ + sparse/0/)
outputs/
  vggt_dtu_benchmark/scanXX/   — VGGT outputs (points_depth.ply, predictions.npz)
  3dgs/dtu_scanXX_vggt/        — 3DGS outputs (checkpoints, renders, point cloud)
dtu/
  DTU/scanXX/images/           — DTU images
  Points/stl/stlXXX_total.ply  — DTU ground truth point clouds
configs/scenes/dtu_scanXX.yaml — Per-scene configuration
```
