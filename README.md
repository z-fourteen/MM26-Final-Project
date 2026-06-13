# Camera Pose Prediction and Point Cloud Reconstruction

> 中文在前，English follows. This repository combines a project-owned incremental SfM pipeline with VGGT and 3D Gaussian Splatting handoff adapters.

## 项目愿景

本项目面向多视图图像中的**相机位姿估计、稀疏点云重建与可渲染 3D 表达生成**问题，目标是把可解释的几何 SfM 流程与前沿 VGGT / 3DGS 工程生态打通。项目既能服务评测者快速复现实验，也能让研究者检查每个几何模块、坐标约定和第三方边界。

核心思想是：在 `src/` 中保留可审计、可诊断、可导出 COLMAP sparse model 的自研几何重建链路；在 `third_party/` 中保持 VGGT 与 3DGS 上游代码边界清晰；最终统一导出到 `data/3dgs_inputs/<scene>_{sfm,vggt}/` 进行对比训练与可视化。

## 关键特性与创新点

**自研 / 项目自主建模部分**

- **Paper-aligned Incremental SfM 控制器**：两视图初始化、增量 PnP 注册、重三角化、局部 BA、过滤、增长触发全局 BA 和最终 refinement 由 `src.tools.run_paper_aligned_sfm` 统一编排。
- **可解释几何模块**：RootSIFT 后处理、匹配策略、F/E/H 几何验证、初始化 pair scoring、DLT 三角化、cheirality / triangulation angle / reprojection gates、鲁棒 residual 评估。
- **Next-Best-View 与注册诊断**：综合 2D-3D 支持、图像网格覆盖、已注册邻居证据、scene graph 信息，记录失败原因并支持 retry / soft accept / degenerate camera diagnostics。
- **本地算力友好的 BA 策略**：基于 SciPy 的局部/全局 BA、共享焦距可选优化、Cauchy/Huber 等鲁棒损失、增长比例触发全局优化，避免小规模实验反复全局重算。
- **3DGS 交付前验证**：导出前后检查 `PINHOLE` 相机、图像尺寸、registered image 引用、sparse point 数量和可选 PLY，降低 CUDA 训练阶段的路径/格式错误。

**对开源 VGGT / 3DGS 的工程接入与扩展**

- **VGGT 分支薄适配**：通过 `src/adapters/vggt_adapter.py` 与 `src/tools/run_vggt_inference.py` 调用 vendored VGGT，不把项目脚本混入上游仓库。
- **统一 COLMAP-style 3DGS 输入**：SfM 与 VGGT 分别导出到 `data/3dgs_inputs/<scene>_sfm/` 和 `data/3dgs_inputs/<scene>_vggt/`，共享 `images/ + sparse/0/` 布局。
- **Graphdeco 3DGS 后端复用**：训练与渲染使用 `third_party/gaussian-splatting/`，项目侧只负责数据准备、检查和实验组织。
- **第三方合规边界**：`third_party/` 尽量保持上游完整；若必须修改，记录在 `docs/third_party_patches.md`。当前管线不需要第三方源码 patch。

## 仓库结构

```text
final/
  README.md
  requirements.txt
  configs/
    default.yaml
    scenes/                       # 每个实验场景的路径、相机与阈值配置
      south_building_small.yaml
      dtu_scan55.yaml
      ...
  src/                            # 项目自研代码与薄适配层
    sfm/                          # 核心创新：几何、匹配、三角化、BA、过滤
      camera.py
      features.py
      matching.py
      geometry.py
      initialization.py
      reconstruction.py
      triangulation.py
      bundle_adjustment.py
      filtering.py
    tools/                        # 可复现实验 CLI 与 3DGS/VGGT handoff
      extract_features.py
      match_features.py
      verify_matches.py
      initialize_reconstruction.py
      run_paper_aligned_sfm.py
      prepare_3dgs_from_sfm.py
      prepare_3dgs_from_vggt.py
      check_3dgs_scene.py
      run_vggt_inference.py
    adapters/
      vggt_adapter.py             # 第三方 VGGT 的薄封装
    innovation/
      README.md                   # 自研创新与工程改造说明
  third_party/                    # 开源引用边界：尽量保持上游结构
    vggt/
    gaussian-splatting/
  data/                           # 本地数据与生成产物，默认不提交
    raw/
    scenes/<scene>/images/
    3dgs_inputs/<scene>_sfm/
    3dgs_inputs/<scene>_vggt/
  outputs/                        # 运行日志、诊断报告、3DGS 输出，默认不提交
  docs/                           # 设计手册、复现实验、合规与 artifact manifest
  report/                         # 最终报告源码与精选图
```

## 快速开始

> **Note**  
> 推荐先跑 `south_building_small` 或一个 30-50 张图像的小场景。VGGT 与 3DGS 依赖 CUDA/PyTorch extension，可能与主 SfM 环境冲突；主项目环境用于 CPU-friendly SfM 与数据检查，VGGT/3DGS 可单独建 CUDA 环境。

### 1. 环境配置

```bash
conda create -n mm26 python=3.10 -y
conda activate mm26
pip install -r requirements.txt
python -m src.tools.check_env
```

如需运行 vendored 3DGS：

```bash
conda env create -f third_party/gaussian-splatting/environment.yml
conda activate gaussian_splatting
```

如需运行 VGGT：

```bash
conda create -n vggt python=3.10 -y
conda activate vggt
pip install -r third_party/vggt/requirements.txt
pip install -e third_party/vggt
```

> **Warning**  
> 3DGS 的 CUDA extension、PyTorch 版本和本机显卡驱动强相关。若安装失败，先确认 CUDA/PyTorch 组合，再回到主环境执行 `check_3dgs_scene` 验证数据格式。

### 2. 数据准备

标准场景目录：

```text
data/scenes/<scene>/
  images/                         # 原始或整理后的输入图像
  features/
  matches/
  verified/
  tracks/
  sparse/0/
```

准备方式：

```bash
# 将数据集图像放入：
# data/scenes/<scene>/images/

# 使用或复制一个场景配置：
# configs/scenes/<scene>.yaml
```

推荐公开数据：COLMAP South Building / Gerrard Hall。自采数据建议 40-80 张，保证相邻图像 60%-80% 重叠，避免纯旋转、玻璃/强反光/大面积弱纹理。

### 3. 运行 SfM 主流程

最小分阶段复现：

```bash
conda activate mm26

python -m src.tools.extract_features --scene configs/scenes/<scene>.yaml
python -m src.tools.match_features --scene configs/scenes/<scene>.yaml --strategy retrieval --retrieval-top-k 20
python -m src.tools.verify_matches --scene configs/scenes/<scene>.yaml
python -m src.tools.initialize_reconstruction --scene configs/scenes/<scene>.yaml
```

推荐主控制器：

```bash
python -m src.tools.run_paper_aligned_sfm \
  --scene configs/scenes/<scene>.yaml \
  --max-register 50 \
  --local-ba-after-registration \
  --global-ba-growth-ratio 1.3 \
  --global-ba-min-interval 4 \
  --filter-after-ba \
  --diagnose-degenerate-cameras \
  --report-name paper_aligned_sfm_report.json
```

导出 COLMAP sparse model：

```bash
python -m src.tools.export_colmap_model --scene configs/scenes/<scene>.yaml
```

### 4. 准备 3DGS 输入

SfM 分支：

```bash
python -m src.tools.prepare_3dgs_from_sfm \
  --scene configs/scenes/<scene>.yaml \
  --output-name <scene>_sfm
```

VGGT 分支：

```bash
python -m src.tools.run_vggt_inference \
  --image_folder data/scenes/<scene>/images \
  --output_dir outputs/<scene>/vggt

python -m src.tools.prepare_3dgs_from_vggt \
  --scene configs/scenes/<scene>.yaml \
  --predictions outputs/<scene>/vggt/predictions.npz \
  --points outputs/<scene>/vggt/points_depth.npz \
  --output-name <scene>_vggt
```

CPU 侧检查 3DGS 输入：

```bash
python -m src.tools.check_3dgs_scene \
  --source-path data/3dgs_inputs/<scene>_sfm \
  --write-ply \
  --report-name outputs/<scene>/reports/<scene>_sfm_3dgs_scene_check.json
```

### 5. 训练、评估与可视化

在 CUDA 环境中训练 3DGS：

```bash
python third_party/gaussian-splatting/train.py \
  -s data/3dgs_inputs/<scene>_sfm \
  -m outputs/3dgs/<scene>_sfm

python third_party/gaussian-splatting/train.py \
  -s data/3dgs_inputs/<scene>_vggt \
  -m outputs/3dgs/<scene>_vggt
```

渲染与指标：

```bash
python third_party/gaussian-splatting/render.py -m outputs/3dgs/<scene>_sfm
python third_party/gaussian-splatting/metrics.py -m outputs/3dgs/<scene>_sfm
```

常用输出位置：

```text
data/scenes/<scene>/sparse/0/                 # SfM sparse reconstruction
data/3dgs_inputs/<scene>_sfm/                 # SfM -> 3DGS source
data/3dgs_inputs/<scene>_vggt/                # VGGT -> 3DGS source
outputs/<scene>/reports/                      # SfM / 3DGS 检查报告
outputs/3dgs/<scene>_{sfm,vggt}/              # 3DGS checkpoints, renders, metrics
```

## 实验与产物策略

- 提交代码、配置、文档、报告源码与小型精选图。
- 不提交原始数据集、复制图像、特征/匹配缓存、大规模 sparse model、3DGS checkpoint、训练日志。
- 重要本地产物记录在 `docs/artifacts_manifest.md`；最终提交时按课程/评审平台要求用 release attachment 或 Git LFS 管理大文件。
- 第三方集成与许可证边界见 `docs/third_party_compliance.md`。

---

## Project Vision

This project addresses **camera pose estimation, sparse point cloud reconstruction, and 3D-renderable scene generation** from multi-view images. It bridges an auditable geometry-first SfM pipeline with modern VGGT and 3D Gaussian Splatting backends.

The repository is designed for two audiences: reviewers who need a smooth reproduction path, and researchers who need to inspect the mathematical assumptions, coordinate conventions, diagnostics, and third-party boundaries.

## Key Features and Innovations

**Project-owned modeling and implementation**

- **Paper-aligned incremental SfM controller**: two-view initialization, PnP registration, triangulation, local BA, filtering, growth-triggered global BA, and final refinement are orchestrated by `src.tools.run_paper_aligned_sfm`.
- **Interpretable geometry pipeline**: RootSIFT post-processing, feature matching, F/E/H verification, initialization scoring, DLT triangulation, cheirality / triangulation-angle / reprojection gates, and residual diagnostics.
- **Next-best-view and registration diagnostics**: candidate scoring combines 2D-3D support, spatial image coverage, registered-neighbor evidence, and scene-graph cues, while failed registrations are logged with actionable reasons.
- **Local-hardware-aware BA**: SciPy-based local/global BA with robust losses, optional shared focal optimization, capped observations, and growth-ratio scheduling.
- **3DGS preflight validation**: project tools validate `PINHOLE` cameras, image paths, image dimensions, sparse points, and optional PLY export before expensive CUDA training.

**Engineering integration with VGGT and 3DGS**

- **Thin VGGT adapter**: VGGT is called through `src/adapters/` and `src/tools/`, keeping the vendored upstream repository clean.
- **Unified COLMAP-style handoff**: SfM and VGGT export into separate `data/3dgs_inputs/<scene>_sfm/` and `data/3dgs_inputs/<scene>_vggt/` roots with the same `images/ + sparse/0/` layout.
- **Vendored 3DGS backend reuse**: training and rendering are delegated to `third_party/gaussian-splatting/`; project code prepares and validates inputs.
- **Clear third-party compliance boundary**: upstream code lives under `third_party/`; project-owned experiments live under `src/`.

## Quick Start

```bash
conda create -n mm26 python=3.10 -y
conda activate mm26
pip install -r requirements.txt
python -m src.tools.check_env
```

Place images under:

```text
data/scenes/<scene>/images/
configs/scenes/<scene>.yaml
```

Run the main SfM pipeline:

```bash
python -m src.tools.extract_features --scene configs/scenes/<scene>.yaml
python -m src.tools.match_features --scene configs/scenes/<scene>.yaml --strategy retrieval --retrieval-top-k 20
python -m src.tools.verify_matches --scene configs/scenes/<scene>.yaml
python -m src.tools.initialize_reconstruction --scene configs/scenes/<scene>.yaml

python -m src.tools.run_paper_aligned_sfm \
  --scene configs/scenes/<scene>.yaml \
  --max-register 50 \
  --local-ba-after-registration \
  --global-ba-growth-ratio 1.3 \
  --global-ba-min-interval 4 \
  --filter-after-ba \
  --diagnose-degenerate-cameras \
  --report-name paper_aligned_sfm_report.json
```

Prepare 3DGS inputs:

```bash
python -m src.tools.prepare_3dgs_from_sfm \
  --scene configs/scenes/<scene>.yaml \
  --output-name <scene>_sfm

python -m src.tools.check_3dgs_scene \
  --source-path data/3dgs_inputs/<scene>_sfm \
  --write-ply \
  --report-name outputs/<scene>/reports/<scene>_sfm_3dgs_scene_check.json
```

Train and render with the vendored 3DGS backend on a CUDA machine:

```bash
python third_party/gaussian-splatting/train.py \
  -s data/3dgs_inputs/<scene>_sfm \
  -m outputs/3dgs/<scene>_sfm

python third_party/gaussian-splatting/render.py -m outputs/3dgs/<scene>_sfm
python third_party/gaussian-splatting/metrics.py -m outputs/3dgs/<scene>_sfm
```

## Reproducibility Notes

- Main CPU-friendly dependencies are listed in `requirements.txt`.
- VGGT and 3DGS may require separate CUDA-enabled environments.
- Large datasets, sparse reconstructions, checkpoints, and logs are intentionally excluded from normal Git history.
- See `docs/artifacts_manifest.md` for artifact policy and `docs/third_party_compliance.md` for attribution and integration rules.
