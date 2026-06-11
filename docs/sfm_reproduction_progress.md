# SfM Reproduction Progress

本文档用于汇总当前已经完成的 Incremental SfM 复现过程。它不是理论手册，而是实际工程执行记录，覆盖：

- 已完成的阶段
- 每阶段实际输入输出
- 已实现代码
- 运行命令
- 验收结果
- 当前已知问题与下一步方向

当前复现主线数据集：

```text
data/scenes/south_building_small/
```

辅助数据集已准备：

```text
data/scenes/drjohnson/
data/scenes/graham_hall/
data/scenes/playroom/
data/scenes/south_building/
data/scenes/train/
data/scenes/truck/
```

---

## 1. 当前复现范围

截至目前，已经完成的阶段为：

```text
Phase 0  环境准备
Phase 1  Feature Extraction
Phase 2  Feature Matching
Phase 3  Geometric Verification
Phase 4  Two-view Initialization
Phase 5A Incremental Image Registration
```

尚未完成的阶段为：

```text
Phase 6  Triangulation
Phase 7  Bundle Adjustment
Phase 8  Sparse Model Export
Phase 9  Interface to 3DGS
```

---

## 2. Phase 0 环境准备

### 2.1 已完成内容

- 建立了项目目录框架。
- 配置了 `configs/default.yaml` 和各 scene 配置文件。
- 安装了 `mm26` 环境依赖。
- 准备了环境检查脚本。
- 明确了数据集下载与整理规范。

### 2.2 关键文件

- `requirements.txt`
- `configs/default.yaml`
- `configs/scenes/*.yaml`
- `src/tools/check_env.py`
- `docs/dataset_download.md`

### 2.3 环境检查命令

```bash
conda run -n mm26 python -m src.tools.check_env
```

### 2.4 验收结果

环境检查通过，依赖包括：

- `numpy`
- `scipy`
- `opencv-contrib-python`
- `pycolmap`
- `open3d`
- `networkx`
- `matplotlib`
- `Pillow`
- `PyYAML`
- `tqdm`

---

## 3. Phase 1 Feature Extraction

### 3.1 已完成内容

实现了基于 OpenCV SIFT 的 RootSIFT 特征提取，并保存为 `.npz` 特征缓存。

### 3.2 已实现代码

- `src/sfm/features.py`
- `src/tools/extract_features.py`
- `src/tools/validate_features.py`

### 3.3 特征文件格式

每张图像输出：

```text
data/scenes/<scene>/features/<image_stem>.npz
```

字段：

- `image_name`
- `image_size`
- `resized_size`
- `scale_factor`
- `keypoints`
- `descriptors`

### 3.4 运行命令

```bash
conda run -n mm26 python -m src.tools.extract_features --scene configs/scenes/south_building_small.yaml
conda run -n mm26 python -m src.tools.extract_features --scene configs/scenes/south_building.yaml
conda run -n mm26 python -m src.tools.validate_features --scene configs/scenes/south_building_small.yaml
conda run -n mm26 python -m src.tools.validate_features --scene configs/scenes/south_building.yaml
```

### 3.5 验收结果

#### `south_building_small`

```text
Images/features: 50 / 50
Keypoints min/mean/max: 8192 / 8192.2 / 8193
Mean descriptor L2 norm: 1.000000
Missing: 0
Invalid: 0
```

#### `south_building`

```text
Images/features: 128 / 128
Keypoints min/mean/max: 8192 / 8192.2 / 8194
Mean descriptor L2 norm: 1.000000
Missing: 0
Invalid: 0
```

### 3.6 可视化输出

```text
outputs/<scene>/figures/keypoints_*.jpg
```

---

## 4. Phase 2 Feature Matching

### 4.1 已完成内容

实现了 RootSIFT 描述子的两两匹配，使用：

- BFMatcher
- `knnMatch(k=2)`
- Lowe ratio test
- mutual check

### 4.2 已实现代码

- `src/sfm/matching.py`
- `src/tools/match_features.py`

### 4.3 匹配输出

每个图像对输出：

```text
data/scenes/<scene>/matches/<image1>__<image2>.npz
```

字段：

- `image_name1`
- `image_name2`
- `feature_path1`
- `feature_path2`
- `matches`
- `distances`
- `ratios`

### 4.4 运行命令

```bash
conda run -n mm26 python -m src.tools.match_features --scene configs/scenes/south_building_small.yaml --preview-count 3
```

### 4.5 验收结果

对 `south_building_small`：

```text
Images: 50
Pairs: 1225
Match files: 1225
Matches min/mean/max: 2 / 189.9 / 3725
Pairs >= 30 matches: 361
```

### 4.6 可视化与报告

- `outputs/south_building_small/reports/matching_report.json`
- `outputs/south_building_small/figures/matches_*.jpg`

---

## 5. Phase 3 Geometric Verification

### 5.1 已完成内容

实现了基于 Fundamental Matrix + RANSAC 的几何验证，将 raw matches 过滤为 verified inlier matches，并构建 scene graph。

当前第一版只实现：

- Fundamental Matrix
- RANSAC
- verified inlier matches
- scene graph

暂未实现：

- homography 分类
- essential calibration check
- watermark / timestamp / frame 过滤
- general / planar / panoramic 分类

### 5.2 已实现代码

- `src/sfm/geometry.py`
- `src/tools/verify_matches.py`

### 5.3 输出文件

```text
data/scenes/south_building_small/verified/*.npz
data/scenes/south_building_small/verified/scene_graph.json
outputs/south_building_small/reports/geometric_verification_report.json
outputs/south_building_small/figures/verified_*.jpg
```

### 5.4 运行命令

```bash
conda run -n mm26 python -m src.tools.verify_matches --scene configs/scenes/south_building_small.yaml --max-pairs 100 --preview-count 2 --force
conda run -n mm26 python -m src.tools.verify_matches --scene configs/scenes/south_building_small.yaml --preview-count 3
```

### 5.5 验收结果

```text
Pairs: 1225
Verified pairs: 283
Status counts:
  verified: 283
  low_inliers: 78
  too_few_matches: 864
Inliers min/mean/max: 0 / 167.1 / 3635
```

scene graph 验收：

```text
num_pairs = 1225
num_verified_pairs = 283
nodes = 50
edges = 283
```

### 5.6 当前意义

这一阶段已经为初始化提供了足够的候选图像对，说明：

- raw matching 能找到稳定重叠图像对
- `south_building_small` 具备进入 two-view initialization 的几何基础

---

## 6. Phase 4 Two-view Initialization

### 6.1 已完成内容

实现了：

- 相机内参 bootstrap 估计
- 从 scene graph 中选择初始化 pair
- `F -> E -> recoverPose`
- DLT 三角化
- 初始 3D 点过滤
- 初始化结果保存

### 6.2 已实现代码

- `src/sfm/camera.py`
- `src/sfm/triangulation.py`
- `src/sfm/initialization.py`
- `src/tools/initialize_reconstruction.py`

### 6.3 输出文件

```text
data/scenes/south_building_small/sparse/0/initial_pair.npz
data/scenes/south_building_small/sparse/0/initial_points.npz
outputs/south_building_small/reports/initialization_report.json
outputs/south_building_small/figures/initial_pair_matches.jpg
```

### 6.4 运行命令

```bash
conda run -n mm26 python -m src.tools.initialize_reconstruction --scene configs/scenes/south_building_small.yaml
```

### 6.5 验收结果

```text
Selected pair: P1180189.JPG <-> P1180190.JPG
Candidate verified pairs: 283
Points candidate/kept: 3635 / 697
Median reprojection error: 6.790 px
Mean reprojection error: 6.178 px
Baseline norm: about 1.0
```

### 6.6 当前解释

`3635 / 697` 的差异是正常现象，因为：

- F-matrix inlier 不等于可稳定三角化点
- 当前内参为近似 bootstrap 值
- 使用了正深度和重投影误差过滤
- DLT 尚未经过 BA 优化

详见专题分析文档：

- `docs/phase4_initialization_analysis.md`

### 6.7 当前结论

虽然保留率约为 `19.2%`，但 697 个初始 3D 点已经足够支撑后续增量注册。

---

## 7. Phase 5A Incremental Image Registration

### 7.1 已完成内容

实现了：

- reconstruction state
- `(image_name, keypoint_idx) -> point3D_id` 查询
- 从 verified matches 构建未注册图像的 2D-3D correspondences
- PnP + RANSAC 注册新图像
- reconstruction state 保存
- 注册顺序报告
- 相机中心可视化

### 7.2 已实现代码

- `src/sfm/reconstruction.py`
- `src/tools/register_images.py`

### 7.3 输出文件

```text
data/scenes/south_building_small/sparse/0/reconstruction_state.json
data/scenes/south_building_small/sparse/0/registered_images.npz
outputs/south_building_small/reports/registration_report.json
outputs/south_building_small/figures/registered_camera_centers.png
```

### 7.4 运行命令

```bash
conda run -n mm26 python -m src.tools.register_images \
  --scene configs/scenes/south_building_small.yaml \
  --max-register 10 \
  --min-2d3d 30 \
  --min-pnp-inliers 20
```

### 7.5 验收结果

```text
Initial registered images: 2
Final registered images: 7
Successful new registrations: 5
Failed attempts: 5
Points3D: 697
Observations: 2436
```

成功注册图像：

```text
P1180188.JPG  323 inliers  mean error 3.196 px
P1180187.JPG  259 inliers  mean error 3.856 px
P1180186.JPG  242 inliers  mean error 4.136 px
P1180185.JPG  162 inliers  mean error 3.342 px
P1180184.JPG   56 inliers  mean error 3.622 px
```

失败图像主要是：

```text
not_enough_2d3d
```

原因不是 PnP 崩溃，而是当前初始点云覆盖范围还不够。这正是 Phase 6 Triangulation 需要解决的问题。

### 7.6 当前结论

当前系统已经能够：

```text
two-view initialization
  -> build initial sparse points
  -> incrementally register new images
```

说明最小 Incremental SfM 主链已经打通。

---

## 8. 当前代码结构

当前已实现的核心模块：

```text
src/
  sfm/
    camera.py
    config.py
    features.py
    geometry.py
    initialization.py
    matching.py
    reconstruction.py
    triangulation.py
  tools/
    check_env.py
    prepare_scenes.py
    extract_features.py
    validate_features.py
    match_features.py
    verify_matches.py
    initialize_reconstruction.py
    register_images.py
```

---

## 9. 当前主要问题

### 9.1 初始化点保留率较低

```text
3635 -> 697
```

虽然目前可接受，但说明初始化 pair 选择和内参估计还有提升空间。

### 9.2 增量注册受初始点云覆盖限制

Phase 5A 中有一部分图像失败，不是因为匹配错误，而是：

```text
not_enough_2d3d
```

这说明仅靠初始 pair 产生的 3D 点无法覆盖更远图像，需要在 Phase 6 中利用更多已注册图像重新三角化，扩展点云。

### 9.3 还没有 BA

当前位姿和点云还没有经过局部或全局 BA 优化，因此：

- 位姿误差仍有累计风险
- 点云几何精度还不够稳定
- 后续导出 COLMAP 前必须做 BA/过滤

---

## 10. 下一步建议

下一阶段建议进入：

```text
Phase 6 Triangulation
```

目标是：

- 利用当前 7 张已注册图像
- 从 verified matches 中建立新的多视图 correspondence
- 三角化更多 3D 点
- 更新 observation graph
- 为后续 BA 提供更强的几何约束

完成 Phase 6 后，再进入：

```text
Phase 7 Bundle Adjustment
```

这样才能真正从“能跑通”进入“结构更完整、位姿更稳定”的状态。

---

## 11. Phase 6 Triangulation 与 Phase 5B 闭环

### 11.1 已完成内容

在 `south_building_small` 上完成了第一版 Phase 6，并验证其可以和增量注册形成闭环：

- 从已注册图像之间的 `verified/*.npz` inlier matches 构建 feature tracks。
- 对已有 3D 点对应的 track 补充缺失 observation。
- 对尚未绑定 3D 点的 track 执行鲁棒三角化。
- 三角化检查包括：
  - positive depth / cheirality
  - triangulation angle
  - reprojection error
  - 简化 pair-sampling RANSAC
- 将扩展后的点云保存到 `reconstruction_points.npz`，并让后续 `register_images.py` 从完整 reconstruction state 继续，而不是每次回退到 initial two-view state。

### 11.2 新增和修改代码

- `src/sfm/triangulation.py`
  - 新增 n-view DLT triangulation。
  - 新增 triangulation angle 计算。
  - 新增 `robust_triangulate_track`。
- `src/sfm/reconstruction.py`
  - 新增 `load_reconstruction_state`。
  - `save_reconstruction_state` 额外写出 `reconstruction_points.npz`。
- `src/tools/register_images.py`
  - 改为从完整 reconstruction state 继续注册。
  - 注册报告记录真实初始 registered image / points / observations 数量。
- `src/tools/triangulate_registered_tracks.py`
  - 新增 Phase 6 命令行工具。

### 11.3 运行命令

```bash
conda run -n mm26 python -m src.tools.triangulate_registered_tracks --scene configs/scenes/south_building_small.yaml

conda run -n mm26 python -m src.tools.register_images \
  --scene configs/scenes/south_building_small.yaml \
  --max-register 20 \
  --min-2d3d 30 \
  --min-pnp-inliers 20
```

实际执行了多轮 `Triangulation -> Registration -> Triangulation` 闭环。

### 11.4 验收结果

第一轮 Phase 6，从 Phase 5A 的 7 张注册图像出发：

```text
Registered images: 7
Tracks: 8683
Points3D: 697 -> 7223
Observations: 2436 -> 21905
New point reprojection error median/mean: 2.473 / 2.720 px
```

随后重新运行增量注册：

```text
Registered images: 7 -> 11
Successful new registrations: 4
```

第二轮闭环后：

```text
Points3D: 7223 -> 12782
Observations: 22529 -> 41258
New point reprojection error median/mean: 1.614 / 1.974 px
Registered images: 11 -> 21
Successful new registrations: 10
```

最终又基于 21 张已注册图像执行一次 Phase 6，得到当前状态：

```text
Registered images: 21
Tracks: 24480
Points3D: 12782 -> 18198
Observations: 46070 -> 66955
New point reprojection error median/mean: 2.593 / 2.866 px
New point triangulation angle median/mean: 23.384 / 29.324 deg
```

输出报告：

```text
outputs/south_building_small/reports/triangulation_report.json
outputs/south_building_small/reports/registration_report.json
```

重建状态：

```text
data/scenes/south_building_small/sparse/0/reconstruction_state.json
data/scenes/south_building_small/sparse/0/reconstruction_points.npz
data/scenes/south_building_small/sparse/0/registered_images.npz
```

### 11.5 当前结论

Phase 6 已经解决 Phase 5A 的主要瓶颈：仅依赖初始 two-view 点云导致的 `not_enough_2d3d`。系统现在已经从：

```text
two-view initialization
  -> limited PnP registration
```

推进为：

```text
two-view initialization
  -> incremental registration
  -> registered-track triangulation
  -> more registration
  -> more triangulation
```

当前仍需注意：

- 还没有 BA，注册到更远图像后位姿误差会继续累积。
- 个别新增注册图像平均重投影误差偏高，例如 P1180174 / P1180170，需要 Phase 7 BA 和 outlier filtering 清理。
- 当前 track 构建仍是简化版，没有完整实现论文中的 track splitting / recursive RANSAC 多模型拆分。

### 11.6 下一步建议

下一阶段建议进入：

```text
Phase 7 Bundle Adjustment
```

优先目标：

- 实现小规模 SciPy least_squares BA。
- 先固定内参，只优化 registered image poses 和 points3D。
- 使用 Huber 或 Cauchy robust loss。
- BA 后按 reprojection error / track length / triangulation angle 过滤点。
- 再重新执行一轮 registration / triangulation，观察 registered images 和误差是否继续改善。

---

## 12. Phase 3B Multi-model Geometric Verification

### 12.1 执行动机

前一版 Phase 3 只使用 Fundamental Matrix + RANSAC，能够过滤明显错误匹配，但与论文 *Structure-from-Motion Revisited* 的 scene graph augmentation 尚未对齐。论文中的 Phase 3 不仅判断图像对是否几何验证通过，还会为 scene graph 边标注几何类型，用于后续初始化和三角化决策。

因此本轮回到前端，补充多模型几何验证：

```text
F verification
  -> H/F model comparison
  -> E/F calibration diagnostic
  -> E decomposition + median triangulation angle
  -> simplified WTF similarity filtering
  -> model_type annotation
```

### 12.2 已完成内容

新增/修改代码：

- `configs/default.yaml`
  - 新增 H/E/WTF/全景判断相关阈值。
- `src/sfm/geometry.py`
  - 新增 `verify_geometric_models`。
  - 保留旧 `verify_fundamental_matrix` 兼容入口。
  - 新增 Homography RANSAC、Essential RANSAC、border similarity WTF 检测、E 分解后三角化角估计。
- `src/tools/verify_matches.py`
  - 输出 F/H/E/S 多模型统计。
  - `scene_graph.json` 边新增 `model_type`、`calibration_status`、`homography_ratio`、`essential_ratio`、`median_triangulation_angle_deg` 等字段。
- `src/sfm/initialization.py`
  - 初始化 pair 选择优先使用 `general`、`calibrated`、高几何支持度、低 homography ratio 的图像对。

### 12.3 分类逻辑

当前实现尽量贴近论文，但对 bootstrap 内参采用保守 fallback：

```text
1. F RANSAC 得到 NF，NF 足够则几何验证通过。
2. H RANSAC 得到 NH，计算 NH / NF。
3. E RANSAC 得到 NE，计算 NE / NF，用于 calibration_status。
4. 若 E 可靠，则 recoverPose 并三角化，计算 median triangulation angle。
5. 图像边界匹配点估计 similarity transform，作为简化 WTF 检测。
6. 输出 model_type:
   general / planar / panoramic / rejected_wtf / unverified
```

注意：当前内参仍是 bootstrap pinhole，因此 `calibration_status` 是诊断信息，不作为唯一硬拒绝条件。

### 12.4 运行命令

```bash
conda run -n mm26 python -m src.tools.verify_matches \
  --scene configs/scenes/south_building_small.yaml \
  --preview-count 3 \
  --force
```

### 12.5 验收结果

`south_building_small` 上重新生成 Phase 3B 结果：

```text
Pairs: 1225
Verified scene graph edges: 283

Status counts:
  verified: 77
  verified_planar: 201
  verified_panoramic: 5
  low_inliers: 78
  too_few_matches: 864

Model type counts:
  general: 77
  planar: 201
  panoramic: 5
  unverified: 942

Calibration counts:
  calibrated: 361
  uncertain: 864

Inliers min/mean/max:
  0 / 167.1 / 3635

Homography ratio min/mean/max:
  0.000 / 0.226 / 0.995
```

Phase 3B 后，当前初始化选择的候选变为：

```text
P1180183.JPG <-> P1180184.JPG
status = verified
model_type = general
calibration_status = calibrated
num_inliers = 2811
inlier_ratio = 0.978
homography_ratio = 0.555
median_triangulation_angle_deg = 6.679
```

相比旧版只看 `num_inliers * inlier_ratio`，新版会避开 `planar` 和 `panoramic` pair，更符合论文中“初始化不应来自 panoramic，且优先 calibrated/general pair”的原则。

### 12.6 当前影响

本轮只重算了 Phase 3B 的 verified pairs 和 scene graph，没有重跑 Phase 4/5/6。因此：

- 已生成的 Phase 6 reconstruction state 仍是旧 scene graph 下的结果。
- 若要让 Phase 3B 真正影响重建质量，下一步应重新执行：

```text
Phase 4B robust initialization
  -> Phase 5 registration
  -> Phase 6 triangulation
```

建议下一步先重新运行 Phase 4 初始化，比较新旧初始 pair 的点云保留率和重投影误差，再决定是否继续补 Phase 4B scoring。
