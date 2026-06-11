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
  - 初始化 pair 选择严格依赖 Phase 3B scene graph 字段，优先使用 `general`、`calibrated`、高几何支持度、低 homography ratio 的图像对。
  - 不再兼容旧版只含 `num_inliers / inlier_ratio` 的 scene graph；后续阶段统一按 Phase 3B 输出契约迭代。

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

---

## 13. Phase 4B Robust Two-view Initialization

### 13.1 执行动机

Phase 3B 已经为 scene graph 边补充了 `model_type`、`calibration_status`、`homography_ratio`、`essential_ratio`、`median_triangulation_angle_deg` 等属性，因此初始化不应再只按 `num_inliers * inlier_ratio` 选择 pair。

本轮将 Phase 4 升级为：

```text
Phase 3B scene graph
  -> candidate ranking
  -> top-k initialization attempts
  -> quality checks
  -> selected robust two-view seed
```

### 13.2 已完成内容

新增/修改代码：

- `configs/default.yaml`
  - 新增 `min_initial_points`、`min_initial_kept_ratio`、`max_initial_median_reproj_error_px`。
- `src/sfm/initialization.py`
  - 新增 `rank_initial_pair_candidates`。
  - 初始化候选排除 `panoramic / rejected_wtf / unverified`。
  - `general` 优先，`planar` 保留但降权。
  - `calibrated` 加分，`uncertain` 保留但不加分。
  - score 综合使用 inliers、inlier ratio、homography ratio、essential ratio、triangulation angle、graph degree。
  - 新增初始化质量指标：pose inliers、kept ratio、cheirality ratio、初始化后三角化角。
  - 重新初始化时自动清理旧的 `reconstruction_state.json`、`registered_images.npz`、`reconstruction_points.npz`。
- `src/tools/initialize_reconstruction.py`
  - 新增 `--max-candidate-pairs`。
  - 输出 `initialization_candidates_report.json`。
  - 从 top-k 候选中逐个尝试，选择第一个满足质量阈值的 pair。

### 13.3 运行命令

```bash
conda run -n mm26 python -m src.tools.initialize_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --max-candidate-pairs 20
```

### 13.4 验收结果

Phase 4B 选择结果：

```text
Selected pair: P1180142.JPG <-> P1180143.JPG
Selected rank: 2 / 20
status = verified
model_type = general
calibration_status = calibrated
num_inliers = 1775
inlier_ratio = 0.981
homography_ratio = 0.550
essential_ratio = 1.004
Phase3B median triangulation angle = 2.187 deg
graph_degree_score = 10.954
```

初始化三角化结果：

```text
Candidate points: 1775
Kept points: 1775
Kept ratio: 1.000
Pose inliers: 1756
Cheirality ratio: 1.000
Median initialization triangulation angle: 2.248 deg
Median / mean reprojection error: 1.609 / 1.710 px
Baseline norm: 1.0
```

对比旧 Phase 4：

```text
Old selected pair: P1180189.JPG <-> P1180190.JPG
Old points candidate/kept: 3635 / 697
Old median / mean reprojection error: 6.790 / 6.178 px

New selected pair: P1180142.JPG <-> P1180143.JPG
New points candidate/kept: 1775 / 1775
New median / mean reprojection error: 1.609 / 1.710 px
```

Phase 4B 明显改善了初始化点云质量：保留点更多，重投影误差显著降低。

### 13.5 当前状态影响

由于 Phase 4B 已重新生成：

```text
data/scenes/south_building_small/sparse/0/initial_pair.npz
data/scenes/south_building_small/sparse/0/initial_points.npz
```

并清理了旧的增量状态：

```text
reconstruction_state.json
registered_images.npz
reconstruction_points.npz
```

因此旧 Phase 5/6 的 21-image reconstruction state 已失效。下一步应基于新的 Phase 4B 初始化重新运行：

```text
Phase 5B incremental registration
Phase 6B registered-track triangulation
```

这样才能评估更强前端初始化对完整增量重建链路的真实影响。

---

## 14. Phase 5C Next Best View Scoring

### 14.1 执行动机

前一版 Phase 5A/5B 的图像注册顺序主要按 2D-3D correspondence 数量排序。论文 *Structure-from-Motion Revisited* 中的 next best view selection 更强调可见 3D 点在候选图像平面上的空间分布，因为 PnP 稳定性不仅取决于点数，也取决于点是否覆盖图像不同区域。

本轮 Phase 5C 按论文思想实现 pyramid visibility scoring：

```text
K_l = 2^l
w_l = K_l^2
score(I) = sum_l w_l * occupied_cells_l(I)
```

其中 `occupied_cells_l` 表示第 `l` 层网格中被 2D-3D 点占据的 cell 数量。

### 14.2 已完成内容

新增/修改代码：

- `src/sfm/reconstruction.py`
  - 新增 `image_pyramid_visibility_score`。
  - 新增 `score_next_best_view`。
  - 新增 scene graph 诊断信息统计，但仅用于报告，不参与主排序。
- `src/tools/register_images.py`
  - 注册候选排序改为：

```text
primary key: pyramid_visibility_score
tie-breaker: num_2d3d
```

  - `num_2d3d < min_2d3d` 的候选只进入报告诊断，不再执行 PnP。
  - 每轮报告 top candidates，包括 eligibility、pyramid score、num_2d3d、scene graph 诊断字段。

### 14.3 运行命令

由于 Phase 4B 已重新初始化，本轮先重置到 Phase 4B，然后运行 Phase 5C：

```bash
conda run -n mm26 python -m src.tools.initialize_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --max-candidate-pairs 20

conda run -n mm26 python -m src.tools.register_images \
  --scene configs/scenes/south_building_small.yaml \
  --max-register 20 \
  --min-2d3d 30 \
  --min-pnp-inliers 20 \
  --visibility-levels 3 \
  --top-candidates 5
```

### 14.4 验收结果

Phase 5C 从 Phase 4B 的两视图初始化出发：

```text
Initial registered images: 2
Final registered images: 10
Successful registrations: 8
Failed registrations: 0
Points3D: 1775
Observations: 3550 -> 5210
Selection method: phase5c_pyramid_visibility
Visibility levels: 3
```

注册顺序：

```text
1  P1180141.JPG  num_2d3d=933  inliers=927  score=3073  mean_error=1.746 px
2  P1180144.JPG  num_2d3d=538  inliers=444  score=2369  mean_error=1.247 px
3  P1180145.JPG  num_2d3d=71   inliers=50   score=1649  mean_error=2.237 px
4  P1180152.JPG  num_2d3d=66   inliers=56   score=585   mean_error=3.050 px
5  P1180149.JPG  num_2d3d=54   inliers=51   score=605   mean_error=1.897 px
6  P1180148.JPG  num_2d3d=49   inliers=49   score=457   mean_error=1.628 px
7  P1180150.JPG  num_2d3d=41   inliers=38   score=381   mean_error=2.027 px
8  P1180151.JPG  num_2d3d=45   inliers=45   score=297   mean_error=2.645 px
```

对比旧 Phase 5A：

```text
Old Phase 5A: 2 -> 7 registered images
New Phase 5C: 2 -> 10 registered images
```

### 14.5 当前结论

Phase 5C 的 next best view selection 已经按论文核心思想从“点数量优先”改为“图像平面分布优先”。scene graph 的 `model_type / homography_ratio` 等信息暂时只用于报告诊断，不进入 NBV 主评分，避免引入过多自定义 heuristic。

当前停止原因不是 PnP 失败，而是剩余候选不足 `min_2d3d = 30`。因此下一步应进入：

```text
Phase 6B Registered-track Triangulation
```

目标是利用当前 10 张已注册图像重新三角化，扩展点云覆盖范围，然后继续 Phase 5C 注册。

---

## 15. Phase 6B Robust Registered-track Triangulation after Phase 5C

### 15.1 执行动机

Phase 5C 基于 Phase 4B 初始化完成后，注册图像从 2 张增加到 10 张，但停止于剩余候选不足 `min_2d3d = 30`。这说明当前点云覆盖范围仍然不足，需要利用已经注册的 10 张图像从 feature tracks 中恢复更多 3D 点。

这与论文中的 incremental SfM 闭环一致：

```text
image registration
  -> triangulation
  -> more 2D-3D correspondences
  -> more image registration
```

### 15.2 已完成内容

新增/修改代码：

- `src/sfm/triangulation.py`
  - `robust_triangulate_track` 新增 `valid_pair_mask`，RANSAC 采样只从合法视图对中取两视图。
- `src/tools/triangulate_registered_tracks.py`
  - 严格依赖 Phase 3B verified schema。
  - 构建 tracks 时只使用：

```text
status in {verified, verified_planar}
model_type not in {panoramic, rejected_wtf}
```

  - RANSAC 采样时同样检查 observation pair 是否来自合法 Phase 3B 边。
  - 报告新增 used/skipped edges、edge status/model type counts、ambiguous/conflicting/no-valid-pair 等跳过原因。

### 15.3 运行命令

```bash
conda run -n mm26 python -m src.tools.triangulate_registered_tracks \
  --scene configs/scenes/south_building_small.yaml
```

### 15.4 验收结果

Phase 6B 从 Phase 5C 的 10 张注册图像出发：

```text
Registered images: 10
Input registered edges used: 43
Input registered edges skipped: 2
Input tracks: 4372

Points3D: 1775 -> 3930
Observations: 5210 -> 11279
New points3D: 2155
New observations: 6069
Augmented existing observations: 805
```

几何质量：

```text
Median / mean new point reprojection error: 0.988 / 1.773 px
Median / mean new point triangulation angle: 5.759 / 9.607 deg
```

跳过统计：

```text
skipped_ambiguous_tracks: 15
skipped_conflicting_point_tracks: 0
skipped_no_valid_view_pair_tracks: 0
skipped_geometry_tracks: 434
```

使用边统计：

```text
edge_status_counts:
  verified: 17
  verified_planar: 26
  low_inliers: 1
  too_few_matches: 1

edge_model_type_counts:
  general: 17
  planar: 26
  unverified: 2
```

没有使用 panoramic pair 进行三角化，符合论文中避免从 panoramic image pairs triangulate 的原则。

### 15.5 Phase 6B 后续 Phase 5C 验证

扩点后继续运行 Phase 5C：

```bash
conda run -n mm26 python -m src.tools.register_images \
  --scene configs/scenes/south_building_small.yaml \
  --max-register 20 \
  --min-2d3d 30 \
  --min-pnp-inliers 20 \
  --visibility-levels 3 \
  --top-candidates 5
```

结果：

```text
Registered images: 10 -> 16
Successful new registrations: 6
Failed attempts: 1
Observations: 11279 -> 11517
```

新增注册图像：

```text
P1180146.JPG  num_2d3d=99   inliers=34  mean_error=3.276 px
P1180153.JPG  num_2d3d=141  inliers=77  mean_error=3.631 px
P1180147.JPG  num_2d3d=75   inliers=58  mean_error=2.114 px
P1180154.JPG  num_2d3d=50   inliers=20  mean_error=2.653 px
P1180156.JPG  num_2d3d=40   inliers=26  mean_error=3.388 px
P1180157.JPG  num_2d3d=33   inliers=23  mean_error=9.013 px
```

失败图像：

```text
P1180155.JPG  pnp_failed  num_2d3d=32
```

### 15.6 当前结论

Phase 6B 扩点有效缓解了 Phase 5C 的 `not_enough_2d3d` 瓶颈，使注册图像数量继续从 10 张增加到 16 张。

当前需要注意：

- `P1180157.JPG` 虽然注册成功，但 mean reprojection error 约 9 px，质量偏低。
- 后续需要更严格的 registration acceptance 或 Phase 7 BA/filtering 清理低质量注册。
- 当前还没有执行 BA，因此 16 张图像后的位姿和点云仍可能存在累计误差。

下一步建议：

```text
Phase 6C / Phase 5C loop
  或
Phase 7 Local Bundle Adjustment + filtering
```

如果目标是继续增加注册图像数量，可以先再执行一轮 Phase 6B -> Phase 5C；如果目标是稳定质量，应进入 Phase 7 BA。

---

## 16. Phase 7A Small-scale Bundle Adjustment

### 16.1 执行动机

Phase 5C/6B 闭环已经将 `south_building_small` 推进到 16 张已注册图像，但新增图像中已经出现较高重投影误差，例如 `P1180157.JPG` 的 mean reprojection error 约 9 px。继续盲目执行 Phase 5C/6B 会放大位姿累计误差，因此本阶段先实现一个小规模 BA 验证版，用于稳定当前局部重建并为后续 filtering / 论文级 BA 做准备。

本阶段定位为：

```text
fixed intrinsics
fixed first camera pose as gauge
optimize remaining camera poses + selected 3D points
robust least-squares
```

固定第一张图像的 `R/t` 是为了消除 SfM 的 gauge freedom：整体坐标系可以任意刚体变换和尺度缩放，如果不固定参考相机，BA 的解不唯一，优化器会在等价坐标系中漂移。

### 16.2 已完成内容

新增代码：

- `src/sfm/bundle_adjustment.py`
  - 新增 BA problem 构建。
  - 使用 Rodrigues 向量参数化相机旋转。
  - 固定首张已注册图像的 pose，不放入优化变量。
  - 优化其余 registered image poses 和选中的 `points3D`。
  - 使用 `scipy.optimize.least_squares` 和 Cauchy robust loss。
  - 显式构建 bundle adjustment sparsity pattern，避免密集有限差分导致运行过慢。
- `src/tools/run_bundle_adjustment.py`
  - 新增 Phase 7A 命令行工具。
  - 支持 `--max-points`、`--max-observations`、`--max-iterations`、`--loss`、`--f-scale`。
  - 自动备份 BA 前状态到 `*_before_ba.*`。
  - 写出 `bundle_adjustment_report.json`。

### 16.3 运行命令

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 800 \
  --max-observations 3000 \
  --min-track-length 2
```

### 16.4 验收结果

本轮 BA 从 Phase 6B + Phase 5C 后的 16-image reconstruction state 出发：

```text
Fixed image: P1180142.JPG
Registered images: 16
Optimized cameras: 15
Total points3D: 3930
Optimized points3D: 545
Total observations: 11517
Optimized observations: 3000
Loss: cauchy
f_scale: 4.0
Function evaluations: 40
Optimizer status: max function evaluations exceeded
```

误差变化：

```text
Median reprojection error: 1.859 -> 1.555 px
Mean reprojection error:   4.760 -> 4.540 px
Observations > 8 px:       180 -> 178
Observations > 16 px:      144 -> 144
```

输出文件：

```text
outputs/south_building_small/reports/bundle_adjustment_report.json
data/scenes/south_building_small/sparse/0/reconstruction_state.json
data/scenes/south_building_small/sparse/0/reconstruction_points.npz
data/scenes/south_building_small/sparse/0/registered_images.npz
```

BA 前备份：

```text
data/scenes/south_building_small/sparse/0/reconstruction_state_before_ba.json
data/scenes/south_building_small/sparse/0/reconstruction_points_before_ba.npz
data/scenes/south_building_small/sparse/0/registered_images_before_ba.npz
```

### 16.5 当前结论

Phase 7A 已经证明当前 reconstruction state 可以进入 bundle adjustment，并且在小规模稀疏 BA 下重投影误差有下降。由于本轮仍达到 `max_nfev` 上限，且离群观测数量基本未变，说明 BA 本身不能替代 outlier filtering。

下一步建议进入：

```text
Phase 7B BA-based filtering
```

优先处理：

- 按 BA 后 reprojection error 过滤高误差 observations / points。
- 按 track length 和 triangulation angle 过滤弱几何点。
- 对 `P1180157.JPG` 等低质量注册图像做质量复核。
- 过滤后再执行一轮局部 BA，比较误差和后续 Phase 5C 注册能力。

论文级别的完整 BA 可以作为 Phase 7C 实现，并在数学建模报告中与 Phase 7A 简化版形成对照：简化版用于解释核心优化思想，论文级版本用于展示更完整的鲁棒性和工程效果。

---

## 17. Phase 7B BA-based Filtering

### 17.1 执行动机

Phase 7A 的 BA 只优化连续变量：

```text
camera poses + 3D points
```

但错误匹配、错误 track 和局部低质量三角化属于离散 outlier，不能指望 BA 自动“优化正确”。因此 Phase 7B 新增基于 BA residual 的过滤步骤：

```text
BA
  -> measure observation residuals
  -> remove high-error observations
  -> remove weak / high-error points
  -> compact point3D ids
  -> run BA again
```

### 17.2 BA 报告指标修正

本阶段首先修正了 BA report 中优化目标和诊断指标混淆的问题。

新增/明确区分：

```text
initial_squared_residual_cost
final_squared_residual_cost
initial_robust_cost
final_robust_cost
scipy_final_robust_cost
```

其中：

- `robust_cost` 对应实际优化目标，例如 Cauchy loss。
- `squared_residual_cost` 只作为离群点诊断指标。
- `median / mean / p90 / p95 / max reprojection error` 用于直观评估重建质量。
- `observations_above_4px / 8px / 16px` 用于判断是否仍有明显 outlier。

同时 BA report 新增逐 observation 误差：

```text
optimized_observation_errors
```

用于 Phase 7B 复现过滤决策。

### 17.3 已完成内容

新增/修改代码：

- `src/sfm/bundle_adjustment.py`
  - 新增 `robust_residual_cost`。
  - 新增 `squared_residual_cost`。
  - 新增 `error_distribution`。
  - `BundleAdjustmentResult` 明确保存 squared cost 与 robust cost。
- `src/tools/run_bundle_adjustment.py`
  - 报告新增 robust/squared cost、p90/p95、逐 observation errors。
  - 新增 `--report-name`，避免多次 BA 覆盖同名报告。
  - BA state 备份改为唯一文件名，避免覆盖已有 baseline。
- `src/sfm/filtering.py`
  - 新增 BA residual filtering 核心逻辑。
  - 支持 observation-level filtering。
  - 支持 point-level filtering。
  - 支持 point3D id compact/remap。
- `src/tools/filter_reconstruction.py`
  - 新增 Phase 7B 命令行工具。
  - 支持 `--ba-report`、`--max-reprojection-error`、`--max-point-median-error`、`--max-point-max-error`、`--min-track-length`。
  - 新增 `--report-name`。
  - filtering state 备份同样改为唯一文件名。

### 17.4 运行命令

过滤前 BA：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 800 \
  --max-observations 3000 \
  --min-track-length 2 \
  --report-name bundle_adjustment_before_filtering_report.json
```

BA-based filtering：

```bash
D:\06_envs\mm26\python.exe -m src.tools.filter_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --ba-report outputs\south_building_small\reports\bundle_adjustment_before_filtering_report.json \
  --max-reprojection-error 8.0 \
  --max-point-median-error 8.0 \
  --max-point-max-error 32.0 \
  --min-track-length 2 \
  --report-name filtering_report.json
```

过滤后 BA：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 800 \
  --max-observations 3000 \
  --min-track-length 2 \
  --report-name bundle_adjustment_after_filtering_report.json
```

### 17.5 验收结果

首次有效执行 Phase 7B filtering 时，从 Phase 6B + Phase 5C 后的 16-image reconstruction 出发，过滤结果为：

```text
Observations: 11517 -> 11209
Points3D: 3930 -> 3890
Removed observations by reprojection: 178
Removed points by track length: 0
Removed points by error: 40
Removed points total: 40
Registered images: 16
```

过滤后同规模 BA 的质量：

```text
Points3D total: 3890
Observations total: 11209
Optimized points: 574
Optimized observations: 3000

Median reprojection error: 1.347 -> 1.332 px
Mean reprojection error:   1.716 -> 1.713 px
P95 reprojection error:    4.720 -> 4.695 px
Observations > 8 px:       0 -> 0
```

相比过滤前 7A 的均值误差约 `4.540 px`，过滤后 BA 子问题的均值误差降至约 `1.713 px`，说明高误差 observation / point filtering 有效清理了主要离群项。

### 17.6 当前注意事项

本阶段执行过程中发现原来的 BA/filtering 备份文件会覆盖同名 `*_before_ba.*` / `*_before_filtering.*` 文件，导致后续重复实验时 baseline 不够稳。该问题已修复：现在备份路径会自动使用唯一文件名。

当前 filtering 第一版只清理 BA 子问题覆盖到的 observation，也就是 `optimized_observation_errors` 中出现过的观测；未进入本轮 BA 的 observations 不会被误删。这是一个保守设计，适合作为 7B 第一版。后续若要更强，可以实现 full-reconstruction residual evaluation 后再过滤全部 observation。

下一步建议：

```text
Phase 7C Local / Global BA policy
```

或先进行：

```text
Phase 7B.2 full residual evaluation + second-pass filtering
```

如果目标是更贴近论文，应继续补充：

- 局部 BA 与全局 BA 触发策略。
- 内参 refinement。
- 更系统的 outlier filtering。
- 与 Phase 5C / 6B 的再次闭环验证。

---

## 18. Phase 7B.2 Registered-observation Residual Evaluation

### 18.1 执行动机

Phase 7B 第一版只过滤 BA 子问题覆盖到的 observation，也就是 `optimized_observation_errors` 中出现的部分观测。这个机制可以验证 BA-based filtering 是否可行，但不能说明当前所有已注册图像上的 observation 都已经干净。

因此本阶段不使用 `full` 命名，而是明确限定为：

```text
registered-observation residual evaluation
```

含义是：只在当前已经 registered 的图像和当前 reconstruction state 中已有的 observations 上计算 reprojection residual。它不是全数据集范围，也不是所有未注册图像范围。

### 18.2 已完成内容

新增/修改代码：

- `src/tools/evaluate_registered_residuals.py`
  - 遍历当前 reconstruction state 中所有 observations。
  - 只评估已经 registered 的图像。
  - 对每条 observation 计算 reprojection error。
  - 输出 `registered_residual_report.json` 风格报告。
- `src/tools/filter_reconstruction.py`
  - 支持读取 `observation_errors`。
  - 兼容旧的 `optimized_observation_errors`。

### 18.3 运行命令

过滤前 registered-observation residual evaluation：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name registered_residual_before_filtering_report.json
```

基于 registered residual 的过滤：

```bash
D:\06_envs\mm26\python.exe -m src.tools.filter_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --ba-report outputs\south_building_small\reports\registered_residual_before_filtering_report.json \
  --max-reprojection-error 8.0 \
  --max-point-median-error 8.0 \
  --max-point-max-error 32.0 \
  --min-track-length 2 \
  --report-name registered_residual_filtering_report.json
```

过滤后 registered-observation residual evaluation：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name registered_residual_after_filtering_report.json
```

过滤后 BA 验证：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 800 \
  --max-observations 3000 \
  --min-track-length 2 \
  --report-name bundle_adjustment_after_registered_residual_filtering_report.json
```

### 18.4 验收结果

注意：由于前一轮 Phase 7B 执行时旧备份文件曾被同名覆盖，当前 Phase 7B.2 是在已经初步过滤后的 16-image reconstruction state 上继续执行，而不是从最原始 7B 前状态重新开始。

过滤前 registered-observation residual：

```text
Registered images: 16
Points3D: 3890
Observations evaluated: 11209 / 11209

Mean reprojection error:   1.867 px
Median reprojection error: 1.239 px
P90 reprojection error:    4.047 px
P95 reprojection error:    5.259 px
Max reprojection error:    373.195 px
Observations > 4 px:       1149
Observations > 8 px:       74
Observations > 16 px:      27
```

Registered residual filtering：

```text
Observations: 11209 -> 11076
Points3D: 3890 -> 3848
Removed observations by reprojection: 74
Removed points by track length: 25
Removed points by error: 20
Removed points total: 42
```

过滤后 registered-observation residual：

```text
Points3D: 3848
Observations evaluated: 11076 / 11076

Mean reprojection error:   1.706 px
Median reprojection error: 1.228 px
P90 reprojection error:    3.906 px
P95 reprojection error:    5.034 px
Max reprojection error:    7.999 px
Observations > 4 px:       1049
Observations > 8 px:       0
Observations > 16 px:      0
```

过滤后 BA 验证：

```text
Optimized points: 576 / 3848
Optimized observations: 3000 / 11076

Median reprojection error: 1.330 -> 1.331 px
Mean reprojection error:   1.707 -> 1.702 px
P95 reprojection error:    4.688 -> 4.689 px
Observations > 8 px:       0 -> 0
```

### 18.5 当前结论

Phase 7B.2 比 7B 第一版更能说明 filtering 的效果，因为它不再只看 BA 子问题，而是评估了当前已注册图像上的全部 observations。

结论应表述为：

```text
Registered-observation filtering effectively removes remaining large residual outliers.
```

而不是：

```text
BA brings a large additional improvement after filtering.
```

过滤之后，registered observations 中 `>8px` 和 `>16px` 的大误差项都降为 0，最大误差从约 `373 px` 降到约 `8 px` 以下；但过滤后 BA 只带来很小的连续优化收益，说明当前主要收益来自离群观测清理，而不是 BA 对相机/点的进一步微调。

下一步更适合进入：

```text
Phase 7C Local / Global BA policy comparison
```

在进入 Phase 7D 内参优化前，仍建议先完成 7C，因为内参 refinement 应建立在相对干净的 observation graph 和明确的 BA 策略之上。

---

## 19. Phase 7C Local / Global BA Scheduling

### 19.1 执行动机

论文 4.4 的 BA 模块并不是只做一次全局 BA，而是明确区分：

```text
local BA:
  after each image registration
  optimize the set of most-connected images

global BA:
  after the model grows by a certain percentage
```

因此在实现 pre-BA RT / post-BA RT 迭代之前，必须先补上 local/global BA 的调度骨架。否则后续无法正确表达：

```text
pre-BA RT -> global BA -> filtering -> post-BA RT -> BA -> filtering
```

### 19.2 已完成内容

新增/修改代码：

- `src/sfm/bundle_adjustment.py`
  - `BundleAdjustmentProblem` 新增 `scope` 和 `local_image_names`。
  - `build_bundle_adjustment_problem` 支持 `scope = global | local`。
  - 新增 `build_covisibility_graph`，按 shared point3D count 统计 registered image 共视边。
  - 新增 `select_local_ba_images`，从 target image 选择 top-K most-connected registered neighbors。
  - local BA 中只优化 local image poses 和相关 points；非 local registered cameras 固定，但其 observations 可作为约束参与 residual。
- `src/tools/run_bundle_adjustment.py`
  - 新增 `--scope global|local`。
  - 新增 `--target-image`。
  - 新增 `--local-neighbors`。
  - BA report 记录 `ba_scope`、`target_image_name`、`local_image_names`、`optimized_image_names`、`num_covisibility_edges`。

### 19.3 运行命令

Phase 7C baseline registered residual：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name phase7c_registered_residual_baseline_report.json
```

Local BA，对 `P1180157.JPG` 及其共视邻居执行：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --scope local \
  --target-image P1180157.JPG \
  --local-neighbors 6 \
  --max-iterations 30 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 500 \
  --max-observations 2500 \
  --min-track-length 2 \
  --report-name phase7c_local_ba_p1180157_report.json
```

Local BA 后 registered residual：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name phase7c_registered_residual_after_local_ba_report.json
```

Global BA：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --scope global \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 900 \
  --max-observations 4000 \
  --min-track-length 2 \
  --report-name phase7c_global_ba_report.json
```

Global BA 后 registered residual：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name phase7c_registered_residual_after_global_ba_report.json
```

### 19.4 验收结果

本阶段从 Phase 7B.2 后的 16-image、已过滤 reconstruction state 出发。

Baseline registered residual：

```text
Observations evaluated: 11076 / 11076
Mean reprojection error:   1.709 px
Median reprojection error: 1.235 px
P90 reprojection error:    3.884 px
P95 reprojection error:    5.053 px
Max reprojection error:    8.164 px
Observations > 4 px:       1025
Observations > 8 px:       8
Observations > 16 px:      0
```

Local BA 选择结果：

```text
Target image: P1180157.JPG
Local images:
  P1180150.JPG
  P1180151.JPG
  P1180154.JPG
  P1180156.JPG
  P1180157.JPG
Fixed image: P1180150.JPG
Optimized cameras: 4
Optimized points: 431
Optimized observations: 2500
```

Local BA 子问题误差：

```text
Median reprojection error: 1.082 -> 1.071 px
Mean reprojection error:   1.547 -> 1.538 px
P95 reprojection error:    4.745 -> 4.722 px
Observations > 8 px:       1 -> 1
```

Local BA 后 registered residual：

```text
Mean reprojection error:   1.699 px
Median reprojection error: 1.231 px
P95 reprojection error:    4.862 px
Observations > 8 px:       6
```

Global BA 子问题：

```text
Fixed image: P1180142.JPG
Optimized cameras: 15
Optimized points: 807
Optimized observations: 4000
```

Global BA 子问题误差：

```text
Median reprojection error: 1.249 -> 1.235 px
Mean reprojection error:   1.633 -> 1.613 px
P95 reprojection error:    4.390 -> 4.390 px
Observations > 8 px:       1 -> 1
```

Global BA 后 registered residual：

```text
Mean reprojection error:   1.638 px
Median reprojection error: 1.170 px
P95 reprojection error:    4.837 px
Observations > 8 px:       2
```

### 19.5 当前结论

Phase 7C 已补上论文 BA 模块中的 local/global BA 区分：

- Local BA 能对目标图像邻域带来小幅改善，运行规模更小，符合“每次注册后局部优化”的定位。
- Global BA 覆盖更多相机和观测，对 registered residual 的整体改善更明显，但运行代价更高，符合“模型增长到一定比例后触发”的定位。

当前仍未实现论文 4.4 中的完整迭代：

```text
pre-BA RT
global BA
filtering
post-BA RT
BA
filtering
repeat until filtered observations and post-BA RT points diminish
```

下一步应进入：

```text
Phase 7D Paper-style BA / RT / Filtering Iterative Refinement
```

其中重点是把已有的 registered-track triangulation 改造成：

- pre-BA RT：global BA 前重新三角化以补偿 drift。
- post-BA RT：BA 改善 pose/points 后，继续之前失败的 tracks；只使用 residual 低于 filtering threshold 的 observations，并尝试 merge tracks。

---

## 20. Phase 7D Paper-style BA / RT / Filtering Iterative Refinement

### 20.1 执行动机

论文 4.4 中的 BA 模块不仅包含 local/global BA，还包含围绕 global BA 的 re-triangulation 和 filtering 迭代：

```text
pre-BA RT
  -> global BA
  -> filtering
  -> post-BA RT
  -> BA
  -> filtering
  -> repeat until filtered observations and post-BA RT points diminish
```

Phase 7C 已补上 local/global BA 的 scope。本阶段开始实现论文式 BA/RT/filtering 第一轮闭环。

### 20.2 已完成内容

新增/修改代码：

- `src/tools/triangulate_registered_tracks.py`
  - 新增 `--stage registered_rt|pre_ba_rt|post_ba_rt`。
  - 新增 `--residual-report`。
  - 新增 `--max-observation-error`。
  - 新增 `--report-name`。
  - post-BA RT 支持读取 residual report，对高 residual observation 做 gating。
  - report 新增 `rt_stage`、`residual_report_path`、`blocked_observations_by_residual`、`skipped_matches_by_residual`、`track_merge_implemented` 等字段。

当前边界：

```text
track merge: not implemented yet
automatic iterative controller: not implemented yet
intrinsic refinement: not implemented yet
```

因此本阶段是论文 BA/RT/filtering 主流程的第一版对齐，不是完整 COLMAP/Ceres 级实现。

### 20.3 运行命令

Baseline registered residual：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name phase7d_registered_residual_baseline_report.json
```

pre-BA RT：

```bash
D:\06_envs\mm26\python.exe -m src.tools.triangulate_registered_tracks \
  --scene configs/scenes/south_building_small.yaml \
  --stage pre_ba_rt \
  --report-name phase7d_pre_ba_rt_report.json
```

Global BA after pre-BA RT：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --scope global \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 1200 \
  --max-observations 6000 \
  --min-track-length 2 \
  --report-name phase7d_global_ba_after_pre_rt_report.json
```

Filtering after global BA：

```bash
D:\06_envs\mm26\python.exe -m src.tools.filter_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --ba-report outputs\south_building_small\reports\phase7d_registered_residual_after_global_ba_report.json \
  --max-reprojection-error 8.0 \
  --max-point-median-error 8.0 \
  --max-point-max-error 32.0 \
  --min-track-length 2 \
  --report-name phase7d_filtering_after_global_ba_report.json
```

post-BA RT：

```bash
D:\06_envs\mm26\python.exe -m src.tools.triangulate_registered_tracks \
  --scene configs/scenes/south_building_small.yaml \
  --stage post_ba_rt \
  --residual-report outputs\south_building_small\reports\phase7d_registered_residual_after_filtering_report.json \
  --max-observation-error 8.0 \
  --report-name phase7d_post_ba_rt_report.json
```

Final BA/filtering/residual：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --scope global \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 1200 \
  --max-observations 6000 \
  --min-track-length 2 \
  --report-name phase7d_final_global_ba_report.json

D:\06_envs\mm26\python.exe -m src.tools.filter_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --ba-report outputs\south_building_small\reports\phase7d_registered_residual_after_post_rt_ba_report.json \
  --max-reprojection-error 8.0 \
  --max-point-median-error 8.0 \
  --max-point-max-error 32.0 \
  --min-track-length 2 \
  --report-name phase7d_final_filtering_report.json

D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name phase7d_final_registered_residual_report.json
```

### 20.4 验收结果

本阶段从 Phase 7C 后的 16-image reconstruction state 出发。

Baseline：

```text
Points3D: 3848
Observations: 11076
Mean registered residual:   1.638 px
Median registered residual: 1.170 px
P95 registered residual:    4.837 px
Observations > 8 px:        2
```

pre-BA RT：

```text
Points3D: 3848 -> 4486
Observations: 11076 -> 13621
New points3D: 638
New observations: 2545
Augmented existing observations: 1173
New point reprojection error median/mean: 2.472 / 2.825 px
```

pre-BA RT 后执行 global BA，registered residual 显示新增观测中包含大量 outlier：

```text
Observations evaluated: 13621
Mean registered residual:   4.532 px
Median registered residual: 1.411 px
P95 registered residual:    14.345 px
Observations > 8 px:        957
```

这与论文 4.4 的动机一致：pre-BA RT 可以增加完整度，但会把相当一部分 outlier 带入 BA，必须配合 filtering 和后续迭代。

Filtering after global BA：

```text
Observations: 13621 -> 12089
Points3D: 4486 -> 4224
Removed observations by reprojection: 957
Removed points total: 262
```

Filtering 后 registered residual：

```text
Mean registered residual:   1.727 px
Median registered residual: 1.252 px
P95 registered residual:    5.038 px
Observations > 8 px:        0
```

post-BA RT：

```text
Points3D: 4224 -> 4503
Observations: 12089 -> 13502
New points3D: 279
New observations: 1413
Augmented existing observations: 714
New point reprojection error median/mean: 3.829 / 3.972 px
```

由于 post-BA RT 使用的 residual report 来自 filtering 后状态，而此时所有 registered observations 都低于 `8 px`，因此：

```text
blocked_observations_by_residual: 0
skipped_matches_by_residual: 0
```

这说明 residual gating 生效条件存在，但当前过滤后状态没有超阈值 observation 需要屏蔽。

Final BA/filtering：

```text
After post-BA RT + BA:
Observations evaluated: 13502
Mean registered residual:   3.985 px
Median registered residual: 1.384 px
P95 registered residual:    8.851 px
Observations > 8 px:        729

Final filtering:
Observations: 13502 -> 12434
Points3D: 4503 -> 4357
Removed observations by reprojection: 729
Removed points total: 146

Final registered residual:
Mean registered residual:   1.776 px
Median registered residual: 1.271 px
P90 registered residual:    4.076 px
P95 registered residual:    5.403 px
Max registered residual:    7.993 px
Observations > 8 px:        0
```

### 20.5 当前结论

Phase 7D 第一版已经跑通论文 4.4 的主流程骨架：

```text
pre-BA RT
global BA
filtering
post-BA RT
global BA
filtering
```

结果符合论文思想：

- pre-BA RT 和 post-BA RT 都提高了 completeness。
- RT 会引入 outlier observations，需要 BA/filtering 清理。
- post-BA RT 的新增点数 `279` 小于 pre-BA RT 的 `638`，有“新增点数减少”的趋势。
- final filtering 删除 observation 数 `729` 小于第一次 filtering 的 `957`，有“filtered observations diminish”的趋势。

但当前还没有达到完整论文级实现：

- 只执行了一轮手动迭代，没有自动循环到收敛。
- track merge 尚未实现。
- post-BA RT 的新点误差偏高，说明仍需更严格的 track continuation 或 triangulation acceptance。
- 当前 BA 仍是 capped sparse SciPy BA，不是 Ceres/Schur 级全量 BA。
- 内参 refinement 和 degenerate camera filtering 尚未实现。

下一步建议：

```text
Phase 7E Automatic BA/RT/Filtering controller
```

或先进行更保守的：

```text
Phase 7D.2 stricter post-BA RT acceptance
```

建议优先做 7D.2，因为 post-BA RT 新点 median/mean reprojection error `3.829 / 3.972 px` 明显高于 pre-BA RT，先改进 RT 接受策略会让后续自动迭代更稳定。

---

## 21. Phase 7E Automatic BA/RT/Filtering Controller

### 21.1 执行动机

Phase 7D 手动跑通了一轮论文式流程：

```text
pre-BA RT
global BA
filtering
post-BA RT
global BA
filtering
```

但论文 4.4 强调的是迭代执行，直到：

```text
filtered observations diminish
post-BA RT points diminish
```

因此本阶段新增自动控制器，用于判断当前未收敛到底是因为迭代次数不足，还是因为 RT / track continuation 本身还不够稳定。

### 21.2 已完成内容

新增代码：

- `src/tools/run_ba_rt_refinement.py`
  - 自动执行 registered residual evaluation。
  - 自动执行 pre-BA RT。
  - 自动执行 global BA。
  - 自动执行 filtering。
  - 自动执行 post-BA RT。
  - 自动执行 final BA/filtering。
  - 每轮写出独立 report。
  - 汇总输出 `ba_rt_refinement_report.json`。

停止条件：

```text
post_ba_rt_new_points < min_new_points
and
final_filtering_removed_observations < min_filtered_observations
```

本轮实验使用：

```text
max_refinement_iterations = 2
min_new_points = 50
min_filtered_observations = 50
```

### 21.3 运行命令

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_ba_rt_refinement \
  --scene configs/scenes/south_building_small.yaml \
  --max-refinement-iterations 2 \
  --min-new-points 50 \
  --min-filtered-observations 50 \
  --max-observation-error 8.0 \
  --max-point-median-error 8.0 \
  --max-point-max-error 32.0 \
  --ba-max-iterations 40 \
  --ba-max-points 1200 \
  --ba-max-observations 6000 \
  --loss cauchy \
  --f-scale 4.0 \
  --min-track-length 2 \
  --report-name phase7e_ba_rt_refinement_report.json
```

### 21.4 验收结果

自动控制器从 Phase 7D final state 出发：

```text
Baseline points3D: 4357
Baseline observations: 12434
Baseline mean residual: 1.776 px
Baseline median residual: 1.271 px
Baseline P95 residual: 5.403 px
Baseline observations > 8 px: 0
```

第 1 轮：

```text
pre-BA RT new points: 145
post-BA RT new points: 246
filtering after pre-BA removed observations: 985
final filtering removed observations: 729
final points3D: 4366
final observations: 12450
final mean residual: 1.784 px
final observations > 8 px: 0
```

第 2 轮：

```text
pre-BA RT new points: 136
post-BA RT new points: 225
filtering after pre-BA removed observations: 969
final filtering removed observations: 739
final points3D: 4375
final observations: 12469
final mean residual: 1.798 px
final observations > 8 px: 0
```

收敛状态：

```text
converged: false
stop_reason: max_refinement_iterations_reached
```

### 21.5 当前结论

Phase 7E 证明了自动迭代控制器可以稳定调度 BA/RT/filtering，但当前模型没有在 2 轮内达到论文式停止条件。

关键观察：

- `post_ba_rt_new_points` 从 `246` 降到 `225`，有下降趋势，但仍远高于 `50`。
- `final_filtering_removed_observations` 从 `729` 到 `739`，没有下降，说明 RT 仍反复引入大量需要过滤的 observations。
- 最终 `>8px` observations 能被 filtering 清回 0，但 mean residual 从 `1.784` 到 `1.798` 略升，说明单纯增加迭代次数并不足以改善质量。

因此，当前不应优先做 intrinsic refinement。更合理的下一步是：

```text
Phase 7F Track Merge / stricter RT continuation
```

原因：

- 论文中的 post-BA RT 明确提到继续 tracks 并尝试 merge tracks，以增加下一轮 BA 的 redundancy。
- 当前 `track_merge_implemented = false`。
- 当前 RT 每轮仍引入大量需要过滤的 observations，说明需要更严格的 track continuation / merge / acceptance 逻辑，而不是先释放内参自由度。

下一步建议：

```text
Phase 7F Track Merge and RT Acceptance Refinement
```

优先实现保守 track merge：

- 只合并没有同图冲突的 points/tracks。
- 合并前后 reprojection error 必须低于阈值。
- 合并后 triangulation angle 必须合格。
- report 中记录 accepted/rejected merge 数量和原因。

随后再重新运行 Phase 7E 自动控制器，观察：

```text
post_ba_rt_new_points 是否下降
final_filtering_removed_observations 是否下降
final residual 是否稳定
```

---

## 22. Phase 7F Conservative Track Merge

### 22.1 执行动机

论文 4.4 在 post-BA RT 中明确提到：

```text
attempt to merge tracks
thereby provide increased redundancy for the next BA step
```

Phase 7E 自动迭代显示，单纯增加 BA/RT/filtering 迭代次数不能让系统收敛：

```text
post-BA RT new points 仍为 225
final filtering removed observations 仍约 739
```

因此本阶段先实现保守 track merge，处理同一个 feature track 已经连接到多个 `point3D_id` 的冲突情况。

### 22.2 已完成内容

修改代码：

- `src/tools/triangulate_registered_tracks.py`
  - 新增 `--enable-track-merge`。
  - 对 `linked_point_ids > 1` 的 conflicting track 尝试合并。
  - 合并前检查：
    - 合并后每张图像最多一个 observation。
    - 合并后 track length 满足阈值。
    - 合并后存在合法 Phase 3B verified view pair。
    - 合并后可通过 robust triangulation 几何验证。
  - 合并成功后：
    - 保留最小 `point3D_id`。
    - 用重新三角化的点更新该 point3D。
    - 将其他 point 的 observations 迁移到保留点。
    - 更新 observation lookup。
  - report 新增：
    - `track_merge_enabled`
    - `merged_tracks`
    - `rejected_merge_same_image_conflict`
    - `rejected_merge_short`
    - `rejected_merge_no_valid_pair`
    - `rejected_merge_geometry`

当前实现是保守 merge，不做大范围 track graph 重写。

### 22.3 运行命令

Baseline residual：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name phase7f_registered_residual_baseline_report.json
```

post-BA RT with track merge：

```bash
D:\06_envs\mm26\python.exe -m src.tools.triangulate_registered_tracks \
  --scene configs/scenes/south_building_small.yaml \
  --stage post_ba_rt \
  --residual-report outputs\south_building_small\reports\phase7f_registered_residual_baseline_report.json \
  --max-observation-error 8.0 \
  --enable-track-merge \
  --report-name phase7f_post_ba_rt_with_merge_report.json
```

Merge 后 BA/filtering 验证：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --scope global \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 1200 \
  --max-observations 6000 \
  --min-track-length 2 \
  --report-name phase7f_global_ba_after_merge_report.json

D:\06_envs\mm26\python.exe -m src.tools.filter_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --ba-report outputs\south_building_small\reports\phase7f_registered_residual_after_merge_ba_report.json \
  --max-reprojection-error 8.0 \
  --max-point-median-error 8.0 \
  --max-point-max-error 32.0 \
  --min-track-length 2 \
  --report-name phase7f_filtering_after_merge_ba_report.json
```

### 22.4 验收结果

Baseline：

```text
Points3D: 4375
Observations: 12469
Mean residual:   1.798 px
Median residual: 1.269 px
P95 residual:    5.635 px
Observations > 8 px: 0
```

post-BA RT with merge：

```text
Points3D: 4375 -> 4506
Observations: 12469 -> 13737
New points3D: 131
New observations: 1268
Augmented existing observations: 940
Merged tracks: 7
Skipped conflicting point tracks: 0
Rejected merge same-image conflict: 0
Rejected merge short: 0
Rejected merge no valid pair: 0
Rejected merge geometry: 0
New point reprojection error median/mean: 2.365 / 2.858 px
```

Merge 后 BA/filtering：

```text
After BA:
Mean registered residual:   4.587 px
Median registered residual: 1.418 px
P95 registered residual:    15.602 px
Observations > 8 px:        996

Filtering:
Observations: 13737 -> 12178
Points3D: 4506 -> 4265
Removed observations by reprojection: 996
Removed points total: 241

Final registered residual:
Mean residual:   1.772 px
Median residual: 1.268 px
P95 residual:    5.448 px
Observations > 8 px: 0
```

### 22.5 当前结论

Track merge 第一版有效解决了 conflicting point tracks：

```text
merged_tracks = 7
skipped_conflicting_point_tracks = 0
```

同时 post-BA RT 新增点数从 Phase 7E 第 2 轮的 `225` 降到 `131`，说明 merge 和当前状态确实减少了一部分重复/冲突点生成。

但 merge 并没有解决主要收敛瓶颈：

```text
filtering removed observations = 996
```

这说明当前 BA/RT/filtering 循环中的主要压力不是 conflicting tracks，而是 RT 仍会添加大量后续被判为高 residual 的 observations。

下一步不建议立刻做 intrinsic refinement。更合理的是：

```text
Phase 7G Stricter RT Continuation / Acceptance
```

候选改进：

- post-BA RT 使用更严格的 `max_reproj_error_px`，例如 4px 而不是 8px。
- 对 augmented existing observations 也做即时 reprojection check，而不是只依赖后续 filtering。
- 新点接受时增加 median/max reprojection error 双阈值。
- 对 post-BA RT 单独设置更高的 min track length 或 min triangulation angle。
- 将 RT 新增 observations 的质量统计加入自动控制器停止条件。

---

## 23. Phase 7G RT Acceptance Policy

### 23.1 执行动机

Phase 7F 表明 track merge 能减少 conflicting tracks，但 RT 后仍会产生大量后续被 filtering 删除的 observations：

```text
filtering removed observations = 996
```

这说明问题不只是 track conflict，而是 post-BA RT 的接受策略仍偏宽松。BA/filtering 能清理 outlier，但如果 RT 每轮持续加入大量低质量 observations，迭代控制器就难以满足论文中的 diminish 停止条件。

本阶段不直接把阈值改到过严的 `4px`，而是先实现 policy 化：

```text
current
post_ba_moderate
post_ba_strict
```

并先测试 `post_ba_moderate`。

### 23.2 已完成内容

修改代码：

- `src/tools/triangulate_registered_tracks.py`
  - 新增 `--rt-policy current|post_ba_moderate|post_ba_strict`。
  - 新增 `--max-new-point-median-error`。
  - 新增 `--max-new-point-max-error`。
  - 新点接受时增加 median/max reprojection error policy。
  - track merge 时同样检查新点 error policy。
  - report 新增：
    - `rt_policy`
    - `max_new_point_median_error_px`
    - `max_new_point_max_error_px`
    - `skipped_new_point_error_policy`
    - `rejected_merge_error_policy`

当前 policy：

```text
current:
  max_reproj_error_px = config default, currently 8.0
  min_track_length = config/default CLI
  min_triangulation_angle_deg = config default

post_ba_moderate:
  max_reproj_error_px = 6.0
  max_new_point_median_error_px = 3.0
  max_new_point_max_error_px = 6.0
  min_track_length >= 3
  min_triangulation_angle_deg = 2.0

post_ba_strict:
  max_reproj_error_px = 4.0
  max_new_point_median_error_px = 2.5
  max_new_point_max_error_px = 4.0
  min_track_length >= 3
  min_triangulation_angle_deg = 2.0
```

### 23.3 运行命令

Baseline residual：

```bash
D:\06_envs\mm26\python.exe -m src.tools.evaluate_registered_residuals \
  --scene configs/scenes/south_building_small.yaml \
  --report-name phase7g_registered_residual_baseline_report.json
```

post-BA RT moderate：

```bash
D:\06_envs\mm26\python.exe -m src.tools.triangulate_registered_tracks \
  --scene configs/scenes/south_building_small.yaml \
  --stage post_ba_rt \
  --rt-policy post_ba_moderate \
  --residual-report outputs\south_building_small\reports\phase7g_registered_residual_baseline_report.json \
  --max-observation-error 8.0 \
  --enable-track-merge \
  --report-name phase7g_post_ba_rt_moderate_report.json
```

moderate RT 后 BA/filtering：

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_bundle_adjustment \
  --scene configs/scenes/south_building_small.yaml \
  --scope global \
  --max-iterations 40 \
  --loss cauchy \
  --f-scale 4.0 \
  --max-points 1200 \
  --max-observations 6000 \
  --min-track-length 2 \
  --report-name phase7g_global_ba_after_moderate_rt_report.json

D:\06_envs\mm26\python.exe -m src.tools.filter_reconstruction \
  --scene configs/scenes/south_building_small.yaml \
  --ba-report outputs\south_building_small\reports\phase7g_registered_residual_after_moderate_rt_ba_report.json \
  --max-reprojection-error 8.0 \
  --max-point-median-error 8.0 \
  --max-point-max-error 32.0 \
  --min-track-length 2 \
  --report-name phase7g_filtering_after_moderate_rt_ba_report.json
```

### 23.4 验收结果

Baseline：

```text
Points3D: 4265
Observations: 12178
Mean residual:   1.772 px
Median residual: 1.268 px
P95 residual:    5.448 px
Observations > 8 px: 0
```

post-BA RT moderate：

```text
Policy: post_ba_moderate
max_reproj_error_px: 6.0
max_new_point_median_error_px: 3.0
max_new_point_max_error_px: 6.0
min_track_length: 3
min_triangulation_angle_deg: 2.0

Points3D: 4265 -> 4326
Observations: 12178 -> 13156
New points3D: 61
New observations: 978
Augmented existing observations: 742
Merged tracks: 0
Skipped new point error policy: 10
New point reprojection error median/mean: 1.798 / 1.828 px
```

对比 Phase 7F 的 post-BA RT with merge：

```text
Phase 7F current policy:
  New points3D: 131
  New observations: 1268
  New point error median/mean: 2.365 / 2.858 px
  Filtering removed observations: 996

Phase 7G moderate policy:
  New points3D: 61
  New observations: 978
  New point error median/mean: 1.798 / 1.828 px
  Filtering removed observations: 744
```

moderate RT 后 BA/filtering：

```text
After BA:
Mean registered residual:   4.550 px
Median registered residual: 1.348 px
P95 registered residual:    9.994 px
Observations > 8 px:        744

Filtering:
Observations: 13156 -> 12092
Points3D: 4326 -> 4197
Removed observations by reprojection: 744
Removed points total: 129

Final registered residual:
Mean residual:   1.734 px
Median residual: 1.242 px
P95 residual:    5.354 px
Observations > 8 px: 0
```

### 23.5 当前结论

`post_ba_moderate` policy 有效降低了 RT 新点数量和过滤压力：

- 新点误差明显下降。
- `filtering removed observations` 从 Phase 7F 的 `996` 降到 `744`。
- 最终 residual 保持干净，`>8px = 0`。

代价是 completeness 增长变慢：

```text
New points3D: 131 -> 61
Final points3D: 4265 -> 4197
```

因此当前结论是：

```text
6px moderate policy 比直接 4px strict 更稳妥；
它能降低 outlier 压力，但仍未让 filtering removed observations 降到收敛阈值。
```

下一步建议：

```text
Phase 7H Integrate RT policy into automatic controller
```

在同一自动迭代框架下比较：

```text
current policy
post_ba_moderate policy
post_ba_strict policy
```

只有在同一 controller / 同一轮数下比较，才能判断 6px 是否真正优于 8px 或 4px。
