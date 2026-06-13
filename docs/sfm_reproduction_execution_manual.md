# Incremental SfM Reproduction Execution Manual

面向任务：复现论文 *Structure-from-Motion Revisited* 的 Incremental SfM pipeline，并保证输出能低成本接入 VGGT 与 3D Gaussian Splatting。

本文不是论文摘要，而是工程执行手册：每个阶段都回答为什么做、做什么、怎么做、自己实现还是调用库、如何验收、如何提交 Git commit。

---

## 1. 数据集选择与下载规划

本项目的数据集要同时服务两个目标：一是验证 Incremental SfM 复现是否稳定，二是保证导出的 sparse model 能进入后续 3DGS。因此不建议一开始使用随手拍摄、纹理稀疏、曝光变化大的数据；第一阶段应使用公开、经典、图像数量适中、COLMAP 可重建的数据。

### 1.1 推荐主数据集

| 优先级 | 数据集 | 用途 | 原因 | 建议规模 |
| --- | --- | --- | --- | --- |
| 第一优先级 | COLMAP South Building | SfM 主复现、COLMAP/3DGS 接口验收 | 128 张图像，室外建筑，视差充分，规模适合课程实现 | 先用 30-50 张调通，再跑全量 |
| 第二优先级 | COLMAP Gerrard Hall | 复现实验对照 | 100 张高分辨率图像，和 South Building 风格接近，可验证泛化 | 先抽样 30 张 |
| 第三优先级 | 自采小场景 | 课程展示补充 | 能展示从数据采集到 3DGS 的完整闭环 | 40-80 张，保证环绕拍摄和足够纹理 |
| 后续接口验证 | 3DGS 官方示例场景 | 验证 3DGS 训练环境 | 官方 3DGS 页面提供示例 scenes；用于确认 3DGS 代码本身可运行，不作为 SfM 手写主数据 | 只下载一个小场景 |

推荐顺序：

```text
South Building 子集
  ↓
South Building 全量
  ↓
Gerrard Hall 对照
  ↓
自采小场景
  ↓
3DGS 官方示例场景接口验证
```

### 1.2 数据来源

- COLMAP 官方数据集页面列出了 South Building、Gerrard Hall、Person Hall、Graham Hall 等数据集。
- COLMAP 数据下载入口：`https://demuc.de/colmap/datasets/`
- COLMAP 文档数据集说明：`https://colmap.github.io/datasets.html`
- 3D Gaussian Splatting 官方页面提供 scenes 下载入口：`https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/`

### 1.3 Phase 0 下载任务

准备阶段必须完成以下下载和整理任务：

```text
data/
  raw/
    south-building/
    gerrard-hall/
  scenes/
    south_building/
      images/
    south_building_small/
      images/
    gerrard_hall/
      images/
  external/
    3dgs_official/
```

下载后处理规范：

1. 原始压缩包和原始解压结果放入 `data/raw/`，不直接在 raw 目录上跑实验。
2. 实验使用的数据复制或软链接到 `data/scenes/<scene_name>/images/`。
3. `south_building_small` 作为首个调试数据集，只保留 30-50 张连续且有重叠的图像。
4. 图像文件名保持原始名称，不要重命名为 `1.jpg, 2.jpg`，避免后续 COLMAP/3DGS 索引错乱。
5. 若图像过大，另建 `images_resized/`，不要覆盖原图。
6. 每个 scene 必须有一个配置文件，例如 `configs/scenes/south_building_small.yaml`。

### 1.4 自采数据集规范

自采数据只作为第二阶段展示，不作为第一阶段算法调试起点。采集要求：

- 使用同一手机或相机，固定焦距，不要频繁变焦。
- 围绕目标物体或建筑缓慢移动，保证相邻图像 60%-80% 重叠。
- 避免大面积玻璃、纯白墙、反光金属、水面、动态行人。
- 每张图都要有足够纹理和稳定视差。
- 不要只原地旋转拍全景；纯旋转可以匹配，但无法稳定三角化。
- 保留原始 EXIF，方便估计焦距初值。

---

## 2. 论文执行逻辑重排

不要按论文章节实现，而应按下面的执行链组织代码。

```text
论文理论
  Incremental SfM 是“数据质量”和“模型质量”互相促进的闭环
  Scene graph 提供重叠关系，BA/过滤/重三角化持续修正模型

数学模型
  特征描述子距离 -> 2D-2D 匹配
  对极几何 F/E/H -> 图像对验证与场景类型判断
  Track graph -> 多视图观测集合
  Two-view initialization -> 初始相对位姿与初始点云
  PnP/RANSAC -> 新图像 world-to-camera pose
  DLT triangulation/RANSAC -> 新 3D 点
  BA -> 最小化鲁棒重投影误差

代码模块
  sfm/features.py
  sfm/matching.py
  sfm/geometry.py
  sfm/tracks.py
  sfm/initialization.py
  sfm/reconstruction.py
  sfm/triangulation.py
  sfm/bundle_adjustment.py
  sfm/export/colmap.py

工程接口
  input: data/scenes/<scene>/images/*
  cache: features, matches, verified_pairs, tracks, reconstruction_state
  output: COLMAP sparse model + images folder + optional PLY/JSON reports

最终输出
  data/scenes/<scene>/sparse/0/cameras.txt
  data/scenes/<scene>/sparse/0/images.txt
  data/scenes/<scene>/sparse/0/points3D.txt
  data/scenes/<scene>/images/*
  output/<scene>/sparse_points.ply
```

核心设计原则：

1. 内部可以自定义 Python 数据结构，但最终必须导出 COLMAP sparse model。
2. 位姿统一保存为 COLMAP 约定：`x_cam = R * x_world + t`。
3. `images.txt` 中的 `qvec/tvec` 是 world-to-camera，不是 camera-to-world。
4. 相机中心仅作为派生量：`C = -R^T t`。
5. 点云坐标、相机位姿、track 中的 image/point id 不允许在导出阶段重新打乱。

---

## 3. SfM 复现执行手册

### Phase 0 环境准备

**功能目标**

建立可复现实验环境、数据集目录、配置文件、日志目录与可视化输出目录。Phase 0 不进入算法实现，但必须把后续每个阶段的输入输出位置固定下来，避免中途出现路径混乱、图像重命名、COLMAP/3DGS 不兼容等问题。

**输入**

- Conda 环境：`mm26`
- 原始公开数据集压缩包或下载链接
- 图像数据：`data/scenes/<scene>/images`
- 相机先验：EXIF 或手动配置的焦距/传感器尺寸

**输出**

- `requirements.txt`
- `configs/default.yaml`
- `configs/scenes/<scene>.yaml`
- `data/raw/<dataset_name>/`
- `data/scenes/<scene>/images/`
- `outputs/<scene>/logs/`
- `outputs/<scene>/figures/`
- `outputs/<scene>/reports/`

**准备阶段要建立的目录框架**

Phase 0 应先建立下面的目录框架。空目录可以放 `.gitkeep`，真实数据目录应被 `.gitignore` 忽略。

```text
final/
  configs/
    default.yaml
    scenes/
      south_building_small.yaml
      south_building.yaml
      gerrard_hall.yaml

  data/
    raw/
      south-building/
      gerrard-hall/
    scenes/
      south_building_small/
        images/
        features/
        matches/
        verified/
        tracks/
        sparse/
          0/
        undistorted/
      south_building/
        images/
        features/
        matches/
        verified/
        tracks/
        sparse/
          0/
        undistorted/
      gerrard_hall/
        images/
        features/
        matches/
        verified/
        tracks/
        sparse/
          0/
        undistorted/
    external/
      3dgs_official/

  outputs/
    south_building_small/
      logs/
      figures/
      reports/
    south_building/
      logs/
      figures/
      reports/
    gerrard_hall/
      logs/
      figures/
      reports/

  third_party/
    gaussian-splatting/
    vggt/
```

目录职责：

| 目录 | 职责 | 是否提交 Git |
| --- | --- | --- |
| `configs/` | 实验配置、阈值、路径、相机模型 | 提交 |
| `data/raw/` | 原始下载数据和解压结果 | 不提交 |
| `data/scenes/<scene>/images/` | 当前实验实际使用图像 | 不提交 |
| `features/` | keypoints/descriptors 缓存 | 不提交 |
| `matches/` | raw matches 缓存 | 不提交 |
| `verified/` | 几何验证后的 inlier matches 和 scene graph | 不提交 |
| `tracks/` | feature track graph | 不提交 |
| `sparse/0/` | COLMAP sparse model 导出目录 | 可提交小样例，不提交大结果 |
| `undistorted/` | 给 3DGS 使用的去畸变图像 | 不提交 |
| `outputs/<scene>/logs/` | 运行日志 | 不提交 |
| `outputs/<scene>/figures/` | 可视化结果 | 可选择提交课程报告图 |
| `outputs/<scene>/reports/` | JSON/Markdown 实验报告 | 可提交 |
| `third_party/` | VGGT、3DGS 等外部项目 | 建议用 submodule 或不提交 |

**数据集下载任务**

Phase 0 必须完成以下任务：

1. 下载 COLMAP South Building 数据集到 `data/raw/south-building/`。
2. 从 South Building 中整理一个 30-50 张图像的调试子集到 `data/scenes/south_building_small/images/`。
3. 将 South Building 全量图像整理到 `data/scenes/south_building/images/`。
4. 下载 Gerrard Hall 作为第二个验证场景，整理到 `data/scenes/gerrard_hall/images/`。
5. 可选下载 3DGS 官方示例场景到 `data/external/3dgs_official/`，只用于确认 3DGS 训练环境，不作为手写 SfM 主数据。

推荐下载命令模板：

```bash
conda activate mm26

# 手动从 COLMAP 数据集页面下载压缩包后解压到 data/raw/
# https://demuc.de/colmap/datasets/
```

Windows PowerShell 可使用：

```powershell
New-Item -ItemType Directory -Force data/raw/south-building
New-Item -ItemType Directory -Force data/scenes/south_building_small/images
New-Item -ItemType Directory -Force data/scenes/south_building/images
New-Item -ItemType Directory -Force data/scenes/gerrard_hall/images
New-Item -ItemType Directory -Force outputs/south_building_small/logs
New-Item -ItemType Directory -Force outputs/south_building_small/figures
New-Item -ItemType Directory -Force outputs/south_building_small/reports
```

**数学模型**

- 针孔相机模型
- 像素坐标、归一化相机坐标
- 相机内参矩阵：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

建议自己推导：像素坐标到归一化坐标 `x_norm = K^{-1} x_pixel`。

**工程实现**

- Python 3.10+
- NumPy, SciPy
- OpenCV contrib
- PyCOLMAP
- Open3D
- Matplotlib
- PyYAML, tqdm

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| 相机模型和坐标约定 | 必须自己实现 |
| 配置读取、目录管理、日志 | 建议自己实现 |
| EXIF 解析 | 可以调用成熟库 |
| COLMAP 安装 | 完全建议使用现成框架 |

**验收标准**

- `python -m src.tools.check_env` 能输出依赖状态。
- 任意 scene 都有固定目录结构。
- 能读取图像尺寸并生成初始相机内参。
- `south_building_small` 至少包含 30 张图像，且相邻图像有明显重叠。
- `data/scenes/<scene>/images/` 中的文件名与后续 `images.txt` 计划使用的名称一致。
- `.gitignore` 已忽略 `data/raw/`、`data/scenes/*/images/`、缓存文件和大规模输出。

**推荐 Commit Message**

`chore(env): add reproducible SfM environment setup`

---

### Phase 1 Feature Extraction

**功能目标**

从每张图像提取局部特征，得到可匹配的 keypoints 和 descriptors。

**输入**

- `Image`
- 可选：相机内参初值

**输出**

- `Keypoints`: `(x, y, scale, orientation)`
- `Descriptors`: RootSIFT or SIFT
- `features/<image_id>.npz`

**数学模型**

- 尺度空间 extrema
- 梯度方向直方图
- SIFT 描述子归一化
- RootSIFT:

```text
d_l1 = d / (sum(d) + eps)
d_root = sqrt(d_l1)
```

建议自己推导：为什么 L1 + sqrt 近似 Hellinger kernel，为什么能提升匹配鲁棒性。

**工程实现**

- OpenCV `cv2.SIFT_create`
- 手动实现 RootSIFT 后处理
- 保存为 `.npz`，字段固定：`keypoints`, `descriptors`, `image_id`, `image_name`

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| RootSIFT 归一化 | 必须自己实现 |
| SIFT 检测器 | 可以调用成熟库 |
| 特征缓存格式 | 必须自己设计 |

**验收标准**

- 每张图像有不少于设定阈值的特征点。
- 可视化前 500 个 keypoints，位置合理。
- RootSIFT descriptor 行向量 L2 norm 近似稳定。

**推荐 Commit Message**

`feat(sfm): implement RootSIFT feature extraction`

---

### Phase 2 Feature Matching

**功能目标**

为可能有重叠的图像对建立 2D-2D 候选匹配。

**输入**

- `Descriptors`
- 图像检索候选对或全图像对

**输出**

- `RawMatches`: `(image_id1, image_id2, idx1, idx2, distance)`
- `matches/<pair_id>.npz`

**数学模型**

- 描述子最近邻
- Lowe ratio test:

```text
dist_1 / dist_2 < ratio
```

- mutual nearest neighbor check

建议自己推导：ratio test 为什么能减少歧义匹配。

**工程实现**

- 小数据集：OpenCV BFMatcher / FLANN
- 中等数据集：FAISS optional
- 论文实验路线：vocabulary tree 找候选图像，再做局部匹配

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| Ratio test / mutual check | 必须自己实现 |
| 最近邻搜索 | 可以调用成熟库 |
| Vocabulary tree | 可以调用成熟库或暂缓 |
| 大规模检索 | 完全建议使用现成框架 |

**验收标准**

- 每个候选图像对有 raw match 统计。
- 随机可视化匹配线，明显错误不过半。
- 保存 pair 命名稳定：`image_id_a_image_id_b.npz`，且 `a < b`。

**推荐 Commit Message**

`feat(sfm): add descriptor matching with ratio test`

---

### Phase 3 Geometric Verification

**功能目标**

用几何模型过滤错误匹配，并为 scene graph 边添加模型类型。

**输入**

- `RawMatches`
- `Keypoints`
- `CameraIntrinsics`

**输出**

- `VerifiedPairs`
- `InlierMatches`
- `SceneGraph`
- 边属性：`NF, NH, NE, NS, model_type`

**数学模型**

- Fundamental matrix:

```text
x2^T F x1 = 0
```

- Essential matrix:

```text
E = K2^T F K1
```

- Homography:

```text
x2 ~ H x1
```

- Similarity transform for WTF filtering
- RANSAC / LO-RANSAC

建议自己推导：

- 8-point algorithm for F
- normalized 8-point algorithm
- Sampson distance
- E 分解得到 `(R, t)` 的四种候选并用 cheirality 选择

**工程实现**

- 课程版：OpenCV `findFundamentalMat`, `findEssentialMat`, `findHomography`
- 展示版：自己实现 normalized 8-point + RANSAC
- Scene graph 用 NetworkX 或自定义 adjacency dict

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| Normalized 8-point | 必须自己实现 |
| RANSAC 框架 | 必须自己实现 |
| E/H/F OpenCV 对照 | 可以调用成熟库 |
| WTF 检测 | 建议自己实现简化版 |

**验收标准**

- 每条 scene graph 边保留 verified inliers。
- 能区分 `general/planar/panoramic/rejected_wtf`。
- 初始化候选不来自 panoramic pair。

**推荐 Commit Message**

`feat(geometry): add multi-model geometric verification`

---

### Phase 4 Initialization

**功能目标**

选择一对高质量图像作为种子，恢复初始两相机位姿和初始 sparse points。

**输入**

- `SceneGraph`
- `InlierMatches`
- `CameraIntrinsics`

**输出**

- 初始 registered images
- 初始 `CameraPose`
- 初始 `SparsePointCloud`
- 初始 `Track` 关联

**数学模型**

- E/F 分解
- Cheirality check
- Triangulation angle
- 初始尺度：SfM 只能恢复到相似变换尺度，通常令 `||t|| = 1`

建议自己推导：

- E 分解为 `R1/R2/t`
- `depth > 0` 的 cheirality 判断
- 为什么初始化尺度不可观测

**工程实现**

- 优先选择 general + calibrated + 高 inlier + 足够 triangulation angle 的 pair
- OpenCV `recoverPose` 作为对照
- 自己实现候选打分：

```text
score = w1 * num_inliers + w2 * median_triangulation_angle - w3 * homography_ratio
```

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| 初始化 pair scoring | 必须自己实现 |
| E 分解与 cheirality | 建议自己实现 |
| SVD 求解 | 可以调用 NumPy/SciPy |

**验收标准**

- 两张初始图像注册成功。
- 初始点云大部分在两个相机前方。
- median reprojection error 小于配置阈值。

**推荐 Commit Message**

`feat(init): implement robust two-view initialization`

---

### Phase 5 Incremental Reconstruction

**功能目标**

不断选择下一张最适合注册的图像，用 2D-3D 对应估计位姿，并加入重建。

**输入**

- 当前 `ReconstructionState`
- `Tracks`
- `SceneGraph`
- 未注册图像的 2D observations

**输出**

- 新增 registered image
- 新增/更新 camera pose
- 新的 2D-3D observations

**数学模型**

- PnP:

```text
x ~ K [R | t] X
```

- PnP RANSAC
- Next Best View pyramid scoring:

```text
K_l = 2^l
w_l = K_l^2
S = sum_l w_l * occupied_cells_l
```

建议自己推导：为什么点分布越均匀，PnP 条件越好。

**工程实现**

- OpenCV `solvePnPRansac`
- 自己实现 next best image scoring
- 注册失败的图像保留失败原因：`not_enough_2d3d`, `pnp_failed`, `low_inlier_ratio`

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| Next Best View scoring | 必须自己实现 |
| 2D-3D 关联查询 | 必须自己实现 |
| PnP RANSAC | 建议先调用 OpenCV，进阶再实现 |
| unknown focal PnP | 可以暂缓 |

**验收标准**

- 每轮能输出候选图像排序。
- 新相机 PnP inliers 足够且分布不集中。
- 注册后局部重投影误差不爆炸。

**推荐 Commit Message**

`feat(recon): add incremental image registration`

---

### Phase 6 Triangulation

**功能目标**

从已注册图像中的多视图 tracks 恢复新的 3D 点，并用鲁棒估计避免错误 track 污染模型。

**输入**

- `RegisteredCameraPose`
- `Tracks`
- `InlierMatches`

**输出**

- 新增 `Point3D`
- 每个点的 RGB
- 每个点的 track observations

**数学模型**

- DLT triangulation:

```text
x_i × (P_i X) = 0
AX = 0
```

- SVD 解齐次坐标
- Cheirality
- Triangulation angle
- Reprojection error
- Recursive RANSAC triangulation

建议自己推导：

- 由 `x × PX = 0` 推出 A 矩阵每个 view 的两行约束
- triangulation angle 的 cos 公式
- 为什么小角度导致深度不稳定

**工程实现**

- 自己实现 DLT
- 自己实现 recursive RANSAC 简化版：
  1. 从 track 随机采样两个 observation
  2. DLT 得到候选 X
  3. 用 cheirality、angle、reprojection error 找 consensus
  4. consensus 足够大则建立点
  5. 从 track 中移除 consensus，递归继续

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| DLT triangulation | 必须自己实现 |
| Cheirality/angle/error 检查 | 必须自己实现 |
| Recursive RANSAC | 建议自己实现 |
| 最优多视图 triangulation | 可以调用成熟库或暂缓 |

**验收标准**

- 新点满足 `depth > 0`。
- triangulation angle 大于阈值，如 `1-2 deg`。
- 点 reprojection error 小于阈值，如 `4-8 px`。
- 错误 track 能被拆分或拒绝。

**推荐 Commit Message**

`feat(triangulation): implement recursive RANSAC triangulation`

---

### Phase 7 Bundle Adjustment

**功能目标**

联合优化相机位姿、3D 点和可选内参，降低全局/局部重投影误差。

**输入**

- `CameraPose`
- `Point3D`
- `Observations`
- `CameraIntrinsics`

**输出**

- 优化后的 poses
- 优化后的 points
- BA report: cost, iterations, residual statistics

**数学模型**

鲁棒重投影误差：

```text
E = sum_j rho( || pi(P_c, X_k) - x_jk ||^2 )
```

- Lie algebra pose update or Rodrigues vector
- Levenberg-Marquardt
- Schur complement
- Cauchy/Huber robust loss

建议自己推导：

- 投影函数 `pi(K, R, t, X)`
- 重投影误差 residual
- 小规模 BA 的参数向量展开

**工程实现**

- 课程版：SciPy `least_squares(loss="cauchy" or "huber")`
- 工程版：PyCOLMAP / COLMAP BA
- 第一版固定主点和畸变，只优化 pose + points + 可选 focal

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| 投影函数和 residual | 必须自己实现 |
| 小规模 BA | 建议自己实现 |
| 大规模稀疏 BA/Ceres | 完全建议使用现成框架 |
| Redundant View Mining | 进阶选做 |

**验收标准**

- BA 后 median reprojection error 下降。
- 没有大量点跑到相机后方。
- 局部 BA 每次注册后运行；全局 BA 按增长比例触发。

**推荐 Commit Message**

`feat(ba): add bundle adjustment with robust loss`

---

### Phase 8 Sparse Model Export

**功能目标**

将内部重建结果导出为标准 COLMAP sparse model，并生成可视化点云。

**输入**

- `ReconstructionState`
- `CameraIntrinsics`
- `Images`
- `Point3D`
- `Tracks`

**输出**

```text
data/scenes/<scene>/sparse/0/cameras.txt
data/scenes/<scene>/sparse/0/images.txt
data/scenes/<scene>/sparse/0/points3D.txt
outputs/<scene>/sparse_points.ply
outputs/<scene>/reconstruction_report.json
```

**数学模型**

- Quaternion rotation
- COLMAP world-to-camera extrinsics:

```text
x_cam = R * x_world + t
C_world = -R^T * t
```

- RGB point color from image observations

建议自己推导：从 camera-to-world 变换求 world-to-camera 的 `R, t`。

**工程实现**

- 自己写 COLMAP text exporter
- 用 PyCOLMAP 读取导出的 sparse model 做合法性校验
- 用 Open3D 导出 PLY

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| COLMAP text export | 必须自己实现 |
| Quaternion conversion | 建议自己实现并单测 |
| COLMAP binary export | 可以调用 PyCOLMAP |
| PLY 可视化 | 可以调用 Open3D |

**验收标准**

- `pycolmap.Reconstruction("data/scenes/<scene>/sparse/0")` 可成功读取。
- `cameras/images/points3D` 数量与内部状态一致。
- 图像名与 `data/<scene>/images` 完全一致。

**推荐 Commit Message**

`feat(export): support COLMAP sparse model export`

---

### Phase 9 Interface to 3DGS

**功能目标**

保证 SfM 输出能直接作为 3DGS 输入，降低后续转换成本。

**输入**

- COLMAP sparse model
- 原始或去畸变图像
- 可选：VGGT pose/depth prior

**输出**

3DGS 标准数据布局：

```text
data/scenes/<scene>/
  images/
  sparse/
    0/
      cameras.bin or cameras.txt
      images.bin or images.txt
      points3D.bin or points3D.txt
```

**数学模型**

- COLMAP 坐标系到 3DGS loader 内部坐标的转换由 3DGS 代码处理。
- 本阶段只保证 COLMAP 语义正确，不自行改轴。

**工程实现**

- 优先输出 COLMAP text，再用 COLMAP/PyCOLMAP 转 binary。
- 若图像有畸变，推荐使用 COLMAP `image_undistorter` 或 PyCOLMAP undistort 后给 3DGS。
- 3DGS 的初始化点云可直接来自 `points3D.txt`。

**是否建议手动复现**

| 模块 | 建议 |
| --- | --- |
| COLMAP 数据规范 | 必须自己掌握 |
| 3DGS loader 适配 | 建议只做薄接口 |
| 去畸变 | 完全建议使用 COLMAP |
| VGGT 到 COLMAP 转换 | 后续单独实现 |

**验收标准**

- Graphdeco 3DGS 或兼容实现能读取 scene。
- 相机视锥和 sparse points 在可视化中空间一致。
- 不出现图像左右/上下翻转、相机朝向反向、点云尺度异常爆炸。

**推荐 Commit Message**

`feat(interface): prepare COLMAP outputs for 3DGS`

---

## 4. 面向 3DGS 的接口设计

### 4.1 必须保留的数据

| 数据 | 是否必须 | 原因 |
| --- | --- | --- |
| Camera intrinsics | 必须 | 3DGS 需要 FoV/焦距计算投影 |
| Camera extrinsics | 必须 | 训练每张图像的视角 |
| Sparse point cloud | 必须 | 3DGS 初始化高斯中心 |
| Point RGB | 必须 | 初始化颜色 |
| Track information | 强烈建议 | COLMAP `points3D.txt` 中包含 track，可用于调试 BA 和过滤 |
| Image id/name mapping | 必须 | 3DGS 按图像名读取训练图像 |
| Distortion parameters | 建议保留 | 可用于去畸变或复查相机模型 |
| Reprojection error | 建议保留 | 可过滤低质量点 |

### 4.2 推荐输出格式：COLMAP

推荐 COLMAP 的原因：

1. 原始 3D Gaussian Splatting 代码默认支持 COLMAP sparse reconstruction。
2. COLMAP 的 `cameras/images/points3D` 已覆盖 3DGS 需要的 intrinsics、extrinsics、point cloud、RGB、track。
3. 后续若切换到 VGGT，只要把 VGGT 输出转换成 COLMAP sparse model，就能复用 3DGS 入口。
4. COLMAP text 格式适合课程展示与调试，binary 格式适合工程运行。

### 4.3 必须提前避免的问题

| 问题 | 后果 | 规范 |
| --- | --- | --- |
| 坐标系混乱 | 3DGS 相机看不到点云 | 内部统一 world-to-camera；导出前明确 `x_cam = R x_world + t` |
| `qvec/tvec` 写反 | 相机全部反向或漂移 | 单测 `C = -R^T t`，可视化相机中心 |
| image id 重排 | points3D track 指向错误图像 | 从导入开始固定 `image_id`，导出不重新排序 |
| point id 重排 | track 信息失效 | 删除点时保留 id 或建立稳定 remap |
| 图像重命名 | 3DGS 找不到训练图像 | `images.txt` 的 `NAME` 必须相对 `images/` 匹配 |
| 尺度漂移 | 3DGS 初始化不稳定 | 初始尺度可任意，但同一重建内必须一致；不要每轮归一化 |
| 畸变未处理 | 训练投影误差大 | 简化版使用 PINHOLE/SIMPLE_PINHOLE；有畸变则先 undistort |
| 点云含大量离群点 | 3DGS 产生噪声高斯 | 导出前按 reprojection error、track length、angle 过滤 |
| 颜色采样错误 | 初始化颜色异常 | 从 track 中可见图像采样 RGB 均值或中位数 |
| 法向缺失 | 一般不影响 3DGS | 3DGS 不依赖点法向；PLY 法向可为空 |

---

## 5. 项目目录规划

推荐从当前轻量项目扩展为如下结构：

```text
final/
  README.md
  requirements.txt
  Structure-from-Motion Revisited.pdf

  configs/
    default.yaml
    scenes/
      example.yaml

  data/
    <scene_name>/
      images/
      masks/
      features/
      matches/
      verified/
      tracks/
      sparse/
        0/
      undistorted/

  docs/
    notes.md
    experiment_manual.md
    sfm_reproduction_execution_manual.md
    api_colmap_3dgs.md

  src/
    sfm/
      __init__.py
      camera.py
      features.py
      matching.py
      geometry.py
      tracks.py
      initialization.py
      reconstruction.py
      triangulation.py
      bundle_adjustment.py
      filtering.py
      io/
        colmap_text.py
        colmap_binary.py
        ply.py
      utils/
        logging.py
        visualization.py
        metrics.py
    tools/
      extract_features.py
      match_features.py
      verify_matches.py
      run_incremental_sfm.py
      export_colmap.py
      check_env.py

  tests/
    test_camera.py
    test_geometry.py
    test_triangulation.py
    test_colmap_export.py

  outputs/
    <scene_name>/
      logs/
      figures/
      reports/
      sparse_points.ply

  third_party/
    gaussian-splatting/
    vggt/
```

解耦原则：

- `src/sfm` 只负责 SfM，不直接依赖 3DGS 训练代码。
- `io/colmap_text.py` 是 SfM 与 3DGS 的主接口。
- `third_party` 只放外部项目或 submodule，不混入自己的核心代码。
- `data/scenes/<scene>/sparse/0` 严格模拟 COLMAP 输出，方便 3DGS 直接读取。
- `VGGT` 升级方案只需要新增 `src/vggt_adapter/`，输出仍落到 `sparse/0`。

---

## 6. 环境管理

当前检查结果：

| 包 | 当前状态 |
| --- | --- |
| numpy | 已安装 |
| scipy | 已安装 |
| networkx | 已安装 |
| matplotlib | 已安装 |
| PyYAML | 已安装 |
| tqdm | 已安装 |
| Pillow | 已安装 |
| opencv-python / cv2 | 缺失 |
| pycolmap | 缺失 |
| open3d | 缺失 |

安装命令：

```bash
conda activate mm26
pip install -r requirements.txt
```

若使用系统 COLMAP：

```bash
conda install -c conda-forge colmap
```

若后续要 C++/Ceres 完整复现 BA：

```bash
conda install -c conda-forge ceres-solver eigen glog gflags suitesparse cmake ninja
```

课程阶段建议：

- 第一版使用 Python + OpenCV + SciPy + PyCOLMAP。
- 不强制手写 Ceres BA。
- 使用 PyCOLMAP/COLMAP 作为工程正确性对照，而不是替代所有自己实现模块。

---

## 7. Git 协作规范

每个阶段提交前都应写清楚目标、修改文件、验收标准和推荐 commit。

| 阶段 | 本阶段目标 | 修改文件 | 验收标准 | 推荐 Commit Message |
| --- | --- | --- | --- | --- |
| Phase 0 | 建立环境、配置、目录与依赖 | `requirements.txt`, `configs/*`, `src/tools/check_env.py` | 环境检查通过 | `chore(env): add reproducible SfM environment setup` |
| Phase 1 | RootSIFT 特征提取 | `src/sfm/features.py`, `src/tools/extract_features.py`, `tests/test_features.py` | 特征缓存和可视化正确 | `feat(sfm): implement RootSIFT feature extraction` |
| Phase 2 | 特征匹配 | `src/sfm/matching.py`, `src/tools/match_features.py` | raw matches 可视化合理 | `feat(sfm): add descriptor matching with ratio test` |
| Phase 3 | 几何验证与 scene graph | `src/sfm/geometry.py`, `src/tools/verify_matches.py` | F/E/H inliers 与类型输出正确 | `feat(geometry): add multi-model geometric verification` |
| Phase 4 | 初始化 | `src/sfm/initialization.py`, `tests/test_initialization.py` | 初始 pair、pose、points 合理 | `feat(init): implement robust two-view initialization` |
| Phase 5 | 增量注册 | `src/sfm/reconstruction.py` | 新图像 PnP 注册成功 | `feat(recon): add incremental image registration` |
| Phase 6 | 鲁棒三角化 | `src/sfm/triangulation.py`, `tests/test_triangulation.py` | 点满足深度/角度/误差阈值 | `feat(triangulation): implement recursive RANSAC triangulation` |
| Phase 7 | BA 与过滤 | `src/sfm/bundle_adjustment.py`, `src/sfm/filtering.py` | BA 后误差下降 | `feat(ba): add bundle adjustment with robust loss` |
| Phase 8 | COLMAP 导出 | `src/sfm/io/colmap_text.py`, `src/tools/export_colmap.py` | PyCOLMAP 可读取 | `feat(export): support COLMAP sparse model export` |
| Phase 9 | 3DGS 接口 | `src/tools/prepare_3dgs_scene.py`, `docs/api_colmap_3dgs.md` | 3DGS loader 可读取 | `feat(interface): prepare COLMAP outputs for 3DGS` |

提交信息规范：

- 使用 `feat(scope): action object` 或 `chore(scope): action object`。
- 避免 `update`, `modify`, `fix bug`, `test` 这类无法追踪意图的消息。
- 每个 commit 只完成一个清晰阶段，不混入无关格式化。

---

## 8. 最小可展示路线

为了适合本科数学建模课程展示，建议采用“两条线并行”：

### 8.1 数学建模展示线

必须自己实现：

- RootSIFT 后处理
- normalized 8-point + RANSAC
- scene graph 边属性与分类
- initialization scoring
- DLT triangulation
- triangulation angle / cheirality / reprojection error
- Next Best View pyramid scoring
- 小规模 BA residual
- COLMAP text export

这些模块最能体现数学推导、建模思想和工程掌控力。

### 8.2 工程可靠性对照线

建议调用成熟库：

- SIFT detector
- 最近邻搜索
- OpenCV PnP
- SciPy least_squares 或 PyCOLMAP BA
- COLMAP undistortion
- PyCOLMAP sparse model validation

这些部分重复造轮子成本高，且不是课程价值最高的部分。

### 8.3 最终验收

一个 scene 的最终验收应包含：

1. `features/matches/verified/tracks` 中间结果可复查。
2. 至少完成两视图初始化和若干图像增量注册。
3. sparse point cloud 可视化形状合理。
4. `data/<scene>/sparse/0` 能被 PyCOLMAP 读取。
5. 3DGS 数据目录符合默认 loader 预期。
6. 文档中能解释每个模块的数学模型和工程取舍。
