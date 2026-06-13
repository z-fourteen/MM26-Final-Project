下面是一份“项目模块规范化与集成规划方案”。我会按高级项目架构 + 开源合规的角度来规划，重点是：**第三方仓库保持原貌、主项目只写薄适配层、两条算法线边界清晰、成果能被评审看懂**。

---

# 1. Proposed Target File Tree

推荐目标结构如下：

```text
USTC_MM_26_final/
├─ README.md
├─ requirements.txt
├─ environment.yml                 # 可选：主项目统一环境
├─ configs/
│  ├─ default.yaml
│  ├─ scenes/
│  │  ├─ dtu_scan55.yaml
│  │  └─ ...
│  ├─ third_party.yaml             # VGGT/3DGS 路径、权重路径、开关
│  └─ pipeline.yaml                # 统一 pipeline 配置
│
├─ src/
│  ├─ sfm/                         # 我们自研 / 复现的 Incremental SfM 主体
│  ├─ tools/
│  │  ├─ extract_features.py
│  │  ├─ run_paper_aligned_sfm.py
│  │  ├─ export_colmap_model.py
│  │  ├─ check_3dgs_scene.py
│  │  ├─ prepare_3dgs_from_sfm.py
│  │  └─ prepare_3dgs_from_vggt.py
│  ├─ adapters/
│  │  ├─ vggt_adapter.py           # 封装 VGGT 输出契约，不改 VGGT 源码
│  │  └─ colmap_3dgs_adapter.py    # 统一 COLMAP text / 3DGS input 规范
│  ├─ innovation/
│  │  ├─ registration_retry.py     # 我们的增量注册改进思想
│  │  ├─ pnp_diagnostics.py        # PnP 诊断、soft accept 策略说明/实现
│  │  └─ scene_quality.py          # 对 SfM/VGGT 输出质量评估
│  └─ __init__.py
│
├─ third_party/
│  ├─ vggt/                        # 原 VGGT 仓库，尽量原貌保留
│  │  ├─ vggt/
│  │  ├─ demo_colmap.py
│  │  ├─ README.md
│  │  ├─ LICENSE.txt
│  │  └─ ...
│  └─ gaussian-splatting/          # 原 3DGS 仓库，尽量原貌保留
│     ├─ scene/
│     ├─ gaussian_renderer/
│     ├─ train.py
│     ├─ render.py
│     ├─ README.md
│     ├─ LICENSE.md
│     └─ ...
│
├─ data/
│  ├─ raw/                         # 原始数据，不提交
│  ├─ scenes/                      # 主项目 scene 工作区
│  │  └─ dtu_scan55/
│  │     ├─ images/
│  │     ├─ features/
│  │     ├─ matches/
│  │     ├─ verified/
│  │     └─ sparse/0/              # SfM 内部状态 / 可导出
│  └─ 3dgs_inputs/                 # 统一给 3DGS 的输入，不混淆来源
│     ├─ dtu_scan55_sfm/
│     │  ├─ images/
│     │  └─ sparse/0/
│     └─ dtu_scan55_vggt/
│        ├─ images/
│        └─ sparse/0/
│
├─ outputs/
│  ├─ dtu_scan55/
│  │  ├─ reports/
│  │  └─ logs/
│  └─ 3dgs/
│     ├─ dtu_scan55_sfm/
│     └─ dtu_scan55_vggt/
│
├─ docs/
│  ├─ sfm_paper_aligned_pipeline.md
│  ├─ sfm_to_3dgs_pipeline.md
│  ├─ vggt_to_3dgs_pipeline.md
│  ├─ third_party_compliance.md
│  └─ experiment_protocol.md
│
├─ report/
│  ├─ main.tex
│  └─ reference.bib
│
└─ licenses/
   ├─ THIRD_PARTY_NOTICES.md
   ├─ VGGT_LICENSE.txt
   └─ 3DGS_LICENSE.md
```

重点是这三层边界：

```text
src/          我们自己的代码、适配器、创新点
third_party/  原始开源仓库，尽量不改
data/3dgs_inputs/  两条线统一交给 3DGS 的标准输入
```

---

# 2. 核心执行策略表

| 类别 | 文件/模块 | 建议去向 | 处理动作 | 原则 |
|---|---|---|---|---|
| 并入本地 `src/` | SfM 复现代码 | `src/sfm/` | 保留为主项目核心算法 | 明确这是自主实现/复现部分 |
| 并入本地 `src/` | SfM -> 3DGS 导出 | `src/tools/prepare_3dgs_from_sfm.py` | 保留并继续完善 | 输出到 `data/3dgs_inputs/<scene>_sfm` |
| 并入本地 `src/` | VGGT -> 3DGS 适配 | `src/tools/prepare_3dgs_from_vggt.py` 或 `src/adapters/vggt_adapter.py` | 从 VGGT 原脚本抽象为项目级薄适配器 | 不改 VGGT 核心模型 |
| 并入本地 `src/` | 3DGS 输入检查 | `src/tools/check_3dgs_scene.py` | 作为统一验收工具 | 两条线都必须通过 |
| 并入本地 `src/` | 创新算法 | `src/innovation/` | 放 PnP retry、soft accept、质量评估等 | 让评审直观看到“我们的贡献” |
| 保留不动 | VGGT 核心模型 | `third_party/vggt/` | 尽量原样迁移 | 方便同步上游 |
| 保留不动 | 3DGS 核心训练/渲染 | `third_party/gaussian-splatting/` | 尽量原样迁移 | 不直接改 `train.py`/renderer |
| 保留不动 | 3DGS CUDA/C++ 扩展 | `third_party/gaussian-splatting/submodules/` | 保留原目录 | 避免构建破坏 |
| 根目录配置 | 主项目依赖 | `requirements.txt` / `environment.yml` | 只放主项目 + 轻量适配依赖 | 不把 VGGT/3DGS 全量依赖混在一起 |
| 根目录配置 | 第三方依赖 | `third_party/vggt/requirements.txt`、`third_party/gaussian-splatting/environment.yml` | 保留原文件 | 作为可选环境说明 |
| 根目录配置 | 权重路径 | `configs/third_party.yaml` | 指定 VGGT checkpoint、3DGS repo path | 不把权重提交进 git |
| 根目录配置 | 训练输出 | `outputs/3dgs/<scene>_{sfm,vggt}` | 统一输出 | 方便横向对比 |
| 合并冲突 | Wandb/TensorBoard | 主项目默认关闭；第三方内部保留 | 用 wrapper 参数控制 | 不让日志系统污染根目录 |
| 合并冲突 | PyTorch/CUDA 版本 | 分环境管理 | `env_sfm_cpu`、`env_vggt_cuda`、`env_3dgs_cuda` | 避免一个 requirements 解决所有问题 |
| 合并冲突 | COLMAP 格式 | 统一为 text `PINHOLE` | 由 adapter 输出 | 避免 3DGS loader 读失败 |

---

# 3. Lean & Intact 原则：扫描与清理清单

## 3.1 VGGT 仓库

建议保留：

```text
vggt/vggt/
vggt/demo_colmap.py
vggt/README.md
vggt/LICENSE.txt
vggt/requirements.txt
vggt/pyproject.toml
```

可保留但标记为第三方 demo：

```text
vggt/demo_gradio.py
vggt/demo_viser.py
vggt/visual_util.py
vggt/examples/
```

可归档或不纳入主项目展示：

```text
vggt/training/
vggt/docs/
vggt/CODE_OF_CONDUCT.md
vggt/CONTRIBUTING.md
vggt/requirements_demo.txt
```

处理建议：

```text
不删除原仓库文件，而是整体移入 third_party/vggt/
然后在主项目 README 中说明：
“VGGT is vendored as third_party/vggt with original license retained.”
```

如果确实要瘦身，可以做：

```text
third_party/vggt/
  vggt/
  demo_colmap.py
  README.md
  LICENSE.txt
  requirements.txt
  pyproject.toml
  THIRD_PARTY_ORIGIN.md
```

不建议改：

```text
vggt/vggt/models/
vggt/vggt/heads/
vggt/vggt/utils/
```

## 3.2 3DGS 仓库

建议保留：

```text
gaussian-splatting/train.py
gaussian-splatting/render.py
gaussian-splatting/metrics.py
gaussian-splatting/arguments/
gaussian-splatting/scene/
gaussian-splatting/gaussian_renderer/
gaussian-splatting/utils/
gaussian-splatting/submodules/
gaussian-splatting/LICENSE.md
gaussian-splatting/README.md
gaussian-splatting/environment.yml
```

可归档或不展示：

```text
gaussian-splatting/assets/
gaussian-splatting/SIBR_viewers/
gaussian-splatting/full_eval.py
gaussian-splatting/results.md
gaussian-splatting/convert.py
```

注意：`submodules/` 通常包含 CUDA rasterizer，不能随意删。

处理建议：

```text
third_party/gaussian-splatting/
  train.py
  render.py
  scene/
  gaussian_renderer/
  submodules/
  ...
```

不建议改：

```text
gaussian_renderer/
scene/gaussian_model.py
submodules/
```

如果需要适配，写 wrapper：

```text
src/tools/run_3dgs_train.py
```

而不是改 `train.py`。

---

# 4. 合规与规范化检查清单

## 4.1 开源许可证处理

必须做：

```text
[ ] 保留 VGGT 原始 LICENSE.txt
[ ] 保留 3DGS 原始 LICENSE.md
[ ] 在 licenses/THIRD_PARTY_NOTICES.md 中列出来源、许可证、用途
[ ] 在 README.md 中说明 VGGT 和 3DGS 是第三方开源组件
[ ] 不删除原作者 copyright header
[ ] 修改第三方文件时，在文件头注明 Modified by <project> on <date>
[ ] 报告和论文中引用 VGGT 与 3DGS 论文
```

尤其注意：

```text
3DGS license 明确偏 research / non-commercial 使用；
VGGT license 包含 acceptable use policy，且不同 checkpoint 商用权限不同。
```

你的项目是学术/数学建模用途，方向上匹配 research use，但报告里仍应写清：

```text
本项目仅用于课程/学术研究展示，不作商业发布。
```

## 4.2 第三方代码修改规范

推荐规则：

```text
原则上不改 third_party/*
必须修改时：
1. 新建 patch 文件记录差异
2. 在 docs/third_party_patches.md 说明原因
3. 文件头保留原 copyright
4. 添加 “Modified for integration only”
```

更推荐：

```text
src/adapters/*
src/tools/*
```

里写适配代码，不动第三方核心。

## 4.3 学术创新表达

建议在项目报告中明确三层贡献：

```text
第一层：自研/复现的几何 SfM
- RootSIFT feature pipeline
- retrieval / exhaustive matching
- geometric verification
- incremental registration
- PnP retry
- soft accept
- BA/filtering
- COLMAP text export

第二层：前沿模型引用
- VGGT 用作 feed-forward geometry prior / alternative reconstruction branch
- 3DGS 用作 neural rendering backend

第三层：统一接口创新
- 两条几何来源统一为 COLMAP-like 3DGS input
- check_3dgs_scene 无 CUDA 校验
- SfM vs VGGT 的同场景可对比实验框架
```

建议目录表达：

```text
src/sfm/          我们复现/实现的几何算法
src/innovation/   我们的改进策略
src/adapters/     第三方模型接口
third_party/      原始开源引用
```

这样评审能一眼看懂：

```text
哪些是我们做的；
哪些是引用的；
我们如何把它们组合成完整创新系统。
```

---

# 5. 推荐执行路线

我建议分三步执行，不要一口气大搬家：

## Step 1：建立合规文档与第三方声明

新增：

```text
licenses/THIRD_PARTY_NOTICES.md
docs/third_party_compliance.md
configs/third_party.yaml
```

## Step 2：目录迁移但不魔改

把：

```text
vggt/ -> third_party/vggt/
gaussian-splatting/ -> third_party/gaussian-splatting/
```

然后更新工具中的路径引用。

## Step 3：完善 wrappers / adapters

保留主项目入口：

```text
src/tools/prepare_3dgs_from_sfm.py
src/tools/prepare_3dgs_from_vggt.py
src/tools/check_3dgs_scene.py
```

未来可加：

```text
src/tools/run_vggt_inference.py
src/tools/run_3dgs_train.py
src/tools/compare_sfm_vggt_3dgs.py
```

---

# 6. 当前状态评价

以你现在项目来看：

```text
SfM 主线：工程闭环基本完整
VGGT 主线：已有第三方脚本和我们新增的 3DGS adapter，但还需要规范化放置
3DGS 主线：作为第三方训练/渲染 backend 存在，接口已被 checker 验证
合规文档：还需要补
目录结构：还需要从根目录堆叠转成 third_party vendoring
```

所以这不是“代码能力不够”，而是到了该做**项目架构治理**的时候了。现在做这个整理是对的，会让项目从“能跑”提升到“像一个高质量学术工程”。

本轮没有改文件，git 语句：

```powershell
git status --short
```