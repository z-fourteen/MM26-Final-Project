# Dataset Download Guide

本项目第一阶段使用 COLMAP 官方数据集作为 SfM 复现主数据。你可以手动下载并解压，Codex 后续会基于固定目录继续执行特征提取、匹配和重建。

## 推荐下载顺序

1. `South Building`：主数据集，优先下载。
2. `Gerrard Hall`：对照数据集，South Building 跑通后再下载。
3. 3DGS 官方示例场景：只用于验证 3DGS 训练环境，不作为手写 SfM 主数据。

## 下载入口

COLMAP 数据集说明页：

```text
https://colmap.github.io/datasets.html
```

COLMAP 数据集目录：

```text
https://demuc.de/colmap/datasets/
```

如果目录直链不可访问，可以从 COLMAP 文档页面点击对应数据集链接下载。

3D Gaussian Splatting 官方数据入口：

```text
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
```

## 解压位置

下载后按下面方式整理：

```text
data/
  raw/
    south-building/
      <原始解压内容>
    gerrard-hall/
      <原始解压内容>
  scenes/
    south_building/
      images/
    south_building_small/
      images/
    gerrard_hall/
      images/
```

## South Building 整理方法

1. 将 South Building 原始数据解压到：

```text
data/raw/south-building/
```

2. 找到其中的图像目录，通常名为 `images` 或包含 `.jpg/.png` 的目录。

3. 将全量图像复制到：

```text
data/scenes/south_building/images/
```

4. 从全量图像中选取 30-50 张连续、有重叠的图像，复制到：

```text
data/scenes/south_building_small/images/
```

建议先选择文件名排序后的前 50 张；如果发现重叠不足，再人工换成连续绕建筑拍摄的一段。

## Gerrard Hall 整理方法

1. 将 Gerrard Hall 原始数据解压到：

```text
data/raw/gerrard-hall/
```

2. 将图像复制到：

```text
data/scenes/gerrard_hall/images/
```

## 注意事项

- 不要重命名图像文件。
- 为了保证复现实验在不同机器上路径清晰，建议把实验用图像真实复制到 `data/scenes/<scene>/images/`，不要依赖硬链接或软链接。
- 不要覆盖原图；若需要缩放，另建 `images_resized/`。
- `data/raw/` 和 `data/scenes/*/images/` 已被 `.gitignore` 忽略，不会提交到 Git。
- Phase 1 默认从 `data/scenes/south_building_small/images/` 开始。
