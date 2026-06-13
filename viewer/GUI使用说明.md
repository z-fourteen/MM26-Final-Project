# 3DGS 相机位姿查看平台使用说明

## 1. 平台作用

本平台用于在浏览器中查看单个 3D Gaussian Splatting（3DGS）实验结果，并展示该实验对应的相机位姿。

主要用途包括：

- 实时渲染 Graphdeco 3DGS 训练生成的 `point_cloud.ply`。
- 显示 SfM 或 VGGT 求得的相机中心、相机轨迹和相机视锥。
- 按图片名称切换相机，并立即从对应相机视角渲染场景。
- 分别检查 SfM、VGGT 或其他方法生成的独立实验结果。
- 上传自定义 PLY 和相机参数，在不修改代码的情况下查看新实验。

## 2. 预置实验

平台目前包含两个服务器本地预置结果。

预置实验由 `viewer/examples/*/scene.json` 自动发现。Git 提交中包含场景清单和相机参数，但大型 `point_cloud.ply` 不直接进入普通 Git 历史。

### 2.1 COLMAP baseline · Truck · 30K

- 使用标准 COLMAP 稀疏重建提供的相机参数和初始点云。
- 使用 3DGS 训练 30,000 次后的 PLY。
- 包含 251 个相机。
- 用于查看完整 Truck 图像序列的 COLMAP + Graphdeco 3DGS 基线结果。
- 该结果不是项目 `src/sfm` 自研 SfM 管线生成的结果。

### 2.2 VGGT + BA · Truck · 7K

- 使用 VGGT 预测并经过 Bundle Adjustment 优化的相机参数。
- 使用 3DGS 训练 7,000 次后的 PLY。
- 包含 32 个相机。
- 这是平台默认加载的预置实验。

选择左侧“实验结果”，然后点击“加载预置结果”即可切换。两个实验使用各自的 PLY 和相机坐标系，互不混合。

## 3. 自定义上传

自定义上传区包含以下字段：

| 字段 | 是否必需 | 作用 |
|---|---:|---|
| 方法标签 | 否 | 标记结果来自 SfM、VGGT 或其他方法，不改变渲染算法。 |
| 场景名称 | 否 | 设置界面中显示的实验名称。 |
| 3DGS PLY | 是 | 提供需要渲染的高斯场景。 |
| 相机参数 | 否 | 提供相机位置、朝向、内参和图片名称。 |

上传完成后点击“加载上传结果”。文件首先上传到服务器的 Gradio 临时目录，后端校验文件并生成临时场景 ID，浏览器随后从服务器流式加载 PLY。

上传结果只在当前服务进程中注册。服务器重启后需要重新上传；预置实验不受影响。

## 3.1 协作者直接接收和放置示例 PLY

Git 仓库已经包含 `scene.json` 和 `cameras.json`，但不包含体积较大的 PLY。发送者只需要把两份 3DGS 结果文件直接发给协作者。

协作者克隆或更新仓库后，必须将收到的文件重命名并放到以下固定位置：

```text
viewer/examples/
├── colmap_truck_30k/
│   ├── scene.json          # Git 已提供
│   ├── cameras.json        # Git 已提供
│   └── point_cloud.ply     # 将收到的 COLMAP 30K PLY 放在这里
└── vggt_ba_truck_7k/
    ├── scene.json          # Git 已提供
    ├── cameras.json        # Git 已提供
    └── point_cloud.ply     # 将收到的 VGGT+BA 7K PLY 放在这里
```

无论发送者原来使用什么文件名，放入示例目录后都必须命名为：

```text
point_cloud.ply
```

两个文件的准确大小为：

| 示例 | 目标路径 | 准确字节数 | 约合大小 |
|---|---|---:|---:|
| COLMAP baseline 30K | `viewer/examples/colmap_truck_30k/point_cloud.ply` | 513,109,068 | 489 MiB |
| VGGT + BA 7K | `viewer/examples/vggt_ba_truck_7k/point_cloud.ply` | 100,329,187 | 95.7 MiB |

Linux 下在仓库根目录直接复制并重命名：

```bash
cp /path/to/收到的_colmap_30k.ply \
  viewer/examples/colmap_truck_30k/point_cloud.ply

cp /path/to/收到的_vggt_ba_7k.ply \
  viewer/examples/vggt_ba_truck_7k/point_cloud.ply
```

Windows 11 PowerShell 下，先进入仓库根目录，然后执行：

```powershell
Copy-Item "C:\收到文件的位置\colmap_30k.ply" `
  ".\viewer\examples\colmap_truck_30k\point_cloud.ply"

Copy-Item "C:\收到文件的位置\vggt_ba_7k.ply" `
  ".\viewer\examples\vggt_ba_truck_7k\point_cloud.ply"
```

放置完成后的检查命令：

```bash
ls -lh viewer/examples/*/point_cloud.ply
stat -c '%s %n' viewer/examples/*/point_cloud.ply
```

输出的字节数应与上表一致。Windows 11 PowerShell 可以在仓库根目录执行：

```powershell
Get-Item viewer\examples\*\point_cloud.ply | Select-Object Length, FullName
```

如果协作者只收到其中一份 PLY，只需放置对应文件。GUI 启动时只会在“预置实验”中显示本地文件完整的示例，不会因为缺少另一份 PLY 而无法启动。

放好文件后直接启动：

```bash
python -m viewer.app --host 127.0.0.1 --port 7860
```

刷新页面后即可在“预置实验”下拉框中看到对应结果。

### 3.2 使用准备脚本放置文件（可选）

不想手动重命名和移动时，可以让脚本校验文件大小并复制到正确位置：

```bash
python -m viewer.prepare_examples --mode copy \
  --source colmap_truck_30k=/path/to/收到的_colmap_30k.ply \
  --source vggt_ba_truck_7k=/path/to/收到的_vggt_ba_7k.ply
```

只准备一个示例时，只传入对应的 `--source` 即可。例如：

```bash
python -m viewer.prepare_examples --mode copy \
  --source vggt_ba_truck_7k=/path/to/收到的_vggt_ba_7k.ply
```

其中 `scene.json` 是 GUI 清单，描述实验名称、方法、文件名、预期 PLY 大小、高斯数量和相机数量。它不参与 3DGS 训练或渲染。

两份 PLY 都被 `viewer/examples/.gitignore` 忽略，不会被普通 `git add` 提交到 GitHub。

如果协作者本机已经保留原始实验目录，也可以建立符号链接，避免额外复制约 585 MiB：

```bash
python -m viewer.prepare_examples --mode symlink
```

## 4. PLY 与相机文件的关系

Graphdeco 训练生成的 `point_cloud.ply` 通常包含：

- 高斯中心位置 `x, y, z`
- 高斯尺度 `scale_*`
- 高斯旋转 `rot_*`
- 不透明度 `opacity`
- 球谐颜色参数 `f_dc_*`、`f_rest_*`

PLY **不包含训练相机的信息**，通常没有：

- 相机位置
- 相机朝向
- 相机内参
- 图片名称
- 图片路径

因此：

- 只上传 PLY：可以自由查看 3DGS 场景，但不能显示相机轨迹或切换到训练相机视角。
- 上传 PLY 和相机参数：可以使用平台的全部相机位姿功能。

Graphdeco 实验推荐上传：

```text
point_cloud/iteration_<迭代次数>/point_cloud.ply
cameras.json
```

## 5. 支持的相机格式

### 5.1 Graphdeco cameras.json

推荐格式。平台读取：

- `position`
- `rotation`
- `fx`、`fy`
- `width`、`height`
- `img_name`

该格式可以完整显示相机中心、轨迹、视锥，并进入真实相机视角。

### 5.2 VGGT cameras.json

支持带有以下字段的 VGGT 导出格式：

```text
world_from_camera_3x4
intrinsic_3x3
camera_center_world
image
```

### 5.3 COLMAP images.txt

平台可以从 COLMAP 四元数和平移向量恢复相机中心和朝向。

`images.txt` 通常不包含完整相机内参，因此视锥尺寸可能使用默认值。需要精确内参时优先使用 Graphdeco `cameras.json`。

### 5.4 CSV

CSV 支持以下任意一组位置列：

```text
camera_center_world_x,camera_center_world_y,camera_center_world_z
```

或：

```text
x,y,z
```

只有相机中心的 CSV 可以显示相机位置和轨迹，但无法显示真实视锥，也不能进入相机视角。

## 6. 界面功能

### 6.1 三维场景操作

- 鼠标左键拖动：围绕目标旋转。
- 鼠标右键拖动：平移。
- 鼠标滚轮：缩放。
- `P` 键：切换普通高斯模式和点模式。

### 6.2 相机选择

“当前相机”下拉框使用相机参数中的图片名称。

选择相机后，平台会立即：

1. 高亮对应的相机中心和视锥。
2. 将 WebGL 相机移动到该相机位置。
3. 应用该相机朝向。
4. 根据 `fy` 和图像高度设置垂直视场角。
5. 重新渲染 3DGS 场景。

“上一相机”和“下一相机”具有相同的实时切换效果。

### 6.3 进入视角

切换相机时已经自动进入相机视角。“进入视角”按钮用于用户手动旋转场景后，重新回到当前选中相机的标准视角。

### 6.4 可视化开关

- 相机中心：显示或隐藏所有相机位置。
- 相机轨迹：显示或隐藏按文件顺序连接的轨迹线。
- 相机视锥：显示或隐藏相机视锥和朝向轴。
- 点模式：将高斯显示为屏幕空间圆点。
- 高斯尺寸：调整所有高斯在屏幕中的显示尺寸。
- 全局视角：返回能观察整个相机轨迹的视角。
- 切换 Y 轴：处理部分数据集上下方向相反的问题。
- 全屏：将三维查看区域切换为浏览器全屏。

## 7. 在服务器启动

启动服务：

```bash
python -m viewer.app --host 127.0.0.1 --port 7860
```

使用 SSH 端口转发时推荐监听 `127.0.0.1`，无需将服务直接暴露到公网。

如果出现：

```text
ERROR: [Errno 98] address already in use
```

说明端口已被其他进程占用。检查：

```bash
ss -ltnp | grep ':7860'
```

也可以改用其他端口：

```bash
python -m viewer.app --host 127.0.0.1 --port 7861
```

## 8. 渲染实现

平台由两部分组成：

- Gradio + FastAPI：负责预置实验、自定义上传、文件校验、场景注册和大文件流式传输。
- GaussianSplats3D + Three.js：负责浏览器端 WebGL 高斯渲染、相机轨迹、视锥和交互控制。

当前主要渲染设置：

- 球谐阶数：0
- 透明度移除阈值：5
- 渐进式 PLY 加载
- GPU 半精度协方差存储
- 不使用 SfM/VGGT 跨方法坐标对齐

大 PLY 使用 HTTP Range 响应传输，不会在 Python 中完整复制一份到内存。

## 10. 常见问题

### 页面显示的相机数量不符合预期

先检查当前选择的预置实验。COLMAP baseline 预置包含 251 个相机，VGGT + BA 预置包含 32 个相机。

### 只看到场景，看不到相机

检查是否上传了相机参数文件，以及“相机中心”“相机轨迹”“相机视锥”开关是否开启。

### 有轨迹但没有视锥

相机文件可能只包含中心位置，没有旋转矩阵。此时无法恢复真实朝向。

### 修改代码后页面没有变化

重启服务器上的查看器，然后在浏览器中执行强制刷新：

```text
Ctrl + F5
```

### 大型 PLY 加载缓慢

首次加载需要从服务器传输并在浏览器解析完整 PLY。文件达到数百 MB 时需要等待，并确保浏览器有足够内存。后续可以考虑将 PLY 转换为压缩的 `.ksplat` 格式以提升加载速度。
