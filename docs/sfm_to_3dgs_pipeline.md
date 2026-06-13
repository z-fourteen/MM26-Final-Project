# SfM to 3DGS Pipeline

This project exports each reconstructed scene as a COLMAP text model under the
same scene root used by 3DGS.

## Expected Layout

```text
data/scenes/<scene_name>/
  images/
  sparse/
    0/
      cameras.txt
      images.txt
      points3D.txt
      points3D.ply  # optional, checker can create it
```

The bundled `gaussian-splatting` loader accepts COLMAP scenes through
`train.py -s <scene_root>`. Its text camera loader requires `PINHOLE` cameras,
so `export_colmap_model` writes `PINHOLE fx fy cx cy` even when `fx == fy`.

## Export

```powershell
D:\06_envs\mm26\python.exe -m src.tools.export_colmap_model --scene configs/scenes/dtu_scan55.yaml --output-dir data/scenes/dtu_scan55/sparse/0
```

## Validate Without CUDA

```powershell
D:\06_envs\mm26\python.exe -m src.tools.check_3dgs_scene --source-path data/scenes/dtu_scan55 --write-ply --report-name outputs/dtu_scan55/reports/3dgs_scene_check.json
```

The checker verifies:

- `images/` and `sparse/0/` exist.
- `cameras.txt`, `images.txt`, and `points3D.txt` are present.
- Cameras use the `PINHOLE` model required by this 3DGS text loader.
- Every registered image referenced by `images.txt` exists on disk.
- Image sizes match `cameras.txt`.
- Sparse points are present.

## Train On A CUDA Machine

```bash
python train.py -s data/scenes/dtu_scan55 -m outputs/3dgs/dtu_scan55
```

If the 3DGS invocation needs an explicit image folder:

```bash
python train.py -s data/scenes/dtu_scan55 -m outputs/3dgs/dtu_scan55 --images images
```
