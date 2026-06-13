# SfM / VGGT to 3DGS Pipeline

Both reconstruction branches hand off to 3DGS through the same COLMAP-style
source layout, but they must use separate source roots so their cameras and
point clouds never overwrite each other.

## Canonical 3DGS Inputs

```text
data/3dgs_inputs/<scene_name>_sfm/
  images/
  sparse/
    0/
      cameras.txt
      images.txt
      points3D.txt
      points3D.ply

data/3dgs_inputs/<scene_name>_vggt/
  images/
  sparse/
    0/
      cameras.txt
      images.txt
      points3D.txt
      points3D.ply
```

`data/scenes/<scene_name>/` remains the project working directory for raw
images, SfM features, matches, reports, and intermediate reconstruction state.
3DGS training should use `data/3dgs_inputs/...` as its `-s` source path.

The bundled `gaussian-splatting` text loader requires `PINHOLE` cameras, so all
exporters write `PINHOLE fx fy cx cy`.

## SfM Branch

Prepare a 3DGS source directory from the current SfM reconstruction:

```powershell
D:\06_envs\mm26\python.exe -m src.tools.prepare_3dgs_from_sfm --scene configs/scenes/dtu_scan55.yaml --output-name dtu_scan55_sfm
```

This command copies scene images, exports the current SfM reconstruction as
COLMAP text, writes `points3D.ply`, and runs the checker.

## VGGT Branch

First run VGGT inference to produce:

```text
predictions.npz   # extrinsic, intrinsic
points_depth.npz  # xyz, rgb, optional confidence
```

Then prepare the VGGT 3DGS source directory:

```powershell
D:\06_envs\mm26\python.exe -m src.tools.prepare_3dgs_from_vggt --scene configs/scenes/dtu_scan55.yaml --predictions <path-to-predictions.npz> --points <path-to-points_depth.npz> --output-name dtu_scan55_vggt
```

The VGGT exporter writes preprocessed VGGT images into `images/`, because VGGT
intrinsics are predicted in that preprocessed image coordinate system.

## Validate Without CUDA

Either branch can be validated independently:

```powershell
D:\06_envs\mm26\python.exe -m src.tools.check_3dgs_scene --source-path data/3dgs_inputs/dtu_scan55_sfm --write-ply --report-name outputs/dtu_scan55/reports/dtu_scan55_sfm_3dgs_scene_check.json
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
python train.py -s data/3dgs_inputs/dtu_scan55_sfm -m outputs/3dgs/dtu_scan55_sfm
python train.py -s data/3dgs_inputs/dtu_scan55_vggt -m outputs/3dgs/dtu_scan55_vggt
```

If the 3DGS invocation needs an explicit image folder:

```bash
python train.py -s data/3dgs_inputs/dtu_scan55_sfm -m outputs/3dgs/dtu_scan55_sfm --images images
```
