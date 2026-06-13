# MM26-Final-Project

This project contains two geometry reconstruction branches that both export a
COLMAP-style scene for 3D Gaussian Splatting:

- `src/sfm/`: project-owned incremental SfM implementation and diagnostics.
- `third_party/vggt/`: vendored VGGT repository used through thin adapters.
- `third_party/gaussian-splatting/`: vendored 3DGS training/rendering backend.

Generated artifacts are intentionally kept out of normal Git history. See
`docs/artifacts_manifest.md` for the local artifact policy and
`docs/third_party_compliance.md` for third-party integration rules.

## Layout

```text
configs/                 Scene and third-party path configuration
src/sfm/                 Project SfM implementation
src/tools/               Project command-line tools and pipeline wrappers
src/adapters/            Thin adapters around vendored repositories
src/innovation/          Notes on project-owned innovations and adaptations
third_party/vggt/        Vendored VGGT upstream repository
third_party/gaussian-splatting/
                         Vendored 3DGS upstream repository
data/3dgs_inputs/        Generated 3DGS-ready inputs, ignored by Git
outputs/                 Generated reports and training outputs, ignored by Git
report/                  Final report sources and curated figures
```

## 3DGS Handoff

SfM and VGGT results should be exported into separate source roots:

```text
data/3dgs_inputs/<scene>_sfm/
data/3dgs_inputs/<scene>_vggt/
```

On a CUDA-enabled machine, train 3DGS through the vendored backend:

```bash
python third_party/gaussian-splatting/train.py -s data/3dgs_inputs/dtu_scan55_sfm -m outputs/3dgs/dtu_scan55_sfm
```

Project-owned VGGT/3DGS experiment helpers live in `src/tools/`, not inside the
vendored VGGT repository:

```bash
python -m src.tools.run_vggt_inference --image_folder data/scenes/dtu_scan55/images --output_dir outputs/dtu_scan55/vggt
python -m src.tools.prepare_3dgs_from_vggt --scene configs/scenes/dtu_scan55.yaml --predictions outputs/dtu_scan55/vggt/predictions.npz --points outputs/dtu_scan55/vggt/points_depth.npz
```
