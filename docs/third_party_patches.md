# Third-Party Patches

No third-party source patches are currently required.

Integration is handled through project-owned files:

- `src/adapters/vggt_adapter.py`
- `src/tools/prepare_3dgs_from_sfm.py`
- `src/tools/prepare_3dgs_from_vggt.py`
- `src/tools/check_3dgs_scene.py`
- `src/tools/run_vggt_inference.py`
- `src/tools/train_vggt_gsplat.py`

The following project-owned scripts were moved out of the vendored VGGT
repository:

- `third_party/vggt/infer_v100.py` -> `src/tools/run_vggt_inference.py`
- `third_party/vggt/train_vggt_gsplat.py` -> `src/tools/train_vggt_gsplat.py`

The older `third_party/vggt/prepare_3dgs_from_vggt.py` wrapper was removed in
favor of the project-level `src/tools/prepare_3dgs_from_vggt.py`.
