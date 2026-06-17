# Innovation Notes

This directory documents the project-owned innovations and adaptations without
duplicating implementation code. The actual implementation stays in `src/sfm/`,
`src/tools/`, and `src/adapters/`.

## SfM Project Adaptations

The SfM branch is not a plain script-level reproduction. It was adapted for the
project datasets, diagnostics, and 3DGS handoff:

- Camera intrinsics prefer EXIF focal length when available, then fall back to a
  simple pinhole estimate.
- Bundle adjustment can optimize a shared focal length so exported 3DGS cameras
  remain consistent with the reconstruction.
- Retrieval-based matching is supported to avoid exhaustive matching costs on
  larger scenes.
- Image registration includes retry, soft accept, cheirality checks, depth
  dispersion checks, spatial coverage gates, and residual-based diagnostics.
- Planar verified pairs can supplement general matches, but are constrained by
  coverage and geometric gates.
- Next-best-view selection combines 2D-3D support, image-grid coverage,
  registered-neighbor evidence, and scene-graph diagnostics.
- Local/global BA scheduling, growth-ratio triggering, and track cache support
  make the incremental pipeline practical on local hardware.
- Post-BA triangulation supports residual gating, track merge, recursive track
  splitting, and detailed reports.

## 3DGS Handoff Adaptations

The project adds a unified handoff layer so SfM and VGGT can be compared through
the same 3DGS backend:

- SfM exports to `data/3dgs_inputs/<scene>_sfm/`.
- VGGT exports to `data/3dgs_inputs/<scene>_vggt/`.
- Both branches use the same COLMAP-style `images/` plus `sparse/0/` layout.
- Exporters write `PINHOLE` cameras for compatibility with the vendored 3DGS
  text loader.
- VGGT handoff uses VGGT-preprocessed images because predicted intrinsics live
  in that coordinate system.
- CPU-side validation checks image presence, camera format, image dimensions,
  sparse points, and optional PLY export before CUDA training.

## Artifact Governance

Large reconstruction and rendering artifacts are preserved locally for the final
report but are not committed through normal Git. The tracked manifest in
`docs/artifacts_manifest.md` records which artifacts matter and where they are
stored.
