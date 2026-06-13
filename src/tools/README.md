# Tool Index

This directory contains project command-line entry points. Shared algorithms
belong in `src/sfm/`; third-party import glue belongs in `src/adapters/`;
project innovation notes belong in `src/innovation/`.

## Environment And Data Preparation

- `check_env.py`: basic environment sanity checks.
- `prepare_scenes.py`: create scene working directories from raw datasets.
- `validate_features.py`: validate extracted feature files.

## SfM Pipeline Stages

- `extract_features.py`: feature extraction.
- `match_features.py`: exhaustive or retrieval-based feature matching.
- `verify_matches.py`: geometric verification and scene-graph creation.
- `initialize_reconstruction.py`: select and initialize the first image pair.
- `register_images.py`: incremental image registration utility.
- `triangulate_registered_tracks.py`: triangulate tracks after registration or BA.
- `run_bundle_adjustment.py`: local/global bundle adjustment entry point.
- `run_paper_aligned_sfm.py`: paper-aligned incremental SfM orchestration.

## SfM Diagnostics And Recovery

- `evaluate_degenerate_cameras.py`: diagnose degenerate camera geometry.
- `evaluate_registered_residuals.py`: evaluate registered-image residuals.
- `manage_reconstruction_checkpoint.py`: inspect and restore reconstruction checkpoints.

## 3DGS Handoff

- `export_colmap_model.py`: export current SfM state as COLMAP text.
- `prepare_3dgs_from_sfm.py`: build a 3DGS source directory from SfM output.
- `prepare_3dgs_from_vggt.py`: build a 3DGS source directory from VGGT output.
- `check_3dgs_scene.py`: CPU-side validation for 3DGS source directories.

## VGGT / 3DGS Experiments

- `run_vggt_inference.py`: lightweight VGGT inference runner without demo dependencies.
- `train_vggt_gsplat.py`: minimal gsplat training initialized from VGGT outputs.
