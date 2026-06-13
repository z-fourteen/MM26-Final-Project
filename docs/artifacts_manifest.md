# Experiment Artifacts Manifest

This project keeps large reconstruction and rendering artifacts outside normal
Git history. The files below are preserved locally for the final report and
3DGS handoff, while the repository tracks the code, configs, commands, and this
manifest.

## Policy

- Commit source code, configs, documentation, report sources, and small report
  figures.
- Do not commit raw datasets, reconstructed sparse models, copied image sets,
  checkpoints, logs, or 3DGS training outputs with normal Git.
- For final delivery, package required artifacts as release attachments or use
  Git LFS if the submission platform requires versioned binary files.

## Local Artifact Roots

| Path | Purpose | Git policy |
|---|---|---|
| `data/3dgs_inputs/` | Standardized 3DGS inputs exported from SfM or VGGT. | Ignored; package externally when needed. |
| `data/scenes/*/sparse/0/` | SfM reconstruction state, COLMAP text exports, checkpoints, and point data. | Ignored except `.gitkeep`. |
| `outputs/` | Reports, diagnostics, logs, and model outputs produced by experiments. | Ignored except `.gitkeep`. |
| `report/figures/` | Small selected figures used by the final report. | Tracked when figures are curated. |

## Current Important Artifacts

| Artifact | Local path | Notes |
|---|---|---|
| DTU scan55 SfM to 3DGS input | `data/3dgs_inputs/dtu_scan55_sfm/` | Contains copied images and `sparse/0` COLMAP text files for 3DGS. |
| DTU scan55 experiment outputs | `outputs/dtu_scan55/` | Contains run diagnostics and intermediate reports. |
| South Building experiment outputs | `outputs/south_building/` | Preserved for comparison with the earlier SfM reconstruction. |
| South Building Small experiment outputs | `outputs/south_building_small/` | Preserved as the initial small-scene baseline. |
| Truck experiment outputs | `outputs/truck/` | Preserved for cross-dataset registration analysis. |

## Reproducibility Notes

The final report should cite the exact scene config, command, and exported
artifact path for every visual result. If an artifact is used in a figure,
copy only the final selected screenshot or plot into `report/figures/`.
