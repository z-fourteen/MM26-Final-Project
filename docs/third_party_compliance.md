# Third-Party Compliance

This project combines an in-project SfM implementation with vendored
third-party research code. The boundary is intentional:

- `src/` contains project-owned code, adapters, tools, and SfM logic.
- `src/innovation/` documents project-owned innovations without duplicating
  implementation code.
- `third_party/` contains upstream repositories kept as intact as practical.
- `data/3dgs_inputs/` and `outputs/` contain generated artifacts and are not
  committed through normal Git.

## Vendored Repositories

| Component | Location | Role | Modification policy |
|---|---|---|---|
| VGGT | `third_party/vggt/` | Feed-forward geometry reconstruction branch. | Keep upstream core intact; integrate through `src/adapters/` and `src/tools/`. |
| 3D Gaussian Splatting | `third_party/gaussian-splatting/` | Neural rendering backend for SfM/VGGT outputs. | Keep upstream core intact; call via wrapper commands. |
| VGGT-OMEGA | `vggt-omega/` | Deferred experimental branch. | Not part of the current normalized pipeline. |

## Required Practices

- Preserve upstream license files and copyright notices.
- Do not remove original headers in third-party files.
- Prefer wrappers in `src/tools/` and adapters in `src/adapters/` over edits
  inside `third_party/`.
- Project-owned scripts must not live inside vendored repositories. Move them
  to `src/tools/` and keep only upstream code under `third_party/`.
- If a third-party file must be modified, document the reason in
  `docs/third_party_patches.md` and add a short "Modified for integration"
  note near the file header.
- Do not commit model checkpoints, raw datasets, generated sparse models, or
  3DGS outputs with normal Git.
- Cite the original VGGT and 3DGS papers/repositories in the final report.

## Environment Policy

The SfM branch can run in the main CPU-friendly environment. VGGT and 3DGS may
require separate CUDA-enabled environments because their PyTorch/CUDA and
extension requirements can conflict. Keep upstream requirement files in their
vendored repositories, and use the root `requirements.txt` only for the main
project and lightweight adapters.
