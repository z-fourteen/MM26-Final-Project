# Third-Party Notices

This repository vendors third-party research projects for academic use. Their
original license files and upstream documentation are retained in place.

| Component | Location | Purpose | License file |
|---|---|---|---|
| VGGT | `third_party/vggt/` | Feed-forward geometry branch and VGGT preprocessing utilities. | `third_party/vggt/LICENSE.txt` |
| 3D Gaussian Splatting | `third_party/gaussian-splatting/` | 3DGS training and rendering backend. | `third_party/gaussian-splatting/LICENSE.md` |

The project-owned implementation and adapters live under `src/`. Vendored code
is kept separate to preserve attribution and make future upstream synchronization
clear.
