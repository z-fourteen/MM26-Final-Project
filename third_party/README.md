# Third-Party Repositories

This directory contains vendored upstream repositories used by the project.

- `vggt/`: VGGT reconstruction branch.
- `gaussian-splatting/`: 3D Gaussian Splatting rendering backend.

Project-specific integration code should live in `src/adapters/` or
`src/tools/`. Keep vendored core code intact whenever possible so upstream
updates and license attribution remain clear.

Generated data, checkpoints, caches, and training outputs under this directory
are ignored by Git.
