# Paper-Aligned SfM Pipeline

This document defines the canonical SfM pipeline used by this project after Phase 7I.
It is intentionally different from the chronological progress log in
`docs/sfm_reproduction_progress.md`: this file is the execution contract for the next
implementation pass.

## 1. Design Goal

The goal is to align the reconstruction control flow with the refinement strategy in
*Structure-from-Motion Revisited* while keeping the current educational Python
implementation usable.

The canonical loop is:

```text
two-view initialization
  -> incremental image registration
  -> triangulation from newly registered views
  -> local bundle adjustment
  -> filtering
  -> growth-triggered global refinement
  -> final global refinement
```

The main entry point should be a controller, not a hand-written sequence of many
commands.

## 2. Canonical Phases

### Phase I: Initialization

Input:

```text
data/scenes/<scene>/verified/*.npz
data/scenes/<scene>/verified/scene_graph.json
```

Command:

```bash
python -m src.tools.initialize_reconstruction --scene configs/scenes/<scene>.yaml
```

Output:

```text
data/scenes/<scene>/sparse/0/initial_pair.npz
data/scenes/<scene>/sparse/0/initial_points.npz
data/scenes/<scene>/sparse/0/reconstruction_state.json
```

### Phase II: Incremental Reconstruction Loop

Each iteration performs:

```text
1. register one new image by PnP
2. triangulate tracks among registered images
3. run local BA around the new image
4. evaluate residuals
5. filter observations and points
6. optionally run degenerate camera diagnostics
7. if model growth threshold is reached, run global refinement
```

The high-level controller is:

```bash
python -m src.tools.run_paper_aligned_sfm --scene configs/scenes/<scene>.yaml
```

The lower-level tools remain available, but they are not the recommended way to run
the main reconstruction.

### Phase III: Growth-Triggered Global Refinement

Triggered when either registered image count or point count grows by a configured
ratio since the previous global refinement.

The global refinement sequence is:

```text
pre-global RT
global BA
residual evaluation
global filtering
post-global RT
global BA
residual evaluation
global filtering
degenerate camera diagnostics / optional removal
```

This matches the paper-level idea that RT, BA, and filtering should be iterated until
new post-BA RT points and filtered observations diminish. The current controller uses
a bounded number of global refinement cycles for safety.

### Phase IV: Final Refinement

After no more images can be registered, the controller runs:

```text
final RT
global BA
residual evaluation
filtering
degenerate camera diagnostics
final residual report
```

## 3. Filtering Semantics

Filtering is one logical stage with three parts:

```text
observation filtering
point filtering
camera diagnostics / optional camera filtering
```

Observation and point filtering are currently implemented in:

```text
src/sfm/filtering.py
src/tools/run_paper_aligned_sfm.py
```

Degenerate camera diagnostics are currently implemented in:

```text
src/tools/evaluate_degenerate_cameras.py
```

Important distinction:

```text
Residual reports are measurements.
Filtering is the state-changing stage.
```

The canonical sequence is therefore:

```text
BA -> residual evaluation -> filtering -> post-filter residual evaluation
```

Camera diagnostics may use either pre-filter or post-filter residuals, but camera
removal should be considered part of global filtering, not a separate refinement
method.

## 4. Canonical vs Experimental Components

Canonical:

```text
initialize_reconstruction.py
run_paper_aligned_sfm.py
triangulate_registered_tracks.py with current RT policy
run_bundle_adjustment.py with local/global scopes
src/sfm/filtering.py through run_paper_aligned_sfm.py
evaluate_registered_residuals.py
evaluate_degenerate_cameras.py as diagnostics
```

Experimental:

```text
post_ba_moderate / post_ba_strict RT policies
balanced capped observation sampling in SciPy BA
recursive track splitting diagnostics
camera removal without manual inspection
```

The balanced capped observation sampling in the SciPy BA implementation is an
engineering workaround for limited compute. It is not claimed to be the paper's BA
selection strategy.

## 5. Current Known Gaps

The current implementation is not a full COLMAP/Ceres reproduction.

Known gaps:

```text
SciPy BA is capped and does not use Schur/Ceres.
Intrinsics and radial distortion are fixed.
Point filtering does not yet enforce triangulation-angle geometry globally.
Degenerate camera filtering is diagnostic-first.
Registration and RT are still separate tools internally.
Global refinement convergence is bounded by controller parameters.
```

## 6. Recommended Baseline Run

Start from a clean initialization state, then run:

```bash
D:\06_envs\mm26\python.exe -m src.tools.run_paper_aligned_sfm \
  --scene configs/scenes/south_building_small.yaml \
  --max-register 50 \
  --local-ba-after-registration \
  --global-ba-growth-ratio 1.3 \
  --global-ba-min-interval 4 \
  --filter-after-ba \
  --diagnose-degenerate-cameras \
  --report-name paper_aligned_sfm_report.json
```

Use `--remove-degenerate-cameras` only after inspecting diagnostic reports.

The `--global-ba-min-interval` option prevents small early reconstructions from
running global refinement after every newly registered image. Local BA remains the
high-frequency stabilizer; global refinement is reserved for larger model growth.

## 7. Frontend, Matching, and Export Notes

The default frontend remains RootSIFT with brute-force ratio matching:

```bash
python -m src.tools.extract_features --scene configs/scenes/<scene>.yaml \
  --feature-backend rootsift

python -m src.tools.match_features --scene configs/scenes/<scene>.yaml \
  --matcher-backend rootsift_bf \
  --strategy exhaustive
```

The feature and matcher stages are now backend-selectable. The DISK + LightGlue path
uses the official vendored LightGlue repository and still writes the same project
NPZ feature and match files consumed by geometric verification:

```bash
python -m pip install -e third_party/LightGlue

python -m src.tools.extract_features --scene configs/scenes/<scene>.yaml \
  --feature-backend disk \
  --force

python -m src.tools.match_features --scene configs/scenes/<scene>.yaml \
  --matcher-backend lightglue \
  --strategy exhaustive \
  --force
```

When using the one-command PowerShell pipeline:

```powershell
.\scripts\run_sfm_full.ps1 `
  -Scene dtu_scan83 `
  -FeatureBackend disk `
  -MatcherBackend lightglue `
  -ReportName dtu_scan83_disk_lightglue_paper_aligned_sfm_report.json
```

The RootSIFT NPZ pipeline also supports retrieval pair generation through
`src.tools.match_features`. The retrieval mode builds a lightweight visual-word
TF-IDF index from the existing `features/*.npz`, selects top-k image pairs, and still
writes the same `matches/*.npz` files consumed by verification:

```bash
python -m src.tools.match_features --scene configs/scenes/<scene>.yaml \
  --matcher-backend rootsift_bf \
  --strategy retrieval \
  --retrieval-top-k 20 \
  --retrieval-num-words 256
```

For 3DGS handoff, export the current reconstruction as a COLMAP text sparse model:

```bash
python -m src.tools.export_colmap_model --scene configs/scenes/<scene>.yaml
```
