from __future__ import annotations

import argparse

from src.sfm.config import load_scene_config
from src.sfm.reconstruction import load_reconstruction_state
from src.sfm.residuals import (
    evaluate_registered_residuals_in_memory,
    write_residual_npz,
    write_residual_report,
)
from src.sfm.runtime import load_runtime_context


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate reprojection residuals for observations in the current registered reconstruction state."
    )
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--report-name", default="registered_residual_report.json")
    parser.add_argument("--compact-report", action="store_true")
    parser.add_argument("--write-errors-npz", action="store_true")
    args = parser.parse_args()

    runtime = load_runtime_context(args.scene, focal_scale=args.focal_scale)
    config = runtime.config
    scene = config["scene"]

    state = load_reconstruction_state(runtime.sparse_dir)
    valid_observations, errors_array, summary = evaluate_registered_residuals_in_memory(
        state=state,
        cameras=runtime.cameras,
        keypoints_by_name=runtime.keypoints_by_name,
    )
    report_path = runtime.report_dir / args.report_name
    npz_name = ""
    if args.write_errors_npz:
        npz_name = f"{report_path.stem}_errors.npz"
        write_residual_npz(runtime.report_dir / npz_name, valid_observations, errors_array)
    report = write_residual_report(
        report_path=report_path,
        scene_name=scene["scene_name"],
        observations=valid_observations,
        errors=errors_array,
        summary=summary,
        compact=args.compact_report,
        npz_name=npz_name or None,
    )

    summary = report["error_summary"]
    print(f"Scene: {scene['scene_name']}")
    print(f"Scope: registered_observations")
    print(f"Observations evaluated: {report['num_observations_evaluated']} / {report['num_observations_total']}")
    print(f"Median reprojection error: {summary['median']:.3f} px")
    print(f"Mean reprojection error: {summary['mean']:.3f} px")
    print(f"P95 reprojection error: {summary['p95']:.3f} px")
    print(f"Observations > 8 px: {summary['observations_above_8px']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
