from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.sfm.config import load_scene_config, resolve_project_path
from src.tools import evaluate_registered_residuals
from src.tools import filter_reconstruction
from src.tools import run_bundle_adjustment
from src.tools import triangulate_registered_tracks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-style BA/RT/filtering iterative refinement.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--max-refinement-iterations", type=int, default=3)
    parser.add_argument("--min-new-points", type=int, default=50)
    parser.add_argument("--min-filtered-observations", type=int, default=50)
    parser.add_argument("--max-observation-error", type=float, default=8.0)
    parser.add_argument("--max-point-median-error", type=float, default=8.0)
    parser.add_argument("--max-point-max-error", type=float, default=32.0)
    parser.add_argument("--ba-max-iterations", type=int, default=40)
    parser.add_argument("--ba-max-points", type=int, default=1200)
    parser.add_argument("--ba-max-observations", type=int, default=6000)
    parser.add_argument("--loss", default="cauchy", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"])
    parser.add_argument("--f-scale", type=float, default=4.0)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--post-ba-rt-policy", default="current", choices=["current", "post_ba_moderate", "post_ba_strict"])
    parser.add_argument("--enable-track-merge", action="store_true")
    parser.add_argument("--run-prefix", default="phase7e")
    parser.add_argument("--report-name", default="ba_rt_refinement_report.json")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    report_dir = resolve_project_path(scene["output_dir"]) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    iteration_summaries = []
    converged = False
    stop_reason = "max_refinement_iterations_reached"

    baseline_residual = run_registered_residual(args.scene, f"{args.run_prefix}_iter00_registered_residual_baseline_report.json")

    for iteration in range(1, args.max_refinement_iterations + 1):
        prefix = f"{args.run_prefix}_iter{iteration:02d}"

        pre_rt = run_rt(
            scene_path=args.scene,
            stage="pre_ba_rt",
            report_name=f"{prefix}_pre_ba_rt_report.json",
        )
        ba_after_pre = run_global_ba(
            args=args,
            report_name=f"{prefix}_global_ba_after_pre_rt_report.json",
        )
        residual_after_pre_ba = run_registered_residual(
            args.scene,
            f"{prefix}_registered_residual_after_pre_ba_report.json",
        )
        filtering_after_pre = run_filtering(
            args=args,
            residual_report=report_dir / f"{prefix}_registered_residual_after_pre_ba_report.json",
            report_name=f"{prefix}_filtering_after_pre_ba_report.json",
        )
        residual_after_filtering = run_registered_residual(
            args.scene,
            f"{prefix}_registered_residual_after_filtering_report.json",
        )
        post_rt = run_rt(
            scene_path=args.scene,
            stage="post_ba_rt",
            report_name=f"{prefix}_post_ba_rt_report.json",
            residual_report=report_dir / f"{prefix}_registered_residual_after_filtering_report.json",
            max_observation_error=args.max_observation_error,
            rt_policy=args.post_ba_rt_policy,
            enable_track_merge=args.enable_track_merge,
        )
        final_ba = run_global_ba(
            args=args,
            report_name=f"{prefix}_final_global_ba_report.json",
        )
        residual_after_final_ba = run_registered_residual(
            args.scene,
            f"{prefix}_registered_residual_after_final_ba_report.json",
        )
        final_filtering = run_filtering(
            args=args,
            residual_report=report_dir / f"{prefix}_registered_residual_after_final_ba_report.json",
            report_name=f"{prefix}_final_filtering_report.json",
        )
        final_residual = run_registered_residual(
            args.scene,
            f"{prefix}_final_registered_residual_report.json",
        )

        post_ba_rt_new_points = int(post_rt["new_points3D"])
        filtered_observations = int(final_filtering["removed_observations_by_reprojection"])
        summary = {
            "iteration": iteration,
            "pre_ba_rt_new_points": int(pre_rt["new_points3D"]),
            "pre_ba_rt_new_observations": int(pre_rt["new_observations"]),
            "filtering_after_pre_ba_removed_observations": int(filtering_after_pre["removed_observations_by_reprojection"]),
            "filtering_after_pre_ba_removed_points": int(filtering_after_pre["removed_points_total"]),
            "post_ba_rt_new_points": post_ba_rt_new_points,
            "post_ba_rt_new_observations": int(post_rt["new_observations"]),
            "final_filtering_removed_observations": filtered_observations,
            "final_filtering_removed_points": int(final_filtering["removed_points_total"]),
            "final_points3D": int(final_filtering["points_after"]),
            "final_observations": int(final_filtering["observations_after"]),
            "final_residual_summary": final_residual["error_summary"],
            "report_names": {
                "pre_ba_rt": f"{prefix}_pre_ba_rt_report.json",
                "global_ba_after_pre_rt": f"{prefix}_global_ba_after_pre_rt_report.json",
                "filtering_after_pre_ba": f"{prefix}_filtering_after_pre_ba_report.json",
                "post_ba_rt": f"{prefix}_post_ba_rt_report.json",
                "final_global_ba": f"{prefix}_final_global_ba_report.json",
                "final_filtering": f"{prefix}_final_filtering_report.json",
                "final_registered_residual": f"{prefix}_final_registered_residual_report.json",
            },
        }
        iteration_summaries.append(summary)

        if post_ba_rt_new_points < args.min_new_points and filtered_observations < args.min_filtered_observations:
            converged = True
            stop_reason = "post_ba_rt_points_and_filtered_observations_below_thresholds"
            break

    report = {
        "scene_name": scene["scene_name"],
        "max_refinement_iterations": args.max_refinement_iterations,
        "min_new_points": args.min_new_points,
        "min_filtered_observations": args.min_filtered_observations,
        "max_observation_error": args.max_observation_error,
        "ba_max_iterations": args.ba_max_iterations,
        "ba_max_points": args.ba_max_points,
        "ba_max_observations": args.ba_max_observations,
        "loss": args.loss,
        "f_scale": args.f_scale,
        "post_ba_rt_policy": args.post_ba_rt_policy,
        "enable_track_merge": bool(args.enable_track_merge),
        "run_prefix": args.run_prefix,
        "baseline_residual_summary": baseline_residual["error_summary"],
        "iterations": iteration_summaries,
        "converged": converged,
        "stop_reason": stop_reason,
    }
    report_path = report_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Iterations completed: {len(iteration_summaries)}")
    print(f"Converged: {converged}")
    print(f"Stop reason: {stop_reason}")
    if iteration_summaries:
        last = iteration_summaries[-1]
        print(f"Last post-BA RT new points: {last['post_ba_rt_new_points']}")
        print(f"Last final filtered observations: {last['final_filtering_removed_observations']}")
        print(f"Final mean residual: {last['final_residual_summary']['mean']:.3f} px")
        print(f"Final >8px observations: {last['final_residual_summary']['observations_above_8px']}")
    print(f"Report: {report_path}")
    return 0


def run_registered_residual(scene_path: str, report_name: str) -> dict:
    call_tool(
        evaluate_registered_residuals.main,
        [
            "evaluate_registered_residuals",
            "--scene",
            scene_path,
            "--report-name",
            report_name,
        ],
    )
    return read_report(scene_path, report_name)


def run_rt(
    scene_path: str,
    stage: str,
    report_name: str,
    residual_report: Path | None = None,
    max_observation_error: float = 8.0,
    rt_policy: str = "current",
    enable_track_merge: bool = False,
) -> dict:
    argv = [
        "triangulate_registered_tracks",
        "--scene",
        scene_path,
        "--stage",
        stage,
        "--report-name",
        report_name,
        "--rt-policy",
        rt_policy,
    ]
    if residual_report is not None:
        argv.extend(["--residual-report", str(residual_report), "--max-observation-error", str(max_observation_error)])
    if enable_track_merge:
        argv.append("--enable-track-merge")
    call_tool(triangulate_registered_tracks.main, argv)
    return read_report(scene_path, report_name)


def run_global_ba(args: argparse.Namespace, report_name: str) -> dict:
    call_tool(
        run_bundle_adjustment.main,
        [
            "run_bundle_adjustment",
            "--scene",
            args.scene,
            "--scope",
            "global",
            "--max-iterations",
            str(args.ba_max_iterations),
            "--loss",
            args.loss,
            "--f-scale",
            str(args.f_scale),
            "--max-points",
            str(args.ba_max_points),
            "--max-observations",
            str(args.ba_max_observations),
            "--min-track-length",
            str(args.min_track_length),
            "--report-name",
            report_name,
        ],
    )
    return read_report(args.scene, report_name)


def run_filtering(args: argparse.Namespace, residual_report: Path, report_name: str) -> dict:
    call_tool(
        filter_reconstruction.main,
        [
            "filter_reconstruction",
            "--scene",
            args.scene,
            "--ba-report",
            str(residual_report),
            "--max-reprojection-error",
            str(args.max_observation_error),
            "--max-point-median-error",
            str(args.max_point_median_error),
            "--max-point-max-error",
            str(args.max_point_max_error),
            "--min-track-length",
            str(args.min_track_length),
            "--report-name",
            report_name,
        ],
    )
    return read_report(args.scene, report_name)


def read_report(scene_path: str, report_name: str) -> dict:
    config = load_scene_config(scene_path)
    report_path = resolve_project_path(config["scene"]["output_dir"]) / "reports" / report_name
    return json.loads(report_path.read_text(encoding="utf-8"))


def call_tool(main_func, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = argv
        exit_code = main_func()
        if exit_code not in (0, None):
            raise RuntimeError(f"Tool failed with exit code {exit_code}: {' '.join(argv)}")
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
