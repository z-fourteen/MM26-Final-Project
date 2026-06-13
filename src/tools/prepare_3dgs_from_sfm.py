from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.sfm.config import load_scene_config, resolve_project_path
from src.tools import check_3dgs_scene, export_colmap_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a 3DGS source directory from the SfM reconstruction.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--output-name", default="", help="Defaults to <scene_name>_sfm")
    parser.add_argument("--output-root", default="data/3dgs_inputs")
    parser.add_argument("--focal-scale", type=float, default=1.2)
    parser.add_argument("--copy-images", action="store_true", default=True)
    parser.add_argument("--link-images", action="store_true", help="Use symlinks instead of copying images when possible")
    parser.add_argument("--skip-check", action="store_true")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    scene_name = str(scene["scene_name"])
    output_name = args.output_name or f"{scene_name}_sfm"
    output_dir = resolve_project_path(args.output_root) / output_name
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    source_images_dir = resolve_project_path(scene["image_dir"])

    if args.copy_images:
        copy_or_link_images(source_images_dir, images_dir, link=args.link_images)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    call_tool(
        export_colmap_model.main,
        [
            "export_colmap_model",
            "--scene",
            args.scene,
            "--output-dir",
            str(sparse_dir),
            "--focal-scale",
            str(args.focal_scale),
        ],
    )

    report_path = resolve_project_path(scene["output_dir"]) / "reports" / f"{output_name}_3dgs_scene_check.json"
    if not args.skip_check:
        call_tool(
            check_3dgs_scene.main,
            [
                "check_3dgs_scene",
                "--source-path",
                str(output_dir),
                "--write-ply",
                "--report-name",
                str(report_path),
            ],
        )

    print(f"3DGS SfM source: {output_dir}")
    print(
        "Train command: python third_party/gaussian-splatting/train.py "
        f"-s {output_dir.as_posix()} -m outputs/3dgs/{output_name}"
    )
    return 0


def copy_or_link_images(source_dir: Path, target_dir: Path, link: bool = False) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(path for path in source_dir.iterdir() if path.is_file()):
        target = target_dir / source.name
        if target.exists():
            continue
        if link:
            try:
                target.symlink_to(source.resolve())
                continue
            except OSError:
                pass
        shutil.copy2(source, target)


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
