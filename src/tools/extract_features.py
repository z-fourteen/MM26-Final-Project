from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import draw_keypoints_preview, extract_rootsift, list_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract RootSIFT features for one configured scene.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute existing feature files.")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]

    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    max_image_size = int(default["sfm"].get("max_image_size", 1600))
    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    results = []
    for image_path in tqdm(images, desc=f"Extracting {scene['scene_name']}"):
        feature_path = feature_dir / f"{image_path.stem}.npz"
        if feature_path.exists() and not args.force:
            data = np.load(feature_path)
            results.append(
                {
                    "image_name": str(data["image_name"]),
                    "num_keypoints": int(data["keypoints"].shape[0]),
                    "feature_path": str(feature_path),
                    "cached": True,
                }
            )
            continue

        result = extract_rootsift(
            image_path=image_path,
            feature_dir=feature_dir,
            max_image_size=max_image_size,
        )
        results.append(
            {
                "image_name": result.image_name,
                "image_size": result.image_size,
                "resized_size": result.resized_size,
                "scale_factor": result.scale_factor,
                "num_keypoints": result.num_keypoints,
                "feature_path": str(result.output_path),
                "cached": False,
            }
        )

    for image_path in images[: args.preview_count]:
        feature_path = feature_dir / f"{image_path.stem}.npz"
        preview_path = figure_dir / f"keypoints_{image_path.stem}.jpg"
        draw_keypoints_preview(image_path, feature_path, preview_path)

    counts = np.array([item["num_keypoints"] for item in results], dtype=np.int32)
    report = {
        "scene_name": scene["scene_name"],
        "num_images": len(images),
        "feature_dir": str(feature_dir),
        "min_keypoints": int(counts.min()),
        "max_keypoints": int(counts.max()),
        "mean_keypoints": float(counts.mean()),
        "results": results,
    }
    report_path = report_dir / "feature_extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Images: {len(images)}")
    print(f"Keypoints min/mean/max: {report['min_keypoints']} / {report['mean_keypoints']:.1f} / {report['max_keypoints']}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
