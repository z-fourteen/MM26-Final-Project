from __future__ import annotations

import argparse
import json

import numpy as np

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate extracted RootSIFT feature files.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    missing = []
    invalid = []
    counts = []
    mean_norms = []
    fields = None

    for image_path in images:
        feature_path = feature_dir / f"{image_path.stem}.npz"
        if not feature_path.exists():
            missing.append(image_path.name)
            continue
        data = np.load(feature_path)
        fields = data.files
        keypoints = data["keypoints"]
        descriptors = data["descriptors"]
        counts.append(int(keypoints.shape[0]))
        if (
            descriptors.ndim != 2
            or descriptors.shape[1] != 128
            or descriptors.shape[0] != keypoints.shape[0]
            or np.isnan(descriptors).any()
            or (descriptors < 0).any()
        ):
            invalid.append(image_path.name)
        if descriptors.size:
            mean_norms.append(float(np.linalg.norm(descriptors, axis=1).mean()))

    report = {
        "scene_name": scene["scene_name"],
        "num_images": len(images),
        "num_feature_files": len(images) - len(missing),
        "fields": fields,
        "missing": missing,
        "invalid": invalid,
        "min_keypoints": min(counts) if counts else 0,
        "mean_keypoints": float(np.mean(counts)) if counts else 0.0,
        "max_keypoints": max(counts) if counts else 0,
        "mean_descriptor_l2_norm": float(np.mean(mean_norms)) if mean_norms else 0.0,
    }
    report_path = report_dir / "feature_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {report['scene_name']}")
    print(f"Images/features: {report['num_images']} / {report['num_feature_files']}")
    print(f"Keypoints min/mean/max: {report['min_keypoints']} / {report['mean_keypoints']:.1f} / {report['max_keypoints']}")
    print(f"Mean descriptor L2 norm: {report['mean_descriptor_l2_norm']:.6f}")
    print(f"Missing: {len(missing)}")
    print(f"Invalid: {len(invalid)}")
    print(f"Report: {report_path}")

    return 0 if not missing and not invalid else 1


if __name__ == "__main__":
    raise SystemExit(main())
