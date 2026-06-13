from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import draw_keypoints_preview, extract_image_features, list_images


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract local features for one configured scene.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--feature-backend", choices=["rootsift", "disk"], default=None)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute existing feature files.")
    parser.add_argument("--alpha-threshold", type=int, default=1, help="Minimum alpha treated as valid foreground.")
    parser.add_argument("--use-alpha-mask", action="store_true", default=True)
    parser.add_argument("--disable-alpha-mask", action="store_false", dest="use_alpha_mask")
    parser.add_argument("--device", default=None, help="Feature device for learned frontends: auto, cpu, cuda.")
    parser.add_argument("--disk-max-num-keypoints", type=int, default=None)
    parser.add_argument("--disk-detection-threshold", type=float, default=None)
    parser.add_argument("--disk-nms-window-size", type=int, default=None)
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

    sfm_config = default["sfm"]
    feature_backend = args.feature_backend or sfm_config.get("feature_backend", sfm_config.get("feature_type", "rootsift"))
    max_image_size = int(sfm_config.get("max_image_size", 1600))
    disk_config = sfm_config.get("disk", {})
    device = args.device or sfm_config.get("device", disk_config.get("device", "auto"))
    disk_max_num_keypoints = (
        args.disk_max_num_keypoints
        if args.disk_max_num_keypoints is not None
        else disk_config.get("max_num_keypoints", 4096)
    )
    disk_detection_threshold = (
        args.disk_detection_threshold
        if args.disk_detection_threshold is not None
        else float(disk_config.get("detection_threshold", 0.0))
    )
    disk_nms_window_size = (
        args.disk_nms_window_size
        if args.disk_nms_window_size is not None
        else int(disk_config.get("nms_window_size", 5))
    )
    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    results = []
    for image_path in tqdm(images, desc=f"Extracting {scene['scene_name']}"):
        feature_path = feature_dir / f"{image_path.stem}.npz"
        if feature_path.exists() and not args.force:
            data = np.load(feature_path)
            cached_feature_type = str(data["feature_type"]) if "feature_type" in data.files else "rootsift"
            cache_is_compatible = (
                cached_feature_type == feature_backend
                and "used_alpha_mask" in data.files
                and "mask_coverage" in data.files
            )
            if cache_is_compatible:
                results.append(
                    {
                        "image_name": str(data["image_name"]),
                        "feature_type": cached_feature_type,
                        "num_keypoints": int(data["keypoints"].shape[0]),
                        "feature_path": str(feature_path),
                        "cached": True,
                        "used_alpha_mask": bool(data["used_alpha_mask"]),
                        "mask_coverage": float(data["mask_coverage"]),
                    }
                )
                continue

        result = extract_image_features(
            image_path=image_path,
            feature_dir=feature_dir,
            feature_backend=feature_backend,
            max_image_size=max_image_size,
            use_alpha_mask=bool(args.use_alpha_mask),
            alpha_threshold=int(args.alpha_threshold),
            disk_max_num_keypoints=disk_max_num_keypoints,
            disk_detection_threshold=disk_detection_threshold,
            disk_nms_window_size=disk_nms_window_size,
            device=device,
        )
        results.append(
            {
                "image_name": result.image_name,
                "feature_type": result.feature_type,
                "image_size": result.image_size,
                "resized_size": result.resized_size,
                "scale_factor": result.scale_factor,
                "num_keypoints": result.num_keypoints,
                "used_alpha_mask": result.used_alpha_mask,
                "mask_coverage": result.mask_coverage,
                "feature_path": str(result.output_path),
                "cached": False,
            }
        )

    for image_path in images[: args.preview_count]:
        feature_path = feature_dir / f"{image_path.stem}.npz"
        preview_path = figure_dir / f"keypoints_{image_path.stem}.jpg"
        draw_keypoints_preview(image_path, feature_path, preview_path)

    counts = np.array([item["num_keypoints"] for item in results], dtype=np.int32)
    mask_coverages = np.array([item.get("mask_coverage", 1.0) for item in results], dtype=np.float32)
    masked_images = sum(1 for item in results if item.get("used_alpha_mask", False))
    report = {
        "scene_name": scene["scene_name"],
        "num_images": len(images),
        "feature_dir": str(feature_dir),
        "feature_backend": feature_backend,
        "device": device,
        "disk": {
            "max_num_keypoints": disk_max_num_keypoints,
            "detection_threshold": disk_detection_threshold,
            "nms_window_size": disk_nms_window_size,
        },
        "use_alpha_mask": bool(args.use_alpha_mask),
        "alpha_threshold": int(args.alpha_threshold),
        "num_alpha_masked_images": int(masked_images),
        "mean_mask_coverage": float(mask_coverages.mean()),
        "min_keypoints": int(counts.min()),
        "max_keypoints": int(counts.max()),
        "mean_keypoints": float(counts.mean()),
        "results": results,
    }
    report_path = report_dir / "feature_extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Images: {len(images)}")
    print(f"Feature backend: {feature_backend}")
    print(f"Keypoints min/mean/max: {report['min_keypoints']} / {report['mean_keypoints']:.1f} / {report['max_keypoints']}")
    print(f"Alpha-masked images: {report['num_alpha_masked_images']} / {len(images)}")
    print(f"Mean mask coverage: {report['mean_mask_coverage']:.3f}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
