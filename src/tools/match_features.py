from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pycolmap
from tqdm import tqdm

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import (
    draw_match_preview,
    load_features,
    match_feature_pair,
    pair_output_path,
    save_pair_matches,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Match RootSIFT features for a configured scene.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--strategy", default="exhaustive", choices=["exhaustive", "sequential", "vocabulary_tree"])
    parser.add_argument("--sequential-window", type=int, default=10)
    parser.add_argument("--database-path", default="", help="COLMAP database path for vocabulary_tree matching.")
    parser.add_argument("--vocab-tree-path", default="", help="COLMAP vocabulary tree path.")
    parser.add_argument("--vocab-tree-num-images", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for debugging; 0 means all pairs.")
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute existing match files.")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    default = config["default"]
    scene = config["scene"]

    image_dir = resolve_project_path(scene["image_dir"])
    feature_dir = resolve_project_path(scene["feature_dir"])
    match_dir = resolve_project_path(scene["match_dir"])
    output_dir = resolve_project_path(scene["output_dir"])
    figure_dir = output_dir / "figures"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    ratio_test = float(default["sfm"].get("ratio_test", 0.8))
    mutual_check = bool(default["sfm"].get("mutual_check", True))
    min_num_matches = int(default["sfm"].get("min_num_matches", 30))

    images = list_images(image_dir)
    if len(images) < 2:
        raise ValueError(f"Need at least two images for matching: {image_dir}")
    if args.strategy == "vocabulary_tree":
        return run_vocab_tree_matching(args, scene)

    feature_paths = [feature_dir / f"{image_path.stem}.npz" for image_path in images]
    missing_features = [str(path) for path in feature_paths if not path.exists()]
    if missing_features:
        raise FileNotFoundError("Missing feature files:\n" + "\n".join(missing_features[:10]))

    features = [load_features(path) for path in feature_paths]
    image_by_name = {image_path.name: image_path for image_path in images}
    feature_by_name = {feature.image_name: feature for feature in features}
    pairs = build_pair_indices(len(features), strategy=args.strategy, sequential_window=args.sequential_window)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    summaries = []
    for index1, index2 in tqdm(pairs, desc=f"Matching {scene['scene_name']}"):
        feature_path1 = feature_paths[index1]
        feature_path2 = feature_paths[index2]
        output_path = pair_output_path(match_dir, feature_path1, feature_path2)

        if output_path.exists() and not args.force:
            data = np.load(output_path)
            num_matches = int(data["matches"].shape[0])
        else:
            result = match_feature_pair(
                features[index1],
                features[index2],
                ratio_test=ratio_test,
                mutual_check=mutual_check,
            )
            save_pair_matches(result, output_path)
            num_matches = result.num_matches

        summaries.append(
            {
                "image_name1": features[index1].image_name,
                "image_name2": features[index2].image_name,
                "match_path": str(output_path),
                "num_matches": num_matches,
            }
        )

    counts = np.array([item["num_matches"] for item in summaries], dtype=np.int32)
    preview_candidates = sorted(summaries, key=lambda item: item["num_matches"], reverse=True)[: args.preview_count]
    for item in preview_candidates:
        feature1 = feature_by_name[item["image_name1"]]
        feature2 = feature_by_name[item["image_name2"]]
        match_data = np.load(item["match_path"])
        preview_path = figure_dir / f"matches_{Path(item['image_name1']).stem}__{Path(item['image_name2']).stem}.jpg"
        draw_match_preview(
            image_by_name[item["image_name1"]],
            image_by_name[item["image_name2"]],
            feature1,
            feature2,
            match_data["matches"],
            preview_path,
        )

    report = {
        "scene_name": scene["scene_name"],
        "strategy": args.strategy,
        "sequential_window": args.sequential_window,
        "num_images": len(images),
        "num_pairs": len(pairs),
        "ratio_test": ratio_test,
        "mutual_check": mutual_check,
        "min_num_matches": min_num_matches,
        "min_matches": int(counts.min()) if len(counts) else 0,
        "mean_matches": float(counts.mean()) if len(counts) else 0.0,
        "max_matches": int(counts.max()) if len(counts) else 0,
        "num_pairs_ge_min_matches": int(np.sum(counts >= min_num_matches)) if len(counts) else 0,
        "pairs": summaries,
    }
    report_path = report_dir / "matching_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Scene: {scene['scene_name']}")
    print(f"Images: {len(images)}")
    print(f"Pairs: {len(pairs)}")
    print(f"Matches min/mean/max: {report['min_matches']} / {report['mean_matches']:.1f} / {report['max_matches']}")
    print(f"Pairs >= {min_num_matches}: {report['num_pairs_ge_min_matches']}")
    print(f"Report: {report_path}")
    return 0


def build_pair_indices(num_images: int, strategy: str, sequential_window: int) -> list[tuple[int, int]]:
    if strategy == "exhaustive":
        return list(itertools.combinations(range(num_images), 2))
    if strategy == "sequential":
        window = max(int(sequential_window), 1)
        return [
            (idx1, idx2)
            for idx1 in range(num_images)
            for idx2 in range(idx1 + 1, min(num_images, idx1 + window + 1))
        ]
    raise ValueError(f"Unsupported NPZ matching strategy: {strategy}")


def run_vocab_tree_matching(args: argparse.Namespace, scene: dict) -> int:
    if not args.database_path:
        raise ValueError("--database-path is required for --strategy vocabulary_tree")
    if not args.vocab_tree_path:
        raise ValueError("--vocab-tree-path is required for --strategy vocabulary_tree")
    pairing_options = pycolmap.VocabTreePairingOptions()
    if hasattr(pairing_options, "vocab_tree_path"):
        pairing_options.vocab_tree_path = args.vocab_tree_path
    if hasattr(pairing_options, "num_images"):
        pairing_options.num_images = int(args.vocab_tree_num_images)
    pycolmap.match_vocabtree(args.database_path, pairing_options=pairing_options)
    print(f"Scene: {scene['scene_name']}")
    print("Strategy: vocabulary_tree")
    print(f"Database: {args.database_path}")
    print(f"Vocab tree: {args.vocab_tree_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
