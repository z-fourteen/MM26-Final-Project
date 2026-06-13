from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.features import list_images
from src.sfm.matching import (
    draw_match_preview,
    load_features,
    match_feature_pair_backend,
    pair_output_path,
    save_pair_matches,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Match local features for a configured scene.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--matcher-backend", choices=["rootsift_bf", "lightglue"], default=None)
    parser.add_argument("--strategy", default="exhaustive", choices=["exhaustive", "sequential", "retrieval"])
    parser.add_argument("--sequential-window", type=int, default=10)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--retrieval-num-words", type=int, default=256)
    parser.add_argument("--retrieval-max-descriptors-per-image", type=int, default=1000)
    parser.add_argument("--retrieval-sequential-window", type=int, default=2)
    parser.add_argument("--retrieval-seed", type=int, default=13)
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for debugging; 0 means all pairs.")
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="Recompute existing match files.")
    parser.add_argument("--device", default=None, help="Matcher device for learned matchers: auto, cpu, cuda.")
    parser.add_argument("--lightglue-filter-threshold", type=float, default=None)
    parser.add_argument("--lightglue-depth-confidence", type=float, default=None)
    parser.add_argument("--lightglue-width-confidence", type=float, default=None)
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

    sfm_config = default["sfm"]
    feature_backend = sfm_config.get("feature_backend", sfm_config.get("feature_type", "rootsift"))
    matcher_backend = args.matcher_backend or sfm_config.get("matcher_backend", "rootsift_bf")
    ratio_test = float(sfm_config.get("ratio_test", 0.8))
    mutual_check = bool(sfm_config.get("mutual_check", True))
    min_num_matches = int(sfm_config.get("min_num_matches", 30))
    lightglue_config = sfm_config.get("lightglue", {})
    device = args.device or sfm_config.get("device", lightglue_config.get("device", "auto"))
    lightglue_features = str(lightglue_config.get("features", feature_backend))
    lightglue_filter_threshold = (
        args.lightglue_filter_threshold
        if args.lightglue_filter_threshold is not None
        else float(lightglue_config.get("filter_threshold", 0.1))
    )
    lightglue_depth_confidence = (
        args.lightglue_depth_confidence
        if args.lightglue_depth_confidence is not None
        else float(lightglue_config.get("depth_confidence", 0.95))
    )
    lightglue_width_confidence = (
        args.lightglue_width_confidence
        if args.lightglue_width_confidence is not None
        else float(lightglue_config.get("width_confidence", 0.99))
    )

    images = list_images(image_dir)
    if len(images) < 2:
        raise ValueError(f"Need at least two images for matching: {image_dir}")

    feature_paths = [feature_dir / f"{image_path.stem}.npz" for image_path in images]
    missing_features = [str(path) for path in feature_paths if not path.exists()]
    if missing_features:
        raise FileNotFoundError("Missing feature files:\n" + "\n".join(missing_features[:10]))

    features = [load_features(path) for path in feature_paths]
    incompatible_features = [
        f"{feature.image_name}: {feature.feature_type}"
        for feature in features
        if matcher_backend == "lightglue" and feature.feature_type != lightglue_features
    ]
    if incompatible_features:
        raise ValueError(
            "LightGlue matcher requires matching feature files. "
            f"Expected {lightglue_features}; examples:\n" + "\n".join(incompatible_features[:10])
        )
    if args.strategy == "retrieval" and matcher_backend == "lightglue":
        raise ValueError("Retrieval pair generation is currently implemented for descriptor BoW; use exhaustive or sequential with LightGlue.")
    image_by_name = {image_path.name: image_path for image_path in images}
    feature_by_name = {feature.image_name: feature for feature in features}
    pair_scores: dict[tuple[int, int], float] = {}
    retrieval_summary = None
    if args.strategy == "retrieval":
        pairs, pair_scores, retrieval_summary = build_retrieval_pairs(
            features,
            top_k=args.retrieval_top_k,
            num_words=args.retrieval_num_words,
            max_descriptors_per_image=args.retrieval_max_descriptors_per_image,
            sequential_window=args.retrieval_sequential_window,
            seed=args.retrieval_seed,
        )
    else:
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
            cached_matcher = str(data["matcher_backend"]) if "matcher_backend" in data.files else "rootsift_bf"
            cached_feature_type1 = str(data["feature_type1"]) if "feature_type1" in data.files else "rootsift"
            cached_feature_type2 = str(data["feature_type2"]) if "feature_type2" in data.files else "rootsift"
            cache_is_compatible = (
                cached_matcher == matcher_backend
                and cached_feature_type1 == features[index1].feature_type
                and cached_feature_type2 == features[index2].feature_type
            )
            if cache_is_compatible:
                num_matches = int(data["matches"].shape[0])
            else:
                result = match_feature_pair_backend(
                    features[index1],
                    features[index2],
                    matcher_backend=matcher_backend,
                    ratio_test=ratio_test,
                    mutual_check=mutual_check,
                    lightglue_features=lightglue_features,
                    lightglue_filter_threshold=lightglue_filter_threshold,
                    lightglue_depth_confidence=lightglue_depth_confidence,
                    lightglue_width_confidence=lightglue_width_confidence,
                    device=device,
                )
                save_pair_matches(result, output_path)
                num_matches = result.num_matches
        else:
            result = match_feature_pair_backend(
                features[index1],
                features[index2],
                matcher_backend=matcher_backend,
                ratio_test=ratio_test,
                mutual_check=mutual_check,
                lightglue_features=lightglue_features,
                lightglue_filter_threshold=lightglue_filter_threshold,
                lightglue_depth_confidence=lightglue_depth_confidence,
                lightglue_width_confidence=lightglue_width_confidence,
                device=device,
            )
            save_pair_matches(result, output_path)
            num_matches = result.num_matches

        summaries.append(
            {
                "image_name1": features[index1].image_name,
                "image_name2": features[index2].image_name,
                "match_path": str(output_path),
                "num_matches": num_matches,
                "retrieval_score": pair_scores.get((index1, index2)),
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
        "feature_backend": feature_backend,
        "matcher_backend": matcher_backend,
        "device": device,
        "strategy": args.strategy,
        "sequential_window": args.sequential_window,
        "retrieval": retrieval_summary,
        "num_images": len(images),
        "num_pairs": len(pairs),
        "ratio_test": ratio_test,
        "mutual_check": mutual_check,
        "lightglue": {
            "features": lightglue_features,
            "filter_threshold": lightglue_filter_threshold,
            "depth_confidence": lightglue_depth_confidence,
            "width_confidence": lightglue_width_confidence,
        },
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
    print(f"Matcher backend: {matcher_backend}")
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


def build_retrieval_pairs(
    features: list,
    top_k: int,
    num_words: int,
    max_descriptors_per_image: int,
    sequential_window: int,
    seed: int,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float], dict]:
    if top_k <= 0:
        raise ValueError("--retrieval-top-k must be positive")
    sampled = sample_descriptors_by_image(features, max_descriptors_per_image=max_descriptors_per_image, seed=seed)
    vocabulary, vocabulary_summary = build_visual_vocabulary(sampled, num_words=num_words, seed=seed)
    image_vectors = build_tfidf_image_vectors(sampled, vocabulary)
    similarities = image_vectors @ image_vectors.T
    np.fill_diagonal(similarities, -np.inf)

    pair_set: set[tuple[int, int]] = set()
    pair_scores: dict[tuple[int, int], float] = {}
    for image_idx in range(len(features)):
        candidate_indices = np.argsort(similarities[image_idx])[::-1][: min(top_k, len(features) - 1)]
        for candidate_idx in candidate_indices:
            if candidate_idx == image_idx or not np.isfinite(similarities[image_idx, candidate_idx]):
                continue
            pair = tuple(sorted((int(image_idx), int(candidate_idx))))
            pair_set.add(pair)
            pair_scores[pair] = max(pair_scores.get(pair, -np.inf), float(similarities[image_idx, candidate_idx]))

    if sequential_window > 0:
        for pair in build_pair_indices(len(features), strategy="sequential", sequential_window=sequential_window):
            pair_set.add(pair)
            pair_scores.setdefault(pair, 0.0)

    pairs = sorted(pair_set, key=lambda pair: (-pair_scores.get(pair, 0.0), pair[0], pair[1]))
    summary = {
        "type": "npz_bow_tfidf",
        "top_k": int(top_k),
        "num_words_requested": int(num_words),
        "num_words_used": int(vocabulary.shape[0]),
        "max_descriptors_per_image": int(max_descriptors_per_image),
        "sequential_window": int(sequential_window),
        "seed": int(seed),
        "num_pairs": len(pairs),
        **vocabulary_summary,
    }
    return pairs, pair_scores, summary


def sample_descriptors_by_image(features: list, max_descriptors_per_image: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    sampled = []
    for image_features in features:
        descriptors = image_features.descriptors.astype(np.float32, copy=False)
        if descriptors.size == 0:
            descriptor_dim = int(descriptors.shape[1]) if descriptors.ndim == 2 else 0
            sampled.append(descriptors.reshape(0, descriptor_dim))
            continue
        limit = min(max(int(max_descriptors_per_image), 1), int(descriptors.shape[0]))
        if descriptors.shape[0] > limit:
            indices = rng.choice(descriptors.shape[0], size=limit, replace=False)
            descriptors = descriptors[indices]
        sampled.append(l2_normalize_rows(descriptors.astype(np.float32, copy=False)))
    return sampled


def build_visual_vocabulary(sampled_descriptors: list[np.ndarray], num_words: int, seed: int) -> tuple[np.ndarray, dict]:
    descriptor_pool = np.vstack([descriptors for descriptors in sampled_descriptors if descriptors.size])
    if descriptor_pool.size == 0:
        raise ValueError("No descriptors available for retrieval matching")
    num_words_used = min(max(int(num_words), 1), int(descriptor_pool.shape[0]))
    if num_words_used == 1:
        vocabulary = descriptor_pool.mean(axis=0, keepdims=True)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        _compactness, _labels, vocabulary = cv2.kmeans(
            descriptor_pool.astype(np.float32, copy=False),
            num_words_used,
            None,
            criteria,
            3,
            cv2.KMEANS_PP_CENTERS,
        )
    vocabulary = l2_normalize_rows(vocabulary.astype(np.float32, copy=False))
    return vocabulary, {"num_descriptors_for_vocabulary": int(descriptor_pool.shape[0])}


def build_tfidf_image_vectors(sampled_descriptors: list[np.ndarray], vocabulary: np.ndarray) -> np.ndarray:
    histograms = np.zeros((len(sampled_descriptors), vocabulary.shape[0]), dtype=np.float32)
    for image_idx, descriptors in enumerate(sampled_descriptors):
        if descriptors.size == 0:
            continue
        word_indices = assign_visual_words(descriptors, vocabulary)
        histograms[image_idx] = np.bincount(word_indices, minlength=vocabulary.shape[0]).astype(np.float32)
    document_frequency = np.count_nonzero(histograms > 0, axis=0)
    idf = np.log((len(sampled_descriptors) + 1.0) / (document_frequency + 1.0)) + 1.0
    vectors = histograms * idf.astype(np.float32)
    return l2_normalize_rows(vectors)


def assign_visual_words(descriptors: np.ndarray, vocabulary: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    assignments = []
    vocab_sq = np.sum(vocabulary * vocabulary, axis=1)
    for start in range(0, descriptors.shape[0], chunk_size):
        chunk = descriptors[start : start + chunk_size]
        chunk_sq = np.sum(chunk * chunk, axis=1, keepdims=True)
        distances = chunk_sq + vocab_sq[None, :] - 2.0 * (chunk @ vocabulary.T)
        assignments.append(np.argmin(distances, axis=1).astype(np.int32))
    return np.concatenate(assignments) if assignments else np.empty((0,), dtype=np.int32)


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


if __name__ == "__main__":
    raise SystemExit(main())
