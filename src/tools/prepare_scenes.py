from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.sfm.config import project_root


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CANONICAL_SUBDIRS = [
    "images",
    "features",
    "matches",
    "verified",
    "tracks",
    "sparse/0",
    "undistorted",
]


def scene_name_from_dataset(dataset_name: str) -> str:
    return dataset_name.replace("-", "_").replace(" ", "_").lower()


def find_image_dir(dataset_dir: Path) -> Path | None:
    candidates = []
    for directory in [dataset_dir, *dataset_dir.rglob("*")]:
        if not directory.is_dir():
            continue
        image_count = sum(
            1
            for file in directory.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count:
            candidates.append((image_count, directory))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], len(item[1].parts)))
    return candidates[0][1]


def ensure_scene_dirs(scene_dir: Path) -> None:
    for rel_path in CANONICAL_SUBDIRS:
        directory = scene_dir / rel_path
        directory.mkdir(parents=True, exist_ok=True)
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")


def write_scene_config(config_path: Path, scene_name: str, dataset_name: str, subset: dict) -> None:
    config = {
        "scene_name": scene_name,
        "dataset_name": dataset_name,
        "image_dir": f"data/scenes/{scene_name}/images",
        "feature_dir": f"data/scenes/{scene_name}/features",
        "match_dir": f"data/scenes/{scene_name}/matches",
        "verified_dir": f"data/scenes/{scene_name}/verified",
        "track_dir": f"data/scenes/{scene_name}/tracks",
        "sparse_dir": f"data/scenes/{scene_name}/sparse/0",
        "undistorted_dir": f"data/scenes/{scene_name}/undistorted",
        "output_dir": f"outputs/{scene_name}",
        "subset": subset,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def prepare_outputs(scene_name: str) -> None:
    root = project_root()
    for rel_path in ["logs", "figures", "reports"]:
        directory = root / "outputs" / scene_name / rel_path
        directory.mkdir(parents=True, exist_ok=True)
        gitkeep = directory / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")


def copy_subset(source_dir: Path, target_dir: Path, num_images: int) -> int:
    import shutil

    images = sorted(
        file
        for file in source_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = images[:num_images]
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in selected:
        target = target_dir / source.name
        if not target.exists():
            shutil.copy2(source, target)
            copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare canonical scene folders from raw datasets.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--subset-source", default="south-building")
    parser.add_argument("--subset-name", default="south_building_small")
    parser.add_argument("--subset-size", type=int, default=50)
    args = parser.parse_args()

    root = project_root()
    raw_root = root / args.raw_root
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw dataset root not found: {raw_root}")

    prepared = []
    for dataset_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        scene_name = scene_name_from_dataset(dataset_name)
        image_dir = find_image_dir(dataset_dir)
        scene_dir = root / "data" / "scenes" / scene_name
        ensure_scene_dirs(scene_dir)
        prepare_outputs(scene_name)
        write_scene_config(
            root / "configs" / "scenes" / f"{scene_name}.yaml",
            scene_name,
            dataset_name,
            {"enabled": False},
        )
        prepared.append((scene_name, str(image_dir) if image_dir else "NO_IMAGES_FOUND"))

    subset_dataset_dir = raw_root / args.subset_source
    subset_image_dir = find_image_dir(subset_dataset_dir)
    if subset_image_dir is None:
        raise FileNotFoundError(f"No images found for subset source: {subset_dataset_dir}")

    subset_scene_dir = root / "data" / "scenes" / args.subset_name
    ensure_scene_dirs(subset_scene_dir)
    prepare_outputs(args.subset_name)
    copied = copy_subset(subset_image_dir, subset_scene_dir / "images", args.subset_size)
    write_scene_config(
        root / "configs" / "scenes" / f"{args.subset_name}.yaml",
        args.subset_name,
        args.subset_source,
        {
            "enabled": True,
            "target_num_images": args.subset_size,
            "selection": "filename_sorted_first_n",
            "source_image_dir": str(subset_image_dir.relative_to(root)),
        },
    )

    print("Prepared scenes:")
    for scene_name, image_dir in prepared:
        print(f"  {scene_name:<24} source={image_dir}")
    print("Images are not copied automatically. Put reproducible image copies under data/scenes/<scene>/images/.")
    print(f"Subset {args.subset_name}: copied {copied} new images from {subset_image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
