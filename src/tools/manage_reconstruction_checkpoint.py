from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from src.sfm.config import load_scene_config, resolve_project_path
from src.sfm.reconstruction import load_reconstruction_state, save_reconstruction_state


STATE_FILES = (
    "reconstruction_state.json",
    "reconstruction_points.npz",
    "registered_images.npz",
)
INITIAL_FILES = (
    "initial_pair.npz",
    "initial_points.npz",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Save, restore, or list reconstruction state checkpoints.")
    parser.add_argument("--scene", required=True, help="Path to configs/scenes/<scene>.yaml")
    parser.add_argument("--action", required=True, choices=["save", "restore", "list"])
    parser.add_argument("--name", default="", help="Checkpoint name. Required for save/restore.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_scene_config(args.scene)
    scene = config["scene"]
    sparse_dir = resolve_project_path(scene["sparse_dir"])
    checkpoint_root = sparse_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    if args.action == "list":
        return list_checkpoints(checkpoint_root)
    if not args.name:
        raise ValueError("--name is required for save/restore.")
    checkpoint_dir = checkpoint_root / args.name
    if args.action == "save":
        return save_checkpoint(sparse_dir, checkpoint_dir, args.overwrite)
    if args.action == "restore":
        return restore_checkpoint(sparse_dir, checkpoint_dir)
    raise ValueError(f"Unsupported action: {args.action}")


def list_checkpoints(checkpoint_root: Path) -> int:
    checkpoints = [path for path in sorted(checkpoint_root.iterdir()) if path.is_dir()]
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_root}")
        return 0
    for checkpoint_dir in checkpoints:
        metadata_path = checkpoint_dir / "checkpoint_metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        files = ", ".join(sorted(path.name for path in checkpoint_dir.iterdir() if path.name != metadata_path.name))
        print(f"{checkpoint_dir.name}: {metadata.get('created_at', 'unknown time')} [{files}]")
    return 0


def save_checkpoint(sparse_dir: Path, checkpoint_dir: Path, overwrite: bool) -> int:
    if checkpoint_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")
        if not checkpoint_dir.is_dir():
            raise ValueError(f"Checkpoint path is not a directory: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    materialized = False
    if not any((sparse_dir / file_name).exists() for file_name in STATE_FILES):
        if all((sparse_dir / file_name).exists() for file_name in INITIAL_FILES):
            state = load_reconstruction_state(sparse_dir)
            save_reconstruction_state(state, sparse_dir)
            materialized = True

    copied = []
    for file_name in STATE_FILES:
        source = sparse_dir / file_name
        if not source.exists():
            continue
        shutil.copy2(source, checkpoint_dir / file_name)
        copied.append(file_name)
    for file_name in INITIAL_FILES:
        source = sparse_dir / file_name
        if not source.exists():
            continue
        shutil.copy2(source, checkpoint_dir / file_name)
        copied.append(file_name)
    if not copied:
        raise FileNotFoundError(f"No reconstruction state files found in {sparse_dir}")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_sparse_dir": str(sparse_dir),
        "materialized_from_initial_state": materialized,
        "files": copied,
    }
    (checkpoint_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_dir}")
    print(f"Files: {', '.join(copied)}")
    return 0


def restore_checkpoint(sparse_dir: Path, checkpoint_dir: Path) -> int:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_dir}")
    restored = []
    for file_name in (*STATE_FILES, *INITIAL_FILES):
        source = checkpoint_dir / file_name
        if not source.exists():
            continue
        shutil.copy2(source, sparse_dir / file_name)
        restored.append(file_name)
    if not restored:
        raise FileNotFoundError(f"No reconstruction state files found in checkpoint: {checkpoint_dir}")
    print(f"Restored checkpoint: {checkpoint_dir}")
    print(f"Files: {', '.join(restored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
