from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from viewer.example_catalog import ExampleSpec, load_example_specs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = Path(__file__).resolve().parent / "examples"


def prepare_example(spec: ExampleSpec, source: Path, mode: str, force: bool) -> Path:
    source = source.resolve(strict=True)
    if source.suffix.lower() != ".ply":
        raise ValueError(f"{spec.example_id}: source is not a PLY file: {source}")
    if spec.expected_ply_bytes is not None and source.stat().st_size != spec.expected_ply_bytes:
        raise ValueError(
            f"{spec.example_id}: expected {spec.expected_ply_bytes} bytes, got {source.stat().st_size}"
        )

    destination = spec.ply_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not force:
            return destination
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        relative_source = os.path.relpath(source, destination.parent)
        destination.symlink_to(relative_source)
    return destination


def resolve_source(spec: ExampleSpec, overrides: dict[str, Path]) -> Path | None:
    override = overrides.get(spec.example_id)
    if override is not None:
        return override
    return next((path for path in spec.source_candidates if path.is_file()), None)


def parse_overrides(values: list[str]) -> dict[str, Path]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --source value: {value}; expected EXAMPLE_ID=/path/to/file.ply")
        example_id, path = value.split("=", 1)
        overrides[example_id] = Path(path).expanduser()
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ignored PLY assets for viewer/examples")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="EXAMPLE_ID=PLY_PATH",
        help="Override a manifest's local source candidate; may be repeated",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    specs = load_example_specs(EXAMPLES_ROOT, PROJECT_ROOT)
    overrides = parse_overrides(args.source)
    prepared = 0
    for spec in specs:
        source = resolve_source(spec, overrides)
        if source is None:
            print(f"SKIP {spec.example_id}: no local PLY source found")
            continue
        destination = prepare_example(spec, source, args.mode, args.force)
        print(f"READY {spec.example_id}: {destination}")
        prepared += 1
    print(f"Prepared {prepared}/{len(specs)} examples")


if __name__ == "__main__":
    main()
