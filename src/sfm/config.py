from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.is_absolute():
        yaml_path = project_root() / yaml_path
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return project_root() / value


def load_scene_config(scene_config_path: str | Path) -> dict[str, Any]:
    default_config = load_yaml(project_root() / "configs" / "default.yaml")
    scene_config = load_yaml(scene_config_path)
    return {
        "default": default_config,
        "scene": scene_config,
    }
