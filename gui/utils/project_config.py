"""Canonical musubi project config helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import toml


PROJECT_CONFIG_FILENAME = "musubi_project.toml"
PROJECT_CONFIG_VERSION = 1

_DEFAULT_PROJECT_CONFIG: dict[str, Any] = {
    "schema_version": PROJECT_CONFIG_VERSION,
    "project": {
        "name": "",
        "root_dir": "",
        "created_by": "qinglong_gui",
    },
    "gui": {
        "active_page": "train",
        "last_workflow": "train",
        "selected_preset": "",
        "recent_import_paths": {
            "dataset_config": "",
            "config_cache": "",
            "config_train": "",
            "config_generate": "",
        },
    },
    "dataset": {
        "general": {},
        "datasets": [],
    },
    "workflows": {
        "cache": {},
        "train": {},
        "generate": {},
    },
    "interop": {
        "dataset_extra": {},
        "workflow_extra": {
            "cache": {},
            "train": {},
            "generate": {},
        },
        "import_sources": {
            "dataset_config": "",
            "config_cache": "",
            "config_train": "",
            "config_generate": "",
        },
    },
}


def create_default_project_config() -> dict[str, Any]:
    """Return a deep-copied canonical default config."""
    return copy.deepcopy(_DEFAULT_PROJECT_CONFIG)


def resolve_project_config_path(project_dir_or_file: str | Path) -> Path:
    """Resolve a project directory or a direct TOML path to musubi_project.toml."""
    path = Path(project_dir_or_file)
    if path.suffix.lower() == ".toml":
        return path
    return path / PROJECT_CONFIG_FILENAME


def _merge_with_defaults(default_value: Any, raw_value: Any) -> Any:
    if isinstance(default_value, dict):
        merged: dict[str, Any] = {}
        raw_dict = raw_value if isinstance(raw_value, dict) else {}

        for key, nested_default in default_value.items():
            merged[key] = _merge_with_defaults(nested_default, raw_dict.get(key))

        for key, nested_raw in raw_dict.items():
            if key not in merged:
                merged[key] = copy.deepcopy(nested_raw)

        return merged

    if isinstance(default_value, list):
        return copy.deepcopy(raw_value) if isinstance(raw_value, list) else copy.deepcopy(default_value)

    if raw_value is None:
        return copy.deepcopy(default_value)

    return copy.deepcopy(raw_value)


def normalize_project_config(raw_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize arbitrary config content into the canonical project schema."""
    if not isinstance(raw_config, Mapping):
        return create_default_project_config()

    normalized = _merge_with_defaults(_DEFAULT_PROJECT_CONFIG, dict(raw_config))

    schema_version = raw_config.get("schema_version", PROJECT_CONFIG_VERSION)
    normalized["schema_version"] = schema_version if isinstance(schema_version, int) else PROJECT_CONFIG_VERSION
    return normalized


def load_project_config(project_dir_or_file: str | Path) -> dict[str, Any]:
    """Load and normalize musubi_project.toml content."""
    path = resolve_project_config_path(project_dir_or_file)
    if not path.exists():
        return create_default_project_config()

    with open(path, "r", encoding="utf-8") as handle:
        raw_config = toml.load(handle)

    return normalize_project_config(raw_config)


def save_project_config(project_dir_or_file: str | Path, config: Mapping[str, Any] | None) -> Path:
    """Normalize and persist musubi_project.toml content."""
    path = resolve_project_config_path(project_dir_or_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_project_config(config)
    with open(path, "w", encoding="utf-8") as handle:
        toml.dump(normalized, handle)

    return path

