"""Helpers for importing/exporting upstream dataset_config.toml files."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping

import toml


DATASET_CONFIG_FILENAME = "dataset_config.toml"

KNOWN_GENERAL_KEYS = (
    "resolution",
    "caption_extension",
    "batch_size",
    "enable_bucket",
    "bucket_no_upscale",
    "num_repeats",
)

KNOWN_DATASET_KEYS = (
    "resolution",
    "image_directory",
    "cache_directory",
    "image_jsonl_file",
    "video_directory",
    "video_jsonl_file",
    "control_directory",
    "caption_extension",
    "batch_size",
    "enable_bucket",
    "bucket_no_upscale",
    "num_repeats",
    "multiple_target",
    "no_resize_control",
    "control_resolution",
    "flux_kontext_no_resize_control",
    "qwen_image_edit_no_resize_control",
    "qwen_image_edit_control_resolution",
    "target_frames",
    "frame_extraction",
    "frame_stride",
    "frame_sample",
    "max_frames",
    "source_fps",
    "fp_latent_window_size",
    "fp_1f_clean_indices",
    "fp_1f_target_index",
    "fp_1f_no_post",
)

_ASSIGNMENT_RE = re.compile(r"^([A-Za-z0-9_-]+)\s*=")


def get_default_project_dir() -> Path:
    """Return the current GUI project root."""
    return Path(__file__).resolve().parents[2]


def get_default_dataset_config_path(project_dir: str | Path | None = None) -> Path:
    """Return the default project-local dataset_config.toml path."""
    base_dir = Path(project_dir) if project_dir is not None else get_default_project_dir()
    return base_dir / DATASET_CONFIG_FILENAME


def _copy_mapping(raw_value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(raw_value) if isinstance(raw_value, Mapping) else {}


def _split_known_fields(raw_value: Mapping[str, Any] | None, supported_keys: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    known: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    key_set = set(supported_keys)

    for key, value in _copy_mapping(raw_value).items():
        target = known if key in key_set else extra
        target[key] = copy.deepcopy(value)

    return known, extra


def _normalize_dataset_extra(raw_extra: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = _copy_mapping(raw_extra)
    datasets_extra = raw.get("datasets")
    normalized_datasets = copy.deepcopy(datasets_extra) if isinstance(datasets_extra, list) else []
    return {
        "root": copy.deepcopy(raw.get("root", {})) if isinstance(raw.get("root"), Mapping) else {},
        "general": copy.deepcopy(raw.get("general", {})) if isinstance(raw.get("general"), Mapping) else {},
        "datasets": normalized_datasets,
    }


def _is_meaningful_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _scan_loose_extra_fields(raw_text: str) -> tuple[dict[str, Any], list[tuple[int, str]]]:
    """Recover extra fields with a forgiving scope heuristic for legacy loose TOML."""
    extras: dict[str, Any] = {
        "root": {},
        "general": {},
        "datasets": [],
    }
    promoted_dataset_keys: list[tuple[int, str]] = []
    scope = "root"
    dataset_index = -1
    blank_after_dataset_entry = False

    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            if scope == "dataset":
                blank_after_dataset_entry = True
            continue

        if stripped.startswith("#"):
            continue

        if stripped.startswith("[[") and stripped.endswith("]]"):
            header = stripped[2:-2].strip()
            if header == "datasets":
                scope = "dataset"
                dataset_index += 1
                extras["datasets"].append({})
            else:
                scope = "root"
            blank_after_dataset_entry = False
            continue

        if stripped.startswith("[") and stripped.endswith("]"):
            header = stripped[1:-1].strip()
            scope = "general" if header == "general" else "root"
            blank_after_dataset_entry = False
            continue

        match = _ASSIGNMENT_RE.match(stripped)
        if match is None:
            blank_after_dataset_entry = False
            continue

        key = match.group(1)
        value = toml.loads(stripped).get(key)

        if scope == "general":
            if key not in KNOWN_GENERAL_KEYS:
                extras["general"][key] = copy.deepcopy(value)
        elif scope == "dataset":
            while len(extras["datasets"]) <= dataset_index:
                extras["datasets"].append({})

            if key not in KNOWN_DATASET_KEYS and blank_after_dataset_entry:
                extras["root"][key] = copy.deepcopy(value)
                promoted_dataset_keys.append((dataset_index, key))
                scope = "root"
            elif key not in KNOWN_DATASET_KEYS:
                extras["datasets"][dataset_index][key] = copy.deepcopy(value)
        else:
            extras["root"][key] = copy.deepcopy(value)

        blank_after_dataset_entry = False

    return extras, promoted_dataset_keys


def load_dataset_config_import(path: str | Path) -> dict[str, Any]:
    """Load upstream dataset_config.toml into canonical dataset/interoperability slices."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    raw_config = toml.loads(raw_text)

    root = copy.deepcopy(raw_config)
    general_known, general_extra = _split_known_fields(root.pop("general", {}), KNOWN_GENERAL_KEYS)

    datasets_known: list[dict[str, Any]] = []
    datasets_extra: list[dict[str, Any]] = []
    for raw_dataset in root.pop("datasets", []):
        known_dataset, extra_dataset = _split_known_fields(raw_dataset, KNOWN_DATASET_KEYS)
        datasets_known.append(known_dataset)
        datasets_extra.append(extra_dataset)

    loose_extra, promoted_dataset_keys = _scan_loose_extra_fields(raw_text)
    root.update(loose_extra["root"])
    general_extra.update(loose_extra["general"])

    for index, dataset_extra in enumerate(loose_extra["datasets"]):
        while len(datasets_extra) <= index:
            datasets_extra.append({})
        datasets_extra[index].update(dataset_extra)

    for dataset_index, key in promoted_dataset_keys:
        if dataset_index < len(datasets_extra):
            datasets_extra[dataset_index].pop(key, None)

    return {
        "dataset": {
            "general": general_known,
            "datasets": datasets_known,
        },
        "interop": {
            "dataset_extra": {
                "root": root,
                "general": general_extra,
                "datasets": datasets_extra,
            },
        },
    }


def build_dataset_config(project_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build an upstream-compatible dataset_config payload from canonical project_config content."""
    dataset_section = _copy_mapping(project_config.get("dataset"))
    interop = _copy_mapping(project_config.get("interop"))
    dataset_extra = _normalize_dataset_extra(interop.get("dataset_extra"))

    payload = copy.deepcopy(dataset_extra["root"])

    general_payload = copy.deepcopy(dataset_extra["general"])
    for key, value in _copy_mapping(dataset_section.get("general")).items():
        if _is_meaningful_value(value):
            general_payload[key] = copy.deepcopy(value)
        else:
            general_payload.pop(key, None)
    if general_payload:
        payload["general"] = general_payload

    datasets_payload: list[dict[str, Any]] = []
    for index, dataset in enumerate(dataset_section.get("datasets", [])):
        merged_dataset = {}
        if index < len(dataset_extra["datasets"]) and isinstance(dataset_extra["datasets"][index], Mapping):
            merged_dataset.update(copy.deepcopy(dataset_extra["datasets"][index]))
        for key, value in _copy_mapping(dataset).items():
            if _is_meaningful_value(value):
                merged_dataset[key] = copy.deepcopy(value)
            else:
                merged_dataset.pop(key, None)
        if merged_dataset:
            datasets_payload.append(merged_dataset)
    if datasets_payload:
        payload["datasets"] = datasets_payload

    return payload


def export_dataset_config(project_config: Mapping[str, Any], output_path: str | Path) -> Path:
    """Export canonical dataset state to upstream dataset_config.toml."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        toml.dump(build_dataset_config(project_config), handle)
    return path


def summarize_dataset_state(project_config: Mapping[str, Any], project_dir: str | Path | None = None) -> dict[str, str]:
    """Return a small summary for readonly consumers like cache/train."""
    dataset_section = _copy_mapping(project_config.get("dataset"))
    general = _copy_mapping(dataset_section.get("general"))
    datasets = dataset_section.get("datasets", [])

    resolution = general.get("resolution")
    if isinstance(resolution, list) and len(resolution) == 2:
        resolution_text = f"{resolution[0]}x{resolution[1]}"
    else:
        resolution_text = "-"

    import_sources = _copy_mapping(_copy_mapping(project_config.get("interop")).get("import_sources"))
    source_path = import_sources.get("dataset_config", "")

    return {
        "dataset_count": str(len(datasets)) if isinstance(datasets, list) else "0",
        "resolution": resolution_text,
        "source_path": str(source_path or get_default_dataset_config_path(project_dir)),
        "has_dataset": "true" if isinstance(datasets, list) and len(datasets) > 0 else "false",
    }
