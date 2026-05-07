"""Step 1 dataset page with overview, tagging, and dataset_config.toml ownership."""

from __future__ import annotations

import copy
import socket
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from nicegui import ui

from components.path_selector import create_path_selector
from theme import COLORS, get_classes
from utils.config_manager import config_manager
from utils.dataset_config import (
    export_dataset_config,
    get_default_dataset_config_path,
    get_default_project_dir,
    load_dataset_config_import,
    summarize_dataset_state,
)
from utils.i18n import t


DATASET_PRESET_DIRNAME = "toml"
CURRENT_PROJECT_PRESET_ID = "__current_project__"
QINGLONG_CAPTIONS_DIRNAME = "qinglong-captions"
QINGLONG_INSTALL_SCRIPT = "1.install-uv-qinglong.ps1"
QINGLONG_START_GUI_SCRIPT = "start_gui.ps1"
QINGLONG_START_GUI_PORT = 7899


def _format_dataset_preset_label(path: Path) -> str:
    return path.stem


def discover_dataset_presets(preset_dir: str | Path | None = None) -> dict[str, str]:
    base_dir = Path(preset_dir) if preset_dir is not None else get_default_project_dir() / DATASET_PRESET_DIRNAME
    if not base_dir.exists():
        return {}

    presets: dict[str, str] = {}
    for path in sorted(base_dir.glob("*.toml")):
        if "dataset" not in path.stem.lower():
            continue
        presets[str(path)] = _format_dataset_preset_label(path)
    return presets


def _normalize_preview_datasets(project_config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_section = project_config.get("dataset", {}) if isinstance(project_config.get("dataset"), dict) else {}
    general = dataset_section.get("general", {}) if isinstance(dataset_section.get("general"), dict) else {}
    datasets = dataset_section.get("datasets", []) if isinstance(dataset_section.get("datasets"), list) else []
    interop = project_config.get("interop", {}) if isinstance(project_config.get("interop"), dict) else {}
    dataset_extra = interop.get("dataset_extra", {}) if isinstance(interop.get("dataset_extra"), dict) else {}
    dataset_extras = dataset_extra.get("datasets", []) if isinstance(dataset_extra.get("datasets"), list) else []
    return general, datasets, dataset_extras


def _merge_preview_dataset(dataset: Any, dataset_extra: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(dataset_extra, dict):
        merged.update(copy.deepcopy(dataset_extra))
    if isinstance(dataset, dict):
        merged.update(copy.deepcopy(dataset))
    return merged


def _collect_preview_dataset_views(project_config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    general, datasets, dataset_extras = _normalize_preview_datasets(project_config)
    dataset_views: list[dict[str, Any]] = []
    dataset_count = max(len(datasets), len(dataset_extras))

    for index in range(dataset_count):
        raw_dataset = datasets[index] if index < len(datasets) else {}
        raw_extra = dataset_extras[index] if index < len(dataset_extras) else {}
        merged_dataset = _merge_preview_dataset(raw_dataset, raw_extra)
        if merged_dataset:
            dataset_views.append(merged_dataset)

    return general, dataset_views


def _unique_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _format_resolution_value(value: Any) -> str | None:
    if isinstance(value, list) and len(value) == 2:
        return f"{value[0]}, {value[1]}"
    return None


def _format_scalar_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return str(value)


def _effective_dataset_value(dataset: dict[str, Any], general: dict[str, Any], key: str) -> Any:
    dataset_value = dataset.get(key)
    if isinstance(dataset_value, str):
        return dataset_value.strip() or general.get(key)
    if dataset_value in (None, [], {}):
        return general.get(key)
    return dataset_value


def _has_dataset_source(dataset: dict[str, Any]) -> bool:
    for key in ("image_directory", "image_jsonl_file", "video_directory", "video_jsonl_file"):
        value = dataset.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _is_preview_runnable(general: dict[str, Any], dataset_views: list[dict[str, Any]]) -> bool:
    if not dataset_views:
        return False

    for dataset in dataset_views:
        if not _has_dataset_source(dataset):
            return False

        cache_directory = dataset.get("cache_directory")
        if not isinstance(cache_directory, str) or not cache_directory.strip():
            return False

        if _format_resolution_value(_effective_dataset_value(dataset, general, "resolution")) is None:
            return False

        if _format_scalar_value(_effective_dataset_value(dataset, general, "batch_size")) is None:
            return False

        if _format_scalar_value(_effective_dataset_value(dataset, general, "num_repeats")) is None:
            return False

    return True


def _resolve_project_path(path_value: Any, project_dir: str | Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(project_dir) / path


def _directory_has_tag_files(path_value: Any, project_dir: str | Path) -> bool:
    directory = _resolve_project_path(path_value, project_dir)
    if directory is None or not directory.exists() or not directory.is_dir():
        return False

    for pattern in ("*.txt", "*.json", "*.jsonl"):
        if next(directory.rglob(pattern), None) is not None:
            return True
    return False


def _file_has_tag_data(path_value: Any, project_dir: str | Path) -> bool:
    file_path = _resolve_project_path(path_value, project_dir)
    if file_path is None or not file_path.exists() or not file_path.is_file():
        return False
    return file_path.suffix.lower() in {".json", ".jsonl"}


def _detect_tagging_status(dataset_views: list[dict[str, Any]], project_dir: str | Path) -> str:
    if not dataset_views:
        return "dataset_untagged"

    for dataset in dataset_views:
        has_tag_data = (
            _directory_has_tag_files(dataset.get("image_directory"), project_dir)
            or _directory_has_tag_files(dataset.get("video_directory"), project_dir)
            or _file_has_tag_data(dataset.get("image_jsonl_file"), project_dir)
            or _file_has_tag_data(dataset.get("video_jsonl_file"), project_dir)
        )
        if not has_tag_data:
            return "dataset_untagged"
    return "dataset_tagged"


def _is_local_port_listening(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _detect_template_type(project_config: dict[str, Any], preset_path: Path | None) -> str:
    general, dataset_views = _collect_preview_dataset_views(project_config)
    source_name = preset_path.stem.lower() if preset_path is not None else ""

    video_markers = ("video", "single-frame", "single frame", "framepack")
    if any(marker in source_name for marker in video_markers):
        return "template_video_generation"

    video_keys = {
        "video_directory",
        "video_jsonl_file",
        "target_frames",
        "frame_extraction",
        "frame_stride",
        "frame_sample",
        "max_frames",
        "source_fps",
    }

    for dataset in dataset_views:
        if any(dataset.get(key) for key in video_keys):
            return "template_video_generation"
        if any(dataset.get(key) for key in ("fp_latent_window_size", "fp_1f_clean_indices", "fp_1f_target_index", "fp_1f_no_post")):
            return "template_video_generation"
        if any("control" in key.lower() for key in dataset.keys()):
            return "template_image_edit"

    if any(marker in source_name for marker in ("edit", "kontext")):
        return "template_image_edit"

    if any(general.get(key) for key in video_keys):
        return "template_video_generation"

    return "template_text_to_image"


def _build_preview_summary(project_config: dict[str, Any], preset_path: Path | None, project_dir: str | Path) -> dict[str, Any]:
    general, dataset_views = _collect_preview_dataset_views(project_config)
    base_summary = summarize_dataset_state(project_config, project_dir)

    directory_lines: list[str] = []
    resolution_values: list[str] = []
    batch_sizes: list[str] = []
    repeat_values: list[str] = []

    for dataset in dataset_views:
        image_directory = dataset.get("image_directory")
        if isinstance(image_directory, str) and image_directory.strip():
            directory_lines.append(f"image::{image_directory}")

        video_directory = dataset.get("video_directory")
        if isinstance(video_directory, str) and video_directory.strip():
            directory_lines.append(f"video::{video_directory}")

        dataset_resolution = _format_resolution_value(_effective_dataset_value(dataset, general, "resolution"))
        if dataset_resolution:
            resolution_values.append(dataset_resolution)

        dataset_batch_size = _format_scalar_value(_effective_dataset_value(dataset, general, "batch_size"))
        if dataset_batch_size:
            batch_sizes.append(dataset_batch_size)

        dataset_num_repeats = _format_scalar_value(_effective_dataset_value(dataset, general, "num_repeats"))
        if dataset_num_repeats:
            repeat_values.append(dataset_num_repeats)

    if not dataset_views:
        general_resolution = _format_resolution_value(general.get("resolution"))
        if general_resolution:
            resolution_values.append(general_resolution)

        general_batch_size = _format_scalar_value(general.get("batch_size"))
        if general_batch_size:
            batch_sizes.append(general_batch_size)

        general_num_repeats = _format_scalar_value(general.get("num_repeats"))
        if general_num_repeats:
            repeat_values.append(general_num_repeats)

    return {
        **base_summary,
        "directories": _unique_ordered(directory_lines),
        "resolution_values": _unique_ordered(resolution_values),
        "batch_sizes": _unique_ordered(batch_sizes),
        "repeat_values": _unique_ordered(repeat_values),
        "template_type": _detect_template_type(project_config, preset_path),
        "is_runnable": "true" if _is_preview_runnable(general, dataset_views) else "false",
        "tagging_status": _detect_tagging_status(dataset_views, project_dir),
    }


def build_dataset_preview(project_config: dict[str, Any], preset_path: str | Path | None, project_dir: str | Path) -> dict[str, Any]:
    if preset_path is None or str(preset_path) == CURRENT_PROJECT_PRESET_ID:
        preview_config = copy.deepcopy(project_config)
        source_label = "Current Project"
        resolved_path = None
    else:
        resolved_path = Path(preset_path)
        imported = load_dataset_config_import(resolved_path)
        preview_config = {
            "dataset": imported["dataset"],
            "interop": {
                "dataset_extra": imported["interop"]["dataset_extra"],
                "import_sources": {"dataset_config": str(resolved_path)},
            },
        }
        source_label = _format_dataset_preset_label(resolved_path)

    return {
        "config": preview_config,
        "summary": _build_preview_summary(preview_config, resolved_path, project_dir),
        "source_label": source_label,
    }


class DatasetStep:
    """Dataset owner page for the GUI workflow."""

    def __init__(self):
        self.project_dir = get_default_project_dir()
        self.qinglong_captions_dir = self.project_dir / QINGLONG_CAPTIONS_DIRNAME
        self.default_dataset_config_path = get_default_dataset_config_path(self.project_dir)
        self.project_config = config_manager.load_project_config(self.project_dir)
        self.dataset_preset_options = discover_dataset_presets(self.project_dir / DATASET_PRESET_DIRNAME)
        self.dataset_preset_select = None
        self.preview_dataset = build_dataset_preview(self.project_config, None, self.project_dir)
        self.preview_directories = None
        self.preview_source_path = None
        self.preview_resolution = None
        self.preview_batch_repeats = None
        self.preview_template_type = None
        self.preview_status = None
        self.tagging_tool_status = None
        self.tagging_uv_status = None
        self.tagging_env_status = None
        self.tagging_port_status = None
        self.dataset_rows_container = None
        self.dataset_row_states: list[dict[str, Any]] = []
        self.dataset_row_controls: list[dict[str, Any]] = []
        self._refresh_dataset_row_states()

    def render(self):
        """Render step 1 as the dataset owner page."""
        with ui.column().classes(get_classes("page_container") + " gap-4"):
            with ui.row().classes("w-full items-center gap-3 q-mb-sm"):
                ui.icon("dataset", size="32px")
                with ui.column().classes("gap-0"):
                    ui.label(t("nav_dataset", "Dataset")).classes("text-h4 text-weight-bold").style("color: var(--color-text);")
                    ui.label(
                        t("dataset_page_desc", "Dataset ownership, tagging, and dataset_config.toml interoperability")
                    ).classes("text-body2").style("color: var(--color-text-secondary);")

            with ui.tabs().classes("w-full") as tabs:
                tab_overview = ui.tab(t("dataset_tab_overview", "Overview"), icon="dashboard")
                tab_tagging = ui.tab(t("dataset_tab_tagging", "Tagging"), icon="label")
                tab_dataset_config = ui.tab(t("dataset_tab_details", "数据集设置"), icon="description")

            with ui.tab_panels(tabs, value=tab_overview).classes("w-full"):
                with ui.tab_panel(tab_overview):
                    self._render_overview_tab()
                with ui.tab_panel(tab_tagging):
                    self._render_tagging_tab()
                with ui.tab_panel(tab_dataset_config):
                    self._render_dataset_config_tab()

    def _render_overview_tab(self):
        summary = self.preview_dataset["summary"]

        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            ui.label(t("dataset_preset_library", "Dataset Preset Library")).classes("text-h6 text-weight-bold q-mb-md").style(
                "color: var(--color-text);"
            )
            if self.dataset_preset_options:
                preset_options = {
                    CURRENT_PROJECT_PRESET_ID: t("current_project_dataset", "Current Project"),
                    **self.dataset_preset_options,
                }
                self.dataset_preset_select = ui.select(
                    preset_options,
                    label=t("dataset_preset", "Dataset Preset"),
                    value=CURRENT_PROJECT_PRESET_ID,
                ).classes("w-full").props(
                    'use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'
                )
                self.dataset_preset_select.on_value_change(self._on_dataset_preset_change)
                ui.label(
                    t("dataset_preset_desc", "Switch presets here to preview dataset contents before importing")
                ).classes("text-caption q-mt-sm").style("color: var(--color-text-secondary);")
                with ui.row().classes("w-full gap-2 q-mt-md flex-wrap"):
                    ui.button(
                        t("import_dataset_preset", "Import Dataset Preset"),
                        on_click=self._import_selected_dataset_preset,
                        icon="playlist_add_check",
                    ).classes("modern-btn-primary")
                    ui.button(
                        t("refresh_dataset_preset_list", "Refresh Preset List"),
                        on_click=self._reload_page,
                        icon="refresh",
                    ).classes("modern-btn-ghost")
            else:
                ui.label(
                    t("no_dataset_presets", "No dataset presets were found in the project's toml folder")
                ).classes("text-body2").style("color: var(--color-text-secondary);")

        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            ui.label(t("dataset_preview_summary", "Dataset Preview Summary")).classes("text-h6 text-weight-bold q-mb-md").style(
                "color: var(--color-text);"
            )
            with ui.row().classes("w-full gap-4 flex-wrap"):
                self.preview_directories = self._render_stat_card(
                    "folder",
                    t("dataset_directories", "Dataset Directories"),
                    self._format_directory_lines(summary["directories"]),
                )
                self.preview_resolution = self._render_stat_card(
                    "aspect_ratio",
                    t("dataset_resolution_wh", "Resolution (W,H)"),
                    self._format_value_lines(summary["resolution_values"]),
                )
                self.preview_batch_repeats = self._render_stat_card(
                    "tag",
                    t("dataset_batch_and_repeats", "Batch Size & Repeats"),
                    self._format_batch_repeat_lines(summary["batch_sizes"], summary["repeat_values"]),
                )
            with ui.row().classes("w-full gap-4 q-mt-md flex-wrap"):
                self.preview_source_path = self._render_stat_card(
                    "description",
                    "dataset_config.toml",
                    summary["source_path"],
                )
                self.preview_template_type = self._render_stat_card(
                    "category",
                    t("dataset_template_type", "Dataset Template Type"),
                    t(summary["template_type"]),
                )
                self.preview_status = self._render_stat_card(
                    "info",
                    t("dataset_status", "Dataset Status"),
                    self._format_preview_status(summary),
                )

        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            ui.label(t("dataset_quick_actions", "Quick Actions")).classes("text-h6 text-weight-bold q-mb-md").style(
                "color: var(--color-text);"
            )
            with ui.row().classes("w-full gap-2 flex-wrap"):
                ui.button(
                    t("import_project_dataset_config", "Import Project dataset_config.toml"),
                    on_click=lambda: self._import_dataset_config_from_path(self.default_dataset_config_path),
                    icon="upload_file",
                ).classes("modern-btn-secondary")
                ui.button(
                    t("export_project_dataset_config", "Export Project dataset_config.toml"),
                    on_click=lambda: self._export_dataset_config_to_path(self.default_dataset_config_path),
                    icon="download",
                ).classes("modern-btn-primary")
                ui.button(
                    t("reload_project_dataset", "Reload Project Dataset"),
                    on_click=self._reload_page,
                    icon="refresh",
                ).classes("modern-btn-ghost")

    def _render_stat_card(self, icon: str, label: str, value: str):
        with ui.card().classes(get_classes("card") + " flex-1 min-w-[220px] q-pa-sm"):
            with ui.row().classes("items-center gap-2 q-mb-xs"):
                ui.icon(icon, size="18px")
                ui.label(label).classes("text-caption text-weight-medium").style("color: var(--color-text-secondary);")
            value_label = ui.label(value).classes("text-body2").style("color: var(--color-text); word-break: break-all; white-space: pre-line;")
        return value_label

    def _format_directory_lines(self, directories: list[str]) -> str:
        if not directories:
            return "-"

        lines: list[str] = []
        for entry in directories:
            prefix, path = entry.split("::", 1)
            if prefix == "video":
                lines.append(f'{t("video_directory", "Video Directory")}: {path}')
            else:
                lines.append(f'{t("image_directory", "Image Directory")}: {path}')
        return "\n".join(lines)

    def _format_value_lines(self, values: list[str]) -> str:
        return "\n".join(values) if values else "-"

    def _format_named_value_lines(self, label: str, values: list[str]) -> str:
        if not values:
            return f"{label}: -"
        if len(values) == 1:
            return f"{label}: {values[0]}"
        return "\n".join(f"{label}: {value}" for value in values)

    def _format_batch_repeat_lines(self, batch_sizes: list[str], repeat_values: list[str]) -> str:
        return (
            f'{self._format_named_value_lines(t("batch_size"), batch_sizes)}\n'
            f'{self._format_named_value_lines(t("num_repeats", "Repeats"), repeat_values)}'
        )

    def _format_preview_status(self, summary: dict[str, Any]) -> str:
        return t(summary.get("tagging_status", "dataset_untagged"))

    def _render_tagging_tab(self):
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            ui.label(t("tagging_external_tool_title", "External Tagging Tool")).classes("text-h6 text-weight-bold q-mb-md").style(
                "color: var(--color-text);"
            )
            ui.label(
                t(
                    "tagging_external_tool_desc",
                    "Tagging is delegated to the qinglong-captions submodule. Install its environment first, then launch the dedicated GUI.",
                )
            ).classes("text-body2 q-mb-md").style("color: var(--color-text-secondary);")

            with ui.column().classes("w-full gap-2 q-mb-md"):
                ui.label(f"qinglong-captions: {self.qinglong_captions_dir}").classes("text-body2").style(
                    "color: var(--color-text); word-break: break-all;"
                )
                ui.label(QINGLONG_INSTALL_SCRIPT).classes("text-caption").style("color: var(--color-text-secondary);")
                ui.label(QINGLONG_START_GUI_SCRIPT).classes("text-caption").style("color: var(--color-text-secondary);")

            with ui.row().classes("w-full gap-4 q-mb-md flex-wrap"):
                self.tagging_tool_status = self._render_stat_card(
                    "hub",
                    t("tagging_tool_status_label", "Tool Status"),
                    "-",
                )
                self.tagging_uv_status = self._render_stat_card(
                    "terminal",
                    "uv",
                    "-",
                )
                self.tagging_env_status = self._render_stat_card(
                    "folder",
                    t("tagging_env_status_label", "Virtual Environment"),
                    "-",
                )
                self.tagging_port_status = self._render_stat_card(
                    "lan",
                    t("tagging_port_status_label", "Port 7899"),
                    "-",
                )

            with ui.row().classes("w-full gap-3 flex-wrap"):
                ui.button(
                    t("tagging_install_button", "Install"),
                    on_click=self._install_qinglong_captions,
                    icon="download",
                ).classes("modern-btn-primary")
                ui.button(
                    t("tagging_launch_button", "Launch"),
                    on_click=self._launch_qinglong_captions_gui,
                    icon="open_in_new",
                ).classes("modern-btn-secondary")
                ui.button(
                    t("refresh_status", "Refresh Status"),
                    on_click=self._refresh_qinglong_status,
                    icon="refresh",
                ).classes("modern-btn-ghost")

        self._refresh_qinglong_status()
        ui.timer(5.0, self._refresh_qinglong_status)

    def _qinglong_script_path(self, script_name: str) -> Path:
        return self.qinglong_captions_dir / script_name

    def _qinglong_env_path(self) -> Path | None:
        for dirname in (".venv", "venv"):
            candidate = self.qinglong_captions_dir / dirname
            if candidate.exists():
                return candidate
        return None

    def _collect_qinglong_status(self) -> dict[str, Any]:
        uv_path = shutil.which("uv")
        env_path = self._qinglong_env_path()
        port_listening = _is_local_port_listening(QINGLONG_START_GUI_PORT)

        if port_listening:
            overall_status = "tagging_tool_running"
        elif uv_path and env_path is not None:
            overall_status = "tagging_tool_installed"
        else:
            overall_status = "tagging_tool_not_installed"

        return {
            "overall_status": overall_status,
            "uv_path": uv_path,
            "env_path": env_path,
            "port_listening": port_listening,
        }

    def _refresh_qinglong_status(self):
        status = self._collect_qinglong_status()
        uv_text = status["uv_path"] or t("tagging_uv_missing", "Missing")
        env_text = str(status["env_path"]) if status["env_path"] is not None else t("tagging_env_missing", "Missing")
        port_text = (
            t("tagging_port_listening", "Listening")
            if status["port_listening"]
            else t("tagging_port_not_listening", "Not Listening")
        )

        updates = (
            (self.tagging_tool_status, t(status["overall_status"])),
            (self.tagging_uv_status, uv_text),
            (self.tagging_env_status, env_text),
            (self.tagging_port_status, port_text),
        )

        for target, text in updates:
            if target is None:
                continue
            try:
                target.set_text(text)
            except RuntimeError:
                return

    def _powershell_command_prefix(self) -> list[str] | None:
        if sys.platform == "win32":
            return ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File"]

        pwsh = shutil.which("pwsh")
        if pwsh:
            return [pwsh, "-File"]

        powershell = shutil.which("powershell")
        if powershell:
            return [powershell, "-File"]

        return None

    def _launch_qinglong_script(self, script_name: str):
        script_path = self._qinglong_script_path(script_name)
        if not script_path.exists():
            ui.notify(f'{t("tagging_script_missing", "Script not found")}: {script_path}', type="warning")
            return

        command_prefix = self._powershell_command_prefix()
        if command_prefix is None:
            ui.notify(t("tagging_powershell_missing", "PowerShell is not available on this system"), type="warning")
            return

        creationflags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0

        try:
            subprocess.Popen(
                [*command_prefix, str(script_path)],
                cwd=str(script_path.parent),
                creationflags=creationflags,
            )
            ui.notify(f'{t("tagging_script_started", "Script started")}: {script_path.name}', type="positive")
        except Exception as exc:
            ui.notify(f'{t("tagging_script_launch_failed", "Failed to launch script")}: {exc}', type="negative")

    def _install_qinglong_captions(self):
        self._launch_qinglong_script(QINGLONG_INSTALL_SCRIPT)
        self._refresh_qinglong_status()

    def _launch_qinglong_captions_gui(self):
        self._launch_qinglong_script(QINGLONG_START_GUI_SCRIPT)
        self._refresh_qinglong_status()

    def _render_dataset_config_tab(self):
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            ui.label(t("dataset_tab_details", "数据集设置")).classes("text-h6 text-weight-bold q-mb-md").style(
                "color: var(--color-text);"
            )
            self.import_dataset_config_path = create_path_selector(
                label=t("import_dataset_config", "Import dataset_config.toml"),
                default_path=str(self.default_dataset_config_path) if self.default_dataset_config_path.exists() else "",
                selection_type="file",
                file_filter="*.toml",
                placeholder=t("select_toml", "Select .toml dataset config file"),
            )
            self.export_dataset_config_path = create_path_selector(
                label=t("export_dataset_config", "Export dataset_config.toml"),
                default_path=str(self.default_dataset_config_path),
                selection_type="save",
                file_filter="*.toml",
                placeholder=str(self.default_dataset_config_path),
            )
            with ui.row().classes("w-full gap-2 q-mt-md flex-wrap"):
                ui.button(t("import_dataset_config", "Import dataset_config.toml"), on_click=self._import_dataset_config, icon="upload_file").classes(
                    "modern-btn-secondary"
                )
                ui.button(t("save_dataset_state", "Save Dataset State"), on_click=self._save_dataset_state, icon="save").classes(
                    "modern-btn-primary"
                )
                ui.button(t("export_dataset_config", "Export dataset_config.toml"), on_click=self._export_dataset_config, icon="download").classes(
                    "modern-btn-secondary"
                )

        general = self.project_config.get("dataset", {}).get("general", {})
        resolution = general.get("resolution", ["", ""])
        if not isinstance(resolution, list) or len(resolution) != 2:
            resolution = ["", ""]

        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            ui.label(t("dataset_general", "General")).classes("text-h6 text-weight-bold q-mb-md").style("color: var(--color-text);")
            with ui.row().classes("w-full gap-4"):
                self.general_resolution_w = ui.input(
                    t("resolution_w", "Resolution Width"), value=str(resolution[0]) if resolution[0] != "" else ""
                ).classes("flex-1 modern-input")
                self.general_resolution_h = ui.input(
                    t("resolution_h", "Resolution Height"), value=str(resolution[1]) if resolution[1] != "" else ""
                ).classes("flex-1 modern-input")
                self.general_batch_size = ui.input(
                    t("batch_size"), value=self._string_value(general.get("batch_size"))
                ).classes("flex-1 modern-input")
            with ui.row().classes("w-full gap-4 q-mt-md"):
                self.general_caption_extension = ui.input(
                    t("caption_extension"), value=self._string_value(general.get("caption_extension", ".txt"))
                ).classes("flex-1 modern-input")
                self.general_num_repeats = ui.input(
                    t("num_repeats", "Repeats"), value=self._string_value(general.get("num_repeats"))
                ).classes("flex-1 modern-input")
                self.general_enable_bucket = ui.checkbox(t("enable_bucket", "Enable Bucket"), value=bool(general.get("enable_bucket", True)))
                self.general_bucket_no_upscale = ui.checkbox(
                    t("bucket_no_upscale", "Bucket No Upscale"), value=bool(general.get("bucket_no_upscale", False))
                )

        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center justify-between q-mb-md"):
                ui.label(t("datasets", "Datasets")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")
                with ui.row().classes("gap-2"):
                    ui.button(t("add_image_dataset", "Add Image Dataset"), on_click=self._add_image_dataset_row, icon="add").classes(
                        "modern-btn-secondary"
                    )
                    ui.button(t("add_video_dataset", "Add Video Dataset"), on_click=self._add_video_dataset_row, icon="movie").classes(
                        "modern-btn-secondary"
                    )

            self.dataset_rows_container = ui.column().classes("w-full gap-4")
            self._render_dataset_rows()

    def _string_value(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    def _on_dataset_preset_change(self, e):
        self._update_dataset_preview(e.value)

    def _update_dataset_preview(self, preset_path: str | None):
        selected_path = None if preset_path in (None, CURRENT_PROJECT_PRESET_ID) else preset_path
        self.preview_dataset = build_dataset_preview(self.project_config, selected_path, self.project_dir)
        summary = self.preview_dataset["summary"]

        if self.preview_directories is not None:
            self.preview_directories.set_text(self._format_directory_lines(summary["directories"]))
        if self.preview_source_path is not None:
            self.preview_source_path.set_text(summary["source_path"])
        if self.preview_resolution is not None:
            self.preview_resolution.set_text(self._format_value_lines(summary["resolution_values"]))
        if self.preview_batch_repeats is not None:
            self.preview_batch_repeats.set_text(self._format_batch_repeat_lines(summary["batch_sizes"], summary["repeat_values"]))
        if self.preview_template_type is not None:
            self.preview_template_type.set_text(t(summary["template_type"]))
        if self.preview_status is not None:
            self.preview_status.set_text(self._format_preview_status(summary))

    def _dataset_source_options(self) -> dict[str, str]:
        return {
            "directory": t("dataset_source_directory", "Caption Files in Directory"),
            "jsonl": t("dataset_source_jsonl", "Metadata JSONL File"),
        }

    def _dataset_template_options(self, dataset_type: str) -> dict[str, str]:
        if dataset_type == "video":
            return {
                "video_generation": t("template_video_generation", "Video Generation"),
                "video_control": t("template_video_control", "Video Control"),
            }
        return {
            "text_to_image": t("template_text_to_image", "Text to Image"),
            "image_edit": t("template_image_edit", "Image Edit"),
            "framepack_one_frame": t("template_framepack_one_frame", "FramePack One Frame"),
        }

    def _normalize_dataset_template(self, dataset_type: str, template: str | None) -> str:
        options = self._dataset_template_options(dataset_type)
        if template in options:
            return str(template)
        return next(iter(options))

    def _dataset_import_source_name(self) -> str:
        interop = self.project_config.get("interop", {}) if isinstance(self.project_config.get("interop"), dict) else {}
        import_sources = interop.get("import_sources", {}) if isinstance(interop.get("import_sources"), dict) else {}
        return str(import_sources.get("dataset_config", "")).lower()

    def _infer_dataset_row_template(self, dataset_type: str, merged_dataset: dict[str, Any]) -> str:
        source_name = self._dataset_import_source_name()

        if dataset_type == "video":
            if merged_dataset.get("control_directory"):
                return "video_control"
            return "video_generation"

        if any(
            merged_dataset.get(key)
            for key in ("fp_latent_window_size", "fp_1f_clean_indices", "fp_1f_target_index", "fp_1f_no_post")
        ):
            return "framepack_one_frame"

        if any(marker in source_name for marker in ("single-frame", "single frame", "framepack")) and merged_dataset.get("control_directory"):
            return "framepack_one_frame"

        if (
            merged_dataset.get("control_directory")
            or merged_dataset.get("no_resize_control")
            or merged_dataset.get("control_resolution")
            or merged_dataset.get("flux_kontext_no_resize_control")
            or merged_dataset.get("qwen_image_edit_no_resize_control")
            or merged_dataset.get("qwen_image_edit_control_resolution")
            or any(marker in source_name for marker in ("edit", "kontext"))
        ):
            return "image_edit"

        return "text_to_image"

    def _build_dataset_row_state(self, raw_dataset: dict[str, Any], raw_extra: dict[str, Any]) -> dict[str, Any]:
        merged_dataset = copy.deepcopy(raw_extra)
        merged_dataset.update(copy.deepcopy(raw_dataset))

        supported_keys = {
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
        }
        clean_extra = {key: copy.deepcopy(value) for key, value in raw_extra.items() if key not in supported_keys}

        dataset_type = "video" if merged_dataset.get("video_directory") or merged_dataset.get("video_jsonl_file") else "image"
        dataset_source = "jsonl" if (
            merged_dataset.get("image_jsonl_file") or merged_dataset.get("video_jsonl_file")
        ) else "directory"
        dataset_template = self._infer_dataset_row_template(dataset_type, merged_dataset)

        resolution = merged_dataset.get("resolution", ["", ""])
        if not isinstance(resolution, list) or len(resolution) != 2:
            resolution = ["", ""]

        control_resolution = merged_dataset.get("control_resolution", merged_dataset.get("qwen_image_edit_control_resolution", ["", ""]))
        if not isinstance(control_resolution, list) or len(control_resolution) != 2:
            control_resolution = ["", ""]

        target_frames = merged_dataset.get("target_frames", [])
        clean_indices = merged_dataset.get("fp_1f_clean_indices", [])

        return {
            "dataset_type": dataset_type,
            "dataset_source": dataset_source,
            "dataset_template": self._normalize_dataset_template(dataset_type, dataset_template),
            "image_directory": self._string_value(merged_dataset.get("image_directory")),
            "image_jsonl_file": self._string_value(merged_dataset.get("image_jsonl_file")),
            "video_directory": self._string_value(merged_dataset.get("video_directory")),
            "video_jsonl_file": self._string_value(merged_dataset.get("video_jsonl_file")),
            "cache_directory": self._string_value(merged_dataset.get("cache_directory")),
            "control_directory": self._string_value(merged_dataset.get("control_directory")),
            "caption_extension": self._string_value(merged_dataset.get("caption_extension")),
            "resolution_w": self._string_value(resolution[0]) if resolution[0] != "" else "",
            "resolution_h": self._string_value(resolution[1]) if resolution[1] != "" else "",
            "batch_size": self._string_value(merged_dataset.get("batch_size")),
            "num_repeats": self._string_value(merged_dataset.get("num_repeats")),
            "multiple_target": bool(merged_dataset.get("multiple_target", False)),
            "no_resize_control": bool(
                merged_dataset.get("no_resize_control", False)
                or merged_dataset.get("flux_kontext_no_resize_control", False)
                or merged_dataset.get("qwen_image_edit_no_resize_control", False)
            ),
            "control_resolution_w": self._string_value(control_resolution[0]) if control_resolution[0] != "" else "",
            "control_resolution_h": self._string_value(control_resolution[1]) if control_resolution[1] != "" else "",
            "target_frames": ", ".join(str(value) for value in target_frames) if isinstance(target_frames, list) else "",
            "frame_extraction": self._string_value(merged_dataset.get("frame_extraction", "head")) or "head",
            "frame_stride": self._string_value(merged_dataset.get("frame_stride")),
            "frame_sample": self._string_value(merged_dataset.get("frame_sample")),
            "max_frames": self._string_value(merged_dataset.get("max_frames")),
            "source_fps": self._string_value(merged_dataset.get("source_fps")),
            "fp_latent_window_size": self._string_value(merged_dataset.get("fp_latent_window_size")),
            "fp_1f_clean_indices": ", ".join(str(value) for value in clean_indices) if isinstance(clean_indices, list) else "",
            "fp_1f_target_index": self._string_value(merged_dataset.get("fp_1f_target_index")),
            "fp_1f_no_post": bool(merged_dataset.get("fp_1f_no_post", False)),
            "extra": clean_extra,
        }

    def _refresh_dataset_row_states(self):
        dataset_section = self.project_config.get("dataset", {})
        datasets = dataset_section.get("datasets", [])
        dataset_extra = self._dataset_extra().get("datasets", [])

        self.dataset_row_states = []
        dataset_count = max(len(datasets), len(dataset_extra))
        for index in range(dataset_count):
            raw_dataset: dict[str, Any] = {}
            if index < len(datasets) and isinstance(datasets[index], dict):
                raw_dataset = datasets[index]
            raw_extra = dataset_extra[index] if index < len(dataset_extra) and isinstance(dataset_extra[index], dict) else {}
            self.dataset_row_states.append(self._build_dataset_row_state(raw_dataset, raw_extra))

        if not self.dataset_row_states:
            self.dataset_row_states.append(self._empty_dataset_row_state())

    def _empty_dataset_row_state(
        self,
        dataset_type: str = "image",
        dataset_template: str | None = None,
        dataset_source: str = "directory",
    ) -> dict[str, Any]:
        return {
            "dataset_type": dataset_type,
            "dataset_source": dataset_source,
            "dataset_template": self._normalize_dataset_template(dataset_type, dataset_template),
            "image_directory": "",
            "image_jsonl_file": "",
            "video_directory": "",
            "video_jsonl_file": "",
            "cache_directory": "",
            "control_directory": "",
            "caption_extension": "",
            "resolution_w": "",
            "resolution_h": "",
            "batch_size": "",
            "num_repeats": "",
            "multiple_target": False,
            "no_resize_control": False,
            "control_resolution_w": "",
            "control_resolution_h": "",
            "target_frames": "",
            "frame_extraction": "head",
            "frame_stride": "",
            "frame_sample": "",
            "max_frames": "",
            "source_fps": "",
            "fp_latent_window_size": "",
            "fp_1f_clean_indices": "",
            "fp_1f_target_index": "",
            "fp_1f_no_post": False,
            "extra": {},
        }

    def _set_dataset_row_mode(self, index: int, key: str, value: str):
        self.dataset_row_states = self._snapshot_dataset_rows()
        if index >= len(self.dataset_row_states):
            return

        self.dataset_row_states[index][key] = value
        dataset_type = self.dataset_row_states[index].get("dataset_type", "image")
        self.dataset_row_states[index]["dataset_template"] = self._normalize_dataset_template(
            dataset_type, self.dataset_row_states[index].get("dataset_template")
        )
        self._render_dataset_rows()

    def _frame_extraction_options(self) -> dict[str, str]:
        return {
            "head": t("frame_extraction_head", "Head"),
            "chunk": t("frame_extraction_chunk", "Chunk"),
            "slide": t("frame_extraction_slide", "Slide"),
            "uniform": t("frame_extraction_uniform", "Uniform"),
            "full": t("frame_extraction_full", "Full"),
        }

    def _render_dataset_rows(self):
        if self.dataset_rows_container is None:
            return

        self.dataset_rows_container.clear()
        self.dataset_row_controls = []
        with self.dataset_rows_container:
            for index, state in enumerate(self.dataset_row_states):
                with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                    with ui.row().classes("w-full items-center justify-between q-mb-md"):
                        dataset_label = t("video_dataset", "Video Dataset") if state.get("dataset_type") == "video" else t("image_dataset", "Image Dataset")
                        ui.label(f"{dataset_label} {index + 1}").classes("text-subtitle1 text-weight-bold").style(
                            "color: var(--color-text);"
                        )
                        ui.button(icon="delete", on_click=lambda idx=index: self._remove_dataset_row(idx)).classes("modern-btn-ghost").props(
                            "dense"
                        )

                    controls: dict[str, Any] = {"__state__": copy.deepcopy(state)}

                    with ui.row().classes("w-full gap-4 flex-wrap q-mb-md"):
                        dataset_type_select = ui.select(
                            {
                                "image": t("image_dataset", "Image Dataset"),
                                "video": t("video_dataset", "Video Dataset"),
                            },
                            label=t("dataset_type", "Dataset Type"),
                            value=state["dataset_type"],
                        ).classes("min-w-[200px]")
                        dataset_type_select.on_value_change(lambda e, idx=index: self._set_dataset_row_mode(idx, "dataset_type", e.value))

                        dataset_source_select = ui.select(
                            self._dataset_source_options(),
                            label=t("dataset_source_mode", "Dataset Source"),
                            value=state["dataset_source"],
                        ).classes("min-w-[220px]")
                        dataset_source_select.on_value_change(lambda e, idx=index: self._set_dataset_row_mode(idx, "dataset_source", e.value))

                        dataset_template_select = ui.select(
                            self._dataset_template_options(state["dataset_type"]),
                            label=t("dataset_template_mode", "Dataset Template"),
                            value=state["dataset_template"],
                        ).classes("min-w-[240px]")
                        dataset_template_select.on_value_change(lambda e, idx=index: self._set_dataset_row_mode(idx, "dataset_template", e.value))

                    if state["dataset_type"] == "image":
                        if state["dataset_source"] == "directory":
                            controls["image_directory"] = create_path_selector(
                                label=t("image_directory", "Image Directory"),
                                default_path=state["image_directory"],
                                selection_type="dir",
                                placeholder="./train/image",
                            )
                        else:
                            controls["image_jsonl_file"] = create_path_selector(
                                label=t("image_jsonl_file", "Image JSONL File"),
                                default_path=state["image_jsonl_file"],
                                selection_type="file",
                                file_filter="*.jsonl",
                                placeholder="metadata.jsonl",
                            )

                        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                            controls["cache_directory"] = create_path_selector(
                                label=t("cache_directory", "Cache Directory"),
                                default_path=state["cache_directory"],
                                selection_type="dir",
                                placeholder="./train/image/cache",
                            )
                            if state["dataset_template"] in {"image_edit", "framepack_one_frame"} and state["dataset_source"] == "directory":
                                controls["control_directory"] = create_path_selector(
                                    label=t("control_directory", "Control Directory"),
                                    default_path=state["control_directory"],
                                    selection_type="dir",
                                    placeholder="./train/image/control",
                                )

                        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                            controls["resolution_w"] = ui.input(
                                t("resolution_w", "Resolution Width"), value=state["resolution_w"]
                            ).classes("min-w-[180px] modern-input")
                            controls["resolution_h"] = ui.input(
                                t("resolution_h", "Resolution Height"), value=state["resolution_h"]
                            ).classes("min-w-[180px] modern-input")
                            controls["batch_size"] = ui.input(
                                t("batch_size"), value=state["batch_size"]
                            ).classes("min-w-[160px] modern-input")
                            controls["num_repeats"] = ui.input(
                                t("num_repeats", "Repeats"), value=state["num_repeats"]
                            ).classes("min-w-[160px] modern-input")

                        if state["dataset_source"] == "directory":
                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["caption_extension"] = ui.input(
                                    t("caption_extension"), value=state["caption_extension"]
                                ).classes("min-w-[220px] modern-input")

                        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                            controls["multiple_target"] = ui.checkbox(
                                t("multiple_target", "Multiple Target"), value=state["multiple_target"]
                            )

                        if state["dataset_template"] in {"image_edit", "framepack_one_frame"}:
                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["no_resize_control"] = ui.checkbox(
                                    t("no_resize_control", "No Resize Control"), value=state["no_resize_control"]
                                )
                                controls["control_resolution_w"] = ui.input(
                                    t("control_resolution_w", "Control Resolution Width"),
                                    value=state["control_resolution_w"],
                                ).classes("min-w-[220px] modern-input")
                                controls["control_resolution_h"] = ui.input(
                                    t("control_resolution_h", "Control Resolution Height"),
                                    value=state["control_resolution_h"],
                                ).classes("min-w-[220px] modern-input")

                        if state["dataset_template"] == "framepack_one_frame":
                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["fp_latent_window_size"] = ui.input(
                                    t("fp_latent_window_size", "FramePack Latent Window Size"),
                                    value=state["fp_latent_window_size"],
                                ).classes("min-w-[220px] modern-input")
                                controls["fp_1f_target_index"] = ui.input(
                                    t("fp_1f_target_index", "FramePack Target Index"),
                                    value=state["fp_1f_target_index"],
                                ).classes("min-w-[220px] modern-input")
                                controls["fp_1f_no_post"] = ui.checkbox(
                                    t("fp_1f_no_post", "FramePack No Post"), value=state["fp_1f_no_post"]
                                )

                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["fp_1f_clean_indices"] = ui.input(
                                    t("fp_1f_clean_indices", "FramePack Clean Indices"),
                                    value=state["fp_1f_clean_indices"],
                                    placeholder="0, 1",
                                ).classes("min-w-[320px] modern-input")
                    else:
                        if state["dataset_source"] == "directory":
                            controls["video_directory"] = create_path_selector(
                                label=t("video_directory", "Video Directory"),
                                default_path=state["video_directory"],
                                selection_type="dir",
                                placeholder="./train/video",
                            )
                        else:
                            controls["video_jsonl_file"] = create_path_selector(
                                label=t("video_jsonl_file", "Video JSONL File"),
                                default_path=state["video_jsonl_file"],
                                selection_type="file",
                                file_filter="*.jsonl",
                                placeholder="video_metadata.jsonl",
                            )

                        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                            controls["cache_directory"] = create_path_selector(
                                label=t("cache_directory", "Cache Directory"),
                                default_path=state["cache_directory"],
                                selection_type="dir",
                                placeholder="./train/video/cache",
                            )
                            if state["dataset_template"] == "video_control" and state["dataset_source"] == "directory":
                                controls["control_directory"] = create_path_selector(
                                    label=t("control_directory", "Control Directory"),
                                    default_path=state["control_directory"],
                                    selection_type="dir",
                                    placeholder="./train/video/control",
                                )

                        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                            controls["resolution_w"] = ui.input(
                                t("resolution_w", "Resolution Width"), value=state["resolution_w"]
                            ).classes("min-w-[180px] modern-input")
                            controls["resolution_h"] = ui.input(
                                t("resolution_h", "Resolution Height"), value=state["resolution_h"]
                            ).classes("min-w-[180px] modern-input")
                            controls["batch_size"] = ui.input(
                                t("batch_size"), value=state["batch_size"]
                            ).classes("min-w-[160px] modern-input")
                            controls["num_repeats"] = ui.input(
                                t("num_repeats", "Repeats"), value=state["num_repeats"]
                            ).classes("min-w-[160px] modern-input")

                        if state["dataset_source"] == "directory":
                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["caption_extension"] = ui.input(
                                    t("caption_extension"), value=state["caption_extension"]
                                ).classes("min-w-[220px] modern-input")

                        with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                            controls["target_frames"] = ui.input(
                                t("target_frames", "Target Frames"),
                                value=state["target_frames"],
                                placeholder="1, 25, 45",
                            ).classes("min-w-[260px] modern-input")
                            controls["frame_extraction"] = ui.select(
                                self._frame_extraction_options(),
                                label=t("frame_extraction", "Frame Extraction"),
                                value=state["frame_extraction"],
                            ).classes("min-w-[220px]")
                            controls["frame_extraction"].on_value_change(
                                lambda e, idx=index: self._set_dataset_row_mode(idx, "frame_extraction", e.value)
                            )
                            controls["source_fps"] = ui.input(
                                t("source_fps", "Source FPS"), value=state["source_fps"]
                            ).classes("min-w-[180px] modern-input")

                        if state["frame_extraction"] == "slide":
                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["frame_stride"] = ui.input(
                                    t("frame_stride", "Frame Stride"), value=state["frame_stride"]
                                ).classes("min-w-[220px] modern-input")
                        if state["frame_extraction"] == "uniform":
                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["frame_sample"] = ui.input(
                                    t("frame_sample", "Frame Sample"), value=state["frame_sample"]
                                ).classes("min-w-[220px] modern-input")
                        if state["frame_extraction"] == "full":
                            with ui.row().classes("w-full gap-4 flex-wrap q-mt-md"):
                                controls["max_frames"] = ui.input(
                                    t("max_frames", "Max Frames"), value=state["max_frames"]
                                ).classes("min-w-[220px] modern-input")

                    self.dataset_row_controls.append(controls)

    def _snapshot_dataset_rows(self) -> list[dict[str, Any]]:
        if not self.dataset_row_controls:
            return list(self.dataset_row_states)

        snapshot = []
        for controls in self.dataset_row_controls:
            base_state = copy.deepcopy(controls.get("__state__", self._empty_dataset_row_state()))
            for key, control in controls.items():
                if key.startswith("__"):
                    continue
                if hasattr(control, "value"):
                    value = control.value
                else:
                    value = control
                if isinstance(value, bool):
                    base_state[key] = value
                else:
                    base_state[key] = self._string_value(value)
            snapshot.append(base_state)
        return snapshot

    def _add_image_dataset_row(self):
        self.dataset_row_states = self._snapshot_dataset_rows()
        self.dataset_row_states.append(self._empty_dataset_row_state("image", "text_to_image"))
        self._render_dataset_rows()

    def _add_video_dataset_row(self):
        self.dataset_row_states = self._snapshot_dataset_rows()
        self.dataset_row_states.append(self._empty_dataset_row_state("video", "video_generation"))
        self._render_dataset_rows()

    def _remove_dataset_row(self, index: int):
        self.dataset_row_states = self._snapshot_dataset_rows()
        if len(self.dataset_row_states) == 1:
            self.dataset_row_states[0] = self._empty_dataset_row_state()
        else:
            self.dataset_row_states.pop(index)
        self._render_dataset_rows()

    def _parse_int(self, raw_value: Any) -> int | None:
        value = self._string_value(raw_value).strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _parse_float(self, raw_value: Any) -> float | None:
        value = self._string_value(raw_value).strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_int_list(self, raw_value: Any) -> list[int]:
        value = self._string_value(raw_value).strip()
        if not value:
            return []

        parsed: list[int] = []
        for item in value.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            try:
                parsed.append(int(item))
            except ValueError:
                continue
        return parsed

    def _collect_general_state(self) -> dict[str, Any]:
        general: dict[str, Any] = {}
        resolution_w = self._parse_int(self.general_resolution_w.value)
        resolution_h = self._parse_int(self.general_resolution_h.value)
        if resolution_w and resolution_h:
            general["resolution"] = [resolution_w, resolution_h]

        caption_extension = self.general_caption_extension.value.strip()
        if caption_extension:
            general["caption_extension"] = caption_extension

        batch_size = self._parse_int(self.general_batch_size.value)
        if batch_size is not None:
            general["batch_size"] = batch_size

        num_repeats = self._parse_int(self.general_num_repeats.value)
        if num_repeats is not None:
            general["num_repeats"] = num_repeats

        general["enable_bucket"] = bool(self.general_enable_bucket.value)
        general["bucket_no_upscale"] = bool(self.general_bucket_no_upscale.value)
        return general

    def _collect_dataset_rows(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        datasets: list[dict[str, Any]] = []
        dataset_extras: list[dict[str, Any]] = []

        for state in self._snapshot_dataset_rows():
            dataset: dict[str, Any] = {}
            dataset_type = state.get("dataset_type", "image")
            dataset_source = state.get("dataset_source", "directory")
            dataset_template = self._normalize_dataset_template(dataset_type, state.get("dataset_template"))

            if dataset_type == "image":
                source_key = "image_jsonl_file" if dataset_source == "jsonl" else "image_directory"
                source_value = self._string_value(state.get(source_key)).strip()
                if source_value:
                    dataset[source_key] = source_value
                if dataset_source == "directory":
                    caption_extension = self._string_value(state.get("caption_extension")).strip()
                    if caption_extension:
                        dataset["caption_extension"] = caption_extension
                if dataset_template in {"image_edit", "framepack_one_frame"} and dataset_source == "directory":
                    control_directory = self._string_value(state.get("control_directory")).strip()
                    if control_directory:
                        dataset["control_directory"] = control_directory
                if dataset_template in {"image_edit", "framepack_one_frame"}:
                    if state.get("no_resize_control"):
                        dataset["no_resize_control"] = True
                    control_resolution_w = self._parse_int(state.get("control_resolution_w"))
                    control_resolution_h = self._parse_int(state.get("control_resolution_h"))
                    if control_resolution_w is not None and control_resolution_h is not None:
                        dataset["control_resolution"] = [control_resolution_w, control_resolution_h]
                if dataset_template == "framepack_one_frame":
                    fp_latent_window_size = self._parse_int(state.get("fp_latent_window_size"))
                    if fp_latent_window_size is not None:
                        dataset["fp_latent_window_size"] = fp_latent_window_size
                    fp_clean_indices = self._parse_int_list(state.get("fp_1f_clean_indices"))
                    if fp_clean_indices:
                        dataset["fp_1f_clean_indices"] = fp_clean_indices
                    fp_target_index = self._parse_int(state.get("fp_1f_target_index"))
                    if fp_target_index is not None:
                        dataset["fp_1f_target_index"] = fp_target_index
                    if state.get("fp_1f_no_post"):
                        dataset["fp_1f_no_post"] = True
                if state.get("multiple_target"):
                    dataset["multiple_target"] = True
            else:
                source_key = "video_jsonl_file" if dataset_source == "jsonl" else "video_directory"
                source_value = self._string_value(state.get(source_key)).strip()
                if source_value:
                    dataset[source_key] = source_value
                if dataset_source == "directory":
                    caption_extension = self._string_value(state.get("caption_extension")).strip()
                    if caption_extension:
                        dataset["caption_extension"] = caption_extension
                if dataset_template == "video_control" and dataset_source == "directory":
                    control_directory = self._string_value(state.get("control_directory")).strip()
                    if control_directory:
                        dataset["control_directory"] = control_directory

                target_frames = self._parse_int_list(state.get("target_frames"))
                if target_frames:
                    dataset["target_frames"] = target_frames

                frame_extraction = self._string_value(state.get("frame_extraction")).strip()
                if frame_extraction:
                    dataset["frame_extraction"] = frame_extraction

                frame_stride = self._parse_int(state.get("frame_stride"))
                if frame_stride is not None:
                    dataset["frame_stride"] = frame_stride

                frame_sample = self._parse_int(state.get("frame_sample"))
                if frame_sample is not None:
                    dataset["frame_sample"] = frame_sample

                max_frames = self._parse_int(state.get("max_frames"))
                if max_frames is not None:
                    dataset["max_frames"] = max_frames

                source_fps = self._parse_float(state.get("source_fps"))
                if source_fps is not None:
                    dataset["source_fps"] = source_fps

            resolution_w = self._parse_int(state.get("resolution_w"))
            resolution_h = self._parse_int(state.get("resolution_h"))
            if resolution_w is not None and resolution_h is not None:
                dataset["resolution"] = [resolution_w, resolution_h]

            cache_directory = self._string_value(state.get("cache_directory")).strip()
            if cache_directory:
                dataset["cache_directory"] = cache_directory

            batch_size = self._parse_int(state.get("batch_size"))
            if batch_size is not None:
                dataset["batch_size"] = batch_size

            num_repeats = self._parse_int(state.get("num_repeats"))
            if num_repeats is not None:
                dataset["num_repeats"] = num_repeats

            has_primary_source = any(
                self._string_value(dataset.get(key)).strip()
                for key in ("image_directory", "image_jsonl_file", "video_directory", "video_jsonl_file")
            )
            if has_primary_source:
                datasets.append(dataset)
                dataset_extras.append(dict(state.get("extra", {})))

        return datasets, dataset_extras

    def _dataset_extra(self) -> dict[str, Any]:
        interop = self.project_config.setdefault("interop", {})
        dataset_extra = interop.setdefault("dataset_extra", {})
        dataset_extra.setdefault("root", {})
        dataset_extra.setdefault("general", {})
        dataset_extra.setdefault("datasets", [])
        return dataset_extra

    def _apply_project_metadata(self):
        project = self.project_config.setdefault("project", {})
        project.setdefault("name", self.project_dir.name)
        project["root_dir"] = str(self.project_dir)
        project.setdefault("created_by", "qinglong_gui")

    def _persist_dataset_to_project_config(self):
        self.project_config = config_manager.load_project_config(self.project_dir)
        self._apply_project_metadata()

        dataset_section = self.project_config.setdefault("dataset", {})
        dataset_section["general"] = self._collect_general_state()
        datasets, dataset_extras = self._collect_dataset_rows()
        dataset_section["datasets"] = datasets

        dataset_extra = self._dataset_extra()
        dataset_extra["datasets"] = dataset_extras
        dataset_extra.setdefault("root", {})
        dataset_extra.setdefault("general", {})

    def _save_dataset_state(self):
        self._persist_dataset_to_project_config()
        if config_manager.save_project_config(self.project_dir, self.project_config):
            ui.notify(t("save_dataset_state_success", "Dataset state saved"), type="positive")
            self._reload_page()
            return
        ui.notify(t("save_dataset_state_failed", "Failed to save dataset state"), type="negative")

    def _import_selected_dataset_preset(self):
        if self.dataset_preset_select is None or not self.dataset_preset_select.value:
            ui.notify(t("dataset_preset_missing", "Please select a dataset preset first"), type="warning")
            return
        if self.dataset_preset_select.value == CURRENT_PROJECT_PRESET_ID:
            ui.notify(t("dataset_preset_current_project_selected", "Current Project is already active"), type="warning")
            return
        self._import_dataset_config_from_path(self.dataset_preset_select.value)

    def _import_dataset_config(self):
        import_path = self.import_dataset_config_path.value or str(self.default_dataset_config_path)
        self._import_dataset_config_from_path(import_path)

    def _import_dataset_config_from_path(self, import_path: str | Path):
        path = Path(import_path)
        if not path.exists():
            ui.notify(t("dataset_import_missing", "dataset_config.toml not found"), type="warning")
            return

        imported = load_dataset_config_import(path)
        self.project_config = config_manager.load_project_config(self.project_dir)
        self._apply_project_metadata()
        self.project_config["dataset"] = imported["dataset"]
        self.project_config.setdefault("interop", {})["dataset_extra"] = imported["interop"]["dataset_extra"]
        self.project_config["interop"].setdefault("import_sources", {})["dataset_config"] = str(path)
        self.project_config.setdefault("gui", {}).setdefault("recent_import_paths", {})["dataset_config"] = str(path)

        if config_manager.save_project_config(self.project_dir, self.project_config):
            ui.notify(t("dataset_import_success", "Dataset config imported"), type="positive")
            self._reload_page()
            return
        ui.notify(t("dataset_import_failed", "Failed to import dataset config"), type="negative")

    def _export_dataset_config(self):
        export_path = self.export_dataset_config_path.value or str(self.default_dataset_config_path)
        self._export_dataset_config_to_path(export_path)

    def _export_dataset_config_to_path(self, export_path: str | Path):
        self._persist_dataset_to_project_config()
        target_path = Path(export_path)
        if not target_path.suffix:
            target_path = target_path / "dataset_config.toml"

        try:
            export_dataset_config(self.project_config, target_path)
            self.project_config.setdefault("gui", {}).setdefault("recent_import_paths", {})["dataset_config"] = str(target_path)
            config_manager.save_project_config(self.project_dir, self.project_config)
            ui.notify(t("dataset_export_success", "Dataset config exported"), type="positive")
        except Exception as exc:
            ui.notify(f'{t("dataset_export_failed", "Failed to export dataset config")}: {exc}', type="negative")

    def _reload_page(self):
        ui.run_javascript("window.location.reload()")


def render_dataset_step():
    """Render the upgraded step 1 dataset page."""
    DatasetStep().render()


def render_tagging_step():
    """Backward-compatible route entry point for /tagging."""
    render_dataset_step()
