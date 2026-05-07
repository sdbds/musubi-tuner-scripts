"""模型架构选择组件 - 使用共享 model_catalog。"""

from typing import Any, Callable, Dict, Optional

from nicegui import ui

from components.advanced_inputs import styled_select
from theme import COLORS, get_classes
from utils import model_catalog
from utils.i18n import t


class ModelSelector:
    """按页面能力显示架构 / 版本 / 任务。"""

    def __init__(
        self,
        on_change: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        default_arch: str = "FLUX.2",
        page_key: str = "train",
    ):
        self.on_change = on_change
        self.page_key = page_key
        self.current_arch = default_arch

        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                ui.icon("view_module", size="24px")
                ui.label(t("model_architecture")).classes("text-h6 text-weight-bold").style("color: var(--color-text);")

            with ui.row().classes("w-full items-start gap-4"):
                with ui.column().classes("flex-grow"):
                    self.arch_select = styled_select(
                        options=model_catalog.get_architecture_names(),
                        value=default_arch,
                        label=t("select_architecture"),
                        icon="hub",
                        placeholder=t("select_architecture"),
                        on_change=self._on_arch_change,
                    )

                with ui.column().classes("flex-grow") as self.version_container:
                    self.version_select = styled_select(
                        options=[],
                        value=None,
                        label=t("model_version"),
                        icon="layers",
                        placeholder=t("model_version"),
                        on_change=self._on_version_change,
                    )

                with ui.column().classes("flex-grow") as self.task_container:
                    self.task_select = styled_select(
                        options=[],
                        value=None,
                        label=t("task_type"),
                        icon="tune",
                        placeholder=t("task_type"),
                        on_change=self._on_task_change,
                    )

                with ui.column().classes("items-center justify-center").style(
                    """
                    width: 60px;
                    height: 60px;
                    border-radius: 16px;
                    """
                ) as self.icon_container:
                    self.arch_icon = ui.label("📦").classes("text-3xl")

            with ui.row().classes("w-full gap-4 q-mt-sm") as self.badge_row:
                self.type_badge = self._create_badge("", "primary")
                self.vae_badge = self._create_badge("需要 VAE", "secondary")
                self.fp8_badge = self._create_badge("支持 FP8", "accent")

        self._apply_arch(default_arch, emit=False)

    def _create_badge(self, text: str, color_key: str = "primary"):
        color = COLORS[color_key]
        with ui.element("span").classes(get_classes("badge")).style(
            f"""
            background: color-mix(in srgb, {color} 14%, transparent) !important;
            color: {color} !important;
            border: 1px solid color-mix(in srgb, {color} 28%, transparent);
            """
        ) as badge:
            label = ui.label(text).classes("text-caption text-weight-medium")
        badge._label = label
        return badge

    def _set_badge_state(self, badge, text: str, visible: bool):
        badge._label.set_text(text)
        badge.visible = visible

    def _update_icon(self, arch_info: Dict[str, Any]):
        self.arch_icon.text = arch_info.get("icon", "📦")
        color = arch_info.get("color", COLORS["primary"])
        self.icon_container.style(
            f"""
            width: 60px;
            height: 60px;
            background: linear-gradient(
                135deg,
                color-mix(in srgb, {color} 18%, transparent),
                color-mix(in srgb, {color} 8%, transparent)
            );
            border: 1px solid color-mix(in srgb, {color} 28%, transparent);
            border-radius: 16px;
            """
        )

    def _refresh_tasks(self, arch_name: str, version: Optional[str] = None):
        tasks = model_catalog.get_tasks_for_page(arch_name, self.page_key, version=version)
        default_task = model_catalog.get_default_task(arch_name, self.page_key, version=version)
        self.task_select.options = tasks
        self.task_select.value = default_task if default_task in tasks else (tasks[0] if tasks else None)
        self.task_container.visible = model_catalog.supports_task_selector(arch_name, self.page_key) and bool(tasks)
        self.task_select.update()

    def _apply_arch(self, arch_name: str, emit: bool = True):
        arch_info = model_catalog.get_architecture(arch_name) or {}
        self.current_arch = arch_name

        versions = model_catalog.get_versions_for_page(arch_name, self.page_key)
        default_version = model_catalog.get_default_version(arch_name, self.page_key)
        self.version_select.options = versions
        self.version_select.value = default_version if default_version in versions else (versions[0] if versions else None)
        self.version_select.update()

        self._refresh_tasks(arch_name, version=self.version_select.value)
        self._update_icon(arch_info)

        self._set_badge_state(self.type_badge, "视频模型" if arch_info.get("is_video") else "图像模型", True)
        self._set_badge_state(self.vae_badge, "需要 VAE", bool(arch_info.get("requires_vae")))
        self._set_badge_state(self.fp8_badge, "支持 FP8", bool(arch_info.get("supports_fp8_scaled")))

        if emit and self.on_change:
            self.on_change(arch_name, arch_info)

    def _on_arch_change(self, value: str):
        self._apply_arch(value, emit=True)

    def _on_version_change(self, value: str):
        self._refresh_tasks(self.current_arch, version=value)
        if self.on_change:
            self.on_change(self.current_arch, self.arch_info)

    def _on_task_change(self, _value: str):
        if self.on_change:
            self.on_change(self.current_arch, self.arch_info)

    @property
    def arch(self) -> str:
        return self.arch_select.value

    @property
    def version(self) -> Optional[str]:
        return self.version_select.value

    @property
    def task(self) -> Optional[str]:
        return self.task_select.value if self.task_container.visible else None

    @property
    def arch_info(self) -> Dict[str, Any]:
        return model_catalog.get_architecture(self.arch) or {}

    def set_arch(self, arch: str):
        if arch in model_catalog.get_architecture_names():
            self.arch_select.value = arch
            self.arch_select.update()
            self._apply_arch(arch, emit=True)

    def set_version(self, version: str):
        if version in model_catalog.get_versions_for_page(self.arch, self.page_key):
            self.version_select.value = version
            self.version_select.update()
            self._refresh_tasks(self.arch, version=version)
            if self.on_change:
                self.on_change(self.arch, self.arch_info)

    def set_task(self, task: str):
        tasks = model_catalog.get_tasks_for_page(self.arch, self.page_key, version=self.version)
        if task in tasks:
            self.task_select.value = task
            self.task_select.update()
            if self.on_change:
                self.on_change(self.arch, self.arch_info)


def create_model_selector(**kwargs) -> ModelSelector:
    return ModelSelector(**kwargs)


def get_arch_info(arch_name: str) -> Optional[Dict[str, Any]]:
    return model_catalog.get_architecture(arch_name)


def get_all_architectures() -> Dict[str, Dict[str, Any]]:
    return model_catalog.get_all_architectures()
