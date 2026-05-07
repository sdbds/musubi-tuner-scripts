"""Settings dialog for GUI-managed environment variables."""

from __future__ import annotations

from typing import Dict

from nicegui import ui

from theme import COLORS
from utils.env_config import ENV_VAR_DEFINITIONS, load_env_config, save_env_config
from utils.i18n import get_i18n, t


_SETTINGS_CSS = """
.settings-dialog-bg {
    background: linear-gradient(180deg, var(--ql-bg), var(--ql-surface-raised)) !important;
}
.settings-dialog-bg .settings-section-card {
    background: var(--ql-card-bg) !important;
    border: 1px solid var(--ql-card-border) !important;
    box-shadow: var(--ql-card-shadow) !important;
    border-radius: 8px;
}
.settings-dialog-bg .settings-section-header {
    background: linear-gradient(135deg, var(--ql-accent-soft), var(--ql-secondary-muted)) !important;
    border-radius: 8px 8px 0 0;
}
"""


class SettingsDialog:
    """Environment variable management dialog."""

    _GROUP_LABELS = {
        "runtime": "env_group_runtime",
        "uv": "env_group_uv",
        "network": "env_group_network",
    }
    _GROUP_ICONS = {
        "runtime": "memory",
        "uv": "inventory_2",
        "network": "language",
    }
    _SECTION_CARD_LAYOUT = "overflow: hidden;"
    _SECTION_HEADER_LAYOUT = "padding: 10px 14px;"
    _ROW_STYLE = "border-bottom: 1px solid var(--ql-inset-border); padding: 6px 0;"
    _LABEL_STYLE = (
        "min-width: 210px; max-width: 210px; color: var(--color-text); "
        "word-break: break-all;"
    )

    def __init__(self) -> None:
        self.dialog = None
        self.env_data: Dict[str, str] = {}
        self._env_container = None

    def _save_current(self) -> None:
        try:
            save_env_config(self.env_data)
            ui.notify(t("save_success", "Configuration saved successfully"), type="positive", position="top")
        except Exception as exc:
            ui.notify(f"{t('save_failed', 'Failed to save configuration')}: {exc}", type="negative", position="top")

    def _reload_current(self) -> None:
        self.env_data = load_env_config()
        self._render_env_tab()
        ui.notify(t("reload_success", "Configuration reloaded from disk"), type="info", position="top")

    def _render_env_tab(self) -> None:
        container = self._env_container
        if container is None:
            return
        container.clear()

        predefined_keys = {item["key"] for item in ENV_VAR_DEFINITIONS}
        groups: dict[str, list[dict[str, str]]] = {}
        for definition in ENV_VAR_DEFINITIONS:
            groups.setdefault(definition["group"], []).append(definition)
        custom_keys = [key for key in self.env_data if key not in predefined_keys]

        with container:
            for group_key, items in groups.items():
                self._render_env_group(group_key, items)

            if custom_keys:
                with ui.column().classes("w-full settings-section-card").style(self._SECTION_CARD_LAYOUT):
                    with ui.row().classes("w-full items-center gap-2 settings-section-header").style(
                        self._SECTION_HEADER_LAYOUT
                    ):
                        ui.icon("tune", size="18px").style(f"color: {COLORS['primary']};")
                        ui.label(t("env_group_custom", "Custom")).classes("text-body1 text-weight-bold").style(
                            "color: var(--color-text); font-size: 14px;"
                        )

                    with ui.column().classes("w-full gap-0").style("padding: 12px 16px;"):
                        for env_key in custom_keys:
                            self._render_env_row(env_key, deletable=True)

            with ui.row().classes("w-full justify-center").style("padding-top: 8px;"):
                add_btn = ui.button(icon="add", on_click=self._open_add_env_dialog).props("flat round dense")
                add_btn.style(
                    f"color: {COLORS['primary']}; "
                    "border: 1px dashed var(--ql-card-border); opacity: 0.75;"
                )
                add_btn.tooltip(t("env_add_title", "Add Environment Variable"))

    def _render_env_group(self, group_key: str, items: list[dict[str, str]]) -> None:
        label_key = self._GROUP_LABELS.get(group_key, group_key)
        icon = self._GROUP_ICONS.get(group_key, "settings")
        with ui.column().classes("w-full settings-section-card").style(self._SECTION_CARD_LAYOUT):
            with ui.row().classes("w-full items-center gap-2 settings-section-header").style(
                self._SECTION_HEADER_LAYOUT
            ):
                ui.icon(icon, size="18px").style(f"color: {COLORS['primary']};")
                ui.label(t(label_key, group_key.title())).classes("text-body1 text-weight-bold").style(
                    "color: var(--color-text); font-size: 14px;"
                )

            with ui.column().classes("w-full gap-0").style("padding: 12px 16px;"):
                for definition in items:
                    self._render_env_row(definition["key"], desc=definition)

    def _render_env_row(
        self,
        env_key: str,
        *,
        deletable: bool = False,
        desc: dict[str, str] | None = None,
    ) -> None:
        current_value = self.env_data.get(env_key, "")

        with ui.row().classes("w-full items-center").style(self._ROW_STYLE):
            ui.label(env_key).classes("text-body2 text-weight-medium").style(self._LABEL_STYLE)
            props = "dense outlined type=password autocomplete=off" if self._is_secret_key(env_key) else "dense outlined"
            inp = ui.input(value=current_value, placeholder=t("env_empty_hint", "(empty = not set)")).props(props).classes(
                "flex-1"
            )
            inp.on_value_change(lambda e, key=env_key: self.env_data.__setitem__(key, e.value or ""))

            if deletable:
                del_btn = ui.button(icon="remove_circle_outline", on_click=lambda _e, key=env_key: self._delete_env(key))
                del_btn.props("flat round dense size=sm")
                del_btn.style("color: var(--color-text-secondary); opacity: 0.55; margin-left: 4px;")
                del_btn.tooltip(f"Delete '{env_key}'")

        if desc:
            lang = get_i18n().lang
            desc_key = "desc_zh" if lang == "zh" else "desc_en"
            hint = desc.get(desc_key, desc.get("desc_en", ""))
            if hint:
                ui.label(hint).classes("text-caption").style(
                    "color: var(--color-text-secondary); "
                    "padding-left: 214px; margin-top: -4px; margin-bottom: 4px;"
                )

    def _open_add_env_dialog(self) -> None:
        with ui.dialog() as dlg, ui.card().style(
            "min-width: 340px; background: var(--color-surface) !important; "
            "border: 1px solid var(--ql-card-border);"
        ):
            ui.label(t("env_add_title", "Add Environment Variable")).classes("text-subtitle1 text-weight-bold").style(
                "color: var(--color-text);"
            )
            key_input = ui.input(label="Key", placeholder="MY_ENV_VAR").props("dense outlined").classes("w-full")
            val_input = (
                ui.input(label="Value", placeholder=t("env_empty_hint", "(empty = not set)"))
                .props("dense outlined")
                .classes("w-full q-mt-sm")
            )

            with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):
                ui.button(icon="close", on_click=dlg.close).props("flat dense")
                ui.button(icon="check", on_click=lambda _e: self._add_env_var(dlg, key_input, val_input)).props(
                    "flat dense"
                ).style(f"color: {COLORS['primary']};")
        dlg.open()

    def _add_env_var(self, dialog, key_input, value_input) -> None:
        name = (key_input.value or "").strip()
        if not name:
            ui.notify(t("env_key_required", "Key cannot be empty"), type="warning")
            return
        if "=" in name:
            ui.notify(t("env_key_invalid", "Key cannot contain '='"), type="warning")
            return
        if name in self.env_data:
            ui.notify(t("env_key_exists", "Environment variable already exists"), type="warning")
            return
        self.env_data[name] = (value_input.value or "").strip()
        dialog.close()
        self._render_env_tab()

    def _delete_env(self, env_key: str) -> None:
        self.env_data.pop(env_key, None)
        self._render_env_tab()

    @staticmethod
    def _is_secret_key(env_key: str) -> bool:
        upper = env_key.upper()
        return upper.endswith(("TOKEN", "SECRET", "PASSWORD", "API_KEY")) or "TOKEN" in upper

    def build(self) -> None:
        ui.add_css(_SETTINGS_CSS)
        self.dialog = ui.dialog().props("maximized transition-show=slide-up transition-hide=slide-down")
        with self.dialog, ui.card().classes("w-full h-full q-pa-none settings-dialog-bg").style(
            "display: flex; flex-direction: column; overflow: hidden;"
        ):
            with ui.row().classes("w-full items-center justify-between q-px-md q-py-sm").style(
                f"border-bottom: 1px solid {COLORS['accent']}; flex-shrink: 0; z-index: 10;"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("settings", size="26px").style(f"color: {COLORS['primary']};")
                    with ui.column().classes("gap-0"):
                        ui.label(t("settings_title", "Settings")).classes("text-h6 text-weight-bold").style(
                            "color: var(--color-text);"
                        )
                        ui.label(t("settings_desc", "Manage GUI environment variables.")).classes("text-caption").style(
                            "color: var(--color-text-secondary);"
                        )

                with ui.row().classes("items-center gap-2"):
                    reload_btn = ui.button(t("reload", "Reload"), icon="refresh", on_click=self._reload_current).props(
                        "flat dense"
                    )
                    reload_btn.classes("modern-btn-secondary")
                    reload_btn.classes(remove="bg-primary bg-secondary text-white")

                    save_btn = ui.button(t("save", "Save"), icon="save", on_click=self._save_current).props("dense")
                    save_btn.classes("modern-btn-primary")
                    save_btn.classes(remove="bg-primary bg-secondary text-white")

                    ui.button(icon="close", on_click=self.dialog.close).props("flat round dense").style(
                        f"color: {COLORS['accent']};"
                    )

            with ui.column().classes("w-full").style("flex: 1; overflow: hidden; display: flex; flex-direction: column;"):
                with ui.tabs().classes("w-full").style("flex-shrink: 0;") as tabs:
                    env_tab = ui.tab(t("env_config", "Environment"), icon="tune")

                with ui.tab_panels(tabs, value=env_tab).classes("w-full").style(
                    "flex: 1; overflow: auto; padding: 8px;"
                ):
                    with ui.tab_panel(env_tab).classes("q-pa-sm"):
                        self._env_container = ui.column().classes("w-full gap-3")

    def open(self) -> None:
        if self.dialog is None:
            self.build()
        self.env_data = load_env_config()
        self._render_env_tab()
        self.dialog.open()


def create_settings_dialog() -> SettingsDialog:
    return SettingsDialog()
