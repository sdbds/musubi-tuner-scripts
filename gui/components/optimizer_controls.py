"""Shared optimizer selector, argument editor and parameter controls."""

from __future__ import annotations

from typing import Callable

from nicegui import ui
from utils.i18n import t
from utils.optimizer_catalog import parse_optimizer_args

from components.advanced_inputs import editable_slider, styled_select


class OptimizerControls:
    def __init__(
        self,
        *,
        options: list[str],
        value: str,
        values: dict,
        norm_key: str = "max_grad_norm",
        extra_args: str = "",
        d_coef: str = "0.5",
        d0: str = "1e-3",
        on_change: Callable,
        on_reset: Callable,
        norm_validation: Callable | None = None,
    ):
        self._on_change = on_change
        self.suspend_updates = False
        with ui.grid().classes("w-full gap-4").style("grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));"):
            self.optimizer_type = styled_select(
                list(options),
                label=t("optimizer_type"),
                value=value,
                new_value_mode="add-unique",
                on_change=self._type_changed,
            )
            self.max_grad_norm = editable_slider(
                "max_grad_norm",
                values,
                norm_key,
                min_val=0,
                max_val=10,
                step=0.1,
                decimals=None,
                hard_max_val=None,
                validate=norm_validation,
            )
        with ui.row().classes("w-full gap-4") as self.coefficient_row:
            self.d_coef = ui.input(t("d_coef"), value=d_coef).classes("flex-1 min-w-0").props("outlined dense")
            self.d_coef.tooltip(t("d_coef_tooltip"))
            self.d0 = ui.input(t("d0"), value=d0).classes("flex-1 min-w-0").props("outlined dense")
            self.d0.tooltip(t("d0_tooltip"))
        with ui.row().classes("w-full items-start no-wrap gap-2"):
            self.optimizer_extra_args = (
                ui.textarea(
                    t("optimizer_extra_args"),
                    value=extra_args,
                    placeholder="key=value",
                )
                .classes("flex-1 min-w-0")
                .props("autogrow outlined")
            )
            self.reset_button = ui.button(icon="restart_alt", on_click=on_reset).classes("modern-btn-ghost")
            self.reset_button.props("dense").tooltip(t("reset_optimizer_template"))
        self.d_coef.on_value_change(lambda event: self._coefficient_changed("d_coef", event.value))
        self.d0.on_value_change(lambda event: self._coefficient_changed("d0", event.value))
        self.optimizer_extra_args.on_value_change(lambda _event: self.sync_coefficients_from_args())
        self.sync_visibility()

    def sync_visibility(self):
        if hasattr(self, "coefficient_row"):
            self.coefficient_row.visible = "prodigy" in str(self.optimizer_type.value).lower()

    def _type_changed(self, _event):
        self.sync_visibility()
        if not self.suspend_updates:
            self._on_change()

    def sync_coefficients_from_args(self):
        if self.suspend_updates or not self.coefficient_row.visible:
            return
        try:
            args = dict(token.split("=", 1) for token in parse_optimizer_args(self.optimizer_extra_args.value))
        except ValueError:
            return
        self.suspend_updates = True
        try:
            for name in ("d_coef", "d0"):
                getattr(self, name).set_value(args.get(name, "").strip())
        finally:
            self.suspend_updates = False

    def _coefficient_changed(self, name, value):
        if self.suspend_updates or not self.coefficient_row.visible:
            return
        try:
            tokens = parse_optimizer_args(self.optimizer_extra_args.value)
        except ValueError:
            return
        tokens = [token for token in tokens if token.split("=", 1)[0].strip() != name]
        if str(value or "").strip():
            tokens.append(f"{name}={value}")
        self.optimizer_extra_args.set_value("\n".join(tokens))
