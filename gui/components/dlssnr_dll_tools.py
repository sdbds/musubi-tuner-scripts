"""DLL operations independent of training and inference form state."""

from pathlib import Path

from nicegui import ui
from utils.command_builder import CommandBuildError
from utils.dlssnr_commands import (
    PACK_DEFAULTS,
    UNPACK_DEFAULTS,
    build_dlssnr_pack_job,
    build_dlssnr_unpack_job,
    validate_nr_numeric_value,
)
from utils.form_state import FormStateMixin
from utils.i18n import t

from components.advanced_inputs import editable_slider, styled_select
from components.execution_panel import ExecutionPanel
from components.path_selector import create_path_selector


class DLSSNRDLLTools(FormStateMixin):
    def __init__(self, *, project_dir: Path):
        self.project_dir = project_dir
        self.unpack_controls = {}
        self.pack_controls = {}
        self.pack_config = dict(PACK_DEFAULTS)
        with ui.dialog() as self.dialog:
            with (
                ui.column()
                .classes("w-full min-w-0 gap-4 q-pa-md")
                .style(
                    "width: 900px; max-width: calc(100vw - 32px); max-height: calc(100dvh - 32px); "
                    "overflow-y: auto; background: var(--color-surface); border-radius: 8px;"
                )
            ):
                with ui.row().classes("w-full items-center justify-between no-wrap"):
                    ui.label(t("nr_dll_tools")).classes("text-h6 text-weight-bold")
                    ui.button(icon="close", on_click=self.dialog.close).props("flat round dense").tooltip(t("close"))
                with ui.tabs().classes("w-full") as tabs:
                    unpack = ui.tab("unpack", label=t("nr_unpack_dll"), icon="unarchive")
                    pack = ui.tab("pack", label=t("nr_pack_dll"), icon="archive")
                with ui.tab_panels(tabs, value=unpack).classes("w-full min-w-0 shrink-0").style("background: transparent;"):
                    with ui.tab_panel(unpack).classes("p-0"):
                        self._render_unpack()
                    with ui.tab_panel(pack).classes("p-0"):
                        self._render_pack()
                self.exec_panel = ExecutionPanel(show_start=False, height="240px")
                self.exec_panel.register_external_start_button(self.unpack_button)
                self.exec_panel.register_external_start_button(self.pack_button)

    def _path(self, controls, defaults, name, label, *, kind="file", file_filter=None, tooltip=None):
        control = create_path_selector(
            label=t(label),
            default_path=defaults[name],
            selection_type=kind,
            file_filter=file_filter,
        )
        control.input.classes("min-w-0 w-0")
        if tooltip:
            control.input.tooltip(t(tooltip))
        controls[name] = control

    def _render_unpack(self):
        controls, defaults = self.unpack_controls, UNPACK_DEFAULTS
        with ui.column().classes("w-full min-w-0 gap-4"):
            self._path(controls, defaults, "nr_source_dll", "nr_source_dll", file_filter="*.dll")
            self._path(controls, defaults, "nr_output_dir", "nr_unpack_output", kind="save", tooltip="nr_unpack_output_tip")
            with ui.grid().classes("w-full gap-4").style("grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));"):
                controls["nr_weights"] = styled_select(
                    {"all": t("nr_all_weights"), "HT": "HT", "CC": "CC"},
                    label=t("nr_dll_weights"),
                    value=defaults["nr_weights"],
                )
                controls["nr_cc_format"] = styled_select(
                    {"records": t("nr_cc_records"), "canonical": t("nr_cc_canonical")},
                    label=t("nr_cc_format"),
                    value=defaults["nr_cc_format"],
                ).tooltip(t("nr_cc_format_tip"))
                controls["nr_save_precision"] = styled_select(
                    {"native": t("nr_storage_native"), "fp32": "FP32"},
                    label=t("nr_save_precision"),
                    value=defaults["nr_save_precision"],
                )
            self.unpack_button = ui.button(t("nr_unpack_dll"), icon="unarchive", on_click=self._unpack).classes(
                "modern-btn-success self-end"
            )

    def _render_pack(self):
        controls, defaults = self.pack_controls, PACK_DEFAULTS
        with ui.column().classes("w-full min-w-0 gap-4"):
            self._path(controls, defaults, "nr_template_dll", "nr_template_dll", file_filter="*.dll")
            self._path(controls, defaults, "nr_model_dir", "nr_pack_model_dir", kind="dir")
            self._path(controls, defaults, "nr_output_dll", "nr_output_dll", kind="save", tooltip="nr_pack_output_tip")
            with ui.grid().classes("w-full gap-4").style("grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));"):
                controls["nr_weights"] = styled_select(
                    {"": t("nr_dll_auto"), "HT": "HT", "CC": "CC"},
                    label=t("nr_dll_weights"),
                    value=defaults["nr_weights"],
                ).tooltip(t("nr_dll_weights_tip"))
                self._pack_number("nr_mix", "nr_pack_mix", minimum=0, maximum=1).tooltip(t("nr_pack_mix_tip"))
                self._pack_number("nr_strength", "nr_pack_strength").tooltip(t("nr_pack_strength_tip"))
            self._path(controls, defaults, "nr_merge_lora", "nr_merge_lora", file_filter="*.safetensors")
            with ui.grid(columns=1).classes("w-full"):
                self._pack_number("nr_lora_multiplier", "nr_pack_lora_multiplier")
            self.pack_button = (
                ui.button(t("nr_pack_dll"), icon="archive", on_click=self._pack)
                .classes("modern-btn-success self-end")
                .tooltip(t("nr_pack_native_tip"))
            )

    def _pack_number(self, name, label_key, *, minimum=None, maximum=None):
        def validate(value):
            try:
                validate_nr_numeric_value(
                    value,
                    name,
                    minimum=float("-inf") if minimum is None else minimum,
                    maximum=maximum,
                )
            except CommandBuildError as exc:
                return str(exc)
            return None

        control = editable_slider(
            label_key,
            self.pack_config,
            name,
            min_val=-2 if minimum is None else minimum,
            max_val=2 if maximum is None else maximum,
            step=0.05,
            decimals=None,
            hard_min_val=minimum,
            hard_max_val=maximum,
            validate=validate,
        )
        self.pack_controls[name] = control
        return control

    async def _unpack(self):
        await self.run("unpack")

    async def _pack(self):
        await self.run("pack")

    async def run(self, operation: str):
        controls = self.unpack_controls if operation == "unpack" else self.pack_controls
        builder = build_dlssnr_unpack_job if operation == "unpack" else build_dlssnr_pack_job
        try:
            job = builder({name: self._read_control_value(control) for name, control in controls.items()}, self.project_dir)
        except CommandBuildError as exc:
            ui.notify(str(exc), type="negative")
            return
        await self.exec_panel.run_job(
            script_key=job.script_key,
            args=job.args,
            name=job.name,
            runner_kwargs=job.runner_kwargs,
        )
