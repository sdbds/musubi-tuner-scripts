"""Manifest-based NR forms which never share diffusion controls."""

from __future__ import annotations

from typing import Any, Mapping

from nicegui import ui
from utils.dlssnr_commands import GENERATE_DEFAULTS, get_train_defaults
from utils.i18n import t

from components.path_selector import create_path_selector

_GRID_STYLE = "grid-template-columns: repeat(auto-fit, minmax(min(100%, 260px), 1fr));"
_DATASET_TEMPLATES = {
    "single_frame": "./toml/qinglong_dlssnr_single.toml",
    "temporal": "./toml/qinglong_dlssnr_temporal.toml",
}


class DLSSNRPanel:
    def __init__(self, *, stage: str):
        self.stage = stage
        self.controls: dict[str, Any] = {}
        self._train_mode = "lora"
        self._applying_config = False
        self._defaults = get_train_defaults() if stage == "train" else dict(GENERATE_DEFAULTS)

    def _number(self, name: str, label_key: str, *, minimum: float = 0, step: float = 1, maximum=None):
        control = (
            ui.number(
                t(label_key),
                value=self._defaults[name],
                min=minimum,
                max=maximum,
                step=step,
            )
            .classes("w-full min-w-0")
            .props("outlined dense")
        )
        self.controls[name] = control
        return control

    def _path(self, name: str, label_key: str, *, kind: str = "file", file_filter: str | None = None):
        control = create_path_selector(
            label=t(label_key),
            default_path=self._defaults[name],
            selection_type=kind,
            file_filter=file_filter,
        )
        control.input.classes("min-w-0 w-0")
        self.controls[name] = control
        return control

    def _select(self, name: str, label_key: str, options, *, on_change=None):
        control = (
            ui.select(
                options,
                label=t(label_key),
                value=self._defaults[name],
                on_change=on_change,
            )
            .classes("w-full min-w-0")
            .props("outlined dense")
        )
        self.controls[name] = control
        return control

    def _checkbox(self, name: str, label_key: str):
        control = ui.checkbox(t(label_key), value=self._defaults[name])
        self.controls[name] = control
        return control

    def render(self):
        with ui.row().classes("w-full items-center gap-2"):
            ui.badge("DLSS-NR 310.8.0")
            ui.badge("FP32", color="secondary")
            ui.badge(t("nr_surrogate"), color="warning")
        if self.stage == "train":
            self._render_train()
        else:
            self._render_generate()

    def _render_train(self):
        with ui.tabs().classes("w-full") as tabs:
            sources = ui.tab(t("dataset_and_model"), icon="folder")
            training = ui.tab(t("training_params"), icon="trending_up")
            optimizer = ui.tab(t("optimizer_settings"), icon="speed")
            network = ui.tab(t("network_settings"), icon="hub")
            evaluation = ui.tab(t("sampling_settings"), icon="fact_check")
        with ui.tab_panels(tabs, value=sources).classes("w-full"):
            with ui.tab_panel(sources).classes("p-0 py-4"):
                with ui.grid().classes("w-full gap-4").style(_GRID_STYLE):
                    self._path("nr_dataset_config", "dataset_config", file_filter="*.toml")
                    self._path("nr_model_dir", "nr_model_dir", kind="dir")
                    self._path("nr_forward_validation_report", "nr_forward_validation_report", file_filter="*.json")
                    self._select("nr_device", "device", ["auto", "cpu", "cuda"])
                    self._select("nr_deployment_target", "nr_deployment_target", ["float_runtime", "native_roundtrip"])
                    self._checkbox("nr_development_smoke", "nr_development_smoke").tooltip(t("nr_smoke_warning"))
                ui.separator().classes("my-4")
                with ui.grid().classes("w-full gap-4").style(_GRID_STYLE):
                    self.controls["nr_output_name"] = (
                        ui.input(
                            t("output_model_name"),
                            value=self._defaults["nr_output_name"],
                        )
                        .classes("w-full min-w-0")
                        .props("outlined dense")
                    )
                    self._path("nr_output_dir", "output_dir", kind="dir")
                    self._path("nr_resume", "resume_path", kind="dir")
                    self._number("nr_save_every_n_steps", "save_every_n_steps")
                    self._checkbox("nr_save_state", "save_state")
            with ui.tab_panel(training).classes("p-0 py-4"):
                with ui.grid().classes("w-full gap-4").style(_GRID_STYLE):
                    self._select(
                        "nr_training_mode",
                        "nr_training_mode",
                        {
                            "single_frame": t("nr_single_frame"),
                            "temporal": t("nr_temporal"),
                        },
                        on_change=lambda: self._sync_temporal(),
                    )
                    self._number("nr_max_train_steps", "max_train_steps", minimum=1)
                    self._number("nr_gradient_accumulation_steps", "gradient_accumulation_steps", minimum=1)
                    self._number("nr_seed", "seed")
                with ui.grid().classes("w-full gap-4 mt-4").style(_GRID_STYLE) as self._temporal_section:
                    self._number("nr_sequence_length", "nr_sequence_length", minimum=2)
                    self._number("nr_burn_in", "nr_burn_in", minimum=1)
                    self._number("nr_tbptt_length", "nr_tbptt_length", minimum=1)
                    self._number("nr_loss_temporal", "nr_loss_temporal", step=0.01)
                ui.separator().classes("my-4")
                with ui.grid().classes("w-full gap-4").style(_GRID_STYLE):
                    self._number("nr_loss_pre", "nr_loss_pre", step=0.05)
                    self._number("nr_loss_out", "nr_loss_out", step=0.05)
                    self._number("nr_loss_edge", "nr_loss_edge", step=0.01)
            with ui.tab_panel(optimizer).classes("p-0 py-4"):
                with ui.grid().classes("w-full gap-4").style(_GRID_STYLE):
                    self.controls["nr_optimizer_type"] = (
                        ui.select(
                            ["AdamW", "SGD", "AdamW8bit", "Adafactor"],
                            label=t("optimizer_type"),
                            value=self._defaults["nr_optimizer_type"],
                            with_input=True,
                            new_value_mode="add-unique",
                        )
                        .classes("w-full min-w-0")
                        .props("outlined dense")
                    )
                    self._number("nr_learning_rate", "learning_rate", minimum=0, step=1e-5)
                    self._number("nr_max_grad_norm", "max_grad_norm", step=0.1)
                    ui.input(t("lr_scheduler"), value="constant").props("outlined dense readonly").classes("w-full")
                self.controls["nr_optimizer_args"] = (
                    ui.textarea(
                        t("optimizer_extra_args"),
                        value=self._defaults["nr_optimizer_args"],
                        placeholder="weight_decay=0.0\nbetas=0.9,0.999",
                    )
                    .classes("w-full mt-4")
                    .props("outlined autogrow")
                )
            with ui.tab_panel(network).classes("p-0 py-4"):
                with ui.column().classes("w-full gap-4") as self._lora_section:
                    self._select(
                        "nr_lora_profile",
                        "nr_lora_profile",
                        {
                            "vit_only": t("nr_vit_only"),
                            "multiscale": t("nr_multiscale"),
                        },
                        on_change=lambda: self._sync_lora(),
                    )
                    with ui.grid().classes("w-full gap-4").style(_GRID_STYLE) as self._vit_section:
                        self._number("nr_network_dim", "network_dim", minimum=1, maximum=1024)
                        self.controls["nr_network_alpha"] = (
                            ui.input(
                                t("network_alpha"),
                                value="",
                                placeholder=t("nr_alpha_auto"),
                            )
                            .classes("w-full")
                            .props("outlined dense type=number")
                        )
                    with ui.column().classes("w-full gap-4") as self._multiscale_section:
                        for name in ("rank_by_width", "alpha_by_width"):
                            self.controls[f"nr_{name}"] = (
                                ui.textarea(
                                    t(f"nr_{name}"),
                                    value="",
                                    placeholder='{"32": 2, "64": 4, "128": 8, "256": 8, "512": 16, "1024": 16}',
                                )
                                .classes("w-full")
                                .props("outlined autogrow")
                            )
                    self._number("nr_network_dropout", "network_dropout", step=0.01, maximum=0.99)
                with ui.grid().classes("w-full gap-4").style(_GRID_STYLE) as self._full_section:
                    for name in ("prior_lr_multiplier", "scale_lr_multiplier", "temporal_blend_lr_multiplier"):
                        self._number(f"nr_{name}", f"nr_{name}", step=0.05)
            with ui.tab_panel(evaluation).classes("p-0 py-4"):
                with ui.grid().classes("w-full gap-4").style(_GRID_STYLE):
                    self._number("nr_sample_every_n_steps", "sample_every_n_steps")
                    self._number("nr_min_sequence_frames", "nr_min_sequence_frames", minimum=1)
                    self._checkbox("nr_compare_baseline", "nr_compare_baseline")
        self._sync_temporal(update_defaults=False)
        self._sync_lora()

    def _render_generate(self):
        with ui.grid().classes("w-full gap-4 mt-4").style(_GRID_STYLE):
            self._select(
                "nr_inference_mode",
                "nr_inference_mode",
                {
                    "image": t("nr_image"),
                    "sequence": t("nr_sequence_output"),
                },
                on_change=lambda: self._sync_inference(),
            )
            self._select("nr_device", "device", ["auto", "cpu", "cuda"])
            self._path("nr_model_dir", "nr_model_dir", kind="dir")
            self._path("nr_output_dir", "output_dir", kind="dir")
            self._path("nr_sample_manifest", "nr_sample_manifest", file_filter="*.jsonl")
            self._path("nr_sequence_manifest", "nr_sequence_manifest", file_filter="*.jsonl")
            self._number("nr_bucket_width", "nr_bucket_width", minimum=33, step=16)
            self._number("nr_bucket_height", "nr_bucket_height", minimum=33, step=16)
            self._number("nr_seed", "seed")
        self._sync_inference()

    def _sync_temporal(self, *, update_defaults: bool = True):
        temporal = self.controls["nr_training_mode"].value == "temporal"
        self._temporal_section.visible = temporal
        if update_defaults and not self._applying_config:
            dataset = self.controls["nr_dataset_config"]
            if dataset.value in _DATASET_TEMPLATES.values():
                dataset.value = _DATASET_TEMPLATES["temporal" if temporal else "single_frame"]
            if not temporal:
                self.controls["nr_loss_temporal"].set_value(0.0)

    def _sync_lora(self):
        lora = self._train_mode == "lora"
        multiscale = self.controls["nr_lora_profile"].value == "multiscale"
        self._lora_section.visible = lora
        self._full_section.visible = not lora
        self._vit_section.visible = not multiscale
        self._multiscale_section.visible = multiscale

    def sync_train_mode(self, mode: str):
        if mode != self._train_mode:
            old_defaults = get_train_defaults(self._train_mode)
            new_defaults = get_train_defaults(mode)
            for name in ("nr_learning_rate", "nr_output_name"):
                control = self.controls[name]
                if control.value == old_defaults[name]:
                    control.set_value(new_defaults[name])
            self._train_mode = mode
        self._sync_lora()

    def _sync_inference(self):
        sequence = self.controls["nr_inference_mode"].value == "sequence"
        self.controls["nr_sample_manifest"].container.visible = not sequence
        self.controls["nr_sequence_manifest"].container.visible = sequence

    def get_state(self) -> dict[str, Any]:
        return {name: control.value for name, control in self.controls.items()}

    def apply_config(self, config: Mapping[str, Any], *, train_mode: str = "lora"):
        defaults = get_train_defaults(train_mode) if self.stage == "train" else GENERATE_DEFAULTS
        resolved = {**defaults, **config}
        self._applying_config = True
        try:
            for name, control in self.controls.items():
                value = resolved[name]
                if name == "nr_optimizer_type" and value not in control.options:
                    control.options.append(value)
                    control.update()
                if hasattr(control, "set_value"):
                    control.set_value(value)
                else:
                    control.value = value
        finally:
            self._applying_config = False
        if self.stage == "train":
            self._train_mode = train_mode
            self._sync_temporal(update_defaults=False)
            self._sync_lora()
        else:
            self._sync_inference()
