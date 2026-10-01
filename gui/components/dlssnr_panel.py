"""NR fields rendered inside the same top-level sections as other architectures."""

from __future__ import annotations

from typing import Any, Mapping

from nicegui import ui
from utils.command_builder import CommandBuildError
from utils.dlssnr_commands import (
    GENERATE_DEFAULTS,
    get_nr_optimizer_types,
    get_train_defaults,
    is_nr_optimizer_supported,
    parse_nr_boolean,
    resolve_nr_optimizer,
    validate_nr_numeric_value,
)
from utils.form_state import FormStateMixin
from utils.i18n import t

from components.advanced_inputs import editable_slider, toggle_switch
from components.optimizer_controls import OptimizerControls
from components.path_selector import create_path_selector

_GRID_STYLE = "grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr));"
_DATASET_TEMPLATES = {
    "single_frame": "./toml/qinglong_dlssnr_single.toml",
    "temporal": "./toml/qinglong_dlssnr_temporal.toml",
}


class DLSSNRPanel(FormStateMixin):
    def __init__(self, *, stage: str):
        self.stage = stage
        self.controls: dict[str, Any] = {}
        self.ready = False
        self._train_mode = "lora"
        self._applying_config = False
        self._defaults = get_train_defaults() if stage == "train" else dict(GENERATE_DEFAULTS)
        self.config = dict(self._defaults)
        self._numeric_validators = {}
        self.section_renderers = (
            {
                "model": self._render_model,
                "training": self._render_training,
                "lr": self._render_lr,
                "network": self._render_network,
                "optimizer": self._render_optimizer,
                "save": self._render_save,
                "sample": self._render_sample,
            }
            if stage == "train"
            else {
                "model": self._render_generate_model,
                "generation": self._render_generate_settings,
            }
        )

    def _numeric_validation(self, name, **rules):
        def validate(value):
            try:
                validate_nr_numeric_value(value, name, **rules)
            except CommandBuildError as exc:
                return str(exc)
            return None

        self._numeric_validators[name] = validate
        return validate

    def _number(
        self,
        name: str,
        label_key: str,
        *,
        minimum=0,
        maximum=100,
        step=1,
        integer=False,
        hard_max=None,
        allow_empty=False,
        positive=False,
        exclusive_max=None,
        on_change=None,
    ):
        validation = self._numeric_validation(
            name,
            integer=integer,
            minimum=minimum,
            maximum=hard_max,
            optional=allow_empty,
            positive=positive,
            exclusive_max=exclusive_max,
        )
        control = editable_slider(
            label_key,
            self.config,
            name,
            min_val=minimum,
            max_val=maximum,
            step=step,
            decimals=0 if integer else None,
            hard_max_val=hard_max,
            allow_empty=allow_empty,
            on_change=on_change,
            validate=validation,
        )
        self.controls[name] = control
        return control

    def _path(self, name: str, label_key: str, *, kind="file", file_filter=None):
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

    def _toggle(self, name: str, label_key: str):
        control = toggle_switch(label_key, self.config, name)
        self.controls[name] = control
        return control

    def _grid(self):
        return ui.grid().classes("w-full gap-4").style(_GRID_STYLE)

    def _heading(self, key: str):
        ui.label(t(key)).classes("text-subtitle1 text-weight-bold").style("color: var(--color-text);")

    def render_section(self, section: str):
        with ui.column().classes("w-full min-w-0 gap-5"):
            self.section_renderers[section]()

    def finish_render(self):
        self.ready = True
        if self.stage == "train":
            self._sync_temporal(update_defaults=False)
            self._sync_lora()
        else:
            self._sync_inference()

    def _render_model(self):
        self._heading("dataset_and_model")
        self._path("nr_model_dir", "nr_model_dir", kind="dir")
        self._path("nr_dataset_config", "dataset_config", file_filter="*.toml")
        with self._grid():
            self._select("nr_device", "device", ["auto", "cpu", "cuda"])
            ui.input(t("nr_precision"), value="FP32").props("outlined dense readonly").classes("w-full")

    def _render_training(self):
        self._heading("basic_train_params")
        with self._grid():
            self._select(
                "nr_training_mode",
                "nr_training_mode",
                {"single_frame": t("nr_single_frame"), "temporal": t("nr_temporal")},
                on_change=lambda: self._sync_temporal(),
            )
            self._number("nr_max_train_steps", "max_train_steps", minimum=1, maximum=10000, integer=True)
            self._number("nr_gradient_accumulation_steps", "gradient_accumulation_steps", minimum=1, maximum=16, integer=True)
            self._number("nr_seed", "seed", maximum=2**32 - 1, integer=True)
        with ui.column().classes("w-full gap-4") as self._temporal_section:
            ui.separator()
            with self._grid():
                self._number("nr_sequence_length", "nr_sequence_length", minimum=2, maximum=64, integer=True)
                self._number("nr_burn_in", "nr_burn_in", minimum=1, maximum=32, integer=True)
                self._number("nr_tbptt_length", "nr_tbptt_length", minimum=1, maximum=32, integer=True)
                self._number("nr_loss_temporal", "nr_loss_temporal", maximum=1, step=0.01)
        ui.separator()
        with self._grid():
            self._number("nr_loss_pre", "nr_loss_pre", maximum=2, step=0.05)
            self._number("nr_loss_out", "nr_loss_out", maximum=2, step=0.05)
            self._number("nr_loss_edge", "nr_loss_edge", maximum=1, step=0.01)

    def _render_lr(self):
        self._heading("lr_settings")
        with self._grid():
            self._number("nr_learning_rate", "learning_rate", maximum=1e-3, step=1e-6, positive=True)
            ui.select(["constant"], value="constant", label=t("lr_scheduler")).props("outlined dense readonly").classes("w-full")
        with ui.column().classes("w-full gap-4") as self._full_section:
            ui.separator()
            with self._grid():
                for name in ("prior_lr_multiplier", "scale_lr_multiplier", "temporal_blend_lr_multiplier"):
                    self._number(f"nr_{name}", f"nr_{name}", maximum=1, step=0.05)

    def _render_network(self):
        self._heading("network_settings")
        with ui.column().classes("w-full gap-4") as self._lora_section:
            self._select(
                "nr_lora_profile",
                "nr_lora_profile",
                {"vit_only": t("nr_vit_only"), "multiscale": t("nr_multiscale")},
                on_change=lambda: self._sync_lora(),
            )
            with self._grid() as self._vit_section:
                self._number("nr_network_dim", "network_dim", minimum=1, maximum=128, integer=True, hard_max=1024)
                self._number("nr_network_alpha", "network_alpha", minimum=0, maximum=128, allow_empty=True, positive=True)
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
            self._number("nr_network_dropout", "network_dropout", maximum=0.99, step=0.01, exclusive_max=1)

    def _render_optimizer(self):
        self._heading("optimizer_settings")
        self.optimizer_controls = OptimizerControls(
            options=get_nr_optimizer_types(),
            value=self._defaults["nr_optimizer_type"],
            values=self.config,
            norm_key="nr_max_grad_norm",
            extra_args=self._defaults["nr_optimizer_args"],
            on_change=self._on_optimizer_change,
            on_reset=self.reset_optimizer_args,
            norm_validation=self._numeric_validation("nr_max_grad_norm"),
        )
        for source, target in (
            ("optimizer_type", "nr_optimizer_type"),
            ("max_grad_norm", "nr_max_grad_norm"),
            ("optimizer_extra_args", "nr_optimizer_args"),
            ("d_coef", "nr_d_coef"),
            ("d0", "nr_d0"),
        ):
            self.controls[target] = getattr(self.optimizer_controls, source)

    def reset_optimizer_args(self):
        if self._applying_config or "nr_optimizer_args" not in self.controls:
            return
        try:
            _, args = resolve_nr_optimizer({**self.get_state(), "train_mode": self._train_mode})
        except CommandBuildError as exc:
            ui.notify(str(exc), type="negative")
            return
        self.controls["nr_optimizer_args"].set_value("\n".join(args))

    def _on_optimizer_change(self):
        if self._applying_config or not self.ready:
            return
        self.reset_optimizer_args()
        rate = self.controls["nr_learning_rate"]
        current = self._read_control_value(rate)
        default = get_train_defaults(self._train_mode)["nr_learning_rate"]
        if current in (default, 1.0):
            adaptive = "prodigy" in str(self.controls["nr_optimizer_type"].value).lower()
            self._write_control_value(rate, 1.0 if adaptive else default)

    def _render_save(self):
        self._heading("model_output")
        self.controls["nr_output_name"] = (
            ui.input(
                t("output_model_name"),
                value=self._defaults["nr_output_name"],
            )
            .classes("w-full min-w-0")
            .props("outlined dense")
        )
        self._path("nr_output_dir", "output_dir", kind="dir")
        with self._grid():
            self._number("nr_save_every_n_steps", "save_every_n_steps", maximum=1000, integer=True)
            self._toggle("nr_save_state", "save_state")
        ui.separator()
        self._heading("resume_training")
        self._path("nr_resume", "resume_path", kind="dir")

    def _render_sample(self):
        self._heading("sampling_settings")
        with self._grid():
            self._number("nr_sample_every_n_steps", "sample_every_n_steps", maximum=1000, integer=True)
            self._number("nr_min_sequence_frames", "nr_min_sequence_frames", minimum=1, maximum=256, integer=True)
        self._toggle("nr_compare_baseline", "nr_compare_baseline")

    def _render_generate_model(self):
        self._heading("model_paths")
        self._path("nr_model_dir", "nr_model_dir", kind="dir")
        self._select(
            "nr_inference_mode",
            "nr_inference_mode",
            {"image": t("nr_image"), "sequence": t("nr_sequence_output")},
            on_change=lambda: self._sync_inference(),
        )
        self._path("nr_sample_manifest", "nr_sample_manifest", file_filter="*.jsonl")
        self._path("nr_sequence_manifest", "nr_sequence_manifest", file_filter="*.jsonl")
        self._path("nr_output_dir", "output_dir", kind="dir")

    def _render_generate_settings(self):
        self._heading("generation_params")
        with self._grid():
            self._number("nr_bucket_width", "nr_bucket_width", minimum=33, maximum=2048, integer=True)
            self._number("nr_bucket_height", "nr_bucket_height", minimum=33, maximum=2048, integer=True)
            self._number("nr_seed", "seed", maximum=2**32 - 1, integer=True)
            self._select("nr_device", "device", ["auto", "cpu", "cuda"])

    def _sync_temporal(self, *, update_defaults=True):
        if not self.ready:
            return
        temporal = self.controls["nr_training_mode"].value == "temporal"
        self._temporal_section.visible = temporal
        if update_defaults and not self._applying_config:
            dataset = self.controls["nr_dataset_config"]
            if dataset.value in _DATASET_TEMPLATES.values():
                dataset.value = _DATASET_TEMPLATES["temporal" if temporal else "single_frame"]
            if not temporal:
                self._write_control_value(self.controls["nr_loss_temporal"], 0.0)

    def _sync_lora(self):
        if not self.ready:
            return
        lora = self._train_mode == "lora"
        multiscale = self.controls["nr_lora_profile"].value == "multiscale"
        self._lora_section.visible = lora
        self._full_section.visible = not lora
        self._vit_section.visible = not multiscale
        self._multiscale_section.visible = multiscale

    def sync_train_mode(self, mode: str):
        if not self.ready:
            return
        if mode != self._train_mode:
            old_defaults = get_train_defaults(self._train_mode)
            new_defaults = get_train_defaults(mode)
            for name in ("nr_learning_rate", "nr_output_name"):
                control = self.controls[name]
                if self._read_control_value(control) == old_defaults[name]:
                    self._write_control_value(control, new_defaults[name])
            self._train_mode = mode
        self._sync_optimizer_choices()
        self._sync_lora()

    def _sync_optimizer_choices(self):
        optimizer = self.controls["nr_optimizer_type"]
        options = get_nr_optimizer_types(self._train_mode)
        value = optimizer.value
        if not is_nr_optimizer_supported(value, self._train_mode):
            value = "AdamW"
            ui.notify(t("nr_optimizer_lora_only"), type="warning")
        if value not in options:
            options.append(value)
        optimizer.set_options(options, value=value)

    def _sync_inference(self):
        if not self.ready:
            return
        sequence = self.controls["nr_inference_mode"].value == "sequence"
        self.controls["nr_sample_manifest"].container.visible = not sequence
        self.controls["nr_sequence_manifest"].container.visible = sequence

    def get_state(self) -> dict[str, Any]:
        return {name: self._read_control_value(control) for name, control in self.controls.items()}

    def validate_config(self, config: Mapping[str, Any], *, train_mode="lora"):
        defaults = get_train_defaults(train_mode) if self.stage == "train" else GENERATE_DEFAULTS
        resolved = {**defaults, **config}
        for name, validate in self._numeric_validators.items():
            if error := validate(resolved[name]):
                raise CommandBuildError(error)
        if self.stage == "train":
            resolved["train_mode"] = train_mode
            resolve_nr_optimizer(resolved)
            for name in ("nr_save_state", "nr_compare_baseline"):
                resolved[name] = parse_nr_boolean(resolved[name], name)
        return resolved

    def apply_config(self, config: Mapping[str, Any], *, train_mode="lora"):
        resolved = self.validate_config(config, train_mode=train_mode)
        if self.stage == "train" and "nr_optimizer_args" not in config:
            _, args = resolve_nr_optimizer(resolved)
            resolved["nr_optimizer_args"] = "\n".join(args)
        self._applying_config = True
        if self.stage == "train":
            self.optimizer_controls.suspend_updates = True
        try:
            for name, control in self.controls.items():
                value = resolved[name]
                if name == "nr_optimizer_type" and value not in control.options:
                    control.options.append(value)
                    control.update()
                self._write_control_value(control, value)
        finally:
            self._applying_config = False
            if self.stage == "train":
                self.optimizer_controls.suspend_updates = False
        if self.stage == "train":
            self._train_mode = train_mode
            self._sync_optimizer_choices()
            self._sync_temporal(update_defaults=False)
            self._sync_lora()
            self.optimizer_controls.sync_visibility()
            self.optimizer_controls.sync_coefficients_from_args()
        else:
            self._sync_inference()
