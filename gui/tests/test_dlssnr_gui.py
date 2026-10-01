import sys
from pathlib import Path

import pytest
from nicegui import ui

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))

from components.model_selector import create_model_selector  # noqa: E402
from utils.command_builder import CommandBuildError  # noqa: E402
from utils.config_manager import ConfigManager  # noqa: E402
from utils.i18n import TRANSLATIONS, t  # noqa: E402
from wizard.step3_train import TrainStep  # noqa: E402
from wizard.step4_generate import GenerateStep  # noqa: E402


@pytest.fixture
def train_step():
    step = TrainStep()
    with ui.column() as container:
        step.render()
    try:
        step.model_selector.set_arch("DLSS-NR")
        yield step
    finally:
        container.delete()


@pytest.fixture
def generate_step():
    step = GenerateStep()
    with ui.column() as container:
        step.render()
    try:
        step.model_selector.set_arch("DLSS-NR")
        yield step
    finally:
        container.delete()


def test_train_page_exposes_only_dlssnr_controls_and_preserves_diffusion_tabs(train_step):
    assert all(section.visible for section in train_step._nr_sections.values())
    assert all(not section.visible for section in train_step._diffusion_sections.values())
    assert train_step.train_mode.options.keys() == {"lora", "finetune"}
    state = train_step._get_config()
    assert "nr_development_smoke" not in state
    assert state["nr_learning_rate"] == 1e-4
    assert "mixed_precision" not in state
    assert "gradient_checkpointing" not in state
    train_step.model_selector.set_arch("FLUX.2")
    assert all(not section.visible for section in train_step._nr_sections.values())
    assert train_step._diffusion_tabs[0].visible
    assert train_step._tab_network.visible


def test_lora_full_toggle_sets_mode_defaults_but_preserves_custom_rate(train_step):
    panel = train_step._dlssnr_panel
    train_step.train_mode.set_value("finetune")
    assert panel.get_state()["nr_learning_rate"] == 1e-5
    assert not panel._lora_section.visible
    assert panel._full_section.visible
    panel.controls["nr_learning_rate"].set_value(3e-5)
    train_step.train_mode.set_value("lora")
    assert panel.get_state()["nr_learning_rate"] == 3e-5
    assert panel._lora_section.visible
    assert not panel._full_section.visible


def test_temporal_mode_switch_changes_template_and_resets_single_frame_loss(train_step):
    panel = train_step._dlssnr_panel
    panel.controls["nr_training_mode"].set_value("temporal")
    assert panel._temporal_section.visible
    assert panel.get_state()["nr_dataset_config"] == "./toml/qinglong_dlssnr_temporal.toml"
    panel.controls["nr_loss_temporal"].set_value(0.1)
    panel.controls["nr_training_mode"].set_value("single_frame")
    assert not panel._temporal_section.visible
    assert panel.get_state()["nr_loss_temporal"] == 0.0
    assert "nr_development_smoke" not in panel.get_state()


def test_multiscale_controls_replace_scalar_rank_inputs(train_step):
    panel = train_step._dlssnr_panel
    panel.controls["nr_lora_profile"].set_value("multiscale")
    assert panel._multiscale_section.visible
    assert not panel._vit_section.visible


def test_train_preset_roundtrip_preserves_explicit_values_and_resets_mode_defaults(train_step):
    custom = {
        "arch": "DLSS-NR",
        "train_mode": "finetune",
        "nr_training_mode": "temporal",
        "nr_model_dir": "D:/models/custom canonical",
        "nr_learning_rate": 2e-5,
        "nr_output_name": "custom_run",
        "nr_dataset_config": "D:/paired data/dataset.toml",
        "nr_sequence_length": 6,
        "nr_burn_in": 2,
        "nr_tbptt_length": 4,
    }
    train_step._apply_config(custom)
    state = train_step._get_config()
    for key, value in custom.items():
        assert state[key] == value
    train_step._apply_config({"arch": "DLSS-NR", "train_mode": "lora"})
    assert "nr_development_smoke" not in train_step._get_config()
    assert train_step._get_config()["nr_learning_rate"] == 1e-4


def test_generate_page_hides_diffusion_inputs_and_keeps_image_sequence_manifests(generate_step):
    panel = generate_step._dlssnr_panel
    assert all(section.visible for section in generate_step._nr_sections.values())
    assert all(not section.visible for section in generate_step._diffusion_sections.values())
    assert not generate_step._page_description.visible
    panel.controls["nr_sample_manifest"].value = "stills.jsonl"
    panel.controls["nr_sequence_manifest"].value = "clip.jsonl"
    panel.controls["nr_inference_mode"].set_value("sequence")
    assert panel.controls["nr_sequence_manifest"].container.visible
    assert not panel.controls["nr_sample_manifest"].container.visible
    assert generate_step._get_config()["nr_sequence_manifest"] == "clip.jsonl"
    generate_step.model_selector.set_arch("Mage-Flow")
    assert all(not section.visible for section in generate_step._nr_sections.values())
    assert generate_step._page_description.visible
    assert generate_step._tab_prompt.visible
    assert not generate_step._tab_inference.visible
    generate_step.model_selector.set_arch("DLSS-NR")
    assert panel.get_state()["nr_sample_manifest"] == "stills.jsonl"
    assert panel.get_state()["nr_sequence_manifest"] == "clip.jsonl"


def test_generate_preset_roundtrip_does_not_add_prompt_or_lora_fields(generate_step):
    config = {
        "arch": "DLSS-NR",
        "nr_inference_mode": "sequence",
        "nr_sequence_manifest": "frames.jsonl",
        "nr_bucket_width": 640,
        "nr_bucket_height": 384,
        "nr_model_dir": "models/merged",
        "nr_output_dir": "out/frames",
        "nr_seed": 123,
    }
    generate_step._apply_config(config)
    state = generate_step._get_config()
    for key, value in config.items():
        assert state[key] == value
    assert "prompt" not in state and "lora_weight" not in state


def test_cache_model_selector_cannot_select_dlssnr():
    with ui.column() as container:
        selector = create_model_selector(default_arch="FLUX.2", page_key="cache")
    try:
        assert "DLSS-NR" not in selector.arch_select.options
        selector.set_arch("DLSS-NR")
        assert selector.arch == "FLUX.2"
    finally:
        container.delete()


def test_dlssnr_labels_and_preset_labels_exist_in_all_languages():
    keys = {
        "nr_model_dir",
        "nr_training_mode",
        "nr_inference_mode",
        "nr_sample_manifest",
        "nr_sequence_manifest",
        "nr_lora_profile",
        "nr_loss_pre",
        "nr_loss_out",
        "nr_loss_edge",
        "nr_loss_temporal",
        "nr_compare_baseline",
    }
    for lang, values in TRANSLATIONS.items():
        assert "dlssnr" in values["model_architecture_list"], lang
        assert all(values.get(key) for key in keys), lang


def test_builtin_presets_do_not_expose_validation_or_experiment_gates():
    manager = ConfigManager()
    for name in ("dlssnr_lora", "dlssnr_full", "dlssnr_lora_temporal", "dlssnr_full_temporal"):
        preset = manager.load_config("train", name)
        assert preset is not None, name
        assert preset["arch"] == "DLSS-NR"
        assert not {"nr_development_smoke", "nr_forward_validation_report", "nr_deployment_target"} & preset.keys()
    for name, mode in (("dlssnr_image", "image"), ("dlssnr_sequence", "sequence")):
        preset = manager.load_config("generate", name)
        assert preset is not None, name
        assert preset["nr_inference_mode"] == mode


def test_nr_uses_existing_top_level_train_sections(train_step):
    labels = {tab.props["name"] for tab in train_step._tabs.default_slot.children if tab.visible}
    assert labels == {
        t(key)
        for key in (
            "basic_settings",
            "model_settings",
            "basic_train_params",
            "lr_settings",
            "network_settings",
            "optimizer_settings",
            "memory_optimization",
            "save_precision",
            "sampling_settings",
        )
    }


def test_nr_uses_common_optimizer_catalog_and_live_templates(train_step):
    panel = train_step._dlssnr_panel
    optimizer = panel.controls["nr_optimizer_type"]
    assert {"AdamW_adv", "Prodigy_adv", "Lion", "SOAP", "Fira", "DAdaptAdam"} <= set(optimizer.options)
    assert "AdamWScheduleFree" not in optimizer.options
    optimizer.set_value("AdamW_adv")
    assert "betas=.95,.98" in panel.get_state()["nr_optimizer_args"]
    optimizer.set_value("adafactor")
    assert "relative_step=False" in panel.get_state()["nr_optimizer_args"]
    assert "warmup_init=False" in panel.get_state()["nr_optimizer_args"]


def test_nr_old_gate_fields_are_not_present_or_exported(train_step):
    train_step._apply_config(
        {
            "arch": "DLSS-NR",
            "nr_development_smoke": True,
            "nr_forward_validation_report": "old-report.json",
            "nr_deployment_target": "native_roundtrip",
        }
    )
    state = train_step._get_config()
    assert not {"nr_development_smoke", "nr_forward_validation_report", "nr_deployment_target"} & state.keys()
    assert not {"nr_development_smoke", "nr_forward_validation_report"} & train_step._dlssnr_panel.controls.keys()


def test_nr_numeric_controls_keep_exact_bound_values(train_step):
    panel = train_step._dlssnr_panel
    train_step._apply_config({"arch": "DLSS-NR", "nr_seed": 9007199254740993, "nr_learning_rate": 1.23456789e-6})
    assert hasattr(panel.controls["nr_seed"], "get_bound_value")
    assert panel.get_state()["nr_seed"] == 9007199254740993
    assert panel.get_state()["nr_learning_rate"] == 1.23456789e-6


def test_nr_generation_uses_existing_top_level_sections(generate_step):
    labels = {tab.props["name"] for tab in generate_step._tabs.default_slot.children if tab.visible}
    assert labels == {t("basic_settings"), t("model_paths"), t("generation_params"), t("inference_settings")}


def test_optimizer_coefficient_edits_update_only_the_corresponding_argument(train_step):
    panel = train_step._dlssnr_panel
    panel.controls["nr_optimizer_type"].set_value("Prodigy_adv")
    panel.controls["nr_optimizer_args"].set_value("d_coef=0.7 d0=0.004 weight_decay=0.03\nbetas=(0.8, 0.99)")
    assert panel.controls["nr_d_coef"].value == "0.7"
    assert panel.controls["nr_d0"].value == "0.004"
    panel.controls["nr_d_coef"].set_value("0.9")
    state = panel.get_state()
    assert "d_coef=0.9" in state["nr_optimizer_args"]
    assert "d0=0.004" in state["nr_optimizer_args"]
    assert "weight_decay=0.03" in state["nr_optimizer_args"]
    assert "betas=(0.8, 0.99)" in state["nr_optimizer_args"]
    assert state["nr_learning_rate"] == 1.0


def test_optimizer_custom_preset_is_not_overwritten_by_control_events(train_step):
    args = "d_coef=0.9\nd0=0.004\nweight_decay=0.03"
    train_step._apply_config(
        {
            "arch": "DLSS-NR",
            "nr_optimizer_type": "Prodigy_adv",
            "nr_learning_rate": 0.2,
            "nr_optimizer_args": args,
            "nr_d_coef": "0.5",
            "nr_d0": "1e-3",
        }
    )
    state = train_step._get_config()
    assert state["nr_optimizer_args"] == args
    assert state["nr_d_coef"] == "0.9"
    assert state["nr_learning_rate"] == 0.2


@pytest.mark.parametrize(
    "override",
    [
        {"nr_max_train_steps": 1.5},
        {"nr_learning_rate": "nan"},
        {"nr_loss_edge": -0.5},
        {"nr_network_alpha": 0},
        {"nr_network_dropout": 1.0},
    ],
)
def test_invalid_numeric_preset_is_rejected_without_silently_rewriting_it(train_step, override):
    before = train_step._get_config()
    with pytest.raises(CommandBuildError):
        train_step._apply_config({"arch": "DLSS-NR", **override})
    assert train_step._get_config() == before


def test_nr_numeric_edit_does_not_truncate_fractional_steps(train_step):
    control = train_step._dlssnr_panel.controls["nr_max_train_steps"]
    control.set_bound_value(1.5)
    assert control.get_bound_value() == 1000


def test_nr_inference_geometry_preserves_valid_non_bucket_sizes(generate_step):
    generate_step._apply_config({"arch": "DLSS-NR", "nr_bucket_width": 33, "nr_bucket_height": 37})
    state = generate_step._get_config()
    assert (state["nr_bucket_width"], state["nr_bucket_height"]) == (33, 37)


def test_nr_dropout_preserves_all_values_below_one(train_step):
    train_step._apply_config({"arch": "DLSS-NR", "nr_network_dropout": 0.995})
    assert train_step._get_config()["nr_network_dropout"] == 0.995


def test_nr_boolean_preset_strings_keep_their_actual_boolean_meaning(train_step):
    train_step._apply_config({"arch": "DLSS-NR", "nr_save_state": "false", "nr_compare_baseline": "false"})
    state = train_step._get_config()
    assert state["nr_save_state"] is False
    assert state["nr_compare_baseline"] is False
    with pytest.raises(CommandBuildError, match="boolean"):
        train_step._apply_config({"arch": "DLSS-NR", "nr_save_state": "maybe"})
    assert train_step._get_config() == state


def test_lorarite_is_not_offered_or_kept_when_switching_to_full_training(train_step):
    panel = train_step._dlssnr_panel
    optimizer = panel.controls["nr_optimizer_type"]
    optimizer.set_value("LoRARite")
    train_step.train_mode.set_value("finetune")
    assert "LoRARite" not in optimizer.options
    assert panel.get_state()["nr_optimizer_type"] == "AdamW"


def test_rejected_preset_reports_an_error_without_changing_the_form(train_step, tmp_path, monkeypatch):
    from components import preset_manager

    storage = ConfigManager(builtin_dir=str(tmp_path / "builtin"), user_dir=str(tmp_path / "user"))
    assert storage.save_config("train", "invalid-nr", {"arch": "DLSS-NR", "nr_max_train_steps": 1.5})
    monkeypatch.setattr(preset_manager, "config_manager", storage)
    notifications = []
    monkeypatch.setattr(ui, "notify", lambda message, **options: notifications.append((str(message), options.get("type"))))
    before = train_step._get_config()
    with ui.column() as container:
        manager = preset_manager.PresetManager(train_step._get_config, train_step._apply_config, "train", default_name="invalid-nr")
    try:
        manager._apply_selected()
        assert train_step._get_config() == before
        assert len(notifications) == 1
        assert notifications[0][1] == "negative"
        assert "nr_max_train_steps" in notifications[0][0]
    finally:
        container.delete()


def test_invalid_pending_numeric_edit_cannot_launch_with_the_old_value(train_step, monkeypatch):
    panel = train_step._dlssnr_panel
    container = panel.controls["nr_max_train_steps"].parent_slot.parent
    button = next(element for element in container.descendants() if type(element).__name__ == "Button")
    edit_input = next(element for element in container.descendants() if type(element).__name__ == "Input")
    click = next(listener.handler for listener in button._event_listeners.values() if listener.type == "click")
    submit = next(listener.handler for listener in edit_input._event_listeners.values() if listener.type == "keyup.enter")
    monkeypatch.setattr(ui, "run_javascript", lambda *_args: None)
    click(None)
    edit_input.set_value("1.5")
    submit()
    assert panel.get_state()["nr_max_train_steps"] == "1.5"
    with pytest.raises(CommandBuildError, match="nr_max_train_steps"):
        train_step._get_config()
    edit_input.set_value("2000")
    submit()
    assert train_step._get_config()["nr_max_train_steps"] == 2000


@pytest.mark.parametrize("train_mode", ["lora", "finetune"])
def test_initial_training_form_uses_checkpoint_sdpa_without_enabling_amp_or_fp8(train_step, train_mode):
    train_step.train_mode.set_value(train_mode)
    panel = train_step._dlssnr_panel
    state = train_step._get_config()
    assert state["nr_gradient_checkpointing"] is True
    assert state["nr_attention_backend"] == "sdpa"
    assert state["nr_numerics_profile"] == "train_experimental"
    assert state["nr_mixed_precision"] == "no"
    assert state["nr_fp8_base"] is False
    assert state["nr_fp8_scaled"] is False
    assert state["nr_num_processes"] == 1
    assert panel.controls["nr_attention_backend"].options["sdpa"] == "PyTorch SDPA"


def test_explicit_legacy_runtime_preset_is_preserved(train_step, tmp_path):
    manager = ConfigManager(builtin_dir=str(tmp_path / "builtin"), user_dir=str(tmp_path / "user"))
    policy = {
        "arch": "DLSS-NR",
        "nr_numerics_profile": "train_surrogate",
        "nr_attention_backend": "native",
        "nr_gradient_checkpointing": False,
    }
    assert manager.save_config("train", "legacy-runtime", policy)
    train_step._apply_config(manager.load_config("train", "legacy-runtime"))
    train_step.model_selector.set_arch("FLUX.2")
    train_step.model_selector.set_arch("DLSS-NR")
    state = train_step._get_config()
    assert {key: state[key] for key in policy} == policy
    assert manager.load_config("train", "legacy-runtime") == policy


def test_runtime_controls_preserve_baseline_and_enable_checkpoint_without_experimental_mode(train_step):
    panel = train_step._dlssnr_panel
    train_step._apply_config(
        {
            "arch": "DLSS-NR",
            "nr_numerics_profile": "train_surrogate",
            "nr_attention_backend": "native",
            "nr_gradient_checkpointing": False,
        }
    )
    state = train_step._get_config()
    assert state["nr_numerics_profile"] == "train_surrogate"
    assert state["nr_mixed_precision"] == "no"
    assert state["nr_attention_backend"] == "native"
    assert state["nr_num_processes"] == 1
    assert not panel.controls["nr_mixed_precision"].enabled
    assert not panel.controls["nr_fp8_base"].enabled
    panel.controls["nr_gradient_checkpointing"].set_toggle_value(True)
    state = train_step._get_config()
    assert state["nr_gradient_checkpointing"] is True
    assert state["nr_numerics_profile"] == "train_surrogate"
    assert train_step._section_tabs["memory"].visible


def test_runtime_selection_links_flash_scope_fp8_and_full_training(train_step):
    panel = train_step._dlssnr_panel
    panel.controls["nr_numerics_profile"].set_value("train_experimental")
    panel.controls["nr_mixed_precision"].set_value("bf16")
    panel.controls["nr_attention_backend"].set_value("flash_attn")
    assert panel.get_state()["nr_attention_scope"] == "global"
    assert not panel.controls["nr_attention_scope"].enabled
    assert "sage_attn" not in panel.controls["nr_attention_backend"].options
    panel.controls["nr_fp8_base"].set_toggle_value(True)
    panel.controls["nr_fp8_scaled"].set_toggle_value(True)
    assert train_step._get_config()["nr_fp8_scaled"] is True
    train_step.train_mode.set_value("finetune")
    state = train_step._get_config()
    assert state["nr_fp8_base"] is False
    assert state["nr_fp8_scaled"] is False
    assert state["nr_mixed_precision"] == "bf16"
    assert not panel.controls["nr_fp8_base"].enabled


def test_switching_back_to_baseline_clears_only_experimental_options(train_step):
    panel = train_step._dlssnr_panel
    panel.controls["nr_numerics_profile"].set_value("train_experimental")
    panel.controls["nr_mixed_precision"].set_value("fp16")
    panel.controls["nr_attention_backend"].set_value("sdpa")
    panel.controls["nr_fp8_base"].set_toggle_value(True)
    panel.controls["nr_gradient_checkpointing"].set_toggle_value(True)
    assert panel._overflow_section.visible
    panel.controls["nr_numerics_profile"].set_value("train_surrogate")
    state = train_step._get_config()
    assert (state["nr_mixed_precision"], state["nr_attention_backend"]) == ("no", "native")
    assert state["nr_fp8_base"] is False
    assert state["nr_gradient_checkpointing"] is True
    assert not panel._overflow_section.visible


def test_runtime_preset_roundtrip_and_architecture_switch_preserve_explicit_policy(train_step):
    policy = {
        "nr_numerics_profile": "train_experimental",
        "nr_mixed_precision": "bf16",
        "nr_gradient_checkpointing": True,
        "nr_fp8_base": True,
        "nr_fp8_scaled": True,
        "nr_attention_backend": "sdpa",
        "nr_attention_scope": "global",
        "nr_num_processes": 2,
        "nr_max_overflow_retries": 7,
    }
    train_step._apply_config({"arch": "DLSS-NR", **policy})
    train_step.model_selector.set_arch("FLUX.2")
    train_step.model_selector.set_arch("DLSS-NR")
    state = train_step._get_config()
    assert {key: state[key] for key in policy} == policy
    train_step._apply_config({"arch": "DLSS-NR"})
    assert train_step._get_config()["nr_num_processes"] == 1
    assert train_step._get_config()["nr_mixed_precision"] == "no"


@pytest.mark.parametrize(
    "override",
    [
        {"nr_numerics_profile": "train_surrogate", "nr_mixed_precision": "bf16"},
        {"nr_numerics_profile": "train_experimental", "nr_fp8_scaled": True},
        {
            "nr_numerics_profile": "train_experimental",
            "nr_attention_backend": "sage_attn",
            "nr_mixed_precision": "bf16",
            "nr_attention_scope": "global",
        },
        {"nr_num_processes": 1.5},
        {"nr_max_overflow_retries": -1},
    ],
)
def test_invalid_runtime_presets_are_rejected_atomically(train_step, override):
    before = train_step._get_config()
    with pytest.raises(CommandBuildError):
        train_step._apply_config({"arch": "DLSS-NR", **override})
    assert train_step._get_config() == before


def test_cpu_selection_disables_amp_without_turning_off_checkpoint(train_step):
    panel = train_step._dlssnr_panel
    panel.controls["nr_numerics_profile"].set_value("train_experimental")
    panel.controls["nr_mixed_precision"].set_value("bf16")
    panel.controls["nr_attention_backend"].set_value("flash_attn")
    panel.controls["nr_gradient_checkpointing"].set_toggle_value(True)
    panel.controls["nr_device"].set_value("cpu")
    state = train_step._get_config()
    assert state["nr_mixed_precision"] == "no"
    assert state["nr_attention_backend"] == "native"
    assert state["nr_gradient_checkpointing"] is True
    assert not panel.controls["nr_mixed_precision"].enabled


def test_inference_runtime_defaults_to_inherit_and_supports_explicit_sage_policy(generate_step):
    panel = generate_step._dlssnr_panel
    assert generate_step._get_config()["nr_runtime_mode"] == "inherit"
    assert not panel._runtime_options.visible
    assert generate_step._tab_inference.visible
    panel.controls["nr_runtime_mode"].set_value("override")
    assert panel._runtime_options.visible
    panel.controls["nr_numerics_profile"].set_value("train_experimental")
    panel.controls["nr_mixed_precision"].set_value("bf16")
    assert "sage_attn" in panel.controls["nr_attention_backend"].options
    panel.controls["nr_attention_backend"].set_value("sage_attn")
    assert generate_step._get_config()["nr_attention_scope"] == "global"
    config = generate_step._get_config()
    generate_step._apply_config({"arch": "DLSS-NR"})
    assert generate_step._get_config()["nr_runtime_mode"] == "inherit"
    generate_step._apply_config(config)
    assert generate_step._get_config()["nr_attention_backend"] == "sage_attn"
