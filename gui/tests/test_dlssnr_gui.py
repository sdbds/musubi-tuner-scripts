import sys
from pathlib import Path

import pytest
from nicegui import ui

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))

from components.model_selector import create_model_selector  # noqa: E402
from utils.config_manager import ConfigManager  # noqa: E402
from utils.i18n import TRANSLATIONS  # noqa: E402
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
    assert train_step._dlssnr_tab.visible
    assert all(not tab.visible for tab in train_step._diffusion_tabs)
    assert train_step.train_mode.options.keys() == {"lora", "finetune"}
    state = train_step._get_config()
    assert state["nr_development_smoke"] is False
    assert state["nr_learning_rate"] == 1e-4
    assert "mixed_precision" not in state
    assert "gradient_checkpointing" not in state
    train_step.model_selector.set_arch("FLUX.2")
    assert not train_step._dlssnr_tab.visible
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
    assert panel.get_state()["nr_development_smoke"] is False


def test_multiscale_controls_replace_scalar_rank_inputs(train_step):
    panel = train_step._dlssnr_panel
    panel.controls["nr_lora_profile"].set_value("multiscale")
    assert panel._multiscale_section.visible
    assert not panel._vit_section.visible


def test_train_preset_roundtrip_preserves_explicit_values_and_resets_omitted_experiment_flag(train_step):
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
        "nr_development_smoke": True,
    }
    train_step._apply_config(custom)
    state = train_step._get_config()
    for key, value in custom.items():
        assert state[key] == value
    train_step._apply_config({"arch": "DLSS-NR", "train_mode": "lora"})
    assert train_step._get_config()["nr_development_smoke"] is False
    assert train_step._get_config()["nr_learning_rate"] == 1e-4


def test_generate_page_hides_diffusion_inputs_and_keeps_image_sequence_manifests(generate_step):
    panel = generate_step._dlssnr_panel
    assert generate_step._dlssnr_tab.visible
    assert all(not tab.visible for tab in generate_step._diffusion_tabs)
    assert not generate_step._page_description.visible
    panel.controls["nr_sample_manifest"].value = "stills.jsonl"
    panel.controls["nr_sequence_manifest"].value = "clip.jsonl"
    panel.controls["nr_inference_mode"].set_value("sequence")
    assert panel.controls["nr_sequence_manifest"].container.visible
    assert not panel.controls["nr_sample_manifest"].container.visible
    assert generate_step._get_config()["nr_sequence_manifest"] == "clip.jsonl"
    generate_step.model_selector.set_arch("Mage-Flow")
    assert not generate_step._dlssnr_tab.visible
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
        "nr_development_smoke",
        "nr_forward_validation_report",
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


def test_builtin_presets_keep_experimental_training_opt_in():
    manager = ConfigManager()
    for name in ("dlssnr_lora", "dlssnr_full", "dlssnr_lora_temporal", "dlssnr_full_temporal"):
        preset = manager.load_config("train", name)
        assert preset is not None, name
        assert preset["arch"] == "DLSS-NR"
        assert preset["nr_development_smoke"] is False
    for name, mode in (("dlssnr_image", "image"), ("dlssnr_sequence", "sequence")):
        preset = manager.load_config("generate", name)
        assert preset is not None, name
        assert preset["nr_inference_mode"] == mode
