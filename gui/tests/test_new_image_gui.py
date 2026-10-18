"""Exercise real form state, including switching from older architectures."""

import sys
from pathlib import Path

import pytest
from nicegui import ui

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))

from test_new_image_commands import ARCHES, PROJECT, parse_job  # noqa: E402
from utils.command_builder import build_cache_jobs, build_generate_job, build_train_job  # noqa: E402
from wizard.step2_cache import CacheStep  # noqa: E402
from wizard.step3_train import TrainStep  # noqa: E402
from wizard.step4_generate import GenerateStep  # noqa: E402


@pytest.mark.parametrize("arch,slug", ARCHES)
@pytest.mark.parametrize("step_type", [CacheStep, TrainStep, GenerateStep])
def test_real_forms_emit_accepted_native_commands(tmp_path, arch, slug, step_type):
    step = step_type()
    step.project_dir = tmp_path
    with ui.column() as container:
        step.render()
    try:
        step.model_selector.set_arch(arch)
        assert step.model_selector.arch == arch
        if step_type is GenerateStep:
            step.prompt.set_value("a greenhouse")
        state = step._get_config()
        assert state["text_encoder_path"].endswith(
            "qwen3vl_8b_bf16.safetensors" if slug == "qwen_image_21" else "qwen3.5_9b_bf16.safetensors"
        )
        if step_type is CacheStep:
            jobs = build_cache_jobs(state, tmp_path, PROJECT)
            parse_job(jobs[0], f"{slug}_cache_latents")
            parse_job(jobs[1], f"{slug}_cache_text_encoder_outputs")
        elif step_type is TrainStep:
            initial = parse_job(build_train_job(state, tmp_path, PROJECT), f"{slug}_train_network")
            assert initial.max_train_epochs is None
            step.train_mode.set_value("finetune")
            assert step._finetune_options_card.visible
            assert not step._tab_network.visible
            full = parse_job(build_train_job(step._get_config(), tmp_path, PROJECT), f"{slug}_train")
            assert full.optimizer_type == ("pytorch_optimizer.AdaFactor" if slug == "moss_image" else "optimi.AdamW")
            assert float(full.learning_rate) == (1e-5 if slug == "moss_image" else 5e-6)
            step.train_mode.set_value("lora")
            restored = parse_job(build_train_job(step._get_config(), tmp_path, PROJECT), f"{slug}_train_network")
            assert restored.optimizer_type == "optimi.AdamW" and float(restored.learning_rate) == 1e-4
        else:
            parse_job(build_generate_job(state, tmp_path), f"{slug}_generate_image")
    finally:
        container.delete()


def test_moss_full_preset_keeps_adafactor_in_selector(tmp_path):
    step = TrainStep()
    step.project_dir = tmp_path
    with ui.column() as container:
        step.render()
    try:
        step._apply_config({"arch": "MOSS-Image", "train_mode": "finetune", "optimizer_type": "Adafactor"})
        state = step._get_config()
        assert state["optimizer_type"] == "adafactor"
        parsed = parse_job(build_train_job(state, tmp_path, PROJECT), "moss_image_train")
        assert parsed.optimizer_type == "pytorch_optimizer.AdaFactor"
        assert parsed.optimizer_args == ["scale_parameter=False", "warmup_init=False", "relative_step=False", "cautious=True"]
    finally:
        container.delete()


def test_full_memory_mode_refreshes_only_untouched_optimizer_templates(tmp_path):
    step = TrainStep()
    step.project_dir = tmp_path
    with ui.column() as container:
        step.render()
    try:
        step.model_selector.set_arch("MOSS-Image")
        step.train_mode.set_value("finetune")
        assert "cautious=True" in step.optimizer_extra_args.value
        step.config["_bound_controls"]["blocks_to_swap"].set_bound_value(2)
        step.config["block_swap_optimizer_patch_params"] = True
        state = step._get_config()
        parsed = parse_job(build_train_job(state, tmp_path, PROJECT), "moss_image_train")
        assert parsed.optimizer_type == "Adafactor"
        assert "cautious=True" not in parsed.optimizer_args
        step.config["_bound_controls"]["blocks_to_swap"].set_bound_value(0)
        assert "cautious=True" in step.optimizer_extra_args.value
        step.optimizer_extra_args.set_value("scale_parameter=False\nrelative_step=False\nweight_decay=0.02")
        custom = step.optimizer_extra_args.value
        step.config["_bound_controls"]["blocks_to_swap"].set_bound_value(2)
        assert step.optimizer_extra_args.value == custom
    finally:
        container.delete()


@pytest.mark.parametrize(
    "arch,version,filename",
    [
        ("Qwen Image 2.1", "turbo", "qwen_image_2.1_turbo_bf16.safetensors"),
        ("MOSS-Image", "sft", "moss_image_sft_bf16.safetensors"),
    ],
)
def test_variant_change_keeps_custom_training_values_and_shared_paths(tmp_path, arch, version, filename):
    step = TrainStep()
    step.project_dir = tmp_path
    with ui.column() as container:
        step.render()
    try:
        step.model_selector.set_arch(arch)
        step.learning_rate.set_value("0.00003")
        step.max_train_steps.set_value("1234")
        step.text_encoder_path.value = "custom text.safetensors"
        step.model_selector.set_version(version)
        state = step._get_config()
        assert state["learning_rate"] == "0.00003" and str(state["max_train_steps"]) == "1234"
        assert state["text_encoder_path"] == "custom text.safetensors"
        assert state["dit_path"].endswith(filename)
    finally:
        container.delete()


def test_qwen_turbo_switch_updates_only_unchanged_inference_defaults(tmp_path):
    step = GenerateStep()
    step.project_dir = tmp_path
    with ui.column() as container:
        step.render()
    try:
        step.model_selector.set_arch("Qwen Image 2.1")
        step.video_size.set_value("768 1024")
        step.control_image_path.set_value("custom reference.png")
        step.model_selector.set_version("turbo")
        state = step._get_config()
        assert state["infer_steps"] == 8
        assert state["video_size"] == "768 1024"
        assert state["control_image_path"] == "custom reference.png"
    finally:
        container.delete()


@pytest.mark.parametrize("step_type", [CacheStep, TrainStep, GenerateStep])
def test_moss_number_control_edits_emit_integer_token_limit(tmp_path, step_type):
    step = step_type()
    step.project_dir = tmp_path
    with ui.column() as container:
        step.render()
    try:
        step.model_selector.set_arch("MOSS-Image")
        step.moss_max_token_length.set_value(2048.0)
        state = step._get_config()
        if step_type is CacheStep:
            job = build_cache_jobs(state, tmp_path, PROJECT)[1]
            module = "moss_image_cache_text_encoder_outputs"
        elif step_type is TrainStep:
            job = build_train_job(state, tmp_path, PROJECT)
            module = "moss_image_train_network"
        else:
            state["prompt"] = "test"
            job = build_generate_job(state, tmp_path)
            module = "moss_image_generate_image"
        assert parse_job(job, module).moss_max_token_length == 2048
    finally:
        container.delete()


def test_moss_form_hides_fp8_and_preserves_custom_preset_values(tmp_path):
    step = TrainStep()
    step.project_dir = tmp_path
    with ui.column() as container:
        step.render()
    try:
        step.config["fp8_base"] = True
        step.model_selector.set_arch("MOSS-Image")
        assert step.model_selector.arch == "MOSS-Image"
        assert not any(control.visible for control in step._finetune_disabled_fp8_controls)
        assert step._get_config()["fp8_base"] is False
        preset = {
            "arch": "MOSS-Image",
            "train_mode": "lora",
            "dit_path": "custom.safetensors",
            "caption_dropout_rate": 0.2,
            "moss_max_token_length": 2048,
            "sample_uncond_skip_layers": "",
            "learning_rate": "0.00003",
            "enable_sample": False,
        }
        step._apply_config(preset)
        state = step._get_config()
        for key in ("dit_path", "caption_dropout_rate", "moss_max_token_length", "sample_uncond_skip_layers"):
            assert state[key] == preset[key]
        step.model_selector.set_arch("Qwen Image 2.1")
        assert all(control.visible for control in step._finetune_disabled_fp8_controls)
        assert step._get_config()["vae_path"].endswith("qwen_image_2.1_vae_bf16.safetensors")
    finally:
        container.delete()
