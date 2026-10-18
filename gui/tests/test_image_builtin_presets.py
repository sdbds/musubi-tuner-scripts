import importlib
import sys
from pathlib import Path

import pytest

GUI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_ROOT))

from utils.command_builder import build_cache_jobs, build_generate_job, build_train_job  # noqa: E402
from utils.config_manager import ConfigManager  # noqa: E402
from utils.script_coverage_manifest import NATIVE_GUI  # noqa: E402
from utils.script_preset_catalog import get_builtin_preset  # noqa: E402

PROJECT = {"dataset": {"general": {}, "datasets": [{"image_directory": "images", "cache_directory": "cache"}]}}
PROFILES = [
    ("qwen_image_21", "Qwen Image 2.1", "standard", "qwen_image_2.1_bf16.safetensors", "qwen_image_21", "13"),
    ("qwen_image_21_turbo", "Qwen Image 2.1", "turbo", "qwen_image_2.1_turbo_bf16.safetensors", "qwen_image_21", "13"),
    ("moss_image", "MOSS-Image", "base", "moss_image_base_bf16.safetensors", "moss_image", "14"),
    ("moss_image_sft", "MOSS-Image", "sft", "moss_image_sft_bf16.safetensors", "moss_image", "14"),
]


def parse_native_job(job, module):
    parser = importlib.import_module(f"musubi_tuner.{module}").setup_parser()
    parsed, unknown = parser.parse_known_args(job.args)
    assert unknown == [], f"Preset emitted unsupported native CLI arguments: {unknown}"
    return parsed


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(user_dir=str(tmp_path / "user_presets"))


@pytest.mark.parametrize("slug,arch,version,dit,module,index", PROFILES)
def test_generation_presets_are_discoverable_and_build_variant_specific_commands(
    manager, tmp_path, slug, arch, version, dit, module, index
):
    entry = next((entry for entry in manager.list_config_entries("generate") if entry["name"] == slug), None)
    assert entry is not None
    assert entry["source"] == "builtin"
    preset = manager.load_config("generate", slug)
    assert (preset["arch"], preset["version"]) == (arch, version)
    job = build_generate_job(preset, tmp_path)
    parsed = parse_native_job(job, f"{module}_generate_image")
    scalars = dict(arg[2:].split("=", 1) for arg in job.args if arg.startswith("--") and "=" in arg)
    assert job.script_key == f"musubi_tuner.{module}_generate_image"
    assert f"--dit=./ckpts/diffusion_models/{dit}" in job.args
    assert parsed.dit == f"./ckpts/diffusion_models/{dit}"
    if arch == "Qwen Image 2.1":
        assert f"--infer_steps={8 if version == 'turbo' else 40}" in job.args
        assert float(scalars["guidance_scale"]) == 1.0
        assert not any(arg.startswith("--flow_shift") for arg in job.args)
    else:
        assert "--infer_steps=50" in job.args
        assert float(scalars["guidance_scale"]) == 3.0
        assert float(scalars["flow_shift"]) == 5.0
        assert "--uncond_skip_layers=22" in job.args
        assert not any(arg.startswith("--fp8") or arg.startswith("--control_image") for arg in job.args)


@pytest.mark.parametrize("slug,arch,version,dit,module,index", PROFILES)
@pytest.mark.parametrize("mode", ["lora", "finetune"])
def test_training_presets_select_native_entry_point_and_safe_defaults(
    manager, tmp_path, slug, arch, version, dit, module, index, mode
):
    name = slug if mode == "lora" else f"{slug}_finetune"
    preset = manager.load_config("train", name)
    assert preset is not None
    assert (preset["arch"], preset["version"], preset["train_mode"]) == (arch, version, mode)
    assert preset["mixed_precision"] == "bf16"
    assert preset["attn_mode"] == "sdpa"
    assert preset["enable_sample"] is False
    assert preset["save_state_on_train_end"] is True
    assert preset["fp8_base"] is False
    assert preset["fp8_scaled"] is False
    assert preset["max_train_steps"] == (1000 if arch == "MOSS-Image" else 2000)
    assert preset["max_train_epochs"] == 0
    assert preset["optimizer_type"] == ("adafactor" if arch == "MOSS-Image" and mode == "finetune" else "AdamW")
    assert preset["save_every_n_steps"] == (100 if arch == "MOSS-Image" else 200)
    expected_lr = "1e-4" if mode == "lora" else "1e-5" if arch == "MOSS-Image" else "5e-6"
    assert float(preset["learning_rate"]) == float(expected_lr)
    job = build_train_job(preset, tmp_path, PROJECT)
    parsed = parse_native_job(job, f"{module}_{'train_network' if mode == 'lora' else 'train'}")
    assert parsed.learning_rate == float(expected_lr)
    assert parsed.max_train_steps == (1000 if arch == "MOSS-Image" else 2000)
    assert not parsed.max_train_epochs
    assert parsed.optimizer_type == (
        "pytorch_optimizer.AdaFactor" if arch == "MOSS-Image" and mode == "finetune" else "optimi.AdamW"
    )
    assert parsed.sample_prompts is None
    assert job.script_key.endswith(str(Path("musubi_tuner") / f"{module}_{'train_network' if mode == 'lora' else 'train'}.py"))
    assert f"--dit=./ckpts/diffusion_models/{dit}" in job.args
    assert "--save_state_on_train_end" in job.args
    assert not any(arg.startswith(("--sample_prompts", "--sample_every", "--sample_at_first")) for arg in job.args)
    if mode == "finetune":
        assert preset["mem_eff_save"] is True
        assert "--mem_eff_save" in job.args
        assert not any(arg.startswith("--network_") for arg in job.args)
    else:
        assert preset["network_dim"] == (32 if arch == "MOSS-Image" else 16)
        assert preset["network_alpha"] == preset["network_dim"]


@pytest.mark.parametrize(
    "slug,arch,vae,encoder,vae_dtype",
    [
        ("qwen_image_21", "Qwen Image 2.1", "qwen_image_2.1_vae_bf16.safetensors", "qwen3vl_8b_bf16.safetensors", "bfloat16"),
        ("moss_image", "MOSS-Image", "flux2-vae.safetensors", "qwen3.5_9b_bf16.safetensors", "float32"),
    ],
)
def test_cache_presets_use_architecture_specific_components(manager, tmp_path, slug, arch, vae, encoder, vae_dtype):
    preset = manager.load_config("cache", slug)
    assert preset is not None
    assert preset["arch"] == arch
    assert preset["vae_dtype"] == vae_dtype
    jobs = build_cache_jobs(preset, tmp_path, PROJECT)
    parse_native_job(jobs[0], f"{slug}_cache_latents")
    parse_native_job(jobs[1], f"{slug}_cache_text_encoder_outputs")
    assert jobs[0].script_key == f"musubi_tuner.{slug}_cache_latents"
    assert f"--vae=./ckpts/vae/{vae}" in jobs[0].args
    assert jobs[1].script_key == f"musubi_tuner.{slug}_cache_text_encoder_outputs"
    assert f"--text_encoder=./ckpts/text_encoder/{encoder}" in jobs[1].args


@pytest.mark.parametrize("slug,arch,version,dit,module,index", PROFILES)
def test_script_source_presets_preserve_typed_defaults_and_match_builtin_consumers(
    manager, slug, arch, version, dit, module, index
):
    for scope, names in (("train", [slug, f"{slug}_finetune"]), ("generate", [slug])):
        for name in names:
            source = get_builtin_preset(scope, name)
            assert source is not None
            preset = manager.load_config(scope, name)
            for key in ("arch", "version", "dit_path", "vae_path", "text_encoder_path"):
                assert source[key] == preset[key]
            for key in (
                ("max_train_steps", "save_every_n_steps", "gradient_checkpointing", "save_state_on_train_end")
                if scope == "train"
                else ("infer_steps", "guidance_scale", "flow_shift")
            ):
                assert source[key] == preset[key]
            if scope == "train":
                assert source["learning_rate"] == float(preset["learning_rate"])
                assert source["optimizer_type"] == preset["optimizer_type"]
            assert source["_source_script"] in NATIVE_GUI
            assert source["_source_script"] == preset["_source_script"]
    cache = get_builtin_preset("cache", module)
    assert cache is not None
    assert cache["_source_script"] == f"2.{index}{module}_cache_latent_and_text_encoder.ps1"
    assert cache["_source_script"] in NATIVE_GUI
