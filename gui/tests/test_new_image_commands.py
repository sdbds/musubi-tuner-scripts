"""Contracts against the actual Qwen 2.1 and MOSS command-line parsers."""

import copy
import importlib
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))

from utils.command_builder import CommandBuildError, build_cache_jobs, build_generate_job, build_train_job  # noqa: E402

PROJECT = {
    "dataset": {
        "general": {"resolution": [1024, 1024]},
        "datasets": [{"image_directory": "images", "cache_directory": "cache", "batch_size": 1}],
    },
    "interop": {"dataset_extra": {"root": {}, "general": {}, "datasets": [{}]}},
}
ARCHES = [("Qwen Image 2.1", "qwen_image_21"), ("MOSS-Image", "moss_image")]


@lru_cache
def parser_for(module):
    return importlib.import_module(f"musubi_tuner.{module}").setup_parser()


def parse_job(job, module):
    parsed, unknown = parser_for(module).parse_known_args(job.args)
    assert unknown == [], f"Unsupported CLI arguments: {unknown}"
    return parsed


@pytest.mark.parametrize("arch,slug", ARCHES)
def test_cache_routes_to_dedicated_modules_with_installed_components(tmp_path, arch, slug):
    jobs = build_cache_jobs({"arch": arch}, tmp_path, PROJECT)
    assert [job.script_key for job in jobs] == [
        f"musubi_tuner.{slug}_cache_latents",
        f"musubi_tuner.{slug}_cache_text_encoder_outputs",
    ]
    latent = parse_job(jobs[0], f"{slug}_cache_latents")
    text = parse_job(jobs[1], f"{slug}_cache_text_encoder_outputs")
    if slug == "qwen_image_21":
        assert latent.vae == "./ckpts/vae/qwen_image_2.1_vae_bf16.safetensors"
        assert text.text_encoder == "./ckpts/text_encoder/qwen3vl_8b_bf16.safetensors"
    else:
        assert latent.vae == "./ckpts/vae/flux2-vae.safetensors"
        assert latent.vae_dtype == "float32"
        assert text.text_encoder == "./ckpts/text_encoder/qwen3.5_9b_bf16.safetensors"
    assert latent.dataset_config == text.dataset_config
    assert Path(latent.dataset_config).is_file()


@pytest.mark.parametrize("arch,slug", ARCHES)
@pytest.mark.parametrize("mode,entry", [("lora", "train_network"), ("finetune", "train")])
def test_train_modes_use_native_parsers_and_keep_lora_args_out_of_full(tmp_path, arch, slug, mode, entry):
    job = build_train_job({"arch": arch, "train_mode": mode}, tmp_path, PROJECT)
    assert job.script_key.endswith(str(Path("musubi_tuner") / f"{slug}_{entry}.py"))
    parsed = parse_job(job, f"{slug}_{entry}")
    assert parsed.sdpa
    assert parsed.mixed_precision == "bf16"
    assert parsed.optimizer_type == (
        "pytorch_optimizer.AdaFactor" if slug == "moss_image" and mode == "finetune" else "optimi.AdamW"
    )
    assert parsed.dit.endswith("qwen_image_2.1_bf16.safetensors" if slug == "qwen_image_21" else "moss_image_base_bf16.safetensors")
    assert job.runner_kwargs["use_accelerate"]
    if mode == "lora":
        assert parsed.network_module.endswith(f"lora_{slug}")
    else:
        assert not any(arg.startswith(("--network_", "--base_weights")) for arg in job.args)


@pytest.mark.parametrize("arch,slug", ARCHES)
def test_training_samples_require_prompt_and_components_but_cache_only_training_does_not(tmp_path, arch, slug):
    state = {"arch": arch, "vae_path": "", "text_encoder_path": ""}
    job = build_train_job(state, tmp_path, PROJECT)
    assert not any(arg.startswith(("--vae=", "--text_encoder=")) for arg in job.args)
    with pytest.raises(CommandBuildError, match="sampling|Sampling"):
        build_train_job({**state, "enable_sample": True, "sample_prompts": "prompts.txt"}, tmp_path, PROJECT)
    job = build_train_job({"arch": arch, "enable_sample": True, "sample_prompts": "prompts.txt"}, tmp_path, PROJECT)
    parsed = parse_job(job, f"{slug}_train_network")
    assert parsed.vae and parsed.text_encoder and parsed.sample_prompts == "prompts.txt"


@pytest.mark.parametrize("arch,slug", ARCHES)
def test_full_training_emits_memory_options_and_rejects_frozen_weight_options(tmp_path, arch, slug):
    job = build_train_job(
        {
            "arch": arch,
            "train_mode": "finetune",
            "full_bf16": True,
            "mem_eff_save": True,
            "blocks_to_swap": 2,
            "block_swap_optimizer_patch_params": True,
            "optimizer_type": "AdamW",
        },
        tmp_path,
        PROJECT,
    )
    parsed = parse_job(job, f"{slug}_train")
    assert parsed.full_bf16 and parsed.mem_eff_save and parsed.block_swap_optimizer_patch_params
    for bad in ({"fp8_base": True}, {"fp8_scaled": True}, {"block_swap_h2d_only": True}):
        with pytest.raises(CommandBuildError):
            build_train_job({"arch": arch, "train_mode": "finetune", **bad}, tmp_path, PROJECT)


@pytest.mark.parametrize(
    "extra",
    [
        {"fp8_base": True},
        {"fp8_scaled": True},
        {"fp8_vl": True},
        {"mixed_precision": "fp16"},
        {"vae_dtype": "bfloat16"},
        {"blocks_to_swap": 41},
        {"moss_max_token_length": 4097},
        {"caption_dropout_rate": 1.1},
        {"log_grad_metrics": True},
        {"compile": True, "compile_fullgraph": True, "split_attn": True},
        {"gradient_checkpointing_cpu_offload": True, "blocks_to_swap": 1},
        {"attn_mode": "xformers", "split_attn": False},
    ],
)
def test_moss_training_rejects_unsupported_settings_before_launch(tmp_path, extra):
    with pytest.raises(CommandBuildError):
        build_train_job({"arch": "MOSS-Image", **extra}, tmp_path, PROJECT)


def test_moss_rejects_reference_dataset_and_qwen21_accepts_it(tmp_path):
    project = copy.deepcopy(PROJECT)
    project["dataset"]["datasets"][0]["control_directory"] = "references"
    for builder in (build_cache_jobs, build_train_job):
        with pytest.raises(CommandBuildError, match="control|reference"):
            builder({"arch": "MOSS-Image"}, tmp_path, project)
        builder({"arch": "Qwen Image 2.1"}, tmp_path, project)


def test_moss_text_cache_options_and_disabled_stage_paths(tmp_path):
    (job,) = build_cache_jobs(
        {
            "arch": "MOSS-Image",
            "cache_latents_enabled": False,
            "vae_path": "",
            "tokenizer_path": "local tokenizer",
            "moss_max_token_length": 1024,
            "disable_numpy_memmap": True,
        },
        tmp_path,
        PROJECT,
    )
    parsed = parse_job(job, "moss_image_cache_text_encoder_outputs")
    assert parsed.tokenizer_path == "local tokenizer"
    assert parsed.moss_max_token_length == 1024 and parsed.disable_numpy_memmap


def test_qwen21_generate_supports_ordered_controls_multiple_loras_and_turbo_defaults(tmp_path):
    job = build_generate_job(
        {
            "arch": "Qwen Image 2.1",
            "version": "turbo",
            "prompt": "change the sky",
            "control_image_path": "first image.png\nsecond image.png",
            "lora_weight": "one.safetensors\ntwo.safetensors",
            "lora_multiplier": "0.7 1.2",
            "vae_tiling": True,
            "fp8_vl": True,
        },
        tmp_path,
    )
    parsed = parse_job(job, "qwen_image_21_generate_image")
    assert parsed.dit.endswith("qwen_image_2.1_turbo_bf16.safetensors")
    assert parsed.infer_steps == 8 and parsed.guidance_scale == 1.0
    assert parsed.control_image_path == ["first image.png", "second image.png"]
    assert parsed.lora_weight == ["one.safetensors", "two.safetensors"]
    assert parsed.lora_multiplier == [0.7, 1.2]
    assert parsed.vae_tiling and parsed.fp8_vl


def test_qwen21_latent_decode_needs_only_vae(tmp_path):
    job = build_generate_job(
        {
            "arch": "Qwen Image 2.1",
            "latent_path": "a.safetensors\nb.safetensors",
            "dit_path": "",
            "text_encoder_path": "",
            "prompt": "",
        },
        tmp_path,
    )
    parsed = parse_job(job, "qwen_image_21_generate_image")
    assert parsed.latent_path == ["a.safetensors", "b.safetensors"]
    assert parsed.vae and not parsed.dit and not parsed.text_encoder


def test_moss_generate_adapts_dimensions_output_and_empty_skip_layers(tmp_path):
    job = build_generate_job(
        {
            "arch": "MOSS-Image",
            "version": "sft",
            "prompt": "a greenhouse",
            "video_size": "768 1024",
            "save_path": "images output",
            "uncond_skip_layers": "",
            "lora_weight": ["one.safetensors", "two.safetensors"],
            "lora_multiplier": [0.5, 1.2],
        },
        tmp_path,
    )
    parsed = parse_job(job, "moss_image_generate_image")
    assert parsed.dit.endswith("moss_image_sft_bf16.safetensors")
    assert (parsed.height, parsed.width) == (768, 1024)
    assert parsed.output_dir == "images output"
    assert parsed.infer_steps == 50 and parsed.guidance_scale == 3.0 and parsed.flow_shift == 5.0
    assert parsed.uncond_skip_layers == ""
    assert parsed.lora_weight == ["one.safetensors", "two.safetensors"] and parsed.lora_multiplier == [0.5, 1.2]


@pytest.mark.parametrize(
    "extra",
    [
        {"control_image_path": "ref.png"},
        {"from_file": "prompts.txt"},
        {"latent_path": "latent.safetensors"},
        {"fp8_scaled": True},
        {"video_size": "1023 1024"},
        {"dtype": "float16"},
        {"lora_weight": "one.safetensors", "lora_multiplier": "1 2"},
    ],
)
def test_moss_generate_rejects_unsupported_options(tmp_path, extra):
    with pytest.raises(CommandBuildError):
        build_generate_job({"arch": "MOSS-Image", "prompt": "a greenhouse", **extra}, tmp_path)


def test_moss_generation_from_native_cache_omits_text_encoder_and_prompts(tmp_path):
    job = build_generate_job(
        {
            "arch": "MOSS-Image",
            "text_cache": "caption_moss_te.safetensors",
            "text_encoder_path": "",
            "prompt": "",
            "negative_prompt": "",
        },
        tmp_path,
    )
    parsed = parse_job(job, "moss_image_generate_image")
    assert parsed.text_cache == "caption_moss_te.safetensors"
    assert parsed.text_encoder is None and parsed.prompt is None


@pytest.mark.parametrize("arch,slug", ARCHES)
def test_explicit_custom_paths_survive_profile_selection(tmp_path, arch, slug):
    state = {
        "arch": arch,
        "dit_path": "my custom.safetensors",
        "vae_path": "my vae.safetensors",
        "text_encoder_path": "my text.safetensors",
        "prompt": "test",
    }
    parsed = parse_job(build_generate_job(state, tmp_path), f"{slug}_generate_image")
    assert (parsed.dit, parsed.vae, parsed.text_encoder) == (state["dit_path"], state["vae_path"], state["text_encoder_path"])


def test_moss_sigmoid_training_keeps_custom_training_shift(tmp_path):
    parsed = parse_job(
        build_train_job({"arch": "MOSS-Image", "discrete_flow_shift": 2.0}, tmp_path, PROJECT), "moss_image_train_network"
    )
    assert parsed.timestep_sampling == "sigmoid" and parsed.discrete_flow_shift == 2.0


def test_moss_skip_indices_use_42_layers_not_swap_limit(tmp_path):
    parsed = parse_job(
        build_generate_job({"arch": "MOSS-Image", "prompt": "test", "uncond_skip_layers": "40,41"}, tmp_path),
        "moss_image_generate_image",
    )
    assert parsed.uncond_skip_layers == "40,41"
    with pytest.raises(CommandBuildError):
        build_generate_job({"arch": "MOSS-Image", "prompt": "test", "uncond_skip_layers": "42"}, tmp_path)


def test_qwen21_broadcasts_one_lora_multiplier_like_native_cli(tmp_path):
    parsed = parse_job(
        build_generate_job(
            {"arch": "Qwen Image 2.1", "prompt": "test", "lora_weight": ["a", "b"], "lora_multiplier": 0.8}, tmp_path
        ),
        "qwen_image_21_generate_image",
    )
    module = importlib.import_module("musubi_tuner.qwen_image_21_generate_image")
    assert module.get_lora_multipliers(parsed) == [0.8, 0.8]


@pytest.mark.parametrize(
    "state",
    [
        {"mask_path": "mask.png"},
        {"mask_path": "mask.png", "control_image_path": "ref.png"},
    ],
)
def test_qwen21_inpainting_requires_control_and_matching_resize(tmp_path, state):
    with pytest.raises(CommandBuildError, match="mask|resize"):
        build_generate_job({"arch": "Qwen Image 2.1", "prompt": "test", **state}, tmp_path)


def test_moss_generation_can_compile_with_supported_options(tmp_path):
    parsed = parse_job(
        build_generate_job(
            {
                "arch": "MOSS-Image",
                "prompt": "test",
                "compile": True,
                "compile_backend": "inductor",
                "compile_mode": "reduce-overhead",
            },
            tmp_path,
        ),
        "moss_image_generate_image",
    )
    assert parsed.compile and parsed.compile_mode == "reduce-overhead"


def test_qwen21_cache_rejects_fp16_text_encoder_before_launch(tmp_path):
    with pytest.raises(CommandBuildError, match="dtype|precision"):
        build_cache_jobs({"arch": "Qwen Image 2.1", "text_encoder_dtype": "float16"}, tmp_path, PROJECT)


@pytest.mark.parametrize("arch,slug", ARCHES)
@pytest.mark.parametrize("mode,entry", [("lora", "train_network"), ("finetune", "train")])
@pytest.mark.parametrize(
    "optimizer,expected",
    [
        ("AdamW", "optimi.AdamW"),
        ("adafactor", "pytorch_optimizer.AdaFactor"),
        ("Prodigy", "pytorch_optimizer.Prodigy"),
        ("custom.optimizer.Class", "custom.optimizer.Class"),
    ],
)
def test_ordinary_training_keeps_existing_optimizer_resolution(tmp_path, arch, slug, mode, entry, optimizer, expected):
    job = build_train_job({"arch": arch, "train_mode": mode, "optimizer_type": optimizer}, tmp_path, PROJECT)
    assert parse_job(job, f"{slug}_{entry}").optimizer_type == expected


@pytest.mark.parametrize("arch,slug", ARCHES)
def test_generation_integral_numbers_emit_integer_cli_tokens(tmp_path, arch, slug):
    state = {"arch": arch, "prompt": "test", "infer_steps": 25.0, "blocks_to_swap": 2.0, "seed": 42.0}
    parsed = parse_job(build_generate_job(state, tmp_path), f"{slug}_generate_image")
    assert (parsed.infer_steps, parsed.blocks_to_swap, parsed.seed) == (25, 2, 42)


@pytest.mark.parametrize(
    "memory_mode",
    [
        {"blocks_to_swap": 2, "block_swap_optimizer_patch_params": True},
        {"fused_backward_pass": True, "max_grad_norm": 0},
    ],
)
def test_moss_full_memory_modes_reject_gui_multi_gpu(tmp_path, memory_mode):
    with pytest.raises(CommandBuildError, match="one process"):
        build_train_job(
            {"arch": "MOSS-Image", "train_mode": "finetune", "multi_gpu": True, **memory_mode},
            tmp_path,
            PROJECT,
        )


@pytest.mark.parametrize(
    "memory_mode",
    [
        {"blocks_to_swap": 2, "block_swap_optimizer_patch_params": True},
        {"fused_backward_pass": True, "max_grad_norm": 0},
    ],
)
def test_moss_full_memory_modes_pin_single_process_launch(tmp_path, memory_mode):
    job = build_train_job({"arch": "MOSS-Image", "train_mode": "finetune", **memory_mode}, tmp_path, PROJECT)
    assert {"--num_processes=1", "--num_machines=1"}.issubset(job.runner_kwargs["accelerate_args"])


@pytest.mark.parametrize("mode", ["lora", "finetune"])
def test_moss_ordinary_training_keeps_multi_gpu_available(tmp_path, mode):
    job = build_train_job({"arch": "MOSS-Image", "train_mode": mode, "multi_gpu": True}, tmp_path, PROJECT)
    assert "--multi_gpu" in job.runner_kwargs["accelerate_args"]
