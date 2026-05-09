"""Build executable musubi-tuner jobs from GUI form state."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils import model_catalog
from utils.dataset_config import export_dataset_config, get_default_dataset_config_path

SCRIPT_DEFAULT_OUTPUT_DIR = "./output_dir"
DOPSD_SOURCE_ROOT = "musubi-tuner-dopsd-zimage"


class CommandBuildError(ValueError):
    """Raised when the current GUI state cannot be converted to a runnable job."""


@dataclass(frozen=True)
class CommandJob:
    """A job ready for ExecutionPanel.run_job."""

    name: str
    script_key: str
    args: list[str]
    runner_kwargs: dict[str, Any] = field(default_factory=dict)


def get_train_optimizer_template_args(optimizer_type: Any, state: Mapping[str, Any] | None = None) -> list[str]:
    """Return default optimizer_args tokens for the selected optimizer."""
    optimizer_state = dict(state or {})
    optimizer_state["optimizer_type"] = optimizer_type
    _, optimizer_args = _resolve_train_optimizer(optimizer_state)
    return optimizer_args


MODEL_PATH_FLAGS = {
    "dit": "--dit",
    "dit_high_noise": "--dit_high_noise",
    "vae": "--vae",
    "text_encoder": "--text_encoder",
    "te1": "--text_encoder1",
    "te2": "--text_encoder2",
    "t5": "--t5",
    "clip": "--clip",
    "image_encoder": "--image_encoder",
    "byt5": "--byt5",
}

MODEL_PATH_STATE_KEYS = {
    "dit": ("dit_path",),
    "dit_high_noise": ("dit_high_noise",),
    "vae": ("vae_path",),
    "text_encoder": ("text_encoder_path", "text_encoder_vl_path"),
    "te1": ("te1_path",),
    "te2": ("te2_path",),
    "t5": ("t5_path",),
    "clip": ("clip_path",),
    "image_encoder": ("image_encoder_path",),
    "byt5": ("byt5_path",),
}

TEXT_ENCODER_PATH_OVERRIDES = {
    ("Qwen Image", "generate"): ("text_encoder_vl_path", "text_encoder_path"),
    ("Long-CAT", "cache"): ("t5_path", "text_encoder_path", "text_encoder_vl_path"),
    ("Long-CAT", "train"): ("t5_path", "text_encoder_path", "text_encoder_vl_path"),
    ("Long-CAT", "generate"): ("t5_path", "text_encoder_path", "text_encoder_vl_path"),
}

NETWORK_MODULE_BY_ARCH = {
    "FLUX.2": "networks.lora_flux_2",
    "Wan2.1": "networks.lora_wan",
    "HunyuanVideo": "networks.lora",
    "FramePack": "networks.lora_framepack",
    "FLUX Kontext": "networks.lora_flux",
    "Qwen Image": "networks.lora_qwen_image",
    "Long-CAT": "networks.lora_longcat",
    "Z-Image": "networks.lora_zimage",
    "HV 1.5": "networks.lora_hv_1_5",
}

CACHE_LATENT_SCALARS = {
    "vae_dtype": "--vae_dtype",
    "device": "--device",
    "batch_size": "--batch_size",
    "vae_chunk_size": "--vae_chunk_size",
    "vae_spatial_tile_sample_min_size": "--vae_spatial_tile_sample_min_size",
    "vae_sample_size": "--vae_sample_size",
}

CACHE_LATENT_BOOLS = {
    "skip_existing": "--skip_existing",
    "vae_tiling": "--vae_tiling",
    "vae_cache_cpu": "--vae_cache_cpu",
    "i2v": "--i2v",
    "one_frame": "--one_frame",
    "one_frame_no_2x": "--one_frame_no_2x",
    "one_frame_no_4x": "--one_frame_no_4x",
    "f1_mode": "--f1",
    "vae_enable_patch_conv": "--vae_enable_patch_conv",
    "edit_mode": "--edit",
    "edit_plus": "--edit_plus",
}

CACHE_TEXT_SCALARS = {
    "te_batch_size": "--batch_size",
    "te_device": "--device",
    "text_encoder_dtype": "--text_encoder_dtype",
}

CACHE_TEXT_BOOLS = {
    "te_skip_existing": "--skip_existing",
    "fp8_text_encoder": "--fp8_text_encoder",
    "fp8_llm": "--fp8_llm",
    "fp8_t5": "--fp8_t5",
    "fp8_vl": "--fp8_vl",
}

DOPSD_TEACHER_EMBED_KEY = "dopsd_teacher_llm_embed"

CACHE_LATENT_ARCH_SCALAR_KEYS = {
    "HunyuanVideo": {"vae_chunk_size", "vae_spatial_tile_sample_min_size"},
    "Qwen Image": {"vae_chunk_size", "vae_spatial_tile_sample_min_size"},
    "HV 1.5": {"vae_sample_size"},
}

CACHE_LATENT_ARCH_BOOL_KEYS = {
    "HunyuanVideo": {"vae_tiling"},
    "Qwen Image": {"vae_tiling", "edit_mode", "edit_plus"},
    "Wan2.1": {"vae_cache_cpu", "i2v", "one_frame"},
    "FramePack": {"f1_mode", "one_frame", "one_frame_no_2x", "one_frame_no_4x"},
    "Long-CAT": {"i2v"},
    "Z-Image": {"i2v"},
    "HV 1.5": {"i2v", "vae_enable_patch_conv"},
}

CACHE_TEXT_ARCH_BOOL_KEYS = {
    "FLUX.2": {"fp8_text_encoder"},
    "FLUX Kontext": {"fp8_t5"},
    "Wan2.1": {"fp8_t5"},
    "HunyuanVideo": {"fp8_llm"},
    "FramePack": {"fp8_llm"},
    "Long-CAT": {"fp8_t5"},
    "Qwen Image": {"fp8_vl"},
    "HV 1.5": {"fp8_vl"},
    "Z-Image": {"fp8_llm"},
}

CACHE_LATENT_ARCH_SCALAR_KEY_UNION = set().union(*CACHE_LATENT_ARCH_SCALAR_KEYS.values())
CACHE_LATENT_ARCH_BOOL_KEY_UNION = set().union(*CACHE_LATENT_ARCH_BOOL_KEYS.values())
CACHE_TEXT_ARCH_BOOL_KEY_UNION = set().union(*CACHE_TEXT_ARCH_BOOL_KEYS.values())

TRAIN_SCALARS = {
    "output_name": "--output_name",
    "seed": "--seed",
    "vae_dtype": "--vae_dtype",
    "max_data_loader_n_workers": "--max_data_loader_n_workers",
    "max_grad_norm": "--max_grad_norm",
    "training_comment": "--training_comment",
    "metadata_title": "--metadata_title",
    "metadata_author": "--metadata_author",
    "metadata_description": "--metadata_description",
    "metadata_license": "--metadata_license",
    "metadata_tags": "--metadata_tags",
    "huggingface_repo_id": "--huggingface_repo_id",
    "huggingface_repo_type": "--huggingface_repo_type",
    "huggingface_path_in_repo": "--huggingface_path_in_repo",
    "huggingface_token": "--huggingface_token",
    "huggingface_repo_visibility": "--huggingface_repo_visibility",
    "text_encoder_dtype": "--text_encoder_dtype",
    "vae_chunk_size": "--vae_chunk_size",
    "vae_spatial_tile_sample_min_size": "--vae_spatial_tile_sample_min_size",
    "vae_sample_size": "--vae_sample_size",
    "timestep_boundary": "--timestep_boundary",
    "latent_window_size": "--latent_window_size",
    "num_layers": "--num_layers",
    "flow_target": "--flow_target",
    "longcat_flow_target": "--flow_target",
}

TRAIN_BOOLS = {
    "gradient_checkpointing": "--gradient_checkpointing",
    "split_attn": "--split_attn",
    "fp8_base": "--fp8_base",
    "fp8_scaled": "--fp8_scaled",
    "fp8_text_encoder": "--fp8_text_encoder",
    "fp8_llm": "--fp8_llm",
    "fp8_t5": "--fp8_t5",
    "fp8_vl": "--fp8_vl",
    "vae_tiling": "--vae_tiling",
    "vae_cache_cpu": "--vae_cache_cpu",
    "vae_enable_patch_conv": "--vae_enable_patch_conv",
    "one_frame": "--one_frame",
    "f1_mode": "--f1",
    "edit_mode": "--edit",
    "edit_plus": "--edit_plus",
    "longcat_i2v": "--longcat_i2v",
    "img_in_txt_in_offloading": "--img_in_txt_in_offloading",
    "disable_numpy_memmap": "--disable_numpy_memmap",
    "use_32bit_attention": "--use_32bit_attention",
    "bulk_decode": "--bulk_decode",
    "force_v2_1_time_embedding": "--force_v2_1_time_embedding",
    "offload_inactive_dit": "--offload_inactive_dit",
    "remove_first_image_from_target": "--remove_first_image_from_target",
    "persistent_workers": "--persistent_data_loader_workers",
    "cuda_allow_tf32": "--cuda_allow_tf32",
    "cuda_cudnn_benchmark": "--cuda_cudnn_benchmark",
    "async_upload": "--async_upload",
    "save_state_to_huggingface": "--save_state_to_huggingface",
    "resume_from_huggingface": "--resume_from_huggingface",
}

TRAIN_PATHS = {
    "resume_path": "--resume",
    "network_weights": "--network_weights",
    "dit_high_noise": "--dit_high_noise",
}

TRAIN_ARCH_SCALAR_KEYS = {
    "HunyuanVideo": {"text_encoder_dtype", "vae_chunk_size", "vae_spatial_tile_sample_min_size"},
    "FramePack": {"vae_chunk_size", "vae_spatial_tile_sample_min_size", "latent_window_size"},
    "HV 1.5": {"vae_sample_size"},
    "Long-CAT": {"flow_target", "longcat_flow_target"},
    "Wan2.1": {"timestep_boundary"},
    "Qwen Image": {"num_layers"},
}

TRAIN_ARCH_BOOL_KEYS = {
    "FLUX.2": {"fp8_scaled", "fp8_text_encoder"},
    "FLUX Kontext": {"fp8_scaled", "fp8_t5"},
    "Qwen Image": {"fp8_scaled", "fp8_vl", "edit_mode", "edit_plus", "remove_first_image_from_target"},
    "Wan2.1": {"fp8_scaled", "fp8_t5", "vae_cache_cpu", "one_frame", "force_v2_1_time_embedding", "offload_inactive_dit"},
    "HunyuanVideo": {"fp8_llm", "vae_tiling"},
    "FramePack": {"fp8_scaled", "fp8_llm", "vae_tiling", "bulk_decode", "f1_mode", "one_frame"},
    "Long-CAT": {"fp8_scaled", "fp8_t5", "vae_cache_cpu", "longcat_i2v"},
    "Z-Image": {"fp8_scaled", "fp8_llm", "use_32bit_attention"},
    "HV 1.5": {"fp8_scaled", "fp8_vl", "vae_enable_patch_conv"},
}

TRAIN_ARCH_PATH_KEYS = {
    "Wan2.1": {"dit_high_noise"},
}

TRAIN_ARCH_SCALAR_KEY_UNION = set().union(*TRAIN_ARCH_SCALAR_KEYS.values())
TRAIN_ARCH_BOOL_KEY_UNION = set().union(*TRAIN_ARCH_BOOL_KEYS.values())
TRAIN_ARCH_PATH_KEY_UNION = set().union(*TRAIN_ARCH_PATH_KEYS.values())

TRAIN_LORA_PATH_KEYS = {"network_weights"}

TRAIN_FINETUNE_BOOLS = {
    "full_bf16": "--full_bf16",
    "fused_backward_pass": "--fused_backward_pass",
    "mem_eff_save": "--mem_eff_save",
    "block_swap_optimizer_patch_params": "--block_swap_optimizer_patch_params",
}

TRAIN_FINETUNE_ARCH_BOOL_KEYS = {
    "Qwen Image": {"full_bf16", "fused_backward_pass", "mem_eff_save"},
    "Z-Image": {"full_bf16", "fused_backward_pass", "mem_eff_save", "block_swap_optimizer_patch_params"},
}

TRAIN_FINETUNE_DISABLED_BOOL_KEYS = {"fp8_base", "fp8_scaled"}

GENERATE_SCALARS = {
    "vae_dtype": "--vae_dtype",
    "lora_multiplier": "--lora_multiplier",
    "lora_multiplier_high_noise": "--lora_multiplier_high_noise",
    "include_patterns": "--include_patterns",
    "exclude_patterns": "--exclude_patterns",
    "prompt": "--prompt",
    "negative_prompt": "--negative_prompt",
    "video_length": "--video_length",
    "fps": "--fps",
    "infer_steps": "--infer_steps",
    "seed": "--seed",
    "guidance_scale": "--guidance_scale",
    "embedded_cfg_scale": "--embedded_cfg_scale",
    "flow_shift": "--flow_shift",
    "sample_solver": "--sample_solver",
    "output_type": "--output_type",
    "device": "--device",
    "attn_mode": "--attn_mode",
    "blocks_to_swap": "--blocks_to_swap",
    "cfg_skip_mode": "--cfg_skip_mode",
    "cfg_apply_ratio": "--cfg_apply_ratio",
    "slg_layers": "--slg_layers",
    "slg_scale": "--slg_scale",
    "slg_start": "--slg_start",
    "slg_end": "--slg_end",
    "slg_mode": "--slg_mode",
    "trim_tail_frames": "--trim_tail_frames",
    "timestep_boundary": "--timestep_boundary",
    "magcache_mag_ratios": "--magcache_mag_ratios",
    "magcache_retention_ratio": "--magcache_retention_ratio",
    "magcache_threshold": "--magcache_threshold",
    "magcache_k": "--magcache_k",
    "vae_chunk_size": "--vae_chunk_size",
    "vae_spatial_tile_sample_min_size": "--vae_spatial_tile_sample_min_size",
    "latent_window_size": "--latent_window_size",
    "video_seconds": "--video_seconds",
    "video_sections": "--video_sections",
    "one_frame_inference": "--one_frame_inference",
    "rcm_threshold": "--rcm_threshold",
    "rcm_kernel_size": "--rcm_kernel_size",
    "rcm_dilate_size": "--rcm_dilate_size",
    "longcat_flow_target": "--flow_target",
    "compile_backend": "--compile_backend",
    "compile_mode": "--compile_mode",
    "compile_dynamic": "--compile_dynamic",
    "compile_cache_size_limit": "--compile_cache_size_limit",
}

GENERATE_BOOLS = {
    "fp8": "--fp8",
    "fp8_base": "--fp8",
    "fp8_scaled": "--fp8_scaled",
    "fp8_fast": "--fp8_fast",
    "fp8_text_encoder": "--fp8_text_encoder",
    "fp8_llm": "--fp8_llm",
    "fp8_t5": "--fp8_t5",
    "text_encoder_cpu": "--text_encoder_cpu",
    "vae_cache_cpu": "--vae_cache_cpu",
    "split_attn": "--split_attn",
    "img_in_txt_in_offloading": "--img_in_txt_in_offloading",
    "offload_inactive_dit": "--offload_inactive_dit",
    "cpu_noise": "--cpu_noise",
    "no_metadata": "--no_metadata",
    "lycoris": "--lycoris",
    "use_pinned_memory": "--use_pinned_memory_for_block_swap",
    "magcache_calibration": "--magcache_calibration",
    "f1_mode": "--f1",
    "bulk_decode": "--bulk_decode",
    "edit_mode": "--edit",
    "edit_plus": "--edit_plus",
    "rcm_relative_threshold": "--rcm_relative_threshold",
    "rcm_debug_save": "--rcm_debug_save",
    "longcat_i2v": "--longcat_i2v",
    "disable_numpy_memmap": "--disable_numpy_memmap",
    "use_32bit_attention": "--use_32bit_attention",
    "vae_enable_patch_conv": "--vae_enable_patch_conv",
    "no_resize_control": "--no_resize_control",
    "compile": "--compile",
    "compile_fullgraph": "--compile_fullgraph",
}

GENERATE_PATHS = {
    "dit_high_noise": "--dit_high_noise",
    "lora_weight": "--lora_weight",
    "lora_weight_high_noise": "--lora_weight_high_noise",
    "from_file": "--from_file",
    "latent_path": "--latent_path",
    "image_path": "--image_path",
    "end_image_path": "--end_image_path",
    "control_image_path": "--control_image_path",
    "control_path": "--control_path",
    "image_mask_path": "--image_mask_path",
    "end_image_mask_path": "--end_image_mask_path",
    "mask_path": "--mask_path",
}

GENERATE_BASE_SCALAR_KEYS = {
    "vae_dtype",
    "lora_multiplier",
    "include_patterns",
    "exclude_patterns",
    "prompt",
    "infer_steps",
    "seed",
    "flow_shift",
    "output_type",
    "device",
    "attn_mode",
    "blocks_to_swap",
}

GENERATE_BASE_BOOL_KEYS = {
    "fp8",
    "fp8_base",
    "no_metadata",
    "lycoris",
    "use_pinned_memory",
}

GENERATE_BASE_PATH_KEYS = {
    "lora_weight",
    "from_file",
    "latent_path",
}

GENERATE_SCALAR_KEYS_BY_ARCH = {
    "FLUX.2": {"embedded_cfg_scale", "negative_prompt", "guidance_scale"},
    "FLUX Kontext": {"embedded_cfg_scale"},
    "Qwen Image": {"embedded_cfg_scale", "negative_prompt", "guidance_scale", "rcm_threshold", "rcm_kernel_size", "rcm_dilate_size"},
    "Z-Image": {"embedded_cfg_scale", "negative_prompt", "guidance_scale"},
    "Wan2.1": {
        "video_length",
        "fps",
        "sample_solver",
        "negative_prompt",
        "guidance_scale",
        "lora_multiplier_high_noise",
        "cfg_skip_mode",
        "cfg_apply_ratio",
        "slg_layers",
        "slg_scale",
        "slg_start",
        "slg_end",
        "slg_mode",
        "trim_tail_frames",
        "timestep_boundary",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_cache_size_limit",
    },
    "HunyuanVideo": {
        "video_length",
        "fps",
        "negative_prompt",
        "guidance_scale",
        "embedded_cfg_scale",
        "vae_chunk_size",
        "vae_spatial_tile_sample_min_size",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_cache_size_limit",
    },
    "FramePack": {
        "fps",
        "sample_solver",
        "negative_prompt",
        "guidance_scale",
        "embedded_cfg_scale",
        "vae_chunk_size",
        "vae_spatial_tile_sample_min_size",
        "latent_window_size",
        "video_seconds",
        "video_sections",
        "one_frame_inference",
        "magcache_mag_ratios",
        "magcache_retention_ratio",
        "magcache_threshold",
        "magcache_k",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_cache_size_limit",
    },
    "HV 1.5": {
        "video_length",
        "fps",
        "negative_prompt",
        "guidance_scale",
        "vae_sample_size",
        "compile_backend",
        "compile_mode",
        "compile_dynamic",
        "compile_cache_size_limit",
    },
}

GENERATE_BOOL_KEYS_BY_ARCH = {
    "FLUX.2": {"fp8_scaled", "fp8_text_encoder", "no_resize_control"},
    "FLUX Kontext": {"fp8_scaled", "fp8_t5", "no_resize_control"},
    "Qwen Image": {"fp8_scaled", "text_encoder_cpu", "edit_mode", "edit_plus", "rcm_relative_threshold", "rcm_debug_save"},
    "Z-Image": {"fp8_scaled", "fp8_llm", "text_encoder_cpu", "use_32bit_attention"},
    "Wan2.1": {
        "fp8_scaled",
        "fp8_fast",
        "fp8_t5",
        "vae_cache_cpu",
        "offload_inactive_dit",
        "cpu_noise",
        "magcache_calibration",
        "compile",
        "compile_fullgraph",
    },
    "HunyuanVideo": {
        "fp8_fast",
        "fp8_llm",
        "split_attn",
        "img_in_txt_in_offloading",
        "compile",
        "compile_fullgraph",
    },
    "FramePack": {
        "fp8_scaled",
        "fp8_llm",
        "f1_mode",
        "bulk_decode",
        "magcache_calibration",
        "compile",
        "compile_fullgraph",
    },
    "HV 1.5": {"fp8_scaled", "text_encoder_cpu", "vae_enable_patch_conv", "cpu_noise", "compile", "compile_fullgraph"},
}

GENERATE_PATH_KEYS_BY_ARCH = {
    "FLUX.2": {"control_image_path"},
    "FLUX Kontext": {"control_image_path"},
    "Qwen Image": {"control_image_path", "mask_path"},
    "Z-Image": set(),
    "Wan2.1": {"dit_high_noise", "lora_weight_high_noise", "image_path", "end_image_path", "control_image_path", "control_path"},
    "HunyuanVideo": {"image_path"},
    "FramePack": {"image_path", "end_image_path", "control_image_path"},
    "HV 1.5": {"image_path"},
}


def build_cache_jobs(
    state: Mapping[str, Any],
    project_dir: str | Path,
    project_config: Mapping[str, Any],
) -> list[CommandJob]:
    arch_name, arch = _resolve_architecture(state)
    dataset_config = _export_dataset(project_dir, project_config)
    latent_module = _required_module(arch, "cache_module", arch_name, "cache latents")
    text_module = _required_module(arch, "cache_te_module", arch_name, "cache text encoder")
    dopsd_teacher_cache = arch_name == "Z-Image" and _truthy(state.get("dopsd_cache_teacher_outputs"))
    zimage_i2v_cache = arch_name == "Z-Image" and _zimage_i2v_cache_requested(state, project_config)
    latent_state = dict(state)
    if zimage_i2v_cache:
        latent_state["i2v"] = True
        _require_zimage_i2v_image_encoder(latent_state)

    latent_args = [f"--dataset_config={dataset_config}"]
    _add_model_version(latent_args, latent_state, arch_name)
    _add_model_path(latent_args, latent_state, arch_name, "cache", "vae")
    if arch_name in {"Wan2.1"}:
        _add_model_path(latent_args, latent_state, arch_name, "cache", "clip")
    if arch_name in {"FramePack"} or zimage_i2v_cache:
        _add_model_path(latent_args, latent_state, arch_name, "cache", "image_encoder")
    _add_mapped_scalars(latent_args, latent_state, _cache_latent_scalars_for_arch(arch_name))
    _add_positive_int_scalar(latent_args, "--num_workers", latent_state.get("num_workers"))
    _add_cache_debug_args(latent_args, latent_state)
    _add_mapped_bools(latent_args, latent_state, _cache_latent_bools_for_arch(arch_name))

    text_args = [f"--dataset_config={dataset_config}"]
    _add_model_version(text_args, state, arch_name)
    for path_key in _cache_text_encoder_paths(arch_name):
        _add_model_path(text_args, state, arch_name, "cache", path_key)
    _add_mapped_scalars(text_args, state, CACHE_TEXT_SCALARS)
    text_bools = dict(CACHE_TEXT_BOOLS)
    if arch_name == "Long-CAT":
        text_bools.pop("fp8_vl", None)
        if _truthy(state.get("fp8_vl")) and not _truthy(state.get("fp8_t5")):
            text_args.append("--fp8_t5")
    _add_mapped_bools(text_args, state, _cache_text_bools_for_arch(arch_name, text_bools))
    _add_zimage_dopsd_cache_teacher_args(text_args, state, arch_name)
    _add_positive_int_scalar(text_args, "--num_workers", state.get("te_num_workers"))

    return [
        CommandJob(
            name=f"{arch_name} Cache Latents",
            script_key=latent_module,
            args=latent_args,
        ),
        CommandJob(
            name=f"{arch_name} Cache Text Encoder",
            script_key=text_module,
            args=text_args,
            runner_kwargs=_dopsd_runner_kwargs(project_dir) if dopsd_teacher_cache else {},
        ),
    ]


def build_train_job(
    state: Mapping[str, Any],
    project_dir: str | Path,
    project_config: Mapping[str, Any],
) -> CommandJob:
    arch_name, arch = _resolve_architecture(state)
    dataset_config = _export_dataset(project_dir, project_config)
    train_mode = _normalize_train_mode(state.get("train_mode"))
    is_lora_train = train_mode == "lora"
    dopsd_train = _truthy(state.get("dopsd"))
    train_module = _train_module_for_mode(arch, arch_name, train_mode)

    args = [f"--dataset_config={dataset_config}"]
    _add_model_version(args, state, arch_name)
    _add_task(args, state)
    _add_required_model_paths(args, state, arch_name, arch, "train")

    output_dir = _default_output_dir(project_dir, state.get("output_dir"))
    _add_scalar(args, "--output_dir", output_dir)
    if is_lora_train:
        _add_network_module(args, state, arch_name)
    _add_mapped_paths(args, state, _train_paths_for_arch(arch_name, include_lora=is_lora_train))
    if is_lora_train:
        _add_train_base_weights_args(args, state)
    _add_train_learning_rate_args(args, state)
    _add_train_core_args(args, state)
    _add_train_timestep_args(args, state, arch_name)
    _add_train_weighting_args(args, state)
    _add_train_soar_args(args, state, arch_name, train_mode)
    _add_train_dopsd_args(args, state, arch_name, train_mode)
    _add_train_lr_scheduler_args(args, state)
    if is_lora_train:
        _add_train_network_args(args, state)
    _add_mapped_scalars(args, state, _train_scalars_for_arch(arch_name))
    train_bools = _train_bools_for_arch(arch_name)
    if not is_lora_train:
        train_bools = {
            key: flag
            for key, flag in train_bools.items()
            if key not in TRAIN_FINETUNE_DISABLED_BOOL_KEYS
        }
    _add_mapped_bools(args, state, train_bools)
    _add_train_gradient_checkpointing_args(args, state)
    _add_train_block_swap_args(args, state)
    _add_train_compile_args(args, state)
    _add_train_ddp_args(args, state)
    _add_train_save_state_args(args, state)
    wandb_api_key = _add_train_wandb_args(args, state)
    _add_train_save_frequency_args(args, state)
    _add_train_sample_args(args, state)
    _add_train_attention_args(args, state)
    if is_lora_train:
        _add_train_network_extra_args(args, state)
    else:
        _add_train_finetune_args(args, state, arch_name)
    _add_train_optimizer_args(args, state)

    mixed_precision = _normalize_train_mixed_precision(state.get("mixed_precision"))
    runner_kwargs = {
        "use_accelerate": True,
        "mixed_precision": mixed_precision,
        "num_cpu_threads_per_process": _as_int(state.get("num_cpu_threads_per_process"), 8),
        "accelerate_args": _train_accelerate_args(state),
    }
    if wandb_api_key:
        runner_kwargs.setdefault("env_vars", {})["WANDB_API_KEY"] = wandb_api_key
    if dopsd_train:
        runner_kwargs.setdefault("env_vars", {}).update(_dopsd_runner_env(project_dir))
    return CommandJob(
        name=f"{arch_name} {'LoRA Train' if is_lora_train else 'Fine-tune'}",
        script_key=_module_to_script_path(train_module, source_root=DOPSD_SOURCE_ROOT if dopsd_train else "musubi-tuner"),
        args=args,
        runner_kwargs=runner_kwargs,
    )


def build_generate_job(state: Mapping[str, Any], project_dir: str | Path) -> CommandJob:
    arch_name, arch = _resolve_architecture(state)
    generate_module = arch.get("generate_module")
    if not generate_module:
        raise CommandBuildError(
            f"{arch_name} generate has no local Python entry point; use the compatibility launcher for the external PS1 workflow."
        )
    _validate_generate_prompt_source(state)

    args: list[str] = []
    _add_model_version(args, state, arch_name)
    _add_task(args, state)
    _add_required_model_paths(args, state, arch_name, arch, "generate")
    save_path = _default_generate_dir(project_dir, state.get("save_path"))
    _add_scalar(args, "--save_path", save_path)
    _add_size(args, "--video_size" if arch.get("is_video") else "--image_size", state.get("video_size"))
    _add_mapped_paths(args, state, _generate_paths_for_arch(arch_name))
    _add_generate_scalars(args, state, arch_name)
    _add_save_merged_model(args, state, save_path)
    _add_mapped_bools(args, state, _generate_bools_for_arch(arch_name))

    return CommandJob(
        name=f"{arch_name} Generate",
        script_key=str(generate_module),
        args=args,
        runner_kwargs=_generate_runner_kwargs(state),
    )


def _validate_generate_prompt_source(state: Mapping[str, Any]) -> None:
    if _has_value(state.get("prompt")) or _has_value(state.get("from_file")) or _has_value(state.get("latent_path")):
        return
    raise CommandBuildError("Generate requires a prompt, prompt file, or latent path.")


def _generate_scalars_for_arch(arch_name: str) -> dict[str, str]:
    keys = set(GENERATE_BASE_SCALAR_KEYS)
    keys.update(GENERATE_SCALAR_KEYS_BY_ARCH.get(arch_name, set()))
    return _select_mapping(GENERATE_SCALARS, keys)


def _generate_bools_for_arch(arch_name: str) -> dict[str, str]:
    keys = set(GENERATE_BASE_BOOL_KEYS)
    keys.update(GENERATE_BOOL_KEYS_BY_ARCH.get(arch_name, set()))
    return _select_mapping(GENERATE_BOOLS, keys)


def _add_generate_scalars(args: list[str], state: Mapping[str, Any], arch_name: str) -> None:
    for key, flag in _generate_scalars_for_arch(arch_name).items():
        value = state.get(key)
        if key == "guidance_scale" and _is_zero_scalar(value):
            continue
        _add_scalar(args, flag, value)


def _generate_paths_for_arch(arch_name: str) -> dict[str, str]:
    keys = set(GENERATE_BASE_PATH_KEYS)
    keys.update(GENERATE_PATH_KEYS_BY_ARCH.get(arch_name, set()))
    return _select_mapping(GENERATE_PATHS, keys)


def _add_save_merged_model(args: list[str], state: Mapping[str, Any], save_path: str) -> None:
    value = state.get("save_merged_model")
    if not _truthy(value):
        return
    if isinstance(value, str) and value.strip().lower() not in {"1", "true", "yes", "on"}:
        _add_scalar(args, "--save_merged_model", value)
        return
    _add_scalar(args, "--save_merged_model", str(Path(save_path) / "merged_model.safetensors"))


def _generate_runner_kwargs(state: Mapping[str, Any]) -> dict[str, Any]:
    if not _truthy(state.get("enable_tf32")):
        return {}
    return {"env_vars": {"NVIDIA_TF32_OVERRIDE": "1"}}


def _select_mapping(mapping: Mapping[str, str], keys: set[str]) -> dict[str, str]:
    return {key: mapping[key] for key in keys if key in mapping}


def _resolve_architecture(state: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    arch_name = str(state.get("arch") or "FLUX.2")
    arch = model_catalog.get_architecture(arch_name)
    if arch is None:
        raise CommandBuildError(f"Unknown model architecture: {arch_name}")
    return arch_name, arch


def _required_module(arch: Mapping[str, Any], key: str, arch_name: str, label: str) -> str:
    module = arch.get(key)
    if not module:
        raise CommandBuildError(f"{arch_name} does not provide a {label} module")
    return str(module)


def _zimage_i2v_cache_requested(state: Mapping[str, Any], project_config: Mapping[str, Any]) -> bool:
    return (
        _truthy(state.get("i2v"))
        or _has_value(_first_value(state, MODEL_PATH_STATE_KEYS.get("image_encoder", ())))
        or _project_has_control_dataset(project_config)
    )


def _require_zimage_i2v_image_encoder(state: Mapping[str, Any]) -> None:
    if _has_value(_first_value(state, MODEL_PATH_STATE_KEYS.get("image_encoder", ()))):
        return
    raise CommandBuildError("Z-Image SOAR/I2V cache requires an Image Encoder path.")


def _project_has_control_dataset(project_config: Mapping[str, Any]) -> bool:
    dataset_section = project_config.get("dataset")
    if not isinstance(dataset_section, Mapping):
        return False

    datasets = dataset_section.get("datasets")
    if not isinstance(datasets, list):
        return False

    return any(isinstance(dataset, Mapping) and _has_value(dataset.get("control_directory")) for dataset in datasets)


def _normalize_train_mode(value: Any) -> str:
    normalized = str(value or "lora").strip().lower().replace("-", "_")
    if normalized in {"finetune", "fine_tune", "full", "db", "dreambooth"}:
        return "finetune"
    return "lora"


def _train_module_for_mode(arch: Mapping[str, Any], arch_name: str, train_mode: str) -> str:
    if train_mode == "finetune":
        return _required_module(arch, "finetune_module", arch_name, "fine-tune")
    return _required_module(arch, "train_module", arch_name, "train")


def _export_dataset(project_dir: str | Path, project_config: Mapping[str, Any]) -> str:
    output_path = get_default_dataset_config_path(project_dir)
    return str(export_dataset_config(project_config, output_path))


def _default_output_dir(project_dir: str | Path, value: Any) -> str:
    if _has_value(value):
        return str(value)
    return SCRIPT_DEFAULT_OUTPUT_DIR


def _default_generate_dir(project_dir: str | Path, value: Any) -> str:
    if _has_value(value):
        return str(value)
    return SCRIPT_DEFAULT_OUTPUT_DIR


def _cache_text_encoder_paths(arch_name: str) -> tuple[str, ...]:
    if arch_name == "Wan2.1":
        return ("t5",)
    if arch_name in {"HunyuanVideo", "FramePack", "FLUX Kontext"}:
        return ("te1", "te2")
    if arch_name == "HV 1.5":
        return ("text_encoder", "byt5")
    return ("text_encoder",)


def _add_required_model_paths(
    args: list[str],
    state: Mapping[str, Any],
    arch_name: str,
    arch: Mapping[str, Any],
    page_key: str,
) -> None:
    page_config = arch.get("pages", {}).get(page_key, {})
    required_paths = tuple(page_config.get("required_paths") or ())
    for path_key in required_paths:
        _add_model_path(args, state, arch_name, page_key, path_key)


def _add_model_path(
    args: list[str],
    state: Mapping[str, Any],
    arch_name: str,
    page_key: str,
    path_key: str,
) -> None:
    flag = MODEL_PATH_FLAGS.get(path_key)
    if not flag:
        return
    candidates = MODEL_PATH_STATE_KEYS.get(path_key, ())
    if path_key == "text_encoder":
        candidates = TEXT_ENCODER_PATH_OVERRIDES.get((arch_name, page_key), candidates)
    value = _first_value(state, candidates)
    _add_scalar(args, flag, value)


def _add_model_version(args: list[str], state: Mapping[str, Any], arch_name: str) -> None:
    if arch_name not in {"FLUX.2", "Qwen Image"}:
        return

    version = state.get("version")
    if not _has_value(version) or str(version).lower() == "default":
        edit_version = state.get("edit_version")
        version = edit_version if _has_value(edit_version) else None

    if not _has_value(version):
        return

    normalized = str(version)
    if arch_name == "Qwen Image":
        normalized = {
            "2509": "edit-2509",
            "2511": "edit-2511",
            "edit_plus": "edit-2509",
            "edit": "edit",
        }.get(normalized, normalized)
    _add_scalar(args, "--model_version", normalized)


def _add_task(args: list[str], state: Mapping[str, Any]) -> None:
    task = state.get("task")
    if _has_value(task):
        _add_scalar(args, "--task", task)


def _add_network_module(args: list[str], state: Mapping[str, Any], arch_name: str) -> None:
    if _truthy(state.get("enable_lycoris")):
        module = "lycoris.kohya"
    else:
        module = NETWORK_MODULE_BY_ARCH.get(arch_name)
    if module:
        _add_scalar(args, "--network_module", module)


def _add_mapped_paths(args: list[str], state: Mapping[str, Any], mapping: Mapping[str, str]) -> None:
    for key, flag in mapping.items():
        _add_path_or_list(args, flag, state.get(key))


def _add_mapped_scalars(args: list[str], state: Mapping[str, Any], mapping: Mapping[str, str]) -> None:
    for key, flag in mapping.items():
        _add_scalar(args, flag, state.get(key))


def _add_mapped_bools(args: list[str], state: Mapping[str, Any], mapping: Mapping[str, str]) -> None:
    for key, flag in mapping.items():
        if _truthy(state.get(key)):
            args.append(flag)


def _cache_latent_scalars_for_arch(arch_name: str) -> dict[str, str]:
    keys = set(CACHE_LATENT_SCALARS) - CACHE_LATENT_ARCH_SCALAR_KEY_UNION
    keys.update(CACHE_LATENT_ARCH_SCALAR_KEYS.get(arch_name, set()))
    return _select_mapping(CACHE_LATENT_SCALARS, keys)


def _cache_latent_bools_for_arch(arch_name: str) -> dict[str, str]:
    keys = set(CACHE_LATENT_BOOLS) - CACHE_LATENT_ARCH_BOOL_KEY_UNION
    keys.update(CACHE_LATENT_ARCH_BOOL_KEYS.get(arch_name, set()))
    return _select_mapping(CACHE_LATENT_BOOLS, keys)


def _cache_text_bools_for_arch(arch_name: str, mapping: Mapping[str, str]) -> dict[str, str]:
    keys = set(mapping) - CACHE_TEXT_ARCH_BOOL_KEY_UNION
    keys.update(CACHE_TEXT_ARCH_BOOL_KEYS.get(arch_name, set()))
    return _select_mapping(mapping, keys)


def _add_zimage_dopsd_cache_teacher_args(args: list[str], state: Mapping[str, Any], arch_name: str) -> None:
    if arch_name != "Z-Image" or not _truthy(state.get("dopsd_cache_teacher_outputs")):
        return

    teacher_encoder = _dopsd_value(state, "dopsd_teacher_text_encoder")
    if not _has_value(teacher_encoder):
        raise CommandBuildError("--dopsd_cache_teacher_outputs requires --dopsd_teacher_text_encoder.")

    reweight_source = _dopsd_value(state, "dopsd_teacher_llm_reweight_source")
    already_reweighted = _truthy(state.get("dopsd_teacher_already_reweighted"))
    allow_raw_vlm = _truthy(state.get("dopsd_teacher_allow_raw_vlm"))
    if not _has_value(reweight_source) and not already_reweighted and not allow_raw_vlm:
        raise CommandBuildError(
            "--dopsd_cache_teacher_outputs requires --dopsd_teacher_llm_reweight_source, "
            "--dopsd_teacher_already_reweighted, or --dopsd_teacher_allow_raw_vlm."
        )

    args.append("--dopsd_cache_teacher_outputs")
    _add_scalar(args, "--dopsd_teacher_text_encoder", teacher_encoder)
    _add_scalar(args, "--dopsd_teacher_processor", _dopsd_value(state, "dopsd_teacher_processor"))
    _add_scalar(args, "--dopsd_teacher_llm_reweight_source", reweight_source)
    if already_reweighted:
        args.append("--dopsd_teacher_already_reweighted")
    if allow_raw_vlm:
        args.append("--dopsd_teacher_allow_raw_vlm")
    _add_scalar(args, "--dopsd_teacher_dtype", state.get("dopsd_teacher_dtype"))
    if _truthy(state.get("dopsd_teacher_trust_remote_code")):
        args.append("--dopsd_teacher_trust_remote_code")
    _add_int_scalar(args, "--dopsd_teacher_hidden_state_index", state.get("dopsd_teacher_hidden_state_index"))
    _add_scalar(args, "--dopsd_teacher_embed_key", state.get("dopsd_teacher_embed_key") or DOPSD_TEACHER_EMBED_KEY)


def _train_scalars_for_arch(arch_name: str) -> dict[str, str]:
    keys = set(TRAIN_SCALARS) - TRAIN_ARCH_SCALAR_KEY_UNION
    keys.update(TRAIN_ARCH_SCALAR_KEYS.get(arch_name, set()))
    return _select_mapping(TRAIN_SCALARS, keys)


def _train_bools_for_arch(arch_name: str) -> dict[str, str]:
    conditional_keys = {"gradient_checkpointing"}
    keys = set(TRAIN_BOOLS) - TRAIN_ARCH_BOOL_KEY_UNION - conditional_keys
    keys.update(TRAIN_ARCH_BOOL_KEYS.get(arch_name, set()))
    return _select_mapping(TRAIN_BOOLS, keys)


def _train_paths_for_arch(arch_name: str, include_lora: bool = True) -> dict[str, str]:
    keys = set(TRAIN_PATHS) - TRAIN_ARCH_PATH_KEY_UNION
    keys.update(TRAIN_ARCH_PATH_KEYS.get(arch_name, set()))
    if not include_lora:
        keys.difference_update(TRAIN_LORA_PATH_KEYS)
    return _select_mapping(TRAIN_PATHS, keys)


def _add_train_core_args(args: list[str], state: Mapping[str, Any]) -> None:
    _add_scalar(args, "--mixed_precision", _normalize_train_mixed_precision(state.get("mixed_precision")))
    _add_positive_int_scalar(args, "--max_train_steps", state.get("max_train_steps"))
    _add_positive_int_scalar(args, "--max_train_epochs", state.get("max_train_epochs"))
    _add_positive_int_scalar(args, "--gradient_accumulation_steps", state.get("gradient_accumulation_steps"))
    if _as_float(state.get("guidance_scale"), 1.0) != 1.0:
        _add_scalar(args, "--guidance_scale", state.get("guidance_scale"))


def _normalize_train_mixed_precision(value: Any) -> str:
    precision = str(value or "bf16").strip().lower()
    return {"bfloat16": "bf16", "float16": "fp16"}.get(precision, precision)


def _add_train_timestep_args(args: list[str], state: Mapping[str, Any], arch_name: str) -> None:
    sampling = _normalize_timestep_sampling(state.get("timestep_sampling"), arch_name)
    if _has_value(sampling) and sampling != "sigma":
        _add_scalar(args, "--timestep_sampling", sampling)

    if sampling == "shift" and _as_float(state.get("discrete_flow_shift"), 1.0) != 1.0:
        _add_scalar(args, "--discrete_flow_shift", state.get("discrete_flow_shift"))
    if sampling in {"sigmoid", "shift"} and _as_float(state.get("sigmoid_scale"), 1.0) != 1.0:
        _add_scalar(args, "--sigmoid_scale", state.get("sigmoid_scale"))

    if _as_float(state.get("min_timestep"), 0.0) != 0.0:
        _add_scalar(args, "--min_timestep", state.get("min_timestep"))
    if _as_float(state.get("max_timestep"), 1000.0) != 1000.0:
        _add_scalar(args, "--max_timestep", state.get("max_timestep"))
    _add_scalar(args, "--show_timesteps", state.get("show_timesteps"))


def _normalize_timestep_sampling(value: Any, arch_name: str) -> str | None:
    if not _has_value(value):
        return None
    sampling = str(value).strip()
    if sampling == "qinglong":
        return "qinglong_qwen" if arch_name in {"Qwen Image", "Long-CAT"} else "qinglong_flux"
    return sampling


def _add_train_weighting_args(args: list[str], state: Mapping[str, Any]) -> None:
    weighting = state.get("weighting_scheme")
    if not _has_value(weighting):
        return

    normalized = str(weighting).strip()
    if normalized == "uniform":
        normalized = "none"
    if normalized not in {"logit_normal", "mode", "cosmap", "sigma_sqrt", "none"}:
        return

    _add_scalar(args, "--weighting_scheme", normalized)
    if normalized == "logit_normal":
        if _as_float(state.get("logit_mean"), 0.0) != 0.0:
            _add_scalar(args, "--logit_mean", state.get("logit_mean"))
        if _as_float(state.get("logit_std"), 1.0) != 1.0:
            _add_scalar(args, "--logit_std", state.get("logit_std"))
    elif normalized == "mode" and _as_float(state.get("mode_scale"), 1.29) != 1.29:
        _add_scalar(args, "--mode_scale", state.get("mode_scale"))


def _add_train_soar_args(args: list[str], state: Mapping[str, Any], arch_name: str, train_mode: str) -> None:
    if not model_catalog.supports_soar_training(arch_name, train_mode):
        return
    if not _truthy(state.get("soar")):
        return
    version = _soar_version(state, arch_name)
    if not model_catalog.supports_soar_training(arch_name, train_mode, version=version):
        raise CommandBuildError(f"{arch_name} SOAR is not supported for model version {version}.")
    if train_mode != "lora" and _truthy(state.get("fused_backward_pass")):
        raise CommandBuildError("--soar is not compatible with --fused_backward_pass.")
    if arch_name == "Qwen Image" and _qwen_soar_incompatible(state):
        raise CommandBuildError("Qwen Image SOAR is only supported for original non-edit training.")

    args.append("--soar")
    _add_min_float_scalar(args, "--soar_lambda_aux", state.get("soar_lambda_aux"), 0.0)
    _add_min_int_scalar(args, "--soar_trajectory_length", state.get("soar_trajectory_length"), 1)
    _add_min_int_scalar(args, "--soar_num_sampling_steps", state.get("soar_num_sampling_steps"), 2)
    _add_min_float_scalar(args, "--soar_sigma_upper_ratio", state.get("soar_sigma_upper_ratio"), 1.0)
    cfg_scale_sampling = state.get("soar_cfg_scale_sampling")
    _add_positive_float_scalar(args, "--soar_cfg_scale_sampling", cfg_scale_sampling)
    if _soar_cfg_rollout_requested(cfg_scale_sampling):
        if not model_catalog.supports_soar_cfg_rollout(arch_name, train_mode, version=version):
            raise CommandBuildError(
                f"{arch_name} SOAR CFG rollout is not supported for train mode {train_mode} and model version {version}."
            )


def _soar_version(state: Mapping[str, Any], arch_name: str) -> Any:
    version = state.get("version") or state.get("edit_version")
    if _has_value(version):
        return version
    return model_catalog.get_default_version(arch_name, "train")


def _soar_cfg_rollout_requested(value: Any) -> bool:
    return _has_value(value) and abs(_as_float(value, 1.0) - 1.0) > 1e-6


def _add_train_dopsd_args(args: list[str], state: Mapping[str, Any], arch_name: str, train_mode: str) -> None:
    if not _truthy(state.get("dopsd")):
        return
    if not model_catalog.supports_dopsd_training(arch_name, train_mode):
        raise CommandBuildError("D-OPSD is only supported for Z-Image LoRA training.")

    args.append("--dopsd")
    _add_positive_float_scalar(args, "--dopsd_loss_weight", state.get("dopsd_loss_weight"))
    _add_min_int_scalar(args, "--dopsd_num_sampling_steps", state.get("dopsd_num_sampling_steps"), 1)
    _add_float_range_scalar(args, "--dopsd_ema_decay", state.get("dopsd_ema_decay"), 0.0, 1.0)
    _add_scalar(args, "--dopsd_teacher_embed_key", state.get("dopsd_teacher_embed_key") or DOPSD_TEACHER_EMBED_KEY)


def _qwen_soar_incompatible(state: Mapping[str, Any]) -> bool:
    version = str(state.get("version") or state.get("edit_version") or "").strip().lower()
    if version in {"2509", "2511", "edit", "edit-2509", "edit-2511", "edit_plus", "layered"}:
        return True
    return any(_truthy(state.get(key)) for key in ("edit_mode", "edit_plus", "remove_first_image_from_target"))


def _add_train_lr_scheduler_args(args: list[str], state: Mapping[str, Any]) -> None:
    if _as_float(state.get("lr_warmup_steps"), 0.0) != 0.0:
        _add_scalar(args, "--lr_warmup_steps", state.get("lr_warmup_steps"))
    if _as_float(state.get("lr_decay_steps"), 0.0) != 0.0:
        _add_scalar(args, "--lr_decay_steps", state.get("lr_decay_steps"))
    _add_scalar(args, "--lr_scheduler_num_cycles", state.get("lr_scheduler_num_cycles"))
    if _as_float(state.get("lr_scheduler_power"), 1.0) != 1.0:
        _add_scalar(args, "--lr_scheduler_power", state.get("lr_scheduler_power"))
    if _as_float(state.get("lr_scheduler_timescale"), 0.0) != 0.0:
        _add_scalar(args, "--lr_scheduler_timescale", state.get("lr_scheduler_timescale"))
    _add_scalar(args, "--lr_scheduler_min_lr_ratio", state.get("lr_scheduler_min_lr_ratio"))


def _add_train_network_args(args: list[str], state: Mapping[str, Any]) -> None:
    _add_positive_int_scalar(args, "--network_dim", state.get("network_dim"))
    if _as_float(state.get("network_alpha"), 0.0) != 0.0:
        _add_scalar(args, "--network_alpha", state.get("network_alpha"))
    if not _truthy(state.get("enable_lycoris")) and _as_float(state.get("network_dropout"), 0.0) != 0.0:
        _add_scalar(args, "--network_dropout", state.get("network_dropout"))
    if _as_float(state.get("scale_weight_norms"), 0.0) != 0.0:
        _add_scalar(args, "--scale_weight_norms", state.get("scale_weight_norms"))
    if _has_value(state.get("network_weights")) and _truthy(state.get("dim_from_weights")):
        args.append("--dim_from_weights")


def _add_train_base_weights_args(args: list[str], state: Mapping[str, Any]) -> None:
    if not _has_value(state.get("base_weights")):
        return
    _add_path_or_list(args, "--base_weights", state.get("base_weights"))
    _add_path_or_list(args, "--base_weights_multiplier", state.get("base_weights_multiplier"))


def _add_train_gradient_checkpointing_args(args: list[str], state: Mapping[str, Any]) -> None:
    if not _truthy(state.get("gradient_checkpointing")):
        return
    args.append("--gradient_checkpointing")
    if _truthy(state.get("gradient_checkpointing_cpu_offload")):
        args.append("--gradient_checkpointing_cpu_offload")


def _add_train_block_swap_args(args: list[str], state: Mapping[str, Any]) -> None:
    if not _is_positive_number(state.get("blocks_to_swap")):
        return
    _add_scalar(args, "--blocks_to_swap", state.get("blocks_to_swap"))
    if _truthy(state.get("use_pinned_memory")):
        args.append("--use_pinned_memory_for_block_swap")


def _add_train_compile_args(args: list[str], state: Mapping[str, Any]) -> None:
    if not _truthy(state.get("compile")):
        return
    args.append("--compile")
    _add_scalar(args, "--compile_backend", state.get("compile_backend"))
    _add_scalar(args, "--compile_mode", state.get("compile_mode"))
    if _truthy(state.get("compile_fullgraph")):
        args.append("--compile_fullgraph")
    _add_scalar(args, "--compile_dynamic", state.get("compile_dynamic"))
    _add_positive_int_scalar(args, "--compile_cache_size_limit", state.get("compile_cache_size_limit"))


def _add_train_ddp_args(args: list[str], state: Mapping[str, Any]) -> None:
    if not _truthy(state.get("multi_gpu")):
        return
    if _as_int(state.get("ddp_timeout"), 0) != 0:
        _add_scalar(args, "--ddp_timeout", state.get("ddp_timeout"))
    if _truthy(state.get("ddp_gradient_as_bucket_view")):
        args.append("--ddp_gradient_as_bucket_view")
    if _truthy(state.get("ddp_static_graph")):
        args.append("--ddp_static_graph")


def _add_train_save_state_args(args: list[str], state: Mapping[str, Any]) -> None:
    if _truthy(state.get("save_state_on_train_end")):
        args.append("--save_state_on_train_end")
        return

    if not _truthy(state.get("save_state")):
        return

    args.append("--save_state")
    _add_positive_int_scalar(args, "--save_last_n_epochs_state", state.get("save_last_n_epochs_state"))
    _add_positive_int_scalar(args, "--save_last_n_steps_state", state.get("save_last_n_steps_state"))


def _add_train_wandb_args(args: list[str], state: Mapping[str, Any]) -> str | None:
    if not _has_value(state.get("wandb_api_key")):
        return None
    api_key = str(state.get("wandb_api_key")).strip()
    _add_scalar(args, "--log_with", "wandb")
    tracker_name = state.get("output_name") if _has_value(state.get("output_name")) else "network_train"
    _add_scalar(args, "--log_tracker_name", tracker_name)
    return api_key


def _add_train_save_frequency_args(args: list[str], state: Mapping[str, Any]) -> None:
    if _is_positive_number(state.get("save_every_n_steps")):
        _add_scalar(args, "--save_every_n_steps", state.get("save_every_n_steps"))
        return
    if _is_positive_number(state.get("save_every_n_epochs")):
        _add_scalar(args, "--save_every_n_epochs", state.get("save_every_n_epochs"))

    _add_positive_int_scalar(args, "--save_last_n_epochs", state.get("save_last_n_epochs"))
    _add_positive_int_scalar(args, "--save_last_n_steps", state.get("save_last_n_steps"))


def _add_train_sample_args(args: list[str], state: Mapping[str, Any]) -> None:
    if not _truthy(state.get("enable_sample")):
        return

    if _truthy(state.get("sample_at_first")):
        args.append("--sample_at_first")

    if _is_positive_number(state.get("sample_every_n_steps")):
        _add_scalar(args, "--sample_every_n_steps", state.get("sample_every_n_steps"))
    elif _is_positive_number(state.get("sample_every_n_epochs")):
        _add_scalar(args, "--sample_every_n_epochs", state.get("sample_every_n_epochs"))

    _add_path_or_list(args, "--sample_prompts", state.get("sample_prompts"))


def _train_accelerate_args(state: Mapping[str, Any]) -> list[str]:
    launch_args: list[str] = []
    precision = state.get("mixed_precision")
    if _has_value(precision) and _normalize_train_mixed_precision(precision) == "bf16":
        launch_args.append("--downcast_bf16")
    if _truthy(state.get("multi_gpu")):
        launch_args.extend(["--multi_gpu", "--rdzv_backend=c10d"])
    return launch_args


def _add_train_learning_rate_args(args: list[str], state: Mapping[str, Any]) -> None:
    optimizer_key = str(state.get("optimizer_type") or "").strip().lower()
    learning_rate = "1" if optimizer_key in {"prodigy", "prodigy_adv", "lion_prodigy_adv"} else state.get("learning_rate")
    _add_scalar(args, "--learning_rate", learning_rate)

    if optimizer_key.endswith("schedulefree"):
        return
    _add_scalar(args, "--lr_scheduler", state.get("lr_scheduler"))


def _add_train_optimizer_args(args: list[str], state: Mapping[str, Any]) -> None:
    optimizer_type = state.get("optimizer_type")
    if not _has_value(optimizer_type):
        return

    resolved_type, template_args = _resolve_train_optimizer(state)
    optimizer_args = (
        _parse_optimizer_args_text(state.get("optimizer_extra_args"))
        if "optimizer_extra_args" in state
        else template_args
    )

    _add_scalar(args, "--optimizer_type", resolved_type)
    if optimizer_args:
        args.append("--optimizer_args")
        args.extend(optimizer_args)


def _resolve_train_optimizer(state: Mapping[str, Any]) -> tuple[str, list[str]]:
    optimizer_type = state.get("optimizer_type")
    raw_name = str(optimizer_type).strip()
    key = raw_name.lower()
    optimizer_args: list[str] = []
    resolved_type = raw_name

    if "." in raw_name:
        resolved_type = raw_name
    elif key == "adam":
        resolved_type = "optimi.Adam"
        optimizer_args.extend(["betas=.95,.98", "decouple_lr=True"])
    elif key == "adamw":
        resolved_type = "optimi.AdamW"
        optimizer_args.extend(["betas=.95,.98", "decouple_lr=True"])
    elif key == "adafactor":
        resolved_type = "pytorch_optimizer.AdaFactor"
        optimizer_args.extend(["scale_parameter=False", "warmup_init=False", "relative_step=False", "cautious=True"])
    elif key in {"pagedadamw8bit", "adamw8bit"}:
        resolved_type = raw_name
    elif key == "lion":
        resolved_type = "pytorch_optimizer.Lion"
        optimizer_args.append("cautious=True")
    elif key in {"lion8bit", "pagedlion8bit"}:
        resolved_type = raw_name
        optimizer_args.extend(["weight_decay=0.01", "betas=.95,.98"])
    elif key in {"ademamix", "ademamix8bit", "pagedademamix8bit"}:
        resolved_type = "pytorch_optimizer.AdEMAMix" if key == "ademamix" else raw_name
        optimizer_args.extend(["alpha=10", "cautious=True"] if key == "ademamix" else ["weight_decay=0.01"])
    elif key == "sophia":
        resolved_type = "pytorch_optimizer.SophiaH"
        optimizer_args.append("weight_decay=0.01")
    elif key == "prodigy":
        resolved_type = "pytorch_optimizer.Prodigy"
        optimizer_args.extend(["weight_decay=0.01", "betas=.9,.99", "decouple=True", "use_bias_correction=True"])
        _append_optimizer_arg_if_has_value(optimizer_args, "d_coef", state.get("d_coef"))
        if _as_float(state.get("lr_warmup_steps"), 0.0) != 0.0:
            optimizer_args.append("safeguard_warmup=True")
        _append_optimizer_arg_if_has_value(optimizer_args, "d0", state.get("d0"))
    elif key in {"ranger", "adan", "stableadamw"}:
        resolved_type = f"optimi.{raw_name}"
        optimizer_args.append("decouple_lr=True")
    elif key == "tiger":
        resolved_type = "pytorch_optimizer.Tiger"
        optimizer_args.append("weight_decay=0.01")
    elif key.endswith("schedulefree"):
        resolved_type = raw_name
        optimizer_args.extend(["weight_decay=0.08", "weight_lr_power=0.001"])
    elif key == "adammini":
        resolved_type = raw_name
        optimizer_args.append("weight_decay=0.01")
    elif key == "adamg":
        resolved_type = raw_name
        optimizer_args.extend(["weight_decay=0.1", "weight_decouple=True"])
    elif key == "came":
        resolved_type = "pytorch_optimizer.CAME"
        optimizer_args.append("weight_decay=0.01")
    elif key == "soap":
        resolved_type = "pytorch_optimizer.SOAP"
    elif key == "sgdsai":
        resolved_type = "pytorch_optimizer.SGDSaI"
    elif key == "adopt":
        resolved_type = "pytorch_optimizer.ADOPT"
        optimizer_args.append("cautious=True")
    elif key == "fira":
        resolved_type = "pytorch_optimizer.Fira"
        rank = state.get("network_dim") if _has_value(state.get("network_dim")) else 32
        optimizer_args.extend(["weight_decay=0.01", f"rank={rank}", "update_proj_gap=50", "scale=1", "projection_type='std'"])
    elif key in {"emonavi", "emofact", "emolynx", "emoneco"}:
        resolved_type = f"pytorch_optimizer.{raw_name}"
        optimizer_args.append("weight_decay=0.01")
    elif key == "emozeal":
        resolved_type = "pytorch_optimizer.EmoZeal"
        optimizer_args.extend(["weight_decay=0.01", "shadow_weight=0.1"])
    elif key == "simplified_ademamix":
        resolved_type = "adv_optm.SimplifiedAdEMAMix"
        if _truthy(state.get("compile")):
            optimizer_args.append("compiled_optimizer=True")
    elif key == "adamuon":
        resolved_type = "pytorch_optimizer.AdaMuon"
        optimizer_args.extend(["weight_decay=0.01", "adamw_lr=2e-4", "adamw_betas=.9,.95"])
    elif key == "adamw_adv":
        resolved_type = "adv_optm.AdamW_adv"
        optimizer_args.append("grams_moment=True")
        if _truthy(state.get("compile")):
            optimizer_args.append("compiled_optimizer=True")
    elif key == "adopt_adv":
        resolved_type = "adv_optm.Adopt_adv"
        optimizer_args.append("grams_moment=True")
        if _truthy(state.get("compile")):
            optimizer_args.append("compiled_optimizer=True")
    elif key == "prodigy_adv":
        resolved_type = "adv_optm.Prodigy_adv"
        optimizer_args.append("grams_moment=True")
        _append_optimizer_arg_if_has_value(optimizer_args, "d_coef", state.get("d_coef"))
        if _truthy(state.get("compile")):
            optimizer_args.append("compiled_optimizer=True")
        if _as_float(state.get("lr_warmup_steps"), 0.0) != 0.0:
            optimizer_args.append("growth_rate=1.02")
        _append_optimizer_arg_if_has_value(optimizer_args, "d0", state.get("d0"))
    elif key == "lion_adv":
        resolved_type = "adv_optm.Lion_adv"
        optimizer_args.append("cautious_mask=True")
        if _truthy(state.get("compile")):
            optimizer_args.append("compiled_optimizer=True")
    elif key == "lion_prodigy_adv":
        resolved_type = "adv_optm.Lion_Prodigy_adv"
        optimizer_args.append("grams_moment=True")
        _append_optimizer_arg_if_has_value(optimizer_args, "d_coef", state.get("d_coef"))
        if _truthy(state.get("compile")):
            optimizer_args.append("compiled_optimizer=True")
        if _as_float(state.get("lr_warmup_steps"), 0.0) != 0.0:
            optimizer_args.append("growth_rate=1.02")
        _append_optimizer_arg_if_has_value(optimizer_args, "d0", state.get("d0"))
    elif key == "bcos":
        resolved_type = "pytorch_optimizer.BCOS"
        optimizer_args.append("simple_cond=True")
    elif key == "ano":
        resolved_type = "pytorch_optimizer.Ano"

    return resolved_type, optimizer_args


def _append_optimizer_arg_if_has_value(optimizer_args: list[str], key: str, value: Any) -> None:
    if _has_value(value):
        optimizer_args.append(f"{key}={value}")


def _parse_optimizer_args_text(value: Any) -> list[str]:
    if value is None:
        return []
    tokens: list[str] = []
    for line in str(value).replace("\r", "\n").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        tokens.extend(part for part in line.split() if part)
    return tokens


def _add_train_attention_args(args: list[str], state: Mapping[str, Any]) -> None:
    mode = state.get("attn_mode")
    if not _has_value(mode):
        return

    normalized = str(mode).strip().lower().replace("-", "_")
    flag_by_mode = {
        "flash": "--flash_attn",
        "flash_attn": "--flash_attn",
        "flash2": "--flash_attn",
        "flash3": "--flash3",
        "xformers": "--xformers",
        "sageattn": "--sage_attn",
        "sage_attn": "--sage_attn",
        "sdpa": "--sdpa",
        "torch": "--sdpa",
    }
    flag = flag_by_mode.get(normalized)
    if flag:
        args.append(flag)


def _add_train_network_extra_args(args: list[str], state: Mapping[str, Any]) -> None:
    if _truthy(state.get("enable_lycoris")):
        _add_train_lycoris_args(args, state)
        return

    network_args: list[str] = []
    if _truthy(state.get("enable_lora_plus")):
        _append_network_arg_if_has_value(network_args, "loraplus_lr_ratio", state.get("loraplus_lr_ratio"))
    elif _truthy(state.get("enable_blocks")):
        _append_network_arg_if_has_value(network_args, "exclude_patterns", state.get("exclude_patterns"))
        _append_network_arg_if_has_value(network_args, "include_patterns", state.get("include_patterns"))

    if network_args:
        args.append("--network_args")
        args.extend(network_args)


def _add_train_finetune_args(args: list[str], state: Mapping[str, Any], arch_name: str) -> None:
    keys = TRAIN_FINETUNE_ARCH_BOOL_KEYS.get(arch_name, set())
    _add_mapped_bools(args, state, _select_mapping(TRAIN_FINETUNE_BOOLS, keys))


def _add_train_lycoris_args(args: list[str], state: Mapping[str, Any]) -> None:
    if not _truthy(state.get("enable_lycoris")):
        return

    network_args: list[str] = []
    algo = str(state.get("lycoris_algo") or "lokr").strip()
    if algo:
        network_args.append(f"algo={algo}")

    if algo not in {"ia3", "diag-oft"}:
        if algo != "full":
            _append_network_arg_if_nonzero(network_args, "conv_dim", state.get("lycoris_conv_dim"))
            if _as_float(state.get("lycoris_conv_dim"), 0.0) != 0.0:
                _append_network_arg_if_nonzero(network_args, "conv_alpha", state.get("lycoris_conv_alpha"))
            if _truthy(state.get("lycoris_use_tucker")):
                network_args.append("use_tucker=True")
            if algo != "dylora":
                if _truthy(state.get("lycoris_dora_wd")):
                    network_args.append("dora_wd=True")
                if _truthy(state.get("lycoris_bypass_mode")):
                    network_args.append("bypass_mode=True")
                if _truthy(state.get("lycoris_use_scalar")):
                    network_args.append("use_scalar=True")

        preset = state.get("lycoris_preset")
        if _has_value(preset):
            network_args.append(f"preset={preset}")

    if algo == "locon":
        _append_network_arg_if_nonzero(network_args, "dropout", state.get("lycoris_dropout"))

    if algo != "ia3" and _truthy(state.get("lycoris_train_norm")):
        network_args.append("train_norm=True")

    if algo == "lokr":
        _append_network_arg_if_has_value(network_args, "factor", state.get("lycoris_factor"))
        if _truthy(state.get("lycoris_decompose_both")):
            network_args.append("decompose_both=True")
        if _truthy(state.get("lycoris_full_matrix")):
            network_args.append("full_matrix=True")
    elif algo == "dylora":
        _append_network_arg_if_has_value(network_args, "block_size", state.get("lycoris_block_size"))
    elif algo == "diag-oft":
        if _as_float(state.get("lycoris_rescaled"), 0.0) != 0.0:
            network_args.append("rescaled=True")
        if _truthy(state.get("lycoris_constrain")):
            network_args.append(f"constrain={state.get('lycoris_constrain')}")

    if network_args:
        args.append("--network_args")
        args.extend(network_args)


def _append_network_arg_if_has_value(network_args: list[str], key: str, value: Any) -> None:
    if _has_value(value):
        network_args.append(f"{key}={value}")


def _append_network_arg_if_nonzero(network_args: list[str], key: str, value: Any) -> None:
    if _has_value(value) and _as_float(value, 0.0) != 0.0:
        network_args.append(f"{key}={value}")


def _add_cache_debug_args(args: list[str], state: Mapping[str, Any]) -> None:
    debug_mode = state.get("debug_mode")
    if not _has_value(debug_mode):
        return

    normalized_mode = str(debug_mode).strip()
    _add_scalar(args, "--debug_mode", normalized_mode)
    if normalized_mode != "console":
        return

    _add_int_scalar(args, "--console_width", state.get("console_width"))
    _add_scalar(args, "--console_back", state.get("console_back"))
    _add_int_scalar(args, "--console_num_images", state.get("console_num_images"))


def _add_int_scalar(args: list[str], flag: str, value: Any) -> None:
    if not _has_value(value):
        return
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return
    args.append(f"{flag}={int_value}")


def _add_positive_int_scalar(args: list[str], flag: str, value: Any) -> None:
    if not _has_value(value):
        return
    try:
        int_value = int(value)
    except (TypeError, ValueError):
        return
    if int_value <= 0:
        return
    args.append(f"{flag}={int_value}")


def _add_min_int_scalar(args: list[str], flag: str, value: Any, min_value: int) -> None:
    if not _has_value(value):
        return
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise CommandBuildError(f"{flag} must be an integer.") from exc
    if int_value < min_value:
        raise CommandBuildError(f"{flag} must be at least {min_value}.")
    args.append(f"{flag}={int_value}")


def _add_min_float_scalar(args: list[str], flag: str, value: Any, min_value: float) -> None:
    if not _has_value(value):
        return
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise CommandBuildError(f"{flag} must be a number.") from exc
    if float_value < min_value:
        raise CommandBuildError(f"{flag} must be at least {min_value}.")
    args.append(f"{flag}={value}")


def _add_positive_float_scalar(args: list[str], flag: str, value: Any) -> None:
    if not _has_value(value):
        return
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise CommandBuildError(f"{flag} must be a number.") from exc
    if float_value <= 0:
        raise CommandBuildError(f"{flag} must be positive.")
    args.append(f"{flag}={value}")


def _add_float_range_scalar(args: list[str], flag: str, value: Any, min_value: float, max_value: float) -> None:
    if not _has_value(value):
        return
    try:
        float_value = float(value)
    except (TypeError, ValueError) as exc:
        raise CommandBuildError(f"{flag} must be a number.") from exc
    if float_value < min_value or float_value > max_value:
        raise CommandBuildError(f"{flag} must be between {min_value} and {max_value}.")
    args.append(f"{flag}={value}")


def _add_path_or_list(args: list[str], flag: str, value: Any) -> None:
    if not _has_value(value):
        return
    if isinstance(value, (list, tuple)):
        args.append(flag)
        args.extend(str(item) for item in value if _has_value(item))
        return

    parts = _split_path_list(str(value))
    if len(parts) > 1 and flag in {
        "--lora_weight",
        "--lora_multiplier",
        "--lora_weight_high_noise",
        "--lora_multiplier_high_noise",
        "--latent_path",
    }:
        args.append(flag)
        args.extend(parts)
        return
    _add_scalar(args, flag, value)


def _add_size(args: list[str], flag: str, value: Any) -> None:
    if not _has_value(value):
        return
    parts = list(value) if isinstance(value, (list, tuple)) else _split_multi_value(str(value))
    if len(parts) != 2:
        raise CommandBuildError(f"{flag} requires width and height, got: {value}")
    args.append(flag)
    args.extend(str(part) for part in parts)


def _add_scalar(args: list[str], flag: str, value: Any) -> None:
    if not _has_value(value):
        return
    if isinstance(value, bool):
        if value:
            args.append(flag)
        return
    args.append(f"{flag}={value}")


def _is_zero_scalar(value: Any) -> bool:
    if not _has_value(value):
        return False
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _first_value(state: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = state.get(key)
        if _has_value(value):
            return value
    return None


def _dopsd_value(state: Mapping[str, Any], key: str) -> Any:
    return _first_value(state, (f"{key}_path", key))


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _split_multi_value(value: str) -> list[str]:
    return [part for part in value.replace(",", " ").split() if part]


def _split_path_list(value: str) -> list[str]:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace(";", "\n")
    return [part.strip() for part in normalized.split("\n") if part.strip()]


def _as_int(value: Any, default: int) -> int:
    if not _has_value(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if not _has_value(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_positive_number(value: Any) -> bool:
    return _has_value(value) and _as_float(value, 0.0) > 0.0


def _dopsd_runner_kwargs(project_dir: str | Path) -> dict[str, Any]:
    return {"env_vars": _dopsd_runner_env(project_dir)}


def _dopsd_runner_env(project_dir: str | Path) -> dict[str, str]:
    source_root = _resolve_dopsd_source_root(project_dir)
    project_root = source_root.parents[1]
    qinglong_captions = project_root / "qinglong-captions"
    if not qinglong_captions.exists():
        qinglong_captions = Path(project_dir) / "qinglong-captions"
    pythonpath = os.pathsep.join([str(source_root), str(project_root), str(qinglong_captions)])
    return {"PYTHONPATH": pythonpath}


def _resolve_dopsd_source_root(project_dir: str | Path) -> Path:
    source_root = Path(project_dir) / DOPSD_SOURCE_ROOT / "src"
    if source_root.exists():
        return source_root
    source_root = Path(__file__).resolve().parents[2] / DOPSD_SOURCE_ROOT / "src"
    if not source_root.exists():
        raise CommandBuildError(f"D-OPSD source tree not found: {source_root}")
    return source_root


def _module_to_script_path(module_name: str, source_root: str = "musubi-tuner") -> str:
    if module_name.endswith(".py") or "/" in module_name or "\\" in module_name:
        return module_name
    if not module_name.startswith("musubi_tuner."):
        return module_name
    relative = module_name.replace(".", "/") + ".py"
    return str(Path(source_root) / "src" / relative)
