"""Built-in preset catalog derived from the external PowerShell scripts."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_STOP_MARKER = "DO NOT MODIFY CONTENTS BELOW"
ASSIGNMENT_RE = re.compile(r"^\$(\w+)\s*=\s*(.+?)\s*(?:#.*)?$")

PRESET_SOURCES: Dict[str, Dict[str, Dict[str, str]]] = {
    "cache": {
        "flux2": {"arch": "FLUX.2", "script": "2.9、flux2_cache_latent_and_text_encoder.ps1"},
        "wan2_1": {"arch": "Wan2.1", "script": "2.2、wan_cache_latent_and_text_encoder.ps1"},
        "hunyuan_video": {"arch": "HunyuanVideo", "script": "2.1、hy_cache_latent_and_text_encoder.ps1"},
        "framepack": {"arch": "FramePack", "script": "2.3、framepack_cache_latent_and_text_encoder.ps1"},
        "flux_kontext": {"arch": "FLUX Kontext", "script": "2.4、flux_kontext_cache_latent_and_text_encoder.ps1"},
        "qwen_image": {"arch": "Qwen Image", "script": "2.5、qwen_image_cache_latent_and_text_encoder.ps1"},
        "hv_1_5": {"arch": "HV 1.5", "script": "2.6、hv_1_5_cache_latent_and_text_encoder.ps1"},
        "long_cat": {"arch": "Long-CAT", "script": "2.7、long_cat_cache_latent_and_text_encoder.ps1"},
        "zimage": {"arch": "Z-Image", "script": "2.8、zimage_cache_latent_and_text_encoder.ps1"},
    },
    "train": {
        "flux2": {"arch": "FLUX.2", "script": "3.9、flux2_train_lora.ps1"},
        "wan2_1": {"arch": "Wan2.1", "script": "3.2、wan_train_lora.ps1"},
        "hunyuan_video": {"arch": "HunyuanVideo", "script": "3.1、hy_train_lora.ps1"},
        "framepack": {"arch": "FramePack", "script": "3.3、framepack_train_lora.ps1"},
        "flux_kontext": {"arch": "FLUX Kontext", "script": "3.4、flux_kontext_train_lora.ps1"},
        "qwen_image": {"arch": "Qwen Image", "script": "3.5、qwen_image_train_lora.ps1"},
        "hv_1_5": {"arch": "HV 1.5", "script": "3.7、hv_1_5_train_lora.ps1"},
        "long_cat": {"arch": "Long-CAT", "script": "3.6、long_cat_train_lora.ps1"},
        "zimage": {"arch": "Z-Image", "script": "3.8、zimage_train_lora.ps1"},
    },
    "generate": {
        "flux2": {"arch": "FLUX.2", "script": "5.9、flux2_generate.ps1"},
        "wan2_1": {"arch": "Wan2.1", "script": "5.2、wan_generate.ps1"},
        "hunyuan_video": {"arch": "HunyuanVideo", "script": "5.1、hy_generate.ps1"},
        "framepack": {"arch": "FramePack", "script": "5.3、famepack_generate.ps1"},
        "flux_kontext": {"arch": "FLUX Kontext", "script": "5.4、flux_kontext_generate.ps1"},
        "qwen_image": {"arch": "Qwen Image", "script": "5.5、qwen_image_generate.ps1"},
        "hv_1_5": {"arch": "HV 1.5", "script": "5.6、hv_1_5_generate.ps1"},
        "long_cat": {"arch": "Long-CAT", "script": "5.7、long_cat_generate.ps1"},
        "zimage": {"arch": "Z-Image", "script": "5.8、zimage_generate.ps1"},
    },
}

KEY_MAPS: Dict[str, Dict[str, str]] = {
    "cache": {
        "dataset_config": "toml_path",
        "model_version": "version",
        "vae": "vae_path",
        "vae_dtype": "vae_dtype",
        "device": "device",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "skip_existing": "skip_existing",
        "debug_mode": "debug_mode",
        "console_width": "console_width",
        "console_back": "console_back",
        "console_num_images": "console_num_images",
        "text_encoder_batch_size": "te_batch_size",
        "text_encoder_device": "te_device",
        "text_encoder_num_workers": "te_num_workers",
        "text_encoder_skip_existing": "te_skip_existing",
        "vae_chunk_size": "vae_chunk_size",
        "vae_tiling": "vae_tiling",
        "vae_spatial_tile_sample_min_size": "vae_spatial_tile_sample_min_size",
        "vae_cache_cpu": "vae_cache_cpu",
        "fp8_text_encoder": "fp8_text_encoder",
        "fp8_t5": "fp8_t5",
        "fp8_llm": "fp8_llm",
        "f1_mode": "f1_mode",
        "one_frame": "one_frame",
        "one_frame_no_2x": "one_frame_no_2x",
        "one_frame_no_4x": "one_frame_no_4x",
        "i2v": "i2v",
        "text_encoder_cpu": "text_encoder_cpu",
        "vae_sample_size": "vae_sample_size",
        "vae_enable_patch_conv": "vae_enable_patch_conv",
        "edit": "edit_mode",
        "edit_plus": "edit_plus",
        "edit_version": "edit_version",
    },
    "train": {
        "dataset_config": "dataset_config",
        "dit": "dit_path",
        "vae": "vae_path",
        "model_version": "version",
        "fp8_text_encoder": "fp8_text_encoder",
        "fp8_scaled": "fp8_scaled",
        "resume": "resume_path",
        "network_weights": "network_weights",
        "base_weights": "base_weights",
        "base_weights_multiplier": "base_weights_multiplier",
        "max_train_steps": "max_train_steps",
        "max_train_epochs": "max_train_epochs",
        "gradient_checkpointing": "gradient_checkpointing",
        "gradient_checkpointing_cpu_offload": "gradient_checkpointing_cpu_offload",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "guidance_scale": "guidance_scale",
        "seed": "seed",
        "timestep_sampling": "timestep_sampling",
        "discrete_flow_shift": "discrete_flow_shift",
        "sigmoid_scale": "sigmoid_scale",
        "weighting_scheme": "weighting_scheme",
        "logit_mean": "logit_mean",
        "logit_std": "logit_std",
        "mode_scale": "mode_scale",
        "soar": "soar",
        "soar_lambda_aux": "soar_lambda_aux",
        "soar_trajectory_length": "soar_trajectory_length",
        "soar_num_sampling_steps": "soar_num_sampling_steps",
        "min_timestep": "min_timestep",
        "max_timestep": "max_timestep",
        "show_timesteps": "show_timesteps",
        "lr": "learning_rate",
        "lr_scheduler": "lr_scheduler",
        "lr_warmup_steps": "lr_warmup_steps",
        "lr_decay_steps": "lr_decay_steps",
        "lr_scheduler_num_cycles": "lr_scheduler_num_cycles",
        "lr_scheduler_power": "lr_scheduler_power",
        "lr_scheduler_timescale": "lr_scheduler_timescale",
        "lr_scheduler_min_lr_ratio": "lr_scheduler_min_lr_ratio",
        "network_dim": "network_dim",
        "network_alpha": "network_alpha",
        "network_dropout": "network_dropout",
        "dim_from_weights": "dim_from_weights",
        "scale_weight_norms": "scale_weight_norms",
        "attn_mode": "attn_mode",
        "split_attn": "split_attn",
        "mixed_precision": "mixed_precision",
        "full_bf16": "full_bf16",
        "compile": "compile",
        "compile_backend": "compile_backend",
        "compile_mode": "compile_mode",
        "compile_fullgraph": "compile_fullgraph",
        "compile_dynamic": "compile_dynamic",
        "compile_cache_size_limit": "compile_cache_size_limit",
        "cuda_allow_tf32": "cuda_allow_tf32",
        "cuda_cudnn_benchmark": "cuda_cudnn_benchmark",
        "vae_dtype": "vae_dtype",
        "fp8_base": "fp8_base",
        "max_data_loader_n_workers": "max_data_loader_n_workers",
        "persistent_data_loader_workers": "persistent_workers",
        "blocks_to_swap": "blocks_to_swap",
        "use_pinned_memory_for_block_swap": "use_pinned_memory",
        "optimizer_type": "optimizer_type",
        "max_grad_norm": "max_grad_norm",
        "d_coef": "d_coef",
        "d0": "d0",
        "wandb_api_key": "wandb_api_key",
        "output_name": "output_name",
        "save_every_n_epochs": "save_every_n_epochs",
        "save_every_n_steps": "save_every_n_steps",
        "save_last_n_epochs": "save_last_n_epochs",
        "save_last_n_steps": "save_last_n_steps",
        "save_state": "save_state",
        "save_state_on_train_end": "save_state_on_train_end",
        "save_last_n_epochs_state": "save_last_n_epochs_state",
        "save_last_n_steps_state": "save_last_n_steps_state",
        "enable_lycoris": "enable_lycoris",
        "conv_dim": "lycoris_conv_dim",
        "conv_alpha": "lycoris_conv_alpha",
        "algo": "lycoris_algo",
        "preset": "lycoris_preset",
        "dropout": "lycoris_dropout",
        "factor": "lycoris_factor",
        "decompose_both": "lycoris_decompose_both",
        "block_size": "lycoris_block_size",
        "use_tucker": "lycoris_use_tucker",
        "use_scalar": "lycoris_use_scalar",
        "train_norm": "lycoris_train_norm",
        "dora_wd": "lycoris_dora_wd",
        "full_matrix": "lycoris_full_matrix",
        "bypass_mode": "lycoris_bypass_mode",
        "rescaled": "lycoris_rescaled",
        "constrain": "lycoris_constrain",
        "enable_sample": "enable_sample",
        "sample_at_first": "sample_at_first",
        "sample_prompts": "sample_prompts",
        "sample_every_n_epochs": "sample_every_n_epochs",
        "sample_every_n_steps": "sample_every_n_steps",
        "training_comment": "training_comment",
        "metadata_title": "metadata_title",
        "metadata_author": "metadata_author",
        "metadata_description": "metadata_description",
        "metadata_license": "metadata_license",
        "metadata_tags": "metadata_tags",
        "async_upload": "async_upload",
        "huggingface_repo_id": "huggingface_repo_id",
        "huggingface_repo_type": "huggingface_repo_type",
        "huggingface_path_in_repo": "huggingface_path_in_repo",
        "huggingface_token": "huggingface_token",
        "huggingface_repo_visibility": "huggingface_repo_visibility",
        "save_state_to_huggingface": "save_state_to_huggingface",
        "resume_from_huggingface": "resume_from_huggingface",
        "multi_gpu": "multi_gpu",
        "ddp_timeout": "ddp_timeout",
        "ddp_gradient_as_bucket_view": "ddp_gradient_as_bucket_view",
        "ddp_static_graph": "ddp_static_graph",
        "fp8_t5": "fp8_t5",
        "vae_cache_cpu": "vae_cache_cpu",
        "vae_chunk_size": "vae_chunk_size",
        "vae_tiling": "vae_tiling",
        "vae_spatial_tile_sample_min_size": "vae_spatial_tile_sample_min_size",
        "vae_sample_size": "vae_sample_size",
        "vae_enable_patch_conv": "vae_enable_patch_conv",
        "text_encoder_cpu": "text_encoder_cpu",
        "fp8_vl": "fp8_vl",
        "edit": "edit_mode",
        "edit_plus": "edit_plus",
        "edit_version": "edit_version",
    },
    "generate": {
        "dit": "dit_path",
        "dit_high_noise": "dit_high_noise",
        "vae": "vae_path",
        "vae_dtype": "vae_dtype",
        "model_version": "version",
        "fp8_text_encoder": "fp8_text_encoder",
        "lora_weight": "lora_weight",
        "lora_multiplier": "lora_multiplier",
        "include_patterns": "include_patterns",
        "exclude_patterns": "exclude_patterns",
        "save_merged_model": "save_merged_model",
        "prompt": "prompt",
        "negative_prompt": "negative_prompt",
        "from_file": "from_file",
        "image_size": "video_size",
        "infer_steps": "infer_steps",
        "save_path": "save_path",
        "seed": "seed",
        "guidance_scale": "guidance_scale",
        "embedded_cfg_scale": "embedded_cfg_scale",
        "control_image_path": "control_image_path",
        "no_resize_control": "no_resize_control",
        "flow_shift": "flow_shift",
        "fp8": "fp8_base",
        "fp8_scaled": "fp8_scaled",
        "device": "device",
        "attn_mode": "attn_mode",
        "blocks_to_swap": "blocks_to_swap",
        "use_pinned_memory_for_block_swap": "use_pinned_memory_for_block_swap",
        "output_type": "output_type",
        "no_metadata": "no_metadata",
        "latent_path": "latent_path",
        "lycoris": "lycoris",
        "compile": "compile",
        "compile_backend": "compile_backend",
        "compile_mode": "compile_mode",
        "compile_fullgraph": "compile_fullgraph",
        "compile_dynamic": "compile_dynamic",
        "compile_cache_size_limit": "compile_cache_size_limit",
        "enable_tf32": "enable_tf32",
        "vae_chunk_size": "vae_chunk_size",
        "vae_spatial_tile_sample_min_size": "vae_spatial_tile_sample_min_size",
        "fp8_t5": "fp8_t5",
        "vae_cache_cpu": "vae_cache_cpu",
        "offload_inactive_dit": "offload_inactive_dit",
        "text_encoder_cpu": "text_encoder_cpu",
        "edit": "edit_mode",
        "edit_plus": "edit_plus",
        "edit_version": "edit_version",
        "vae_sample_size": "vae_sample_size",
        "vae_enable_patch_conv": "vae_enable_patch_conv",
    },
}

PATH_KEY_OVERRIDES: Dict[str, Dict[str, Dict[str, str]]] = {
    "cache": {
        "flux2": {"text_encoder": "text_encoder_path"},
        "wan2_1": {"t5": "t5_path", "clip": "clip_path", "image_encoder": "image_encoder_path", "text_encoder1": "te1_path", "text_encoder2": "te2_path"},
        "hunyuan_video": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "clip": "clip_path", "t5": "t5_path"},
        "framepack": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "flux_kontext": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "qwen_image": {"text_encoder": "text_encoder_path"},
        "hv_1_5": {"text_encoder": "text_encoder_path", "byt5": "byt5_path", "image_encoder": "image_encoder_path"},
        "long_cat": {"text_encoder": "text_encoder_path"},
        "zimage": {"text_encoder": "text_encoder_path", "image_encoder": "image_encoder_path"},
    },
    "train": {
        "flux2": {"text_encoder": "text_encoder_path"},
        "wan2_1": {"t5": "t5_path", "clip": "clip_path", "image_encoder": "image_encoder_path", "text_encoder1": "te1_path", "text_encoder2": "te2_path"},
        "hunyuan_video": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "clip": "clip_path", "t5": "t5_path"},
        "framepack": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "flux_kontext": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "qwen_image": {"text_encoder": "text_encoder_path", "image_encoder": "image_encoder_path", "t5": "t5_path", "clip": "clip_path", "text_encoder1": "te1_path", "text_encoder2": "te2_path"},
        "hv_1_5": {"text_encoder": "text_encoder_path", "byt5": "byt5_path", "image_encoder": "image_encoder_path"},
        "long_cat": {"text_encoder": "text_encoder_path", "image_encoder": "image_encoder_path", "t5": "t5_path", "clip": "clip_path", "text_encoder1": "te1_path", "text_encoder2": "te2_path"},
        "zimage": {"text_encoder": "text_encoder_path"},
    },
    "generate": {
        "flux2": {"text_encoder": "text_encoder_path"},
        "wan2_1": {"t5": "t5_path", "image_encoder": "image_encoder_path", "text_encoder1": "te1_path", "text_encoder2": "te2_path"},
        "hunyuan_video": {"text_encoder1": "te1_path", "text_encoder2": "te2_path"},
        "framepack": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "flux_kontext": {"text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "qwen_image": {"text_encoder": "text_encoder_vl_path", "text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "hv_1_5": {"text_encoder": "text_encoder_path", "byt5": "byt5_path", "image_encoder": "image_encoder_path"},
        "long_cat": {"text_encoder": "text_encoder_vl_path", "text_encoder1": "te1_path", "text_encoder2": "te2_path", "image_encoder": "image_encoder_path"},
        "zimage": {"text_encoder": "text_encoder_path"},
    },
}

_BUILTIN_CACHE: Dict[str, Dict[str, Dict[str, Any]]] = {}


def _parse_value(raw_value: str) -> Any:
    value = raw_value.strip()
    lower = value.lower()

    if lower == "$true":
        return True
    if lower == "$false":
        return False
    if lower == "$null":
        return None

    if value.startswith('"""') and value.endswith('"""') and len(value) >= 6:
        return value[3:-3]

    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)

    return value


def _parse_script_variables(script_name: str) -> Dict[str, Any]:
    script_path = REPO_ROOT / script_name
    parsed: Dict[str, Any] = {}

    for line in script_path.read_text(encoding="utf-8").splitlines():
        if SCRIPT_STOP_MARKER in line:
            break
        match = ASSIGNMENT_RE.match(line.strip())
        if not match:
            continue
        key, raw_value = match.groups()
        parsed[key] = _parse_value(raw_value)

    return parsed


def _translate_to_preset(scope: str, slug: str, arch_name: str, parsed: Dict[str, Any], script_name: str) -> Dict[str, Any]:
    preset: Dict[str, Any] = {
        "arch": arch_name,
        "_source_script": script_name,
    }

    key_map = KEY_MAPS[scope]
    path_overrides = PATH_KEY_OVERRIDES.get(scope, {}).get(slug, {})

    for key, value in parsed.items():
        if value is None:
            continue

        target_key = path_overrides.get(key) or key_map.get(key)
        if not target_key:
            continue

        preset[target_key] = value

    return preset


def _build_scope_presets(scope: str) -> Dict[str, Dict[str, Any]]:
    built: Dict[str, Dict[str, Any]] = {}

    for slug, entry in PRESET_SOURCES[scope].items():
        parsed = _parse_script_variables(entry["script"])
        built[slug] = _translate_to_preset(scope, slug, entry["arch"], parsed, entry["script"])

    return built


def get_builtin_presets(scope: str) -> Dict[str, Dict[str, Any]]:
    if scope not in _BUILTIN_CACHE:
        _BUILTIN_CACHE[scope] = _build_scope_presets(scope)
    return copy.deepcopy(_BUILTIN_CACHE[scope])


def get_builtin_preset(scope: str, name: str) -> Optional[Dict[str, Any]]:
    return get_builtin_presets(scope).get(name)


def list_builtin_preset_entries(scope: str) -> list[Dict[str, str]]:
    presets = get_builtin_presets(scope)
    return [
        {
            "name": name,
            "label": preset.get("arch", name),
            "source": "builtin",
        }
        for name, preset in sorted(presets.items())
    ]
