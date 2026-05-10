"""Shared model catalog aligned with the external PowerShell scripts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


WAN_TASKS_BY_VERSION = {
    "1.3B": ["t2v-1.3B", "t2v-1.3B-FC"],
    "14B": ["t2v-14B", "i2v-14B", "t2i-14B", "flf2v-14B", "t2v-14B-FC", "i2v-14B-FC"],
    "A14B": ["t2v-A14B", "i2v-A14B"],
    "5B": ["ti2v-5B"],
}

COMMON_VIDEO_TASKS = ["t2v-14B", "t2v-1.3B", "i2v-14B", "t2i-14B", "t2v-1.3B-FC", "t2v-14B-FC", "i2v-14B-FC"]

SOAR_TRAIN_ARCH_MODES = {
    "FLUX.2": {"lora"},
    "Qwen Image": {"lora"},
    "Z-Image": {"lora", "finetune"},
}
DOPSD_TRAIN_ARCH_MODES = {
    "FLUX.2": {"lora"},
    "Z-Image": {"lora", "finetune"},
}
QWEN_SOAR_INCOMPATIBLE_VERSIONS = {"2509", "2511", "edit", "edit-2509", "edit-2511", "edit_plus", "layered"}
FLUX2_SOAR_CFG_INCOMPATIBLE_VERSIONS = {"dev", "klein-4b", "klein-9b"}
FLUX2_DOPSD_SUPPORTED_VERSIONS = {"klein-4b", "klein-9b"}


MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "FLUX.2": {
        "id": "flux2",
        "cache_module": "musubi_tuner.flux_2_cache_latents",
        "cache_te_module": "musubi_tuner.flux_2_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.flux_2_train_network",
        "generate_module": "musubi_tuner.flux_2_generate_image",
        "versions": ["dev", "klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b"],
        "defaults": {
            "cache": {"version": "klein-base-4b"},
            "train": {"version": "klein-base-4b"},
            "generate": {"version": "dev"},
        },
        "path_defaults": {
            "cache": {
                "common": {
                    "vae_path": "./ckpts/vae/flux2-vae.safetensors",
                },
                "versions": {
                    "dev": {"text_encoder_path": "./ckpts/text_encoder/mistral3_model.safetensors"},
                    "klein-4b": {"text_encoder_path": "./ckpts/text_encoder/qwen_3_VL_4b.safetensors"},
                    "klein-base-4b": {"text_encoder_path": "./ckpts/text_encoder/qwen_3_4b.safetensors"},
                    "klein-9b": {"text_encoder_path": "./ckpts/text_encoder/qwen_3_8b.safetensors"},
                    "klein-base-9b": {"text_encoder_path": "./ckpts/text_encoder/qwen_3_8b.safetensors"},
                },
            },
            "train": {
                "common": {
                    "vae_path": "./ckpts/vae/flux2-vae.safetensors",
                },
                "versions": {
                    "dev": {
                        "dit_path": "./ckpts/diffusion_models/flux2-dev.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/mistral3_model.safetensors",
                    },
                    "klein-4b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-4b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_VL_4b.safetensors",
                    },
                    "klein-base-4b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-base-4b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_4b.safetensors",
                    },
                    "klein-9b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-9b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_8b.safetensors",
                    },
                    "klein-base-9b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-base-9b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_8b.safetensors",
                    },
                },
            },
            "generate": {
                "common": {
                    "vae_path": "./ckpts/vae/flux2-vae.safetensors",
                },
                "versions": {
                    "dev": {
                        "dit_path": "./ckpts/diffusion_models/flux2-dev.safetensors",
                        "vae_path": "./ckpts/vae/ae.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/mistral3_model.safetensors",
                    },
                    "klein-4b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-4b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_VL_4b.safetensors",
                    },
                    "klein-base-4b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-base-4b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_4b.safetensors",
                    },
                    "klein-9b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-9b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_8b.safetensors",
                    },
                    "klein-base-9b": {
                        "dit_path": "./ckpts/diffusion_models/flux-2-klein-base-9b.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_8b.safetensors",
                    },
                },
            },
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": True,
        "supports_fp8_scaled": True,
        "requires_vae": True,
        "default_timestep_sampling": "flux2_shift",
        "default_weighting_scheme": "none",
        "default_guidance_scale": 4.0,
        "is_video": False,
        "icon": "🎨",
        "color": "#6366f1",
        "pages": {
            "cache": {
                "supports_task_selector": False,
                "required_paths": ["dit", "vae", "text_encoder"],
                "flags": ["fp8_text_encoder", "dopsd"],
            },
            "train": {
                "supports_task_selector": False,
                "required_paths": ["dit", "vae", "text_encoder"],
                "flags": ["fp8_text_encoder", "soar", "dopsd"],
            },
            "generate": {
                "supports_task_selector": False,
                "required_paths": ["dit", "vae", "text_encoder"],
                "flags": ["fp8_text_encoder", "no_resize_control"],
            },
        },
    },
    "HiDream O1": {
        "id": "hidream_o1",
        "cache_module": "musubi_tuner.hidream_o1_cache_pixel",
        "cache_te_module": "musubi_tuner.hidream_o1_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.hidream_o1_train_network",
        "generate_module": "musubi_tuner.hidream_o1_generate_image",
        "versions": ["full", "dev"],
        "defaults": {
            "cache": {"version": "full"},
            "train": {"version": "full"},
            "generate": {"version": "full"},
        },
        "path_defaults": {
            "cache": {
                "versions": {
                    "full": {"text_encoder_path": "./ckpts/hidream-o1-image"},
                    "dev": {"text_encoder_path": "./ckpts/hidream-o1-image-dev"},
                },
            },
            "train": {
                "versions": {
                    "full": {
                        "dit_path": "./ckpts/hidream-o1-image",
                        "text_encoder_path": "./ckpts/hidream-o1-image",
                    },
                    "dev": {
                        "dit_path": "./ckpts/hidream-o1-image-dev",
                        "text_encoder_path": "./ckpts/hidream-o1-image-dev",
                    },
                },
            },
            "generate": {
                "versions": {
                    "full": {
                        "dit_path": "./ckpts/hidream-o1-image",
                        "text_encoder_path": "./ckpts/hidream-o1-image",
                    },
                    "dev": {
                        "dit_path": "./ckpts/hidream-o1-image-dev",
                        "text_encoder_path": "./ckpts/hidream-o1-image-dev",
                    },
                },
            },
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": True,
        "supports_fp8_scaled": False,
        "requires_vae": False,
        "default_timestep_sampling": "sigma",
        "default_weighting_scheme": "none",
        "default_guidance_scale": 5.0,
        "is_video": False,
        "icon": "O1",
        "color": "#0f766e",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["text_encoder"], "flags": ["fp8_te"]},
            "train": {"supports_task_selector": False, "required_paths": ["dit", "text_encoder"], "flags": []},
            "generate": {"supports_task_selector": False, "required_paths": ["dit", "text_encoder"], "flags": ["keep_original_aspect"]},
        },
    },
    "Wan2.1": {
        "id": "wan",
        "cache_module": "musubi_tuner.wan_cache_latents",
        "cache_te_module": "musubi_tuner.wan_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.wan_train_network",
        "generate_module": "musubi_tuner.wan_generate_video",
        "versions": ["1.3B", "14B", "A14B", "5B"],
        "defaults": {
            "cache": {"version": "14B"},
            "train": {"version": "A14B", "task": "t2v-A14B"},
            "generate": {"version": "14B", "task": "t2v-14B"},
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": False,
        "supports_fp8_scaled": True,
        "requires_vae": True,
        "requires_clip": True,
        "default_timestep_sampling": "shift",
        "default_weighting_scheme": None,
        "default_guidance_scale": 5.0,
        "is_video": True,
        "icon": "🎬",
        "color": "#ec4899",
        "pages": {
            "cache": {
                "supports_task_selector": False,
                "required_paths": ["dit", "t5", "clip"],
                "flags": ["vae_cache_cpu", "i2v", "one_frame", "one_frame_no_2x", "one_frame_no_4x", "fp8_t5"],
            },
            "train": {
                "supports_task_selector": True,
                "required_paths": ["dit", "t5", "clip"],
                "flags": ["vae_cache_cpu", "one_frame", "fp8_t5"],
                "tasks_by_version": WAN_TASKS_BY_VERSION,
            },
            "generate": {
                "supports_task_selector": True,
                "required_paths": ["dit", "t5", "clip"],
                "flags": ["vae_cache_cpu", "fp8_t5", "offload_inactive_dit", "enable_megacache"],
                "tasks_by_version": WAN_TASKS_BY_VERSION,
            },
        },
    },
    "HunyuanVideo": {
        "id": "hy",
        "cache_module": "musubi_tuner.cache_latents",
        "cache_te_module": "musubi_tuner.cache_text_encoder_outputs",
        "train_module": "musubi_tuner.hv_train_network",
        "generate_module": "musubi_tuner.hv_generate_video",
        "versions": ["default"],
        "defaults": {
            "cache": {"version": "default"},
            "train": {"version": "default", "task": "t2v-14B"},
            "generate": {"version": "default", "task": "t2v-14B"},
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": False,
        "supports_fp8_scaled": False,
        "requires_vae": False,
        "default_timestep_sampling": "shift",
        "default_weighting_scheme": None,
        "default_guidance_scale": 1.0,
        "is_video": True,
        "icon": "🎭",
        "color": "#8b5cf6",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["dit", "te1", "te2"], "flags": ["vae_tiling", "fp8_llm"]},
            "train": {"supports_task_selector": True, "required_paths": ["dit", "te1", "te2"], "flags": ["fp8_llm"], "tasks": COMMON_VIDEO_TASKS},
            "generate": {"supports_task_selector": True, "required_paths": ["dit", "te1", "te2"], "flags": ["fp8_llm"], "tasks": COMMON_VIDEO_TASKS},
        },
    },
    "FramePack": {
        "id": "framepack",
        "cache_module": "musubi_tuner.fpack_cache_latents",
        "cache_te_module": "musubi_tuner.fpack_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.fpack_train_network",
        "generate_module": "musubi_tuner.fpack_generate_video",
        "versions": ["default"],
        "defaults": {
            "cache": {"version": "default"},
            "train": {"version": "default", "task": "t2v-14B"},
            "generate": {"version": "default", "task": "t2v-14B"},
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": False,
        "supports_fp8_scaled": True,
        "requires_vae": True,
        "requires_image_encoder": True,
        "default_timestep_sampling": "shift",
        "default_weighting_scheme": None,
        "default_guidance_scale": 1.0,
        "is_video": True,
        "icon": "🎪",
        "color": "#06b6d4",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["dit", "vae", "te1", "te2", "image_encoder"], "flags": ["fp8_llm", "f1_mode", "one_frame", "one_frame_no_2x", "one_frame_no_4x"]},
            "train": {"supports_task_selector": True, "required_paths": ["dit", "vae", "te1", "te2", "image_encoder"], "flags": ["fp8_llm", "f1_mode"], "tasks": COMMON_VIDEO_TASKS},
            "generate": {"supports_task_selector": True, "required_paths": ["dit", "vae", "te1", "te2", "image_encoder"], "flags": ["fp8_llm", "f1_mode", "bulk_decode"], "tasks": COMMON_VIDEO_TASKS},
        },
    },
    "Long-CAT": {
        "id": "longcat",
        "cache_module": "musubi_tuner.longcat_cache_latents",
        "cache_te_module": "musubi_tuner.longcat_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.longcat_train_network",
        "generate_module": None,
        "versions": ["default"],
        "defaults": {
            "cache": {"version": "default"},
            "train": {"version": "default", "task": "t2v-14B"},
            "generate": {"version": "default", "task": "t2v-14B"},
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": False,
        "supports_fp8_scaled": True,
        "requires_vae": False,
        "default_timestep_sampling": "shift",
        "default_weighting_scheme": None,
        "default_guidance_scale": 1.0,
        "is_video": True,
        "icon": "🎫",
        "color": "#f59e0b",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["vae", "text_encoder"], "flags": ["i2v", "fp8_t5"]},
            "train": {"supports_task_selector": True, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["fp8_t5", "vae_cache_cpu", "longcat_i2v", "disable_numpy_memmap"], "tasks": COMMON_VIDEO_TASKS},
            "generate": {"supports_task_selector": True, "native_supported": False, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["fp8_t5", "longcat_i2v", "disable_numpy_memmap"], "tasks": COMMON_VIDEO_TASKS},
        },
    },
    "Z-Image": {
        "id": "zimage",
        "cache_module": "musubi_tuner.zimage_cache_latents",
        "cache_te_module": "musubi_tuner.zimage_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.zimage_train_network",
        "finetune_module": "musubi_tuner.zimage_train",
        "generate_module": "musubi_tuner.zimage_generate_image",
        "versions": ["base", "turbo"],
        "defaults": {
            "cache": {"version": "base"},
            "train": {"version": "base", "train_mode": "lora"},
            "generate": {"version": "base"},
        },
        "path_defaults": {
            "cache": {
                "common": {
                    "vae_path": "./ckpts/vae/ae.safetensors",
                    "dopsd_teacher_text_encoder_path": "./ckpts/text_encoder/qwen_3_VL_4b.safetensors",
                },
                "versions": {
                    "base": {"text_encoder_path": "./ckpts/text_encoder/qwen_3_4b.safetensors"},
                    "turbo": {"text_encoder_path": "./ckpts/text_encoder/qwen_3_VL_4b.safetensors"},
                },
            },
            "train": {
                "common": {
                    "vae_path": "./ckpts/vae/ae.safetensors",
                },
                "versions": {
                    "base": {
                        "dit_path": "./ckpts/diffusion_models/z_image_bf16.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_4b.safetensors",
                    },
                    "turbo": {
                        "dit_path": "./ckpts/diffusion_models/z_image_turbo_bf16.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_VL_4b.safetensors",
                    },
                },
            },
            "generate": {
                "common": {
                    "vae_path": "./ckpts/vae/ae.safetensors",
                },
                "versions": {
                    "base": {
                        "dit_path": "./ckpts/diffusion_models/z_image_bf16.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_4b.safetensors",
                    },
                    "turbo": {
                        "dit_path": "./ckpts/diffusion_models/z_image_turbo_bf16.safetensors",
                        "text_encoder_path": "./ckpts/text_encoder/qwen_3_VL_4b.safetensors",
                    },
                },
            },
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": True,
        "supports_fp8_scaled": True,
        "requires_vae": True,
        "default_timestep_sampling": "shift",
        "default_weighting_scheme": None,
        "default_guidance_scale": 1.0,
        "is_video": False,
        "icon": "🎯",
        "color": "#10b981",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["fp8_llm", "text_encoder_cpu", "i2v", "dopsd"]},
            "train": {"supports_task_selector": False, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["fp8_llm", "text_encoder_cpu", "soar", "dopsd"]},
            "generate": {"supports_task_selector": False, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["fp8_llm", "text_encoder_cpu", "use_32bit_attention"]},
        },
    },
    "HV 1.5": {
        "id": "hv15",
        "cache_module": "musubi_tuner.hv_1_5_cache_latents",
        "cache_te_module": "musubi_tuner.hv_1_5_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.hv_1_5_train_network",
        "generate_module": "musubi_tuner.hv_1_5_generate_video",
        "versions": ["default"],
        "defaults": {
            "cache": {"version": "default"},
            "train": {"version": "default", "task": "t2v"},
            "generate": {"version": "default", "task": "t2v"},
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": False,
        "supports_fp8_scaled": False,
        "requires_vae": True,
        "default_timestep_sampling": "uniform",
        "default_weighting_scheme": None,
        "default_guidance_scale": 1.0,
        "is_video": True,
        "icon": "🔮",
        "color": "#ef4444",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["dit", "vae", "text_encoder", "byt5"], "flags": ["text_encoder_cpu", "vae_enable_patch_conv"]},
            "train": {"supports_task_selector": True, "required_paths": ["dit", "vae", "text_encoder", "byt5"], "flags": ["text_encoder_cpu", "vae_enable_patch_conv"], "tasks": ["t2v", "i2v"]},
            "generate": {"supports_task_selector": True, "required_paths": ["dit", "vae", "text_encoder", "byt5", "image_encoder"], "flags": ["text_encoder_cpu", "vae_enable_patch_conv"], "tasks": ["t2v", "i2v"]},
        },
    },
    "Qwen Image": {
        "id": "qwen_image",
        "cache_module": "musubi_tuner.qwen_image_cache_latents",
        "cache_te_module": "musubi_tuner.qwen_image_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.qwen_image_train_network",
        "finetune_module": "musubi_tuner.qwen_image_train",
        "generate_module": "musubi_tuner.qwen_image_generate_image",
        "versions": ["default", "original", "2509", "2511"],
        "defaults": {
            "cache": {"version": "default"},
            "train": {"version": "default", "train_mode": "lora"},
            "generate": {"version": "default"},
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": True,
        "supports_fp8_scaled": True,
        "requires_vae": True,
        "default_timestep_sampling": "shift",
        "default_weighting_scheme": None,
        "default_guidance_scale": 1.0,
        "is_video": False,
        "supports_edit_version": True,
        "supports_edit_modes": True,
        "icon": "🧩",
        "color": "#14b8a6",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["fp8_vl", "text_encoder_cpu", "edit_mode", "edit_plus"]},
            "train": {"supports_task_selector": False, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["fp8_vl", "text_encoder_cpu", "edit_mode", "edit_plus", "soar"]},
            "generate": {"supports_task_selector": False, "required_paths": ["dit", "vae", "text_encoder"], "flags": ["text_encoder_cpu", "edit_mode", "edit_plus"]},
        },
    },
    "FLUX Kontext": {
        "id": "flux_kontext",
        "cache_module": "musubi_tuner.flux_kontext_cache_latents",
        "cache_te_module": "musubi_tuner.flux_kontext_cache_text_encoder_outputs",
        "train_module": "musubi_tuner.flux_kontext_train_network",
        "generate_module": "musubi_tuner.flux_kontext_generate_image",
        "versions": ["dev"],
        "defaults": {
            "cache": {"version": "dev"},
            "train": {"version": "dev"},
            "generate": {"version": "dev"},
        },
        "supports_text_encoder": True,
        "supports_fp8_text_encoder": True,
        "supports_fp8_scaled": True,
        "requires_vae": True,
        "default_timestep_sampling": "shift",
        "default_weighting_scheme": None,
        "default_guidance_scale": 1.0,
        "is_video": False,
        "icon": "🎨",
        "color": "#6366f1",
        "pages": {
            "cache": {"supports_task_selector": False, "required_paths": ["vae", "te1", "te2"], "flags": ["fp8_t5"]},
            "train": {"supports_task_selector": False, "required_paths": ["dit", "vae", "te1", "te2"], "flags": ["fp8_t5"]},
            "generate": {"supports_task_selector": False, "required_paths": ["dit", "vae", "te1", "te2"], "flags": ["no_resize_control"]},
        },
    },
}


def get_architecture(name: str) -> Optional[Dict[str, Any]]:
    architecture = MODEL_CATALOG.get(name)
    return deepcopy(architecture) if architecture else None


def get_all_architectures() -> Dict[str, Dict[str, Any]]:
    return deepcopy(MODEL_CATALOG)


def get_architecture_names() -> List[str]:
    return list(MODEL_CATALOG.keys())


def get_page_config(name: str, page_key: str) -> Dict[str, Any]:
    architecture = MODEL_CATALOG.get(name, {})
    return deepcopy(architecture.get("pages", {}).get(page_key, {}))


def get_versions_for_page(name: str, page_key: str) -> List[str]:
    architecture = MODEL_CATALOG.get(name, {})
    page_config = architecture.get("pages", {}).get(page_key, {})
    versions = page_config.get("versions") or architecture.get("versions", [])
    return list(versions)


def get_default_version(name: str, page_key: str) -> Optional[str]:
    architecture = MODEL_CATALOG.get(name, {})
    default_value = architecture.get("defaults", {}).get(page_key, {}).get("version")
    if default_value:
        return default_value
    versions = get_versions_for_page(name, page_key)
    return versions[0] if versions else None


def get_tasks_for_page(name: str, page_key: str, version: Optional[str] = None) -> List[str]:
    architecture = MODEL_CATALOG.get(name, {})
    page_config = architecture.get("pages", {}).get(page_key, {})
    if not page_config.get("supports_task_selector", False):
        return []

    tasks_by_version = page_config.get("tasks_by_version")
    if tasks_by_version:
        resolved_version = version or get_default_version(name, page_key)
        return list(tasks_by_version.get(resolved_version, []))

    return list(page_config.get("tasks", []))


def get_default_task(name: str, page_key: str, version: Optional[str] = None) -> Optional[str]:
    architecture = MODEL_CATALOG.get(name, {})
    default_value = architecture.get("defaults", {}).get(page_key, {}).get("task")
    if default_value:
        return default_value
    tasks = get_tasks_for_page(name, page_key, version=version)
    return tasks[0] if tasks else None


def supports_task_selector(name: str, page_key: str) -> bool:
    architecture = MODEL_CATALOG.get(name, {})
    return bool(architecture.get("pages", {}).get(page_key, {}).get("supports_task_selector", False))


def get_train_modes(name: str) -> Dict[str, str]:
    architecture = MODEL_CATALOG.get(name, {})
    modes = {"lora": "LoRA"}
    if architecture.get("finetune_module"):
        modes["finetune"] = "Fine-tune"
    return modes


def get_default_train_mode(name: str) -> str:
    architecture = MODEL_CATALOG.get(name, {})
    default_value = architecture.get("defaults", {}).get("train", {}).get("train_mode")
    modes = get_train_modes(name)
    return str(default_value) if default_value in modes else next(iter(modes), "lora")


def supports_soar_training(name: str, train_mode: str | None = "lora", version: str | None = None) -> bool:
    if str(train_mode or "lora") not in SOAR_TRAIN_ARCH_MODES.get(name, set()):
        return False
    if name == "Qwen Image" and str(version or "").strip().lower() in QWEN_SOAR_INCOMPATIBLE_VERSIONS:
        return False
    return True


def supports_soar_cfg_rollout(name: str, train_mode: str | None = "lora", version: str | None = None) -> bool:
    resolved_mode = str(train_mode or "lora")
    resolved_version = str(version or get_default_version(name, "train") or "").strip().lower()
    if not supports_soar_training(name, resolved_mode, version=resolved_version):
        return False
    if name == "FLUX.2":
        return resolved_version not in FLUX2_SOAR_CFG_INCOMPATIBLE_VERSIONS
    if name == "Qwen Image":
        return resolved_mode == "lora" and resolved_version not in QWEN_SOAR_INCOMPATIBLE_VERSIONS
    if name == "Z-Image":
        return resolved_mode == "lora"
    return False


def supports_dopsd_training(name: str, train_mode: str | None = "lora", version: str | None = None) -> bool:
    if str(train_mode or "lora") not in DOPSD_TRAIN_ARCH_MODES.get(name, set()):
        return False
    return supports_dopsd_cache(name, version=version, page_key="train")


def supports_dopsd_cache(name: str, version: str | None = None, page_key: str | None = None) -> bool:
    del page_key
    resolved_version = str(version or get_default_version(name, "cache") or "").strip().lower()
    if name == "FLUX.2":
        return resolved_version in FLUX2_DOPSD_SUPPORTED_VERSIONS
    if name == "Z-Image":
        return True
    return False


def get_path_defaults(name: str, page_key: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Return default model paths for an architecture/page/version selection."""
    architecture = MODEL_CATALOG.get(name, {})
    page_defaults = architecture.get("path_defaults", {}).get(page_key, {})
    resolved_version = version or get_default_version(name, page_key)

    defaults: Dict[str, Any] = {}
    defaults.update(page_defaults.get("common", {}))
    defaults.update(page_defaults.get("versions", {}).get(resolved_version, {}))
    return deepcopy(defaults)
