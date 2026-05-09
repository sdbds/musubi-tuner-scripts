"""Root PowerShell script coverage classification for the GUI."""

from __future__ import annotations


NATIVE_GUI = (
    "2.1hy_cache_latent_and_text_encoder.ps1",
    "2.2wan_cache_latent_and_text_encoder.ps1",
    "2.3framepack_cache_latent_and_text_encoder.ps1",
    "2.4flux_kontext_cache_latent_and_text_encoder.ps1",
    "2.5、qwen_image_cache_latent_and_text_encoder.ps1",
    "2.6hv_1_5_cache_latent_and_text_encoder.ps1",
    "2.7long_cat_cache_latent_and_text_encoder.ps1",
    "2.8zimage_cache_latent_and_text_encoder.ps1",
    "2.9flux2_cache_latent_and_text_encoder.ps1",
    "3.1hy_train_lora.ps1",
    "3.2wan_train_lora.ps1",
    "3.3framepack_train_lora.ps1",
    "3.4flux_kontext_train_lora.ps1",
    "3.5qwen_image_train_lora.ps1",
    "3.6long_cat_train_lora.ps1",
    "3.7hv_1_5_train_lora.ps1",
    "3.8zimage_train_lora.ps1",
    "3.9、flux2_train_lora.ps1",
    "5.1hy_generate.ps1",
    "5.2wan_generate.ps1",
    "5.3famepack_generate.ps1",
    "5.4flux_kontext_generate.ps1",
    "5.5qwen_image_generate.ps1",
    "5.6hv_1_5_generate.ps1",
    "5.8zimage_generate.ps1",
    "5.9flux2_generate.ps1",
)


COMPATIBILITY_LAUNCHER = (
    "1.5.qwen_vl_captions.ps1",
    "1.6.GUI.ps1",
    "2cache_latent_and_text_encoder.ps1",
    "3train_db.ps1",
    "3train_lora.ps1",
    "3.5.1qwen_image_train_db.ps1",
    "3.8.1zimage_train_db.ps1",
    "4convert_lora.ps1",
    "4.1lora_post_hoc_ema.py.ps1",
    "5generate.ps1",
    "5.5qwen_image_generate_cosplay.ps1",
    "5.7long_cat_generate.ps1",
    "tensorboard.ps1",
)


UNSUPPORTED = ()


IGNORED = (
    "1.install-uv-qinglong.ps1",
)


def all_classified_scripts() -> set[str]:
    return set(NATIVE_GUI) | set(COMPATIBILITY_LAUNCHER) | set(UNSUPPORTED)


def ignored_scripts() -> set[str]:
    return set(IGNORED)
