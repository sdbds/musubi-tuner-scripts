"""Root PowerShell script coverage classification for the GUI."""

from __future__ import annotations


NATIVE_GUI = (
    "2.1、hy_cache_latent_and_text_encoder.ps1",
    "2.2、wan_cache_latent_and_text_encoder.ps1",
    "2.3、framepack_cache_latent_and_text_encoder.ps1",
    "2.4、flux_kontext_cache_latent_and_text_encoder.ps1",
    "2.5、qwen_image_cache_latent_and_text_encoder.ps1",
    "2.6、hv_1_5_cache_latent_and_text_encoder.ps1",
    "2.7、long_cat_cache_latent_and_text_encoder.ps1",
    "2.8、zimage_cache_latent_and_text_encoder.ps1",
    "2.9、flux2_cache_latent_and_text_encoder.ps1",
    "3.1、hy_train_lora.ps1",
    "3.2、wan_train_lora.ps1",
    "3.3、framepack_train_lora.ps1",
    "3.4、flux_kontext_train_lora.ps1",
    "3.5、qwen_image_train_lora.ps1",
    "3.6、long_cat_train_lora.ps1",
    "3.7、hv_1_5_train_lora.ps1",
    "3.8、zimage_train_lora.ps1",
    "3.9、flux2_train_lora.ps1",
    "5.1、hy_generate.ps1",
    "5.2、wan_generate.ps1",
    "5.3、famepack_generate.ps1",
    "5.4、flux_kontext_generate.ps1",
    "5.5、qwen_image_generate.ps1",
    "5.6、hv_1_5_generate.ps1",
    "5.8、zimage_generate.ps1",
    "5.9、flux2_generate.ps1",
)


COMPATIBILITY_LAUNCHER = (
    "1.5.qwen_vl_captions.ps1",
    "1.6、GUI.ps1",
    "2、cache_latent_and_text_encoder.ps1",
    "3、train_db.ps1",
    "3、train_lora.ps1",
    "3.5.1、qwen_image_train_db.ps1",
    "3.8.1、zimage_train_db.ps1",
    "4、convert_lora.ps1",
    "4.1、lora_post_hoc_ema.py.ps1",
    "5、generate.ps1",
    "5.5、qwen_image_generate_cosplay.ps1",
    "5.7、long_cat_generate.ps1",
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
