import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils.command_builder import (  # noqa: E402
    CommandBuildError,
    build_cache_jobs,
    build_generate_job,
    build_train_job,
    get_train_optimizer_template_args,
)


PROJECT_CONFIG = {
    "dataset": {
        "general": {"resolution": [512, 512]},
        "datasets": [
            {
                "image_directory": "images",
                "caption_extension": ".txt",
                "batch_size": 1,
            }
        ],
    },
    "interop": {"dataset_extra": {"root": {}, "general": {}, "datasets": [{}]}},
}


class TestCommandBuilder(unittest.TestCase):
    def test_flux2_cache_builds_latent_and_text_encoder_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/mistral.safetensors",
                "batch_size": 2,
                "fp8_text_encoder": True,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0].script_key, "musubi_tuner.flux_2_cache_latents")
            self.assertEqual(jobs[1].script_key, "musubi_tuner.flux_2_cache_text_encoder_outputs")
            self.assertIn("--model_version=klein-base-4b", jobs[0].args)
            self.assertIn("--vae=ckpts/ae.safetensors", jobs[0].args)
            self.assertIn("--batch_size=2", jobs[0].args)
            self.assertIn("--text_encoder=ckpts/mistral.safetensors", jobs[1].args)
            self.assertIn("--fp8_text_encoder", jobs[1].args)
            self.assertTrue((Path(tmp) / "dataset_config.toml").exists())

    def test_cache_omits_console_args_without_debug_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "debug_mode": "",
                "console_width": "$Host.UI.RawUI.WindowSize.Width",
                "console_back": "black",
                "console_num_images": 16,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertNotIn("--debug_mode=", jobs[0].args)
            self.assertNotIn("--console_width=$Host.UI.RawUI.WindowSize.Width", jobs[0].args)
            self.assertFalse(any(arg.startswith("--console_width=") for arg in jobs[0].args))
            self.assertFalse(any(arg.startswith("--console_back=") for arg in jobs[0].args))
            self.assertFalse(any(arg.startswith("--console_num_images=") for arg in jobs[0].args))

    def test_cache_omits_zero_num_workers_to_use_script_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "num_workers": 0,
                "te_num_workers": 0,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertFalse(any(arg.startswith("--num_workers=") for arg in jobs[0].args))
            self.assertFalse(any(arg.startswith("--num_workers=") for arg in jobs[1].args))

    def test_cache_adds_positive_num_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "num_workers": 2,
                "te_num_workers": 3,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertIn("--num_workers=2", jobs[0].args)
            self.assertIn("--num_workers=3", jobs[1].args)

    def test_cache_only_adds_console_args_for_console_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "debug_mode": "image",
                "console_width": 120,
                "console_back": "black",
                "console_num_images": 8,
            }

            image_jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertIn("--debug_mode=image", image_jobs[0].args)
            self.assertFalse(any(arg.startswith("--console_width=") for arg in image_jobs[0].args))
            self.assertFalse(any(arg.startswith("--console_back=") for arg in image_jobs[0].args))
            self.assertFalse(any(arg.startswith("--console_num_images=") for arg in image_jobs[0].args))

            state["debug_mode"] = "console"
            console_jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertIn("--debug_mode=console", console_jobs[0].args)
            self.assertIn("--console_width=120", console_jobs[0].args)
            self.assertIn("--console_back=black", console_jobs[0].args)
            self.assertIn("--console_num_images=8", console_jobs[0].args)

    def test_wan_train_uses_accelerate_and_network_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Wan2.1",
                "task": "i2v-A14B",
                "dit_path": "ckpts/wan.safetensors",
                "vae_path": "ckpts/wan_vae.safetensors",
                "t5_path": "ckpts/t5.pth",
                "clip_path": "ckpts/clip.pth",
                "learning_rate": "1e-4",
                "mixed_precision": "fp16",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "wan_train_network.py")))
            self.assertIn("--task=i2v-A14B", job.args)
            self.assertIn("--network_module=networks.lora_wan", job.args)
            self.assertIn("--learning_rate=1e-4", job.args)
            self.assertEqual(job.runner_kwargs["use_accelerate"], True)
            self.assertEqual(job.runner_kwargs["mixed_precision"], "fp16")
            self.assertEqual(job.runner_kwargs["num_cpu_threads_per_process"], 8)

    def test_flux2_train_maps_attention_and_ignores_disabled_lycoris_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "learning_rate": "1e-4",
                "attn_mode": "flash",
                "optimizer_type": "AdamW_adv",
                "enable_lycoris": False,
                "lycoris_algo": "lokr",
                "lycoris_preset": "attn-mlp",
                "lycoris_conv_dim": 0,
                "lycoris_conv_alpha": 0,
                "lycoris_factor": 8,
                "lycoris_dropout": 0,
                "lycoris_block_size": 4,
                "lycoris_rescaled": 1,
                "lycoris_dora_wd": True,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--flash_attn", job.args)
            self.assertNotIn("--attn_mode=flash", job.args)
            self.assertIn("--optimizer_type=adv_optm.AdamW_adv", job.args)
            self.assertIn("--optimizer_args", job.args)
            self.assertIn("grams_moment=True", job.args)
            self.assertNotIn("--optimizer_type=AdamW_adv", job.args)
            self.assertNotIn("--lycoris_algo=lokr", job.args)
            self.assertNotIn("--lycoris_preset=attn-mlp", job.args)
            self.assertNotIn("--dora_wd", job.args)
            self.assertNotIn("--network_args", job.args)

    def test_train_optimizer_mappings_follow_script_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "learning_rate": "1e-4",
                "d_coef": "0.5",
                "d0": "1e-3",
                "network_dim": 32,
            }

            prodigy_job = build_train_job({**base_state, "optimizer_type": "Prodigy_adv"}, tmp, PROJECT_CONFIG)
            self.assertIn("--learning_rate=1", prodigy_job.args)
            self.assertIn("--optimizer_type=adv_optm.Prodigy_adv", prodigy_job.args)
            self.assertIn("grams_moment=True", prodigy_job.args)
            self.assertIn("d_coef=0.5", prodigy_job.args)
            self.assertIn("d0=1e-3", prodigy_job.args)

            fira_job = build_train_job({**base_state, "optimizer_type": "fira"}, tmp, PROJECT_CONFIG)
            self.assertIn("--learning_rate=1e-4", fira_job.args)
            self.assertIn("--optimizer_type=pytorch_optimizer.Fira", fira_job.args)
            self.assertIn("rank=32", fira_job.args)
            self.assertIn("projection_type='std'", fira_job.args)

    def test_train_optimizer_extra_args_can_override_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "optimizer_type": "AdamW_adv",
                "optimizer_extra_args": "grams_moment=False\nweight_decay=0.01",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--optimizer_type=adv_optm.AdamW_adv", job.args)
            self.assertIn("--optimizer_args", job.args)
            self.assertIn("grams_moment=False", job.args)
            self.assertIn("weight_decay=0.01", job.args)
            self.assertNotIn("grams_moment=True", job.args)

    def test_train_optimizer_template_args_are_available_for_ui(self):
        template = get_train_optimizer_template_args("Fira", {"network_dim": 64})

        self.assertEqual(
            template,
            ["weight_decay=0.01", "rank=64", "update_proj_gap=50", "scale=1", "projection_type='std'"],
        )

    def test_train_sample_zero_steps_falls_back_to_epoch_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "enable_sample": True,
                "sample_at_first": True,
                "sample_every_n_epochs": 1,
                "sample_every_n_steps": 0,
                "sample_prompts": "prompts.txt",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--sample_at_first", job.args)
            self.assertIn("--sample_every_n_epochs=1", job.args)
            self.assertNotIn("--sample_every_n_steps=0", job.args)
            self.assertFalse(any(arg.startswith("--sample_every_n_steps=") for arg in job.args))
            self.assertIn("--sample_prompts=prompts.txt", job.args)

    def test_train_disabled_sampling_omits_all_sample_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "enable_sample": False,
                "sample_at_first": True,
                "sample_every_n_epochs": 1,
                "sample_every_n_steps": 100,
                "sample_prompts": "prompts.txt",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertNotIn("--sample_at_first", job.args)
            self.assertFalse(any(arg.startswith("--sample_every_n_epochs=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--sample_every_n_steps=") for arg in job.args))
            self.assertNotIn("--sample_prompts=prompts.txt", job.args)

    def test_train_save_steps_zero_falls_back_to_epoch_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "save_every_n_epochs": "2",
                "save_every_n_steps": "0",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--save_every_n_epochs=2", job.args)
            self.assertFalse(any(arg.startswith("--save_every_n_steps=") for arg in job.args))

    def test_train_zero_epoch_frequencies_are_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "enable_sample": True,
                "save_every_n_epochs": "0",
                "save_every_n_steps": "0",
                "sample_every_n_epochs": "0",
                "sample_every_n_steps": "0",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertFalse(any(arg.startswith("--save_every_n_epochs=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--save_every_n_steps=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--sample_every_n_epochs=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--sample_every_n_steps=") for arg in job.args))

    def test_train_filters_unsupported_arch_specific_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "HunyuanVideo",
                "dit_path": "ckpts/hv.pt",
                "te1_path": "ckpts/llm.safetensors",
                "te2_path": "ckpts/clip.safetensors",
                "fp8_scaled": True,
                "vae_cache_cpu": True,
                "vae_tiling": True,
                "vae_chunk_size": 32,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertNotIn("--fp8_scaled", job.args)
            self.assertNotIn("--vae_cache_cpu", job.args)
            self.assertIn("--vae_tiling", job.args)
            self.assertIn("--vae_chunk_size=32", job.args)

    def test_train_multi_gpu_gates_ddp_and_launch_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "multi_gpu": True,
                "ddp_timeout": 120,
                "ddp_gradient_as_bucket_view": True,
                "ddp_static_graph": True,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--ddp_timeout=120", job.args)
            self.assertIn("--ddp_gradient_as_bucket_view", job.args)
            self.assertIn("--ddp_static_graph", job.args)
            self.assertIn("--multi_gpu", job.runner_kwargs["accelerate_args"])
            self.assertIn("--rdzv_backend=c10d", job.runner_kwargs["accelerate_args"])

    def test_cache_filters_unsupported_arch_specific_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX Kontext",
                "vae_path": "ckpts/ae.safetensors",
                "te1_path": "ckpts/t5.safetensors",
                "te2_path": "ckpts/clip.safetensors",
                "vae_chunk_size": 32,
                "vae_spatial_tile_sample_min_size": 256,
                "vae_tiling": True,
                "vae_cache_cpu": True,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertFalse(any(arg.startswith("--vae_chunk_size=") for arg in jobs[0].args))
            self.assertFalse(any(arg.startswith("--vae_spatial_tile_sample_min_size=") for arg in jobs[0].args))
            self.assertNotIn("--vae_tiling", jobs[0].args)
            self.assertNotIn("--vae_cache_cpu", jobs[0].args)

    def test_generate_filters_unsupported_arch_specific_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            flux_kontext = build_generate_job(
                {
                    "arch": "FLUX Kontext",
                    "dit_path": "ckpts/flux.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "te1_path": "ckpts/t5.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                    "video_size": "1024 1024",
                    "prompt": "test",
                    "guidance_scale": 5.0,
                },
                tmp,
            )
            self.assertFalse(any(arg.startswith("--guidance_scale=") for arg in flux_kontext.args))

            hunyuan = build_generate_job(
                {
                    "arch": "HunyuanVideo",
                    "dit_path": "ckpts/hv.pt",
                    "te1_path": "ckpts/llm.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                    "video_size": "512 512",
                    "prompt": "test",
                    "fp8_scaled": True,
                },
                tmp,
            )
            self.assertNotIn("--fp8_scaled", hunyuan.args)

    def test_train_lycoris_uses_network_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "enable_lycoris": True,
                "lycoris_algo": "lokr",
                "lycoris_preset": "attn-mlp",
                "lycoris_factor": 8,
                "lycoris_dora_wd": True,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--network_module=lycoris.kohya", job.args)
            self.assertIn("--network_args", job.args)
            self.assertIn("algo=lokr", job.args)
            self.assertIn("preset=attn-mlp", job.args)
            self.assertIn("factor=8", job.args)
            self.assertIn("dora_wd=True", job.args)

    def test_qwen_finetune_uses_finetune_module_and_filters_lora_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Qwen Image",
                "train_mode": "finetune",
                "version": "2509",
                "dit_path": "ckpts/qwen_dit.safetensors",
                "vae_path": "ckpts/qwen_vae.safetensors",
                "text_encoder_path": "ckpts/qwen_vl.safetensors",
                "network_weights": "ckpts/old_lora.safetensors",
                "base_weights": "ckpts/merge_lora.safetensors",
                "base_weights_multiplier": "0.5",
                "network_dim": 32,
                "network_alpha": 16,
                "network_dropout": 0.1,
                "scale_weight_norms": 1,
                "dim_from_weights": True,
                "enable_lycoris": True,
                "lycoris_algo": "lokr",
                "enable_lora_plus": True,
                "full_bf16": True,
                "fused_backward_pass": True,
                "mem_eff_save": True,
                "fp8_base": True,
                "fp8_scaled": True,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "qwen_image_train.py")))
            self.assertIn("--full_bf16", job.args)
            self.assertIn("--fused_backward_pass", job.args)
            self.assertIn("--mem_eff_save", job.args)
            forbidden_finetune_flags = {
                "--network_module",
                "--network_weights",
                "--network_dim",
                "--network_alpha",
                "--network_dropout",
                "--network_args",
                "--dim_from_weights",
                "--scale_weight_norms",
                "--base_weights",
                "--base_weights_multiplier",
                "--fp8_base",
                "--fp8_scaled",
            }
            self.assertFalse(any(arg.split("=", 1)[0] in forbidden_finetune_flags for arg in job.args))

    def test_zimage_finetune_adds_zimage_specific_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "train_mode": "finetune",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "block_swap_optimizer_patch_params": True,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "zimage_train.py")))
            self.assertIn("--block_swap_optimizer_patch_params", job.args)
            self.assertNotIn("--network_module=networks.lora_zimage", job.args)

    def test_unsupported_architecture_rejects_finetune_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "train_mode": "finetune",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
            }

            with self.assertRaises(CommandBuildError):
                build_train_job(state, tmp, PROJECT_CONFIG)

    def test_qwen_generate_uses_local_image_entry_and_image_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Qwen Image",
                "version": "2509",
                "dit_path": "ckpts/qwen_dit.safetensors",
                "vae_path": "ckpts/qwen_vae.safetensors",
                "text_encoder_vl_path": "ckpts/qwen_vl.safetensors",
                "save_path": str(Path(tmp) / "samples"),
                "video_size": "768 1024",
                "prompt": "a product photo",
                "text_encoder_cpu": True,
            }

            job = build_generate_job(state, tmp)

            self.assertEqual(job.script_key, "musubi_tuner.qwen_image_generate_image")
            self.assertIn("--model_version=edit-2509", job.args)
            self.assertIn("--text_encoder=ckpts/qwen_vl.safetensors", job.args)
            self.assertIn("--image_size", job.args)
            self.assertNotIn("--video_size", job.args)
            self.assertIn("--prompt=a product photo", job.args)
            self.assertIn("--text_encoder_cpu", job.args)

    def test_flux2_generate_filters_video_only_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "dev",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/mistral.safetensors",
                "video_size": "1024 1024",
                "prompt": "a studio portrait",
                "fps": 24,
                "video_length": 1,
                "sample_solver": "vanilla",
                "split_attn": True,
                "img_in_txt_in_offloading": True,
                "cfg_apply_ratio": 0.3,
                "compile_backend": "inductor",
                "fp8": True,
            }

            job = build_generate_job(state, tmp)

            self.assertIn("--fp8", job.args)
            self.assertNotIn("--fps=24", job.args)
            self.assertNotIn("--video_length=1", job.args)
            self.assertNotIn("--sample_solver=vanilla", job.args)
            self.assertNotIn("--split_attn", job.args)
            self.assertNotIn("--img_in_txt_in_offloading", job.args)
            self.assertNotIn("--cfg_apply_ratio=0.3", job.args)
            self.assertNotIn("--compile_backend=inductor", job.args)

    def test_generate_keeps_single_lora_path_with_spaces_as_one_arg(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "prompt": "a studio portrait",
                "video_size": "1024 1024",
                "lora_weight": r"E:\Models\my lora.safetensors",
            }

            job = build_generate_job(state, tmp)

            self.assertIn(r"--lora_weight=E:\Models\my lora.safetensors", job.args)
            self.assertNotIn("--lora_weight", job.args)

    def test_generate_splits_multiple_lora_paths_only_on_newlines_or_semicolons(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "prompt": "a studio portrait",
                "video_size": "1024 1024",
                "lora_weight": "E:/Models/my lora.safetensors\nE:/Models/other lora.safetensors",
            }

            job = build_generate_job(state, tmp)

            lora_index = job.args.index("--lora_weight")
            self.assertEqual(
                job.args[lora_index + 1:lora_index + 3],
                ["E:/Models/my lora.safetensors", "E:/Models/other lora.safetensors"],
            )

    def test_generate_requires_prompt_file_or_latent(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/mistral.safetensors",
                "video_size": "1024 1024",
            }

            with self.assertRaises(CommandBuildError):
                build_generate_job(state, tmp)

    def test_longcat_generate_requires_compatibility_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {"arch": "Long-CAT"}

            with self.assertRaises(CommandBuildError):
                build_generate_job(state, tmp)


if __name__ == "__main__":
    unittest.main()
