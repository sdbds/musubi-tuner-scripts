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

    def test_lens_cache_uses_text_encoder_and_omits_removed_text_metadata_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Lens",
                "vae_path": "ckpts/lens/vae/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors",
                "vae_dtype": "float32",
                "text_encoder_dtype": "bfloat16",
                "disable_numpy_memmap": True,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertEqual(jobs[0].script_key, "musubi_tuner.lens_cache_latents")
            self.assertEqual(jobs[1].script_key, "musubi_tuner.lens_cache_text_encoder_outputs")
            self.assertIn("--vae=ckpts/lens/vae/flux2-vae.safetensors", jobs[0].args)
            self.assertIn("--vae_dtype=float32", jobs[0].args)
            self.assertIn("--text_encoder=ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors", jobs[1].args)
            self.assertFalse(any(arg.startswith("--text_encoder_config=") for arg in jobs[1].args))
            self.assertFalse(any(arg.startswith("--tokenizer=") for arg in jobs[1].args))
            self.assertIn("--text_encoder_dtype=bfloat16", jobs[1].args)
            self.assertIn("--disable_numpy_memmap", jobs[1].args)

    def test_ideogram4_cache_uses_bf16_qwen3vl_text_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Ideogram-4",
                "vae_path": "ckpts/vae/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/text_encoder/qwen3vl_8b_bf16.safetensors",
                "vae_dtype": "bfloat16",
                "text_encoder_dtype": "float16",
                "text_cache_dtype": "bf16",
                "disable_numpy_memmap": True,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertEqual(jobs[0].script_key, "musubi_tuner.ideogram4_cache_latents")
            self.assertEqual(jobs[1].script_key, "musubi_tuner.ideogram4_cache_text_encoder_outputs")
            self.assertIn("--vae=ckpts/vae/flux2-vae.safetensors", jobs[0].args)
            self.assertIn("--vae_dtype=bfloat16", jobs[0].args)
            self.assertIn("--text_encoder=ckpts/text_encoder/qwen3vl_8b_bf16.safetensors", jobs[1].args)
            self.assertIn("--text_cache_dtype=bf16", jobs[1].args)
            self.assertIn("--disable_numpy_memmap", jobs[1].args)
            self.assertFalse(any(arg.startswith("--text_encoder_dtype=") for arg in jobs[1].args))
            self.assertFalse(any("qwen3vl_8b_fp8_scaled" in arg for job in jobs for arg in job.args))

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

    def test_hidream_o1_cache_uses_optional_dit_without_text_encoder_or_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "HiDream O1",
                "version": "full",
                "dit_path": "ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors",
                "text_encoder_path": "ckpts/hidream-qwen3vl",
                "vae_path": "ckpts/stale-vae.safetensors",
                "vae_dtype": "float32",
                "batch_size": 1,
                "te_batch_size": 16,
                "fp8_te": True,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertEqual(jobs[0].name, "HiDream O1 Cache Pixels")
            self.assertEqual(jobs[0].script_key, "musubi_tuner.hidream_o1_cache_pixel")
            self.assertEqual(jobs[1].script_key, "musubi_tuner.hidream_o1_cache_text_encoder_outputs")
            self.assertIn("--batch_size=1", jobs[0].args)
            self.assertIn("--model_type=full", jobs[1].args)
            self.assertIn("--dit=ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors", jobs[1].args)
            self.assertFalse(any(arg.startswith("--text_encoder=") for arg in jobs[1].args))
            self.assertIn("--fp8_te", jobs[1].args)
            self.assertIn("--batch_size=16", jobs[1].args)
            self.assertFalse(any(arg.startswith("--vae=") for arg in jobs[0].args + jobs[1].args))
            self.assertFalse(any(arg.startswith("--vae_dtype=") for arg in jobs[0].args + jobs[1].args))
            self.assertFalse(any(arg.startswith("--model_version=") for arg in jobs[0].args + jobs[1].args))

    def test_hidream_o1_cache_requires_dit_for_fp8_text_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "HiDream O1",
                "version": "dev",
                "fp8_te": True,
            }

            with self.assertRaises(CommandBuildError):
                build_cache_jobs(state, tmp, PROJECT_CONFIG)

    def test_zimage_soar_cache_passes_i2v_and_image_encoder_to_latent_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "image_encoder_path": "ckpts/siglip2.safetensors",
                "i2v": True,
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertEqual(jobs[0].script_key, "musubi_tuner.zimage_cache_latents")
            self.assertIn("--i2v", jobs[0].args)
            self.assertIn("--image_encoder=ckpts/siglip2.safetensors", jobs[0].args)
            self.assertIn("--vae=ckpts/ae.safetensors", jobs[0].args)
            self.assertEqual(jobs[1].script_key, "musubi_tuner.zimage_cache_text_encoder_outputs")
            self.assertIn("--text_encoder=ckpts/qwen3.safetensors", jobs[1].args)
            self.assertNotIn("--i2v", jobs[1].args)
            self.assertFalse(any(arg.startswith("--image_encoder=") for arg in jobs[1].args))

    def test_zimage_soar_cache_requires_image_encoder_when_i2v_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "i2v": True,
            }

            with self.assertRaises(CommandBuildError):
                build_cache_jobs(state, tmp, PROJECT_CONFIG)

    def test_zimage_control_dataset_enables_soar_cache_when_image_encoder_is_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "image_encoder_path": "ckpts/siglip2.safetensors",
                "i2v": False,
            }
            project_config = {
                **PROJECT_CONFIG,
                "dataset": {
                    **PROJECT_CONFIG["dataset"],
                    "datasets": [
                        {
                            "image_directory": "images",
                            "control_directory": "controls",
                            "caption_extension": ".txt",
                            "batch_size": 1,
                        }
                    ],
                },
            }

            jobs = build_cache_jobs(state, tmp, project_config)

            self.assertIn("--i2v", jobs[0].args)
            self.assertIn("--image_encoder=ckpts/siglip2.safetensors", jobs[0].args)

    def test_zimage_cache_passes_dopsd_teacher_args_to_text_encoder_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd_cache_teacher_outputs": True,
                "dopsd_teacher_text_encoder_path": "ckpts/qwen3-vl",
                "dopsd_teacher_dtype": "bfloat16",
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertNotIn("--dopsd_cache_teacher_outputs", jobs[0].args)
            self.assertIn("--dopsd_cache_teacher_outputs", jobs[1].args)
            self.assertIn("--dopsd_teacher_text_encoder=ckpts/qwen3-vl", jobs[1].args)
            self.assertIn("--dopsd_teacher_dtype=bfloat16", jobs[1].args)
            self.assertFalse(any(arg.startswith("--dopsd_teacher_llm_reweight_source") for arg in jobs[1].args))
            self.assertNotIn("--dopsd_teacher_processor=ckpts/qwen3-vl-processor", jobs[1].args)
            self.assertFalse(any(arg.startswith("--dopsd_teacher_embed_key") for arg in jobs[1].args))
            self.assertIn("musubi-tuner", jobs[1].runner_kwargs["env_vars"]["PYTHONPATH"])
            self.assertNotIn("musubi-tuner-dopsd-zimage", jobs[1].runner_kwargs["env_vars"]["PYTHONPATH"])

    def test_flux2_cache_uses_identity_edit_dopsd_teacher_without_vlm_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-4b",
                "vae_path": "ckpts/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd_cache_teacher_outputs": True,
                "dopsd_teacher_text_encoder_path": "ckpts/qwen3-vl",
                "dopsd_teacher_dtype": "float16",
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertIn("--model_version=klein-4b", jobs[1].args)
            self.assertIn("--dopsd_cache_teacher_outputs", jobs[1].args)
            self.assertNotIn("--dopsd_teacher_text_encoder=ckpts/qwen3-vl", jobs[1].args)
            self.assertNotIn("--dopsd_teacher_dtype=float16", jobs[1].args)
            self.assertFalse(any(arg.startswith("--dopsd_teacher_llm_reweight_source") for arg in jobs[1].args))
            self.assertIn("musubi-tuner", jobs[1].runner_kwargs["env_vars"]["PYTHONPATH"])
            self.assertNotIn("musubi-tuner-dopsd-zimage", jobs[1].runner_kwargs["env_vars"]["PYTHONPATH"])

    def test_qwen_image_cache_rejects_dopsd_teacher_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Qwen Image",
                "version": "original",
                "vae_path": "ckpts/qwen_vae.safetensors",
                "text_encoder_path": "ckpts/qwen2_5_vl.safetensors",
                "dopsd_cache_teacher_outputs": True,
                "dopsd_teacher_text_encoder_path": "ckpts/unused-qwen3-vl",
            }

            with self.assertRaises(CommandBuildError):
                build_cache_jobs(state, tmp, PROJECT_CONFIG)

    def test_zimage_dopsd_teacher_cache_requires_teacher_text_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd_cache_teacher_outputs": True,
            }

            with self.assertRaises(CommandBuildError):
                build_cache_jobs(state, tmp, PROJECT_CONFIG)

    def test_zimage_dopsd_teacher_cache_uses_student_text_encoder_for_reweight_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd_cache_teacher_outputs": True,
                "dopsd_teacher_text_encoder_path": "ckpts/qwen3-vl",
            }

            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)

            self.assertIn("--dopsd_teacher_text_encoder=ckpts/qwen3-vl", jobs[1].args)
            self.assertFalse(any(arg.startswith("--dopsd_teacher_llm_reweight_source") for arg in jobs[1].args))

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
            self.assertIn("--mixed_precision=fp16", job.args)
            self.assertEqual(job.runner_kwargs["use_accelerate"], True)
            self.assertEqual(job.runner_kwargs["mixed_precision"], "fp16")
            self.assertEqual(job.runner_kwargs["num_cpu_threads_per_process"], 8)

    def test_lens_train_uses_sdpa_and_supports_block_swap(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Lens",
                "dit_path": "ckpts/lens/diffusion_models/lens_bf16.safetensors",
                "vae_path": "ckpts/lens/vae/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors",
                "text_encoder_dtype": "bfloat16",
                "learning_rate": "1e-4",
                "mixed_precision": "bf16",
                "attn_mode": "flash",
                "split_attn": True,
                "blocks_to_swap": 8,
                "use_pinned_memory": True,
                "fp8_base": True,
                "fp8_scaled": True,
                "optimizer_type": "AdamW_adv",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "lens_train_network.py")))
            self.assertIn("--dit=ckpts/lens/diffusion_models/lens_bf16.safetensors", job.args)
            self.assertIn("--vae=ckpts/lens/vae/flux2-vae.safetensors", job.args)
            self.assertIn("--text_encoder=ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors", job.args)
            self.assertFalse(any(arg.startswith("--text_encoder_config=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--tokenizer=") for arg in job.args))
            self.assertIn("--network_module=networks.lora_lens", job.args)
            self.assertIn("--text_encoder_dtype=bfloat16", job.args)
            self.assertIn("--sdpa", job.args)
            self.assertNotIn("--flash_attn", job.args)
            self.assertNotIn("--split_attn", job.args)
            self.assertIn("--fp8_base", job.args)
            self.assertIn("--fp8_scaled", job.args)
            self.assertIn("--blocks_to_swap=8", job.args)
            self.assertIn("--use_pinned_memory_for_block_swap", job.args)
            self.assertIn("--optimizer_type=adv_optm.AdamW_adv", job.args)
            self.assertIn("betas=.95,.98", job.args)

    def test_ideogram4_train_uses_unconditional_dit_and_specialized_lora_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Ideogram-4",
                "dit_path": "ckpts/diffusion_models/ideogram4_fp8_scaled.safetensors",
                "unconditional_dit_path": "ckpts/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
                "vae_path": "ckpts/vae/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/text_encoder/qwen3vl_8b_bf16.safetensors",
                "learning_rate": "1e-4",
                "mixed_precision": "bf16",
                "timestep_sampling": "sigma",
                "sampler_preset": "V4_DEFAULT_20",
                "ideogram4_timestep_mu": 0.0,
                "ideogram4_timestep_std": 1.0,
                "disable_numpy_memmap": True,
                "fp8_scaled": True,
                "split_attn": True,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "ideogram4_train_network.py")))
            self.assertIn("--dit=ckpts/diffusion_models/ideogram4_fp8_scaled.safetensors", job.args)
            self.assertIn(
                "--unconditional_dit=ckpts/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
                job.args,
            )
            self.assertIn("--vae=ckpts/vae/flux2-vae.safetensors", job.args)
            self.assertIn("--text_encoder=ckpts/text_encoder/qwen3vl_8b_bf16.safetensors", job.args)
            self.assertIn("--network_module=networks.lora_ideogram4", job.args)
            self.assertIn("--sampler_preset=V4_DEFAULT_20", job.args)
            self.assertIn("--ideogram4_timestep_mu=0.0", job.args)
            self.assertIn("--ideogram4_timestep_std=1.0", job.args)
            self.assertIn("--disable_numpy_memmap", job.args)
            self.assertNotIn("--timestep_sampling=sigma", job.args)
            self.assertNotIn("--fp8_scaled", job.args)
            self.assertNotIn("--split_attn", job.args)
            self.assertFalse(any("qwen3vl_8b_fp8_scaled" in arg for arg in job.args))

    def test_lens_train_requires_fp8_base_and_scaled_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_state = {
                "arch": "Lens",
                "dit_path": "ckpts/lens/diffusion_models/lens_bf16.safetensors",
                "vae_path": "ckpts/lens/vae/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors",
                "learning_rate": "1e-4",
                "mixed_precision": "bf16",
                "optimizer_type": "AdamW_adv",
            }

            with self.assertRaises(CommandBuildError):
                build_train_job({**base_state, "fp8_base": True, "fp8_scaled": False}, tmp, PROJECT_CONFIG)
            with self.assertRaises(CommandBuildError):
                build_train_job({**base_state, "fp8_base": False, "fp8_scaled": True}, tmp, PROJECT_CONFIG)

    def test_train_passes_mixed_precision_to_every_script(self):
        cases = [
            (
                "FLUX.2",
                {
                    "version": "klein-base-4b",
                    "dit_path": "ckpts/flux2.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "text_encoder_path": "ckpts/qwen3.safetensors",
                },
            ),
            (
                "Wan2.1",
                {
                    "task": "t2v-A14B",
                    "dit_path": "ckpts/wan.safetensors",
                    "t5_path": "ckpts/t5.pth",
                    "clip_path": "ckpts/clip.pth",
                },
            ),
            (
                "HunyuanVideo",
                {
                    "task": "t2v-14B",
                    "dit_path": "ckpts/hv.safetensors",
                    "te1_path": "ckpts/llava.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                },
            ),
            (
                "FramePack",
                {
                    "dit_path": "ckpts/framepack.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "te1_path": "ckpts/llava.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                    "image_encoder_path": "ckpts/siglip.safetensors",
                },
            ),
            (
                "Long-CAT",
                {
                    "task": "t2v-14B",
                    "dit_path": "ckpts/longcat.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "text_encoder_path": "ckpts/t5.safetensors",
                },
            ),
            (
                "Z-Image",
                {
                    "version": "base",
                    "dit_path": "ckpts/zimage.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "text_encoder_path": "ckpts/qwen3.safetensors",
                },
            ),
            (
                "HV 1.5",
                {
                    "task": "t2v",
                    "dit_path": "ckpts/hv15.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "text_encoder_path": "ckpts/qwen25.safetensors",
                    "byt5_path": "ckpts/byt5.safetensors",
                },
            ),
            (
                "Qwen Image",
                {
                    "version": "default",
                    "dit_path": "ckpts/qwen.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "text_encoder_path": "ckpts/qwen_vl.safetensors",
                },
            ),
            (
                "FLUX Kontext",
                {
                    "version": "dev",
                    "dit_path": "ckpts/kontext.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "te1_path": "ckpts/t5.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                },
            ),
            (
                "HiDream O1",
                {
                    "version": "full",
                    "dit_path": "ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors",
                    "text_encoder_path": "ckpts/hidream-qwen3vl",
                },
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            for arch, arch_state in cases:
                with self.subTest(arch=arch):
                    state = {"arch": arch, "mixed_precision": "fp16", **arch_state}

                    job = build_train_job(state, tmp, PROJECT_CONFIG)

                    self.assertIn("--mixed_precision=fp16", job.args)
                    self.assertEqual(job.runner_kwargs["mixed_precision"], "fp16")

    def test_hunyuan_train_maps_dit_dtype_and_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "HunyuanVideo",
                    "task": "i2v-14B",
                    "dit_path": "ckpts/hv.safetensors",
                    "te1_path": "ckpts/llava.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                    "dit_dtype": "bfloat16",
                    "dit_in_channels": 32,
                },
                tmp,
                PROJECT_CONFIG,
            )

            self.assertIn("--dit_dtype=bfloat16", job.args)
            self.assertIn("--dit_in_channels=32", job.args)

    def test_hv15_train_maps_dit_dtype_without_unsupported_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "HV 1.5",
                    "task": "t2v",
                    "dit_path": "ckpts/hv15.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "text_encoder_path": "ckpts/qwen25.safetensors",
                    "byt5_path": "ckpts/byt5.safetensors",
                    "dit_dtype": "float16",
                    "dit_in_channels": 32,
                },
                tmp,
                PROJECT_CONFIG,
            )

            self.assertIn("--dit_dtype=float16", job.args)
            self.assertNotIn("--dit_in_channels=32", job.args)

    def test_train_default_output_dir_matches_script_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--output_dir=./output_dir", job.args)
            self.assertFalse(any(arg.endswith("/output") or arg.endswith("\\output") for arg in job.args))

    def test_train_defaults_to_local_tensorboard_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--logging_dir=./logs", job.args)
            self.assertIn("--log_with=tensorboard", job.args)
            self.assertNotIn("env_vars", job.runner_kwargs)

    def test_train_uses_custom_logging_dir_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "logging_dir": "./logs/custom",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--logging_dir=./logs/custom", job.args)
            self.assertIn("--log_with=tensorboard", job.args)

    def test_train_wandb_key_uses_environment_not_command_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "output_name": "wandb-run",
                "wandb_api_key": "  test-wandb-key  ",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--logging_dir=./logs", job.args)
            self.assertIn("--log_with=wandb", job.args)
            self.assertNotIn("--log_with=tensorboard", job.args)
            self.assertIn("--log_tracker_name=wandb-run", job.args)
            self.assertFalse(any(arg.startswith("--wandb_api_key") for arg in job.args))
            self.assertNotIn("test-wandb-key", "\n".join(job.args))
            self.assertEqual(job.runner_kwargs["env_vars"]["WANDB_API_KEY"], "test-wandb-key")

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
            self.assertIn("betas=.95,.98", job.args)
            self.assertNotIn("grams_moment=True", job.args)
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

            for optimizer_type, resolved_type in {
                "LoRARite": "pytorch_optimizer.LoRARite",
                "lorarite": "pytorch_optimizer.LoRARite",
                "FlashAdamW": "pytorch_optimizer.FlashAdamW",
                "flashadamw": "pytorch_optimizer.FlashAdamW",
                "DualAdam": "pytorch_optimizer.DualAdam",
                "dualadam": "pytorch_optimizer.DualAdam",
                "ROSE": "pytorch_optimizer.ROSE",
                "rose": "pytorch_optimizer.ROSE",
            }.items():
                job = build_train_job({**base_state, "optimizer_type": optimizer_type}, tmp, PROJECT_CONFIG)
                self.assertIn(f"--optimizer_type={resolved_type}", job.args)

    def test_train_optimizer_extra_args_can_override_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "optimizer_type": "AdamW_adv",
                "optimizer_extra_args": "weight_decay=0.01\nbetas=.9,.99\ndecouple=True\nuse_bias_correction=True\nd_coef=0.5\nd0=1e-3",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertIn("--optimizer_type=adv_optm.AdamW_adv", job.args)
            self.assertIn("--optimizer_args", job.args)
            self.assertIn("weight_decay=0.01", job.args)
            self.assertIn("betas=.9,.99", job.args)
            self.assertIn("decouple=True", job.args)
            self.assertIn("use_bias_correction=True", job.args)
            self.assertIn("d_coef=0.5", job.args)
            self.assertIn("d0=1e-3", job.args)
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

    def test_hidream_o1_train_uses_model_type_and_lora_module_without_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "HiDream O1",
                "version": "dev",
                "dit_path": "ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors",
                "text_encoder_path": "ckpts/hidream-qwen3vl",
                "vae_path": "ckpts/stale-vae.safetensors",
                "vae_dtype": "float32",
                "learning_rate": "1e-4",
                "mixed_precision": "bf16",
                "attn_mode": "flash",
                "blocks_to_swap": 8,
                "use_pinned_memory": True,
                "fp8_base": True,
                "optimizer_type": "AdamW_adv",
                "timestep_sampling": "uniform",
                "noise_scale_start": 7.5,
                "noise_scale_end": 6.5,
                "noise_clip_std": 2.5,
                "dino_loss_weight": 0.02,
                "dino_loss_backend": "vit",
                "dino_loss_model_type": "small",
                "dino_loss_layer": -4,
                "dino_loss_feature_mode": "patch",
                "dino_loss_resize": 224,
                "dino_loss_every_n_steps": 2,
                "dino_loss_use_gram": True,
                "dino_loss_no_norm": True,
                "fp8_scaled": True,
                "enable_sample": True,
                "sample_at_first": 1,
                "sample_prompts": "toml/qinglong_hidream_o1.txt",
                "sample_every_n_epochs": 1,
                "sample_every_n_steps": 0,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "hidream_o1_train_network.py")))
            self.assertIn("--model_type=dev", job.args)
            self.assertIn("--dit=ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors", job.args)
            self.assertIn("--timestep_sampling=uniform", job.args)
            self.assertIn("--noise_scale_start=7.5", job.args)
            self.assertIn("--noise_scale_end=6.5", job.args)
            self.assertIn("--noise_clip_std=2.5", job.args)
            self.assertIn("--dino_loss_weight=0.02", job.args)
            self.assertIn("--dino_loss_backend=vit", job.args)
            self.assertIn("--dino_loss_model_type=small", job.args)
            self.assertIn("--dino_loss_layer=-4", job.args)
            self.assertIn("--dino_loss_feature_mode=patch", job.args)
            self.assertIn("--dino_loss_resize=224", job.args)
            self.assertIn("--dino_loss_every_n_steps=2", job.args)
            self.assertIn("--dino_loss_use_gram", job.args)
            self.assertIn("--dino_loss_no_norm", job.args)
            self.assertIn("--fp8_base", job.args)
            self.assertIn("--fp8_scaled", job.args)
            self.assertIn("--sample_at_first", job.args)
            self.assertIn("--sample_prompts=toml/qinglong_hidream_o1.txt", job.args)
            self.assertIn("--sample_every_n_epochs=1", job.args)
            self.assertFalse(any(arg.startswith("--text_encoder=") for arg in job.args))
            self.assertIn("--network_module=networks.lora_hidream_o1", job.args)
            self.assertIn("--flash_attn", job.args)
            self.assertIn("--optimizer_type=adv_optm.AdamW_adv", job.args)
            self.assertIn("betas=.95,.98", job.args)
            self.assertNotIn("grams_moment=True", job.args)
            self.assertIn("--blocks_to_swap=8", job.args)
            self.assertIn("--use_pinned_memory_for_block_swap", job.args)
            self.assertNotIn("--vae=ckpts/stale-vae.safetensors", job.args)
            self.assertFalse(any(arg.startswith("--vae_dtype=") for arg in job.args))

    def test_hidream_o1_train_omits_dino_args_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "HiDream O1",
                "version": "full",
                "dit_path": "ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors",
                "dino_loss_weight": 0.0,
                "dino_loss_backend": "vit",
                "dino_loss_feature_mode": "patch",
                "dino_loss_resize": 224,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertFalse(any(arg.startswith("--dino_loss") for arg in job.args))

    def test_zimage_train_passes_soar_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "soar": True,
                "soar_lambda_aux": 0.5,
                "soar_trajectory_length": 4,
                "soar_num_sampling_steps": 24,
                "soar_sigma_upper_ratio": 1.2,
                "soar_cfg_scale_sampling": 4.5,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "zimage_train_network.py")))
            self.assertIn("--soar", job.args)
            self.assertIn("--soar_lambda_aux=0.5", job.args)
            self.assertIn("--soar_trajectory_length=4", job.args)
            self.assertIn("--soar_num_sampling_steps=24", job.args)
            self.assertIn("--soar_sigma_upper_ratio=1.2", job.args)
            self.assertIn("--soar_cfg_scale_sampling=4.5", job.args)

    def test_zimage_lora_passes_dopsd_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd": True,
                "dopsd_loss_weight": 1.25,
                "dopsd_num_sampling_steps": 8,
                "dopsd_ema_decay": 0.9999,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(
                job.script_key.endswith(str(Path("musubi-tuner") / "src" / "musubi_tuner" / "zimage_train_network.py"))
            )
            self.assertIn("--dopsd", job.args)
            self.assertIn("--dopsd_loss_weight=1.25", job.args)
            self.assertIn("--dopsd_num_sampling_steps=8", job.args)
            self.assertIn("--dopsd_ema_decay=0.9999", job.args)
            self.assertFalse(any(arg.startswith("--dopsd_teacher_embed_key") for arg in job.args))
            self.assertIn("musubi-tuner", job.runner_kwargs["env_vars"]["PYTHONPATH"])
            self.assertNotIn("musubi-tuner-dopsd-zimage", job.runner_kwargs["env_vars"]["PYTHONPATH"])

    def test_flux2_lora_and_zimage_finetune_pass_dopsd_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            flux_state = {
                "arch": "FLUX.2",
                "version": "klein-4b",
                "dit_path": "ckpts/flux2-klein.safetensors",
                "vae_path": "ckpts/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd": True,
                "dopsd_loss_weight": 0.75,
                "dopsd_num_sampling_steps": 4,
                "dopsd_ema_decay": 1.0,
            }
            zimage_finetune_state = {
                "arch": "Z-Image",
                "train_mode": "finetune",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "fused_backward_pass": False,
                "dopsd": True,
                "dopsd_loss_weight": 1.0,
                "dopsd_num_sampling_steps": 6,
                "dopsd_ema_decay": 0.999,
                "dopsd_full_ema_device": "gpu",
            }

            flux_job = build_train_job(flux_state, tmp, PROJECT_CONFIG)
            zimage_job = build_train_job(zimage_finetune_state, tmp, PROJECT_CONFIG)

            self.assertTrue(
                flux_job.script_key.endswith(str(Path("musubi-tuner") / "src" / "musubi_tuner" / "flux_2_train_network.py"))
            )
            self.assertIn("--dopsd_num_sampling_steps=4", flux_job.args)
            self.assertTrue(
                zimage_job.script_key.endswith(str(Path("musubi-tuner") / "src" / "musubi_tuner" / "zimage_train.py"))
            )
            self.assertIn("--dopsd_num_sampling_steps=6", zimage_job.args)
            self.assertIn("--dopsd_full_ema_device=gpu", zimage_job.args)
            self.assertNotIn("--fused_backward_pass", zimage_job.args)

    def test_dopsd_rejects_unsupported_train_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            flux_base_state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2-base.safetensors",
                "vae_path": "ckpts/flux2-vae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd": True,
            }
            qwen_state = {
                "arch": "Qwen Image",
                "version": "original",
                "dit_path": "ckpts/qwen_image.safetensors",
                "vae_path": "ckpts/qwen_vae.safetensors",
                "text_encoder_path": "ckpts/qwen2_5_vl.safetensors",
                "dopsd": True,
            }
            zimage_fused_backward_state = {
                "arch": "Z-Image",
                "train_mode": "finetune",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd": True,
                "fused_backward_pass": True,
            }

            with self.assertRaises(CommandBuildError):
                build_train_job(flux_base_state, tmp, PROJECT_CONFIG)
            with self.assertRaises(CommandBuildError):
                build_train_job(qwen_state, tmp, PROJECT_CONFIG)
            with self.assertRaises(CommandBuildError):
                build_train_job(zimage_fused_backward_state, tmp, PROJECT_CONFIG)

    def test_dopsd_validates_rollout_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_state = {
                "arch": "Z-Image",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "dopsd": True,
            }

            with self.assertRaises(CommandBuildError):
                build_train_job({**base_state, "dopsd_loss_weight": 0}, tmp, PROJECT_CONFIG)
            with self.assertRaises(CommandBuildError):
                build_train_job({**base_state, "dopsd_num_sampling_steps": 0}, tmp, PROJECT_CONFIG)
            with self.assertRaises(CommandBuildError):
                build_train_job({**base_state, "dopsd_ema_decay": 1.1}, tmp, PROJECT_CONFIG)

    def test_zimage_finetune_passes_soar_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "train_mode": "finetune",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "soar": True,
                "soar_lambda_aux": "1.5",
                "soar_trajectory_length": "6",
                "soar_num_sampling_steps": "40",
                "soar_sigma_upper_ratio": "1.5",
                "soar_cfg_scale_sampling": "1.0",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "zimage_train.py")))
            self.assertIn("--soar", job.args)
            self.assertIn("--soar_lambda_aux=1.5", job.args)
            self.assertIn("--soar_trajectory_length=6", job.args)
            self.assertIn("--soar_num_sampling_steps=40", job.args)
            self.assertIn("--soar_sigma_upper_ratio=1.5", job.args)
            self.assertIn("--soar_cfg_scale_sampling=1.0", job.args)

    def test_qwen_finetune_omits_unsupported_soar_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Qwen Image",
                "train_mode": "finetune",
                "dit_path": "ckpts/qwen_dit.safetensors",
                "vae_path": "ckpts/qwen_vae.safetensors",
                "text_encoder_path": "ckpts/qwen_vl.safetensors",
                "soar": True,
                "soar_lambda_aux": 0.5,
                "soar_trajectory_length": 4,
                "soar_num_sampling_steps": 24,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "qwen_image_train.py")))
            self.assertNotIn("--soar", job.args)
            self.assertFalse(any(arg.startswith("--soar_") for arg in job.args))

    def test_qwen_lora_passes_soar_args_for_original_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Qwen Image",
                "version": "original",
                "dit_path": "ckpts/qwen_dit.safetensors",
                "vae_path": "ckpts/qwen_vae.safetensors",
                "text_encoder_path": "ckpts/qwen_vl.safetensors",
                "soar": True,
                "soar_lambda_aux": 0.25,
                "soar_trajectory_length": 3,
                "soar_num_sampling_steps": 12,
                "soar_sigma_upper_ratio": 1.5,
                "soar_cfg_scale_sampling": 4.5,
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "qwen_image_train_network.py")))
            self.assertIn("--soar", job.args)
            self.assertIn("--soar_lambda_aux=0.25", job.args)
            self.assertIn("--soar_trajectory_length=3", job.args)
            self.assertIn("--soar_num_sampling_steps=12", job.args)
            self.assertIn("--soar_sigma_upper_ratio=1.5", job.args)
            self.assertIn("--soar_cfg_scale_sampling=4.5", job.args)

    def test_flux2_soar_cfg_rollout_rejects_guidance_distilled_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_state = {
                "arch": "FLUX.2",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "soar": True,
                "soar_cfg_scale_sampling": 4.5,
            }

            for version in ("dev", "klein-4b", "klein-9b"):
                with self.subTest(version=version):
                    with self.assertRaises(CommandBuildError):
                        build_train_job({**base_state, "version": version}, tmp, PROJECT_CONFIG)

    def test_zimage_finetune_rejects_soar_cfg_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "train_mode": "finetune",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "soar": True,
                "soar_cfg_scale_sampling": 4.5,
            }

            with self.assertRaises(CommandBuildError):
                build_train_job(state, tmp, PROJECT_CONFIG)

    def test_soar_validates_new_rollout_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_state = {
                "arch": "Z-Image",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "soar": True,
            }

            with self.assertRaises(CommandBuildError):
                build_train_job({**base_state, "soar_sigma_upper_ratio": 0.9}, tmp, PROJECT_CONFIG)
            with self.assertRaises(CommandBuildError):
                build_train_job({**base_state, "soar_cfg_scale_sampling": 0}, tmp, PROJECT_CONFIG)

    def test_qwen_lora_rejects_soar_for_edit_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Qwen Image",
                "version": "2509",
                "dit_path": "ckpts/qwen_dit.safetensors",
                "vae_path": "ckpts/qwen_vae.safetensors",
                "text_encoder_path": "ckpts/qwen_vl.safetensors",
                "soar": True,
            }

            with self.assertRaises(CommandBuildError):
                build_train_job(state, tmp, PROJECT_CONFIG)

    def test_soar_rejects_fused_backward_for_zimage_finetune(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "train_mode": "finetune",
                "dit_path": "ckpts/zimage.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "soar": True,
                "fused_backward_pass": True,
            }

            with self.assertRaises(CommandBuildError):
                build_train_job(state, tmp, PROJECT_CONFIG)

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

    def test_generate_maps_wan_hunyuan_missing_inference_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            wan = build_generate_job(
                {
                    "arch": "Wan2.1",
                    "task": "v2v-A14B",
                    "ckpt_dir": "ckpts/wan_official",
                    "dit_path": "ckpts/wan.safetensors",
                    "vae_path": "ckpts/wan_vae.safetensors",
                    "t5_path": "ckpts/t5.pth",
                    "clip_path": "ckpts/clip.pth",
                    "video_size": "512 512",
                    "prompt": "test",
                    "guidance_scale_high_noise": 4.5,
                    "video_path": "inputs/source.mp4",
                    "control_image_mask_path": "inputs/mask-a.png\ninputs/mask-b.png",
                    "one_frame_inference": "target_index=9,control_indices=0",
                    "lazy_loading": True,
                    "force_v2_1_time_embedding": True,
                    "disable_numpy_memmap": True,
                    "compile_args": "inductor default auto false",
                },
                tmp,
            )

            self.assertIn("--ckpt_dir=ckpts/wan_official", wan.args)
            self.assertIn("--vae=ckpts/wan_vae.safetensors", wan.args)
            self.assertIn("--guidance_scale_high_noise=4.5", wan.args)
            self.assertIn("--video_path=inputs/source.mp4", wan.args)
            mask_index = wan.args.index("--control_image_mask_path")
            self.assertEqual(wan.args[mask_index + 1:mask_index + 3], ["inputs/mask-a.png", "inputs/mask-b.png"])
            self.assertIn("--one_frame_inference=target_index=9,control_indices=0", wan.args)
            self.assertIn("--lazy_loading", wan.args)
            self.assertIn("--force_v2_1_time_embedding", wan.args)
            self.assertIn("--disable_numpy_memmap", wan.args)
            compile_index = wan.args.index("--compile_args")
            self.assertEqual(wan.args[compile_index + 1:compile_index + 5], ["inductor", "default", "auto", "false"])

            hunyuan = build_generate_job(
                {
                    "arch": "HunyuanVideo",
                    "dit_path": "ckpts/hv.pt",
                    "vae_path": "ckpts/hv_vae.safetensors",
                    "te1_path": "ckpts/llm.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                    "video_size": "512 512",
                    "prompt": "test",
                    "dit_in_channels": 32,
                    "video_path": "inputs/source.mp4",
                    "strength": 0.65,
                    "split_uncond": True,
                    "exclude_single_blocks": True,
                    "compile_args": ["inductor", "default", "false", "false"],
                },
                tmp,
            )

            self.assertIn("--dit_in_channels=32", hunyuan.args)
            self.assertIn("--vae=ckpts/hv_vae.safetensors", hunyuan.args)
            self.assertIn("--video_path=inputs/source.mp4", hunyuan.args)
            self.assertIn("--strength=0.65", hunyuan.args)
            self.assertIn("--split_uncond", hunyuan.args)
            self.assertIn("--exclude_single_blocks", hunyuan.args)
            compile_index = hunyuan.args.index("--compile_args")
            self.assertEqual(hunyuan.args[compile_index + 1:compile_index + 5], ["inductor", "default", "false", "false"])

    def test_generate_maps_framepack_qwen_and_hv15_missing_inference_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            framepack = build_generate_job(
                {
                    "arch": "FramePack",
                    "dit_path": "ckpts/framepack.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "te1_path": "ckpts/llava.safetensors",
                    "te2_path": "ckpts/clip.safetensors",
                    "image_encoder_path": "ckpts/siglip.safetensors",
                    "video_size": "512 512",
                    "prompt": "test",
                    "control_image_mask_path": "inputs/mask.png",
                    "custom_system_prompt": "system prompt",
                    "guidance_rescale": 0.2,
                    "latent_paddings": "1,2,3",
                    "one_frame_auto_resize": True,
                    "rope_scaling_factor": 0.75,
                    "rope_scaling_timestep_threshold": 800,
                    "vae_tiling": True,
                    "disable_numpy_memmap": True,
                },
                tmp,
            )

            self.assertIn("--control_image_mask_path=inputs/mask.png", framepack.args)
            self.assertIn("--custom_system_prompt=system prompt", framepack.args)
            self.assertIn("--guidance_rescale=0.2", framepack.args)
            self.assertIn("--latent_paddings=1,2,3", framepack.args)
            self.assertIn("--one_frame_auto_resize", framepack.args)
            self.assertIn("--rope_scaling_factor=0.75", framepack.args)
            self.assertIn("--rope_scaling_timestep_threshold=800", framepack.args)
            self.assertIn("--vae_tiling", framepack.args)
            self.assertIn("--disable_numpy_memmap", framepack.args)

            qwen = build_generate_job(
                {
                    "arch": "Qwen Image",
                    "version": "2511",
                    "dit_path": "ckpts/qwen.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "text_encoder_vl_path": "ckpts/qwen_vl.safetensors",
                    "video_size": "1024 1024",
                    "prompt": "test",
                    "automatic_prompt_lang_for_layered": "cn",
                    "num_layers": 60,
                    "output_layers": 3,
                    "append_original_name": True,
                    "resize_control_to_image_size": True,
                    "resize_control_to_official_size": True,
                    "vae_enable_tiling": True,
                    "disable_numpy_memmap": True,
                    "bell": True,
                },
                tmp,
            )

            self.assertIn("--automatic_prompt_lang_for_layered=cn", qwen.args)
            self.assertIn("--num_layers=60", qwen.args)
            self.assertIn("--output_layers=3", qwen.args)
            self.assertIn("--append_original_name", qwen.args)
            self.assertIn("--resize_control_to_image_size", qwen.args)
            self.assertIn("--resize_control_to_official_size", qwen.args)
            self.assertIn("--vae_enable_tiling", qwen.args)
            self.assertIn("--disable_numpy_memmap", qwen.args)
            self.assertIn("--bell", qwen.args)

            hv15 = build_generate_job(
                {
                    "arch": "HV 1.5",
                    "task": "i2v",
                    "dit_path": "ckpts/hv15.safetensors",
                    "vae_path": "ckpts/vae.safetensors",
                    "text_encoder_path": "ckpts/qwen25.safetensors",
                    "byt5_path": "ckpts/byt5.safetensors",
                    "image_encoder_path": "ckpts/siglip.safetensors",
                    "video_size": "512 512",
                    "prompt": "test",
                    "vae_sample_size": 256,
                    "disable_numpy_memmap": True,
                },
                tmp,
            )

            self.assertIn("--vae_sample_size=256", hv15.args)
            self.assertIn("--disable_numpy_memmap", hv15.args)

    def test_generate_maps_flux2_and_zimage_memory_completion_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            flux2 = build_generate_job(
                {
                    "arch": "FLUX.2",
                    "version": "dev",
                    "dit_path": "ckpts/flux2.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "text_encoder_path": "ckpts/mistral.safetensors",
                    "video_size": "1024 1024",
                    "prompt": "test",
                    "disable_numpy_memmap": True,
                },
                tmp,
            )
            self.assertIn("--disable_numpy_memmap", flux2.args)

            zimage = build_generate_job(
                {
                    "arch": "Z-Image",
                    "version": "base",
                    "dit_path": "ckpts/zimage.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "text_encoder_path": "ckpts/qwen3.safetensors",
                    "video_size": "1024 1024",
                    "prompt": "test",
                    "cpu_noise": True,
                    "disable_numpy_memmap": True,
                    "bell": True,
                },
                tmp,
            )
            self.assertIn("--cpu_noise", zimage.args)
            self.assertIn("--disable_numpy_memmap", zimage.args)
            self.assertIn("--bell", zimage.args)

    def test_generate_maps_compile_only_when_enabled_and_validates_legacy_compile_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            disabled = build_generate_job(
                {
                    "arch": "FLUX.2",
                    "version": "dev",
                    "dit_path": "ckpts/flux2.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "text_encoder_path": "ckpts/mistral.safetensors",
                    "video_size": "1024 1024",
                    "prompt": "test",
                    "compile_backend": "inductor",
                    "compile_mode": "default",
                    "compile_fullgraph": True,
                },
                tmp,
            )
            self.assertNotIn("--compile", disabled.args)
            self.assertFalse(any(arg.startswith("--compile_backend=") for arg in disabled.args))
            self.assertNotIn("--compile_fullgraph", disabled.args)

            enabled = build_generate_job(
                {
                    "arch": "FLUX.2",
                    "version": "dev",
                    "dit_path": "ckpts/flux2.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "text_encoder_path": "ckpts/mistral.safetensors",
                    "video_size": "1024 1024",
                    "prompt": "test",
                    "compile": True,
                    "compile_backend": "inductor",
                    "compile_mode": "default",
                    "compile_dynamic": "auto",
                    "compile_cache_size_limit": 32,
                    "compile_fullgraph": True,
                },
                tmp,
            )
            self.assertIn("--compile", enabled.args)
            self.assertIn("--compile_backend=inductor", enabled.args)
            self.assertIn("--compile_mode=default", enabled.args)
            self.assertIn("--compile_dynamic=auto", enabled.args)
            self.assertIn("--compile_cache_size_limit=32", enabled.args)
            self.assertIn("--compile_fullgraph", enabled.args)

            with self.assertRaises(CommandBuildError):
                build_generate_job(
                    {
                        "arch": "Wan2.1",
                        "dit_path": "ckpts/wan.safetensors",
                        "vae_path": "ckpts/wan_vae.safetensors",
                        "t5_path": "ckpts/t5.pth",
                        "clip_path": "ckpts/clip.pth",
                        "video_size": "512 512",
                        "prompt": "test",
                        "compile_args": "inductor default auto",
                    },
                    tmp,
                )

    def test_hidream_o1_generate_uses_native_args_without_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_generate_job(
                {
                    "arch": "HiDream O1",
                    "version": "full",
                    "dit_path": "ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors",
                    "text_encoder_path": "ckpts/hidream-qwen3vl",
                    "vae_path": "ckpts/stale-vae.safetensors",
                    "vae_dtype": "float32",
                    "prompt": "a studio portrait",
                    "video_size": "2048 2048",
                    "save_path": "./output_dir",
                    "attn_mode": "flash",
                    "blocks_to_swap": 8,
                    "use_pinned_memory": True,
                    "dtype": "bfloat16",
                    "noise_scale_start": 8.0,
                    "noise_scale_end": 8.0,
                    "noise_clip_std": 0.0,
                    "editing_scheduler": "flow_match",
                    "layout_bboxes": "layout.json",
                    "ref_images": "ref one.png\nref two.png",
                    "keep_original_aspect": True,
                    "fp8": True,
                    "from_file": "prompts.txt",
                    "latent_path": "latent.safetensors",
                    "save_merged_model": True,
                },
                tmp,
            )

            self.assertEqual(job.script_key, "musubi_tuner.hidream_o1_generate_image")
            self.assertIn("--model_type=full", job.args)
            self.assertIn("--dit=ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors", job.args)
            self.assertFalse(any(arg.startswith("--text_encoder=") for arg in job.args))
            self.assertIn("--save_path=./output_dir/hidream_o1.png", job.args)
            size_index = job.args.index("--image_size")
            self.assertEqual(job.args[size_index + 1:size_index + 3], ["2048", "2048"])
            self.assertIn("--flash_attn", job.args)
            self.assertIn("--dtype=bfloat16", job.args)
            self.assertFalse(any(arg.startswith("--noise_scale_start=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--noise_scale_end=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--noise_clip_std=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--editing_scheduler=") for arg in job.args))
            self.assertIn("--layout_bboxes=layout.json", job.args)
            self.assertIn("--keep_original_aspect", job.args)
            ref_index = job.args.index("--ref_images")
            self.assertEqual(job.args[ref_index + 1:ref_index + 3], ["ref one.png", "ref two.png"])
            self.assertFalse(any(arg.startswith("--vae=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--vae_dtype=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--attn_mode=") for arg in job.args))
            self.assertNotIn("--fp8", job.args)
            self.assertFalse(any(arg.startswith("--from_file") for arg in job.args))
            self.assertFalse(any(arg.startswith("--latent_path") for arg in job.args))
            self.assertFalse(any(arg.startswith("--save_merged_model") for arg in job.args))

    def test_lens_generate_uses_prompt_file_output_and_filters_common_unsupported_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_generate_job(
                {
                    "arch": "Lens",
                    "dit_path": "ckpts/lens/diffusion_models/lens_bf16.safetensors",
                    "vae_path": "ckpts/lens/vae/flux2-vae.safetensors",
                    "text_encoder_path": "ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors",
                    "prompt": "a studio portrait",
                    "negative_prompt": "low quality",
                    "video_size": "1024 1024",
                    "save_path": "./output_dir",
                    "infer_steps": 20,
                    "guidance_scale": 5.0,
                    "seed": 1026,
                    "dit_dtype": "bfloat16",
                    "vae_dtype": "float32",
                    "text_encoder_dtype": "bfloat16",
                    "attn_mode": "flash",
                    "flow_shift": 3.0,
                    "output_type": "images",
                    "lora_weight": "lora.safetensors",
                    "lora_multiplier": "1.0",
                    "from_file": "prompts.txt",
                    "latent_path": "latent.safetensors",
                    "save_merged_model": True,
                    "disable_numpy_memmap": True,
                    "fp8_base": True,
                    "fp8_scaled": True,
                },
                tmp,
            )

            self.assertEqual(job.script_key, "musubi_tuner.lens_generate_image")
            self.assertIn("--dit=ckpts/lens/diffusion_models/lens_bf16.safetensors", job.args)
            self.assertIn("--vae=ckpts/lens/vae/flux2-vae.safetensors", job.args)
            self.assertIn("--text_encoder=ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors", job.args)
            self.assertFalse(any(arg.startswith("--text_encoder_config=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--tokenizer=") for arg in job.args))
            self.assertIn("--save_path=./output_dir/lens.png", job.args)
            size_index = job.args.index("--image_size")
            self.assertEqual(job.args[size_index + 1:size_index + 3], ["1024", "1024"])
            self.assertIn("--prompt=a studio portrait", job.args)
            self.assertIn("--negative_prompt=low quality", job.args)
            self.assertIn("--guidance_scale=5.0", job.args)
            self.assertIn("--dit_dtype=bfloat16", job.args)
            self.assertIn("--vae_dtype=float32", job.args)
            self.assertIn("--text_encoder_dtype=bfloat16", job.args)
            self.assertIn("--disable_numpy_memmap", job.args)
            self.assertFalse(any(arg.startswith("--attn_mode=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--flow_shift=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--output_type=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--lora_weight") for arg in job.args))
            self.assertFalse(any(arg.startswith("--from_file") for arg in job.args))
            self.assertFalse(any(arg.startswith("--latent_path") for arg in job.args))
            self.assertFalse(any(arg.startswith("--save_merged_model") for arg in job.args))
            self.assertNotIn("--fp8", job.args)
            self.assertNotIn("--fp8_scaled", job.args)

    def test_ideogram4_generate_uses_sampler_preset_and_filters_common_unsupported_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_generate_job(
                {
                    "arch": "Ideogram-4",
                    "dit_path": "ckpts/diffusion_models/ideogram4_fp8_scaled.safetensors",
                    "unconditional_dit_path": "ckpts/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
                    "vae_path": "ckpts/vae/flux2-vae.safetensors",
                    "text_encoder_path": "ckpts/text_encoder/qwen3vl_8b_bf16.safetensors",
                    "prompt": "a studio product photo with readable label text",
                    "negative_prompt": "low quality",
                    "video_size": "1024 1024",
                    "save_path": "./output_dir",
                    "infer_steps": 20,
                    "guidance_scale": 3.0,
                    "sampler_preset": "V4_DEFAULT_20",
                    "seed": 1026,
                    "dtype": "bfloat16",
                    "disable_numpy_memmap": True,
                    "lora_weight": "lora.safetensors",
                    "from_file": "prompts.txt",
                    "latent_path": "latent.safetensors",
                    "save_merged_model": True,
                    "no_metadata": True,
                    "fp8_scaled": True,
                },
                tmp,
            )

            self.assertEqual(job.script_key, "musubi_tuner.ideogram4_generate_image")
            self.assertIn("--dit=ckpts/diffusion_models/ideogram4_fp8_scaled.safetensors", job.args)
            self.assertIn(
                "--unconditional_dit=ckpts/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
                job.args,
            )
            self.assertIn("--vae=ckpts/vae/flux2-vae.safetensors", job.args)
            self.assertIn("--text_encoder=ckpts/text_encoder/qwen3vl_8b_bf16.safetensors", job.args)
            self.assertIn("--save_path=./output_dir/ideogram4.png", job.args)
            size_index = job.args.index("--image_size")
            self.assertEqual(job.args[size_index + 1:size_index + 3], ["1024", "1024"])
            self.assertIn("--prompt=a studio product photo with readable label text", job.args)
            self.assertIn("--sampler_preset=V4_DEFAULT_20", job.args)
            self.assertIn("--seed=1026", job.args)
            self.assertIn("--dtype=bfloat16", job.args)
            self.assertIn("--disable_numpy_memmap", job.args)
            self.assertFalse(any(arg.startswith("--infer_steps=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--guidance_scale=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--negative_prompt=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--lora_weight") for arg in job.args))
            self.assertFalse(any(arg.startswith("--from_file") for arg in job.args))
            self.assertFalse(any(arg.startswith("--latent_path") for arg in job.args))
            self.assertFalse(any(arg.startswith("--save_merged_model") for arg in job.args))
            self.assertNotIn("--no_metadata", job.args)
            self.assertNotIn("--fp8_scaled", job.args)
            self.assertFalse(any("qwen3vl_8b_fp8_scaled" in arg for arg in job.args))

    def test_lens_generate_requires_direct_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CommandBuildError):
                build_generate_job(
                    {
                        "arch": "Lens",
                        "dit_path": "ckpts/lens/diffusion_models/lens_bf16.safetensors",
                        "vae_path": "ckpts/lens/vae/flux2-vae.safetensors",
                        "text_encoder_path": "ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors",
                        "from_file": "prompts.txt",
                    },
                    tmp,
                )

    def test_ideogram4_generate_requires_direct_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(CommandBuildError):
                build_generate_job(
                    {
                        "arch": "Ideogram-4",
                        "dit_path": "ckpts/diffusion_models/ideogram4_fp8_scaled.safetensors",
                        "unconditional_dit_path": "ckpts/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
                        "vae_path": "ckpts/vae/flux2-vae.safetensors",
                        "text_encoder_path": "ckpts/text_encoder/qwen3vl_8b_bf16.safetensors",
                        "from_file": "prompts.txt",
                    },
                    tmp,
                )

    def test_hidream_o1_dev_flash_generate_recipe_uses_dev_native_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_generate_job(
                {
                    "arch": "HiDream O1",
                    "version": "dev",
                    "dit_path": "ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors",
                    "prompt": "a studio portrait",
                    "video_size": "2048 2048",
                    "infer_steps": 28,
                    "save_path": "./output_dir/hidream_o1_dev_flash.png",
                    "guidance_scale": 0.0,
                    "flow_shift": 1.0,
                    "dtype": "bfloat16",
                    "noise_scale_start": 7.5,
                    "noise_scale_end": 7.5,
                    "noise_clip_std": 2.5,
                    "editing_scheduler": "flash",
                },
                tmp,
            )

            self.assertIn("--model_type=dev", job.args)
            self.assertIn("--dit=ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors", job.args)
            self.assertIn("--infer_steps=28", job.args)
            self.assertFalse(any(arg.startswith("--guidance_scale=") for arg in job.args))
            self.assertIn("--flow_shift=1.0", job.args)
            self.assertIn("--editing_scheduler=flash", job.args)
            self.assertIn("--noise_scale_start=7.5", job.args)
            self.assertIn("--noise_scale_end=7.5", job.args)
            self.assertIn("--noise_clip_std=2.5", job.args)

    def test_hidream_o1_dev_edit_flow_recipe_uses_flow_match_without_noise_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_generate_job(
                {
                    "arch": "HiDream O1",
                    "version": "dev",
                    "dit_path": "ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors",
                    "prompt": "edit the reference image",
                    "video_size": "2048 2048",
                    "infer_steps": 28,
                    "save_path": "./output_dir/hidream_o1_dev_edit_flow.png",
                    "guidance_scale": 0.0,
                    "flow_shift": 1.0,
                    "dtype": "bfloat16",
                    "editing_scheduler": "flow_match",
                    "ref_images": "ref.png",
                    "layout_bboxes": "layout.json",
                    "noise_scale_start": 7.5,
                    "noise_scale_end": 7.5,
                    "noise_clip_std": 2.5,
                },
                tmp,
            )

            self.assertIn("--model_type=dev", job.args)
            self.assertIn("--editing_scheduler=flow_match", job.args)
            self.assertIn("--layout_bboxes=layout.json", job.args)
            self.assertIn("--ref_images=ref.png", job.args)
            self.assertFalse(any(arg.startswith("--noise_scale_start=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--noise_scale_end=") for arg in job.args))
            self.assertFalse(any(arg.startswith("--noise_clip_std=") for arg in job.args))

    def test_generate_omits_zero_guidance_scale_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_generate_job(
                {
                    "arch": "Z-Image",
                    "dit_path": "ckpts/zimage.safetensors",
                    "vae_path": "ckpts/ae.safetensors",
                    "text_encoder_path": "ckpts/qwen3.safetensors",
                    "video_size": "1024 1024",
                    "prompt": "test",
                    "attn_mode": "torch",
                    "blocks_to_swap": 0,
                    "embedded_cfg_scale": 2.5,
                    "flow_shift": "3.0",
                    "guidance_scale": 0.0,
                    "infer_steps": 25,
                    "lora_multiplier": 1.0,
                    "output_type": "images",
                },
                tmp,
            )

            self.assertFalse(any(arg.startswith("--guidance_scale=") for arg in job.args))
            self.assertIn("--embedded_cfg_scale=2.5", job.args)
            self.assertIn("--flow_shift=3.0", job.args)
            self.assertIn("--infer_steps=25", job.args)
            self.assertIn("--lora_multiplier=1.0", job.args)
            self.assertIn("--output_type=images", job.args)
            self.assertIn("--blocks_to_swap=0", job.args)
            self.assertIn("--attn_mode=torch", job.args)

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

    def test_zimage_lora_passes_dit_and_mixed_precision_to_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "Z-Image",
                "train_mode": "lora",
                "version": "base",
                "dit_path": "./ckpts/diffusion_models/z_image_bf16.safetensors",
                "vae_path": "./ckpts/vae/ae.safetensors",
                "text_encoder_path": "./ckpts/text_encoder/qwen_3_4b.safetensors",
            }

            job = build_train_job(state, tmp, PROJECT_CONFIG)

            self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "zimage_train_network.py")))
            self.assertIn("--dit=./ckpts/diffusion_models/z_image_bf16.safetensors", job.args)
            self.assertIn("--mixed_precision=bf16", job.args)
            self.assertIn("--network_module=networks.lora_zimage", job.args)
            self.assertEqual(job.runner_kwargs["mixed_precision"], "bf16")

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

    def test_generate_default_save_path_matches_script_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {
                "arch": "FLUX.2",
                "version": "klein-base-4b",
                "dit_path": "ckpts/flux2.safetensors",
                "vae_path": "ckpts/ae.safetensors",
                "text_encoder_path": "ckpts/qwen3.safetensors",
                "prompt": "a studio portrait",
                "video_size": "1024 1024",
            }

            job = build_generate_job(state, tmp)

            self.assertIn("--save_path=./output_dir", job.args)
            self.assertFalse(any("output/generated" in arg or "output\\generated" in arg for arg in job.args))

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
