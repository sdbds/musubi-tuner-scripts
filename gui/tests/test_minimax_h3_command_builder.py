import ast
import subprocess
import sys
import tempfile
import tomllib
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
)


PROJECT_CONFIG = {
    "dataset": {
        "general": {"resolution": [768, 1344], "batch_size": 1},
        "datasets": [
            {
                "video_directory": "videos",
                "cache_directory": "cache",
                "caption_extension": ".txt",
                "target_frames": [124],
            }
        ],
    },
    "interop": {"dataset_extra": {"root": {}, "general": {}, "datasets": [{}]}},
}


IMAGE_PROJECT_CONFIG = {
    "dataset": {
        "general": {"resolution": [768, 1344], "batch_size": 1},
        "datasets": [
            {
                "image_directory": "images",
                "cache_directory": "image-cache",
                "caption_extension": ".txt",
            }
        ],
    },
    "interop": {"dataset_extra": {"root": {}, "general": {}, "datasets": [{}]}},
}


MIXED_PROJECT_CONFIG = {
    "dataset": {
        "general": {"resolution": [768, 1344], "batch_size": 1},
        "datasets": [
            IMAGE_PROJECT_CONFIG["dataset"]["datasets"][0],
            PROJECT_CONFIG["dataset"]["datasets"][0],
        ],
    },
    "interop": {
        "dataset_extra": {"root": {}, "general": {}, "datasets": [{}, {}]}
    },
}


PATHS = {
    "dit_path": "ckpts/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "video_vae_path": "ckpts/vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae_path": "ckpts/vae/minimax_h3_audio_vae_fp32.safetensors",
    "text_encoder_path": "ckpts/text_encoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
}

H3_SUBMODULE_TARGET_SHA = "c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993"


def _add_argument_flags(source: str) -> set[str]:
    tree = ast.parse(source)
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        flags.update(
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("--")
        )
    return flags


def _indexed_submodule_source(relative_path: str) -> str:
    entry = subprocess.run(
        ["git", "ls-files", "--stage", "--", "musubi-tuner"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mode, commit, _ = entry.split(maxsplit=2)
    if mode != "160000":
        raise AssertionError("musubi-tuner must be recorded as a git submodule")
    return subprocess.run(
        ["git", "-C", str(ROOT / "musubi-tuner"), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout


def _parent_tree_submodule_commit(treeish: str = "HEAD") -> str:
    entry = subprocess.run(
        ["git", "ls-tree", treeish, "--", "musubi-tuner"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mode, object_type, commit, path = entry.split(maxsplit=3)
    if (mode, object_type, path) != ("160000", "commit", "musubi-tuner"):
        raise AssertionError(f"Unexpected musubi-tuner tree entry: {entry}")
    return commit


H3_DEFERRED_FLAGS_BY_PARSER = {
    "minimax_h3_cache_latents.py": set(),
    "minimax_h3_cache_text_encoder_outputs.py": {"--teacher_conditions"},
    "minimax_h3_train_network.py": {
        "--h3_teacher_matching",
        "--h3_teacher_conditions",
        "--h3_teacher_condition_sigma_max",
        "--h3_teacher_loss_dc_weight",
        "--h3_teacher_loss_mag_weight",
        "--h3_teacher_preservation_weight",
        "--h3_timestep_focus_min",
        "--h3_timestep_focus_max",
        "--h3_timestep_focus_prob",
    },
    "minimax_h3_generate_video.py": {
        "--interactive",
        "--ref",
        "--trajectory_dir",
        "--trajectory_stride",
        "--lora_runtime_attach",
        "--one_frame",
        "--from_file",
        "--latent_path",
        "--bell",
    },
}


H3_SUPPORTED_FLAGS_BY_PARSER = {
    "minimax_h3_cache_latents.py": {
        "--allow_experimental_duration",
        "--audio_vae",
        "--cache_seed",
        "--disable_mmap",
        "--one_frame",
        "--task",
        "--video_vae",
    },
    "minimax_h3_cache_text_encoder_outputs.py": {
        "--disable_mmap",
        "--nvfp4_scaled_mm",
        "--one_frame",
        "--task",
        "--text_cache_dtype",
        "--text_encoder",
        "--text_encoder_attn_mode",
        "--text_encoder_blocks_to_swap",
        "--uncond_output",
        "--uncond_text",
    },
    "minimax_h3_train_network.py": {
        "--audio_vae",
        "--convrot_int8",
        "--convrot_int8_bwd",
        "--dit_dtype",
        "--h3_allow_experimental_sample_duration",
        "--h3_audio_cond_clean",
        "--h3_best_of_k",
        "--h3_best_of_k_stream",
        "--h3_guidance_loss_scale",
        "--h3_guidance_loss_scale_audio",
        "--h3_guidance_loss_sigma_min",
        "--h3_guidance_loss_uncond_cache",
        "--h3_shift_audio",
        "--h3_shift_video",
        "--h3_visual_cond_clean",
        "--nvfp4_scaled_mm",
        "--one_frame",
        "--prune_adaln",
        "--task",
        "--text_encoder",
        "--text_encoder_attn_mode",
        "--text_encoder_blocks_to_swap",
        "--video_vae",
    },
    "minimax_h3_generate_video.py": {
        "--allow_experimental_duration",
        "--attn_mode",
        "--audio_vae",
        "--blocks_to_swap",
        "--convrot_int8",
        "--device",
        "--disable_numpy_memmap",
        "--dit",
        "--exclude_patterns",
        "--first_frame",
        "--frame_count",
        "--h3_audio_cond_clean",
        "--h3_shift_audio",
        "--h3_shift_video",
        "--h3_visual_cond_clean",
        "--height",
        "--include_patterns",
        "--last_frame",
        "--lora_multiplier",
        "--lora_weight",
        "--nvfp4_scaled_mm",
        "--output",
        "--prompt",
        "--prune_adaln",
        "--reference_index",
        "--reference_jsonl",
        "--seed",
        "--split_attn",
        "--steps",
        "--task",
        "--text_cache",
        "--text_encoder",
        "--text_encoder_attn_mode",
        "--text_encoder_blocks_to_swap",
        "--use_pinned_memory_for_block_swap",
        "--video_vae",
        "--width",
    },
}


class TestMiniMaxH3CommandBuilder(unittest.TestCase):
    def test_parent_tree_pins_final_h3_best_of_k_commit(self):
        self.assertEqual(_parent_tree_submodule_commit(), H3_SUBMODULE_TARGET_SHA)

    def test_h3_specific_upstream_flags_are_mapped_or_explicitly_deferred(self):
        for filename, supported_flags in H3_SUPPORTED_FLAGS_BY_PARSER.items():
            with self.subTest(parser=filename):
                parser_flags = _add_argument_flags(
                    _indexed_submodule_source(f"src/musubi_tuner/{filename}")
                )
                classified_flags = supported_flags | H3_DEFERRED_FLAGS_BY_PARSER[filename]
                self.assertEqual(parser_flags, classified_flags)

    def test_h3_gui_flags_are_supported_by_indexed_submodule_parsers(self):
        expected_by_parser = {
            "src/musubi_tuner/minimax_h3_cache_latents.py": {
                "--one_frame",
            },
            "src/musubi_tuner/minimax_h3_cache_text_encoder_outputs.py": {
                "--one_frame",
                "--uncond_output",
                "--uncond_text",
            },
            "src/musubi_tuner/minimax_h3_train_network.py": {
                "--one_frame",
                "--h3_guidance_loss_scale",
                "--h3_guidance_loss_scale_audio",
                "--h3_guidance_loss_sigma_min",
                "--h3_guidance_loss_uncond_cache",
                "--prune_adaln",
            },
            "src/musubi_tuner/minimax_h3_generate_video.py": {"--prune_adaln"},
        }
        parser_flags = {}
        for source_path, expected_flags in expected_by_parser.items():
            with self.subTest(parser=source_path):
                parser_flags[source_path] = _add_argument_flags(_indexed_submodule_source(source_path))
                self.assertEqual(expected_flags - parser_flags[source_path], set())

        preset_path = ROOT / "gui" / "presets" / "train" / "minimax_h3.toml"
        with preset_path.open("rb") as handle:
            preset = tomllib.load(handle)
        with tempfile.TemporaryDirectory() as tmp:
            default_job = build_train_job(preset, tmp, PROJECT_CONFIG)
        default_h3_flags = {
            argument.split("=", 1)[0]
            for argument in default_job.args
            if argument.startswith("--h3_guidance_loss_") or argument == "--prune_adaln"
        }
        self.assertEqual(
            default_h3_flags - parser_flags["src/musubi_tuner/minimax_h3_train_network.py"],
            set(),
        )

    def test_cache_experimental_duration_is_latent_only_and_off_by_default(self):
        base_state = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            **PATHS,
            "cache_seed": 42,
        }

        with tempfile.TemporaryDirectory() as tmp:
            default_jobs = build_cache_jobs(base_state, tmp, PROJECT_CONFIG)
            enabled_jobs = build_cache_jobs(
                {**base_state, "allow_experimental_duration": True},
                tmp,
                PROJECT_CONFIG,
            )

        self.assertNotIn("--allow_experimental_duration", default_jobs[0].args)
        self.assertNotIn("--allow_experimental_duration", default_jobs[1].args)
        self.assertIn("--allow_experimental_duration", enabled_jobs[0].args)
        self.assertNotIn("--allow_experimental_duration", enabled_jobs[1].args)

    def test_cache_builds_dual_vae_and_text_jobs_with_one_authoritative_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_cache_jobs(
                {
                    "arch": "MiniMax-H3",
                    "version": "fl2va",
                    "task": "t2va",
                    **PATHS,
                    "cache_seed": 42,
                    "skip_existing": True,
                    "te_skip_existing": True,
                    "text_cache_dtype": "bf16",
                    "text_encoder_blocks_to_swap": 50,
                    "text_encoder_attn_mode": "flash_attention_2",
                    "nvfp4_scaled_mm": True,
                    "disable_numpy_memmap": True,
                },
                tmp,
                PROJECT_CONFIG,
            )

        self.assertEqual([job.script_key for job in jobs], [
            "musubi_tuner.minimax_h3_cache_latents",
            "musubi_tuner.minimax_h3_cache_text_encoder_outputs",
        ])
        self.assertIn("--task=t2va", jobs[0].args)
        self.assertIn("--video_vae=ckpts/vae/minimax_h3_video_vae_fp16.safetensors", jobs[0].args)
        self.assertIn("--audio_vae=ckpts/vae/minimax_h3_audio_vae_fp32.safetensors", jobs[0].args)
        self.assertIn("--cache_seed=42", jobs[0].args)
        self.assertIn("--skip_existing", jobs[0].args)
        self.assertIn("--disable_mmap", jobs[0].args)
        self.assertIn("--task=t2va", jobs[1].args)
        self.assertIn(
            "--text_encoder=ckpts/text_encoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            jobs[1].args,
        )
        self.assertIn("--text_cache_dtype=bf16", jobs[1].args)
        self.assertIn("--text_encoder_blocks_to_swap=50", jobs[1].args)
        self.assertIn("--text_encoder_attn_mode=flash_attention_2", jobs[1].args)
        self.assertIn("--nvfp4_scaled_mm", jobs[1].args)
        self.assertIn("--disable_mmap", jobs[1].args)
        self.assertNotIn("--text_encoder_blocks_to_swap=50", jobs[0].args)
        self.assertNotIn("--text_encoder_attn_mode=flash_attention_2", jobs[0].args)
        self.assertNotIn("--nvfp4_scaled_mm", jobs[0].args)

    def test_one_frame_cache_marks_both_jobs_and_accepts_image_or_mixed_datasets(self):
        state = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            "one_frame": True,
            **PATHS,
        }

        for project_config in (IMAGE_PROJECT_CONFIG, MIXED_PROJECT_CONFIG):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(
                dataset_count=len(project_config["dataset"]["datasets"])
            ):
                jobs = build_cache_jobs(state, tmp, project_config)

            self.assertEqual(len(jobs), 2)
            self.assertIn("--one_frame", jobs[0].args)
            self.assertIn("--one_frame", jobs[1].args)

    def test_one_frame_cache_rejects_non_t2va_tasks_and_teacher_conditions(self):
        valid = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            "one_frame": True,
            **PATHS,
        }
        invalid_states = (
            ({**valid, "task": "fl2va"}, "one_frame.*t2va"),
            (
                {**valid, "version": "ref2va", "task": "ref2va"},
                "one_frame.*t2va",
            ),
            ({**valid, "teacher_conditions": True}, "teacher_conditions"),
        )

        for state, message in invalid_states:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(
                message=message
            ), self.assertRaisesRegex(CommandBuildError, message):
                build_cache_jobs(state, tmp, IMAGE_PROJECT_CONFIG)

    def test_cache_text_job_maps_optional_guidance_uncond_output(self):
        state = {
            "arch": "MiniMax-H3",
            "version": "ref2va",
            "task": "ref2va",
            **PATHS,
            "uncond_output": "cache/h3_uncond.safetensors",
            "uncond_text": "custom probe",
        }

        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_cache_jobs(state, tmp, PROJECT_CONFIG)
            whitespace_jobs = build_cache_jobs(
                {**state, "uncond_text": "  "},
                tmp,
                PROJECT_CONFIG,
            )
            without_output = build_cache_jobs(
                {**state, "uncond_output": ""},
                tmp,
                PROJECT_CONFIG,
            )

        self.assertIn("--uncond_output=cache/h3_uncond.safetensors", jobs[1].args)
        self.assertIn("--uncond_text=custom probe", jobs[1].args)
        self.assertIn("--uncond_text=  ", whitespace_jobs[1].args)
        self.assertFalse(any(arg.startswith("--uncond_") for arg in jobs[0].args))
        self.assertFalse(any(arg.startswith("--uncond_") for arg in without_output[1].args))

    def test_cache_rejects_boolean_uncond_scalar_values(self):
        valid = {
            "arch": "MiniMax-H3",
            "version": "ref2va",
            "task": "ref2va",
            **PATHS,
        }
        invalid_states = (
            ({**valid, "uncond_output": True}, "uncond_output must be a path"),
            (
                {
                    **valid,
                    "uncond_output": "cache/h3_uncond.safetensors",
                    "uncond_text": True,
                },
                "uncond_text must be text",
            ),
        )
        for state, message in invalid_states:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(message=message), self.assertRaisesRegex(
                CommandBuildError, message
            ):
                build_cache_jobs(state, tmp, PROJECT_CONFIG)

    def test_sampled_train_adds_future_joint_av_sampling_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "MiniMax-H3",
                    "version": "fl2va",
                    "task": "t2va",
                    **PATHS,
                    "train_mode": "lora",
                    "mixed_precision": "bf16",
                    "timestep_sampling": "uniform",
                    "weighting_scheme": "none",
                    "discrete_flow_shift": 1.0,
                    "h3_shift_video": 12.0,
                    "h3_shift_audio": 3.0,
                    "h3_visual_cond_clean": 0.999,
                    "h3_audio_cond_clean": 1.0,
                    "video_only": True,
                    "audio_loss_weight": 0.75,
                    "convrot_int8": False,
                    "convrot_int8_bwd": "bf16",
                    "h3_allow_experimental_sample_duration": True,
                    "text_encoder_blocks_to_swap": 50,
                    "text_encoder_attn_mode": "flash_attention_2",
                    "nvfp4_scaled_mm": True,
                    "dit_dtype": "bfloat16",
                    "gradient_checkpointing": True,
                    "blocks_to_swap": 48,
                    "block_swap_h2d_only": True,
                    "block_swap_ring_size": 2,
                    "enable_sample": True,
                    "sample_at_first": True,
                    "sample_every_n_epochs": 1,
                    "sample_prompts": "toml/qinglong_minimaxh3.txt",
                    "network_dim": 32,
                    "optimizer_type": "AdamW_adv",
                    "learning_rate": "1e-4",
                },
                tmp,
                PROJECT_CONFIG,
            )

        self.assertTrue(
            job.script_key.endswith(
                str(Path("musubi-tuner") / "src" / "musubi_tuner" / "minimax_h3_train_network.py")
            )
        )
        for expected in (
            "--task=t2va",
            "--dit=ckpts/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "--video_vae=ckpts/vae/minimax_h3_video_vae_fp16.safetensors",
            "--audio_vae=ckpts/vae/minimax_h3_audio_vae_fp32.safetensors",
            "--text_encoder=ckpts/text_encoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "--network_module=networks.lora_minimax_h3",
            "--h3_shift_video=12.0",
            "--h3_shift_audio=3.0",
            "--h3_visual_cond_clean=0.999",
            "--h3_audio_cond_clean=1.0",
            "--video_only",
            "--audio_loss_weight=0.75",
            "--convrot_int8_bwd=bf16",
            "--h3_allow_experimental_sample_duration",
            "--text_encoder_blocks_to_swap=50",
            "--text_encoder_attn_mode=flash_attention_2",
            "--nvfp4_scaled_mm",
            "--network_dim=32",
            "--optimizer_type=adv_optm.AdamW_adv",
            "--blocks_to_swap=48",
            "--block_swap_h2d_only",
            "--block_swap_ring_size=2",
            "--sample_at_first",
            "--sample_every_n_epochs=1",
            "--sample_prompts=toml/qinglong_minimaxh3.txt",
        ):
            self.assertIn(expected, job.args)
        self.assertNotIn("--convrot_int8", job.args)

    def test_train_can_dynamically_quantize_a_bf16_dit_and_select_int8_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "MiniMax-H3",
                    "version": "fl2va",
                    "task": "t2va",
                    **PATHS,
                    "dit_path": "ckpts/diffusion_models/minimax_h3_fl2va_bf16.safetensors",
                    "enable_sample": False,
                    "mixed_precision": "bf16",
                    "timestep_sampling": "uniform",
                    "weighting_scheme": "none",
                    "convrot_int8": True,
                    "convrot_int8_bwd": "int8",
                    "optimizer_type": "AdamW8bit",
                },
                tmp,
                PROJECT_CONFIG,
            )

        self.assertIn("--convrot_int8", job.args)
        self.assertIn("--convrot_int8_bwd=int8", job.args)

    def test_train_maps_guidance_loss_and_adaln_pruning(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "MiniMax-H3",
                    "version": "ref2va",
                    "task": "ref2va",
                    **PATHS,
                    "dit_path": "ckpts/diffusion_models/minimax_h3_ref2va_bf16.safetensors",
                    "enable_sample": False,
                    "mixed_precision": "bf16",
                    "timestep_sampling": "uniform",
                    "weighting_scheme": "none",
                    "optimizer_type": "AdamW8bit",
                    "h3_guidance_loss_scale": 4.0,
                    "h3_guidance_loss_scale_audio": 3.0,
                    "h3_guidance_loss_sigma_min": 0.15,
                    "h3_guidance_loss_uncond_cache": "cache/h3_uncond.safetensors",
                    "prune_adaln": True,
                },
                tmp,
                PROJECT_CONFIG,
            )

        for expected in (
            "--h3_guidance_loss_scale=4.0",
            "--h3_guidance_loss_scale_audio=3.0",
            "--h3_guidance_loss_sigma_min=0.15",
            "--h3_guidance_loss_uncond_cache=cache/h3_uncond.safetensors",
            "--prune_adaln",
        ):
            self.assertIn(expected, job.args)

    def test_one_frame_train_preserves_zero_audio_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "MiniMax-H3",
                    "version": "fl2va",
                    "task": "t2va",
                    "one_frame": True,
                    "video_only": True,
                    **PATHS,
                    "enable_sample": False,
                    "mixed_precision": "bf16",
                    "timestep_sampling": "uniform",
                    "weighting_scheme": "none",
                    "optimizer_type": "AdamW8bit",
                    "audio_loss_weight": 0,
                    "h3_guidance_loss_scale": 4.0,
                    "h3_guidance_loss_scale_audio": 0,
                    "h3_guidance_loss_sigma_min": 0.15,
                    "h3_guidance_loss_uncond_cache": "cache/h3-image-uncond.safetensors",
                },
                tmp,
                IMAGE_PROJECT_CONFIG,
            )

        for expected in (
            "--one_frame",
            "--video_only",
            "--audio_loss_weight=0",
            "--h3_guidance_loss_scale=4.0",
            "--h3_guidance_loss_scale_audio=0",
            "--h3_guidance_loss_sigma_min=0.15",
            "--h3_guidance_loss_uncond_cache=cache/h3-image-uncond.safetensors",
        ):
            self.assertIn(expected, job.args)

    def test_train_rejects_invalid_h3_specific_values(self):
        valid = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            **PATHS,
            "enable_sample": False,
            "mixed_precision": "bf16",
            "timestep_sampling": "uniform",
            "weighting_scheme": "none",
            "optimizer_type": "AdamW8bit",
        }
        invalid_states = (
            ({**valid, "seed": -1}, "seed"),
            ({**valid, "seed": 2**32}, "seed"),
            ({**valid, "audio_loss_weight": -0.1}, "audio_loss_weight"),
            ({**valid, "convrot_int8_bwd": "fp8"}, "convrot_int8_bwd"),
            ({**valid, "h3_guidance_loss_scale": -0.1}, "h3_guidance_loss_scale"),
            (
                {
                    **valid,
                    "h3_guidance_loss_scale": True,
                    "h3_guidance_loss_uncond_cache": "cache/h3_uncond.safetensors",
                },
                "h3_guidance_loss_scale must be a number",
            ),
            ({**valid, "h3_guidance_loss_scale_audio": -0.1}, "h3_guidance_loss_scale_audio"),
            (
                {**valid, "h3_guidance_loss_scale_audio": True},
                "h3_guidance_loss_scale_audio must be a number",
            ),
            ({**valid, "h3_guidance_loss_sigma_min": -0.1}, "h3_guidance_loss_sigma_min"),
            ({**valid, "h3_guidance_loss_sigma_min": 1.1}, "h3_guidance_loss_sigma_min"),
            (
                {**valid, "h3_guidance_loss_sigma_min": True},
                "h3_guidance_loss_sigma_min must be a number",
            ),
            (
                {
                    **valid,
                    "h3_guidance_loss_scale": 4.0,
                    "h3_guidance_loss_uncond_cache": True,
                },
                "h3_guidance_loss_uncond_cache must be a path",
            ),
            (
                {
                    **valid,
                    "h3_guidance_loss_scale": 4.0,
                    "h3_guidance_loss_uncond_cache": False,
                },
                "h3_guidance_loss_uncond_cache must be a path",
            ),
            ({**valid, "h3_guidance_loss_scale": 4.0}, "h3_guidance_loss_uncond_cache"),
            ({**valid, "one_frame": True, "task": "fl2va"}, "one_frame.*t2va"),
            (
                {
                    **valid,
                    "version": "ref2va",
                    "task": "ref2va",
                    "one_frame": True,
                },
                "one_frame.*t2va",
            ),
            ({**valid, "h3_teacher_matching": True}, "h3_teacher_matching"),
        )
        for state, message in invalid_states:
            with tempfile.TemporaryDirectory() as tmp, self.subTest(message=message), self.assertRaisesRegex(
                CommandBuildError, message
            ):
                build_train_job(state, tmp, PROJECT_CONFIG)

    def test_non_sampled_train_omits_joint_av_sampling_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "MiniMax-H3",
                    "version": "fl2va",
                    "task": "t2va",
                    **PATHS,
                    "enable_sample": False,
                    "mixed_precision": "bf16",
                    "timestep_sampling": "uniform",
                    "weighting_scheme": "none",
                    "optimizer_type": "AdamW8bit",
                },
                tmp,
                PROJECT_CONFIG,
            )

        self.assertFalse(any(arg.startswith("--video_vae=") for arg in job.args))
        self.assertFalse(any(arg.startswith("--audio_vae=") for arg in job.args))
        self.assertFalse(any(arg.startswith("--text_encoder=") for arg in job.args))
        self.assertFalse(any(arg.startswith("--sample_") for arg in job.args))

    def test_sampled_train_requires_all_joint_av_sampling_paths(self):
        state = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            "dit_path": PATHS["dit_path"],
            "video_vae_path": PATHS["video_vae_path"],
            "text_encoder_path": PATHS["text_encoder_path"],
            "enable_sample": True,
            "sample_prompts": "toml/qinglong_minimaxh3.txt",
        }
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(CommandBuildError, "audio VAE"):
            build_train_job(state, tmp, PROJECT_CONFIG)

    def test_t2va_generate_uses_native_output_geometry_and_step_flags(self):
        job = build_generate_job(
            {
                "arch": "MiniMax-H3",
                "version": "fl2va",
                "task": "t2va",
                **PATHS,
                "prompt": "A singer performs under stage lights.",
                "width": 768,
                "height": 1344,
                "frame_count": 124,
                "infer_steps": 30,
                "seed": 42,
                "blocks_to_swap": 48,
                "h3_shift_video": 12.0,
                "h3_shift_audio": 3.0,
                "save_path": "output/h3.mp4",
                "attn_mode": "flash",
                "lora_weight": "output/h3.safetensors",
                "lora_multiplier": "0.8",
                "convrot_int8": False,
                "text_encoder_blocks_to_swap": 50,
                "text_encoder_attn_mode": "flash_attention_2",
                "nvfp4_scaled_mm": True,
            },
            ROOT,
        )

        self.assertEqual(job.script_key, "musubi_tuner.minimax_h3_generate_video")
        for expected in (
            "--task=t2va",
            "--prompt=A singer performs under stage lights.",
            "--width=768",
            "--height=1344",
            "--frame_count=124",
            "--steps=30",
            "--output=output/h3.mp4",
            "--attn_mode=flash",
            "--text_encoder_blocks_to_swap=50",
            "--text_encoder_attn_mode=flash_attention_2",
            "--nvfp4_scaled_mm",
            "--lora_weight",
            "output/h3.safetensors",
            "--lora_multiplier",
            "0.8",
        ):
            self.assertIn(expected, job.args)
        self.assertFalse(any(arg.startswith("--save_path") for arg in job.args))
        self.assertFalse(any(arg.startswith("--infer_steps") for arg in job.args))
        self.assertNotIn("--convrot_int8", job.args)

    def test_t2va_generate_can_use_text_cache_without_loading_a_text_encoder(self):
        state = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            **PATHS,
            "text_encoder_path": "",
            "text_cache_path": "cache/prompt_mmh3_te.safetensors",
            "text_encoder_blocks_to_swap": 50,
            "text_encoder_attn_mode": "flash_attention_2",
            "nvfp4_scaled_mm": True,
            "prompt": "A singer performs under stage lights.",
            "width": 768,
            "height": 1344,
            "frame_count": 124,
            "save_path": "output/h3.mp4",
        }

        job = build_generate_job(state, ROOT)

        self.assertIn("--text_cache=cache/prompt_mmh3_te.safetensors", job.args)
        self.assertFalse(any(arg.startswith("--text_encoder=") for arg in job.args))
        self.assertFalse(any(arg.startswith("--text_encoder_blocks_to_swap=") for arg in job.args))
        self.assertFalse(any(arg.startswith("--text_encoder_attn_mode=") for arg in job.args))
        self.assertNotIn("--nvfp4_scaled_mm", job.args)

    def test_fl2va_generate_rejects_text_cache(self):
        with self.assertRaisesRegex(CommandBuildError, "FL2VA.*text_cache"):
            build_generate_job(
                {
                    "arch": "MiniMax-H3",
                    "version": "fl2va",
                    "task": "fl2va",
                    **PATHS,
                    "text_cache_path": "cache/prompt_mmh3_te.safetensors",
                    "prompt": "Continue this scene.",
                    "first_frame_path": "input/first.png",
                    "last_frame_path": "input/last.png",
                    "width": 768,
                    "height": 1344,
                    "frame_count": 124,
                    "save_path": "output/h3.mp4",
                },
                ROOT,
            )

    def test_generate_can_dynamically_quantize_a_bf16_dit(self):
        job = build_generate_job(
            {
                "arch": "MiniMax-H3",
                "version": "fl2va",
                "task": "t2va",
                **PATHS,
                "dit_path": "ckpts/diffusion_models/minimax_h3_fl2va_bf16.safetensors",
                "prompt": "test",
                "width": 768,
                "height": 1344,
                "frame_count": 124,
                "save_path": "output/h3.mp4",
                "convrot_int8": True,
                "prune_adaln": True,
            },
            ROOT,
        )

        self.assertIn("--convrot_int8", job.args)
        self.assertIn("--prune_adaln", job.args)

    def test_fl2va_and_ref2va_generate_map_only_their_task_inputs(self):
        common = {
            "arch": "MiniMax-H3",
            **PATHS,
            "width": 768,
            "height": 1344,
            "frame_count": 124,
            "infer_steps": 30,
            "save_path": "output/h3.mp4",
        }
        fl2va = build_generate_job(
            {
                **common,
                "version": "fl2va",
                "task": "fl2va",
                "prompt": "Continue this scene.",
                "first_frame_path": "input/first.png",
                "last_frame_path": "input/last.png",
            },
            ROOT,
        )
        ref2va = build_generate_job(
            {
                **common,
                "version": "ref2va",
                "task": "ref2va",
                "dit_path": "ckpts/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "reference_jsonl_path": "input/references.jsonl",
                "reference_index": 3,
            },
            ROOT,
        )

        self.assertIn("--first_frame=input/first.png", fl2va.args)
        self.assertIn("--last_frame=input/last.png", fl2va.args)
        self.assertFalse(any(arg.startswith("--reference_jsonl") for arg in fl2va.args))
        self.assertIn("--reference_jsonl=input/references.jsonl", ref2va.args)
        self.assertIn("--reference_index=3", ref2va.args)
        self.assertFalse(any(arg.startswith("--first_frame") for arg in ref2va.args))

    def test_generate_uses_h3_gui_aliases_and_ignores_multiplier_without_lora(self):
        job = build_generate_job(
            {
                "arch": "MiniMax-H3",
                "version": "fl2va",
                "task": "t2va",
                **PATHS,
                "prompt": "test",
                "h3_width": 768,
                "h3_height": 1344,
                "h3_frame_count": 124,
                "h3_steps": 31,
                "h3_seed": 7,
                "h3_output_path": "output/aliased.mp4",
                "h3_attn_mode": "flash",
                "h3_blocks_to_swap": 48,
                "lora_multiplier": 1.0,
            },
            ROOT,
        )

        for expected in (
            "--width=768",
            "--height=1344",
            "--frame_count=124",
            "--steps=31",
            "--seed=7",
            "--output=output/aliased.mp4",
        ):
            self.assertIn(expected, job.args)
        self.assertNotIn("--lora_multiplier", job.args)

    def test_generate_preserves_seed_beyond_float_integer_precision(self):
        job = build_generate_job(
            {
                "arch": "MiniMax-H3",
                "version": "fl2va",
                "task": "t2va",
                **PATHS,
                "prompt": "test",
                "h3_width": 768,
                "h3_height": 1344,
                "h3_frame_count": 124,
                "h3_steps": 30,
                "h3_seed": 9007199254740993,
                "h3_output_path": "output/exact-seed.mp4",
            },
            ROOT,
        )

        self.assertIn("--seed=9007199254740993", job.args)

    def test_generate_enforces_torch_seed_range(self):
        valid = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            **PATHS,
            "prompt": "test",
            "h3_width": 768,
            "h3_height": 1344,
            "h3_frame_count": 124,
            "h3_steps": 30,
            "h3_output_path": "output/seed-range.mp4",
        }

        job = build_generate_job({**valid, "h3_seed": 2**64 - 1}, ROOT)
        self.assertIn(f"--seed={2**64 - 1}", job.args)

        for seed in (-1, 2**64):
            with self.subTest(seed=seed), self.assertRaisesRegex(CommandBuildError, "seed"):
                build_generate_job({**valid, "h3_seed": seed}, ROOT)

    def test_generate_accepts_arbitrary_precision_experimental_frame_count(self):
        frame_count = 17 * 10**400 + 5
        job = build_generate_job(
            {
                "arch": "MiniMax-H3",
                "version": "fl2va",
                "task": "t2va",
                **PATHS,
                "prompt": "test",
                "h3_width": 768,
                "h3_height": 1344,
                "h3_frame_count": frame_count,
                "h3_steps": 30,
                "h3_seed": 7,
                "h3_allow_experimental_duration": True,
                "h3_output_path": "output/large-frame-count.mp4",
            },
            ROOT,
        )

        self.assertIn(f"--frame_count={frame_count}", job.args)

    def test_generate_rejects_invalid_version_task_geometry_and_missing_inputs(self):
        valid = {
            "arch": "MiniMax-H3",
            "version": "fl2va",
            "task": "t2va",
            **PATHS,
            "prompt": "test",
            "width": 768,
            "height": 1344,
            "frame_count": 124,
            "save_path": "output/h3.mp4",
        }
        invalid_states = (
            ({**valid, "version": "ref2va", "task": "t2va"}, "version ref2va"),
            ({**valid, "width": 770}, "multiple of 32"),
            ({**valid, "frame_count": 125}, r"17\*n\+5"),
            (
                {**valid, "frame_count": 125, "allow_experimental_duration": True},
                r"17\*n\+5",
            ),
            ({**valid, "task": "fl2va"}, "first and last frame"),
        )
        for state, message in invalid_states:
            with self.subTest(message=message), self.assertRaisesRegex(CommandBuildError, message):
                build_generate_job(state, ROOT)


if __name__ == "__main__":
    unittest.main()
