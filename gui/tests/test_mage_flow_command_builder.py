import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils.command_builder import CommandBuildError, build_cache_jobs, build_generate_job, build_train_job  # noqa: E402


PROJECT_CONFIG = {
    "dataset": {
        "general": {"resolution": [512, 512]},
        "datasets": [{"image_directory": "images", "caption_extension": ".txt", "batch_size": 1}],
    },
    "interop": {"dataset_extra": {"root": {}, "general": {}, "datasets": [{}]}},
}


class TestMageFlowCommandBuilder(unittest.TestCase):
    def test_edit_cache_passes_same_identity_to_both_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_cache_jobs(
                {
                    "arch": "Mage-Flow",
                    "is_edit": True,
                    "vae_path": "ckpts/vae/mage.safetensors",
                    "text_encoder_path": "ckpts/te/qwen.safetensors",
                    "cache_seed": 17,
                },
                tmp,
                PROJECT_CONFIG,
            )
        self.assertEqual(
            [job.script_key for job in jobs],
            [
                "musubi_tuner.mage_flow_cache_latents",
                "musubi_tuner.mage_flow_cache_text_encoder_outputs",
            ],
        )
        self.assertIn("--is_edit", jobs[0].args)
        self.assertIn("--is_edit", jobs[1].args)
        self.assertIn("--seed=17", jobs[0].args)
        self.assertNotIn("--seed=17", jobs[1].args)
        self.assertFalse(any("--processor" in arg or "--tokenizer" in arg for job in jobs for arg in job.args))

    def test_t2i_cache_omits_edit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_cache_jobs({"arch": "Mage-Flow", "is_edit": False}, tmp, PROJECT_CONFIG)
        self.assertNotIn("--is_edit", jobs[0].args)
        self.assertNotIn("--is_edit", jobs[1].args)

    def test_edit_train_uses_fixed_lora_contract_and_optional_sampling_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "Mage-Flow",
                    "version": "standard",
                    "is_edit": True,
                    "mixed_precision": "bf16",
                    "attn_mode": "flash2",
                    "fp8_base": True,
                    "fp8_scaled": True,
                    "vae_path": "ckpts/vae/mage.safetensors",
                    "text_encoder_path": "ckpts/te/qwen.safetensors",
                    "enable_sample": True,
                    "sample_prompts": "prompts.txt",
                },
                tmp,
                PROJECT_CONFIG,
            )
        self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "mage_flow_train_network.py")))
        self.assertIn("--dit=./ckpts/diffusion_models/mage_flow_edit_bf16.safetensors", job.args)
        self.assertIn("--vae=ckpts/vae/mage.safetensors", job.args)
        self.assertIn("--text_encoder=ckpts/te/qwen.safetensors", job.args)
        self.assertIn("--network_module=musubi_tuner.networks.lora_mage_flow", job.args)
        self.assertIn("--is_edit", job.args)
        self.assertIn("--fp8_base", job.args)
        self.assertIn("--fp8_scaled", job.args)
        self.assertIn("--flash_attn", job.args)

    def test_train_rejects_mage_flow_unsupported_combinations(self):
        invalid_states = (
            {"mixed_precision": "fp16"},
            {"fp8_base": True, "fp8_scaled": False},
            {"blocks_to_swap": 11},
            {"compile_fullgraph": True},
            {"attn_mode": "sageattn"},
            {"enable_lycoris": True},
            {"enable_blocks": True, "include_patterns": ".*q_proj.*"},
            {"dim_from_weights": True, "network_weights": ""},
            {"enable_sample": True, "sample_prompts": "prompts.txt", "vae_path": "", "text_encoder_path": ""},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for extra in invalid_states:
                state = {"arch": "Mage-Flow", "mixed_precision": "bf16", "attn_mode": "sdpa", **extra}
                with self.subTest(extra=extra), self.assertRaises(CommandBuildError):
                    build_train_job(state, tmp, PROJECT_CONFIG)

    def test_generate_uses_each_recommended_profile_without_generic_flags(self):
        expected = {
            (False, "standard"): ("mage_flow_bf16.safetensors", "20", "5.0"),
            (False, "turbo"): ("mage_flow_turbo_bf16.safetensors", "4", "1.0"),
            (True, "standard"): ("mage_flow_edit_bf16.safetensors", "30", "5.0"),
            (True, "turbo"): ("mage_flow_edit_turbo_bf16.safetensors", "4", "1.0"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for (is_edit, variant), values in expected.items():
                controls = "source.png" if is_edit else ""
                job = build_generate_job(
                    {
                        "arch": "Mage-Flow",
                        "version": variant,
                        "is_edit": is_edit,
                        "prompt": "replace the sky" if is_edit else "a glass greenhouse",
                        "mage_control_images": controls,
                    },
                    tmp,
                )
                filename, steps, cfg = values
                with self.subTest(is_edit=is_edit, variant=variant):
                    self.assertTrue(any(arg.endswith(filename) for arg in job.args))
                    self.assertIn("--vae=./ckpts/vae/mage_flow_vae_bf16.safetensors", job.args)
                    self.assertIn("--text_encoder=./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors", job.args)
                    self.assertIn(f"--steps={steps}", job.args)
                    self.assertIn(f"--cfg_scale={cfg}", job.args)
                    self.assertIn("--output=./output_dir/mage_flow.png", job.args)
                    self.assertFalse(any(arg.startswith("--save_path") for arg in job.args))
                    self.assertFalse(any(arg.startswith("--infer_steps") for arg in job.args))
                    self.assertFalse(any(arg.startswith("--output_type") for arg in job.args))
                    self.assertFalse(any(arg.startswith("--processor") for arg in job.args))

    def test_edit_generate_preserves_ordered_repeated_control_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_generate_job(
                {
                    "arch": "Mage-Flow",
                    "version": "standard",
                    "is_edit": True,
                    "prompt": "restyle the subject",
                    "mage_control_images": "source.png\nstyle.png;pose.png",
                    "mage_lora_weights": "one.safetensors\ntwo.safetensors",
                    "mage_lora_multipliers": "0.8\n1.1",
                    "mage_renormalize_cfg": True,
                },
                tmp,
            )
        controls = [arg.split("=", 1)[1] for arg in job.args if arg.startswith("--control_image=")]
        self.assertEqual(controls, ["source.png", "style.png", "pose.png"])
        lora_index = job.args.index("--lora_weight")
        self.assertEqual(job.args[lora_index + 1 : lora_index + 3], ["one.safetensors", "two.safetensors"])
        multiplier_index = job.args.index("--lora_multiplier")
        self.assertEqual(job.args[multiplier_index + 1 : multiplier_index + 3], ["0.8", "1.1"])
        self.assertIn("--renormalize_cfg", job.args)

    def test_generate_rejects_invalid_mage_flow_inputs(self):
        invalid_states = (
            {"is_edit": True, "mage_control_images": ""},
            {"is_edit": True, "mage_control_images": "1.png\n2.png\n3.png\n4.png"},
            {"is_edit": False, "mage_control_images": "source.png"},
            {"is_edit": False, "mage_width": 1024, "mage_height": ""},
            {"is_edit": False, "mage_max_size": 1024},
            {"is_edit": False, "mage_steps": 0},
            {"is_edit": False, "mage_flow_shift": 0},
            {"is_edit": False, "mage_dtype": "float8"},
            {"is_edit": False, "mage_attn_mode": "sageattn"},
            {"is_edit": False, "mage_lora_weights": "one.safetensors", "mage_lora_multipliers": "1.0\n0.5"},
            {"is_edit": False, "from_file": "prompts.txt"},
            {"is_edit": False, "prompt": ""},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for extra in invalid_states:
                state = {"arch": "Mage-Flow", "version": "standard", "prompt": "test", **extra}
                with self.subTest(extra=extra), self.assertRaises(CommandBuildError):
                    build_generate_job(state, tmp)


if __name__ == "__main__":
    unittest.main()
