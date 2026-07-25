import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils.command_builder import CommandBuildError, build_cache_jobs, build_train_job  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
