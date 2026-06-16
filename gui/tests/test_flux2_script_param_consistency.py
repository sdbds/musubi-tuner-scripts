import re
import unittest
from pathlib import Path


class TestFlux2ScriptParamConsistency(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    SCRIPT_FILES = {
        "cache": "2.9flux2_cache_latent_and_text_encoder.ps1",
        "train": "3.9\u3001flux2_train_lora.ps1",
        "generate": "5.9flux2_generate.ps1",
    }

    @classmethod
    def setUpClass(cls):
        cls.scripts = {}
        for key, file_name in cls.SCRIPT_FILES.items():
            script_path = cls.ROOT / file_name
            cls.scripts[key] = script_path.read_text(encoding="utf-8")

    def assertHasFlag(self, script_text: str, flag: str):
        self.assertIn(flag, script_text, msg=f"Missing flag: {flag}")

    def test_script_files_exist(self):
        for file_name in self.SCRIPT_FILES.values():
            path = self.ROOT / file_name
            self.assertTrue(path.exists(), msg=f"Script not found: {path}")

    def test_cache_flux2_core_flags_exist(self):
        cache = self.scripts["cache"]
        for flag in [
            "--model_version",
            "--vae_dtype",
            "--device",
            "--batch_size",
            "--num_workers",
            "--skip_existing",
            "--debug_mode",
            "--console_width",
            "--console_back",
            "--console_num_images",
            "--text_encoder",
            "--fp8_text_encoder",
        ]:
            self.assertHasFlag(cache, flag)

    def test_train_flux2_core_flags_exist(self):
        train = self.scripts["train"]
        for flag in [
            "--model_version",
            "--text_encoder",
            "--fp8_text_encoder",
            "--guidance_scale",
            "--timestep_sampling",
            "--weighting_scheme",
            "--network_dim",
            "--network_alpha",
            "--optimizer_type",
            "--compile",
            "--compile_backend",
            "--compile_mode",
            "--compile_fullgraph",
            "--compile_dynamic",
            "--compile_cache_size_limit",
        ]:
            self.assertHasFlag(train, flag)

    def test_generate_flux2_core_flags_exist(self):
        generate = self.scripts["generate"]
        for flag in [
            "--model_version",
            "--text_encoder",
            "--fp8_text_encoder",
            "--guidance_scale",
            "--embedded_cfg_scale",
            "--flow_shift",
            "--control_image_path",
            "--no_resize_control",
            "--blocks_to_swap",
            "--compile",
            "--compile_backend",
            "--compile_mode",
            "--compile_fullgraph",
            "--compile_dynamic",
            "--compile_cache_size_limit",
        ]:
            self.assertHasFlag(generate, flag)

    def test_shared_flags_are_consistent_across_scripts(self):
        cache = self.scripts["cache"]
        train = self.scripts["train"]
        generate = self.scripts["generate"]

        for shared_flag in ["--model_version", "--text_encoder", "--fp8_text_encoder"]:
            self.assertHasFlag(cache, shared_flag)
            self.assertHasFlag(train, shared_flag)
            self.assertHasFlag(generate, shared_flag)

        for shared_flag in [
            "--compile",
            "--compile_backend",
            "--compile_mode",
            "--compile_fullgraph",
            "--compile_dynamic",
            "--compile_cache_size_limit",
            "--blocks_to_swap",
            "--use_pinned_memory_for_block_swap",
        ]:
            self.assertHasFlag(train, shared_flag)
            self.assertHasFlag(generate, shared_flag)

    def test_cache_adds_model_version_to_both_latent_and_te_commands(self):
        cache = self.scripts["cache"]
        self.assertRegex(cache, r"\$ext_args\.Add\(\"--model_version=\$model_version\"\)")
        self.assertRegex(cache, r"\$ext2_args\.Add\(\"--model_version=\$model_version\"\)")
        self.assertRegex(cache, r"\$ext2_args\.Add\(\"--text_encoder=\$text_encoder\"\)")

    def test_blocks_to_swap_logic_matches_train_and_generate(self):
        pattern = re.compile(
            r"if \(\$blocks_to_swap -ne 0\) \{"
            r"[\s\S]*?Add\(\"--blocks_to_swap=\$blocks_to_swap\"\)"
            r"[\s\S]*?if \(\$use_pinned_memory_for_block_swap\) \{"
            r"[\s\S]*?Add\(\"--use_pinned_memory_for_block_swap\"\)",
            re.MULTILINE,
        )

        self.assertRegex(self.scripts["train"], pattern)
        self.assertRegex(self.scripts["generate"], pattern)

    def test_execution_targets_match_flux2_scripts(self):
        cache = self.scripts["cache"]
        train = self.scripts["train"]
        generate = self.scripts["generate"]

        self.assertIn("flux_2_cache_latents.py", cache)
        self.assertIn("flux_2_cache_text_encoder_outputs.py", cache)

        self.assertIn("$laungh_script = \"flux_2_train_network\"", train)
        self.assertIn("--dataset_config=\"$dataset_config\"", train)
        self.assertIn("--dit=$dit", train)
        self.assertIn("--vae=$vae", train)

        self.assertIn("$script = \"flux_2_generate_image.py\"", generate)
        self.assertIn("--dit=$dit", generate)
        self.assertIn("--vae=$vae", generate)
        self.assertIn("--save_path=$save_path", generate)


if __name__ == "__main__":
    unittest.main()
