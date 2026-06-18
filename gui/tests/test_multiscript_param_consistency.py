import unittest
from pathlib import Path


class TestMultiScriptParamConsistency(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]

    @classmethod
    def setUpClass(cls):
        cls.cache_scripts = sorted(cls.ROOT.glob("2*cache_latent_and_text_encoder.ps1"))
        cls.train_scripts = sorted(cls.ROOT.glob("3*train*.ps1"))
        cls.train_lora_scripts = sorted(cls.ROOT.glob("3*train_lora.ps1"))
        cls.generate_scripts = sorted(cls.ROOT.glob("5*generate.ps1"))

        cls.cache_texts = {p.name: p.read_text(encoding="utf-8") for p in cls.cache_scripts}
        cls.train_script_texts = {p.name: p.read_text(encoding="utf-8") for p in cls.train_scripts}
        cls.train_texts = {p.name: p.read_text(encoding="utf-8") for p in cls.train_lora_scripts}
        cls.generate_texts = {p.name: p.read_text(encoding="utf-8") for p in cls.generate_scripts}

    def test_script_family_discovery_not_empty(self):
        self.assertGreater(len(self.cache_scripts), 0, "No cache scripts found")
        self.assertGreater(len(self.train_lora_scripts), 0, "No train_lora scripts found")
        self.assertGreater(len(self.generate_scripts), 0, "No generate scripts found")

    def test_cache_family_has_dual_cache_execution(self):
        for name, text in self.cache_texts.items():
            with self.subTest(script=name):
                has_accelerate_launch = "accelerate.commands.launch" in text
                has_direct_python_launch = "python \"./musubi-tuner/" in text
                self.assertTrue(
                    has_accelerate_launch or has_direct_python_launch,
                    msg=f"Missing expected launcher pattern in {name}",
                )
                self.assertIn("cache_latents.py", text)
                self.assertIn("cache_text_encoder_outputs.py", text)
                self.assertIn("--dataset_config", text)
                self.assertIn("--vae=", text)

    def test_train_lora_family_has_core_execution_flags(self):
        for name, text in self.train_texts.items():
            with self.subTest(script=name):
                self.assertIn("accelerate.commands.launch", text)
                self.assertIn("--dataset_config", text)
                self.assertIn("--dit=$dit", text)
                self.assertIn("--vae=$vae", text)
                self.assertIn("--learning_rate", text)
                self.assertIn("$ext_args", text)

    def test_train_scripts_do_not_force_rdzv_backend_for_multi_gpu(self):
        for name, text in self.train_script_texts.items():
            with self.subTest(script=name):
                self.assertNotIn("--rdzv_backend=c10d", text)

    def test_generate_family_has_core_execution_flags(self):
        for name, text in self.generate_texts.items():
            with self.subTest(script=name):
                has_accelerate_launch = "accelerate.commands.launch" in text
                has_direct_python_launch = "python \"./musubi-tuner/$script\"" in text
                self.assertTrue(
                    has_accelerate_launch or has_direct_python_launch,
                    msg=f"Missing expected launcher pattern in {name}",
                )
                self.assertIn("--dit=$dit", text)
                self.assertIn("--vae=$vae", text)
                self.assertIn("--save_path=$save_path", text)

    def test_base_train_and_generate_share_compile_and_block_swap_logic(self):
        train_base = (self.ROOT / "3train_lora.ps1").read_text(encoding="utf-8")
        generate_base = (self.ROOT / "5generate.ps1").read_text(encoding="utf-8")

        for flag in [
            "--compile",
            "--compile_backend",
            "--compile_mode",
            "--compile_fullgraph",
            "--compile_dynamic",
            "--compile_cache_size_limit",
            "--blocks_to_swap",
        ]:
            self.assertIn(flag, train_base)
            self.assertIn(flag, generate_base)

    def test_qwen_vl_caption_script_core_args(self):
        caption_script = self.ROOT / "1.5.qwen_vl_captions.ps1"
        self.assertTrue(caption_script.exists(), f"Script not found: {caption_script}")

        text = caption_script.read_text(encoding="utf-8")
        for flag in [
            "--image_dir=$image_dir",
            "--model_path=$model_path",
            "--output_format=$output_format",
            "--output_file=$output_file",
            "--max_new_tokens=$max_new_tokens",
            "--max_size=$max_size",
            "--prompt=$prompt",
            "--fp8_vl",
        ]:
            self.assertIn(flag, text)

        self.assertIn("caption_images_by_qwen_vl.py", text)
        self.assertIn("$caption_args", text)


if __name__ == "__main__":
    unittest.main()
