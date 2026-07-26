import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestMageFlowGuiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = (ROOT / "gui/wizard/step2_cache.py").read_text(encoding="utf-8")
        cls.train = (ROOT / "gui/wizard/step3_train.py").read_text(encoding="utf-8")
        cls.generate = (ROOT / "gui/wizard/step4_generate.py").read_text(encoding="utf-8")

    def test_cache_exposes_explicit_mode_text_encoder_and_seed(self):
        self.assertIn('elif arch_name == "Mage-Flow"', self.cache)
        self.assertIn('"is_edit"', self.cache)
        self.assertIn('"cache_seed"', self.cache)
        self.assertIn("on_change=lambda e: self._on_mage_flow_cache_mode_change", self.cache)
        mage_paths = self.cache.split('elif arch_name == "Mage-Flow"', 1)[1].split(
            'elif arch_name == "Lens"',
            1,
        )[0]
        self.assertIn("qwen_3_VL_4b.safetensors", mage_paths)
        self.assertNotIn("qwen3vl_4b_bf16.safetensors", mage_paths)
        self.assertNotIn("mage_processor", self.cache)

    def test_train_limits_mage_flow_to_supported_controls(self):
        self.assertIn("def _sync_mage_flow_train_ui", self.train)
        self.assertIn('["sdpa", "flash2"]', self.train)
        self.assertIn("get_mage_flow_profile", self.train)
        self.assertIn("compile_fullgraph", self.train)
        self.assertIn("mage_flow", self.train.lower())
        self.assertIn('"mage_flow_lora_qinglong"', self.train)
        self.assertIn('"mage_flow_edit_lora_qinglong"', self.train)

    def test_generate_exposes_bespoke_mage_flow_fields(self):
        for field in (
            "mage_output_path",
            "mage_control_images",
            "mage_width",
            "mage_height",
            "mage_max_size",
            "mage_steps",
            "mage_cfg_scale",
            "mage_flow_shift",
            "mage_seed",
            "mage_device",
            "mage_dtype",
            "mage_attn_mode",
            "mage_renormalize_cfg",
            "mage_allow_architecture_mismatch",
            "mage_lora_weights",
            "mage_lora_multipliers",
        ):
            self.assertIn(field, self.generate)
        self.assertIn("def _apply_mage_flow_generate_profile", self.generate)
        self.assertIn("def _sync_mage_flow_generate_ui", self.generate)
        self.assertIn('selection_type="file"', self.generate)
        self.assertGreaterEqual(self.generate.count('classes("flex-1 min-w-[112px]")'), 4)


if __name__ == "__main__":
    unittest.main()
