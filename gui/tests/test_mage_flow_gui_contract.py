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
        self.assertIn("qwen3vl_4b_bf16.safetensors", self.cache)
        self.assertNotIn("mage_processor", self.cache)

    def test_train_limits_mage_flow_to_supported_controls(self):
        self.assertIn("def _sync_mage_flow_train_ui", self.train)
        self.assertIn('["sdpa", "flash2"]', self.train)
        self.assertIn("get_mage_flow_profile", self.train)
        self.assertIn("compile_fullgraph", self.train)
        self.assertIn("mage_flow", self.train.lower())


if __name__ == "__main__":
    unittest.main()
