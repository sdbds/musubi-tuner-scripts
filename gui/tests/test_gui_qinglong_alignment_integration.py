import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestGuiQinglongAlignmentIntegration(unittest.TestCase):

    MAIN_PATH = ROOT / "gui" / "main.py"
    MODEL_SELECTOR_PATH = ROOT / "gui" / "components" / "model_selector.py"
    CACHE_PATH = ROOT / "gui" / "wizard" / "step2_cache.py"
    TRAIN_PATH = ROOT / "gui" / "wizard" / "step3_train.py"
    GENERATE_PATH = ROOT / "gui" / "wizard" / "step4_generate.py"

    @classmethod
    def setUpClass(cls):
        cls.main_text = cls.MAIN_PATH.read_text(encoding="utf-8")
        cls.model_selector_text = cls.MODEL_SELECTOR_PATH.read_text(encoding="utf-8")
        cls.cache_text = cls.CACHE_PATH.read_text(encoding="utf-8")
        cls.train_text = cls.TRAIN_PATH.read_text(encoding="utf-8")
        cls.generate_text = cls.GENERATE_PATH.read_text(encoding="utf-8")

    def test_main_uses_qinglong_branding(self):
        self.assertIn('ui.label("🐉")', self.main_text)
        self.assertTrue('favicon="🐉"' in self.main_text or '"favicon": "🐉"' in self.main_text)
        self.assertNotIn('ui.label("🎨")', self.main_text)

    def test_model_selector_uses_shared_catalog(self):
        self.assertIn("model_catalog", self.model_selector_text)
        self.assertNotIn("MODEL_ARCHITECTURES = {", self.model_selector_text)

    def test_pages_no_longer_return_empty_preset_state(self):
        self.assertNotIn("return {}", self.cache_text)
        self.assertNotIn("return {}", self.train_text)
        self.assertNotIn("return {}", self.generate_text)
        self.assertNotIn("pass\n", self.cache_text)
        self.assertNotIn("pass\n", self.train_text)
        self.assertNotIn("pass\n", self.generate_text)

    def test_pages_do_not_keep_hardcoded_flux2_and_wan_option_lists(self):
        self.assertNotIn("['dev', 'klein-4b', 'klein-base-4b', 'klein-9b', 'klein-base-9b']", self.cache_text)
        self.assertNotIn("['dev', 'klein-4b', 'klein-base-4b', 'klein-9b', 'klein-base-9b']", self.generate_text)
        self.assertNotIn("['t2v-1.3B', 't2v-14B', 'i2v-14B', 't2i-14B'", self.cache_text)
        self.assertNotIn("['t2v-1.3B', 't2v-14B', 'i2v-14B',", self.generate_text)


if __name__ == "__main__":
    unittest.main()
