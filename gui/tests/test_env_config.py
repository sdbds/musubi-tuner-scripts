import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class TestEnvConfig(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    GUI_ROOT = ROOT / "gui"

    @classmethod
    def setUpClass(cls):
        if str(cls.GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(cls.GUI_ROOT))
        from utils import env_config

        cls.env_config = env_config

    def test_defaults_include_musubi_runtime_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_vars.json"
            with mock.patch.object(self.env_config, "CONFIG_PATH", config_path):
                with mock.patch.object(self.env_config, "_detect_system_language", lambda: "en"):
                    cfg = self.env_config.load_env_config()

        self.assertEqual(cfg["HF_HOME"], "huggingface")
        self.assertEqual(cfg["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(cfg["XFORMERS_FORCE_DISABLE_TRITON"], "1")
        self.assertEqual(cfg["VSLANG"], "1033")
        self.assertIn("CUDA_VISIBLE_DEVICES", cfg)
        self.assertIn("UV_INDEX_STRATEGY", cfg)

    def test_chinese_defaults_use_common_mirrors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_vars.json"
            with mock.patch.object(self.env_config, "CONFIG_PATH", config_path):
                with mock.patch.object(self.env_config, "_detect_system_language", lambda: "zh"):
                    cfg = self.env_config.load_env_config()

        self.assertEqual(cfg["HF_ENDPOINT"], "https://hf-mirror.com")
        self.assertEqual(cfg["UV_INDEX_URL"], "https://pypi.tuna.tsinghua.edu.cn/simple/")

    def test_save_load_roundtrip_and_subprocess_filter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_vars.json"
            with mock.patch.object(self.env_config, "CONFIG_PATH", config_path):
                with mock.patch.object(self.env_config, "_detect_system_language", lambda: "en"):
                    self.env_config.save_env_config({"HF_HOME": "custom-cache", "EMPTY_VALUE": ""})
                    cfg = self.env_config.load_env_config()
                    env = self.env_config.get_env_for_subprocess()

        self.assertEqual(cfg["HF_HOME"], "custom-cache")
        self.assertEqual(cfg["EMPTY_VALUE"], "")
        self.assertEqual(env["HF_HOME"], "custom-cache")
        self.assertNotIn("EMPTY_VALUE", env)

    def test_cuda_home_defaults_from_cuda_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env_vars.json"
            with mock.patch.object(self.env_config, "CONFIG_PATH", config_path):
                with mock.patch.object(self.env_config, "_detect_system_language", lambda: "en"):
                    with mock.patch.dict(os.environ, {"CUDA_PATH": r"C:\CUDA"}, clear=False):
                        cfg = self.env_config.load_env_config()

        self.assertEqual(cfg["CUDA_HOME"], r"C:\CUDA")


if __name__ == "__main__":
    unittest.main()
