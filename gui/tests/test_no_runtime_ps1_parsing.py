import importlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestNoRuntimePs1Parsing(unittest.TestCase):

    def setUp(self):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        sys.modules.pop("utils.script_preset_catalog", None)

    def test_config_manager_does_not_import_script_preset_catalog_for_presets(self):
        config_manager_module = importlib.import_module("utils.config_manager")
        config_manager_module = importlib.reload(config_manager_module)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            builtin_dir = tmp_path / "builtin"
            user_dir = tmp_path / "user"
            (builtin_dir / "train").mkdir(parents=True)
            (builtin_dir / "train" / "flux2.toml").write_text('arch = "FLUX.2"\n', encoding="utf-8")

            manager = config_manager_module.ConfigManager(
                builtin_dir=str(builtin_dir),
                user_dir=str(user_dir),
            )

            self.assertIn("flux2", manager.list_configs("train"))
            self.assertEqual(manager.load_config("train", "flux2"), {"arch": "FLUX.2"})

        self.assertNotIn("utils.script_preset_catalog", sys.modules)


if __name__ == "__main__":
    unittest.main()
