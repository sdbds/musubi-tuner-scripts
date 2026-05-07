import importlib
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestProjectConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))

        cls.project_config_module = importlib.import_module("utils.project_config")
        cls.config_manager_module = importlib.import_module("utils.config_manager")

    def test_missing_project_config_returns_canonical_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.config_manager_module.ConfigManager()
            config = manager.load_project_config(tmp)

            self.assertEqual(config["schema_version"], 1)
            self.assertEqual(config["gui"]["active_page"], "train")
            self.assertEqual(config["gui"]["last_workflow"], "train")
            self.assertEqual(config["workflows"]["cache"], {})
            self.assertEqual(config["workflows"]["train"], {})
            self.assertEqual(config["dataset"]["datasets"], [])

    def test_save_and_load_project_config_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.config_manager_module.ConfigManager()
            project_dir = Path(tmp)

            raw_config = {
                "project": {
                    "name": "demo-project",
                },
                "gui": {
                    "selected_preset": "wan_a14b_lora",
                },
                "dataset": {
                    "datasets": [{"image_dir": "./images"}],
                },
                "workflows": {
                    "train": {
                        "learning_rate": "1e-4",
                        "network_dim": 32,
                    },
                },
                "interop": {
                    "workflow_extra": {
                        "train": {
                            "unknown_flag": True,
                        },
                    },
                },
                "legacy_key": "legacy-value",
            }

            self.assertTrue(manager.save_project_config(project_dir, raw_config))

            config_path = manager.get_project_config_path(project_dir)
            self.assertEqual(config_path.name, self.project_config_module.PROJECT_CONFIG_FILENAME)
            self.assertTrue(config_path.exists())

            loaded = manager.load_project_config(project_dir)
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(loaded["project"]["name"], "demo-project")
            self.assertEqual(loaded["gui"]["selected_preset"], "wan_a14b_lora")
            self.assertEqual(loaded["dataset"]["datasets"][0]["image_dir"], "./images")
            self.assertEqual(loaded["workflows"]["train"]["learning_rate"], "1e-4")
            self.assertEqual(loaded["workflows"]["train"]["network_dim"], 32)
            self.assertEqual(loaded["workflows"]["cache"], {})
            self.assertTrue(loaded["interop"]["workflow_extra"]["train"]["unknown_flag"])
            self.assertEqual(loaded["legacy_key"], "legacy-value")

    def test_legacy_flat_project_keys_are_preserved_during_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self.config_manager_module.ConfigManager()
            project_dir = Path(tmp)
            config_path = manager.get_project_config_path(project_dir)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                'model_arch = "Qwen-Image"\n'
                "resolution_w = 1328\n"
                "resolution_h = 1328\n"
                '[project]\n'
                'name = "legacy-project"\n',
                encoding="utf-8",
            )

            loaded = manager.load_project_config(project_dir)

            self.assertEqual(loaded["project"]["name"], "legacy-project")
            self.assertEqual(loaded["model_arch"], "Qwen-Image")
            self.assertEqual(loaded["resolution_w"], 1328)
            self.assertEqual(loaded["resolution_h"], 1328)
            self.assertIn("dataset", loaded)
            self.assertIn("workflows", loaded)


if __name__ == "__main__":
    unittest.main()
