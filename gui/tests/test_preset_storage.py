import contextlib
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestPresetStorage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.config_manager_module = importlib.import_module("utils.config_manager")

    def test_builtin_and_user_presets_are_separated_by_scope_and_saved_as_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            builtin_dir = tmp_path / "builtin"
            user_dir = tmp_path / "user"
            (builtin_dir / "train").mkdir(parents=True)

            (builtin_dir / "train" / "flux_builtin.toml").write_text('arch = "FLUX.2"\n', encoding="utf-8")

            manager = self.config_manager_module.ConfigManager(
                builtin_dir=str(builtin_dir),
                user_dir=str(user_dir),
            )

            self.assertEqual(manager.get_config_source("train", "flux_builtin"), "builtin")
            self.assertIn("flux_builtin", manager.list_configs("train"))

            self.assertTrue(manager.save_config("train", "my_custom", {"arch": "Wan2.1"}))
            self.assertEqual(manager.get_config_source("train", "my_custom"), "user")
            self.assertIn("my_custom", manager.list_configs("train"))
            self.assertEqual(manager.load_config("train", "my_custom"), {"arch": "Wan2.1"})
            self.assertTrue((user_dir / "train" / "my_custom.toml").exists())
            self.assertFalse((user_dir / "train" / "my_custom.json").exists())
            self.assertEqual(manager.list_configs("cache"), [])

    def test_builtin_presets_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            builtin_dir = tmp_path / "builtin"
            user_dir = tmp_path / "user"
            (builtin_dir / "generate").mkdir(parents=True)

            (builtin_dir / "generate" / "flux_builtin.toml").write_text('arch = "FLUX.2"\n', encoding="utf-8")

            manager = self.config_manager_module.ConfigManager(
                builtin_dir=str(builtin_dir),
                user_dir=str(user_dir),
            )

            self.assertFalse(manager.delete_config("generate", "flux_builtin"))
            self.assertEqual(manager.get_config_source("generate", "flux_builtin"), "builtin")

    def test_legacy_json_user_presets_are_visible_from_all_scopes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            builtin_dir = tmp_path / "builtin"
            user_dir = tmp_path / "user"
            builtin_dir.mkdir(parents=True)
            user_dir.mkdir(parents=True)

            (user_dir / "legacy_shared.json").write_text('{"arch": "FLUX.2", "seed": 42}', encoding="utf-8")

            manager = self.config_manager_module.ConfigManager(
                builtin_dir=str(builtin_dir),
                user_dir=str(user_dir),
            )

            for scope in ("cache", "train", "generate"):
                self.assertEqual(manager.get_config_source(scope, "legacy_shared"), "user")
                self.assertIn("legacy_shared", manager.list_configs(scope))
                self.assertEqual(manager.load_config(scope, "legacy_shared"), {"arch": "FLUX.2", "seed": 42})

    def test_builtin_file_labels_use_arch_name_instead_of_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            builtin_dir = tmp_path / "builtin"
            user_dir = tmp_path / "user"
            (builtin_dir / "train").mkdir(parents=True)

            (builtin_dir / "train" / "flux2.toml").write_text('arch = "FLUX.2"\n', encoding="utf-8")

            manager = self.config_manager_module.ConfigManager(
                builtin_dir=str(builtin_dir),
                user_dir=str(user_dir),
            )

            entries = {entry["name"]: entry for entry in manager.list_config_entries("train")}
            self.assertEqual(entries["flux2"]["label"], "FLUX.2")

    def test_user_preset_names_cannot_escape_user_scope_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            builtin_dir = tmp_path / "builtin"
            user_dir = tmp_path / "user"
            manager = self.config_manager_module.ConfigManager(
                builtin_dir=str(builtin_dir),
                user_dir=str(user_dir),
            )

            unsafe_names = [
                r"..\escaped",
                "../escaped",
                r"C:\temp\escaped",
                "name:stream",
                "..",
            ]
            for name in unsafe_names:
                with self.subTest(name=name):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertFalse(manager.save_config("train", name, {"arch": "FLUX.2"}))
                        self.assertIsNone(manager.load_config("train", name))
                    self.assertFalse(manager.delete_config("train", name))

            self.assertFalse((tmp_path / "escaped.toml").exists())
            self.assertFalse((tmp_path / "escaped.json").exists())

    def test_unknown_preset_scopes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manager = self.config_manager_module.ConfigManager(
                builtin_dir=str(tmp_path / "builtin"),
                user_dir=str(tmp_path / "user"),
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(manager.save_config("unknown", "preset", {"arch": "FLUX.2"}))
                self.assertIsNone(manager.load_config("unknown", "preset"))
            self.assertFalse(manager.delete_config("unknown", "preset"))
            self.assertEqual(manager.list_config_entries("unknown"), [])


if __name__ == "__main__":
    unittest.main()
