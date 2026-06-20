import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestScriptPresetCatalogSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.catalog = importlib.import_module("utils.script_preset_catalog")

    def test_all_preset_source_scripts_exist(self):
        missing = []
        for scope, entries in self.catalog.PRESET_SOURCES.items():
            for slug, entry in entries.items():
                if not (ROOT / entry["script"]).exists():
                    missing.append((scope, slug, entry["script"]))
        self.assertEqual(missing, [])

    def test_builtin_presets_can_be_built_from_sources(self):
        for scope in self.catalog.PRESET_SOURCES:
            with self.subTest(scope=scope):
                presets = self.catalog.get_builtin_presets(scope)
                self.assertEqual(set(presets), set(self.catalog.PRESET_SOURCES[scope]))


if __name__ == "__main__":
    unittest.main()
