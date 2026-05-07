import importlib
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestScriptCoverageManifest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.manifest = importlib.import_module("utils.script_coverage_manifest")

    def test_all_root_ps1_scripts_are_classified(self):
        root_scripts = {path.name for path in ROOT.glob("*.ps1")} - self.manifest.ignored_scripts()
        self.assertEqual(root_scripts, self.manifest.all_classified_scripts())

    def test_install_script_is_explicitly_ignored(self):
        self.assertIn("1.install-uv-qinglong.ps1", self.manifest.ignored_scripts())
        self.assertNotIn("1.install-uv-qinglong.ps1", self.manifest.all_classified_scripts())

    def test_script_families_2_3_4_are_covered(self):
        covered = self.manifest.all_classified_scripts()
        scripts_234 = {
            path.name
            for path in ROOT.glob("*.ps1")
            if path.name.startswith(("2", "3", "4"))
        }
        self.assertTrue(scripts_234)
        self.assertEqual(scripts_234 - covered, set())

    def test_scripts_are_not_double_classified(self):
        classified = (
            list(self.manifest.NATIVE_GUI)
            + list(self.manifest.COMPATIBILITY_LAUNCHER)
            + list(self.manifest.UNSUPPORTED)
            + list(self.manifest.IGNORED)
        )
        duplicates = sorted(name for name, count in Counter(classified).items() if count > 1)
        self.assertEqual(duplicates, [])

    def test_longcat_generate_stays_compatibility_until_entry_point_exists(self):
        self.assertIn("5.7、long_cat_generate.ps1", self.manifest.COMPATIBILITY_LAUNCHER)
        self.assertNotIn("5.7、long_cat_generate.ps1", self.manifest.NATIVE_GUI)


if __name__ == "__main__":
    unittest.main()
