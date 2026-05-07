import unittest
from pathlib import Path


class TestGuiPortIntegration(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    MAIN_PATH = ROOT / "gui" / "main.py"

    @classmethod
    def setUpClass(cls):
        cls.main_text = cls.MAIN_PATH.read_text(encoding="utf-8")

    def test_main_uses_dynamic_port_resolution(self):
        self.assertIn("resolve_gui_port", self.main_text)
        self.assertIn("resolve_gui_native", self.main_text)
        self.assertIn("resolve_gui_show", self.main_text)
        self.assertNotIn("port=8080", self.main_text)
        self.assertIn('"host": host', self.main_text)
        self.assertIn('"port": port', self.main_text)
        self.assertIn('"native": native', self.main_text)
        self.assertIn('"show": show', self.main_text)


if __name__ == "__main__":
    unittest.main()
