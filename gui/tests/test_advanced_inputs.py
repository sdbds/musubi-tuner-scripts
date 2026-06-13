import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestAdvancedInputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.advanced_inputs = importlib.import_module("components.advanced_inputs")

    def test_slider_value_coercion_accepts_numeric_strings(self):
        value = self.advanced_inputs._coerce_slider_value("7.26", min_val=0, max_val=10, decimals=1)

        self.assertEqual(value, 7.3)
        self.assertIsInstance(value, float)

    def test_slider_value_coercion_keeps_integral_sliders_as_int(self):
        value = self.advanced_inputs._coerce_slider_value("42", min_val=0, max_val=100, decimals=0)

        self.assertEqual(value, 42)
        self.assertIsInstance(value, int)

    def test_slider_value_coercion_clamps_and_can_fall_back(self):
        self.assertEqual(
            self.advanced_inputs._coerce_slider_value("999", min_val=0, max_val=10, decimals=0),
            10,
        )
        self.assertEqual(
            self.advanced_inputs._coerce_slider_value("not-a-number", min_val=0, max_val=10, decimals=0, fallback=3),
            3,
        )


if __name__ == "__main__":
    unittest.main()
