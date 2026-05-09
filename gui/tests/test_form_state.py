import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class _FakeControl:

    def __init__(self, value=None):
        self.value = value

    def update(self):
        return None


class _FakeBoundControl:

    def __init__(self):
        self.bound_value = None

    def set_bound_value(self, value):
        self.bound_value = value


class _FakeSelector:

    def __init__(self):
        self.arch = "FLUX.2"
        self.version = ""
        self.task = None

    def set_arch(self, value):
        self.arch = value

    def set_version(self, value):
        self.version = value

    def set_task(self, value):
        self.task = value


class TestFormState(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.form_state_module = importlib.import_module("utils.form_state")

    def test_apply_form_state_drops_unknown_keys_instead_of_hiding_them(self):
        mixin = self.form_state_module.FormStateMixin()
        mixin.config = {"known_flag": False}
        mixin.model_selector = _FakeSelector()
        mixin.visible_input = _FakeControl()
        mixin._init_form_state()

        mixin._apply_form_state(
            {
                "arch": "Qwen Image",
                "known_flag": True,
                "visible_input": "visible",
                "unknown_hidden_field": "should_not_stick",
            }
        )

        self.assertEqual(mixin.model_selector.arch, "Qwen Image")
        self.assertTrue(mixin.config["known_flag"])
        self.assertEqual(mixin.visible_input.value, "visible")
        self.assertNotIn("unknown_hidden_field", mixin.config)

    def test_apply_form_state_updates_registered_bound_controls(self):
        mixin = self.form_state_module.FormStateMixin()
        mixin.config = {"guidance_scale": 1.0}
        mixin.model_selector = _FakeSelector()
        mixin.guidance_scale = _FakeBoundControl()
        mixin._init_form_state()

        mixin._apply_form_state({"arch": "Z-Image", "guidance_scale": 0.0})

        self.assertEqual(mixin.model_selector.arch, "Z-Image")
        self.assertEqual(mixin.config["guidance_scale"], 0.0)
        self.assertEqual(mixin.guidance_scale.bound_value, 0.0)


if __name__ == "__main__":
    unittest.main()
