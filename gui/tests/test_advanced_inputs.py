import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from nicegui import ui

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

    def test_slider_value_coercion_keeps_legacy_float_rounding_without_step_snap(self):
        cases = (
            ("2.25", 1, 2.2),
            ("2.675", 2, 2.67),
            ("1.005", 2, 1.0),
        )

        for raw_value, decimals, expected in cases:
            with self.subTest(raw_value=raw_value):
                value = self.advanced_inputs._coerce_slider_value(
                    raw_value,
                    min_val=0,
                    max_val=10,
                    decimals=decimals,
                )
                self.assertEqual(value, expected)

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

    def test_slider_value_coercion_preserves_integers_beyond_float_precision(self):
        value = self.advanced_inputs._coerce_slider_value(
            "9007199254740993",
            min_val=0,
            max_val=9999999999,
            decimals=0,
            step=1,
            hard_max_val=None,
        )

        self.assertEqual(value, 9007199254740993)
        self.assertIsInstance(value, int)

    def test_slider_value_coercion_preserves_arbitrary_precision_integers(self):
        value = self.advanced_inputs._coerce_slider_value(
            str(10**30 + 1),
            min_val=0,
            max_val=9999999999,
            decimals=0,
            step=1,
            hard_max_val=None,
        )

        self.assertEqual(value, 10**30 + 1)
        self.assertIsInstance(value, int)

    def test_editable_slider_preserves_off_step_values_by_default(self):
        cases = (
            ("warmup", 50, 0, 10000, 100, 0),
            ("guidance", 7.2, 0, 20, 0.5, 1),
            ("timeout", 125, 30, 600, 10, 0),
        )

        for key, initial, minimum, maximum, step, decimals in cases:
            with self.subTest(key=key):
                state = {key: initial}
                with ui.column() as container:
                    slider = self.advanced_inputs.editable_slider(
                        key,
                        state,
                        key,
                        min_val=minimum,
                        max_val=maximum,
                        step=step,
                        decimals=decimals,
                    )
                try:
                    self.assertEqual(state[key], initial)
                    self.assertEqual(slider.get_bound_value(), initial)
                finally:
                    slider.dispose_form_binding()
                    container.delete()

    def test_editable_slider_can_preserve_an_optional_empty_value(self):
        state = {"audio_scale": ""}
        with ui.column() as container:
            slider = self.advanced_inputs.editable_slider(
                "audio_scale",
                state,
                "audio_scale",
                min_val=0,
                max_val=10,
                step=0.1,
                decimals=None,
                hard_max_val=None,
                allow_empty=True,
            )

        try:
            self.assertEqual(state["audio_scale"], "")
            self.assertEqual(slider.get_bound_value(), "")

            slider.set_bound_value(0.75)
            self.assertEqual(state["audio_scale"], 0.75)

            slider.set_bound_value("")
            self.assertEqual(state["audio_scale"], "")
            self.assertEqual(slider.get_bound_value(), "")
        finally:
            slider.dispose_form_binding()
            container.delete()

    def test_editable_slider_precise_edit_snaps_to_step_and_notifies_once(self):
        state = {"frames": 39}
        callbacks = []
        with ui.column() as container:
            slider = self.advanced_inputs.editable_slider(
                "frames",
                state,
                "frames",
                min_val=5,
                max_val=100,
                step=17,
                snap_to_step=True,
                on_change=callbacks.append,
            )

        try:
            button = next(element for element in container.descendants() if type(element).__name__ == "Button")
            edit_input = next(element for element in container.descendants() if type(element).__name__ == "Input")
            click = next(
                listener.handler
                for listener in button._event_listeners.values()
                if listener.type == "click"
            )
            submit = next(
                listener.handler
                for listener in edit_input._event_listeners.values()
                if listener.type == "keyup.enter"
            )
            blur = next(
                listener.handler
                for listener in edit_input._event_listeners.values()
                if listener.type == "blur"
            )

            with patch.object(self.advanced_inputs.ui, "run_javascript"):
                click(None)
            edit_input.value = "23"
            submit()
            blur()

            self.assertEqual(state["frames"], 22)
            self.assertEqual(slider.value, 22)
            self.assertEqual(callbacks, [22])
        finally:
            dispose = getattr(slider, "dispose_form_binding", None)
            if dispose:
                dispose()
            container.delete()

    def test_editable_slider_preserves_values_beyond_its_soft_track_maximum(self):
        state = {"steps": 150}
        with ui.column() as container:
            slider = self.advanced_inputs.editable_slider(
                "steps",
                state,
                "steps",
                min_val=1,
                max_val=100,
                step=1,
                hard_max_val=None,
            )
        value_button = next(
            element for element in container.descendants() if type(element).__name__ == "Button"
        )

        try:
            self.assertEqual(state["steps"], 150)
            self.assertEqual(slider.value, 150)
            self.assertEqual(float(slider._props["max"]), 150)

            slider.set_bound_value(175)
            self.assertEqual(state["steps"], 175)
            self.assertEqual(slider.value, 175)
            self.assertEqual(float(slider._props["max"]), 175)
            self.assertIsInstance(slider._props["max"], (int, float))

            slider.set_bound_value(10000000001)
            self.assertEqual(state["steps"], 10000000001)
            self.assertEqual(float(slider._props["max"]), 10000000001)

            slider.set_bound_value(9007199254740993)
            self.assertEqual(state["steps"], 9007199254740993)
            self.assertEqual(slider.get_bound_value(), 9007199254740993)
            self.assertEqual(slider.value, 100)
            self.assertEqual(value_button.text, "9007199254740993")

            slider.set_bound_value(50)
            self.assertEqual(state["steps"], 50)
            self.assertEqual(float(slider._props["max"]), 100)
        finally:
            dispose = getattr(slider, "dispose_form_binding", None)
            if dispose:
                dispose()
            container.delete()

    def test_editable_slider_disposal_releases_state_and_translation_bindings(self):
        state = {"steps": 30}
        i18n = self.advanced_inputs.get_i18n()
        binding_count = len(i18n._bindings)
        with ui.column() as container:
            slider = self.advanced_inputs.editable_slider(
                "steps",
                state,
                "steps",
                min_val=1,
                max_val=100,
            )

        try:
            self.assertIs(state["_bound_controls"]["steps"], slider)
            self.assertEqual(len(i18n._bindings), binding_count + 1)

            slider.dispose_form_binding()

            self.assertNotIn("steps", state["_bound_controls"])
            self.assertEqual(len(i18n._bindings), binding_count)
        finally:
            dispose = getattr(slider, "dispose_form_binding", None)
            if dispose:
                dispose()
            container.delete()

    def test_editable_slider_container_deletion_releases_bindings_automatically(self):
        state = {"steps": 30}
        i18n = self.advanced_inputs.get_i18n()
        binding_count = len(i18n._bindings)
        with ui.column() as container:
            slider = self.advanced_inputs.editable_slider(
                "steps",
                state,
                "steps",
                min_val=1,
                max_val=100,
            )

        self.assertIs(state["_bound_controls"]["steps"], slider)
        container.delete()

        self.assertNotIn("steps", state["_bound_controls"])
        self.assertEqual(len(i18n._bindings), binding_count)


if __name__ == "__main__":
    unittest.main()
