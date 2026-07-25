import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from wizard.step2_cache import CacheStep  # noqa: E402
from wizard.step3_train import TrainStep  # noqa: E402
from wizard.step4_generate import GenerateStep  # noqa: E402


class TestMageFlowGuiState(unittest.TestCase):
    def test_cache_mode_change_updates_persistent_state(self):
        step = CacheStep.__new__(CacheStep)
        step.config = {"is_edit": False}

        step._on_mage_flow_cache_mode_change("true")

        self.assertTrue(step.config["is_edit"])

    def test_train_defaults_disable_sampling_without_prompt_file(self):
        step = TrainStep.__new__(TrainStep)
        step.config = {
            "is_edit": False,
            "enable_sample": True,
            "sample_at_first": True,
        }
        step.model_selector = SimpleNamespace(version="standard")

        step._apply_mage_flow_train_defaults("Mage-Flow")

        self.assertFalse(step.config["enable_sample"])
        self.assertFalse(step.config["sample_at_first"])

    def test_train_preset_values_are_preserved_when_defaults_are_filled(self):
        step = TrainStep.__new__(TrainStep)
        custom = {
            "is_edit": False,
            "dit_path": "custom/mage.safetensors",
            "timestep_sampling": "sigmoid",
            "discrete_flow_shift": 4.5,
            "weighting_scheme": "cosmap",
            "vae_dtype": "float32",
            "enable_sample": False,
        }
        step.config = dict(custom)
        step.model_selector = SimpleNamespace(version="standard")
        step.dit_path = SimpleNamespace(value=custom["dit_path"])
        step.timestep_sampling = SimpleNamespace(value=custom["timestep_sampling"])
        step.discrete_flow_shift = SimpleNamespace(value=custom["discrete_flow_shift"])
        step.weighting_scheme = SimpleNamespace(value=custom["weighting_scheme"])
        step.vae_dtype = SimpleNamespace(value=custom["vae_dtype"])

        step._apply_mage_flow_train_defaults("Mage-Flow", preserve_keys=set(custom))

        for key, value in custom.items():
            self.assertEqual(step.config[key], value)
        self.assertEqual(step.dit_path.value, custom["dit_path"])
        self.assertEqual(step.timestep_sampling.value, custom["timestep_sampling"])
        self.assertEqual(step.discrete_flow_shift.value, custom["discrete_flow_shift"])
        self.assertEqual(step.weighting_scheme.value, custom["weighting_scheme"])
        self.assertEqual(step.vae_dtype.value, custom["vae_dtype"])

    def test_apply_train_config_marks_all_explicit_keys_for_preservation(self):
        step = TrainStep.__new__(TrainStep)
        step._selected_arch = "Mage-Flow"
        step.train_mode = None
        step._apply_form_state = lambda _config: None
        step._refresh_train_mode_options = lambda _arch: None
        step._sync_mage_flow_train_ui = lambda: None
        step._set_optimizer_args_template = lambda force=False: None
        captured = {}
        step._apply_mage_flow_train_defaults = (
            lambda arch, preserve_keys=None: captured.update(
                arch=arch,
                preserve_keys=preserve_keys,
            )
        )
        config = {
            "arch": "Mage-Flow",
            "dit_path": "custom/mage.safetensors",
            "discrete_flow_shift": 4.5,
        }

        step._apply_config(config)

        self.assertEqual(captured["arch"], "Mage-Flow")
        self.assertEqual(captured["preserve_keys"], set(config))

    def test_t2i_keeps_edit_reference_images_in_strict_job_state(self):
        step = GenerateStep.__new__(GenerateStep)
        step.config = {"is_edit": True}
        step._selected_arch = "Mage-Flow"
        step._applying_config = True
        step.mage_control_images = SimpleNamespace(value="source.png\nstyle.png")
        step._sync_mage_flow_mode_fields = lambda: None

        step._on_mage_flow_mode_change(False)

        self.assertEqual(step.mage_control_images.value, "source.png\nstyle.png")
        step._collect_form_state = lambda: {
            "arch": "Mage-Flow",
            "is_edit": False,
            "mage_control_images": step.mage_control_images.value,
        }
        self.assertEqual(
            step._get_config()["mage_control_images"],
            "source.png\nstyle.png",
        )

        step._on_mage_flow_mode_change(True)
        self.assertEqual(step.mage_control_images.value, "source.png\nstyle.png")


if __name__ == "__main__":
    unittest.main()
