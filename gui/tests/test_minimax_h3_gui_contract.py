import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from wizard.step3_train import TrainStep  # noqa: E402
from wizard.step4_generate import GenerateStep  # noqa: E402
from utils.i18n import TRANSLATIONS  # noqa: E402


H3_TRANSLATION_KEYS = {
    "cache_seed",
    "text_cache_dtype",
    "h3_video_vae",
    "h3_audio_vae",
    "h3_text_encoder_int8_recommended",
    "h3_video_vae_sampling",
    "h3_audio_vae_sampling",
    "h3_text_encoder_sampling",
    "h3_text_encoder_blocks_to_swap",
    "h3_text_encoder_blocks_to_swap_tooltip",
    "h3_text_encoder_attn_mode",
    "h3_text_encoder_attn_auto",
    "h3_nvfp4_scaled_mm",
    "h3_nvfp4_scaled_mm_tooltip",
    "h3_text_cache",
    "h3_text_cache_tooltip",
    "h3_dit_dtype",
    "h3_shift_video",
    "h3_shift_audio",
    "h3_visual_cond_clean",
    "h3_audio_cond_clean",
    "h3_audio_loss_weight",
    "h3_convrot_backward",
    "h3_bf16_backward",
    "h3_int8_backward",
    "h3_video_only",
    "h3_video_only_tooltip",
    "h3_quantize_convrot_int8",
    "h3_convrot_int8_tooltip",
    "h3_allow_experimental_sample_duration",
    "h3_allow_experimental_duration",
    "h3_frames_formula",
    "h3_output_video",
    "h3_fl2va_inputs",
    "h3_first_frame",
    "h3_select_first_frame",
    "h3_last_frame",
    "h3_select_last_frame",
    "h3_ref2va_inputs",
    "h3_reference_jsonl",
    "h3_select_reference_jsonl",
    "h3_reference_index",
    "h3_flow_memory",
    "h3_split_attention",
    "h3_pinned_memory",
}


class TestMiniMaxH3GuiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = (ROOT / "gui/wizard/step2_cache.py").read_text(encoding="utf-8")
        cls.train = (ROOT / "gui/wizard/step3_train.py").read_text(encoding="utf-8")
        cls.generate = (ROOT / "gui/wizard/step4_generate.py").read_text(encoding="utf-8")

    def test_cache_exposes_dual_vae_text_encoder_and_native_cache_options(self):
        branch = self.cache.split('elif arch_name == "MiniMax-H3"', 1)[1]
        for field in (
            "video_vae_path",
            "audio_vae_path",
            "text_encoder_path",
            "cache_seed",
            "allow_experimental_duration",
            "text_cache_dtype",
            "text_encoder_blocks_to_swap",
            "text_encoder_attn_mode",
            "nvfp4_scaled_mm",
            "disable_numpy_memmap",
        ):
            self.assertIn(field, branch)
        self.assertIn("self.config.setdefault('allow_experimental_duration', False)", branch)
        self.assertIn("self.config.setdefault('text_encoder_attn_mode', 'flash_attention_2')", branch)
        self.assertIn('arch_name not in {"HiDream O1", "MiniMax-H3"}', self.cache)

    def test_train_exposes_sampling_dependencies_and_h3_flow_controls(self):
        branch = self.train.split('elif arch_name == "MiniMax-H3"', 1)[1]
        for field in (
            "video_vae_path",
            "audio_vae_path",
            "text_encoder_path",
            "text_encoder_blocks_to_swap",
            "text_encoder_attn_mode",
            "nvfp4_scaled_mm",
            "dit_dtype",
            "h3_shift_video",
            "h3_shift_audio",
            "h3_visual_cond_clean",
            "h3_audio_cond_clean",
            "video_only",
            "audio_loss_weight",
            "convrot_int8",
            "convrot_int8_bwd",
            "h3_allow_experimental_sample_duration",
        ):
            self.assertIn(field, branch)
        self.assertIn("*.txt *.toml *.json", self.train)
        self.assertIn("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", self.train)
        self.assertIn("self.config.setdefault('h3_allow_experimental_sample_duration', True)", branch)
        self.assertIn("self.config.setdefault('text_encoder_attn_mode', 'flash_attention_2')", branch)
        self.assertIn("def _apply_minimax_h3_train_defaults", self.train)
        self.assertIn("def _sync_minimax_h3_train_ui", self.train)
        self.assertIn("max_val=48", self.train.split("self._blocks_to_swap_slider = editable_slider", 1)[1])

    def test_generate_exposes_native_geometry_output_and_task_inputs(self):
        branch = self.generate.split('elif arch_name == "MiniMax-H3"', 1)[1]
        for field in (
            "video_vae_path",
            "audio_vae_path",
            "text_encoder_path",
            "text_cache_path",
            "text_encoder_blocks_to_swap",
            "text_encoder_attn_mode",
            "nvfp4_scaled_mm",
            "first_frame_path",
            "last_frame_path",
            "reference_jsonl_path",
            "reference_index",
            "h3_width",
            "h3_height",
            "h3_frame_count",
            "h3_steps",
            "h3_output_path",
            "h3_shift_video",
            "h3_shift_audio",
            "convrot_int8",
        ):
            self.assertIn(field, branch)
        self.assertIn("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", self.generate)
        self.assertIn("self.config.setdefault('text_encoder_attn_mode', 'flash_attention_2')", branch)
        self.assertIn("def _sync_minimax_h3_task_ui", self.generate)

    def test_h3_localization_is_complete_for_every_supported_language(self):
        for language in ("en", "zh", "ja", "ko"):
            with self.subTest(language=language):
                missing = H3_TRANSLATION_KEYS.difference(TRANSLATIONS[language])
                self.assertEqual(missing, set())
                self.assertTrue(all(TRANSLATIONS[language][key].strip() for key in H3_TRANSLATION_KEYS))

        self.assertEqual(TRANSLATIONS["en"]["cache_seed"], "Latent Seed")
        for language in ("zh", "ja", "ko"):
            with self.subTest(language=language):
                self.assertNotEqual(
                    TRANSLATIONS[language]["h3_allow_experimental_duration"],
                    TRANSLATIONS["en"]["h3_allow_experimental_duration"],
                )

    def test_h3_visible_text_uses_translation_keys(self):
        expected_by_source = {
            "cache": (
                "h3_video_vae",
                "h3_audio_vae",
                "h3_text_encoder_int8_recommended",
                "text_cache_dtype",
                "h3_text_encoder_blocks_to_swap",
                "h3_text_encoder_attn_mode",
                "h3_nvfp4_scaled_mm",
                "h3_allow_experimental_duration",
            ),
            "train": (
                "h3_video_vae_sampling",
                "h3_audio_vae_sampling",
                "h3_text_encoder_sampling",
                "h3_text_encoder_blocks_to_swap",
                "h3_text_encoder_attn_mode",
                "h3_nvfp4_scaled_mm",
                "h3_dit_dtype",
                "h3_audio_loss_weight",
                "h3_convrot_backward",
                "h3_video_only",
                "h3_quantize_convrot_int8",
                "h3_allow_experimental_sample_duration",
            ),
            "generate": (
                "h3_video_vae",
                "h3_audio_vae",
                "h3_text_encoder_int8_recommended",
                "h3_frames_formula",
                "h3_output_video",
                "h3_text_cache",
                "h3_text_encoder_blocks_to_swap",
                "h3_text_encoder_attn_mode",
                "h3_nvfp4_scaled_mm",
                "h3_fl2va_inputs",
                "h3_first_frame",
                "h3_last_frame",
                "h3_ref2va_inputs",
                "h3_reference_jsonl",
                "h3_reference_index",
                "h3_flow_memory",
                "h3_split_attention",
                "h3_pinned_memory",
                "h3_allow_experimental_duration",
            ),
        }
        for source_name, keys in expected_by_source.items():
            source = getattr(self, source_name)
            for key in keys:
                with self.subTest(source=source_name, key=key):
                    self.assertIn(f"t('{key}'", source)

    def test_train_defaults_refresh_h3_controls_and_recommended_profile(self):
        step = TrainStep.__new__(TrainStep)
        bound_controls = {
            "network_dim": SimpleNamespace(value=16),
            "block_swap_h2d_only": SimpleNamespace(value=False),
            "h3_allow_experimental_sample_duration": SimpleNamespace(value=False),
            "text_encoder_blocks_to_swap": SimpleNamespace(value=0),
            "nvfp4_scaled_mm": SimpleNamespace(value=True),
        }
        step.config = {
            "audio_loss_weight": 0.25,
            "convrot_int8_bwd": "int8",
            "network_dim": 16,
            "optimizer_type": "AdamW8bit",
            "block_swap_h2d_only": False,
            "h3_allow_experimental_sample_duration": False,
            "text_encoder_blocks_to_swap": 0,
            "text_encoder_attn_mode": "eager",
            "nvfp4_scaled_mm": True,
            "_bound_controls": bound_controls,
        }
        step.audio_loss_weight = SimpleNamespace(value=0.25)
        step.convrot_int8_bwd = SimpleNamespace(value="int8")
        step.optimizer_type = SimpleNamespace(value="AdamW8bit")
        step.text_encoder_attn_mode = SimpleNamespace(value="eager")

        step._apply_minimax_h3_train_defaults("MiniMax-H3")

        self.assertEqual(step.audio_loss_weight.value, 1.0)
        self.assertEqual(step.convrot_int8_bwd.value, "bf16")
        self.assertEqual(step.config["network_dim"], 32)
        self.assertEqual(bound_controls["network_dim"].value, 32)
        self.assertEqual(step.config["optimizer_type"], "AdamW_adv")
        self.assertEqual(step.optimizer_type.value, "AdamW_adv")
        self.assertTrue(step.config["block_swap_h2d_only"])
        self.assertTrue(bound_controls["block_swap_h2d_only"].value)
        self.assertTrue(step.config["h3_allow_experimental_sample_duration"])
        self.assertTrue(bound_controls["h3_allow_experimental_sample_duration"].value)
        self.assertTrue(step.config["gradient_checkpointing"])
        self.assertEqual(step.config["text_encoder_blocks_to_swap"], 50)
        self.assertEqual(bound_controls["text_encoder_blocks_to_swap"].value, 50)
        self.assertEqual(step.config["text_encoder_attn_mode"], "flash_attention_2")
        self.assertEqual(step.text_encoder_attn_mode.value, "flash_attention_2")
        self.assertFalse(step.config["nvfp4_scaled_mm"])
        self.assertFalse(bound_controls["nvfp4_scaled_mm"].value)

    def test_train_block_swap_limits_preserve_h3_default(self):
        self.assertEqual(TrainStep._block_swap_max_for_arch("MiniMax-H3"), 48)
        self.assertEqual(TrainStep._block_swap_max_for_arch("Mage-Flow"), 10)
        self.assertEqual(TrainStep._block_swap_max_for_arch("FLUX.2"), 40)

    def test_generate_task_switch_only_changes_visibility(self):
        step = GenerateStep.__new__(GenerateStep)
        step._selected_arch = "MiniMax-H3"
        step.model_selector = SimpleNamespace(task="fl2va")
        step._h3_fl2va_inputs = SimpleNamespace(visible=False)
        step._h3_ref2va_inputs = SimpleNamespace(visible=False)
        step.first_frame_path = SimpleNamespace(value="first.png")
        step.last_frame_path = SimpleNamespace(value="last.png")
        step.reference_jsonl_path = SimpleNamespace(value="refs.jsonl")

        step._sync_minimax_h3_task_ui()
        self.assertTrue(step._h3_fl2va_inputs.visible)
        self.assertFalse(step._h3_ref2va_inputs.visible)

        step.model_selector.task = "ref2va"
        step._sync_minimax_h3_task_ui()
        self.assertFalse(step._h3_fl2va_inputs.visible)
        self.assertTrue(step._h3_ref2va_inputs.visible)
        self.assertEqual(step.first_frame_path.value, "first.png")
        self.assertEqual(step.last_frame_path.value, "last.png")
        self.assertEqual(step.reference_jsonl_path.value, "refs.jsonl")

    def test_generate_preset_keys_are_mapped_to_native_h3_controls(self):
        step = GenerateStep.__new__(GenerateStep)
        step.config = {}
        step._selected_arch = "MiniMax-H3"
        step._sync_mage_flow_generate_ui = lambda: None
        step._sync_minimax_h3_generate_ui = lambda: None
        captured = {}
        step._apply_form_state = lambda config: captured.update(config)

        step._apply_config(
            {
                "arch": "MiniMax-H3",
                "width": 768,
                "height": 1344,
                "frame_count": 124,
                "infer_steps": 30,
                "seed": 1026,
                "save_path": "./output_dir/h3.mp4",
                "attn_mode": "flash",
                "blocks_to_swap": 48,
            }
        )

        self.assertEqual(captured["h3_width"], 768)
        self.assertEqual(captured["h3_height"], 1344)
        self.assertEqual(captured["h3_frame_count"], 124)
        self.assertEqual(captured["h3_steps"], 30)
        self.assertEqual(captured["h3_seed"], 1026)
        self.assertEqual(captured["h3_output_path"], "./output_dir/h3.mp4")
        self.assertEqual(captured["h3_attn_mode"], "flash")
        self.assertEqual(captured["h3_blocks_to_swap"], 48)


if __name__ == "__main__":
    unittest.main()
