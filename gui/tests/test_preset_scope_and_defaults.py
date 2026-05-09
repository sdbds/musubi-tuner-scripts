import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestPresetScopeAndDefaults(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.config_manager_module = importlib.import_module("utils.config_manager")
        cls.preset_manager_text = (GUI_ROOT / "components" / "preset_manager.py").read_text(encoding="utf-8")
        cls.cache_step_text = (GUI_ROOT / "wizard" / "step2_cache.py").read_text(encoding="utf-8")
        cls.train_step_text = (GUI_ROOT / "wizard" / "step3_train.py").read_text(encoding="utf-8")

    def test_builtin_presets_are_split_by_page_scope(self):
        manager = self.config_manager_module.ConfigManager()

        self.assertIn("flux2", manager.list_configs("cache"))
        self.assertIn("flux2", manager.list_configs("train"))
        self.assertIn("flux2", manager.list_configs("generate"))
        self.assertNotEqual(manager.load_config("cache", "flux2"), manager.load_config("generate", "flux2"))

    def test_flux2_train_preset_uses_script_default_model_paths(self):
        manager = self.config_manager_module.ConfigManager()

        preset = manager.load_config("train", "flux2")

        self.assertEqual(preset["arch"], "FLUX.2")
        self.assertEqual(preset["version"], "klein-base-4b")
        self.assertEqual(preset["dataset_config"], "./toml/qinglong-qwen-image-datasets.toml")
        self.assertEqual(preset["dit_path"], "./ckpts/diffusion_models/flux-2-klein-base-4b.safetensors")
        self.assertEqual(preset["vae_path"], "./ckpts/vae/flux2-vae.safetensors")
        self.assertEqual(preset["text_encoder_path"], "./ckpts/text_encoder/qwen_3_4b.safetensors")

    def test_qwen_generate_preset_uses_script_default_model_paths(self):
        manager = self.config_manager_module.ConfigManager()

        preset = manager.load_config("generate", "qwen_image")

        self.assertEqual(preset["arch"], "Qwen Image")
        self.assertEqual(preset["dit_path"], "./ckpts/diffusion_models/qwen_image_edit_bf16.safetensors")
        self.assertEqual(preset["vae_path"], "./ckpts/vae/qwen_image_vae.safetensors")
        self.assertEqual(preset["text_encoder_vl_path"], "./ckpts/text_encoder/qwen_2.5_vl_7b.safetensors")
        self.assertEqual(preset["image_encoder_path"], "./ckpts/framepack/sigclip_vision_patch14_384.safetensors")

    def test_zimage_presets_use_base_model_paths(self):
        manager = self.config_manager_module.ConfigManager()

        train = manager.load_config("train", "zimage")
        self.assertEqual(train["arch"], "Z-Image")
        self.assertEqual(train["version"], "base")
        self.assertEqual(train["dit_path"], "./ckpts/diffusion_models/z_image_bf16.safetensors")
        self.assertEqual(train["vae_path"], "./ckpts/vae/ae.safetensors")
        self.assertEqual(train["text_encoder_path"], "./ckpts/text_encoder/qwen_3_4b.safetensors")

        generate = manager.load_config("generate", "zimage")
        self.assertEqual(generate["arch"], "Z-Image")
        self.assertEqual(generate["version"], "base")
        self.assertEqual(generate["dit_path"], "./ckpts/diffusion_models/z_image_bf16.safetensors")
        self.assertEqual(generate["vae_path"], "./ckpts/vae/ae.safetensors")

    def test_zimage_dopsd_presets_are_available_for_cache_and_train(self):
        manager = self.config_manager_module.ConfigManager()

        entries = {
            scope: {entry["name"]: entry for entry in manager.list_config_entries(scope)}
            for scope in ("cache", "train")
        }
        self.assertEqual(entries["cache"]["zimage_dopsd"]["label"], "Z-Image D-OPSD Teacher Cache")
        self.assertEqual(entries["train"]["zimage_dopsd"]["label"], "Z-Image D-OPSD LoRA")
        self.assertEqual(entries["train"]["zimage_dopsd_finetune"]["label"], "Z-Image D-OPSD 微调")

        cache = manager.load_config("cache", "zimage_dopsd")
        self.assertEqual(cache["arch"], "Z-Image")
        self.assertTrue(cache["dopsd_cache_teacher_outputs"])
        self.assertEqual(cache["dopsd_teacher_text_encoder_path"], "./ckpts/text_encoder/qwen3-vl-4b")
        self.assertFalse(cache["dopsd_teacher_already_reweighted"])
        self.assertFalse(cache["dopsd_teacher_allow_raw_vlm"])
        self.assertNotIn("dopsd_teacher_llm_reweight_source_path", cache)
        self.assertNotIn("dopsd_teacher_embed_key", cache)
        self.assertNotIn("dopsd_teacher_processor_path", cache)

        train = manager.load_config("train", "zimage_dopsd")
        self.assertEqual(train["arch"], "Z-Image")
        self.assertEqual(train["train_mode"], "lora")
        self.assertTrue(train["dopsd"])
        self.assertEqual(train["dopsd_num_sampling_steps"], 8)
        self.assertEqual(train["dopsd_ema_decay"], 0.9999)
        self.assertNotIn("dopsd_teacher_embed_key", train)

        finetune = manager.load_config("train", "zimage_dopsd_finetune")
        self.assertEqual(finetune["arch"], "Z-Image")
        self.assertEqual(finetune["train_mode"], "finetune")
        self.assertTrue(finetune["dopsd"])
        self.assertFalse(finetune["fused_backward_pass"])
        self.assertEqual(finetune["dopsd_num_sampling_steps"], 8)

    def test_train_finetune_presets_use_finetune_mode_without_lora_network_keys(self):
        manager = self.config_manager_module.ConfigManager()

        entries = {entry["name"]: entry for entry in manager.list_config_entries("train")}
        self.assertEqual(entries["qwen_image_finetune"]["label"], "Qwen Image 微调")
        self.assertEqual(entries["zimage_finetune"]["label"], "Z-Image 微调")

        qwen = manager.load_config("train", "qwen_image_finetune")
        self.assertEqual(qwen["arch"], "Qwen Image")
        self.assertEqual(qwen["train_mode"], "finetune")
        self.assertEqual(qwen["learning_rate"], "1e-5")
        self.assertNotIn("network_weights", qwen)
        self.assertNotIn("network_dim", qwen)
        self.assertNotIn("enable_lycoris", qwen)

        zimage = manager.load_config("train", "zimage_finetune")
        self.assertEqual(zimage["arch"], "Z-Image")
        self.assertEqual(zimage["train_mode"], "finetune")
        self.assertEqual(zimage["optimizer_type"], "BCOS")
        self.assertNotIn("network_weights", zimage)
        self.assertNotIn("network_dim", zimage)

    def test_preset_manager_no_longer_renders_redundant_apply_button(self):
        self.assertNotIn("apply_btn = ui.button", self.preset_manager_text)
        self.assertIn("on_change=self._on_preset_change", self.preset_manager_text)

    def test_cache_and_train_steps_expose_script_required_model_paths_for_complex_arches(self):
        self.assertIn(
            'elif arch_name == "FLUX Kontext":\n'
            '                self._set_control("te1_path"',
            self.cache_step_text,
        )
        self.assertIn(
            'elif arch_name == "Wan2.1":\n'
            '                self._set_control("te1_path"',
            self.cache_step_text,
        )
        self.assertIn(
            'elif arch_name in ("Qwen Image", "Long-CAT"):\n'
            '                self._set_control("te1_path"',
            self.train_step_text,
        )
        self.assertIn(
            'self._set_control("image_encoder_path"',
            self.train_step_text,
        )

    def test_cache_and_train_steps_expose_dopsd_controls(self):
        self.assertIn("dopsd_cache_teacher_outputs", self.cache_step_text)
        self.assertIn("dopsd_teacher_text_encoder_path", self.cache_step_text)
        self.assertIn("model_catalog.supports_dopsd_cache", self.cache_step_text)
        self.assertNotIn("dopsd_teacher_llm_reweight_source_path", self.cache_step_text)
        self.assertNotIn("dopsd_teacher_processor_path", self.cache_step_text)
        self.assertNotIn("dopsd_teacher_hidden_state_index", self.cache_step_text)
        self.assertIn("self.config.setdefault('dopsd_num_sampling_steps', 8)", self.train_step_text)
        self.assertIn("self._sync_dopsd_options_ui()", self.train_step_text)
        self.assertNotIn("dopsd_teacher_embed_key", self.train_step_text)

    def test_train_guidance_scale_slider_is_registered_for_preset_sync(self):
        self.assertIn('self._set_control(\n                    "guidance_scale"', self.train_step_text)


if __name__ == "__main__":
    unittest.main()
