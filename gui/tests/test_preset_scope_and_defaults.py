import ast
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
        cls.model_selector_text = (GUI_ROOT / "components" / "model_selector.py").read_text(encoding="utf-8")
        cls.advanced_inputs_text = (GUI_ROOT / "components" / "advanced_inputs.py").read_text(encoding="utf-8")
        cls.cache_step_text = (GUI_ROOT / "wizard" / "step2_cache.py").read_text(encoding="utf-8")
        cls.train_step_text = (GUI_ROOT / "wizard" / "step3_train.py").read_text(encoding="utf-8")
        cls.generate_step_text = (GUI_ROOT / "wizard" / "step4_generate.py").read_text(encoding="utf-8")
        cls.ideogram4_sampler_text = (
            ROOT / "musubi-tuner" / "src" / "musubi_tuner" / "ideogram4" / "sampler_configs.py"
        ).read_text(encoding="utf-8")
        cls.theme_text = (GUI_ROOT / "theme.py").read_text(encoding="utf-8")

    @staticmethod
    def _list_constant(source: str, name: str) -> list[str]:
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
        raise AssertionError(f"{name} constant not found")

    @staticmethod
    def _dict_keys_constant(source: str, name: str) -> list[str]:
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            if isinstance(node.target, ast.Name) and node.target.id == name and isinstance(node.value, ast.Dict):
                return [key.value for key in node.value.keys if isinstance(key, ast.Constant)]
        raise AssertionError(f"{name} constant not found")

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

    def test_builtin_train_presets_default_attention_mode_to_flash(self):
        manager = self.config_manager_module.ConfigManager()

        for name in manager.list_configs("train"):
            preset = manager.load_config("train", name)
            if not preset or "attn_mode" not in preset:
                continue
            with self.subTest(preset=name):
                self.assertEqual(preset["attn_mode"], "flash")

    def test_lens_presets_are_available_for_cache_train_and_generate(self):
        manager = self.config_manager_module.ConfigManager()

        for scope in ("cache", "train", "generate"):
            with self.subTest(scope=scope):
                self.assertIn("lens", manager.list_configs(scope))
                preset = manager.load_config(scope, "lens")
                self.assertEqual(preset["arch"], "Lens")
                self.assertEqual(preset["version"], "lens_bf16")
                self.assertEqual(
                    preset["text_encoder_path"],
                    "./ckpts/text_encoder/gpt_oss_20b_nvfp4.safetensors",
                )
                self.assertEqual(preset["vae_path"], "./ckpts/vae/flux2-vae.safetensors")
                self.assertNotIn("text_encoder_config_path", preset)
                self.assertNotIn("tokenizer_path", preset)

        train = manager.load_config("train", "lens")
        self.assertEqual(train["optimizer_type"], "AdamW_adv")
        self.assertEqual(train["attn_mode"], "flash")
        self.assertFalse(train["split_attn"])
        self.assertEqual(train["timestep_sampling"], "flux2_shift")
        self.assertEqual(train["network_dim"], 16)
        self.assertEqual(train["network_alpha"], 16)

        train_entries = {entry["name"]: entry for entry in manager.list_config_entries("train")}
        self.assertEqual(train_entries["lens_low_vram"]["label"], "Lens LoRA Low VRAM")
        self.assertEqual(train_entries["lens_finetune"]["label"], "Lens 微调")
        self.assertEqual(train_entries["lens_finetune_low_vram"]["label"], "Lens 微调 Low VRAM")

        low_vram = manager.load_config("train", "lens_low_vram")
        self.assertEqual(low_vram["arch"], "Lens")
        self.assertEqual(low_vram["version"], "lens_bf16")
        self.assertEqual(low_vram["dit_path"], "./ckpts/diffusion_models/lens_bf16.safetensors")
        self.assertEqual(low_vram["text_encoder_path"], "./ckpts/text_encoder/gpt_oss_20b_nvfp4.safetensors")
        self.assertEqual(low_vram["vae_path"], "./ckpts/vae/flux2-vae.safetensors")
        self.assertEqual(low_vram["blocks_to_swap"], 8)
        self.assertTrue(low_vram["use_pinned_memory"])
        self.assertEqual(low_vram["timestep_sampling"], "flux2_shift")
        self.assertEqual(low_vram["optimizer_type"], "AdamW_adv")

        finetune = manager.load_config("train", "lens_finetune")
        self.assertEqual(finetune["arch"], "Lens")
        self.assertEqual(finetune["train_mode"], "finetune")
        self.assertEqual(finetune["dit_path"], "./ckpts/diffusion_models/lens_bf16.safetensors")
        self.assertEqual(finetune["text_encoder_path"], "./ckpts/text_encoder/gpt_oss_20b_nvfp4.safetensors")
        self.assertEqual(finetune["vae_path"], "./ckpts/vae/flux2-vae.safetensors")
        self.assertEqual(finetune["attn_mode"], "flash")
        self.assertFalse(finetune["fp8_base"])
        self.assertFalse(finetune["fp8_scaled"])
        self.assertNotIn("network_weights", finetune)
        self.assertNotIn("network_dim", finetune)
        self.assertNotIn("enable_lycoris", finetune)

        finetune_low_vram = manager.load_config("train", "lens_finetune_low_vram")
        self.assertEqual(finetune_low_vram["arch"], "Lens")
        self.assertEqual(finetune_low_vram["train_mode"], "finetune")
        self.assertTrue(finetune_low_vram["full_bf16"])
        self.assertTrue(finetune_low_vram["fused_backward_pass"])
        self.assertTrue(finetune_low_vram["mem_eff_save"])
        self.assertEqual(finetune_low_vram["blocks_to_swap"], 8)
        self.assertTrue(finetune_low_vram["use_pinned_memory"])

        generate = manager.load_config("generate", "lens")
        self.assertEqual(generate["save_path"], "./output_dir/lens.png")
        self.assertEqual(generate["infer_steps"], 20)
        self.assertEqual(generate["guidance_scale"], 5.0)

    def test_ideogram4_presets_are_available_for_cache_train_and_generate(self):
        manager = self.config_manager_module.ConfigManager()

        expected_text_encoder = "./ckpts/text_encoder/qwen3vl_8b_bf16.safetensors"
        for scope in ("cache", "train", "generate"):
            with self.subTest(scope=scope):
                self.assertIn("ideogram4", manager.list_configs(scope))
                preset = manager.load_config(scope, "ideogram4")
                self.assertEqual(preset["arch"], "Ideogram-4")
                self.assertEqual(preset["version"], "fp8")
                self.assertEqual(preset["text_encoder_path"], expected_text_encoder)
                self.assertEqual(preset["vae_path"], "./ckpts/vae/flux2-vae.safetensors")
                self.assertNotIn("unconditional_dit_path", preset)
                self.assertNotIn("qwen3vl_8b_fp8_scaled.safetensors", str(preset))

        train = manager.load_config("train", "ideogram4")
        self.assertEqual(train["dit_path"], "./ckpts/diffusion_models/ideogram4_fp8_scaled.safetensors")
        self.assertNotIn("unconditional_dit_path", train)
        self.assertEqual(train["timestep_sampling"], "ideogram4_shift")
        self.assertEqual(train["attn_mode"], "flash")
        self.assertEqual(train["lr_scheduler"], "cosine_with_min_lr")
        self.assertEqual(train["lr_scheduler_min_lr_ratio"], 0.1)
        self.assertNotIn("ideogram4_timestep_mu", train)
        self.assertNotIn("ideogram4_timestep_std", train)
        self.assertEqual(train["initial_sigma"], 1.004)
        self.assertEqual(train["network_dim"], 64)
        self.assertEqual(train["network_alpha"], 32)
        self.assertTrue(train["enable_sample"])
        self.assertEqual(train["sample_at_first"], 1)
        self.assertEqual(train["sample_prompts"], "./toml/qinglong_ideogram4.txt")

        generate = manager.load_config("generate", "ideogram4")
        self.assertEqual(generate["save_path"], "./output_dir/ideogram4.png")
        self.assertEqual(generate["sampler_preset"], "V4_DEFAULT_20")
        self.assertEqual(generate["initial_sigma"], 1.004)
        self.assertNotIn("infer_steps", generate)
        self.assertNotIn("guidance_scale", generate)

    def test_ideogram4_gui_sampler_presets_match_backend_presets(self):
        expected = self._dict_keys_constant(self.ideogram4_sampler_text, "PRESETS")

        self.assertEqual(self._list_constant(self.train_step_text, "IDEOGRAM4_SAMPLER_PRESETS"), expected)
        self.assertEqual(self._list_constant(self.generate_step_text, "IDEOGRAM4_SAMPLER_PRESETS"), expected)
        self.assertIn("ideogram4_shift", self._list_constant(self.train_step_text, "TIMESTEP_SAMPLING_METHODS"))
        self.assertNotIn("ideogram4_timestep_mu", self.train_step_text)
        self.assertNotIn("ideogram4_timestep_std", self.train_step_text)

    def test_hidream_o1_presets_use_single_checkpoint_without_vae(self):
        manager = self.config_manager_module.ConfigManager()

        for scope in ("cache", "train", "generate"):
            with self.subTest(scope=scope):
                preset = manager.load_config(scope, "hidream_o1")
                self.assertEqual(preset["arch"], "HiDream O1")
                self.assertEqual(preset["version"], "full")
                self.assertNotIn("text_encoder_path", preset)
                self.assertEqual(
                    preset["dit_path"],
                    "./ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors",
                )
                self.assertNotIn("vae_path", preset)

        generate = manager.load_config("generate", "hidream_o1")
        self.assertEqual(generate["_label"], "HiDream O1 Full T2I")
        self.assertEqual(generate["save_path"], "./output_dir/hidream_o1.png")
        self.assertEqual(generate["infer_steps"], 50)
        self.assertEqual(generate["guidance_scale"], 5.0)
        self.assertEqual(generate["flow_shift"], 3.0)
        self.assertEqual(generate["attn_mode"], "flash")
        self.assertEqual(generate["blocks_to_swap"], 8)
        self.assertEqual(generate["noise_scale_start"], 8.0)
        self.assertEqual(generate["noise_scale_end"], 8.0)
        self.assertEqual(generate["noise_clip_std"], 0.0)

        dev_flash = manager.load_config("generate", "hidream_o1_dev_flash")
        self.assertEqual(dev_flash["_label"], "HiDream O1 Dev Flash T2I")
        self.assertEqual(dev_flash["arch"], "HiDream O1")
        self.assertEqual(dev_flash["version"], "dev")
        self.assertEqual(dev_flash["dit_path"], "./ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors")
        self.assertNotIn("text_encoder_path", dev_flash)
        self.assertNotIn("vae_path", dev_flash)
        self.assertEqual(dev_flash["infer_steps"], 28)
        self.assertEqual(dev_flash["guidance_scale"], 0.0)
        self.assertEqual(dev_flash["flow_shift"], 1.0)
        self.assertEqual(dev_flash["noise_scale_start"], 7.5)
        self.assertEqual(dev_flash["noise_scale_end"], 7.5)
        self.assertEqual(dev_flash["noise_clip_std"], 2.5)
        self.assertEqual(dev_flash["editing_scheduler"], "flash")
        self.assertEqual(dev_flash["layout_bboxes"], "")

        dev_edit_flow = manager.load_config("generate", "hidream_o1_dev_edit_flow")
        self.assertEqual(dev_edit_flow["_label"], "HiDream O1 Dev Edit Flow I2I")
        self.assertEqual(dev_edit_flow["arch"], "HiDream O1")
        self.assertEqual(dev_edit_flow["version"], "dev")
        self.assertEqual(dev_edit_flow["dit_path"], "./ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors")
        self.assertNotIn("text_encoder_path", dev_edit_flow)
        self.assertNotIn("vae_path", dev_edit_flow)
        self.assertEqual(dev_edit_flow["infer_steps"], 28)
        self.assertEqual(dev_edit_flow["guidance_scale"], 0.0)
        self.assertEqual(dev_edit_flow["flow_shift"], 1.0)
        self.assertEqual(dev_edit_flow["editing_scheduler"], "flow_match")
        self.assertEqual(dev_edit_flow["ref_images"], "")
        self.assertEqual(dev_edit_flow["layout_bboxes"], "")
        self.assertNotIn("noise_scale_start", dev_edit_flow)
        self.assertNotIn("noise_scale_end", dev_edit_flow)
        self.assertNotIn("noise_clip_std", dev_edit_flow)

        train = manager.load_config("train", "hidream_o1")
        self.assertEqual(train["timestep_sampling"], "uniform")
        self.assertEqual(train["blocks_to_swap"], 8)
        self.assertEqual(train["guidance_scale"], 5.0)
        self.assertEqual(train["discrete_flow_shift"], 3.0)
        self.assertTrue(train["enable_sample"])
        self.assertEqual(train["sample_at_first"], 1)
        self.assertEqual(train["sample_prompts"], "./toml/qinglong_hidream_o1.txt")
        self.assertEqual(train["sample_every_n_epochs"], 1)
        self.assertEqual(train["sample_every_n_steps"], 0)
        self.assertEqual(train["noise_scale_start"], 8.0)
        self.assertEqual(train["noise_scale_end"], 8.0)
        self.assertEqual(train["noise_clip_std"], 0.0)
        self.assertEqual(train["dino_loss_weight"], 0.0)
        self.assertEqual(train["dino_loss_backend"], "vit")
        self.assertFalse(train["fp8_scaled"])

        train_entries = {entry["name"]: entry for entry in manager.list_config_entries("train")}
        self.assertIn("hidream_o1_dev", train_entries)
        self.assertEqual(train_entries["hidream_o1_dev"]["label"], "HiDream O1 Dev 2604 LoRA")

        dev_train = manager.load_config("train", "hidream_o1_dev")
        self.assertEqual(dev_train["_label"], "HiDream O1 Dev 2604 LoRA")
        self.assertEqual(dev_train["arch"], "HiDream O1")
        self.assertEqual(dev_train["version"], "dev")
        self.assertEqual(
            dev_train["dit_path"],
            "./ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors",
        )
        self.assertNotIn("text_encoder_path", dev_train)
        self.assertNotIn("vae_path", dev_train)
        self.assertEqual(dev_train["guidance_scale"], 0.0)
        self.assertEqual(dev_train["discrete_flow_shift"], 1.0)
        self.assertEqual(dev_train["noise_scale_start"], 7.5)
        self.assertEqual(dev_train["noise_scale_end"], 7.5)
        self.assertEqual(dev_train["noise_clip_std"], 2.5)
        self.assertEqual(dev_train["output_name"], "hidream_o1_dev_2604_lora_qinglong")
        self.assertTrue(dev_train["enable_sample"])
        self.assertEqual(dev_train["sample_prompts"], "./toml/qinglong_hidream_o1.txt")

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
        self.assertEqual(cache["version"], "turbo")
        self.assertEqual(cache["text_encoder_path"], "./ckpts/text_encoder/qwen_3_VL_4b.safetensors")
        self.assertTrue(cache["dopsd_cache_teacher_outputs"])
        self.assertEqual(
            cache["dopsd_teacher_text_encoder_path"],
            "./ckpts/text_encoder/qwen_3_VL_4b.safetensors",
        )
        self.assertTrue(cache["dopsd_teacher_already_reweighted"])
        self.assertFalse(cache["dopsd_teacher_allow_raw_vlm"])
        self.assertNotIn("dopsd_teacher_llm_reweight_source_path", cache)
        self.assertNotIn("dopsd_teacher_embed_key", cache)
        self.assertNotIn("dopsd_teacher_processor_path", cache)

        train = manager.load_config("train", "zimage_dopsd")
        self.assertEqual(train["arch"], "Z-Image")
        self.assertEqual(train["train_mode"], "lora")
        self.assertEqual(train["version"], "turbo")
        self.assertEqual(train["dit_path"], "./ckpts/diffusion_models/z_image_turbo_bf16.safetensors")
        self.assertEqual(train["text_encoder_path"], "./ckpts/text_encoder/qwen_3_VL_4b.safetensors")
        self.assertEqual(train["sample_prompts"], "./toml/qinglong_z_image_turbo.txt")
        self.assertTrue(train["dopsd"])
        self.assertEqual(train["dopsd_num_sampling_steps"], 8)
        self.assertEqual(train["dopsd_ema_decay"], 0.9999)
        self.assertNotIn("dopsd_teacher_embed_key", train)

        finetune = manager.load_config("train", "zimage_dopsd_finetune")
        self.assertEqual(finetune["arch"], "Z-Image")
        self.assertEqual(finetune["train_mode"], "finetune")
        self.assertEqual(finetune["version"], "turbo")
        self.assertEqual(finetune["dit_path"], "./ckpts/diffusion_models/z_image_turbo_bf16.safetensors")
        self.assertEqual(finetune["text_encoder_path"], "./ckpts/text_encoder/qwen_3_VL_4b.safetensors")
        self.assertEqual(finetune["sample_prompts"], "./toml/qinglong_z_image_turbo.txt")
        self.assertTrue(finetune["dopsd"])
        self.assertFalse(finetune["fused_backward_pass"])
        self.assertEqual(finetune["dopsd_num_sampling_steps"], 8)
        self.assertEqual(finetune["dopsd_full_ema_device"], "auto")

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

    def test_select_rows_align_actions_to_select_control(self):
        self.assertIn("w-full items-end gap-3 q-mt-sm", self.preset_manager_text)
        self.assertIn('classes("w-full items-end gap-4")', self.model_selector_text)

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
        self.assertIn('if arch_name == "HiDream O1":', self.cache_step_text)
        self.assertIn("HiDream O1 checkpoint (optional for text embedding cache)", self.cache_step_text)
        self.assertIn("selection_type='file_or_dir'", self.cache_step_text)
        self.assertIn('def _sync_vae_model_card', self.cache_step_text)
        self.assertIn('def _sync_vae_path_ui', self.train_step_text)
        self.assertIn('if arch_name == "HiDream O1":\n            return', self.train_step_text)
        self.assertIn('if arch_name == "HiDream O1":\n            return', self.generate_step_text)
        self.assertIn("noise_scale_start", self.train_step_text)
        self.assertIn("noise_scale_end", self.train_step_text)
        self.assertIn("noise_clip_std", self.train_step_text)
        self.assertIn("editing_scheduler", self.generate_step_text)
        self.assertIn("layout_bboxes", self.generate_step_text)
        self.assertIn("dino_loss_weight", self.train_step_text)
        self.assertIn("_sync_hidream_train_options_ui", self.train_step_text)
        self.assertIn("HIDREAM_TRAIN_VERSION_DEFAULTS", self.train_step_text)
        self.assertIn("'dev': {\n        'guidance_scale': 0.0", self.train_step_text)
        self.assertIn("'noise_clip_std': 2.5", self.train_step_text)
        self.assertIn("self._apply_hidream_train_version_defaults(arch_name, version)", self.train_step_text)
        self.assertIn('elif arch_name == "Ideogram-4":', self.cache_step_text)
        self.assertNotIn('self._set_control("unconditional_dit_path"', self.train_step_text)
        self.assertNotIn('self._set_control("unconditional_dit_path"', self.generate_step_text)
        self.assertIn("qwen3vl_8b_bf16.safetensors", self.cache_step_text)

    def test_cache_and_train_steps_expose_dopsd_controls(self):
        self.assertIn("dopsd_cache_teacher_outputs", self.cache_step_text)
        self.assertIn("dopsd_teacher_text_encoder_path", self.cache_step_text)
        self.assertIn('if arch_name == "Z-Image":', self.cache_step_text)
        self.assertNotIn('if arch_name in {"FLUX.2", "Z-Image"}:', self.cache_step_text)
        self.assertIn('self._set_control("dopsd_cache_teacher_outputs", toggle_switch(', self.cache_step_text)
        self.assertIn('self._set_control("dopsd_teacher_already_reweighted", toggle_switch(', self.cache_step_text)
        self.assertIn('self._set_control("dopsd_teacher_allow_raw_vlm", toggle_switch(', self.cache_step_text)
        self.assertLess(
            self.cache_step_text.index('self._set_control("dopsd_teacher_dtype"'),
            self.cache_step_text.index('self._set_control("dopsd_teacher_text_encoder_path"'),
        )
        self.assertIn("placeholder='Qwen3-VL weights file'", self.cache_step_text)
        self.assertIn("selection_type='file',\n                        file_filter='*.safetensors *.pt *.pth *.bin'", self.cache_step_text)
        self.assertIn("model_catalog.supports_dopsd_cache", self.cache_step_text)
        self.assertNotIn("dopsd_teacher_llm_reweight_source_path", self.cache_step_text)
        self.assertNotIn("dopsd_teacher_processor_path", self.cache_step_text)
        self.assertNotIn("dopsd_teacher_hidden_state_index", self.cache_step_text)
        self.assertIn("self.config.setdefault('dopsd_num_sampling_steps', 8)", self.train_step_text)
        self.assertIn("self.config.setdefault('dopsd_full_ema_device', 'auto')", self.train_step_text)
        self.assertIn("self._sync_dopsd_options_ui()", self.train_step_text)
        self.assertNotIn("dopsd_teacher_embed_key", self.train_step_text)

    def test_cache_step_exposes_default_enabled_cache_stage_controls(self):
        self.assertIn("self.config.setdefault('cache_latents_enabled', True)", self.cache_step_text)
        self.assertIn("self.config.setdefault('cache_text_encoder_enabled', True)", self.cache_step_text)
        self.assertIn("t('cache_latents_enabled'", self.cache_step_text)
        self.assertIn("t('cache_text_encoder_enabled'", self.cache_step_text)

    def test_form_controls_use_consistent_vertical_rhythm(self):
        self.assertIn("min-height: 56px", self.advanced_inputs_text)
        self.assertIn("modern-select force-light-bg", self.advanced_inputs_text)
        self.assertIn('value_ref.setdefault("_bound_controls", {})[value_key] = control', self.advanced_inputs_text)
        self.assertIn(".q-select:not(.modern-select):not(.lang-selector).q-field--labeled .q-field__native", self.theme_text)
        self.assertIn(".q-select:not(.modern-select):not(.lang-selector).q-field--labeled .q-field__label", self.theme_text)
        self.assertIn(".q-input:not(.slider-edit-input) .q-field__control-container", self.theme_text)
        self.assertIn(".q-input:not(.slider-edit-input).q-field--labeled:not(.q-field--float) .q-field__label", self.theme_text)
        self.assertNotIn(".q-input:not(.slider-edit-input).q-field--labeled.q-field--float .q-field__label", self.theme_text)
        self.assertIn("text-align: center", self.theme_text)
        self.assertIn(".editable-slider > .row:first-child", self.theme_text)

    def test_train_guidance_scale_slider_is_registered_for_preset_sync(self):
        self.assertIn('self._set_control(\n                    "guidance_scale"', self.train_step_text)

    def test_train_step_exposes_h2d_only_block_swap_controls(self):
        self.assertIn("self.config.setdefault('block_swap_h2d_only', False)", self.train_step_text)
        self.assertIn("self.config.setdefault('block_swap_ring_size', 2)", self.train_step_text)
        self.assertIn("toggle_switch(t('block_swap_h2d_only'", self.train_step_text)
        self.assertIn("editable_slider(t('block_swap_ring_size'", self.train_step_text)
        self.assertIn("_h2d_block_swap_row.visible = is_lora", self.train_step_text)

    def test_train_sample_step_interval_has_no_gui_maximum(self):
        self.assertIn("self.sample_every_n_steps = ui.input", self.train_step_text)
        self.assertIn('props(\'type="number" min="0" step="1"\')', self.train_step_text)
        self.assertNotIn("'sample_every_n_steps', min_val=0, max_val=1000", self.train_step_text)

    def test_generate_compile_args_stays_registered_across_arch_changes(self):
        dynamic_fields = self.generate_step_text.split("self._dynamic_field_names = {", 1)[1].split("}", 1)[0]

        self.assertIn("self.compile_args = ui.input", self.generate_step_text)
        self.assertNotIn("'compile_args'", dynamic_fields)


if __name__ == "__main__":
    unittest.main()
