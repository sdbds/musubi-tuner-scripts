import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestModelCatalog(unittest.TestCase):

    MODULE_PATH = ROOT / "gui" / "utils" / "model_catalog.py"

    @classmethod
    def setUpClass(cls):
        src_root = ROOT / "musubi-tuner" / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        spec = importlib.util.spec_from_file_location("gui_model_catalog", cls.MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.catalog = module

        i18n_path = ROOT / "gui" / "utils" / "i18n.py"
        i18n_spec = importlib.util.spec_from_file_location("gui_i18n", i18n_path)
        i18n_module = importlib.util.module_from_spec(i18n_spec)
        assert i18n_spec.loader is not None
        i18n_spec.loader.exec_module(i18n_module)
        cls.i18n = i18n_module

    def test_flux2_versions_match_script_contract(self):
        flux2 = self.catalog.get_architecture("FLUX.2")
        self.assertEqual(
            flux2["versions"],
            ["dev", "klein-4b", "klein-base-4b", "klein-9b", "klein-base-9b"],
        )
        self.assertEqual(flux2["defaults"]["cache"]["version"], "klein-base-4b")
        self.assertEqual(flux2["defaults"]["train"]["version"], "klein-base-4b")
        self.assertEqual(flux2["defaults"]["generate"]["version"], "dev")

    def test_flux2_version_path_defaults_follow_variant(self):
        train_defaults = self.catalog.get_path_defaults("FLUX.2", "train", version="klein-base-9b")
        self.assertEqual(
            train_defaults["dit_path"],
            "./ckpts/diffusion_models/flux-2-klein-base-9b.safetensors",
        )
        self.assertEqual(
            train_defaults["text_encoder_path"],
            "./ckpts/text_encoder/qwen_3_8b.safetensors",
        )
        self.assertEqual(train_defaults["vae_path"], "./ckpts/vae/flux2-vae.safetensors")

        cache_defaults = self.catalog.get_path_defaults("FLUX.2", "cache", version="klein-9b")
        self.assertEqual(cache_defaults["text_encoder_path"], "./ckpts/text_encoder/qwen_3_8b.safetensors")
        self.assertNotIn("dit_path", cache_defaults)

        klein_4b_train = self.catalog.get_path_defaults("FLUX.2", "train", version="klein-4b")
        self.assertEqual(klein_4b_train["text_encoder_path"], "./ckpts/text_encoder/qwen_3_VL_4b.safetensors")
        klein_4b_generate = self.catalog.get_path_defaults("FLUX.2", "generate", version="klein-4b")
        self.assertEqual(klein_4b_generate["text_encoder_path"], "./ckpts/text_encoder/qwen_3_VL_4b.safetensors")

        generate_defaults = self.catalog.get_path_defaults("FLUX.2", "generate", version="dev")
        self.assertEqual(generate_defaults["dit_path"], "./ckpts/diffusion_models/flux2-dev.safetensors")
        self.assertEqual(generate_defaults["vae_path"], "./ckpts/vae/ae.safetensors")

    def test_lens_catalog_exposes_mvp_entry_points_and_paths(self):
        lens = self.catalog.get_architecture("Lens")
        self.assertEqual(lens["versions"], ["lens_bf16"])
        self.assertEqual(lens["cache_module"], "musubi_tuner.lens_cache_latents")
        self.assertEqual(lens["cache_te_module"], "musubi_tuner.lens_cache_text_encoder_outputs")
        self.assertEqual(lens["train_module"], "musubi_tuner.lens_train_network")
        self.assertEqual(lens["generate_module"], "musubi_tuner.lens_generate_image")
        self.assertTrue(lens["supports_fp8_scaled"])
        self.assertEqual(lens["default_timestep_sampling"], "flux2_shift")
        self.assertEqual(lens["pages"]["cache"]["required_paths"], ["vae", "text_encoder"])
        self.assertEqual(lens["pages"]["train"]["required_paths"], ["dit", "vae", "text_encoder"])
        self.assertEqual(lens["pages"]["generate"]["required_paths"], ["dit", "vae", "text_encoder"])
        self.assertIn("fp8_base", lens["pages"]["train"]["flags"])
        self.assertIn("fp8_scaled", lens["pages"]["train"]["flags"])

        defaults = self.catalog.get_path_defaults("Lens", "generate", version="lens_bf16")
        self.assertEqual(defaults["dit_path"], "./ckpts/diffusion_models/lens_bf16.safetensors")
        self.assertEqual(defaults["text_encoder_path"], "./ckpts/text_encoder/gpt_oss_20b_nvfp4.safetensors")
        self.assertEqual(defaults["vae_path"], "./ckpts/vae/flux2-vae.safetensors")
        self.assertNotIn("text_encoder_config_path", defaults)
        self.assertNotIn("tokenizer_path", defaults)

    def test_ideogram4_catalog_exposes_entry_points_and_bf16_qwen3vl_paths(self):
        ideogram = self.catalog.get_architecture("Ideogram-4")
        self.assertEqual(ideogram["versions"], ["fp8"])
        self.assertEqual(ideogram["cache_module"], "musubi_tuner.ideogram4_cache_latents")
        self.assertEqual(ideogram["cache_te_module"], "musubi_tuner.ideogram4_cache_text_encoder_outputs")
        self.assertEqual(ideogram["train_module"], "musubi_tuner.ideogram4_train_network")
        self.assertEqual(ideogram["generate_module"], "musubi_tuner.ideogram4_generate_image")
        self.assertTrue(ideogram["supports_text_encoder"])
        self.assertFalse(ideogram["supports_fp8_text_encoder"])
        self.assertFalse(ideogram["supports_fp8_scaled"])
        self.assertEqual(ideogram["default_timestep_sampling"], "ideogram4_shift")
        self.assertEqual(ideogram["pages"]["cache"]["required_paths"], ["vae", "text_encoder"])
        self.assertEqual(ideogram["pages"]["train"]["required_paths"], ["dit", "vae", "text_encoder"])
        self.assertEqual(ideogram["pages"]["generate"]["required_paths"], ["dit", "vae", "text_encoder"])

        defaults = self.catalog.get_path_defaults("Ideogram-4", "generate", version="fp8")
        self.assertEqual(defaults["dit_path"], "./ckpts/diffusion_models/ideogram4_fp8_scaled.safetensors")
        self.assertNotIn("unconditional_dit_path", defaults)
        self.assertEqual(defaults["text_encoder_path"], "./ckpts/text_encoder/qwen3vl_8b_bf16.safetensors")
        self.assertEqual(defaults["vae_path"], "./ckpts/vae/flux2-vae.safetensors")
        self.assertNotIn("qwen3vl_8b_fp8_scaled.safetensors", str(defaults))

    def test_wan_generate_tasks_are_filtered_by_version_family(self):
        tasks_14b = self.catalog.get_tasks_for_page("Wan2.1", "generate", version="14B")
        self.assertIn("t2v-14B", tasks_14b)
        self.assertIn("i2v-14B", tasks_14b)
        self.assertIn("t2v-14B-FC", tasks_14b)
        self.assertNotIn("t2v-A14B", tasks_14b)
        self.assertNotIn("ti2v-5B", tasks_14b)

        tasks_a14b = self.catalog.get_tasks_for_page("Wan2.1", "generate", version="A14B")
        self.assertEqual(tasks_a14b, ["t2v-A14B", "i2v-A14B"])

    def test_wan_cache_exposes_capability_flags_instead_of_task_selector(self):
        wan = self.catalog.get_architecture("Wan2.1")
        cache = wan["pages"]["cache"]
        self.assertFalse(cache.get("supports_task_selector", True))
        self.assertIn("i2v", cache["flags"])
        self.assertIn("one_frame", cache["flags"])
        self.assertIn("clip", cache["required_paths"])
        self.assertIn("t5", cache["required_paths"])

    def test_zimage_cache_exposes_soar_i2v_capability(self):
        zimage = self.catalog.get_architecture("Z-Image")
        cache = zimage["pages"]["cache"]
        self.assertIn("i2v", cache["flags"])

    def test_zimage_versions_follow_base_and_turbo_model_paths(self):
        zimage = self.catalog.get_architecture("Z-Image")
        self.assertEqual(zimage["versions"], ["base", "turbo"])
        self.assertEqual(zimage["defaults"]["cache"]["version"], "base")
        self.assertEqual(zimage["defaults"]["train"]["version"], "base")
        self.assertEqual(zimage["defaults"]["generate"]["version"], "base")

        base_train = self.catalog.get_path_defaults("Z-Image", "train", version="base")
        self.assertEqual(base_train["dit_path"], "./ckpts/diffusion_models/z_image_bf16.safetensors")
        self.assertEqual(base_train["vae_path"], "./ckpts/vae/ae.safetensors")
        self.assertEqual(base_train["text_encoder_path"], "./ckpts/text_encoder/qwen_3_4b.safetensors")

        turbo_cache = self.catalog.get_path_defaults("Z-Image", "cache", version="turbo")
        self.assertEqual(turbo_cache["text_encoder_path"], "./ckpts/text_encoder/qwen_3_VL_4b.safetensors")
        self.assertEqual(
            turbo_cache["dopsd_teacher_text_encoder_path"],
            "./ckpts/text_encoder/qwen_3_VL_4b.safetensors",
        )

        turbo_generate = self.catalog.get_path_defaults("Z-Image", "generate", version="turbo")
        self.assertEqual(turbo_generate["dit_path"], "./ckpts/diffusion_models/z_image_turbo_bf16.safetensors")
        self.assertEqual(turbo_generate["vae_path"], "./ckpts/vae/ae.safetensors")
        self.assertEqual(turbo_generate["text_encoder_path"], "./ckpts/text_encoder/qwen_3_VL_4b.safetensors")

    def test_hidream_o1_uses_single_checkpoint_without_vae(self):
        hidream = self.catalog.get_architecture("HiDream O1")
        self.assertEqual(hidream["versions"], ["full", "dev"])
        self.assertFalse(hidream["requires_vae"])
        self.assertTrue(hidream["supports_fp8_scaled"])
        self.assertEqual(hidream["default_timestep_sampling"], "uniform")

        self.assertEqual(hidream["cache_module"], "musubi_tuner.hidream_o1_cache_pixel")
        self.assertEqual(hidream["pages"]["cache"]["required_paths"], [])
        self.assertIn("fp8_te", hidream["pages"]["cache"]["flags"])
        self.assertIn("fp8_base", hidream["pages"]["train"]["flags"])
        self.assertIn("fp8_scaled", hidream["pages"]["train"]["flags"])
        self.assertEqual(hidream["pages"]["train"]["required_paths"], ["dit"])
        self.assertEqual(hidream["pages"]["generate"]["required_paths"], ["dit"])
        for page in ("cache", "train", "generate"):
            self.assertNotIn("vae", hidream["pages"][page]["required_paths"])
            self.assertNotIn("text_encoder", hidream["pages"][page]["required_paths"])

        self.assertEqual(
            self.catalog.get_path_defaults("HiDream O1", "train", version="full")["dit_path"],
            "./ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors",
        )
        self.assertEqual(
            self.catalog.get_path_defaults("HiDream O1", "generate", version="dev")["dit_path"],
            "./ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors",
        )
        self.assertNotIn("text_encoder_path", self.catalog.get_path_defaults("HiDream O1", "train", version="full"))
        self.assertNotIn("text_encoder_path", self.catalog.get_path_defaults("HiDream O1", "generate", version="dev"))
        self.assertEqual(
            self.catalog.get_path_defaults("HiDream O1", "cache", version="dev")["dit_path"],
            "./ckpts/hidream-o1-image-dev/hidream_o1_image_dev_2604_bf16.safetensors",
        )

    def test_soar_train_support_matches_submodule_entry_points(self):
        self.assertTrue(self.catalog.supports_soar_training("FLUX.2", "lora"))
        self.assertTrue(self.catalog.supports_soar_training("Qwen Image", "lora"))
        self.assertTrue(self.catalog.supports_soar_training("Z-Image", "lora"))
        self.assertTrue(self.catalog.supports_soar_training("Z-Image", "finetune"))
        self.assertFalse(self.catalog.supports_soar_training("Qwen Image", "lora", version="2509"))
        self.assertFalse(self.catalog.supports_soar_training("Qwen Image", "finetune"))
        self.assertFalse(self.catalog.supports_soar_training("HunyuanVideo", "lora"))

        for arch_name in ("FLUX.2", "Qwen Image", "Z-Image"):
            train = self.catalog.get_architecture(arch_name)["pages"]["train"]
            self.assertIn("soar", train["flags"])

    def test_soar_cfg_rollout_support_matches_submodule_limits(self):
        self.assertTrue(self.catalog.supports_soar_cfg_rollout("FLUX.2", "lora", version="klein-base-4b"))
        self.assertTrue(self.catalog.supports_soar_cfg_rollout("FLUX.2", "lora", version="klein-base-9b"))
        self.assertFalse(self.catalog.supports_soar_cfg_rollout("FLUX.2", "lora", version="dev"))
        self.assertFalse(self.catalog.supports_soar_cfg_rollout("FLUX.2", "lora", version="klein-4b"))
        self.assertFalse(self.catalog.supports_soar_cfg_rollout("FLUX.2", "lora", version="klein-9b"))
        self.assertTrue(self.catalog.supports_soar_cfg_rollout("Qwen Image", "lora", version="original"))
        self.assertFalse(self.catalog.supports_soar_cfg_rollout("Qwen Image", "lora", version="2509"))
        self.assertTrue(self.catalog.supports_soar_cfg_rollout("Z-Image", "lora", version="base"))
        self.assertFalse(self.catalog.supports_soar_cfg_rollout("Z-Image", "finetune", version="base"))

    def test_dopsd_support_matches_zimage_lora_entry_point(self):
        self.assertTrue(self.catalog.supports_dopsd_training("Z-Image", "lora"))
        self.assertTrue(self.catalog.supports_dopsd_training("Z-Image", "finetune"))
        self.assertTrue(self.catalog.supports_dopsd_training("FLUX.2", "lora", version="klein-4b"))
        self.assertTrue(self.catalog.supports_dopsd_training("FLUX.2", "lora", version="klein-9b"))
        self.assertFalse(self.catalog.supports_dopsd_training("FLUX.2", "lora", version="klein-base-4b"))
        self.assertFalse(self.catalog.supports_dopsd_training("FLUX.2", "lora", version="dev"))
        self.assertFalse(self.catalog.supports_dopsd_training("Qwen Image", "lora", version="original"))
        self.assertFalse(self.catalog.supports_dopsd_training("Qwen Image", "lora", version="edit-2509"))
        self.assertFalse(self.catalog.supports_dopsd_training("Qwen Image", "lora", version="layered"))
        self.assertFalse(self.catalog.supports_dopsd_training("Qwen Image", "finetune", version="original"))

        zimage = self.catalog.get_architecture("Z-Image")
        flux2 = self.catalog.get_architecture("FLUX.2")
        qwen = self.catalog.get_architecture("Qwen Image")
        self.assertIn("dopsd", zimage["pages"]["cache"]["flags"])
        self.assertIn("dopsd", zimage["pages"]["train"]["flags"])
        self.assertIn("dopsd", flux2["pages"]["cache"]["flags"])
        self.assertIn("dopsd", flux2["pages"]["train"]["flags"])
        self.assertNotIn("dopsd", qwen["pages"]["cache"]["flags"])
        self.assertNotIn("dopsd", qwen["pages"]["train"]["flags"])

    def test_catalog_modules_resolve_to_local_entry_points(self):
        for arch_name, arch_info in self.catalog.get_all_architectures().items():
            for key in ("cache_module", "cache_te_module", "train_module", "finetune_module", "generate_module"):
                module_name = arch_info.get(key)
                if not module_name:
                    continue
                with self.subTest(arch=arch_name, key=key, module=module_name):
                    self.assertIsNotNone(importlib.util.find_spec(module_name))

    def test_only_finetune_entry_points_expose_finetune_mode(self):
        self.assertEqual(self.catalog.get_train_modes("Qwen Image"), {"lora": "LoRA", "finetune": "Fine-tune"})
        self.assertEqual(self.catalog.get_train_modes("Z-Image"), {"lora": "LoRA", "finetune": "Fine-tune"})
        self.assertEqual(self.catalog.get_train_modes("FLUX.2"), {"lora": "LoRA"})
        self.assertEqual(self.catalog.get_default_train_mode("Qwen Image"), "lora")

    def test_longcat_generate_is_not_native_without_entry_point(self):
        longcat = self.catalog.get_architecture("Long-CAT")
        self.assertIsNone(longcat["generate_module"])
        self.assertFalse(longcat["pages"]["generate"].get("native_supported", True))

    def test_home_model_translations_cover_catalog(self):
        catalog_ids = {arch_info["id"] for arch_info in self.catalog.get_all_architectures().values()}
        for lang, values in self.i18n.TRANSLATIONS.items():
            translations = values.get("model_architecture_list", {})
            missing = sorted(catalog_ids - set(translations))
            with self.subTest(lang=lang):
                self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
