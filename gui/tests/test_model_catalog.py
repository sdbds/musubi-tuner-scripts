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

        generate_defaults = self.catalog.get_path_defaults("FLUX.2", "generate", version="dev")
        self.assertEqual(generate_defaults["dit_path"], "./ckpts/diffusion_models/flux2-dev.safetensors")
        self.assertEqual(generate_defaults["vae_path"], "./ckpts/vae/ae.safetensors")

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

        turbo_generate = self.catalog.get_path_defaults("Z-Image", "generate", version="turbo")
        self.assertEqual(turbo_generate["dit_path"], "./ckpts/diffusion_models/z_image_turbo_bf16.safetensors")
        self.assertEqual(turbo_generate["vae_path"], "./ckpts/vae/ae.safetensors")

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


if __name__ == "__main__":
    unittest.main()
