import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import toml
from nicegui import ui


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from wizard.step2_cache import CacheStep  # noqa: E402
from wizard.step3_train import TrainStep  # noqa: E402
from wizard.step4_generate import GenerateStep  # noqa: E402
from components.model_selector import create_model_selector, get_arch_info  # noqa: E402
from utils.command_builder import build_cache_jobs, build_generate_job, build_train_job  # noqa: E402
from utils.i18n import TRANSLATIONS, get_i18n  # noqa: E402
from utils import model_catalog  # noqa: E402


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
    "h3_uncond_output",
    "h3_uncond_output_tooltip",
    "h3_uncond_text",
    "h3_uncond_text_tooltip",
    "h3_one_frame_image_mode",
    "h3_one_frame_image_mode_tooltip",
    "h3_dit_dtype",
    "h3_shift_video",
    "h3_shift_audio",
    "h3_visual_cond_clean",
    "h3_audio_cond_clean",
    "h3_audio_loss_weight",
    "h3_best_of_k",
    "h3_best_of_k_tooltip",
    "h3_best_of_k_stream",
    "h3_best_of_k_stream_tooltip",
    "h3_best_of_k_stream_video",
    "h3_best_of_k_stream_audio",
    "h3_convrot_backward",
    "h3_bf16_backward",
    "h3_int8_backward",
    "h3_video_only",
    "h3_video_only_tooltip",
    "h3_quantize_convrot_int8",
    "h3_convrot_int8_tooltip",
    "h3_guidance_loss_scale",
    "h3_guidance_loss_scale_tooltip",
    "h3_guidance_loss_scale_audio",
    "h3_guidance_loss_scale_audio_tooltip",
    "h3_guidance_loss_sigma_min",
    "h3_guidance_loss_sigma_min_tooltip",
    "h3_guidance_loss_uncond_cache",
    "h3_guidance_loss_uncond_cache_tooltip",
    "h3_prune_adaln",
    "h3_prune_adaln_tooltip",
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

H3_PROJECT_CONFIG = {
    "dataset": {
        "general": {"resolution": [768, 1344], "batch_size": 1},
        "datasets": [
            {
                "video_directory": "videos",
                "cache_directory": "cache",
                "caption_extension": ".txt",
                "target_frames": [124],
            }
        ],
    },
    "interop": {"dataset_extra": {"root": {}, "general": {}, "datasets": [{}]}},
}

H3_PATHS = {
    "dit_path": "ckpts/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "video_vae_path": "ckpts/vae/minimax_h3_video_vae_fp16.safetensors",
    "audio_vae_path": "ckpts/vae/minimax_h3_audio_vae_fp32.safetensors",
    "text_encoder_path": "ckpts/text_encoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
}

H3_TRAIN_STATE = {
    "arch": "MiniMax-H3",
    "version": "fl2va",
    "task": "t2va",
    **H3_PATHS,
    "train_mode": "lora",
    "mixed_precision": "bf16",
    "timestep_sampling": "uniform",
    "weighting_scheme": "none",
    "discrete_flow_shift": 1.0,
    "h3_shift_video": 12.0,
    "h3_shift_audio": 3.0,
    "h3_visual_cond_clean": 0.999,
    "h3_audio_cond_clean": 1.0,
    "video_only": False,
    "convrot_int8": False,
    "convrot_int8_bwd": "bf16",
    "text_encoder_blocks_to_swap": 50,
    "text_encoder_attn_mode": "flash_attention_2",
    "dit_dtype": "bfloat16",
    "gradient_checkpointing": True,
    "blocks_to_swap": 48,
    "block_swap_h2d_only": True,
    "block_swap_ring_size": 2,
    "enable_sample": False,
    "network_dim": 32,
    "optimizer_type": "AdamW_adv",
    "learning_rate": "1e-4",
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
            "uncond_output",
            "uncond_text",
            "one_frame",
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
            "h3_best_of_k",
            "h3_best_of_k_stream",
            "video_only",
            "audio_loss_weight",
            "convrot_int8",
            "convrot_int8_bwd",
            "h3_guidance_loss_scale",
            "h3_guidance_loss_scale_audio",
            "h3_guidance_loss_sigma_min",
            "h3_guidance_loss_uncond_cache",
            "prune_adaln",
            "h3_allow_experimental_sample_duration",
            "one_frame",
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
            "prune_adaln",
        ):
            self.assertIn(field, branch)
        self.assertIn("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", self.generate)
        self.assertIn("self.config.setdefault('text_encoder_attn_mode', 'flash_attention_2')", branch)
        self.assertIn("def _sync_minimax_h3_task_ui", self.generate)

    def test_h3_numeric_controls_use_editable_sliders(self):
        cases = (
            (
                CacheStep(),
                lambda step: step._render_dynamic_arch_specific("MiniMax-H3"),
                {"cache_seed", "text_encoder_blocks_to_swap"},
                "arch_specific",
            ),
            (
                TrainStep(),
                lambda step: step._render_dynamic_te_paths("MiniMax-H3"),
                {
                    "text_encoder_blocks_to_swap",
                    "h3_shift_video",
                    "h3_shift_audio",
                    "h3_visual_cond_clean",
                    "h3_audio_cond_clean",
                    "audio_loss_weight",
                    "h3_guidance_loss_scale",
                    "h3_guidance_loss_scale_audio",
                    "h3_guidance_loss_sigma_min",
                },
                "model_paths",
            ),
            (
                GenerateStep(),
                lambda step: step._render_dynamic_arch_specific("MiniMax-H3"),
                {
                    "h3_width",
                    "h3_height",
                    "h3_frame_count",
                    "h3_steps",
                    "h3_seed",
                    "reference_index",
                    "text_encoder_blocks_to_swap",
                    "h3_blocks_to_swap",
                    "h3_shift_video",
                    "h3_shift_audio",
                    "h3_visual_cond_clean",
                    "h3_audio_cond_clean",
                },
                "arch_specific",
            ),
        )

        for step, render_h3, expected_keys, scope in cases:
            with self.subTest(step=type(step).__name__):
                i18n = get_i18n()
                binding_count = len(i18n._bindings)
                initial_bound_keys = set(step.config.get("_bound_controls", {}))
                with ui.column() as container:
                    render_h3(step)
                try:
                    bound_controls = step.config.get("_bound_controls", {})
                    created_bound_keys = set(bound_controls).difference(initial_bound_keys)
                    missing = expected_keys.difference(bound_controls)
                    self.assertEqual(missing, set())
                    for key in expected_keys:
                        self.assertTrue(
                            hasattr(bound_controls[key], "set_bound_value"),
                            f"{type(step).__name__}.{key} is not an editable slider",
                        )
                finally:
                    step._clear_control_scope(scope)
                    container.delete()
                remaining_bound_keys = set(step.config.get("_bound_controls", {}))
                self.assertTrue(created_bound_keys.isdisjoint(remaining_bound_keys))
                self.assertEqual(len(i18n._bindings), binding_count)

    def test_h3_new_native_controls_render_and_round_trip(self):
        cache = CacheStep()
        train = TrainStep()
        generate = GenerateStep()

        with ui.column() as cache_container:
            cache._render_dynamic_arch_specific("MiniMax-H3")
        with ui.column() as train_container:
            train._render_dynamic_te_paths("MiniMax-H3")
        with ui.column() as generate_container:
            generate._render_dynamic_arch_specific("MiniMax-H3")

        try:
            self.assertEqual(cache.uncond_output.selection_type, "save")
            cache._write_control_value(cache.uncond_output, "cache/h3_uncond.safetensors")
            cache._write_control_value(cache.uncond_text, "negative prompt")
            train._write_control_value(train.h3_guidance_loss_scale, 1.5)
            train._write_control_value(train.h3_guidance_loss_scale_audio, 0.75)
            train._write_control_value(train.h3_guidance_loss_sigma_min, 0.25)
            train._write_control_value(
                train.h3_guidance_loss_uncond_cache,
                "cache/h3_uncond.safetensors",
            )
            train._write_control_value(train.prune_adaln, True)
            generate._write_control_value(generate.prune_adaln, True)

            cache_state = cache._collect_form_state()
            train_state = train._collect_form_state()
            generate_state = generate._collect_form_state()

            self.assertEqual(cache_state["uncond_output"], "cache/h3_uncond.safetensors")
            self.assertEqual(cache_state["uncond_text"], "negative prompt")
            self.assertEqual(train_state["h3_guidance_loss_scale"], 1.5)
            self.assertEqual(train_state["h3_guidance_loss_scale_audio"], 0.75)
            self.assertEqual(train_state["h3_guidance_loss_sigma_min"], 0.25)
            self.assertEqual(
                train_state["h3_guidance_loss_uncond_cache"],
                "cache/h3_uncond.safetensors",
            )
            self.assertTrue(train_state["prune_adaln"])
            self.assertTrue(generate_state["prune_adaln"])
        finally:
            cache._clear_control_scope("arch_specific")
            train._clear_control_scope("model_paths")
            generate._clear_control_scope("arch_specific")
            cache_container.delete()
            train_container.delete()
            generate_container.delete()

    def test_h3_best_of_k_controls_render_normalize_and_serialize_as_toml(self):
        train = TrainStep()

        with ui.column() as container:
            train._render_dynamic_te_paths("MiniMax-H3")
        try:
            self.assertIs(type(train.h3_best_of_k.value), int)
            self.assertEqual(train.h3_best_of_k.value, 1)
            self.assertEqual(train.h3_best_of_k._props.get("min"), 1)
            self.assertEqual(train.h3_best_of_k._props.get("step"), 1)
            self.assertEqual(train.h3_best_of_k_stream.value, "video")
            for control in (train.h3_best_of_k, train.h3_best_of_k_stream):
                self.assertEqual(control._style.get("min-width"), "min(100%, 18rem)")

            train._write_control_value(train.h3_best_of_k, 3.0)
            train._write_control_value(train.h3_best_of_k_stream, "audio")
            state = train._get_config()
            serialized = toml.dumps(state)

            self.assertIs(type(state["h3_best_of_k"]), int)
            self.assertEqual(state["h3_best_of_k"], 3)
            self.assertEqual(state["h3_best_of_k_stream"], "audio")
            self.assertRegex(serialized, r"(?m)^h3_best_of_k = 3$")
            self.assertNotIn("h3_best_of_k = 3.0", serialized)
            self.assertNotIn('h3_best_of_k = "3"', serialized)
        finally:
            train._clear_control_scope("model_paths")
            container.delete()

    def test_h3_best_of_k_state_survives_architecture_round_trip(self):
        step = TrainStep()
        with ui.column() as container:
            step._model_path_container = ui.column()
        try:
            step._on_arch_change("MiniMax-H3", get_arch_info("MiniMax-H3"))
            step._write_control_value(step.h3_best_of_k, 4.0)
            step._write_control_value(step.h3_best_of_k_stream, "audio")

            step._on_arch_change("FLUX.2", get_arch_info("FLUX.2"))
            self.assertFalse(hasattr(step, "h3_best_of_k"))
            self.assertFalse(hasattr(step, "h3_best_of_k_stream"))
            self.assertEqual(step.config["h3_best_of_k"], 4)
            self.assertEqual(step.config["h3_best_of_k_stream"], "audio")

            step._on_arch_change("MiniMax-H3", get_arch_info("MiniMax-H3"))
            self.assertEqual(step.h3_best_of_k.value, 4)
            self.assertEqual(step.h3_best_of_k_stream.value, "audio")
        finally:
            step._clear_control_scope("model_paths")
            container.delete()

    def test_non_h3_config_keeps_best_of_k_fields_dormant(self):
        step = TrainStep()
        step._selected_arch = "FLUX.2"

        step._apply_config(
            {
                "arch": "FLUX.2",
                "h3_best_of_k": 5.0,
                "h3_best_of_k_stream": " AUDIO ",
            }
        )

        self.assertEqual(step.config["h3_best_of_k"], 5)
        self.assertEqual(step.config["h3_best_of_k_stream"], "audio")
        self.assertFalse(hasattr(step, "h3_best_of_k"))
        self.assertFalse(hasattr(step, "h3_best_of_k_stream"))

    def test_h3_catalog_and_cards_expose_one_frame_training(self):
        architecture = model_catalog.get_architecture("MiniMax-H3")
        self.assertIn("one_frame", architecture["pages"]["cache"]["flags"])
        self.assertIn("one_frame", architecture["pages"]["train"]["flags"])

        cases = (
            (CacheStep(), lambda step: step._render_dynamic_arch_specific("MiniMax-H3"), "arch_specific"),
            (TrainStep(), lambda step: step._render_dynamic_te_paths("MiniMax-H3"), "model_paths"),
        )
        for step, render, scope in cases:
            with self.subTest(step=type(step).__name__), ui.column() as container:
                render(step)
                self.assertIn("one_frame", step.config.get("_bound_controls", {}))
                step._write_control_value(step.one_frame, True)
                self.assertTrue(step._collect_form_state()["one_frame"])
                step._clear_control_scope(scope)
                container.delete()

    def test_h3_real_task_callbacks_hide_and_clear_one_frame_mode(self):
        cases = (
            (CacheStep(), "cache", ("model_paths", "arch_specific")),
            (TrainStep(), "train", ("model_paths",)),
        )

        for step, page_key, scopes in cases:
            with self.subTest(step=type(step).__name__), ui.column() as container:
                if isinstance(step, CacheStep):
                    step._model_path_container = ui.column()
                    step._model_specific_container = ui.column()
                else:
                    step._model_path_container = ui.column()
                step.model_selector = create_model_selector(
                    on_change=step._on_arch_change,
                    default_arch="MiniMax-H3",
                    page_key=page_key,
                )
                step._on_arch_change("MiniMax-H3", get_arch_info("MiniMax-H3"))

                step._write_control_value(step.one_frame, True)
                self.assertTrue(step._h3_one_frame_row.visible)
                self.assertTrue(step.config["one_frame"])

                step.model_selector.set_task("fl2va")
                self.assertFalse(step._h3_one_frame_row.visible)
                self.assertFalse(step.config["one_frame"])

                step.model_selector.set_task("t2va")
                self.assertTrue(step._h3_one_frame_row.visible)
                self.assertFalse(step.config["one_frame"])

                step._write_control_value(step.one_frame, True)
                step.model_selector.set_version("ref2va")
                self.assertFalse(step._h3_one_frame_row.visible)
                self.assertFalse(step.config["one_frame"])

                for scope in scopes:
                    step._clear_control_scope(scope)
                container.delete()

    def test_h3_unbounded_numeric_values_survive_slider_rendering(self):
        cache = CacheStep()
        cache.config["cache_seed"] = 10**30 + 1
        generate = GenerateStep()
        generate.config.update(
            {
                "h3_width": 2080,
                "h3_height": 2080,
                "h3_frame_count": 1042,
                "h3_steps": 150,
                "h3_seed": 9007199254740993,
                "reference_index": 10000,
            }
        )
        train = TrainStep()
        train.config["audio_loss_weight"] = 11.0

        with ui.column() as cache_container:
            cache._render_dynamic_arch_specific("MiniMax-H3")
        with ui.column() as generate_container:
            generate._render_dynamic_arch_specific("MiniMax-H3")
        with ui.column() as train_container:
            train._render_dynamic_te_paths("MiniMax-H3")

        try:
            self.assertEqual(cache.config["cache_seed"], 10**30 + 1)
            self.assertEqual(generate.config["h3_width"], 2080)
            self.assertEqual(generate.config["h3_height"], 2080)
            self.assertEqual(generate.config["h3_frame_count"], 1042)
            self.assertEqual(generate.config["h3_steps"], 150)
            self.assertEqual(generate.config["h3_seed"], 9007199254740993)
            self.assertEqual(generate.config["reference_index"], 10000)
            self.assertEqual(train.config["audio_loss_weight"], 11.0)
        finally:
            cache._clear_control_scope("arch_specific")
            generate._clear_control_scope("arch_specific")
            train._clear_control_scope("model_paths")
            cache_container.delete()
            generate_container.delete()
            train_container.delete()

    def test_h3_cache_seed_stays_exact_from_slider_render_to_argv(self):
        seed = 10**30 + 1
        cache = CacheStep()
        cache.config.update(
            {
                "arch": "MiniMax-H3",
                "version": "fl2va",
                "task": "t2va",
                "cache_seed": seed,
                **H3_PATHS,
            }
        )

        with ui.column() as container:
            cache._render_dynamic_arch_specific("MiniMax-H3")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                jobs = build_cache_jobs(cache.config, tmp, H3_PROJECT_CONFIG)

            self.assertEqual(cache.config["cache_seed"], seed)
            self.assertIn(f"--cache_seed={seed}", jobs[0].args)
        finally:
            cache._clear_control_scope("arch_specific")
            container.delete()

    def test_h3_audio_loss_precision_survives_render_collect_and_argv(self):
        train = TrainStep()
        train.config["audio_loss_weight"] = 0.75

        with ui.column() as container:
            train._render_dynamic_te_paths("MiniMax-H3")
        try:
            collected = train._collect_form_state()
            with tempfile.TemporaryDirectory() as tmp:
                job = build_train_job(
                    {**H3_TRAIN_STATE, "audio_loss_weight": collected["audio_loss_weight"]},
                    tmp,
                    H3_PROJECT_CONFIG,
                )

            self.assertEqual(train.config["audio_loss_weight"], 0.75)
            self.assertEqual(collected["audio_loss_weight"], 0.75)
            self.assertIn("--audio_loss_weight=0.75", job.args)

            train.config["_bound_controls"]["audio_loss_weight"].set_bound_value(1e30)
            self.assertEqual(train.config["audio_loss_weight"], 1e30)
            self.assertEqual(train._collect_form_state()["audio_loss_weight"], 1e30)
        finally:
            train._clear_control_scope("model_paths")
            container.delete()

    def test_h3_training_seed_uses_accelerate_integer_domain_from_control_to_argv(self):
        train = TrainStep()
        train.config["seed"] = 2**32

        with ui.column() as container:
            train._render_model_tab()
        try:
            collected = train._collect_form_state()
            self.assertEqual(collected["seed"], 2**32 - 1)

            seed_control = train.config["_bound_controls"]["seed"]
            seed_control.set_bound_value(1.9)
            self.assertEqual(train._collect_form_state()["seed"], 2)

            with tempfile.TemporaryDirectory() as tmp:
                job = build_train_job(
                    {**H3_TRAIN_STATE, "seed": 2**32 - 1},
                    tmp,
                    H3_PROJECT_CONFIG,
                )
            self.assertIn(f"--seed={2**32 - 1}", job.args)
        finally:
            train._clear_control_scope("model_paths")
            container.delete()

    def test_h3_train_continuous_flow_values_survive_render_collect_and_argv(self):
        values = {
            "h3_shift_video": 12.345,
            "h3_shift_audio": 3.4567,
            "h3_visual_cond_clean": 0.9995,
            "h3_audio_cond_clean": 0.12345,
        }
        train = TrainStep()
        train.config.update(values)

        with ui.column() as container:
            train._render_dynamic_te_paths("MiniMax-H3")
        try:
            collected = train._collect_form_state()
            with tempfile.TemporaryDirectory() as tmp:
                job = build_train_job(
                    {**H3_TRAIN_STATE, **{key: collected[key] for key in values}},
                    tmp,
                    H3_PROJECT_CONFIG,
                )

            self.assertEqual({key: collected[key] for key in values}, values)
            for key, value in values.items():
                self.assertIn(f"--{key}={value}", job.args)
        finally:
            train._clear_control_scope("model_paths")
            container.delete()

    def test_h3_generate_continuous_flow_values_survive_render_collect_and_argv(self):
        values = {
            "h3_shift_video": 12.345,
            "h3_shift_audio": 3.4567,
            "h3_visual_cond_clean": 0.9995,
            "h3_audio_cond_clean": 0.12345,
        }
        generate = GenerateStep()
        generate.config.update(
            {
                "arch": "MiniMax-H3",
                "version": "fl2va",
                "task": "t2va",
                **H3_PATHS,
                "prompt": "A singer performs under stage lights.",
                "h3_width": 768,
                "h3_height": 1344,
                "h3_frame_count": 124,
                "h3_steps": 30,
                "h3_seed": 42,
                "h3_blocks_to_swap": 48,
                "h3_output_path": "output/h3.mp4",
                **values,
            }
        )

        with ui.column() as container:
            generate._render_dynamic_arch_specific("MiniMax-H3")
        try:
            collected = generate._collect_form_state()
            job = build_generate_job(collected, ROOT)

            self.assertEqual({key: collected[key] for key in values}, values)
            for key, value in values.items():
                self.assertIn(f"--{key}={value}", job.args)
        finally:
            generate._clear_control_scope("arch_specific")
            container.delete()

    def test_h3_guidance_controls_preserve_continuous_and_optional_values(self):
        train = TrainStep()
        train.config.update(
            {
                "h3_guidance_loss_scale": 0.333,
                "h3_guidance_loss_scale_audio": "",
                "h3_guidance_loss_sigma_min": 0.255,
            }
        )

        with ui.column() as container:
            train._render_dynamic_te_paths("MiniMax-H3")
        try:
            state = train._collect_form_state()
            self.assertEqual(state["h3_guidance_loss_scale"], 0.333)
            self.assertEqual(state["h3_guidance_loss_scale_audio"], "")
            self.assertEqual(state["h3_guidance_loss_sigma_min"], 0.255)

            audio_control = train.config["_bound_controls"]["h3_guidance_loss_scale_audio"]
            audio_control.set_bound_value(0.75)
            self.assertEqual(train._collect_form_state()["h3_guidance_loss_scale_audio"], 0.75)
            audio_control.set_bound_value("")
            self.assertEqual(train._collect_form_state()["h3_guidance_loss_scale_audio"], "")
        finally:
            train._clear_control_scope("model_paths")
            container.delete()

    def test_cache_architecture_switch_releases_outgoing_dynamic_bindings(self):
        cache = CacheStep()
        i18n = get_i18n()
        binding_count = len(i18n._bindings)
        with ui.column() as container:
            cache._model_specific_container = ui.column()

        try:
            cache._on_arch_change("Mage-Flow", {})
            mage_slider = cache.config["_bound_controls"]["cache_seed"]
            self.assertEqual(len(i18n._bindings), binding_count + 1)

            cache._on_arch_change("MiniMax-H3", {})

            self.assertTrue(mage_slider.is_deleted)
            self.assertIsNot(cache.config["_bound_controls"]["cache_seed"], mage_slider)
            h3_slider = cache.config["_bound_controls"]["cache_seed"]
            self.assertGreater(len(i18n._bindings), binding_count + 5)

            cache._on_arch_change("Mage-Flow", {})

            self.assertTrue(h3_slider.is_deleted)
            self.assertEqual(len(i18n._bindings), binding_count + 1)
        finally:
            cache._clear_control_scope("arch_specific")
            container.delete()
        self.assertEqual(len(i18n._bindings), binding_count)

    def test_h3_bound_component_labels_follow_language_changes(self):
        cache = CacheStep()
        i18n = get_i18n()
        original_language = i18n.lang
        i18n.lang = "zh"

        with ui.column() as container:
            cache._render_dynamic_arch_specific("MiniMax-H3")
        try:
            slider = cache.config["_bound_controls"]["text_encoder_blocks_to_swap"]
            slider_label = next(
                element
                for element in slider.parent_slot.parent.descendants()
                if type(element).__name__ == "Label"
            )
            toggle = cache.config["_bound_controls"]["nvfp4_scaled_mm"]
            toggle_label = next(
                element
                for element in toggle.descendants()
                if type(element).__name__ == "Label" and element.text != TRANSLATIONS["zh"]["status_off"]
            )

            i18n.lang = "en"

            self.assertEqual(slider_label.text, TRANSLATIONS["en"]["h3_text_encoder_blocks_to_swap"])
            self.assertEqual(toggle_label.text, TRANSLATIONS["en"]["h3_nvfp4_scaled_mm"])
        finally:
            cache._clear_control_scope("arch_specific")
            container.delete()
            i18n.lang = original_language

    def test_h3_standard_component_labels_follow_language_changes(self):
        i18n = get_i18n()
        original_language = i18n.lang
        cases = (
            (
                CacheStep(),
                lambda step: step._render_dynamic_model_paths("MiniMax-H3"),
                "model_paths",
                {
                    "video_vae_path": "h3_video_vae",
                    "audio_vae_path": "h3_audio_vae",
                    "text_encoder_path": "h3_text_encoder_int8_recommended",
                },
            ),
            (
                CacheStep(),
                lambda step: step._render_dynamic_arch_specific("MiniMax-H3"),
                "arch_specific",
                {
                    "text_cache_dtype": "text_cache_dtype",
                    "text_encoder_attn_mode": "h3_text_encoder_attn_mode",
                    "uncond_output": "h3_uncond_output",
                    "uncond_text": "h3_uncond_text",
                },
            ),
            (
                TrainStep(),
                lambda step: step._render_dynamic_te_paths("MiniMax-H3"),
                "model_paths",
                {
                    "video_vae_path": "h3_video_vae_sampling",
                    "text_encoder_attn_mode": "h3_text_encoder_attn_mode",
                    "dit_dtype": "h3_dit_dtype",
                    "convrot_int8_bwd": "h3_convrot_backward",
                    "h3_guidance_loss_uncond_cache": "h3_guidance_loss_uncond_cache",
                },
            ),
            (
                GenerateStep(),
                lambda step: step._render_dynamic_te_paths("MiniMax-H3"),
                "model_paths",
                {
                    "video_vae_path": "h3_video_vae",
                    "audio_vae_path": "h3_audio_vae",
                    "text_encoder_path": "h3_text_encoder_int8_recommended",
                },
            ),
            (
                GenerateStep(),
                lambda step: step._render_dynamic_arch_specific("MiniMax-H3"),
                "arch_specific",
                {
                    "h3_output_path": "h3_output_video",
                    "text_cache_path": "h3_text_cache",
                    "first_frame_path": "h3_first_frame",
                    "reference_jsonl_path": "h3_reference_jsonl",
                    "text_encoder_attn_mode": "h3_text_encoder_attn_mode",
                },
            ),
        )

        try:
            for step, render, scope, labels in cases:
                with self.subTest(step=type(step).__name__):
                    i18n.lang = "zh"
                    with ui.column() as container:
                        render(step)
                    try:
                        self.assertEqual(
                            {name: getattr(step, name).label for name in labels},
                            {name: TRANSLATIONS["zh"][key] for name, key in labels.items()},
                        )

                        i18n.lang = "en"

                        self.assertEqual(
                            {name: getattr(step, name).label for name in labels},
                            {name: TRANSLATIONS["en"][key] for name, key in labels.items()},
                        )
                    finally:
                        step._clear_control_scope(scope)
                        container.delete()
        finally:
            i18n.lang = original_language

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
                "h3_uncond_output",
                "h3_uncond_text",
                "h3_one_frame_image_mode",
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
                "h3_best_of_k",
                "h3_best_of_k_tooltip",
                "h3_best_of_k_stream",
                "h3_best_of_k_stream_tooltip",
                "h3_best_of_k_stream_video",
                "h3_best_of_k_stream_audio",
                "h3_convrot_backward",
                "h3_video_only",
                "h3_quantize_convrot_int8",
                "h3_guidance_loss_scale",
                "h3_guidance_loss_scale_audio",
                "h3_guidance_loss_sigma_min",
                "h3_guidance_loss_uncond_cache",
                "h3_prune_adaln",
                "h3_allow_experimental_sample_duration",
                "h3_one_frame_image_mode",
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
                "h3_prune_adaln",
                "h3_allow_experimental_duration",
            ),
        }
        for source_name, keys in expected_by_source.items():
            source = getattr(self, source_name)
            for key in keys:
                with self.subTest(source=source_name, key=key):
                    translated_directly = f"t('{key}'" in source
                    translated_by_bound_component = re.search(
                        rf"(?:editable_slider|toggle_switch)\(\s*['\"]{re.escape(key)}['\"]",
                        source,
                    )
                    self.assertTrue(translated_directly or translated_by_bound_component)

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

    def test_train_task_sampling_defaults_only_enable_bundled_prompt_for_t2va(self):
        enable_sample = SimpleNamespace(value=True)
        sample_at_first = SimpleNamespace(value=True)
        step = TrainStep.__new__(TrainStep)
        step.config = {
            "enable_sample": True,
            "sample_at_first": True,
            "_bound_controls": {
                "enable_sample": enable_sample,
                "sample_at_first": sample_at_first,
            },
        }
        step._selected_arch = "MiniMax-H3"
        step.model_selector = SimpleNamespace(task="t2va")
        step.sample_prompts = SimpleNamespace(value="./toml/qinglong_minimaxh3.txt")

        for task in ("fl2va", "ref2va"):
            with self.subTest(task=task):
                step.model_selector.task = task
                step._apply_minimax_h3_task_sampling_defaults()

                self.assertFalse(step.config["enable_sample"])
                self.assertFalse(step.config["sample_at_first"])
                self.assertFalse(enable_sample.value)
                self.assertFalse(sample_at_first.value)
                self.assertEqual(step.sample_prompts.value, "")

        step.model_selector.task = "t2va"
        step._apply_minimax_h3_task_sampling_defaults()

        self.assertTrue(step.config["enable_sample"])
        self.assertTrue(step.config["sample_at_first"])
        self.assertEqual(step.sample_prompts.value, "./toml/qinglong_minimaxh3.txt")

        step.model_selector.task = "fl2va"
        step.sample_prompts.value = "./toml/custom_fl2va.txt"
        step._apply_minimax_h3_task_sampling_defaults(
            preserve_keys={"enable_sample", "sample_at_first", "sample_prompts"}
        )

        self.assertTrue(step.config["enable_sample"])
        self.assertTrue(step.config["sample_at_first"])
        self.assertEqual(step.sample_prompts.value, "./toml/custom_fl2va.txt")

    def test_train_task_change_refreshes_h3_sampling_defaults(self):
        step = TrainStep.__new__(TrainStep)
        step._selected_arch = "MiniMax-H3"
        step._selected_version = "fl2va"
        step._current_model_version = lambda _arch: "fl2va"
        step._refresh_train_mode_options = lambda _arch: None
        step._apply_mage_flow_train_defaults = lambda _arch: None
        step._sync_mage_flow_train_ui = lambda: None
        step._sync_minimax_h3_train_ui = lambda: None
        calls = []
        step._apply_minimax_h3_task_sampling_defaults = lambda: calls.append("sampling")

        step._on_arch_change("MiniMax-H3", {})

        self.assertEqual(calls, ["sampling"])

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
