import importlib
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nicegui import ui


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestDatasetPageRefactor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.config_manager_module = importlib.import_module("utils.config_manager")
        cls.dataset_config_module = importlib.import_module("utils.dataset_config")
        cls.command_builder_module = importlib.import_module("utils.command_builder")
        cls.i18n_module = importlib.import_module("utils.i18n")
        cls.step1_module = importlib.import_module("wizard.step1_tagging")
        cls.main_text = (GUI_ROOT / "main.py").read_text(encoding="utf-8")
        cls.step1_text = (GUI_ROOT / "wizard" / "step1_tagging.py").read_text(encoding="utf-8")
        cls.cache_text = (GUI_ROOT / "wizard" / "step2_cache.py").read_text(encoding="utf-8")
        cls.train_text = (GUI_ROOT / "wizard" / "step3_train.py").read_text(encoding="utf-8")
        cls.generate_text = (GUI_ROOT / "wizard" / "step4_generate.py").read_text(encoding="utf-8")

    def test_main_keeps_tagging_route_but_labels_step_one_as_dataset(self):
        self.assertIn('(t("nav_dataset", "Dataset"), "/tagging", "label")', self.main_text)
        self.assertIn('("label", t("step") + " 1", t("nav_dataset", "Dataset")', self.main_text)
        self.assertIn('_load_wizard_attr("wizard.step1_tagging"', self.main_text)

    def test_step1_dataset_page_has_required_tabs(self):
        self.assertIn('t("dataset_tab_overview"', self.step1_text)
        self.assertIn('t("dataset_tab_tagging"', self.step1_text)
        self.assertIn('t("dataset_tab_details"', self.step1_text)
        self.assertIn('self.dataset_preset_select = ui.select(', self.step1_text)
        self.assertIn('t("dataset_preset_library"', self.step1_text)
        self.assertIn('ui.button(icon="refresh", on_click=self._reload_page)', self.step1_text)
        self.assertIn('self.dataset_preset_select.on_value_change(self._on_dataset_preset_change)', self.step1_text)
        self.assertIn('self._import_dataset_config_from_path(e.value)', self.step1_text)
        self.assertNotIn('t("import_dataset_preset"', self.step1_text)
        self.assertNotIn("on_click=self._import_selected_dataset_preset", self.step1_text)
        self.assertIn('t("dataset_directories"', self.step1_text)
        self.assertIn('t("dataset_resolution_wh"', self.step1_text)
        self.assertIn('t("dataset_batch_and_repeats"', self.step1_text)
        self.assertIn('classes("w-full gap-4 flex-wrap items-stretch")', self.step1_text)
        self.assertIn('classes("w-full gap-4 q-mt-md flex-wrap items-stretch")', self.step1_text)
        self.assertIn("dataset-stat-card", self.step1_text)
        self.assertIn("min-height: 118px; display: flex; flex-direction: column;", self.step1_text)
        self.assertIn("from components.advanced_inputs import toggle_switch", self.step1_text)
        self.assertIn('self.general_enable_bucket = toggle_switch(', self.step1_text)
        self.assertIn('controls["multiple_target"] = toggle_switch(', self.step1_text)
        self.assertNotIn("ui.checkbox(", self.step1_text)
        self.assertIn('t("dataset_template_type"', self.step1_text)
        self.assertIn('t("add_video_dataset"', self.step1_text)
        self.assertIn('t("dataset_source_mode"', self.step1_text)
        self.assertIn('t("control_directory"', self.step1_text)
        self.assertIn('t("target_frames"', self.step1_text)
        self.assertIn('t("frame_extraction"', self.step1_text)
        self.assertIn('t("tagging_external_tool_title"', self.step1_text)
        self.assertIn('t("tagging_install_button"', self.step1_text)
        self.assertIn('t("tagging_launch_button"', self.step1_text)
        self.assertIn("self._collect_qinglong_status()", self.step1_text)
        self.assertIn("ui.timer(5.0, self._refresh_qinglong_status)", self.step1_text)
        self.assertIn("QINGLONG_INSTALL_SCRIPT", self.step1_text)
        self.assertIn("QINGLONG_START_GUI_SCRIPT", self.step1_text)
        self.assertIn("subprocess.Popen(", self.step1_text)
        self.assertLess(
            self.step1_text.index('t("dataset_preset_library"'),
            self.step1_text.index('t("dataset_preview_summary"'),
        )
        self.assertNotIn('t("working_dir"', self.step1_text)
        self.assertNotIn('self._render_stat_card("tune", "musubi_project.toml"', self.step1_text)

    def test_cache_and_train_pages_stop_owning_editable_dataset_paths(self):
        self.assertNotIn('self.toml_path = create_path_selector(', self.cache_text)
        self.assertNotIn('self.dataset_config = create_path_selector(', self.train_text)
        self.assertIn("ui.navigate.to('/tagging')", self.cache_text)
        self.assertIn("ui.navigate.to('/tagging')", self.train_text)

    def test_train_and_generate_output_paths_are_editable_with_script_default(self):
        self.assertIn("SCRIPT_DEFAULT_OUTPUT_DIR", self.train_text)
        self.assertIn("self.output_dir = create_path_selector(", self.train_text)
        self.assertIn("default_path=SCRIPT_DEFAULT_OUTPUT_DIR", self.train_text)
        self.assertIn("SCRIPT_DEFAULT_OUTPUT_DIR", self.generate_text)
        self.assertIn("self.save_path = create_path_selector(", self.generate_text)
        self.assertIn("default_path=SCRIPT_DEFAULT_OUTPUT_DIR", self.generate_text)

    def test_i18n_contains_dataset_page_keys_for_all_languages(self):
        required_keys = {
            "nav_dataset",
            "dataset_page_desc",
            "dataset_tab_overview",
            "dataset_tab_tagging",
            "dataset_tab_details",
            "dataset_directories",
            "dataset_resolution_wh",
            "dataset_batch_and_repeats",
            "dataset_template_type",
            "dataset_type",
            "dataset_source_mode",
            "dataset_source_directory",
            "dataset_source_jsonl",
            "dataset_template_mode",
            "dataset_tagged",
            "dataset_untagged",
            "dataset_runnable",
            "dataset_not_runnable",
            "template_text_to_image",
            "template_image_edit",
            "template_video_generation",
            "template_video_control",
            "template_framepack_one_frame",
            "template_minimax_h3_one_frame",
            "dataset_preset_library",
            "dataset_preset",
            "add_video_dataset",
            "image_dataset",
            "video_dataset",
            "control_directory",
            "control_resolution_w",
            "control_resolution_h",
            "target_frames",
            "frame_extraction",
            "frame_extraction_head",
            "frame_extraction_chunk",
            "frame_extraction_slide",
            "frame_extraction_uniform",
            "frame_extraction_full",
            "frame_stride",
            "frame_sample",
            "max_frames",
            "source_fps",
            "fp_latent_window_size",
            "fp_1f_clean_indices",
            "fp_1f_target_index",
            "fp_1f_no_post",
            "minimax_h3_control_indices",
            "minimax_h3_control_indices_tooltip",
            "minimax_h3_target_index",
            "minimax_h3_target_index_tooltip",
            "dataset_reference",
            "open_dataset_page",
            "cache_dataset_reference_desc",
            "train_dataset_reference_desc",
            "tagging_external_tool_title",
            "tagging_external_tool_desc",
            "tagging_install_button",
            "tagging_launch_button",
            "tagging_tool_status_label",
            "tagging_env_status_label",
            "tagging_port_status_label",
            "tagging_tool_not_installed",
            "tagging_tool_installed",
            "tagging_tool_running",
            "tagging_uv_missing",
            "tagging_env_missing",
            "tagging_port_listening",
            "tagging_port_not_listening",
            "tagging_script_missing",
            "tagging_script_started",
            "tagging_script_launch_failed",
            "tagging_powershell_missing",
            "refresh_status",
            "invalid_tagging_dataset_dir",
            "tagging_finished",
            "tagging_failed",
            "tagging_requires_images",
            "tagging_model_missing",
            "tagging_method_not_supported",
        }

        for lang, translations in self.i18n_module.TRANSLATIONS.items():
            with self.subTest(lang=lang):
                missing = sorted(required_keys - set(translations.keys()))
                self.assertEqual(missing, [])

        self.assertIn(
            "supports T2VA and FL2VA tasks",
            self.i18n_module.TRANSLATIONS["en"]["h3_one_frame_image_mode_tooltip"],
        )

    def test_dataset_preset_discovery_uses_dataset_toml_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_config = (
                "[general]\n"
                "resolution = [512, 512]\n"
                "\n"
                "[[datasets]]\n"
                'image_directory = "./images"\n'
                'cache_directory = "./cache"\n'
            )
            (tmp_path / "qinglong-wan-datasets.toml").write_text(dataset_config, encoding="utf-8")
            (tmp_path / "portrait_dataset.toml").write_text(dataset_config, encoding="utf-8")
            (tmp_path / "qinglong_minimax_h3.toml").write_text(dataset_config, encoding="utf-8")
            (tmp_path / "notes.toml").write_text("", encoding="utf-8")
            (tmp_path / "readme.txt").write_text("", encoding="utf-8")

            presets = self.step1_module.discover_dataset_presets(tmp_path)

            self.assertEqual(len(presets), 3)
            self.assertIn(str(tmp_path / "qinglong-wan-datasets.toml"), presets)
            self.assertIn(str(tmp_path / "portrait_dataset.toml"), presets)
            self.assertIn(str(tmp_path / "qinglong_minimax_h3.toml"), presets)
            self.assertEqual(presets[str(tmp_path / "qinglong-wan-datasets.toml")], "qinglong-wan-datasets")
            self.assertEqual(presets[str(tmp_path / "portrait_dataset.toml")], "portrait_dataset")
            self.assertEqual(presets[str(tmp_path / "qinglong_minimax_h3.toml")], "qinglong_minimax_h3")

    def test_dataset_preview_summary_changes_with_selected_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "project-images-a").mkdir()
            (tmp_path / "project-images-a" / "1.txt").write_text("tag-a", encoding="utf-8")
            (tmp_path / "project-images-b").mkdir()
            (tmp_path / "project-images-b" / "1.txt").write_text("tag-b", encoding="utf-8")
            (tmp_path / "train" / "video").mkdir(parents=True)
            (tmp_path / "train" / "video" / "1.txt").write_text("video-tag", encoding="utf-8")
            (tmp_path / "train" / "video-2").mkdir(parents=True)
            (tmp_path / "train" / "video-2" / "1.json").write_text('{"caption":"ok"}', encoding="utf-8")

            project_config = self.config_manager_module.create_default_project_config()
            project_config["dataset"]["general"]["resolution"] = [512, 512]
            project_config["dataset"]["general"]["batch_size"] = 2
            project_config["dataset"]["general"]["num_repeats"] = 4
            project_config["dataset"]["datasets"] = [
                {"image_directory": "./project-images-a", "cache_directory": "./cache/a"},
                {"image_directory": "./project-images-b", "cache_directory": "./cache/b"},
            ]

            preset_path = tmp_path / "video-datasets.toml"
            preset_path.write_text(
                '[general]\n'
                'caption_extension = ".txt"\n'
                'enable_bucket = true\n'
                'bucket_no_upscale = false\n'
                '\n'
                '[[datasets]]\n'
                'resolution = [832, 480]\n'
                'batch_size = 1\n'
                'num_repeats = 6\n'
                'video_directory = "./train/video"\n'
                'cache_directory = "./train/video/cache"\n'
                '\n'
                '[[datasets]]\n'
                'resolution = [1024, 576]\n'
                'batch_size = 2\n'
                'num_repeats = 6\n'
                'video_directory = "./train/video-2"\n'
                'cache_directory = "./train/video-2/cache"\n',
                encoding="utf-8",
            )

            project_preview = self.step1_module.build_dataset_preview(project_config, None, tmp_path)
            preset_preview = self.step1_module.build_dataset_preview(project_config, preset_path, tmp_path)

            self.assertEqual(project_preview["summary"]["dataset_count"], "2")
            self.assertEqual(project_preview["summary"]["resolution"], "512x512")
            self.assertEqual(
                project_preview["summary"]["directories"],
                ["image::./project-images-a", "image::./project-images-b"],
            )
            self.assertEqual(project_preview["summary"]["resolution_values"], ["512, 512"])
            self.assertEqual(project_preview["summary"]["batch_sizes"], ["2"])
            self.assertEqual(project_preview["summary"]["repeat_values"], ["4"])
            self.assertEqual(project_preview["summary"]["template_type"], "template_text_to_image")
            self.assertEqual(project_preview["summary"]["tagging_status"], "dataset_tagged")
            self.assertEqual(project_preview["source_label"], "Current Project")

            self.assertEqual(preset_preview["summary"]["dataset_count"], "2")
            self.assertEqual(
                preset_preview["summary"]["directories"],
                ["video::./train/video", "video::./train/video-2"],
            )
            self.assertEqual(preset_preview["summary"]["resolution_values"], ["832, 480", "1024, 576"])
            self.assertEqual(preset_preview["summary"]["batch_sizes"], ["1", "2"])
            self.assertEqual(preset_preview["summary"]["repeat_values"], ["6"])
            self.assertEqual(preset_preview["summary"]["template_type"], "template_video_generation")
            self.assertEqual(preset_preview["summary"]["tagging_status"], "dataset_tagged")
            self.assertEqual(preset_preview["source_label"], "video-datasets")
            self.assertEqual(preset_preview["summary"]["source_path"], str(preset_path))

    def test_dataset_preview_status_detects_untagged_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "project-images").mkdir()

            project_config = self.config_manager_module.create_default_project_config()
            project_config["dataset"]["datasets"] = [{"image_directory": "./project-images"}]

            preview = self.step1_module.build_dataset_preview(project_config, None, tmp_path)

            self.assertEqual(preview["summary"]["directories"], ["image::./project-images"])
            self.assertEqual(preview["summary"]["tagging_status"], "dataset_untagged")

    def test_tagging_tab_uses_qinglong_captions_submodule_scripts(self):
        step = self.step1_module.DatasetStep()

        install_script = step._qinglong_script_path(self.step1_module.QINGLONG_INSTALL_SCRIPT)
        launch_script = step._qinglong_script_path(self.step1_module.QINGLONG_START_GUI_SCRIPT)

        self.assertEqual(install_script, ROOT / "qinglong-captions" / "1.install-uv-qinglong.ps1")
        self.assertEqual(launch_script, ROOT / "qinglong-captions" / "start_gui.ps1")
        self.assertTrue(install_script.exists())
        self.assertTrue(launch_script.exists())
        self.assertIsNotNone(step._powershell_command_prefix())

    def test_qinglong_status_detection_reports_not_installed_installed_and_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            step = self.step1_module.DatasetStep()
            step.qinglong_captions_dir = tmp_path

            with patch.object(self.step1_module.shutil, "which", return_value=None), patch.object(
                self.step1_module, "_is_local_port_listening", return_value=False
            ):
                status = step._collect_qinglong_status()
                self.assertEqual(status["overall_status"], "tagging_tool_not_installed")

            (tmp_path / ".venv").mkdir()

            with patch.object(self.step1_module.shutil, "which", return_value="C:/tools/uv.exe"), patch.object(
                self.step1_module, "_is_local_port_listening", return_value=False
            ):
                status = step._collect_qinglong_status()
                self.assertEqual(status["overall_status"], "tagging_tool_installed")
                self.assertEqual(status["env_path"], tmp_path / ".venv")

            with patch.object(self.step1_module.shutil, "which", return_value="C:/tools/uv.exe"), patch.object(
                self.step1_module, "_is_local_port_listening", return_value=True
            ):
                status = step._collect_qinglong_status()
                self.assertEqual(status["overall_status"], "tagging_tool_running")
                self.assertTrue(status["port_listening"])

    def test_dataset_import_export_round_trip_preserves_unknown_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "sample_dataset.toml"
            dataset_path.write_text(
                '[general]\n'
                'resolution = [1024, 1024]\n'
                'caption_extension = ".txt"\n'
                'batch_size = 1\n'
                'enable_bucket = true\n'
                'bucket_no_upscale = false\n'
                'mystery_general = "keep-me"\n'
                '\n'
                '[[datasets]]\n'
                'image_directory = "./images"\n'
                'cache_directory = "./cache"\n'
                'num_repeats = 4\n'
                'surprise_flag = true\n'
                '\n'
                'top_level_unknown = "preserve-root"\n',
                encoding="utf-8",
            )

            imported = self.dataset_config_module.load_dataset_config_import(dataset_path)
            self.assertEqual(imported["dataset"]["general"]["resolution"], [1024, 1024])
            self.assertEqual(imported["dataset"]["datasets"][0]["image_directory"], "./images")
            self.assertEqual(imported["interop"]["dataset_extra"]["general"]["mystery_general"], "keep-me")
            self.assertTrue(imported["interop"]["dataset_extra"]["datasets"][0]["surprise_flag"])
            self.assertEqual(imported["interop"]["dataset_extra"]["root"]["top_level_unknown"], "preserve-root")

            project_config = self.config_manager_module.create_default_project_config()
            project_config["dataset"] = imported["dataset"]
            project_config["interop"]["dataset_extra"] = imported["interop"]["dataset_extra"]

            export_path = tmp_path / "dataset_config.toml"
            self.dataset_config_module.export_dataset_config(project_config, export_path)

            exported_text = export_path.read_text(encoding="utf-8")
            self.assertIn('mystery_general = "keep-me"', exported_text)
            self.assertIn("surprise_flag = true", exported_text)
            self.assertIn('top_level_unknown = "preserve-root"', exported_text)

    def _bare_dataset_step(self):
        step = self.step1_module.DatasetStep.__new__(self.step1_module.DatasetStep)
        step.project_config = {
            "dataset": {"general": {}, "datasets": []},
            "interop": {
                "dataset_extra": {"root": {}, "general": {}, "datasets": []}
            },
        }
        step.dataset_row_states = []
        step.dataset_row_controls = []
        step.dataset_rows_container = None
        return step

    def _h3_row_state(
        self,
        step,
        source: str = "directory",
        target_index="0",
        control_directory: str = "",
        clean_indices: str = "",
    ):
        state = step._empty_dataset_row_state(
            "image", "minimax_h3_one_frame", source
        )
        state.update(
            {
                "image_directory": "./train/h3-images" if source == "directory" else "",
                "image_jsonl_file": "./train/h3-images.jsonl" if source == "jsonl" else "",
                "cache_directory": "./train/h3-image-cache",
                "caption_extension": ".txt",
                "resolution_w": "1024",
                "resolution_h": "1024",
                "batch_size": "1",
                "num_repeats": "1",
                "control_directory": control_directory,
                "fp_1f_clean_indices": clean_indices,
                "fp_1f_target_index": target_index,
            }
        )
        return state

    def test_h3_control_indices_require_one_or_two_nonnegative_integers(self):
        parser = self.step1_module._parse_h3_control_indices

        for raw_value, expected in (
            ("0", [0]),
            ("0, 48", [0, 48]),
            ([24], [24]),
            ((0, 48), [0, 48]),
        ):
            with self.subTest(raw_value=raw_value):
                self.assertEqual(
                    parser(raw_value, "H3 control frame indices"),
                    expected,
                )

        for raw_value in (
            "",
            [],
            "-1",
            "0, 24, 48",
            "0, nope",
            "1.5",
            True,
            [0, True],
        ):
            with self.subTest(raw_value=raw_value), self.assertRaisesRegex(
                ValueError,
                "one or two nonnegative integers",
            ):
                parser(raw_value, "H3 control frame indices")

    def test_minimax_h3_template_inference_recognizes_explicit_zero(self):
        step = self._bare_dataset_step()
        self.assertIn(
            "minimax_h3_one_frame", step._dataset_template_options("image")
        )
        step.project_config["interop"]["import_sources"] = {
            "dataset_config": "./toml/qinglong_minimax_h3_image.toml"
        }

        self.assertEqual(
            step._infer_dataset_row_template(
                "image", {"fp_1f_target_index": 0}
            ),
            "minimax_h3_one_frame",
        )

    def test_manual_h3_template_identity_persists_and_drives_preview(self):
        step = self._bare_dataset_step()
        step.project_dir = ROOT
        step.dataset_row_states = [self._h3_row_state(step, target_index="")]
        step.general_resolution_w = SimpleNamespace(value="1024")
        step.general_resolution_h = SimpleNamespace(value="1024")
        step.general_caption_extension = SimpleNamespace(value=".txt")
        step.general_batch_size = SimpleNamespace(value="1")
        step.general_num_repeats = SimpleNamespace(value="1")
        step.general_enable_bucket = SimpleNamespace(value=True)
        step.general_bucket_no_upscale = SimpleNamespace(value=False)
        empty_project = self.config_manager_module.create_default_project_config()

        with patch.object(
            self.step1_module.config_manager,
            "load_project_config",
            return_value=empty_project,
        ):
            step._persist_dataset_to_project_config()

        self.assertEqual(
            step.project_config["interop"]["dataset_templates"],
            ["minimax_h3_one_frame"],
        )
        preview = self.step1_module.build_dataset_preview(
            step.project_config, None, ROOT
        )
        self.assertEqual(
            preview["summary"]["template_type"],
            "template_minimax_h3_one_frame",
        )

        restored = self._bare_dataset_step()
        restored.project_config = step.project_config
        restored._refresh_dataset_row_states()
        self.assertEqual(
            restored.dataset_row_states[0]["dataset_template"],
            "minimax_h3_one_frame",
        )

        exported = self.dataset_config_module.build_dataset_config(step.project_config)
        self.assertNotIn("dataset_templates", exported)

    def test_h3_target_index_exports_zero_for_directory_and_jsonl_sources(self):
        step = self._bare_dataset_step()

        for source, expected_source_key in (
            ("directory", "image_directory"),
            ("jsonl", "image_jsonl_file"),
        ):
            with self.subTest(source=source):
                step.dataset_row_states = [self._h3_row_state(step, source)]
                datasets, extras, templates = step._collect_dataset_rows()

                self.assertEqual(datasets[0]["fp_1f_target_index"], 0)
                self.assertIn(expected_source_key, datasets[0])
                self.assertEqual(extras, [{}])
                self.assertEqual(templates, ["minimax_h3_one_frame"])
                if source == "jsonl":
                    self.assertNotIn("caption_extension", datasets[0])

    def test_h3_target_index_empty_is_omitted_and_invalid_values_fail(self):
        step = self._bare_dataset_step()
        step.dataset_row_states = [self._h3_row_state(step, target_index="")]
        datasets, _, _ = step._collect_dataset_rows()
        self.assertNotIn("fp_1f_target_index", datasets[0])

        for raw_value in ("-1", "not-a-number"):
            step.dataset_row_states = [
                self._h3_row_state(step, target_index=raw_value)
            ]
            with self.subTest(raw_value=raw_value), self.assertRaisesRegex(
                ValueError, "nonnegative integer"
            ):
                step._collect_dataset_rows()

    def test_h3_controlled_directory_rows_round_trip_one_or_two_controls(self):
        step = self._bare_dataset_step()

        for clean_indices, expected_indices in (
            ("0", [0]),
            ("0, 48", [0, 48]),
        ):
            with self.subTest(clean_indices=clean_indices):
                step.dataset_row_states = [
                    self._h3_row_state(
                        step,
                        target_index="24",
                        control_directory="./train/h3-control",
                        clean_indices=clean_indices,
                    )
                ]
                datasets, extras, templates = step._collect_dataset_rows()

                self.assertEqual(
                    datasets[0]["control_directory"],
                    "./train/h3-control",
                )
                self.assertEqual(
                    datasets[0]["fp_1f_clean_indices"],
                    expected_indices,
                )
                self.assertEqual(datasets[0]["fp_1f_target_index"], 24)
                self.assertEqual(extras, [{}])
                self.assertEqual(templates, ["minimax_h3_one_frame"])

                restored_state = step._build_dataset_row_state(
                    datasets[0],
                    {},
                    "minimax_h3_one_frame",
                )
                self.assertEqual(
                    restored_state["fp_1f_clean_indices"],
                    ", ".join(str(value) for value in expected_indices),
                )
                step.dataset_row_states = [restored_state]
                round_trip, _, _ = step._collect_dataset_rows()
                self.assertEqual(round_trip, datasets)

    def test_h3_controlled_jsonl_rows_export_indices_without_control_directory(self):
        step = self._bare_dataset_step()
        step.dataset_row_states = [
            self._h3_row_state(
                step,
                source="jsonl",
                target_index="24",
                clean_indices="0, 48",
            )
        ]

        datasets, _, _ = step._collect_dataset_rows()

        self.assertEqual(datasets[0]["fp_1f_clean_indices"], [0, 48])
        self.assertEqual(datasets[0]["fp_1f_target_index"], 24)
        self.assertNotIn("control_directory", datasets[0])

    def test_h3_controlled_rows_require_paired_directory_fields_and_target(self):
        step = self._bare_dataset_step()
        invalid_states = (
            (
                self._h3_row_state(
                    step,
                    control_directory="./train/h3-control",
                ),
                "control directory.*indices.*together",
            ),
            (
                self._h3_row_state(step, clean_indices="0"),
                "control directory.*indices.*together",
            ),
            (
                self._h3_row_state(
                    step,
                    target_index="",
                    control_directory="./train/h3-control",
                    clean_indices="0",
                ),
                "require.*target frame index",
            ),
            (
                self._h3_row_state(
                    step,
                    source="jsonl",
                    target_index="",
                    clean_indices="0",
                ),
                "require.*target frame index",
            ),
            (
                self._h3_row_state(
                    step,
                    control_directory="./train/h3-control",
                    clean_indices="-1",
                ),
                "one or two nonnegative integers",
            ),
            (
                self._h3_row_state(
                    step,
                    control_directory="./train/h3-control",
                    clean_indices="0, 24, 48",
                ),
                "one or two nonnegative integers",
            ),
        )

        for state, message in invalid_states:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                step.dataset_row_states = [state]
                step._collect_dataset_rows()

    def test_switching_to_h3_template_clears_unsupported_hidden_state(self):
        step = self._bare_dataset_step()
        for source_template in ("framepack_one_frame", "image_edit"):
            with self.subTest(source_template=source_template):
                state = step._empty_dataset_row_state("image", source_template)
                state.update(
                    {
                        "control_directory": "./control",
                        "control_resolution_w": "512",
                        "control_resolution_h": "512",
                        "fp_latent_window_size": "9",
                        "fp_1f_clean_indices": "0, 1",
                        "fp_1f_target_index": "0",
                        "fp_1f_no_post": True,
                        "multiple_target": True,
                        "no_resize_control": True,
                    }
                )
                step.dataset_row_states = [state]
                step.dataset_row_controls = []

                step._set_dataset_row_mode(
                    0, "dataset_template", "minimax_h3_one_frame"
                )

                updated = step.dataset_row_states[0]
                self.assertEqual(updated["control_directory"], "")
                self.assertEqual(updated["control_resolution_w"], "")
                self.assertEqual(updated["control_resolution_h"], "")
                self.assertEqual(updated["fp_latent_window_size"], "")
                self.assertEqual(updated["fp_1f_clean_indices"], "")
                self.assertFalse(updated["fp_1f_no_post"])
                self.assertFalse(updated["multiple_target"])
                self.assertFalse(updated["no_resize_control"])
                self.assertEqual(updated["fp_1f_target_index"], "0")

    def test_h3_template_renders_control_indices_without_resize_controls(self):
        step = self._bare_dataset_step()
        step.dataset_row_states = [self._h3_row_state(step)]

        with ui.column() as container:
            step.dataset_rows_container = ui.column()
            step._render_dataset_rows()
        try:
            controls = step.dataset_row_controls[0]
            self.assertIn("control_directory", controls)
            self.assertIn("fp_1f_clean_indices", controls)
            self.assertIn("fp_1f_target_index", controls)
            self.assertTrue(
                {
                    "control_resolution_w",
                    "control_resolution_h",
                    "fp_latent_window_size",
                    "fp_1f_no_post",
                    "multiple_target",
                    "no_resize_control",
                }.isdisjoint(controls)
            )
        finally:
            container.delete()

    def test_dataset_save_reports_h3_target_validation_errors(self):
        step = self._bare_dataset_step()
        with patch.object(
            step,
            "_persist_dataset_to_project_config",
            side_effect=ValueError("Target index must be a nonnegative integer"),
        ), patch.object(self.step1_module.ui, "notify") as notify, patch.object(
            self.step1_module.config_manager, "save_project_config"
        ) as save:
            step._save_dataset_state()

        notify.assert_called_once_with(
            "Target index must be a nonnegative integer", type="negative"
        )
        save.assert_not_called()

    def test_minimax_h3_image_examples_match_one_frame_contract(self):
        dataset_path = ROOT / "toml" / "qinglong_minimax_h3_image.toml"
        prompt_path = ROOT / "toml" / "qinglong_minimaxh3_image.txt"

        with dataset_path.open("rb") as handle:
            dataset = tomllib.load(handle)
        self.assertEqual(dataset["general"]["resolution"], [1024, 1024])
        self.assertEqual(dataset["general"]["batch_size"], 1)
        self.assertTrue(dataset["general"]["enable_bucket"])
        self.assertEqual(dataset["datasets"][0]["fp_1f_target_index"], 0)
        self.assertNotEqual(
            dataset["datasets"][0]["image_directory"],
            dataset["datasets"][0]["cache_directory"],
        )

        prompt_lines = prompt_path.read_text(encoding="utf-8").splitlines()
        prompt = next(
            line for line in prompt_lines if line and not line.startswith("#")
        )
        tokens = prompt.split()
        for flag, value in (
            ("--w", "1024"),
            ("--h", "1024"),
            ("--f", "1"),
            ("--s", "30"),
            ("--d", "1026"),
        ):
            self.assertEqual(tokens[tokens.index(flag) + 1], value)

    def test_minimax_h3_controlled_image_examples_match_one_frame_contract(self):
        workflows = {
            "edit": {
                "clean_indices": [0],
                "one_frame": "target_index=24,control_index=0",
                "has_end_image": False,
            },
            "inbetween": {
                "clean_indices": [0, 48],
                "one_frame": "target_index=24,control_index=0;48",
                "has_end_image": True,
            },
        }

        for name, expected in workflows.items():
            with self.subTest(workflow=name):
                dataset_path = (
                    ROOT / "toml" / f"qinglong_minimax_h3_image_{name}.toml"
                )
                prompt_path = (
                    ROOT / "toml" / f"qinglong_minimaxh3_image_{name}.txt"
                )
                with dataset_path.open("rb") as handle:
                    dataset = tomllib.load(handle)

                self.assertEqual(dataset["general"]["resolution"], [1024, 1024])
                self.assertEqual(dataset["general"]["batch_size"], 1)
                row = dataset["datasets"][0]
                self.assertEqual(
                    row["fp_1f_clean_indices"],
                    expected["clean_indices"],
                )
                self.assertEqual(row["fp_1f_target_index"], 24)
                self.assertEqual(
                    len(
                        {
                            row["image_directory"],
                            row["control_directory"],
                            row["cache_directory"],
                        }
                    ),
                    3,
                )
                self.assertTrue(
                    {
                        "multiple_target",
                        "no_resize_control",
                        "control_resolution",
                    }.isdisjoint(row)
                )

                prompt = next(
                    line
                    for line in prompt_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line and not line.startswith("#")
                )
                tokens = prompt.split()
                self.assertEqual(tokens[tokens.index("--w") + 1], "1024")
                self.assertEqual(tokens[tokens.index("--h") + 1], "1024")
                self.assertEqual(tokens[tokens.index("--f") + 1], "1")
                self.assertEqual(tokens.count("--i"), 1)
                self.assertEqual(
                    "--ei" in tokens,
                    expected["has_end_image"],
                )
                self.assertEqual(
                    tokens[tokens.index("--of") + 1],
                    expected["one_frame"],
                )

    def test_minimax_h3_image_import_preset_and_command_flow(self):
        dataset_path = ROOT / "toml" / "qinglong_minimax_h3_image.toml"
        imported = self.dataset_config_module.load_dataset_config_import(dataset_path)
        project_config = self.config_manager_module.create_default_project_config()
        project_config["dataset"] = imported["dataset"]
        project_config["interop"]["dataset_extra"] = imported["interop"][
            "dataset_extra"
        ]
        project_config["interop"]["import_sources"] = {
            "dataset_config": str(dataset_path)
        }

        manager = self.config_manager_module.ConfigManager()
        cache_preset = manager.load_config("cache", "minimax_h3_image")
        train_preset = manager.load_config("train", "minimax_h3_image")
        with tempfile.TemporaryDirectory() as tmp:
            cache_jobs = self.command_builder_module.build_cache_jobs(
                cache_preset, tmp, project_config
            )
            train_job = self.command_builder_module.build_train_job(
                train_preset, tmp, project_config
            )
            with (Path(tmp) / "dataset_config.toml").open("rb") as handle:
                exported_dataset = tomllib.load(handle)

        self.assertEqual(len(cache_jobs), 2)
        self.assertTrue(all("--one_frame" in job.args for job in cache_jobs))
        self.assertIn("--one_frame", train_job.args)
        self.assertIn("--video_only", train_job.args)
        self.assertIn("--h3_guidance_loss_scale=4.0", train_job.args)
        self.assertEqual(
            exported_dataset["datasets"][0]["fp_1f_target_index"], 0
        )

    def test_minimax_h3_controlled_image_import_preset_and_command_flow(self):
        manager = self.config_manager_module.ConfigManager()
        workflows = {
            "minimax_h3_image_edit": [0],
            "minimax_h3_image_inbetween": [0, 48],
        }

        for name, clean_indices in workflows.items():
            with self.subTest(workflow=name):
                dataset_path = ROOT / "toml" / f"qinglong_{name}.toml"
                imported = self.dataset_config_module.load_dataset_config_import(
                    dataset_path
                )
                project_config = (
                    self.config_manager_module.create_default_project_config()
                )
                project_config["dataset"] = imported["dataset"]
                project_config["interop"]["dataset_extra"] = imported["interop"][
                    "dataset_extra"
                ]
                project_config["interop"]["import_sources"] = {
                    "dataset_config": str(dataset_path)
                }

                cache_preset = manager.load_config("cache", name)
                train_preset = manager.load_config("train", name)
                with tempfile.TemporaryDirectory() as tmp:
                    cache_jobs = self.command_builder_module.build_cache_jobs(
                        cache_preset,
                        tmp,
                        project_config,
                    )
                    train_job = self.command_builder_module.build_train_job(
                        train_preset,
                        tmp,
                        project_config,
                    )
                    with (Path(tmp) / "dataset_config.toml").open("rb") as handle:
                        exported_dataset = tomllib.load(handle)

                self.assertEqual(len(cache_jobs), 2)
                for job in cache_jobs:
                    self.assertEqual(job.args.count("--task=fl2va"), 1)
                    self.assertEqual(job.args.count("--one_frame"), 1)
                self.assertEqual(train_job.args.count("--task=fl2va"), 1)
                self.assertEqual(train_job.args.count("--one_frame"), 1)
                self.assertIn("--video_only", train_job.args)
                self.assertNotIn("--sample_at_first", train_job.args)
                row = exported_dataset["datasets"][0]
                self.assertEqual(row["fp_1f_clean_indices"], clean_indices)
                self.assertEqual(row["fp_1f_target_index"], 24)


if __name__ == "__main__":
    unittest.main()
