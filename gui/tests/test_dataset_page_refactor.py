import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"


class TestDatasetPageRefactor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if str(GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(GUI_ROOT))
        cls.config_manager_module = importlib.import_module("utils.config_manager")
        cls.dataset_config_module = importlib.import_module("utils.dataset_config")
        cls.i18n_module = importlib.import_module("utils.i18n")
        cls.step1_module = importlib.import_module("wizard.step1_tagging")
        cls.main_text = (GUI_ROOT / "main.py").read_text(encoding="utf-8")
        cls.step1_text = (GUI_ROOT / "wizard" / "step1_tagging.py").read_text(encoding="utf-8")
        cls.cache_text = (GUI_ROOT / "wizard" / "step2_cache.py").read_text(encoding="utf-8")
        cls.train_text = (GUI_ROOT / "wizard" / "step3_train.py").read_text(encoding="utf-8")

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
        self.assertIn('t("dataset_directories"', self.step1_text)
        self.assertIn('t("dataset_resolution_wh"', self.step1_text)
        self.assertIn('t("dataset_batch_and_repeats"', self.step1_text)
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
            "dataset_preset_library",
            "dataset_preset",
            "import_dataset_preset",
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

    def test_dataset_preset_discovery_filters_and_formats_dataset_presets(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "qinglong-wan-datasets.toml").write_text("", encoding="utf-8")
            (tmp_path / "portrait_dataset.toml").write_text("", encoding="utf-8")
            (tmp_path / "notes.toml").write_text("", encoding="utf-8")
            (tmp_path / "readme.txt").write_text("", encoding="utf-8")

            presets = self.step1_module.discover_dataset_presets(tmp_path)

            self.assertEqual(len(presets), 2)
            self.assertIn(str(tmp_path / "qinglong-wan-datasets.toml"), presets)
            self.assertIn(str(tmp_path / "portrait_dataset.toml"), presets)
            self.assertEqual(presets[str(tmp_path / "qinglong-wan-datasets.toml")], "qinglong-wan-datasets")
            self.assertEqual(presets[str(tmp_path / "portrait_dataset.toml")], "portrait_dataset")

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


if __name__ == "__main__":
    unittest.main()
