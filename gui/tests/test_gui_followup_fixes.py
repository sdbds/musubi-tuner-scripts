import ast
import unittest
from pathlib import Path


class TestGuiFollowupFixes(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    GUI_ROOT = ROOT / "gui"
    MAIN_PATH = ROOT / "gui" / "main.py"
    LOG_VIEWER_PATH = ROOT / "gui" / "components" / "log_viewer.py"
    PATH_SELECTOR_PATH = ROOT / "gui" / "components" / "path_selector.py"
    ADVANCED_INPUTS_PATH = ROOT / "gui" / "components" / "advanced_inputs.py"
    EXECUTION_PANEL_PATH = ROOT / "gui" / "components" / "execution_panel.py"
    JOB_MANAGER_PATH = ROOT / "gui" / "utils" / "job_manager.py"
    I18N_PATH = ROOT / "gui" / "utils" / "i18n.py"
    CONSOLE_PAGE_PATH = ROOT / "gui" / "wizard" / "console_page.py"

    @classmethod
    def setUpClass(cls):
        cls.main_text = cls.MAIN_PATH.read_text(encoding="utf-8")
        cls.main_ast = ast.parse(cls.main_text)
        cls.log_viewer_text = cls.LOG_VIEWER_PATH.read_text(encoding="utf-8")
        cls.path_selector_text = cls.PATH_SELECTOR_PATH.read_text(encoding="utf-8")
        cls.advanced_inputs_text = cls.ADVANCED_INPUTS_PATH.read_text(encoding="utf-8")
        cls.execution_panel_text = cls.EXECUTION_PANEL_PATH.read_text(encoding="utf-8")
        cls.job_manager_text = cls.JOB_MANAGER_PATH.read_text(encoding="utf-8")
        cls.i18n_text = cls.I18N_PATH.read_text(encoding="utf-8")
        cls.console_page_text = cls.CONSOLE_PAGE_PATH.read_text(encoding="utf-8")

    def test_main_uses_lazy_loader_for_wizard_pages(self):
        self.assertIn("def _load_wizard_attr(", self.main_text)
        self.assertIn("import_module(", self.main_text)

    def test_main_has_no_eager_top_level_wizard_step_imports(self):
        eager_step_imports = []
        for node in self.main_ast.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("wizard.step"):
                    eager_step_imports.append(node.module)

        self.assertEqual(
            eager_step_imports,
            [],
            msg=f"Top-level wizard step imports should be lazy-loaded, found: {eager_step_imports}",
        )

    def test_parent_slot_suppression_has_single_choke_point(self):
        self.assertIn("def _install_exception_filter(", self.main_text)
        self.assertNotIn("class _SilenceParentSlotFilter", self.main_text)
        self.assertNotIn('logging.getLogger("nicegui").addFilter', self.main_text)

    def test_language_switch_does_not_reload_browser(self):
        handlers = [
            node
            for node in ast.walk(self.main_ast)
            if isinstance(node, ast.FunctionDef) and node.name == "on_lang_change"
        ]
        self.assertEqual(len(handlers), 1)
        handler_source = ast.get_source_segment(self.main_text, handlers[0]) or ""
        self.assertIn("set_language(lang)", handler_source)
        self.assertIn("_sync_visible_language_text(old_lang, lang)", handler_source)
        self.assertNotIn("ui.run_javascript", handler_source)
        self.assertNotIn("window.location.reload", handler_source)

    def test_language_switch_syncs_visible_text_without_page_rebuild(self):
        self.assertIn("def _sync_visible_language_text(", self.main_text)
        self.assertIn("get_translation_pairs(old_lang, new_lang)", self.main_text)
        self.assertIn("document.createTreeWalker", self.main_text)
        self.assertNotIn("window.location.reload", self.main_text)

    def test_i18n_exposes_translation_pairs_for_dom_sync(self):
        self.assertIn("def get_translation_pairs(", self.i18n_text)
        self.assertIn("def _flatten_translation_strings(", self.i18n_text)

    def test_auto_scroll_uses_i18n_and_button_toggle(self):
        self.assertIn("'auto_scroll': 'Auto Scroll'", self.i18n_text)
        self.assertIn("'auto_scroll': '自动滚动'", self.i18n_text)
        self.assertIn('toggle_switch_simple(', self.log_viewer_text)
        self.assertIn('"auto_scroll"', self.log_viewer_text)
        self.assertIn('toggle_switch(', self.advanced_inputs_text)
        self.assertIn("btn.value = bool(value)", self.advanced_inputs_text)
        self.assertIn("btn.value = new_value", self.advanced_inputs_text)
        self.assertNotIn('ui.switch(label', self.advanced_inputs_text)
        self.assertIn('toggle_switch_simple(', self.console_page_text)
        self.assertNotIn('ui.switch("Auto Scroll"', self.console_page_text)

    def test_tooltips_use_i18n(self):
        def is_t_call(node: ast.AST) -> bool:
            return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"

        def is_i18n_tooltip_arg(node: ast.AST) -> bool:
            if is_t_call(node):
                return True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                return is_t_call(node.func.value)
            return False

        violations = []
        for path in self.GUI_ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "tooltip":
                    continue
                if not node.args or not is_i18n_tooltip_arg(node.args[0]):
                    violations.append(f"{path.relative_to(self.ROOT)}:{node.lineno}")

        self.assertEqual(violations, [])

    def test_tooltip_i18n_keys_exist_for_all_languages(self):
        for key in [
            "browse_file",
            "browse_folder",
            "open_console",
            "num_workers_tooltip",
            "debug_mode_tooltip",
            "delete_env_var",
        ]:
            self.assertGreaterEqual(self.i18n_text.count(f"'{key}':"), 4)

    def test_header_exposes_settings_dialog(self):
        self.assertIn('_load_wizard_attr("wizard.step7_settings", "create_settings_dialog")', self.main_text)
        self.assertIn('ui.button(icon="settings"', self.main_text)
        self.assertIn('settings_btn.tooltip(t("nav_settings", "Settings"))', self.main_text)

    def test_log_viewer_copy_serializes_text_as_data(self):
        self.assertIn("json.dumps(text)", self.log_viewer_text)
        self.assertNotIn("navigator.clipboard.writeText(`", self.log_viewer_text)

    def test_path_selector_copy_serializes_text_as_data(self):
        self.assertIn("json.dumps(self.input.value or '')", self.path_selector_text)
        self.assertNotIn("navigator.clipboard.writeText(`", self.path_selector_text)

    def test_cancelled_jobs_are_converted_to_results(self):
        self.assertIn("except asyncio.CancelledError", self.execution_panel_text)
        self.assertIn("except asyncio.CancelledError", self.job_manager_text)
        self.assertIn("return job.result", self.job_manager_text)


if __name__ == "__main__":
    unittest.main()
