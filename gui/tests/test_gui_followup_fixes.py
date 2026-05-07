import ast
import unittest
from pathlib import Path


class TestGuiFollowupFixes(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    MAIN_PATH = ROOT / "gui" / "main.py"
    LOG_VIEWER_PATH = ROOT / "gui" / "components" / "log_viewer.py"
    PATH_SELECTOR_PATH = ROOT / "gui" / "components" / "path_selector.py"
    EXECUTION_PANEL_PATH = ROOT / "gui" / "components" / "execution_panel.py"
    JOB_MANAGER_PATH = ROOT / "gui" / "utils" / "job_manager.py"

    @classmethod
    def setUpClass(cls):
        cls.main_text = cls.MAIN_PATH.read_text(encoding="utf-8")
        cls.main_ast = ast.parse(cls.main_text)
        cls.log_viewer_text = cls.LOG_VIEWER_PATH.read_text(encoding="utf-8")
        cls.path_selector_text = cls.PATH_SELECTOR_PATH.read_text(encoding="utf-8")
        cls.execution_panel_text = cls.EXECUTION_PANEL_PATH.read_text(encoding="utf-8")
        cls.job_manager_text = cls.JOB_MANAGER_PATH.read_text(encoding="utf-8")

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
