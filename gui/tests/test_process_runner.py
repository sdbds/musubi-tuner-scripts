import importlib.util
import sys
import unittest
from pathlib import Path


class TestProcessRunnerOutput(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    GUI_ROOT = ROOT / "gui"
    MODULE_PATH = GUI_ROOT / "utils" / "process_runner.py"

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(cls.GUI_ROOT))
        spec = importlib.util.spec_from_file_location("gui_process_runner", cls.MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.module = module
        cls.ProcessRunner = module.ProcessRunner

    def test_extract_output_lines_splits_tqdm_carriage_returns(self):
        lines, pending = self.ProcessRunner._extract_output_lines(
            "steps:  95%| 1977/2080, avr_loss=0.145\r"
            "steps:  95%| 1978/2080, avr_loss=0.146\r",
            "",
        )

        self.assertEqual(
            lines,
            [
                "steps:  95%| 1977/2080, avr_loss=0.145",
                "steps:  95%| 1978/2080, avr_loss=0.146",
            ],
        )
        self.assertEqual(pending, "")

    def test_extract_output_lines_preserves_partial_chunks(self):
        lines, pending = self.ProcessRunner._extract_output_lines("hello", "")
        self.assertEqual(lines, [])
        self.assertEqual(pending, "hello")

        lines, pending = self.ProcessRunner._extract_output_lines(" world\nnext", pending)
        self.assertEqual(lines, ["hello world"])
        self.assertEqual(pending, "next")

    def test_powershell_call_quotes_arguments(self):
        command = self.module._powershell_call(["python", "a'b", "spaced value"])
        self.assertEqual(command, "& 'python' 'a''b' 'spaced value'")

    def test_native_console_wrapper_files_exist(self):
        self.assertTrue(Path(self.module._WRAPPER_PATH).exists())
        self.assertTrue((self.GUI_ROOT / "utils" / "_color_inject" / "sitecustomize.py").exists())


if __name__ == "__main__":
    unittest.main()
