import importlib.util
import asyncio
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

    def test_build_env_applies_gui_env_config_before_call_overrides(self):
        original_get_env = self.module.get_env_for_subprocess
        self.module.get_env_for_subprocess = lambda: {
            "HF_HOME": "gui-cache",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
        try:
            env = self.ProcessRunner()._build_env({"NVIDIA_TF32_OVERRIDE": "1"})
        finally:
            self.module.get_env_for_subprocess = original_get_env

        self.assertEqual(env["HF_HOME"], "gui-cache")
        self.assertEqual(env["NVIDIA_TF32_OVERRIDE"], "1")

    def test_env_log_includes_cuda_and_omits_sensitive_keys(self):
        text = "\n".join(
            self.ProcessRunner._format_env_for_log(
                {
                    "CUDA_VISIBLE_DEVICES": "1",
                    "HF_TOKEN": "hf_secret",
                    "OPENAI_APIKEY": "sk-secret",
                    "CUSTOM_FLAG": "enabled",
                }
            )
        )

        self.assertIn("CUDA_VISIBLE_DEVICES=1", text)
        self.assertIn("CUSTOM_FLAG=enabled", text)
        self.assertIn("sensitive environment variable(s) omitted", text)
        self.assertNotIn("HF_TOKEN", text)
        self.assertNotIn("OPENAI_APIKEY", text)
        self.assertNotIn("hf_secret", text)
        self.assertNotIn("sk-secret", text)

    def test_command_log_lists_only_dash_dash_parameters(self):
        lines = self.ProcessRunner._format_command_args_for_log(
            [
                "python",
                "-m",
                "example.module",
                "--arg",
                "spaced value",
                "--flag",
                "--named=value",
            ]
        )

        self.assertEqual(
            lines,
            [
                "  --arg spaced value",
                "  --flag",
                "  --named=value",
            ],
        )
        self.assertNotIn("python", "\n".join(lines))
        self.assertNotIn("example.module", "\n".join(lines))

    def test_command_log_keeps_multi_value_parameters_together(self):
        lines = self.ProcessRunner._format_command_args_for_log(
            [
                "python",
                "-m",
                "example.module",
                "--optimizer_args",
                "weight_decay=0.01",
                "betas=.9,.99",
                "decouple=True",
                "use_bias_correction=True",
                "d_coef=0.5",
                "d0=1e-3",
                "--flag",
            ]
        )

        self.assertEqual(
            lines,
            [
                "  --optimizer_args weight_decay=0.01 betas=.9,.99 decouple=True use_bias_correction=True d_coef=0.5 d0=1e-3",
                "  --flag",
            ],
        )

    def test_command_log_omits_sensitive_dash_dash_parameters(self):
        text = "\n".join(
            self.ProcessRunner._format_command_args_for_log(
                [
                    "python",
                    "--log_with=wandb",
                    "--wandb_api_key=secret-key",
                    "--huggingface_token",
                    "hf-secret",
                    "--normal",
                    "visible",
                ]
            )
        )

        self.assertIn("--log_with=wandb", text)
        self.assertIn("--normal visible", text)
        self.assertIn("sensitive command parameter(s) omitted", text)
        self.assertNotIn("wandb_api_key", text)
        self.assertNotIn("huggingface_token", text)
        self.assertNotIn("secret-key", text)
        self.assertNotIn("hf-secret", text)

    def test_start_log_uses_config_env_not_full_process_env(self):
        original_get_env = self.module.get_env_for_subprocess
        self.module.get_env_for_subprocess = lambda: {"CUDA_VISIBLE_DEVICES": "1"}
        runner = self.ProcessRunner()
        logs: list[str] = []

        async def fake_run_logged_subprocess(cmd, work_dir, env):
            return 0

        runner._notify_log = logs.append
        runner._run_logged_subprocess = fake_run_logged_subprocess
        try:
            asyncio.run(
                runner._run_with_status(
                    ["python", "-m", "example.module", "--arg=value"],
                    None,
                    {"CALL_ONLY": "1"},
                    native_console=False,
                )
            )
        finally:
            self.module.get_env_for_subprocess = original_get_env

        text = "\n".join(logs)
        self.assertIn("启动命令参数:", text)
        self.assertIn("  --arg=value", text)
        self.assertNotIn("python -m example.module", text)
        self.assertIn("GUI 配置环境变量:", text)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", text)
        self.assertNotIn("CALL_ONLY=1", text)


if __name__ == "__main__":
    unittest.main()
