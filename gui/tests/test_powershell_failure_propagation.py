import re
import shutil
import subprocess
import unittest
from pathlib import Path


class TestPowerShellFailurePropagation(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    HELPER = ROOT / "powershell" / "native_command.ps1"

    def test_checked_native_command_preserves_failure_exit_code(self):
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")

        command = (
            f". '{self.HELPER}'; "
            f"& '{pwsh}' -NoProfile -NonInteractive -Command 'exit 7'; "
            "Assert-NativeCommandSucceeded 'expected failure'"
        )
        result = subprocess.run(
            [pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 7)
        self.assertIn("expected failure", result.stderr)

    def test_python_workflows_check_every_native_exit(self):
        python_line = re.compile(r"^\s*python(?:\s|$)", re.MULTILINE)
        guard_line = re.compile(r"^\s*Assert-NativeCommandSucceeded\b", re.MULTILINE)
        workflows = []
        for path in self.ROOT.glob("*.ps1"):
            text = path.read_text(encoding="utf-8-sig")
            python_calls = len(python_line.findall(text))
            if not python_calls:
                continue
            workflows.append(path.name)
            self.assertIn("powershell/native_command.ps1", text, path.name)
            self.assertEqual(python_calls, len(guard_line.findall(text)), path.name)

        self.assertTrue(workflows)

    def test_installer_uses_last_exit_code_and_nonzero_failure_exit(self):
        text = (self.ROOT / "1.install-uv-qinglong.ps1").read_text(encoding="utf-8-sig")
        self.assertIn("$LASTEXITCODE", text)
        self.assertNotIn("if (!($?))", text)
        self.assertRegex(text, r"(?i)exit\s+\$ExitCode")


if __name__ == "__main__":
    unittest.main()
