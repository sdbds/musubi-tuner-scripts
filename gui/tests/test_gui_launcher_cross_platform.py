import unittest
from pathlib import Path


class TestGuiLauncherCrossPlatform(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    LAUNCHER = ROOT / "1.6.GUI.ps1"

    @classmethod
    def setUpClass(cls):
        cls.launcher_text = cls.LAUNCHER.read_text(encoding="utf-8")

    def test_launcher_does_not_pass_windows_relative_path_to_python(self):
        self.assertNotIn('".\\launch.py"', self.launcher_text)
        self.assertIn("$LaunchScript", self.launcher_text)
        self.assertIn('$launchArgs = @($LaunchScript, "--port=$Port")', self.launcher_text)


if __name__ == "__main__":
    unittest.main()
