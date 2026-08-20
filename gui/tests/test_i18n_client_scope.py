import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock


class FakeClient:
    def __init__(self):
        self.storage = {}


class TestI18nClientScope(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    GUI_ROOT = ROOT / "gui"

    @classmethod
    def setUpClass(cls):
        if str(cls.GUI_ROOT) not in sys.path:
            sys.path.insert(0, str(cls.GUI_ROOT))
        from utils import i18n

        cls.i18n = i18n

    def setUp(self):
        self.i18n._client_i18n.clear()
        self.i18n._fallback_i18n.lang = "zh"

    def test_language_is_isolated_and_remembered_per_client(self):
        first = FakeClient()
        second = FakeClient()

        with mock.patch.object(self.i18n, "_current_client", return_value=first):
            self.i18n.set_language("en")
            first_i18n = self.i18n.get_i18n()
            self.assertEqual(first.storage["language"], "en")

        with mock.patch.object(self.i18n, "_current_client", return_value=second):
            second_i18n = self.i18n.get_i18n()
            self.assertEqual(second_i18n.lang, "zh")

        with mock.patch.object(self.i18n, "_current_client", return_value=first):
            self.assertIs(self.i18n.get_i18n(), first_i18n)
            self.assertEqual(self.i18n.get_i18n().lang, "en")

        self.assertIsNot(first_i18n, second_i18n)

    def test_language_survives_navigation_for_same_user(self):
        first_page = FakeClient()
        next_page = FakeClient()
        user_storage = {}

        with (
            mock.patch.object(self.i18n, "_current_client", return_value=first_page),
            mock.patch.object(self.i18n, "_current_user_storage", return_value=user_storage, create=True),
        ):
            self.i18n.set_language("en")

        with (
            mock.patch.object(self.i18n, "_current_client", return_value=next_page),
            mock.patch.object(self.i18n, "_current_user_storage", return_value=user_storage, create=True),
        ):
            next_page_language = self.i18n.get_i18n().lang

        self.assertEqual(next_page_language, "en")

    def test_translation_before_ui_start_does_not_enable_script_mode(self):
        script = textwrap.dedent(
            f"""
            import sys

            sys.path.insert(0, {str(self.GUI_ROOT)!r})

            from nicegui import core
            from utils.i18n import t

            assert not core.script_mode
            t("app_title")
            assert not core.script_mode, "translation lookup activated NiceGUI script mode"
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=self.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gui_storage_secret_is_stable_across_process_restarts(self):
        from main import _resolve_storage_secret

        with tempfile.TemporaryDirectory() as tmp:
            storage_dir = Path(tmp)
            with mock.patch.dict("os.environ", {"MUSUBI_GUI_STORAGE_SECRET": ""}):
                first = _resolve_storage_secret(storage_dir)
                second = _resolve_storage_secret(storage_dir)

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 32)

    def test_gui_storage_path_is_independent_of_launch_directory(self):
        script = textwrap.dedent(
            f"""
            import sys

            sys.path.insert(0, {str(self.GUI_ROOT)!r})

            import main
            from nicegui.storage import Storage

            print(Storage.path)
            """
        )
        env = os.environ.copy()
        env.pop("NICEGUI_STORAGE_PATH", None)
        env["MUSUBI_GUI_STORAGE_SECRET"] = "test-storage-secret"

        resolved_paths = []
        for cwd in (self.ROOT, self.GUI_ROOT):
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            resolved_paths.append(Path(result.stdout.strip()))

        expected = (self.ROOT / ".nicegui").resolve()
        self.assertEqual(resolved_paths, [expected, expected])


if __name__ == "__main__":
    unittest.main()
