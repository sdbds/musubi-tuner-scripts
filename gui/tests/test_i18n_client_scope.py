import sys
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


if __name__ == "__main__":
    unittest.main()
