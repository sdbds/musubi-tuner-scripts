import os
import importlib.util
import socket
import unittest
from pathlib import Path
from unittest import mock


class TestPortUtils(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[2]
    MODULE_PATH = ROOT / "gui" / "utils" / "port_utils.py"

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("gui_port_utils", cls.MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.port_utils = module

    def test_default_gui_port_is_7788(self):
        self.assertEqual(self.port_utils.DEFAULT_GUI_PORT, 7788)
        self.assertEqual(self.port_utils.parse_port(None), 7788)

    def test_default_gui_host_is_loopback(self):
        self.assertEqual(self.port_utils.DEFAULT_GUI_HOST, "127.0.0.1")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.port_utils.resolve_gui_host(), "127.0.0.1")

    def test_gui_host_can_be_overridden_explicitly(self):
        with mock.patch.dict(os.environ, {"MUSUBI_GUI_HOST": "0.0.0.0"}, clear=True):
            self.assertEqual(self.port_utils.resolve_gui_host(), "0.0.0.0")

    def test_parse_port_falls_back_on_invalid_values(self):
        self.assertEqual(self.port_utils.parse_port("not-a-port"), 7788)
        self.assertEqual(self.port_utils.parse_port("70000"), 7788)
        self.assertEqual(self.port_utils.parse_port(" 7789 "), 7789)

    def test_parse_bool_accepts_environment_style_values(self):
        self.assertTrue(self.port_utils.parse_bool("1"))
        self.assertTrue(self.port_utils.parse_bool("true"))
        self.assertFalse(self.port_utils.parse_bool("0", default=True))
        self.assertFalse(self.port_utils.parse_bool("false", default=True))
        self.assertTrue(self.port_utils.parse_bool("unknown", default=True))

    def test_find_available_port_prefers_requested_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            preferred_port = sock.getsockname()[1] + 1

        selected_port = self.port_utils.find_available_port("127.0.0.1", preferred_port, search_span=2)
        self.assertEqual(selected_port, preferred_port)

    def test_find_available_port_advances_when_port_is_busy(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            busy_port = sock.getsockname()[1]

            selected_port = self.port_utils.find_available_port("127.0.0.1", busy_port, search_span=3)

        self.assertGreaterEqual(selected_port, busy_port + 1)


if __name__ == "__main__":
    unittest.main()
