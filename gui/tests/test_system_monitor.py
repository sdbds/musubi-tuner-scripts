import subprocess
import sys
import unittest
from pathlib import Path
from collections import namedtuple
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils import system_monitor  # noqa: E402


class TestSystemMonitor(unittest.TestCase):
    def test_parse_nvidia_smi_csv(self):
        output = "0, NVIDIA GeForce RTX 4090, 87, 21000, 24564, 68, 320.5, 450.0\n"

        gpus = system_monitor.parse_nvidia_smi_csv(output)

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["index"], "0")
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(gpus[0]["utilization_percent"], 87.0)
        self.assertEqual(gpus[0]["memory_used_mib"], 21000.0)
        self.assertEqual(gpus[0]["memory_total_mib"], 24564.0)
        self.assertEqual(gpus[0]["temperature_c"], 68.0)
        self.assertEqual(gpus[0]["power_draw_w"], 320.5)

    def test_parse_nvidia_smi_csv_preserves_multiple_gpus(self):
        output = (
            "0, NVIDIA GeForce RTX 4090, 87, 21000, 24564, 68, 320.5, 450.0\n"
            "1, NVIDIA GeForce RTX 3090, 12, 8000, 24576, 51, 110.0, 350.0\n"
        )

        gpus = system_monitor.parse_nvidia_smi_csv(output)

        self.assertEqual([gpu["index"] for gpu in gpus], ["0", "1"])
        self.assertEqual(gpus[1]["name"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(gpus[1]["utilization_percent"], 12.0)

    def test_query_nvidia_gpus_degrades_when_command_missing(self):
        with mock.patch.object(system_monitor.shutil, "which", return_value=None):
            gpus, error = system_monitor.query_nvidia_gpus()

        self.assertEqual(gpus, [])
        self.assertEqual(error, "nvidia-smi not found")

    def test_query_nvidia_gpus_uses_structured_query(self):
        def fake_run(cmd, **kwargs):
            self.assertEqual(cmd, [r"C:\Tools\nvidia-smi.exe", *system_monitor.NVIDIA_SMI_QUERY[1:]])
            self.assertTrue(kwargs["capture_output"])
            return subprocess.CompletedProcess(cmd, 0, stdout="0, RTX, 12, 100, 1000, 40, N/A, N/A\n", stderr="")

        with mock.patch.object(system_monitor.shutil, "which", return_value=r"C:\Tools\nvidia-smi.exe"):
            gpus, error = system_monitor.query_nvidia_gpus(run=fake_run)

        self.assertIsNone(error)
        self.assertEqual(gpus[0]["utilization_percent"], 12.0)
        self.assertIsNone(gpus[0]["power_draw_w"])

    def test_collect_cpu_temperature_degrades_when_sensor_api_missing(self):
        with mock.patch.object(system_monitor.psutil, "sensors_temperatures", None, create=True):
            self.assertIsNone(system_monitor.collect_cpu_temperature_c())

    def test_collect_cpu_temperature_prefers_cpu_sensor_peak(self):
        Sensor = namedtuple("Sensor", "label current")

        def fake_sensors():
            return {
                "nvme": [Sensor("Composite", 43.0)],
                "coretemp": [Sensor("Core 0", 71.0), Sensor("Core 1", 74.0)],
            }

        with mock.patch.object(system_monitor.psutil, "sensors_temperatures", fake_sensors, create=True):
            self.assertEqual(system_monitor.collect_cpu_temperature_c(), 74.0)


if __name__ == "__main__":
    unittest.main()
