import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils import training_log_service  # noqa: E402


class FakeLogBuffer:
    def __init__(self, lines):
        self._lines = [(index + 1, line) for index, line in enumerate(lines)]

    def get_all_lines(self):
        return list(self._lines)


def fake_job(name, args=None, lines=None):
    return SimpleNamespace(
        name=name,
        script_key="musubi_tuner.example_train_network",
        args=args or [],
        log_buffer=FakeLogBuffer(lines or []),
    )


class TestTrainingLogService(unittest.TestCase):
    def test_find_wandb_url_strips_trailing_punctuation(self):
        url = training_log_service.find_wandb_url(["View run at https://wandb.ai/user/proj/runs/abc123)."])

        self.assertEqual(url, "https://wandb.ai/user/proj/runs/abc123")

    def test_resolve_training_log_context_prefers_wandb_job(self):
        job = fake_job(
            "Wan Train",
            args=["--log_with=wandb", "--logging_dir=./logs/custom"],
            lines=["wandb: View run at https://wandb.ai/user/project/runs/run-id"],
        )

        context = training_log_service.resolve_training_log_context(jobs=[job], project_root=ROOT)

        self.assertEqual(context.mode, "wandb")
        self.assertEqual(context.wandb_url, "https://wandb.ai/user/project/runs/run-id")
        self.assertEqual(context.log_dir, ROOT / "logs" / "custom")

    def test_resolve_training_log_context_defaults_to_tensorboard_logs(self):
        context = training_log_service.resolve_training_log_context(jobs=[], project_root=ROOT)

        self.assertEqual(context.mode, "tensorboard")
        self.assertEqual(context.log_dir, ROOT / "logs")

    def test_tensorboard_service_reuses_existing_process_for_same_log_dir(self):
        class FakeProcess:
            pid = 12345

            def __init__(self):
                self.terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

        fake_process = FakeProcess()
        service = training_log_service.TensorBoardService()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(training_log_service.importlib.util, "find_spec", return_value=object()),
                mock.patch.object(training_log_service, "find_available_port", return_value=6123),
                mock.patch.object(training_log_service.subprocess, "Popen", return_value=fake_process) as popen,
            ):
                first = service.ensure_started(tmp_dir)
                second = service.ensure_started(tmp_dir)

        self.assertTrue(first.available)
        self.assertEqual(first.url, "http://127.0.0.1:6123")
        self.assertEqual(second.pid, 12345)
        self.assertEqual(popen.call_count, 1)
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[:3], [sys.executable, "-m", "tensorboard.main"])


if __name__ == "__main__":
    unittest.main()

