import asyncio
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils.job_manager import JobManager, JobStatus  # noqa: E402
from utils.process_runner import ProcessResult, ProcessStatus  # noqa: E402


class TestJobManagerCancel(unittest.TestCase):
    def test_cancel_terminates_runner_without_cancelling_wait_task(self):
        async def scenario():
            import utils.process_runner as process_runner_module

            created_runners = []
            original_runner = process_runner_module.ProcessRunner

            class FakeRunner:
                def __init__(self, log_buffer=None):
                    self.log_buffer = log_buffer
                    self.terminate_called = False
                    self.release = asyncio.Event()
                    created_runners.append(self)

                async def run_python_script(self, script_key, args, **runner_kwargs):
                    await self.release.wait()
                    return ProcessResult(ProcessStatus.ERROR, -1, "stopped")

                def terminate(self):
                    self.terminate_called = True
                    self.release.set()

            process_runner_module.ProcessRunner = FakeRunner
            try:
                manager = JobManager()
                job = await manager.submit("example.module", [], name="Example")
                await asyncio.sleep(0)

                self.assertTrue(manager.cancel(job.id))
                self.assertEqual(job.status, JobStatus.RUNNING)
                self.assertTrue(job.cancel_requested)
                self.assertFalse(job._task.cancelled())

                result = await job.wait()

                self.assertEqual(result.return_code, -1)
                self.assertEqual(job.status, JobStatus.CANCELLED)
                self.assertTrue(created_runners[0].terminate_called)
            finally:
                process_runner_module.ProcessRunner = original_runner

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
