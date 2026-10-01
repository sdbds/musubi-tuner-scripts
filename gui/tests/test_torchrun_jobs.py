import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))

from utils.job_manager import JobManager, JobStatus  # noqa: E402
from utils.process_runner import ProcessRunner  # noqa: E402


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "2"])
def test_torchrun_rejects_invalid_counts_before_starting(count):
    runner = ProcessRunner()
    with pytest.raises(ValueError, match="num_processes"):
        asyncio.run(runner.run_torchrun("example.module", [], num_processes=count, native_console=False))
    assert not runner.is_running


def test_conflicting_launchers_are_rejected_before_queuing_a_job():
    manager = JobManager()
    with pytest.raises(ValueError, match="launcher"):
        asyncio.run(manager.submit("example.module", [], "Invalid", use_accelerate=True, use_torchrun=True))
    assert manager.get_all_jobs() == []


def test_job_manager_launches_two_cpu_ranks_with_a_real_collective(tmp_path):
    async def run():
        manager = JobManager()
        job = await manager.submit(
            "dlssnr_ddp_probe",
            [str(tmp_path)],
            "CPU DDP probe",
            use_torchrun=True,
            num_processes=2,
            cwd=str(ROOT / "gui/tests/fixtures"),
            native_console=False,
            env_vars={"CUDA_VISIBLE_DEVICES": "", "USE_LIBUV": "0", "OMP_NUM_THREADS": "1"},
        )
        try:
            result = await asyncio.wait_for(asyncio.shield(job.wait()), timeout=90)
        finally:
            if job.runner.is_running:
                manager.cancel(job.id)
                await job.wait()
        assert job.status == JobStatus.SUCCESS, "\n".join(line for _, line in job.log_buffer.get_all_lines())
        assert result.return_code == 0
        assert manager.get_active_jobs() == []

    asyncio.run(run())
    assert [json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in (0, 1)] == [
        {"rank": 0, "world_size": 2, "sum": 3.0},
        {"rank": 1, "world_size": 2, "sum": 3.0},
    ]
