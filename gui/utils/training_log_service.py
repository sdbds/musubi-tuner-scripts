"""Training log discovery and TensorBoard process lifecycle helpers."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from utils.job_manager import Job, job_manager
from utils.port_utils import find_available_port, parse_port

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TENSORBOARD_PORT = 6006
WANDB_HOME_URL = "https://wandb.ai/home"
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^\s\"'<>]+")


@dataclass(frozen=True)
class TrainingLogContext:
    mode: str
    log_dir: Path
    wandb_url: str | None = None
    source_job_name: str | None = None


@dataclass(frozen=True)
class TensorBoardLaunch:
    available: bool
    url: str | None = None
    log_dir: Path | None = None
    port: int | None = None
    pid: int | None = None
    message: str = ""


def _arg_value(args: Sequence[str], flag: str) -> str | None:
    for index, arg in enumerate(args):
        text = str(arg)
        if text == flag and index + 1 < len(args):
            return str(args[index + 1])
        if text.startswith(f"{flag}="):
            return text.split("=", 1)[1]
    return None


def _is_training_job(job: Job) -> bool:
    text = f"{job.name} {job.script_key}".lower()
    return "train" in text or "训练" in text


def _job_uses_wandb(job: Job) -> bool:
    args = _job_args(job)
    return str(_arg_value(args, "--log_with") or "").strip().lower() == "wandb"


def _job_args(job: Job) -> Sequence[str]:
    args = getattr(job, "args", [])
    return args if isinstance(args, list) else []


def latest_training_jobs(jobs: Iterable[Job] | None = None) -> list[Job]:
    source = list(jobs) if jobs is not None else job_manager.get_all_jobs()
    return [job for job in source if _is_training_job(job)]


def find_wandb_url(lines: Iterable[str]) -> str | None:
    for line in lines:
        match = _WANDB_URL_RE.search(str(line))
        if match:
            return match.group(0).rstrip(").,;]")
    return None


def find_wandb_url_from_job(job: Job) -> str | None:
    history = getattr(job.log_buffer, "get_all_lines", lambda: [])()
    return find_wandb_url(line for _seq, line in history)


def resolve_log_dir_from_args(args: Sequence[str], project_root: Path = PROJECT_ROOT) -> Path:
    raw = _arg_value(args, "--logging_dir")
    if not raw:
        return project_root / "logs"
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path


def resolve_training_log_context(
    *,
    jobs: Iterable[Job] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> TrainingLogContext:
    training_jobs = latest_training_jobs(jobs)
    if not training_jobs:
        return TrainingLogContext(mode="tensorboard", log_dir=project_root / "logs")

    log_dir = project_root / "logs"
    for job in training_jobs:
        args = _job_args(job)
        if args:
            log_dir = resolve_log_dir_from_args(args, project_root)
            break

    for job in training_jobs:
        if _job_uses_wandb(job):
            return TrainingLogContext(
                mode="wandb",
                log_dir=log_dir,
                wandb_url=find_wandb_url_from_job(job),
                source_job_name=job.name,
            )

    for job in training_jobs:
        wandb_url = find_wandb_url_from_job(job)
        if wandb_url:
            return TrainingLogContext(
                mode="wandb",
                log_dir=log_dir,
                wandb_url=wandb_url,
                source_job_name=job.name,
            )

    return TrainingLogContext(mode="tensorboard", log_dir=log_dir, source_job_name=training_jobs[0].name)


class TensorBoardService:
    """Small singleton manager for a background TensorBoard process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._log_dir: Path | None = None
        self._port: int | None = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def ensure_started(self, log_dir: str | Path, *, preferred_port: int | None = None) -> TensorBoardLaunch:
        path = Path(log_dir).resolve()
        if self.is_running() and self._log_dir == path and self._port is not None:
            return TensorBoardLaunch(
                available=True,
                url=f"http://127.0.0.1:{self._port}",
                log_dir=path,
                port=self._port,
                pid=self._process.pid if self._process else None,
                message="TensorBoard is already running",
            )

        if self.is_running():
            self.stop()

        if importlib.util.find_spec("tensorboard") is None:
            return TensorBoardLaunch(available=False, log_dir=path, message="tensorboard is not installed")

        path.mkdir(parents=True, exist_ok=True)
        port = find_available_port("127.0.0.1", preferred_port or _preferred_tensorboard_port())
        cmd = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except OSError as exc:
            self._process = None
            return TensorBoardLaunch(available=False, log_dir=path, port=port, message=str(exc))

        self._log_dir = path
        self._port = port
        return TensorBoardLaunch(
            available=True,
            url=f"http://127.0.0.1:{port}",
            log_dir=path,
            port=port,
            pid=self._process.pid,
            message="TensorBoard started",
        )

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
        self._process = None
        self._log_dir = None
        self._port = None


def _preferred_tensorboard_port() -> int:
    return parse_port(os.getenv("MUSUBI_TENSORBOARD_PORT"), DEFAULT_TENSORBOARD_PORT)


tensorboard_service = TensorBoardService()
