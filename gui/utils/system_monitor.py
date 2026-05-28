"""Lightweight system resource probes for the GUI monitor dialog."""

from __future__ import annotations

import csv
import shutil
import subprocess
from io import StringIO
from typing import Any, Callable

try:
    import psutil
except ImportError:  # pragma: no cover - exercised by runtime environments without psutil
    psutil = None


NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
    "--format=csv,noheader,nounits",
]


def _float_or_none(value: str) -> float | None:
    text = str(value or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_nvidia_smi_csv(output: str) -> list[dict[str, Any]]:
    """Parse `nvidia-smi --query-gpu ... --format=csv,noheader,nounits` output."""
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(StringIO(output or "")):
        fields = [field.strip() for field in row]
        if len(fields) < 8:
            continue

        index, name, util, mem_used, mem_total, temp, power_draw, power_limit = fields[:8]
        gpus.append(
            {
                "index": index,
                "name": name,
                "utilization_percent": _float_or_none(util),
                "memory_used_mib": _float_or_none(mem_used),
                "memory_total_mib": _float_or_none(mem_total),
                "temperature_c": _float_or_none(temp),
                "power_draw_w": _float_or_none(power_draw),
                "power_limit_w": _float_or_none(power_limit),
            }
        )
    return gpus


def query_nvidia_gpus(
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    timeout: float = 1.5,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return NVIDIA GPU metrics plus an optional degraded-state message."""
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return [], "nvidia-smi not found"

    try:
        result = run(
            [nvidia_smi_path, *NVIDIA_SMI_QUERY[1:]],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], f"nvidia-smi unavailable: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return [], detail or f"nvidia-smi exited with {result.returncode}"

    return parse_nvidia_smi_csv(result.stdout), None


def collect_cpu_metrics() -> dict[str, Any]:
    if psutil is None:
        return {"available": False, "message": "psutil not installed"}

    freq = psutil.cpu_freq()
    return {
        "available": True,
        "percent": psutil.cpu_percent(interval=None),
        "logical_count": psutil.cpu_count(logical=True),
        "physical_count": psutil.cpu_count(logical=False),
        "frequency_mhz": getattr(freq, "current", None) if freq else None,
        "temperature_c": collect_cpu_temperature_c(),
    }


def collect_cpu_temperature_c() -> float | None:
    if psutil is None:
        return None

    sensors_temperatures = getattr(psutil, "sensors_temperatures", None)
    if sensors_temperatures is None:
        return None

    try:
        sensor_groups = sensors_temperatures() or {}
    except (AttributeError, OSError, RuntimeError):
        return None

    candidates: list[tuple[int, float]] = []
    preferred_names = ("cpu", "core", "package", "k10temp", "zenpower", "acpitz")
    for sensor_name, entries in sensor_groups.items():
        sensor_text = str(sensor_name or "").lower()
        name_score = 1 if any(part in sensor_text for part in preferred_names) else 0
        for entry in entries or []:
            current = getattr(entry, "current", None)
            if current is None:
                continue
            try:
                value = float(current)
            except (TypeError, ValueError):
                continue
            if value <= 0:
                continue
            label_text = str(getattr(entry, "label", "") or "").lower()
            label_score = 1 if any(part in label_text for part in preferred_names) else 0
            candidates.append((name_score + label_score, value))

    if not candidates:
        return None

    best_score = max(score for score, _value in candidates)
    return max(value for score, value in candidates if score == best_score)


def collect_memory_metrics() -> dict[str, Any]:
    if psutil is None:
        return {"available": False, "message": "psutil not installed"}

    memory = psutil.virtual_memory()
    return {
        "available": True,
        "percent": memory.percent,
        "used_bytes": memory.used,
        "total_bytes": memory.total,
        "available_bytes": memory.available,
    }


def collect_system_metrics() -> dict[str, Any]:
    gpus, gpu_error = query_nvidia_gpus()
    return {
        "cpu": collect_cpu_metrics(),
        "memory": collect_memory_metrics(),
        "gpus": gpus,
        "gpu_error": gpu_error,
    }


def format_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024.0 or unit == "TiB":
            return f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{number:.1f} TiB"


def format_mib(value: int | float | None) -> str:
    if value is None:
        return "-"
    return format_bytes(float(value) * 1024 * 1024)
