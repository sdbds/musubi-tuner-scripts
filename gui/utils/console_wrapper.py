"""Native console wrapper for GUI-launched jobs.

The GUI starts this script in a new PowerShell window. The wrapper then runs
the real command, forwards output to the visible console, mirrors it to a log
file for the NiceGUI LogViewer, and writes the final exit code to a signal
file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


_COLOR_INJECT_DIR = str(Path(__file__).parent / "_color_inject")


def _normalize_color_system(value: str) -> str:
    normalized = (value or "").strip().lower()
    aliases = {
        "24bit": "truecolor",
        "24-bit": "truecolor",
        "full": "truecolor",
        "256color": "256",
    }
    return aliases.get(normalized, normalized)


def _setup_windows_console() -> int:
    """Enable UTF-8/ANSI output and return a usable console column count."""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)

        std_output_handle = ctypes.c_ulong(-11)
        enable_virtual_terminal_processing = 0x0004
        handle = kernel32.GetStdHandle(std_output_handle)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            mode.value |= enable_virtual_terminal_processing
            kernel32.SetConsoleMode(handle, mode)

        cols = 160
        subprocess.run(
            f"mode con cols={cols}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return cols
    except Exception:
        return 0


def _build_child_env(log_file: str, console_cols: int) -> dict:
    env = os.environ.copy()
    color_system = _normalize_color_system(env.get("_MUSUBI_RICH_COLOR_SYSTEM", "truecolor"))

    env["FORCE_COLOR"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["_MUSUBI_FORCE_TTY"] = "1"

    if color_system == "truecolor":
        env["COLORTERM"] = "truecolor"
        env["TERM"] = "xterm-256color"
    elif color_system == "256":
        env.pop("COLORTERM", None)
        env["TERM"] = "xterm-256color"
    elif color_system == "standard":
        env.pop("COLORTERM", None)
        env["TERM"] = "xterm"
    elif color_system == "windows":
        env.pop("COLORTERM", None)
        env["TERM"] = "windows"
    elif color_system == "auto":
        pass
    else:
        env["COLORTERM"] = "truecolor"
        env["TERM"] = "xterm-256color"

    if console_cols > 0:
        env["COLUMNS"] = str(console_cols)

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        _COLOR_INJECT_DIR + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else _COLOR_INJECT_DIR
    )
    env["_MUSUBI_GUI_LOG_FILE"] = log_file
    return env


def _write_exit_code(exit_file: str, return_code: int) -> None:
    try:
        with open(exit_file, "w", encoding="utf-8") as handle:
            handle.write(str(return_code))
    except OSError:
        pass


def main() -> int:
    if len(sys.argv) < 4:
        print("Usage: console_wrapper.py <exit_file> <log_file> <command...>")
        return 1

    exit_file = sys.argv[1]
    log_file = sys.argv[2]
    cmd = sys.argv[3:]

    console_cols = _setup_windows_console() if sys.platform == "win32" else 0
    env = _build_child_env(log_file, console_cols)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        bufsize=0,
    )

    try:
        with open(log_file, "ab") as log_handle:
            while True:
                chunk = proc.stdout.read(4096) if proc.stdout else b""
                if not chunk:
                    break
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except Exception:
                    pass
                try:
                    log_handle.write(chunk)
                    log_handle.flush()
                except Exception:
                    pass
    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()
    return_code = int(proc.returncode or 0)
    _write_exit_code(exit_file, return_code)

    status = "成功" if return_code == 0 else f"失败 (返回码: {return_code})"
    print(f"\n{'=' * 50}")
    print(f"  任务{status}")
    print("  按 Enter 关闭此窗口...")
    print(f"{'=' * 50}")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
