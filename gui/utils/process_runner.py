"""进程运行工具 - 管理外部 Python 脚本的调用和日志输出

关键：使用 asyncio.create_subprocess_exec 进行非阻塞 I/O，
避免阻塞 NiceGUI 事件循环导致 WebSocket 断开。
"""

import asyncio
import codecs
import shutil
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

from utils.log_buffer import log_buffer as _global_log_buffer, LogBuffer
from utils.env_config import get_env_for_subprocess


_WRAPPER_PATH = str(Path(__file__).parent / "console_wrapper.py")
_SENSITIVE_ENV_MARKERS = ("TOKEN", "APIKEY", "API_KEY", "SECRET", "PASSWORD", "ACCESS_KEY", "PRIVATE_KEY")


def _ps_escape(value: str) -> str:
    """Escape a string for a single-quoted PowerShell argument."""
    return str(value).replace("'", "''")


def _powershell_call(parts: List[str]) -> str:
    """Render argv as a PowerShell call expression."""
    return "& " + " ".join(f"'{_ps_escape(part)}'" for part in parts)


class ProcessStatus(Enum):
    IDLE = "空闲"
    RUNNING = "运行中"
    SUCCESS = "成功"
    ERROR = "失败"


@dataclass
class ProcessResult:
    status: ProcessStatus
    return_code: int = 0
    message: str = ""


class ProcessRunner:
    """运行外部进程并捕获输出（非阻塞）

    使用 asyncio.create_subprocess_exec 保证不阻塞 NiceGUI 事件循环。
    """

    TASK_DIVIDER = "================ New Task ================"

    # 项目根目录 (gui/utils/../../ = 项目根)
    PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())

    def __init__(self, log_buffer: Optional[LogBuffer] = None):
        """
        Args:
            log_buffer: 可选，指定此实例的日志缓冲区。
                        若未提供，使用全局 log_buffer（向后兼容）。
        """
        self._log_buffer: LogBuffer = log_buffer if log_buffer is not None else _global_log_buffer
        self.process = None
        self.status_callback: Optional[Callable[[ProcessStatus], None]] = None
        self._running = False
        self._task_divider_emitted = False
        self._tail_task: Optional[asyncio.Task] = None
        self._last_gui_env_overrides: dict = {}

    def set_callbacks(
        self,
        log_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[ProcessStatus], None]] = None,
    ):
        """设置回调函数（log_callback 已废弃，保留参数以兼容旧调用方）"""
        self.status_callback = status_callback

    def _notify_log(self, message: str):
        """推送日志到此实例的 log_buffer（订阅者自动收到）"""
        self._log_buffer.push(message)

    def _notify_status(self, status: ProcessStatus):
        """通知状态回调"""
        if self.status_callback:
            self.status_callback(status)

    def begin_task_log(self):
        """在保留历史的前提下，为新任务插入分隔线。"""
        if self._task_divider_emitted:
            return
        if self._log_buffer.get_all_lines():
            self._notify_log(self.TASK_DIVIDER)
        self._task_divider_emitted = True

    def log(self, message: str):
        """推送一条日志到共享缓冲区。"""
        self._notify_log(message)

    @staticmethod
    def _extract_output_lines(text: str, pending: str = "") -> tuple[List[str], str]:
        """Split subprocess output on LF and CR progress-bar refreshes.

        Pipes do not apply terminal carriage-return semantics. Without this,
        tqdm writes like ``step 1\rstep 2\r...`` arrive as one very long GUI
        log line. Treat CR as a line boundary so the browser log remains
        readable.
        """
        if not text:
            return [], pending

        normalized = (pending + text).replace("\r\n", "\n").replace("\r", "\n")
        parts = normalized.split("\n")

        if normalized.endswith("\n"):
            complete = parts[:-1]
            next_pending = ""
        else:
            complete = parts[:-1]
            next_pending = parts[-1]

        return [line.rstrip() for line in complete if line.strip()], next_pending

    def _emit_output_chunk(self, text: str, pending: str) -> str:
        lines, pending = self._extract_output_lines(text, pending)
        for line in lines:
            self._notify_log(line)
        return pending

    def _flush_pending_output(self, pending: str) -> None:
        if pending.strip():
            self._notify_log(pending.rstrip())

    def _build_env(self, env_vars: Optional[dict] = None) -> dict:
        """构建子进程环境变量"""
        env = os.environ.copy()

        # 强制彩色输出
        env["FORCE_COLOR"] = "1"
        env["COLORTERM"] = "truecolor"
        env["TERM"] = "xterm-256color"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        # PYTHONPATH: 确保项目根目录和常用源码根目录都在搜索路径中
        existing = env.get("PYTHONPATH", "")
        extra_roots = [
            self.PROJECT_ROOT,
            str(Path(self.PROJECT_ROOT) / "musubi-tuner" / "src"),
            str(Path(self.PROJECT_ROOT) / "qinglong-captions"),
        ]
        pythonpath_parts = [part for part in existing.split(os.pathsep) if part]
        for root in reversed(extra_roots):
            if root not in pythonpath_parts:
                pythonpath_parts.insert(0, root)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        gui_env_overrides = get_env_for_subprocess()
        self._last_gui_env_overrides = dict(gui_env_overrides)
        env.update(gui_env_overrides)

        # 调用方传入的 env_vars 优先级最高
        if env_vars:
            env.update(env_vars)

        return env

    @staticmethod
    def _is_sensitive_env_key(key: str) -> bool:
        upper_key = key.upper()
        return any(marker in upper_key for marker in _SENSITIVE_ENV_MARKERS)

    @classmethod
    def _format_env_for_log(cls, env: dict) -> list[str]:
        lines: list[str] = []
        omitted = 0
        for key in sorted(env):
            if cls._is_sensitive_env_key(key):
                omitted += 1
                continue
            lines.append(f"  {key}={env.get(key, '')}")
        if omitted:
            lines.append(f"  <{omitted} sensitive environment variable(s) omitted>")
        return lines or ["  <empty>"]

    @staticmethod
    def _format_command_args_for_log(cmd: list[str]) -> list[str]:
        lines: list[str] = []
        index = 0
        while index < len(cmd):
            part = str(cmd[index])
            if not part.startswith("--"):
                index += 1
                continue

            if "=" in part or index + 1 >= len(cmd):
                lines.append(f"  {part}")
                index += 1
                continue

            next_part = str(cmd[index + 1])
            if next_part.startswith("--"):
                lines.append(f"  {part}")
                index += 1
                continue

            lines.append(f"  {part} {next_part}")
            index += 2

        return lines or ["  <no -- parameters>"]

    def _build_python_command(self, script_module: str, args: List[str]) -> List[str]:
        """构建 Python 调用命令，支持模块路径和脚本路径。"""
        candidate = Path(script_module)
        absolute_candidate = candidate if candidate.is_absolute() else Path(self.PROJECT_ROOT) / candidate

        if script_module.endswith(".py") or absolute_candidate.exists():
            script_path = absolute_candidate if absolute_candidate.exists() else candidate
            return [sys.executable, str(script_path)] + args

        return [sys.executable, "-m", script_module] + args

    # ------------------------------------------------------------------
    #  非阻塞读取子进程输出
    # ------------------------------------------------------------------
    async def _stream_output(self, proc: asyncio.subprocess.Process):
        """非阻塞读取 stdout，并把 LF/CR 分隔的日志推送到 log_buffer"""
        assert proc.stdout is not None
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            pending = self._emit_output_chunk(text, pending)
        tail = decoder.decode(b"", final=True)
        if tail:
            pending = self._emit_output_chunk(tail, pending)
        self._flush_pending_output(pending)

    async def _stream_output_popen(self, proc: subprocess.Popen):
        """在 selector event loop 下通过线程读取 Popen 输出。"""
        assert proc.stdout is not None
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await asyncio.to_thread(proc.stdout.read, 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            pending = self._emit_output_chunk(text, pending)
        tail = decoder.decode(b"", final=True)
        if tail:
            pending = self._emit_output_chunk(tail, pending)
        self._flush_pending_output(pending)

    @staticmethod
    def _requires_threaded_subprocess() -> bool:
        if sys.platform != "win32":
            return False
        return type(asyncio.get_event_loop_policy()).__name__ == "WindowsSelectorEventLoopPolicy"

    async def _run_pipe_with_popen(self, cmd: List[str], work_dir: Path, env: dict) -> int:
        """Windows reload 模式下，避免 asyncio 子进程不可用。"""
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(work_dir),
            env=env,
            bufsize=0,
        )
        await self._stream_output_popen(self.process)
        return await asyncio.to_thread(self.process.wait)

    async def _run_logged_subprocess(self, cmd: List[str], work_dir: Path, env: dict) -> int:
        """在 GUI 日志中同步显示命令输出。"""
        if self._requires_threaded_subprocess():
            self._notify_log("检测到 Windows reload 模式，使用线程子进程回退")
            return await self._run_pipe_with_popen(cmd, work_dir, env)

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(work_dir),
                env=env,
            )
        except NotImplementedError:
            self._notify_log("当前事件循环不支持 asyncio 子进程，回退到线程子进程模式")
            return await self._run_pipe_with_popen(cmd, work_dir, env)

        await self._stream_output(self.process)
        return await self.process.wait()

    async def _tail_log_file(self, log_file: str, exit_file: str) -> None:
        """Tail the native console mirror log into the GUI log buffer."""
        offset = 0
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while self._running and not os.path.exists(exit_file):
                if os.path.exists(log_file):
                    with open(log_file, "rb") as handle:
                        handle.seek(offset)
                        chunk = handle.read()
                    if chunk:
                        offset += len(chunk)
                        pending = self._emit_output_chunk(decoder.decode(chunk), pending)
                await asyncio.sleep(0.3)

            if os.path.exists(log_file):
                with open(log_file, "rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                if chunk:
                    pending = self._emit_output_chunk(decoder.decode(chunk), pending)
            tail = decoder.decode(b"", final=True)
            if tail:
                pending = self._emit_output_chunk(tail, pending)
            self._flush_pending_output(pending)
        except asyncio.CancelledError:
            tail = decoder.decode(b"", final=True)
            if tail:
                pending = self._emit_output_chunk(tail, pending)
            self._flush_pending_output(pending)
            raise
        except Exception as exc:
            self._notify_log(f"日志尾随失败: {type(exc).__name__}: {exc}")

    @staticmethod
    def _native_console_supported() -> bool:
        return sys.platform == "win32"

    @staticmethod
    def _build_native_wrapper_env(env: dict, console_color_system: Optional[str]) -> dict:
        wrapper_env = env.copy()
        if console_color_system:
            wrapper_env["_MUSUBI_RICH_COLOR_SYSTEM"] = console_color_system
        else:
            wrapper_env.pop("_MUSUBI_RICH_COLOR_SYSTEM", None)
        return wrapper_env

    async def _run_native(
        self,
        cmd: List[str],
        work_dir: Path,
        env: dict,
        console_color_system: Optional[str] = "truecolor",
    ) -> int:
        """Run a command in a new PowerShell console and mirror output to GUI."""
        tmp_dir = tempfile.gettempdir()
        token = f"{os.getpid()}_{uuid4().hex}"
        exit_file = os.path.join(tmp_dir, f"musubi_gui_exit_{token}.tmp")
        log_file = os.path.join(tmp_dir, f"musubi_gui_log_{token}.tmp")

        for path in (exit_file, log_file):
            try:
                os.unlink(path)
            except OSError:
                pass

        parts = [sys.executable, _WRAPPER_PATH, exit_file, log_file] + cmd
        ps_exe = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
        wrapper_cmd = [
            ps_exe,
            "-NoProfile",
            "-NoLogo",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            _powershell_call(parts),
        ]

        self.process = subprocess.Popen(
            wrapper_cmd,
            cwd=str(work_dir),
            env=self._build_native_wrapper_env(env, console_color_system),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        self._notify_log("已在 PowerShell 控制台窗口中启动，输出会同步镜像到 GUI 日志")

        self._tail_task = asyncio.create_task(self._tail_log_file(log_file, exit_file))

        while not os.path.exists(exit_file):
            if not self._running:
                break
            if self.process and self.process.poll() is not None:
                break
            await asyncio.sleep(0.5)

        if os.path.exists(exit_file):
            try:
                with open(exit_file, "r", encoding="utf-8") as handle:
                    return_code = int(handle.read().strip())
            except (OSError, ValueError):
                return_code = -1
        elif self.process and self.process.returncode is not None:
            return_code = self.process.returncode
        else:
            return_code = -1

        if self._tail_task:
            try:
                await asyncio.wait_for(self._tail_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._tail_task.cancel()
            finally:
                self._tail_task = None

        for path in (exit_file, log_file):
            try:
                os.unlink(path)
            except OSError:
                pass

        return return_code

    # ------------------------------------------------------------------
    #  公共执行核心
    # ------------------------------------------------------------------
    async def _run_with_status(
        self,
        cmd: List[str],
        cwd: Optional[str],
        env_vars: Optional[dict],
        native_console: bool = True,
        console_color_system: Optional[str] = "truecolor",
    ) -> ProcessResult:
        """构建环境、打印头部信息、执行子进程并返回 ProcessResult。
        调用方负责在调用前设置 self._running=True 并 emit begin_task_log()。
        """
        try:
            work_dir = Path(cwd) if cwd else Path(self.PROJECT_ROOT)
            env = self._build_env(env_vars)

            self._notify_log("启动命令参数:")
            for line in self._format_command_args_for_log(cmd):
                self._notify_log(line)
            self._notify_log(f"工作目录: {work_dir.absolute()}")
            self._notify_log("GUI 配置环境变量:")
            for line in self._format_env_for_log(self._last_gui_env_overrides):
                self._notify_log(line)
            self._notify_log("=" * 60)

            if native_console and self._native_console_supported():
                return_code = await self._run_native(cmd, work_dir, env, console_color_system)
            else:
                return_code = await self._run_logged_subprocess(cmd, work_dir, env)
            self._running = False

            if return_code == 0:
                self._notify_status(ProcessStatus.SUCCESS)
                return ProcessResult(ProcessStatus.SUCCESS, return_code, "执行成功")
            else:
                self._notify_status(ProcessStatus.ERROR)
                return ProcessResult(ProcessStatus.ERROR, return_code, f"进程返回错误码: {return_code}")

        except Exception as e:
            self._running = False
            self._notify_status(ProcessStatus.ERROR)
            detail = str(e) or repr(e)
            error_msg = f"执行出错: {type(e).__name__}: {detail}"
            self._notify_log(error_msg)
            return ProcessResult(ProcessStatus.ERROR, -1, error_msg)
        finally:
            self._task_divider_emitted = False
            if self._tail_task and not self._tail_task.done():
                self._tail_task.cancel()
            self._tail_task = None
            self.process = None

    # ------------------------------------------------------------------
    #  主要运行方法
    # ------------------------------------------------------------------
    async def run_python_script(
        self,
        script_module: str,  # 如: "musubi_tuner.flux_2_train_network"
        args: List[str],
        cwd: Optional[str] = None,
        env_vars: Optional[dict] = None,
        native_console: bool = True,
        console_color_system: Optional[str] = "truecolor",
    ) -> ProcessResult:
        """异步运行 Python 模块（完全非阻塞）

        Args:
            script_module: Python 模块路径，如 'musubi_tuner.cache_latents'
            args: 传递给脚本的参数列表
            cwd: 工作目录（默认项目根目录）
            env_vars: 额外环境变量
        """
        if self._running:
            return ProcessResult(ProcessStatus.ERROR, -1, "已有任务在运行")

        self._running = True
        self._notify_status(ProcessStatus.RUNNING)
        self.begin_task_log()

        cmd = self._build_python_command(script_module, args)
        return await self._run_with_status(cmd, cwd, env_vars, native_console, console_color_system)

    # ------------------------------------------------------------------
    #  accelerate 运行 (训练用)
    # ------------------------------------------------------------------
    async def run_accelerate(
        self,
        script_module: str,  # 如: "musubi_tuner.flux_2_train_network"
        args: List[str],
        num_cpu_threads_per_process: int = 1,
        mixed_precision: str = "bf16",
        accelerate_args: Optional[List[str]] = None,
        cwd: Optional[str] = None,
        env_vars: Optional[dict] = None,
        native_console: bool = True,
        console_color_system: Optional[str] = "truecolor",
    ) -> ProcessResult:
        """使用 accelerate 运行训练脚本（非阻塞）"""
        if self._running:
            return ProcessResult(ProcessStatus.ERROR, -1, "已有任务在运行")

        self._running = True
        self._notify_status(ProcessStatus.RUNNING)
        self.begin_task_log()

        cmd = [
            sys.executable, "-m", "accelerate.commands.launch",
            f"--num_cpu_threads_per_process={num_cpu_threads_per_process}",
        ]
        if mixed_precision:
            cmd.append(f"--mixed_precision={mixed_precision}")
        if accelerate_args:
            cmd.extend(accelerate_args)
        cmd.extend(script_module.split())
        cmd.extend(args)

        return await self._run_with_status(cmd, cwd, env_vars, native_console, console_color_system)

    def run_script_sync(self, script_module: str, args: List[str], **kwargs) -> ProcessResult:
        """同步运行脚本（仅供独立测试脚本使用）。
        警告：在 NiceGUI 的 asyncio 事件循环中调用此方法会抛出
        'This event loop is already running'。GUI 代码应使用
        await run_python_script() 或 await run_accelerate()。
        """
        return asyncio.run(self.run_python_script(script_module, args, **kwargs))

    def terminate(self):
        """终止当前进程（包括子进程树）"""
        if self.process and self._running:
            self._notify_log("正在终止进程...")
            try:
                if sys.platform == "win32":
                    # 杀掉整个进程树
                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    self.process.terminate()
            except (ProcessLookupError, OSError):
                pass
            self._running = False
            self._notify_status(ProcessStatus.IDLE)

    @property
    def is_running(self) -> bool:
        """检查是否有任务在运行"""
        return self._running


# 全局进程运行器（保留向后兼容）
process_runner = ProcessRunner()

# 向后兼容别名：允许 `from utils.process_runner import log_buffer`
log_buffer = _global_log_buffer
