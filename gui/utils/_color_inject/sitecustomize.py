"""Make GUI-launched child processes behave like they are attached to a TTY."""

from __future__ import annotations

import os
import sys


_REQUESTED_COLOR_SYSTEM = os.environ.get("_MUSUBI_RICH_COLOR_SYSTEM", "").strip().lower()
_FORCE_TTY = os.environ.get("_MUSUBI_FORCE_TTY", "") == "1"


class _FakeTTY:
    def __init__(self, wrapped):
        self._wrapped = wrapped
        self.encoding = getattr(wrapped, "encoding", "utf-8")
        self.errors = getattr(wrapped, "errors", "replace")
        self.softspace = 0

    def isatty(self) -> bool:
        return True

    def write(self, value) -> int:
        return self._wrapped.write(value)

    def flush(self) -> None:
        self._wrapped.flush()

    def fileno(self) -> int:
        return self._wrapped.fileno()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    @property
    def buffer(self):
        return getattr(self._wrapped, "buffer", self._wrapped)


if _FORCE_TTY:
    try:
        sys.stdout = _FakeTTY(sys.stdout)
        sys.stderr = _FakeTTY(sys.stderr)
    except Exception:
        pass


def _patch_rich() -> None:
    try:
        import rich.console as rich_console

        original_init = rich_console.Console.__init__

        def patched_init(self, *args, **kwargs):
            kwargs.setdefault("legacy_windows", False)
            if _REQUESTED_COLOR_SYSTEM and _REQUESTED_COLOR_SYSTEM != "auto":
                kwargs["color_system"] = _REQUESTED_COLOR_SYSTEM
            original_init(self, *args, **kwargs)

        rich_console.Console.__init__ = patched_init
    except Exception:
        pass


_patch_rich()
