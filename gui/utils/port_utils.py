"""Helpers for choosing a GUI port without colliding with an already running service."""

from __future__ import annotations

import os
import socket

DEFAULT_GUI_HOST = "127.0.0.1"
DEFAULT_GUI_PORT = 7788
DEFAULT_PORT_SEARCH_SPAN = 50


def parse_port(value: str | None, default: int = DEFAULT_GUI_PORT) -> int:
    """Parse a port number from environment text, falling back on invalid input."""
    if value is None:
        return default
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    if 1 <= port <= 65535:
        return port
    return default


def parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse common environment-style booleans."""
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def port_is_available(host: str, port: int) -> bool:
    """Return whether a TCP port can be bound on the requested host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_available_port(host: str, preferred_port: int, search_span: int = DEFAULT_PORT_SEARCH_SPAN) -> int:
    """Find the first available port, starting from the preferred one."""
    max_port = min(65535, preferred_port + max(0, search_span))
    for port in range(preferred_port, max_port + 1):
        if port_is_available(host, port):
            return port
    raise RuntimeError(f"No available port found in range {preferred_port}-{max_port}")


def resolve_gui_host() -> str:
    """Return the bind host for the NiceGUI server."""
    return os.getenv("MUSUBI_GUI_HOST", DEFAULT_GUI_HOST).strip() or DEFAULT_GUI_HOST


def resolve_gui_port() -> tuple[int, int]:
    """Resolve the preferred and selected GUI ports."""
    preferred_port = parse_port(os.getenv("MUSUBI_GUI_PORT"), DEFAULT_GUI_PORT)
    selected_port = find_available_port(resolve_gui_host(), preferred_port)
    return preferred_port, selected_port


def resolve_gui_native() -> bool:
    """Return whether NiceGUI should launch in pywebview/native window mode."""
    return parse_bool(os.getenv("MUSUBI_GUI_NATIVE"), False)


def resolve_gui_show() -> bool:
    """Return whether NiceGUI should open a browser/native window automatically."""
    return not parse_bool(os.getenv("MUSUBI_GUI_NO_BROWSER"), False)
