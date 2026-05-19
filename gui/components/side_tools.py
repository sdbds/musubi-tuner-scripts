"""Global draggable side tools for resource and training log panels."""

from __future__ import annotations

import asyncio
import html
import json
from datetime import datetime
from typing import Any

from nicegui import ui
from theme import get_classes
from utils.env_config import load_env_config, save_env_config
from utils.i18n import t
from utils.system_monitor import collect_system_metrics, format_bytes, format_mib
from utils.training_log_service import WANDB_HOME_URL, resolve_training_log_context, tensorboard_service

_FLOATING_TOOLS_ASSETS = """
<style>
.ql-floating-side-tool {
    position: fixed !important;
    z-index: 3000;
    top: 50%;
    width: 46px;
    height: 46px;
    min-width: 46px;
    min-height: 46px;
    border-radius: 999px !important;
    border: 1px solid var(--ql-accent-border) !important;
    background: var(--ql-surface) !important;
    color: var(--ql-accent) !important;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.24) !important;
    cursor: grab;
    touch-action: none;
}

.ql-floating-side-tool:active {
    cursor: grabbing;
}

.ql-floating-side-tool .q-icon {
    color: var(--ql-accent) !important;
    font-size: 23px;
    line-height: 1;
    width: 1em;
    height: 1em;
    aspect-ratio: 1 / 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    transform: none !important;
}

.ql-floating-side-tool .q-btn__content {
    width: 100%;
    height: 100%;
    min-width: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

.ql-floating-side-tool:hover {
    background: var(--ql-accent-muted) !important;
    border-color: var(--ql-accent) !important;
}

.ql-floating-tool-left {
    left: 14px;
}

.ql-floating-tool-right {
    right: 14px;
}

.ql-side-panel {
    position: fixed;
    top: 58px;
    bottom: 12px;
    z-index: 2990;
    display: flex;
    flex-direction: column;
    background: var(--ql-surface);
    color: var(--ql-text);
    border: 1px solid var(--ql-border);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.34);
    transition: transform 0.22s ease, opacity 0.18s ease;
    opacity: 1;
    overflow: hidden;
}

.ql-side-panel-left {
    left: 12px;
    width: clamp(390px, 46vw, 760px);
    border-radius: 0 12px 12px 0;
}

.ql-side-panel-right {
    right: 12px;
    width: clamp(520px, 72vw, 1240px);
    border-radius: 12px 0 0 12px;
}

.ql-side-panel-left.ql-side-panel-closed {
    transform: translateX(calc(-100% - 18px));
    opacity: 0;
    pointer-events: none;
}

.ql-side-panel-right.ql-side-panel-closed {
    transform: translateX(calc(100% + 18px));
    opacity: 0;
    pointer-events: none;
}

.ql-side-panel-header {
    flex: 0 0 auto;
    padding: 14px 16px;
    border-bottom: 1px solid var(--ql-border);
    background: var(--ql-surface-raised);
}

.ql-side-panel-body {
    flex: 1 1 auto;
    overflow: auto;
    padding: 14px;
}

.ql-side-panel-close {
    border: 1px solid var(--ql-border) !important;
    background: transparent !important;
}

.ql-monitor-metric-label {
    color: var(--ql-text-secondary);
    font-size: 12px;
}

.ql-monitor-metric-value {
    color: var(--ql-text);
    font-size: 21px;
    font-weight: 700;
}

.ql-monitor-graph-card {
    min-height: 230px;
}

.ql-monitor-graph {
    height: 190px;
    width: 100%;
}

.ql-monitor-meter {
    width: 92px;
    min-width: 92px;
    height: 92px;
}

.ql-monitor-gpu-meter {
    width: 96px;
    min-width: 96px;
    height: 96px;
}

.ql-history-chart {
    width: 100%;
    height: 100%;
    min-height: 168px;
}

.ql-history-header {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    color: var(--ql-text-muted);
    font-size: 11px;
    line-height: 1.2;
}

.ql-history-legend {
    display: inline-flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 8px;
    min-width: 0;
}

.ql-history-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    white-space: nowrap;
}

.ql-history-legend-swatch {
    width: 8px;
    height: 8px;
    border-radius: 2px;
    display: inline-block;
}

.ql-history-body {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 8px;
    height: 144px;
    margin-top: 8px;
    align-items: stretch;
}

.ql-history-y-axis {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    align-items: flex-end;
    color: var(--ql-text-muted);
    font-size: 10px;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}

.ql-history-plot {
    min-width: 0;
    position: relative;
    display: flex;
    align-items: stretch;
    gap: 1px;
    padding: 6px 4px 4px;
    border: 1px solid rgba(146, 151, 168, 0.20);
    background:
        linear-gradient(to bottom, rgba(146,151,168,0.16) 1px, transparent 1px) 0 0 / 100% 25%,
        rgba(255,255,255,0.025);
}

.ql-history-sample {
    flex: 1 1 0;
    min-width: 1px;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 1px;
    height: 100%;
}

.ql-history-bar {
    flex: 1 1 0;
    min-width: 1px;
    max-width: 8px;
    border-radius: 2px 2px 0 0;
    opacity: 0.88;
}

.ql-history-x-axis {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 8px;
    margin-top: 4px;
    color: var(--ql-text-muted);
    font-size: 10px;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}

.ql-history-x-labels {
    min-width: 0;
    display: flex;
    justify-content: space-between;
}

.ql-tensorboard-frame {
    display: block;
    width: 100% !important;
    max-width: none !important;
    height: calc(100vh - 220px);
    min-height: 520px;
    border: 1px solid var(--ql-border);
    border-radius: 8px;
    background: var(--ql-surface);
}

.ql-tensorboard-host,
.ql-tensorboard-host > div,
.ql-tensorboard-host iframe {
    width: 100% !important;
    max-width: none !important;
    box-sizing: border-box;
}

@media (max-width: 900px) {
    .ql-side-panel {
        top: 72px;
        left: 8px;
        right: 8px;
        bottom: 8px;
        width: auto;
        border-radius: 12px;
    }

    .ql-side-panel-left.ql-side-panel-closed {
        transform: translateX(calc(-100% - 16px));
    }

    .ql-side-panel-right.ql-side-panel-closed {
        transform: translateX(calc(100% + 16px));
    }

    .ql-tensorboard-frame {
        min-height: 420px;
        height: calc(100vh - 240px);
    }
}
</style>
<script>
(function() {
    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }

    function storageKey(button) {
        return button.classList.contains('ql-floating-tool-left')
            ? 'musubi_floating_resource_pos'
            : 'musubi_floating_logs_pos';
    }

    function applyPosition(button, pos) {
        var rect = button.getBoundingClientRect();
        var x = clamp(pos.x, 8, Math.max(8, window.innerWidth - rect.width - 8));
        var y = clamp(pos.y, 72, Math.max(72, window.innerHeight - rect.height - 8));
        button.style.left = x + 'px';
        button.style.top = y + 'px';
        button.style.right = 'auto';
        button.style.transform = 'none';
    }

    function readSavedPosition(button) {
        try {
            var raw = localStorage.getItem(storageKey(button));
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            return null;
        }
    }

    function savePosition(button) {
        var rect = button.getBoundingClientRect();
        localStorage.setItem(storageKey(button), JSON.stringify({ x: rect.left, y: rect.top }));
    }

    function installButton(button) {
        if (!button || button.dataset.qlFloatingInstalled === '1') return;
        button.dataset.qlFloatingInstalled = '1';

        requestAnimationFrame(function() {
            var saved = readSavedPosition(button);
            if (saved) {
                applyPosition(button, saved);
                return;
            }
            var rect = button.getBoundingClientRect();
            applyPosition(button, { x: rect.left, y: rect.top });
        });

        var state = null;
        button.addEventListener('pointerdown', function(event) {
            if (event.button !== undefined && event.button !== 0) return;
            var rect = button.getBoundingClientRect();
            state = {
                pointerId: event.pointerId,
                startClientX: event.clientX,
                startClientY: event.clientY,
                startX: rect.left,
                startY: rect.top,
                moved: false
            };
            button.setPointerCapture(event.pointerId);
        });

        button.addEventListener('pointermove', function(event) {
            if (!state || state.pointerId !== event.pointerId) return;
            var dx = event.clientX - state.startClientX;
            var dy = event.clientY - state.startClientY;
            if (Math.abs(dx) + Math.abs(dy) > 5) state.moved = true;
            if (!state.moved) return;
            event.preventDefault();
            applyPosition(button, { x: state.startX + dx, y: state.startY + dy });
        });

        button.addEventListener('pointerup', function(event) {
            if (!state || state.pointerId !== event.pointerId) return;
            button.dataset.qlWasDragged = state.moved ? '1' : '0';
            if (state.moved) savePosition(button);
            try { button.releasePointerCapture(event.pointerId); } catch (e) {}
            state = null;
        });

        button.addEventListener('click', function(event) {
            if (button.dataset.qlWasDragged === '1') {
                event.preventDefault();
                event.stopImmediatePropagation();
                button.dataset.qlWasDragged = '0';
            }
        }, true);

        window.addEventListener('resize', function() {
            var rect = button.getBoundingClientRect();
            applyPosition(button, { x: rect.left, y: rect.top });
            savePosition(button);
        });
    }

    window.installMusubiFloatingTools = function() {
        document.querySelectorAll('.ql-floating-side-tool').forEach(installButton);
    };

    setTimeout(window.installMusubiFloatingTools, 0);
    setTimeout(window.installMusubiFloatingTools, 250);
})();
</script>
"""

_MAX_HISTORY = 60
_CHART_COLORS = ["#7ee2a5", "#d7b455", "#bf6c83", "#74a8ff", "#b58cff", "#54d6d6"]


def _format_percent(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "-"


def _progress_value(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value) / 100.0))
    except (TypeError, ValueError):
        return 0.0


def _number_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bounded_history(history: list[Any], value: Any) -> None:
    history.append(value)
    if len(history) > _MAX_HISTORY:
        del history[: len(history) - _MAX_HISTORY]


def _split_cuda_visible_devices(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _open_url(url: str) -> None:
    ui.run_javascript(f"window.open({json.dumps(url)}, '_blank')")


def _usage_color(percent: float) -> str:
    if percent >= 90:
        return "#bf6c83"
    if percent >= 70:
        return "#d7b455"
    return "#7ee2a5"


def _render_ring_svg(value: Any, *, label: str = "") -> str:
    percent = max(0.0, min(100.0, _number_or_zero(value)))
    color = _usage_color(percent)
    display_value = "-" if value is None else f"{percent:.2f}%"
    return f"""
<svg viewBox="0 0 96 96" preserveAspectRatio="xMidYMid meet" style="width:100%;height:100%;display:block;">
  <circle cx="48" cy="48" r="34" fill="none" stroke="rgba(146,151,168,0.20)" stroke-width="10" />
  <circle cx="48" cy="48" r="34" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"
          pathLength="100" stroke-dasharray="{percent:.2f} 100" transform="rotate(-90 48 48)" />
  <circle cx="48" cy="48" r="24" fill="rgba(255,255,255,0.025)" />
  <text x="48" y="46" text-anchor="middle" fill="var(--ql-text)" font-size="14" font-weight="700">{display_value}</text>
  <text x="48" y="62" text-anchor="middle" fill="var(--ql-text-muted)" font-size="10">{html.escape(label)}</text>
</svg>
"""


def _render_chart_svg(title: str, labels: list[str], series: list[tuple[str, list[float]]]) -> str:
    max_series_length = max((len(values) for _name, values in series), default=0)
    sample_count = max(1, len(labels), max_series_length)
    visible_labels = list(labels[-sample_count:]) if labels else [""] * sample_count
    if len(visible_labels) < sample_count:
        visible_labels = [""] * (sample_count - len(visible_labels)) + visible_labels

    normalized_series = []
    legend = []
    for index, (name, values) in enumerate(series):
        color = _CHART_COLORS[index % len(_CHART_COLORS)]
        visible_values = list(values[-sample_count:])
        if len(visible_values) < sample_count:
            visible_values = [0.0] * (sample_count - len(visible_values)) + visible_values
        normalized_series.append((name, visible_values, color))
        latest_value = max(0.0, min(100.0, float(visible_values[-1]))) if visible_values else 0.0
        escaped_name = html.escape(name)
        legend.append(
            '<span class="ql-history-legend-item">'
            f'<span class="ql-history-legend-swatch" style="background:{color};"></span>'
            f"{escaped_name} {latest_value:.2f}%</span>"
        )

    samples = []
    for point_index in range(sample_count):
        sample_bars = []
        for name, visible_values, color in normalized_series:
            bounded = max(0.0, min(100.0, float(visible_values[point_index])))
            title_text = html.escape(f"{name}: {bounded:.2f}%")
            sample_bars.append(
                f'<span class="ql-history-bar" title="{title_text}" '
                f'style="height:{bounded:.2f}%; background:{color};"></span>'
            )
        samples.append(f'<span class="ql-history-sample">{"".join(sample_bars)}</span>')

    first_label = html.escape(visible_labels[0]) if visible_labels else ""
    last_label = html.escape(visible_labels[-1]) if visible_labels else ""
    legend_html = "".join(legend)
    return f"""
<div class="ql-history-chart">
  <div class="ql-history-header">
    <span>{html.escape(title)}</span><span class="ql-history-legend">{legend_html}</span>
  </div>
  <div class="ql-history-body">
    <div class="ql-history-y-axis"><span>100</span><span>75</span><span>50</span><span>25</span><span>0</span></div>
    <div class="ql-history-plot">{"".join(samples)}</div>
  </div>
  <div class="ql-history-x-axis">
    <span></span>
    <div class="ql-history-x-labels"><span>{first_label}</span><span>{last_label}</span></div>
  </div>
</div>
"""


class SidePanel:
    """A fixed side panel without a modal backdrop."""

    def __init__(self, side: str, title: str, icon: str, *, extra_header=None, on_close=None) -> None:
        self.side = side
        self._on_close = on_close
        self.is_open = False
        with ui.element("div").classes(f"ql-side-panel ql-side-panel-{side} ql-side-panel-closed") as self.panel:
            with ui.row().classes("ql-side-panel-header w-full items-center justify-between gap-3"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(icon, size="24px")
                    ui.label(title).classes("text-h6 text-weight-bold")
                with ui.row().classes("items-center gap-2"):
                    if extra_header:
                        extra_header()
                    close_btn = ui.button(icon="close", on_click=self._handle_close).props("flat round dense")
                    close_btn.classes("ql-side-panel-close")
            with ui.element("div").classes("ql-side-panel-body") as self.body:
                pass

    def open(self) -> None:
        self.is_open = True
        self.panel.classes(add="ql-side-panel-open", remove="ql-side-panel-closed")

    def close(self) -> None:
        self.is_open = False
        self.panel.classes(add="ql-side-panel-closed", remove="ql-side-panel-open")

    def _handle_close(self) -> None:
        if self._on_close:
            self._on_close()
        else:
            self.close()


class ResourceMonitorPanel:
    def __init__(self) -> None:
        self._refreshing = False
        self._built = False
        self._labels: list[str] = []
        self._cpu_history: list[float] = []
        self._memory_history: list[float] = []
        self._gpu_util_history: dict[str, list[float]] = {}
        self._gpu_memory_history: dict[str, list[float]] = {}
        self._known_gpu_ids: tuple[str, ...] = ()
        self._gpu_select_controls: dict[str, Any] = {}
        self.cuda_status_label = None

        self.side_panel = SidePanel("left", t("resource_monitor", "系统资源监控"), "monitor_heart", on_close=self.close)
        with self.side_panel.body:
            self.content = ui.column().classes("w-full gap-3")
            with self.content:
                ui.label(t("click_to_load_monitor", "点击后加载系统资源曲线...")).classes("ql-monitor-metric-label")

        self.timer = ui.timer(1.5, self._refresh, active=False)

    async def toggle(self) -> None:
        if self.side_panel.is_open:
            self.close()
            return
        await self.open()

    async def open(self) -> None:
        self._build_content()
        self.side_panel.open()
        self.timer.active = True
        await self._refresh()

    def close(self) -> None:
        self.timer.active = False
        self.side_panel.close()

    def _build_content(self) -> None:
        if self._built:
            return
        self._built = True
        self.content.clear()
        with self.content:
            with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 gap-3 w-full"):
                self.cpu_card = _metric_card("developer_board", "CPU")
                self.memory_card = _metric_card("memory", t("system_memory", "系统内存"))

            with ui.card().classes(get_classes("card") + " w-full q-pa-md ql-monitor-graph-card"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("CPU").classes("text-subtitle1 text-weight-bold")
                    self.cpu_graph_value = ui.label("-").classes("ql-monitor-metric-label")
                self.cpu_chart = ui.html(_render_chart_svg("CPU", [], [("CPU", [])]), sanitize=False).classes("ql-monitor-graph")

            with ui.card().classes(get_classes("card") + " w-full q-pa-md ql-monitor-graph-card"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(t("system_memory", "系统内存")).classes("text-subtitle1 text-weight-bold")
                    self.memory_graph_value = ui.label("-").classes("ql-monitor-metric-label")
                self.memory_chart = ui.html(
                    _render_chart_svg("Memory", [], [("Memory", [])]),
                    sanitize=False,
                ).classes("ql-monitor-graph")

            self.gpu_summary_container = ui.column().classes("w-full gap-3")
            self.gpu_selector_container = ui.column().classes("w-full gap-3")

            with ui.card().classes(get_classes("card") + " w-full q-pa-md ql-monitor-graph-card"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("GPU Utilization").classes("text-subtitle1 text-weight-bold")
                    self.gpu_graph_value = ui.label("-").classes("ql-monitor-metric-label")
                self.gpu_util_chart = ui.html(_render_chart_svg("GPU", [], [("GPU", [])]), sanitize=False).classes(
                    "ql-monitor-graph"
                )

            with ui.card().classes(get_classes("card") + " w-full q-pa-md ql-monitor-graph-card"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("GPU VRAM").classes("text-subtitle1 text-weight-bold")
                    self.gpu_memory_graph_value = ui.label("-").classes("ql-monitor-metric-label")
                self.gpu_memory_chart = ui.html(_render_chart_svg("VRAM", [], [("VRAM", [])]), sanitize=False).classes(
                    "ql-monitor-graph"
                )

    async def _refresh(self) -> None:
        if self._refreshing:
            return
        self._refreshing = True
        try:
            metrics = await asyncio.to_thread(collect_system_metrics)
            timestamp = datetime.now().strftime("%H:%M:%S")
            _bounded_history(self._labels, timestamp)
            self._render_cpu(metrics.get("cpu", {}))
            self._render_memory(metrics.get("memory", {}))
            self._render_gpus(metrics.get("gpus", []), metrics.get("gpu_error"))
        except RuntimeError:
            return
        finally:
            self._refreshing = False

    def _render_cpu(self, cpu: dict[str, Any]) -> None:
        percent = cpu.get("percent") if cpu.get("available") else None
        _bounded_history(self._cpu_history, _number_or_zero(percent))
        self.cpu_card["value"].set_text(_format_percent(percent) if cpu.get("available") else "-")
        self.cpu_card["detail"].set_text(
            f"{cpu.get('physical_count') or '-'}C / {cpu.get('logical_count') or '-'}T"
            if cpu.get("available")
            else str(cpu.get("message") or "unavailable")
        )
        self.cpu_card["meter"].set_content(_render_ring_svg(percent, label="CPU"))
        self.cpu_graph_value.set_text(_format_percent(percent))
        _update_chart(self.cpu_chart, "CPU", self._labels, [("CPU", self._cpu_history)])

    def _render_memory(self, memory: dict[str, Any]) -> None:
        percent = memory.get("percent") if memory.get("available") else None
        _bounded_history(self._memory_history, _number_or_zero(percent))
        self.memory_card["value"].set_text(_format_percent(percent) if memory.get("available") else "-")
        self.memory_card["detail"].set_text(
            f"{format_bytes(memory.get('used_bytes'))} / {format_bytes(memory.get('total_bytes'))}"
            if memory.get("available")
            else str(memory.get("message") or "unavailable")
        )
        self.memory_card["meter"].set_content(_render_ring_svg(percent, label="RAM"))
        self.memory_graph_value.set_text(_format_percent(percent))
        _update_chart(self.memory_chart, "Memory", self._labels, [("Memory", self._memory_history)])

    def _render_gpus(self, gpus: list[dict[str, Any]], gpu_error: str | None) -> None:
        self._render_gpu_selector(gpus)
        self.gpu_summary_container.clear()
        with self.gpu_summary_container:
            if not gpus:
                with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("videocard_off", size="22px")
                        ui.label(t("gpu_unavailable", "GPU 信息不可用")).classes("text-subtitle1 text-weight-bold")
                    ui.label(gpu_error or "No NVIDIA GPU detected").classes("text-caption").style(
                        "color: var(--ql-text-secondary);"
                    )
                self.gpu_graph_value.set_text(gpu_error or "No GPU")
                self.gpu_memory_graph_value.set_text(gpu_error or "No GPU")
                _update_chart(self.gpu_util_chart, "GPU", self._labels, [("GPU", [0 for _ in self._labels])])
                _update_chart(self.gpu_memory_chart, "VRAM", self._labels, [("VRAM", [0 for _ in self._labels])])
                return

            util_series = []
            memory_series = []
            active_util = []
            active_memory = []
            for gpu in gpus:
                gpu_key = f"GPU {gpu.get('index')}"
                util = gpu.get("utilization_percent")
                used = gpu.get("memory_used_mib")
                total = gpu.get("memory_total_mib")
                mem_percent = (float(used) / float(total) * 100.0) if used is not None and total else None
                _bounded_history(self._gpu_util_history.setdefault(gpu_key, []), _number_or_zero(util))
                _bounded_history(self._gpu_memory_history.setdefault(gpu_key, []), _number_or_zero(mem_percent))
                active_util.append(_number_or_zero(util))
                active_memory.append(_number_or_zero(mem_percent))
                util_series.append((gpu_key, self._gpu_util_history[gpu_key]))
                memory_series.append((gpu_key, self._gpu_memory_history[gpu_key]))

                with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                    with ui.row().classes("w-full items-center justify-between gap-3"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("memory", size="22px")
                            ui.label(f"{gpu_key}: {gpu.get('name')}").classes("text-subtitle1 text-weight-bold")
                        ui.label(_format_percent(util)).classes("ql-monitor-metric-value")
                    with ui.row().classes("w-full items-center gap-4 q-mt-sm"):
                        with ui.column().classes("items-center gap-1"):
                            ui.html(_render_ring_svg(util, label="GPU"), sanitize=False).classes("ql-monitor-gpu-meter")
                            ui.label(t("gpu_utilization", "GPU 利用率")).classes("ql-monitor-metric-label")
                        with ui.column().classes("items-center gap-1"):
                            ui.html(_render_ring_svg(mem_percent, label="VRAM"), sanitize=False).classes("ql-monitor-gpu-meter")
                            ui.label("VRAM").classes("ql-monitor-metric-label")
                        with ui.column().classes("gap-1"):
                            ui.label(f"VRAM {format_mib(used)} / {format_mib(total)}").classes("ql-monitor-metric-label")
                            ui.label(f"VRAM {_format_percent(mem_percent)}").classes("ql-monitor-metric-label")
                    details = [
                        f"{gpu.get('temperature_c'):.0f} C" if gpu.get("temperature_c") is not None else None,
                        (
                            f"{gpu.get('power_draw_w'):.0f} W / {gpu.get('power_limit_w'):.0f} W"
                            if gpu.get("power_draw_w") is not None and gpu.get("power_limit_w") is not None
                            else None
                        ),
                    ]
                    ui.label(" | ".join(item for item in details if item) or "-").classes("ql-monitor-metric-label")

        self.gpu_graph_value.set_text(f"avg {_format_percent(sum(active_util) / len(active_util))}")
        self.gpu_memory_graph_value.set_text(f"avg {_format_percent(sum(active_memory) / len(active_memory))}")
        _update_chart(self.gpu_util_chart, "GPU", self._labels, util_series)
        _update_chart(self.gpu_memory_chart, "VRAM", self._labels, memory_series)

    def _render_gpu_selector(self, gpus: list[dict[str, Any]]) -> None:
        gpu_ids = tuple(str(gpu.get("index")) for gpu in gpus if gpu.get("index") is not None)
        if gpu_ids == self._known_gpu_ids:
            self._sync_cuda_status()
            return

        self._known_gpu_ids = gpu_ids
        self._gpu_select_controls = {}
        self.gpu_selector_container.clear()
        with self.gpu_selector_container:
            with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
                with ui.row().classes("w-full items-center justify-between gap-3"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("tune", size="22px")
                        ui.label(t("gpu_training_device", "训练使用 GPU")).classes("text-subtitle1 text-weight-bold")
                    ui.button(
                        t("clear", "清空"),
                        icon="clear",
                        on_click=self._clear_cuda_visible_devices,
                    ).classes("modern-btn-ghost")

                current_ids = set(_split_cuda_visible_devices(load_env_config().get("CUDA_VISIBLE_DEVICES", "")))
                if not gpu_ids:
                    ui.label(t("gpu_selection_unavailable", "未检测到可选择的 NVIDIA GPU")).classes("ql-monitor-metric-label")
                else:
                    with ui.row().classes("w-full gap-3 q-mt-sm flex-wrap"):
                        for gpu in gpus:
                            gpu_id = str(gpu.get("index"))
                            label = f"GPU {gpu_id}: {gpu.get('name')}"
                            checkbox = ui.checkbox(label, value=(gpu_id in current_ids if current_ids else False))
                            checkbox.classes("ql-monitor-metric-label")
                            self._gpu_select_controls[gpu_id] = checkbox
                    ui.button(
                        t("apply_gpu_selection", "应用到训练环境"),
                        icon="done",
                        on_click=self._apply_cuda_visible_devices,
                    ).classes("modern-btn-secondary q-mt-sm")
                self.cuda_status_label = ui.label("").classes("ql-monitor-metric-label q-mt-sm")
        self._sync_cuda_status()

    def _selected_gpu_ids(self) -> list[str]:
        return [gpu_id for gpu_id, control in self._gpu_select_controls.items() if bool(control.value)]

    def _apply_cuda_visible_devices(self) -> None:
        selected_ids = self._selected_gpu_ids()
        config = load_env_config()
        config["CUDA_VISIBLE_DEVICES"] = ",".join(selected_ids)
        config["CUDA_DEVICE_ORDER"] = config.get("CUDA_DEVICE_ORDER") or "PCI_BUS_ID"
        save_env_config(config)
        self._sync_cuda_status()
        saved_value = config["CUDA_VISIBLE_DEVICES"] or "<all>"
        ui.notify(t("gpu_selection_saved", "已保存训练 GPU 环境变量") + f": CUDA_VISIBLE_DEVICES={saved_value}", type="positive")

    def _clear_cuda_visible_devices(self) -> None:
        config = load_env_config()
        config["CUDA_VISIBLE_DEVICES"] = ""
        save_env_config(config)
        for control in self._gpu_select_controls.values():
            control.set_value(False)
        self._sync_cuda_status()
        ui.notify(t("gpu_selection_cleared", "已清空训练 GPU 限制，将使用所有可见 GPU"), type="info")

    def _sync_cuda_status(self) -> None:
        if self.cuda_status_label is None:
            return
        value = load_env_config().get("CUDA_VISIBLE_DEVICES", "")
        self.cuda_status_label.set_text(f"CUDA_VISIBLE_DEVICES={value or '<all>'}")


class TrainingLogsPanel:
    def __init__(self) -> None:
        def header_actions() -> None:
            refresh_btn = ui.button(icon="refresh", on_click=self.refresh).props("flat round dense")
            refresh_btn.classes("ql-side-panel-close")

        self.side_panel = SidePanel("right", t("training_logs", "训练日志"), "query_stats", extra_header=header_actions)
        with self.side_panel.body:
            self.content = ui.column().classes("w-full gap-3")

    async def toggle(self) -> None:
        if self.side_panel.is_open:
            self.side_panel.close()
            return
        await self.open()

    async def open(self) -> None:
        self.side_panel.open()
        await self.refresh()

    async def refresh(self) -> None:
        context = resolve_training_log_context()
        self.content.clear()
        with self.content:
            if context.mode == "wandb":
                self._render_wandb(context.wandb_url or WANDB_HOME_URL)
            else:
                self._render_tensorboard(context.log_dir)

    def _render_wandb(self, url: str) -> None:
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("open_in_new", size="22px")
                    ui.label("W&B").classes("text-subtitle1 text-weight-bold")
                ui.button(t("open", "打开"), icon="open_in_new", on_click=lambda: _open_url(url)).classes("modern-btn-secondary")
            ui.label(url).classes("text-caption").style("color: var(--ql-text-secondary); word-break: break-all;")
        _open_url(url)

    def _render_tensorboard(self, log_dir) -> None:
        launch = tensorboard_service.ensure_started(log_dir)
        with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
            with ui.row().classes("w-full items-center justify-between gap-3"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("analytics", size="22px")
                    ui.label("TensorBoard").classes("text-subtitle1 text-weight-bold")
                if launch.url:
                    ui.button(t("open", "打开"), icon="open_in_new", on_click=lambda: _open_url(launch.url)).classes(
                        "modern-btn-secondary"
                    )
            ui.label(str(launch.log_dir or log_dir)).classes("text-caption").style(
                "color: var(--ql-text-secondary); word-break: break-all;"
            )

        if not launch.available or not launch.url:
            ui.label(launch.message or "TensorBoard unavailable").classes("text-body2").style("color: var(--ql-error);")
            return

        iframe_url = html.escape(launch.url, quote=True)
        ui.html(f'<iframe src="{iframe_url}" class="ql-tensorboard-frame"></iframe>', sanitize=False).classes(
            "w-full ql-tensorboard-host"
        )


def _metric_card(icon: str, title: str) -> dict[str, Any]:
    with ui.card().classes(get_classes("card") + " w-full q-pa-md") as card:
        with ui.row().classes("w-full items-center justify-between gap-3"):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon(icon, size="22px")
                    ui.label(title).classes("text-subtitle1 text-weight-bold")
                value = ui.label("-").classes("ql-monitor-metric-value")
                detail = ui.label("-").classes("ql-monitor-metric-label")
            meter = ui.html(_render_ring_svg(0, label=title), sanitize=False).classes("ql-monitor-meter")
    return {"card": card, "value": value, "meter": meter, "detail": detail}


def _update_chart(chart, title: str, labels: list[str], series: list[tuple[str, list[float]]]) -> None:
    chart.set_content(_render_chart_svg(title, labels, series))


def create_side_tools() -> None:
    """Create the resource/log floating buttons for the current page."""
    ui.add_body_html(_FLOATING_TOOLS_ASSETS)

    resource_panel = ResourceMonitorPanel()
    logs_panel = TrainingLogsPanel()

    resource_btn = ui.button(icon="monitor_heart", on_click=resource_panel.toggle)
    resource_btn.props("round unelevated")
    resource_btn.classes("ql-floating-side-tool ql-floating-tool-left")
    resource_btn.tooltip(t("resource_monitor", "系统资源监控"))

    logs_btn = ui.button(icon="query_stats", on_click=logs_panel.toggle)
    logs_btn.props("round unelevated")
    logs_btn.classes("ql-floating-side-tool ql-floating-tool-right")
    logs_btn.tooltip(t("training_logs", "训练日志"))

    ui.run_javascript("window.installMusubiFloatingTools && window.installMusubiFloatingTools();")
