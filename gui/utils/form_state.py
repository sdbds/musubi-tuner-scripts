"""Helpers for collecting and applying page form state."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class FormStateMixin:
    """Collects UI control values without duplicating page-specific preset logic."""

    _STATE_SKIP_KEYS = {
        "config",
        "model_selector",
        "exec_panel",
        "arch_info",
        "_model_path_container",
        "_model_specific_container",
        "_arch_specific_container",
        "_dynamic_control_scopes",
    }

    def _init_form_state(self) -> None:
        self._dynamic_control_scopes: Dict[str, set[str]] = {}

    def _set_control(self, name: str, control: Any, scope: str | None = None) -> Any:
        setattr(self, name, control)
        if scope:
            self._dynamic_control_scopes.setdefault(scope, set()).add(name)
        return control

    def _clear_control_scope(self, scope: str) -> None:
        for name in self._dynamic_control_scopes.pop(scope, set()):
            if hasattr(self, name):
                delattr(self, name)

    def _iter_state_controls(self) -> Iterable[tuple[str, Any]]:
        for key, value in self.__dict__.items():
            if key.startswith("_") or key in self._STATE_SKIP_KEYS:
                continue
            if key in {"render"}:
                continue
            yield key, value

    def _read_control_value(self, control: Any) -> Any:
        if hasattr(control, "value"):
            return control.value
        return None

    def _write_control_value(self, control: Any, value: Any) -> None:
        if hasattr(control, "set_toggle_value"):
            control.set_toggle_value(value)
            return
        if hasattr(control, "set_bound_value"):
            control.set_bound_value(value)
            return
        if hasattr(control, "set_value"):
            control.set_value(value)
            return
        if hasattr(control, "value"):
            control.value = value
            if hasattr(control, "update"):
                control.update()

    def _write_bound_control_values(self, config: Dict[str, Any]) -> None:
        bound_controls = self.config.get("_bound_controls", {})
        if not isinstance(bound_controls, dict):
            return
        for key, control in bound_controls.items():
            if key in config:
                self._write_control_value(control, config[key])

    def _collect_form_state(self) -> Dict[str, Any]:
        state = {key: value for key, value in self.config.items() if not key.startswith("_")}

        if getattr(self, "model_selector", None):
            state["arch"] = self.model_selector.arch
            state["version"] = self.model_selector.version
            if self.model_selector.task is not None:
                state["task"] = self.model_selector.task

        for key, control in self._iter_state_controls():
            value = self._read_control_value(control)
            if value is None:
                continue
            state[key] = value

        return state

    def _apply_form_state(self, config: Dict[str, Any]) -> None:
        if getattr(self, "model_selector", None):
            arch = config.get("arch")
            if arch:
                self.model_selector.set_arch(arch)
            version = config.get("version")
            if version:
                self.model_selector.set_version(version)
            task = config.get("task")
            if task:
                self.model_selector.set_task(task)

        config_keys = set(self.config.keys())
        self.config.update(
            {
                key: value
                for key, value in config.items()
                if not key.startswith("_") and key in config_keys
            }
        )
        self._write_bound_control_values(config)

        for key, value in config.items():
            control = getattr(self, key, None)
            if control is None:
                continue
            self._write_control_value(control, value)
