"""配置管理工具 - 处理 TOML 预设与项目配置的读写"""
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Optional

import toml

from utils.project_config import (
    create_default_project_config,
    load_project_config as load_project_toml,
    resolve_project_config_path,
    save_project_config as save_project_toml,
)

KNOWN_PRESET_SCOPES = ("cache", "train", "generate")


def _contains_control_char(value: str) -> bool:
    return any(ord(char) < 32 for char in value)


class ConfigManager:
    """管理 GUI 配置的保存和加载"""
    
    def __init__(self, builtin_dir: str | None = None, user_dir: str | None = None):
        base_gui_dir = Path(__file__).resolve().parents[1]
        self.builtin_dir = Path(builtin_dir) if builtin_dir else (base_gui_dir / "presets")
        self.user_dir = Path(user_dir) if user_dir else (self.builtin_dir / "user")
        self.builtin_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.user_config_path = self.user_dir / "user_settings.json"
        self.config_dir = self.user_dir

    def _validate_scope(self, scope: str) -> str:
        normalized = str(scope).strip()
        if normalized not in KNOWN_PRESET_SCOPES:
            raise ValueError(f"Invalid preset scope: {scope}")
        return normalized

    def _validate_preset_name(self, name: str) -> str:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("Preset name is empty")
        if normalized in {".", ".."}:
            raise ValueError("Preset name cannot be a path segment")
        if _contains_control_char(normalized):
            raise ValueError("Preset name contains control characters")
        if "/" in normalized or "\\" in normalized or ":" in normalized:
            raise ValueError("Preset name cannot contain path separators or drive markers")

        windows_path = PureWindowsPath(normalized)
        posix_path = PurePosixPath(normalized)
        if windows_path.is_absolute() or windows_path.drive or windows_path.root or posix_path.is_absolute():
            raise ValueError("Preset name cannot be an absolute path")
        return normalized

    def _resolve_under_root(self, root: Path, child_name: str) -> Path:
        root_path = root.resolve()
        child_path = (root / child_name).resolve()
        if child_path != root_path and not child_path.is_relative_to(root_path):
            raise ValueError("Preset path escapes its storage directory")
        return child_path

    def _scope_dir(self, root: Path, scope: str) -> Path:
        return root / self._validate_scope(scope)

    def _user_config_path(self, scope: str, name: str) -> Path:
        scope_dir = self._scope_dir(self.user_dir, scope)
        preset_name = self._validate_preset_name(name)
        return self._resolve_under_root(scope_dir, f"{preset_name}.toml")

    def _builtin_config_path(self, scope: str, name: str) -> Path:
        scope_dir = self._scope_dir(self.builtin_dir, scope)
        preset_name = self._validate_preset_name(name)
        return self._resolve_under_root(scope_dir, f"{preset_name}.toml")

    def _legacy_user_config_path(self, name: str) -> Path:
        preset_name = self._validate_preset_name(name)
        return self._resolve_under_root(self.user_dir, f"{preset_name}.json")

    def _has_builtin_preset_files(self) -> bool:
        for scope in KNOWN_PRESET_SCOPES:
            if any(self._scope_dir(self.builtin_dir, scope).glob("*.toml")):
                return True
        return False

    def _builtin_entry_from_file(self, path: Path) -> Dict[str, str]:
        label = path.stem
        config = self._load_toml_file(path)
        if config is not None:
            configured_label = config.get("_label") or config.get("arch")
            if isinstance(configured_label, str) and configured_label.strip():
                label = configured_label.strip()
        return {"name": path.stem, "label": label, "source": "builtin"}

    def _resolve_scope_and_name(self, scope_or_name: str, name: str | None) -> tuple[str, str]:
        if name is None:
            return "train", scope_or_name
        return scope_or_name, name

    def _load_toml_file(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return toml.load(handle)

    def get_config_source(self, scope_or_name: str, name: str | None = None) -> Optional[str]:
        scope, preset_name = self._resolve_scope_and_name(scope_or_name, name)
        try:
            user_path = self._user_config_path(scope, preset_name)
            legacy_path = self._legacy_user_config_path(preset_name)
            builtin_path = self._builtin_config_path(scope, preset_name)
        except ValueError:
            return None

        if user_path.exists():
            return "user"
        if legacy_path.exists():
            return "user"
        if builtin_path.exists():
            return "builtin"
        return None

    def list_config_entries(self, scope: str = "train") -> list[Dict[str, str]]:
        entries: Dict[str, Dict[str, str]] = {}
        try:
            scope = self._validate_scope(scope)
        except ValueError:
            return []

        builtin_scope_dir = self._scope_dir(self.builtin_dir, scope)
        if builtin_scope_dir.exists():
            for path in sorted(builtin_scope_dir.glob("*.toml")):
                entries[path.stem] = self._builtin_entry_from_file(path)

        user_scope_dir = self._scope_dir(self.user_dir, scope)
        if user_scope_dir.exists():
            for path in sorted(user_scope_dir.glob("*.toml")):
                entries[path.stem] = {"name": path.stem, "label": path.stem, "source": "user"}

        for path in sorted(self.user_dir.glob("*.json")):
            if path.name == self.user_config_path.name or path.stem in entries:
                continue
            entries[path.stem] = {"name": path.stem, "label": path.stem, "source": "user"}

        return sorted(entries.values(), key=lambda entry: entry["name"])
    
    def save_config(self, scope_or_name: str, name_or_config: str | Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        """保存配置到文件"""
        if config is None:
            scope = "train"
            name = str(scope_or_name)
            config = name_or_config if isinstance(name_or_config, dict) else {}
        else:
            scope = str(scope_or_name)
            name = str(name_or_config)

        try:
            scope = self._validate_scope(scope)
            name = self._validate_preset_name(name)
            scope_dir = self._scope_dir(self.user_dir, scope)
            scope_dir.mkdir(parents=True, exist_ok=True)
            config_path = self._user_config_path(scope, name)
            with open(config_path, 'w', encoding='utf-8') as f:
                toml.dump(config, f)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def load_config(self, scope_or_name: str, name: str | None = None) -> Optional[Dict[str, Any]]:
        """从文件加载配置"""
        scope, preset_name = self._resolve_scope_and_name(scope_or_name, name)
        try:
            user_path = self._user_config_path(scope, preset_name)
            legacy_path = self._legacy_user_config_path(preset_name)
            builtin_path = self._builtin_config_path(scope, preset_name)

            user_config = self._load_toml_file(user_path)
            if user_config is not None:
                return user_config

            if legacy_path.exists():
                with open(legacy_path, 'r', encoding='utf-8') as f:
                    return json.load(f)

            builtin_config = self._load_toml_file(builtin_path)
            if builtin_config is not None:
                return builtin_config

            return None
        except Exception as e:
            print(f"加载配置失败: {e}")
            return None
    
    def list_configs(self, scope: str = "train") -> list:
        """列出所有可用配置"""
        return [entry["name"] for entry in self.list_config_entries(scope)]

    def delete_config(self, scope_or_name: str, name: str | None = None) -> bool:
        """删除用户预设，内置预设只读"""
        scope, preset_name = self._resolve_scope_and_name(scope_or_name, name)
        source = self.get_config_source(scope, preset_name)
        if source != "user":
            return False

        try:
            config_path = self._user_config_path(scope, preset_name)
            if config_path.exists():
                config_path.unlink()
            else:
                legacy_path = self._legacy_user_config_path(preset_name)
                if legacy_path.exists():
                    legacy_path.unlink()
            return True
        except Exception as e:
            print(f"删除配置失败: {e}")
            return False
    
    def save_user_settings(self, settings: Dict[str, Any]):
        """保存用户设置"""
        try:
            with open(self.user_config_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存用户设置失败: {e}")
    
    def load_user_settings(self) -> Dict[str, Any]:
        """加载用户设置"""
        try:
            if not self.user_config_path.exists():
                return {}
            with open(self.user_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载用户设置失败: {e}")
            return {}
    
    def get_default_config(self, model_type: str) -> Optional[Dict[str, Any]]:
        """获取预设的默认配置"""
        defaults = {
            "flux2_lora": {
                "model_version": "klein-base-4b",
                "timestep_sampling": "flux2_shift",
                "weighting_scheme": "none",
                "mixed_precision": "bf16",
                "attn_mode": "flash",
                "optimizer_type": "AdamW_adv",
                "lr": "1e-4",
                "network_dim": 32,
                "network_alpha": 16,
                "gradient_checkpointing": True,
                "max_train_epochs": 20,
                "save_every_n_epochs": 2,
            },
            "wan_lora": {
                "version": "A14B",
                "task": "t2v-A14B",
                "timestep_sampling": "shift",
                "discrete_flow_shift": 1.0,
                "mixed_precision": "fp16",
                "attn_mode": "flash",
                "optimizer_type": "fira",
                "lr": "2e-4",
                "network_dim": 32,
                "network_alpha": 16,
                "gradient_checkpointing": True,
                "max_train_epochs": 8,
                "blocks_to_swap": 26,
                "fp8_t5": False,
                "vae_cache_cpu": True,
                "one_frame": False,
            },
            "hy_video_lora": {
                "mixed_precision": "bf16",
                "attn_mode": "flash",
                "optimizer_type": "AdamW_adv",
                "lr": "2e-4",
                "network_dim": 32,
                "network_alpha": 16,
                "gradient_checkpointing": True,
                "max_train_epochs": 20,
            },
        }
        return defaults.get(model_type)

    def get_project_config_path(self, project_dir: str | Path) -> Path:
        """获取项目配置文件路径。"""
        return resolve_project_config_path(project_dir)

    def load_project_config(self, project_dir: str | Path) -> Dict[str, Any]:
        """加载 musubi_project.toml，缺失时返回规范化默认值。"""
        try:
            return load_project_toml(project_dir)
        except Exception as e:
            print(f"加载项目配置失败: {e}")
            return create_default_project_config()

    def save_project_config(self, project_dir: str | Path, config: Dict[str, Any]) -> bool:
        """保存 musubi_project.toml。"""
        try:
            save_project_toml(project_dir, config)
            return True
        except Exception as e:
            print(f"保存项目配置失败: {e}")
            return False


# 全局配置管理器实例
config_manager = ConfigManager()
