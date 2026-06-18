"""GUI-managed environment variable configuration.

Values are persisted to config/env_vars.json and injected into subprocesses
launched from the GUI.
"""

from __future__ import annotations

import json
import locale
import os
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "env_vars.json"


ENV_VAR_DEFINITIONS: list[dict[str, str]] = [
    {
        "key": "HF_HOME",
        "default": "huggingface",
        "group": "runtime",
        "desc_en": "Hugging Face cache directory",
        "desc_zh": "Hugging Face 缓存目录",
    },
    {
        "key": "HF_ENDPOINT",
        "default": "",
        "group": "runtime",
        "desc_en": "Hugging Face mirror endpoint",
        "desc_zh": "Hugging Face 镜像地址",
    },
    {
        "key": "HF_TOKEN",
        "default": "",
        "group": "runtime",
        "desc_en": "Hugging Face access token for gated or private models",
        "desc_zh": "Hugging Face 访问令牌，用于 gated 或私有模型",
    },
    {
        "key": "CUDA_VISIBLE_DEVICES",
        "default": "",
        "group": "runtime",
        "desc_en": "GPU device IDs, for example 0, 0,1, or -1 for CPU",
        "desc_zh": "GPU 设备号，例如 0、0,1，或 -1 表示 CPU",
    },
    {
        "key": "CUDA_DEVICE_ORDER",
        "default": "PCI_BUS_ID",
        "group": "runtime",
        "desc_en": "Use PCI bus order so CUDA indexes match nvidia-smi more closely",
        "desc_zh": "使用 PCI 总线顺序，让 CUDA 编号尽量匹配 nvidia-smi",
    },
    {
        "key": "CUDA_HOME",
        "default": "",
        "group": "runtime",
        "desc_en": "CUDA toolkit root; defaults from CUDA_PATH when available",
        "desc_zh": "CUDA 工具链根目录；可用时默认来自 CUDA_PATH",
    },
    {
        "key": "XFORMERS_FORCE_DISABLE_TRITON",
        "default": "1",
        "group": "runtime",
        "desc_en": "Disable Triton paths in xformers",
        "desc_zh": "禁用 xformers 中的 Triton 路径",
    },
    {
        "key": "USE_LIBUV",
        "default": "0",
        "group": "runtime",
        "desc_en": "Disable libuv TCPStore on Windows multi-GPU launches",
        "desc_zh": "禁用 Windows 多 GPU 启动中的 libuv TCPStore",
    },
    {
        "key": "PILLOW_IGNORE_XMP_DATA_IS_TOO_LONG",
        "default": "1",
        "group": "runtime",
        "desc_en": "Suppress Pillow warnings for overlong XMP metadata",
        "desc_zh": "忽略 Pillow 的 XMP 数据过长警告",
    },
    {
        "key": "VSLANG",
        "default": "1033",
        "group": "runtime",
        "desc_en": "Force English compiler/toolchain diagnostics on Windows",
        "desc_zh": "在 Windows 上强制编译器/工具链输出英文诊断",
    },
    {
        "key": "NVIDIA_TF32_OVERRIDE",
        "default": "",
        "group": "runtime",
        "desc_en": "Optional NVIDIA TF32 override; leave empty to let each task decide",
        "desc_zh": "可选 NVIDIA TF32 覆盖；留空则由具体任务决定",
    },
    {
        "key": "UV_INDEX_URL",
        "default": "",
        "group": "uv",
        "desc_en": "UV primary index URL, for example a PyPI mirror",
        "desc_zh": "UV 主索引地址，例如 PyPI 镜像",
    },
    {
        "key": "UV_EXTRA_INDEX_URL",
        "default": "",
        "group": "uv",
        "desc_en": "UV extra index URLs",
        "desc_zh": "UV 额外索引地址",
    },
    {
        "key": "UV_CACHE_DIR",
        "default": "",
        "group": "uv",
        "desc_en": "UV cache directory",
        "desc_zh": "UV 缓存目录",
    },
    {
        "key": "UV_NO_BUILD_ISOLATION",
        "default": "0",
        "group": "uv",
        "desc_en": "Disable UV build isolation when set to 1",
        "desc_zh": "设为 1 时禁用 UV 构建隔离",
    },
    {
        "key": "UV_NO_CACHE",
        "default": "0",
        "group": "uv",
        "desc_en": "Disable UV cache when set to 1",
        "desc_zh": "设为 1 时禁用 UV 缓存",
    },
    {
        "key": "UV_LINK_MODE",
        "default": "symlink",
        "group": "uv",
        "desc_en": "UV link mode, such as symlink or copy",
        "desc_zh": "UV 链接模式，例如 symlink 或 copy",
    },
    {
        "key": "UV_INDEX_STRATEGY",
        "default": "unsafe-best-match",
        "group": "uv",
        "desc_en": "UV index resolution strategy",
        "desc_zh": "UV 索引解析策略",
    },
    {
        "key": "PIP_DISABLE_PIP_VERSION_CHECK",
        "default": "1",
        "group": "uv",
        "desc_en": "Disable pip version check",
        "desc_zh": "禁用 pip 版本检查",
    },
    {
        "key": "PIP_NO_CACHE_DIR",
        "default": "1",
        "group": "uv",
        "desc_en": "Disable pip cache",
        "desc_zh": "禁用 pip 缓存",
    },
    {
        "key": "GIT_LFS_SKIP_SMUDGE",
        "default": "1",
        "group": "uv",
        "desc_en": "Skip Git LFS downloads during clone/checkout",
        "desc_zh": "克隆/检出时跳过 Git LFS 文件下载",
    },
    {
        "key": "HTTP_PROXY",
        "default": "",
        "group": "network",
        "desc_en": "HTTP proxy, for example http://127.0.0.1:7890",
        "desc_zh": "HTTP 代理，例如 http://127.0.0.1:7890",
    },
    {
        "key": "HTTPS_PROXY",
        "default": "",
        "group": "network",
        "desc_en": "HTTPS proxy, for example http://127.0.0.1:7890",
        "desc_zh": "HTTPS 代理，例如 http://127.0.0.1:7890",
    },
]


def _detect_system_language() -> str:
    env_lang = os.environ.get("LANG") or os.environ.get("LANGUAGE") or ""
    locale_lang = locale.getlocale()[0] or ""
    value = f"{env_lang} {locale_lang}".lower()
    return "zh" if "zh" in value or "chinese" in value else "en"


def _defaults() -> Dict[str, str]:
    values = {item["key"]: item["default"] for item in ENV_VAR_DEFINITIONS}
    if _detect_system_language() == "zh":
        values.setdefault("HF_ENDPOINT", "")
        values.setdefault("UV_INDEX_URL", "")
        if not values["HF_ENDPOINT"]:
            values["HF_ENDPOINT"] = "https://hf-mirror.com"
        if not values["UV_INDEX_URL"]:
            values["UV_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple/"

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and not values.get("CUDA_HOME"):
        values["CUDA_HOME"] = cuda_path

    return values


def load_env_config() -> Dict[str, str]:
    data = _defaults()
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            saved = {}
        if isinstance(saved, dict):
            data.update({str(key): str(value) for key, value in saved.items()})
    return data


def save_env_config(data: Dict[str, str]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = {str(key): str(value) for key, value in data.items()}
    CONFIG_PATH.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_env_for_subprocess() -> Dict[str, str]:
    return {key: value for key, value in load_env_config().items() if value}
