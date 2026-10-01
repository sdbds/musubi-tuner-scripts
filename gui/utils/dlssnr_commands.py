"""DLSS-NR commands, isolated from diffusion form state and cache exports."""

from __future__ import annotations

import ast
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

from utils.command_builder import CommandBuildError, CommandJob, resolve_train_optimizer
from utils.optimizer_catalog import OPTIMIZER_TYPES, parse_optimizer_args

DLSSNR_ARCH = "DLSS-NR"
RUNTIME_DEFAULTS = {
    "nr_numerics_profile": "train_surrogate",
    "nr_mixed_precision": "no",
    "nr_attention_backend": "native",
    "nr_attention_scope": "all",
    "nr_fp8_base": False,
    "nr_fp8_scaled": False,
}
TRAIN_DEFAULTS = {
    **RUNTIME_DEFAULTS,
    "nr_numerics_profile": "train_experimental",
    "nr_attention_backend": "sdpa",
    "nr_gradient_checkpointing": True,
    "nr_max_overflow_retries": 16,
    "nr_num_processes": 1,
    "nr_dataset_config": "./toml/qinglong_dlssnr_single.toml",
    "nr_model_dir": "./ckpts/dlssnr/canonical",
    "nr_device": "auto",
    "nr_training_mode": "single_frame",
    "nr_seed": 42,
    "nr_max_train_steps": 1000,
    "nr_gradient_accumulation_steps": 1,
    "nr_sequence_length": 4,
    "nr_burn_in": 2,
    "nr_tbptt_length": 2,
    "nr_optimizer_type": "AdamW",
    "nr_optimizer_args": "weight_decay=0.0",
    "nr_d_coef": "0.5",
    "nr_d0": "1e-3",
    "nr_max_grad_norm": 0.0,
    "nr_loss_pre": 1.0,
    "nr_loss_out": 1.0,
    "nr_loss_edge": 0.05,
    "nr_loss_temporal": 0.0,
    "nr_sample_every_n_steps": 0,
    "nr_min_sequence_frames": 64,
    "nr_compare_baseline": True,
    "nr_output_dir": "./output_dir/dlssnr",
    "nr_save_every_n_steps": 100,
    "nr_save_state": False,
    "nr_resume": "",
    "nr_lora_profile": "vit_only",
    "nr_network_dim": 16,
    "nr_network_alpha": "",
    "nr_network_dropout": 0.0,
    "nr_rank_by_width": "",
    "nr_alpha_by_width": "",
    "nr_prior_lr_multiplier": 0.1,
    "nr_scale_lr_multiplier": 0.1,
    "nr_temporal_blend_lr_multiplier": 0.1,
}
GENERATE_DEFAULTS = {
    **RUNTIME_DEFAULTS,
    "nr_runtime_mode": "inherit",
    "nr_model_dir": "./ckpts/dlssnr/canonical",
    "nr_inference_mode": "image",
    "nr_sample_manifest": "",
    "nr_sequence_manifest": "",
    "nr_bucket_width": 512,
    "nr_bucket_height": 512,
    "nr_seed": 0,
    "nr_device": "auto",
    "nr_output_dir": "./output_dir/dlssnr_preview",
}
_INTEGER_FIELDS = (
    "seed",
    "max_train_steps",
    "gradient_accumulation_steps",
    "sample_every_n_steps",
    "min_sequence_frames",
    "save_every_n_steps",
)
_NUMBER_FIELDS = ("learning_rate", "max_grad_norm", "loss_pre", "loss_out", "loss_edge", "loss_temporal")
_FULL_FIELDS = ("prior_lr_multiplier", "scale_lr_multiplier", "temporal_blend_lr_multiplier")


def is_nr_optimizer_supported(name: str, train_mode: str = "lora") -> bool:
    key = name.rsplit(".", 1)[-1].lower().replace("_", "")
    return (
        key not in {"lbfgs", "adammini", "sophia", "sophiah"}
        and "schedulefree" not in name.lower()
        and not (key == "lorarite" and train_mode != "lora")
    )


def get_nr_optimizer_types(train_mode: str = "lora") -> list[str]:
    return [name for name in OPTIMIZER_TYPES if is_nr_optimizer_supported(name, train_mode)] + ["SGD", "Adam"]


def resolve_nr_optimizer(state: Mapping[str, Any]) -> tuple[str, list[str]]:
    name = str(state.get("nr_optimizer_type", "AdamW")).strip()
    if not name or not is_nr_optimizer_supported(name, state.get("train_mode", "lora")):
        raise CommandBuildError(f"DLSS-NR optimizer {name!r} requires an unsupported update protocol.")
    # Keep the existing NR defaults; all other aliases use the shared GUI resolver.
    if name.lower() == "adamw":
        return "AdamW", ["weight_decay=0.0"]
    if name.lower() == "adafactor":
        return "Adafactor", ["scale_parameter=False", "warmup_init=False", "relative_step=False"]
    common = {key.removeprefix("nr_"): value for key, value in state.items() if key.startswith("nr_")}
    common["optimizer_type"] = name
    return resolve_train_optimizer(common)


def get_train_defaults(train_mode: str = "lora") -> dict[str, Any]:
    return {
        **TRAIN_DEFAULTS,
        "nr_learning_rate": 1e-4 if train_mode == "lora" else 1e-5,
        "nr_output_name": "dlssnr_lora" if train_mode == "lora" else "dlssnr_full",
    }


def _integer(value: Any, name: str) -> int:
    try:
        number = Decimal(str(value).strip())
        if isinstance(value, bool) or not number.is_finite() or number != number.to_integral_value():
            raise ValueError
        return int(number)
    except (TypeError, ValueError, OverflowError, InvalidOperation):
        raise CommandBuildError(f"DLSS-NR {name} must be an integer.") from None


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
        if isinstance(value, bool) or not math.isfinite(result):
            raise ValueError
        return result
    except (TypeError, ValueError, OverflowError):
        raise CommandBuildError(f"DLSS-NR {name} must be a finite number.") from None


def validate_nr_numeric_value(
    value, name, *, integer=False, minimum=0, maximum=None, positive=False, optional=False, exclusive_max=None
):
    if optional and (value is None or isinstance(value, str) and not value.strip()):
        return
    number = _integer(value, name) if integer else _number(value, name)
    if number < minimum or positive and number == 0:
        raise CommandBuildError(f"DLSS-NR {name} must be {'positive' if positive else f'>= {minimum}'}.")
    if maximum is not None and number > maximum:
        raise CommandBuildError(f"DLSS-NR {name} must be <= {maximum}.")
    if exclusive_max is not None and number >= exclusive_max:
        raise CommandBuildError(f"DLSS-NR {name} must be < {exclusive_max}.")


def parse_nr_boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise CommandBuildError(f"DLSS-NR {name} must be a boolean.")


def validate_nr_runtime_config(state: Mapping[str, Any], *, stage: str, train_mode: str = "lora") -> dict:
    from musubi_tuner.dlssnr.runtime import default_runtime_policy, validate_runtime_policy

    if stage not in ("train", "generate"):
        raise CommandBuildError("DLSS-NR runtime stage must be train or generate.")
    defaults = TRAIN_DEFAULTS if stage == "train" else GENERATE_DEFAULTS
    values = {key: state.get(key, defaults[key]) for key in RUNTIME_DEFAULTS}
    choices = {
        "nr_numerics_profile": ("train_surrogate", "train_experimental"),
        "nr_mixed_precision": ("no", "fp16", "bf16"),
        "nr_attention_backend": ("native", "sdpa", "flash_attn", "xformers", "sage_attn"),
        "nr_attention_scope": ("all", "global"),
    }
    for name, options in choices.items():
        if values[name] not in options:
            raise CommandBuildError(f"DLSS-NR {name} must be one of {', '.join(options)}.")
    for name in ("nr_fp8_base", "nr_fp8_scaled"):
        values[name] = parse_nr_boolean(values[name], name)
    if stage == "generate":
        mode = state.get("nr_runtime_mode", "inherit")
        if mode not in ("inherit", "override"):
            raise CommandBuildError("DLSS-NR runtime_mode must be inherit or override.")
        values["nr_runtime_mode"] = mode
        if mode == "inherit":
            return values
    else:
        values["nr_gradient_checkpointing"] = parse_nr_boolean(
            state.get("nr_gradient_checkpointing", defaults["nr_gradient_checkpointing"]), "gradient_checkpointing"
        )
        for name, default, minimum in (("nr_max_overflow_retries", 16, 0), ("nr_num_processes", 1, 1)):
            values[name] = _integer(state.get(name, default), name)
            validate_nr_numeric_value(values[name], name, integer=True, minimum=minimum)
        if values["nr_fp8_base"] and train_mode != "lora":
            raise CommandBuildError("DLSS-NR FP8 base storage is supported only for LoRA training.")
    policy = default_runtime_policy()
    for key in policy:
        if f"nr_{key}" in values:
            policy[key] = values[f"nr_{key}"]
    try:
        validate_runtime_policy(policy, training=stage == "train")
    except ValueError as exc:
        raise CommandBuildError(str(exc)) from exc
    if state.get("nr_device", "auto") == "cpu":
        if policy["mixed_precision"] != "no" or policy["attention_backend"] in ("flash_attn", "xformers", "sage_attn"):
            raise CommandBuildError("DLSS-NR mixed precision and optional attention extensions require CUDA.")
    return values


def _path(value: Any, project_dir: str | Path, name: str, *, required: bool = False) -> str | None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise CommandBuildError(f"DLSS-NR {name} is required.")
        return None
    path = Path(text).expanduser()
    return str((path if path.is_absolute() else Path(project_dir) / path).resolve())


def _tokens(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        try:
            return parse_optimizer_args(value)
        except ValueError as exc:
            raise CommandBuildError(str(exc)) from exc
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise CommandBuildError(f"DLSS-NR {name} must contain key=value lines or a list.")
    return [str(item).strip() for item in values if str(item).strip()]


def _network_args(state: Mapping[str, Any]) -> list[str]:
    profile = state["nr_lora_profile"]
    args = [f"profile={profile}"]
    if profile == "multiscale":
        for name in ("rank_by_width", "alpha_by_width"):
            value = state[f"nr_{name}"]
            if value in (None, ""):
                continue
            try:
                parsed = ast.literal_eval(value) if isinstance(value, str) else value
            except (ValueError, SyntaxError):
                raise CommandBuildError(f"DLSS-NR {name} must be a dictionary.") from None
            if not isinstance(parsed, dict):
                raise CommandBuildError(f"DLSS-NR {name} must be a dictionary.")
            args.append(f"{name}={parsed!r}")
    return args


def build_dlssnr_train_job(state: Mapping[str, Any], project_dir: str | Path) -> CommandJob:
    mode = state.get("train_mode", "lora")
    if mode not in ("lora", "finetune"):
        raise CommandBuildError("DLSS-NR train_mode must be lora or finetune.")
    lora = mode == "lora"
    resolved = {**get_train_defaults(mode), **{key: value for key, value in state.items() if key.startswith("nr_")}}
    resolved["train_mode"] = mode
    resolved.update(validate_nr_runtime_config(resolved, stage="train", train_mode=mode))
    dataset_path = _path(resolved["nr_dataset_config"], project_dir, "dataset_config", required=True)
    if not Path(dataset_path).is_file():
        raise CommandBuildError(f"DLSS-NR dataset_config does not exist: {dataset_path}")

    values = {name: _integer(resolved[f"nr_{name}"], name) for name in _INTEGER_FIELDS}
    values.update({name: _number(resolved[f"nr_{name}"], name) for name in _NUMBER_FIELDS})
    for name in ("dataset_config", "model_dir", "output_dir", "resume"):
        values[name] = _path(
            resolved[f"nr_{name}"], project_dir, name, required=name in {"dataset_config", "model_dir", "output_dir"}
        )
    for name in ("training_mode", "device", "output_name"):
        values[name] = str(resolved[f"nr_{name}"]).strip()
    for name in ("compare_baseline", "save_state"):
        values[name] = parse_nr_boolean(resolved[f"nr_{name}"], name)
    optimizer_type, template_args = resolve_nr_optimizer(resolved)
    values.update(
        profile="dlss_nr_310_8_0",
        lr_scheduler=resolved.get("nr_lr_scheduler", "constant"),
        optimizer_type=optimizer_type,
        optimizer_args=_tokens(state.get("nr_optimizer_args", template_args), "optimizer_args"),
        development_smoke=False,
        forward_validation_report=None,
        deployment_target="float_runtime",
    )
    for key in (*RUNTIME_DEFAULTS, "nr_gradient_checkpointing", "nr_max_overflow_retries"):
        values[key.removeprefix("nr_")] = resolved[key]
    if values["training_mode"] == "temporal":
        for name in ("sequence_length", "burn_in", "tbptt_length"):
            values[name] = _integer(resolved[f"nr_{name}"], name)
    else:
        values.update(sequence_length=None, burn_in=None, tbptt_length=None)
    if lora:
        multiscale = resolved["nr_lora_profile"] == "multiscale"
        values["network_dim"] = None if multiscale else _integer(resolved["nr_network_dim"], "lora.rank")
        alpha = resolved["nr_network_alpha"]
        values["network_alpha"] = None if multiscale or alpha in (None, "") else _number(alpha, "lora.alpha")
        values["network_dropout"] = _number(resolved["nr_network_dropout"], "lora.dropout")
        values["network_args"] = _network_args(resolved)
    else:
        values.update({name: _number(resolved[f"nr_{name}"], name) for name in _FULL_FIELDS})

    args = []
    for name, value in values.items():
        if name == "compare_baseline":
            if not value:
                args.append("--no-compare_baseline")
        elif isinstance(value, bool):
            if value:
                args.append(f"--{name}")
        elif isinstance(value, list):
            if value:
                args.extend([f"--{name}", *value])
        elif value is not None:
            args.append(f"--{name}={value}")
    # Parse the actual argv so newly added backend defaults cannot break the GUI namespace.
    try:
        from musubi_tuner.dlssnr.config import build_train_config
        from musubi_tuner.training.dlssnr_parser import setup_parser

        def invalid_arguments(message):
            raise CommandBuildError(message)

        parser = setup_parser(lora=lora)
        parser.error = invalid_arguments
        build_train_config(parser.parse_args(args), lora=lora)
    except ImportError as exc:
        raise CommandBuildError(f"DLSS-NR backend is unavailable: {exc}") from exc
    except (OSError, TypeError, ValueError, SystemExit) as exc:
        raise CommandBuildError(f"Invalid DLSS-NR training configuration: {exc}") from exc
    runner_kwargs = {
        "use_accelerate": False,
        "env_vars": {
            "ACCELERATE_MIXED_PRECISION": values["mixed_precision"],
            "ACCELERATE_USE_CPU": "true" if values["device"] == "cpu" else "false",
            **{f"ACCELERATE_USE_{name}": "false" for name in ("DEEPSPEED", "FSDP", "TP", "MEGATRON_LM")},
        },
    }
    if resolved["nr_num_processes"] > 1:
        runner_kwargs.update(use_torchrun=True, num_processes=resolved["nr_num_processes"])
    return CommandJob(
        name=f"DLSS-NR {'LoRA' if lora else 'Full'} Train",
        script_key=f"musubi_tuner.dlssnr_{'train_network' if lora else 'train'}",
        args=args,
        runner_kwargs=runner_kwargs,
    )


def build_dlssnr_generate_job(state: Mapping[str, Any], project_dir: str | Path) -> CommandJob:
    resolved = {**GENERATE_DEFAULTS, **{key: value for key, value in state.items() if key.startswith("nr_")}}
    resolved.update(validate_nr_runtime_config(resolved, stage="generate"))
    mode = resolved["nr_inference_mode"]
    if mode not in ("image", "sequence"):
        raise CommandBuildError("DLSS-NR inference_mode must be image or sequence.")
    manifest_key = "sample_manifest" if mode == "image" else "sequence_manifest"
    values = {
        name: _path(resolved[f"nr_{name}"], project_dir, name, required=True) for name in ("model_dir", manifest_key, "output_dir")
    }
    for name in ("bucket_width", "bucket_height", "seed"):
        values[name] = _integer(resolved[f"nr_{name}"], name)
    if values["seed"] < 0:
        raise CommandBuildError("DLSS-NR seed must be nonnegative.")
    device = resolved["nr_device"]
    if device not in ("auto", "cpu", "cuda"):
        raise CommandBuildError("DLSS-NR device must be auto, cpu or cuda.")
    try:
        from musubi_tuner.dlssnr.geometry import resolve_geometry

        resolve_geometry(values["bucket_width"], values["bucket_height"])
    except ImportError as exc:
        raise CommandBuildError(f"DLSS-NR backend is unavailable: {exc}") from exc
    except ValueError as exc:
        raise CommandBuildError(str(exc)) from exc
    values["device"] = device
    args = [f"--{name}={value}" for name, value in values.items()]
    if resolved["nr_runtime_mode"] == "override":
        for key in RUNTIME_DEFAULTS:
            name, value = key.removeprefix("nr_"), resolved[key]
            args.append(f"--{'' if value else 'no-'}{name}" if isinstance(value, bool) else f"--{name}={value}")
    return CommandJob(
        name=f"DLSS-NR {'Image' if mode == 'image' else 'PNG Sequence'}",
        script_key=f"musubi_tuner.dlssnr_generate_{'image' if mode == 'image' else 'video'}",
        args=args,
        runner_kwargs={"use_accelerate": False},
    )
