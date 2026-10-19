import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "musubi-tuner" / "src"))

from musubi_tuner.dlssnr.config import build_train_config, load_dataset_config  # noqa: E402
from musubi_tuner.dlssnr.infer import add_runtime_arguments, runtime_overrides_from_args  # noqa: E402
from musubi_tuner.training.dlssnr_parser import setup_parser  # noqa: E402
from utils import script_coverage_manifest  # noqa: E402
from utils.command_builder import build_train_job  # noqa: E402

SCRIPTS = ("3.12dlssnr_train_lora.ps1", "3.12.1dlssnr_train_db.ps1", "5.12dlssnr_generate.ps1")
PWSH = shutil.which("pwsh")


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def training_inputs(dataset):
    return f"-dataset_config {ps_quote(dataset)} -model_dir {ps_quote(dataset.parent / 'canonical')}"


def capture_script(script, parameters="", *, exit_code=0, culture="en-US"):
    if not PWSH:
        pytest.skip("PowerShell 7 is unavailable")
    command = (
        f"[System.Threading.Thread]::CurrentThread.CurrentCulture = '{culture}'; "
        "$env:ACCELERATE_MIXED_PRECISION = 'bf16'; "
        "function python { "
        "[Console]::WriteLine('NR_ARGV=' + (ConvertTo-Json -Compress -InputObject @($args))); "
        "[Console]::WriteLine('NR_PRECISION=' + $env:ACCELERATE_MIXED_PRECISION); "
        f"$global:LASTEXITCODE = {exit_code}; "
        "}; "
        f"& {ps_quote(ROOT / script)} -python_executable python {parameters}; "
        "[Console]::WriteLine('NR_RESTORED=' + $env:ACCELERATE_MIXED_PRECISION)"
    )
    result = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, encoding="utf-8", timeout=15
    )
    arguments = next((json.loads(line[8:]) for line in result.stdout.splitlines() if line.startswith("NR_ARGV=")), None)
    return result, arguments


@pytest.fixture
def dataset(tmp_path):
    model = tmp_path / "canonical"
    model.mkdir()
    (model / "model_config.json").write_text(
        json.dumps({"schema": "dlssnr_canonical_v1", "profile": "dlss_nr_310_8_0"}), encoding="utf-8"
    )
    path = tmp_path / "paired data.toml"
    path.write_text(
        "[general]\nresolution=[512,512]\nbatch_size=1\nenable_bucket=true\nbucket_no_upscale=true\n"
        '[[datasets]]\ntrain_manifest="train pairs.jsonl"\n',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "script, mode, entry",
    [
        (SCRIPTS[0], "lora", "dlssnr_train_network.py"),
        (SCRIPTS[1], "finetune", "dlssnr_train.py"),
    ],
)
def test_training_wrappers_match_gui_mode_defaults_with_explicit_attention(script, mode, entry, dataset):
    result, arguments = capture_script(script, training_inputs(dataset), culture="fr-FR")
    assert result.returncode == 0, result.stderr
    assert Path(arguments[0]).name == entry
    parsed = setup_parser(lora=mode == "lora").parse_args(arguments[1:])
    assert parsed.development_smoke is False
    full = mode == "finetune"
    assert parsed.mixed_precision == ("fp16" if full else "no")
    assert parsed.numerics_profile == ("train_experimental" if full else "train_surrogate")
    assert parsed.native_weight_qat is full
    assert parsed.ema_decay == (0.99 if full else None)
    assert parsed.profile == "auto" and parsed.save_precision == "native"
    assert parsed.attention_backend == "native"
    assert parsed.gradient_checkpointing is False
    assert parsed.gradient_checkpointing_min_pixels == 0
    assert f"NR_PRECISION={'fp16' if full else 'no'}" in result.stdout
    assert "NR_RESTORED=bf16" in result.stdout
    gui_job = build_train_job(
        {
            "arch": "DLSS-NR",
            "train_mode": mode,
            "nr_dataset_config": str(dataset),
            "nr_model_dir": str(dataset.parent / "canonical"),
            "nr_numerics_profile": "train_experimental" if full else "train_surrogate",
            "nr_attention_backend": "native",
            "nr_gradient_checkpointing": False,
        },
        ROOT,
        {},
    )
    gui_parsed = setup_parser(lora=mode == "lora").parse_args(gui_job.args)
    assert build_train_config(parsed, lora=mode == "lora") == build_train_config(gui_parsed, lora=mode == "lora")


@pytest.mark.parametrize("script, lora", [(SCRIPTS[0], True), (SCRIPTS[1], False)])
def test_training_wrapper_normalizes_case_insensitive_powershell_choices(script, lora, dataset):
    result, arguments = capture_script(
        script, f"-dataset_config {ps_quote(dataset)} -device CPU -deployment_target FLOAT_RUNTIME -training_mode SINGLE_FRAME"
    )
    assert result.returncode == 0, result.stderr
    parsed = setup_parser(lora=lora).parse_args(arguments[1:])
    assert parsed.device == "cpu"
    assert parsed.deployment_target == "float_runtime"
    assert parsed.training_mode == "single_frame"


@pytest.mark.parametrize("script,lora", [(SCRIPTS[0], True), (SCRIPTS[1], False)])
def test_training_wrappers_forward_advanced_options(script, lora, dataset):
    dataset.write_text(
        '[general]\nresolution=[64,48]\n[[datasets]]\ntrain_manifest="train.jsonl"\n'
        'validation_manifest="validation.jsonl"\nnr_controls_mode="fixed"\n',
        encoding="utf-8",
    )
    result, arguments = capture_script(
        script,
        f"{training_inputs(dataset)} -native_weight_qat -shuffle_dataset $false -ema_decay 0.99 "
        "-numerics_profile train_surrogate -mixed_precision no "
        "-loss_profile FREQUENCY_SPLIT -loss_lowpass_sigma 4 -base_anchor_weight 0.2 "
        "-control_randomization -control_residual_sigma 5 -control_anchor_probability 0.4 -control_corner_probability 0.2 "
        "-dino_loss_weight 0.1 -dino_loss_model_type BASE -dino_loss_layer -2 -dino_loss_resize 256 "
        "-dino_loss_use_gram $false -dino_loss_use_norm $false "
        "-eval_native -eval_detail_diagnostics -eval_content_preservation",
    )
    assert result.returncode == 0 and arguments is not None, result.stderr
    parsed = setup_parser(lora=lora).parse_args(arguments[1:])
    config = build_train_config(parsed, lora=lora)
    assert config["model"]["native_weight_qat"] and not config["training"]["shuffle_dataset"]
    assert config["training"]["ema_decay"] == 0.99
    assert config["loss_profile"] == {"name": "frequency_split", "lowpass_sigma": 4}
    assert config["loss"]["base_anchor"] == 0.2 and config["loss"]["dino"] == 0.1
    assert config["control_randomization"]["corner_probability"] == 0.2
    assert config["dino_loss"] == {"model_type": "base", "layer": -2, "resize": 256, "use_gram": False, "use_norm": False}
    assert all(config["evaluation"][name] for name in ("native", "detail_diagnostics", "content_preservation"))
    job = build_train_job(
        {
            "arch": "DLSS-NR",
            "train_mode": "lora" if lora else "finetune",
            "nr_dataset_config": str(dataset),
            "nr_model_dir": str(dataset.parent / "canonical"),
            "nr_numerics_profile": "train_surrogate",
            "nr_mixed_precision": "no",
            "nr_attention_backend": "native",
            "nr_gradient_checkpointing": False,
            "nr_native_weight_qat": True,
            "nr_shuffle_dataset": False,
            "nr_ema_decay": 0.99,
            "nr_loss_profile": "frequency_split",
            "nr_loss_lowpass_sigma": 4,
            "nr_base_anchor_weight": 0.2,
            "nr_control_randomization": True,
            "nr_control_residual_sigma": 5,
            "nr_control_anchor_probability": 0.4,
            "nr_control_corner_probability": 0.2,
            "nr_dino_loss_weight": 0.1,
            "nr_dino_loss_model_type": "base",
            "nr_dino_loss_layer": -2,
            "nr_dino_loss_resize": 256,
            "nr_dino_loss_use_gram": False,
            "nr_dino_loss_use_norm": False,
            "nr_eval_native": True,
            "nr_eval_detail_diagnostics": True,
            "nr_eval_content_preservation": True,
        },
        ROOT,
        {},
    )
    assert build_train_config(setup_parser(lora=lora).parse_args(job.args), lora=lora) == config


@pytest.mark.parametrize(
    "extra,message",
    [
        ("-dino_loss_model_type small", "DINO options require"),
        ("-loss_lowpass_sigma 6", "loss_lowpass_sigma requires"),
        ("-control_anchor_probability 0.25", "control-specific options require"),
    ],
)
def test_training_wrappers_do_not_silently_drop_explicit_inactive_options(dataset, extra, message):
    result, arguments = capture_script(SCRIPTS[0], f"{training_inputs(dataset)} {extra}")
    assert result.returncode == 0 and arguments is not None, result.stderr
    with pytest.raises(ValueError, match=message):
        build_train_config(setup_parser(lora=True).parse_args(arguments[1:]), lora=True)


@pytest.mark.parametrize("script,lora", [(SCRIPTS[0], True), (SCRIPTS[1], False)])
def test_training_wrappers_expose_runtime_and_scheduler_instead_of_hardcoding(script, lora, dataset):
    result, arguments = capture_script(
        script,
        f"{training_inputs(dataset)} -numerics_profile TRAIN_EXPERIMENTAL -mixed_precision BF16 "
        "-attention_backend SDPA -attention_scope ALL -gradient_checkpointing -gradient_checkpointing_min_pixels 20000 "
        "-max_overflow_retries 3 "
        "-lr_scheduler COSINE_WITH_MIN_LR -lr_warmup_steps 0.1 -lr_decay_steps 0.2 "
        "-lr_scheduler_num_cycles 2 -lr_scheduler_power 2 -lr_scheduler_timescale 16 -lr_scheduler_min_lr_ratio 0.05",
        culture="fr-FR",
    )
    assert result.returncode == 0 and arguments is not None, result.stderr
    parsed = setup_parser(lora=lora).parse_args(arguments[1:])
    config = build_train_config(parsed, lora=lora)
    assert parsed.mixed_precision == "bf16" and parsed.attention_backend == "sdpa" and parsed.gradient_checkpointing
    assert parsed.gradient_checkpointing_min_pixels == 20000
    assert parsed.max_overflow_retries == 3
    assert config["optimizer"]["scheduler"]["lr_warmup_steps"] == 0.1
    assert config["optimizer"]["scheduler"]["lr_scheduler_min_lr_ratio"] == 0.05
    assert "NR_PRECISION=bf16" in result.stdout and "NR_RESTORED=bf16" in result.stdout


def test_synthetic_temporal_template_is_consumable_by_backend():
    path = ROOT / "toml/qinglong_dlssnr_synthetic_temporal.toml"
    assert path.is_file()
    data = load_dataset_config(path)
    assert data["bucket_size"] == [1024, 1024]
    assert data["synthetic_temporal"] == {"schema": "dlssnr_synthetic_temporal_v1", "max_shift_px": 0.5}


@pytest.mark.parametrize("script,lora", [(SCRIPTS[0], True), (SCRIPTS[1], False)])
@pytest.mark.parametrize(
    "options,message",
    [
        ("-gradient_checkpointing_min_pixels 20000", "requires --gradient_checkpointing"),
        ("-gradient_checkpointing -gradient_checkpointing_min_pixels -1", "gradient_checkpointing_min_pixels"),
    ],
)
def test_wrappers_preserve_checkpoint_threshold_validation(script, lora, dataset, options, message):
    result, arguments = capture_script(script, f"{training_inputs(dataset)} {options}")
    assert result.returncode == 0 and arguments is not None, result.stderr
    with pytest.raises(ValueError, match=message):
        build_train_config(setup_parser(lora=lora).parse_args(arguments[1:]), lora=lora)


@pytest.mark.parametrize("script,lora", [(SCRIPTS[0], True), (SCRIPTS[1], False)])
def test_training_wrappers_accept_cc_profile_and_fp32_storage(script, lora, dataset):
    (dataset.parent / "canonical" / "model_config.json").write_text(
        json.dumps({"schema": "dlssnr_canonical_v1", "profile": "dlss_nr_310_9_16_cc"}), encoding="utf-8"
    )
    result, arguments = capture_script(script, f"{training_inputs(dataset)} -profile DLSS_NR_310_9_16_CC -save_precision FP32")
    assert result.returncode == 0 and arguments is not None, result.stderr
    config = build_train_config(setup_parser(lora=lora).parse_args(arguments[1:]), lora=lora)
    assert config["model"]["profile"] == "dlss_nr_310_9_16_cc"
    assert config["output"]["save_precision"] == "fp32"


@pytest.mark.parametrize("ema_option", ["-no_ema", "-ema_decay ''"])
def test_full_wrapper_can_disable_qat_amp_and_ema_defaults(dataset, ema_option):
    result, arguments = capture_script(
        SCRIPTS[1], f"{training_inputs(dataset)} -native_weight_qat:$false -mixed_precision no {ema_option}"
    )
    assert result.returncode == 0 and arguments is not None, result.stderr
    parsed = setup_parser().parse_args(arguments[1:])
    assert parsed.native_weight_qat is False and parsed.ema_decay is None and parsed.mixed_precision == "no"
    assert "--no-native_weight_qat" in arguments and "--no-ema" in arguments
    config = build_train_config(parsed)
    assert "ema_decay" not in config["training"]


@pytest.mark.parametrize("mode", ["image", "sequence"])
def test_generation_wrapper_inherits_the_saved_model_runtime(mode):
    result, arguments = capture_script(SCRIPTS[2], f"-inference_mode {mode}")
    assert result.returncode == 0 and arguments is not None, result.stderr
    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parsed, _ = parser.parse_known_args(arguments[1:])
    assert runtime_overrides_from_args(parsed) == {}


def test_generation_wrapper_normalizes_case_insensitive_device():
    result, arguments = capture_script(SCRIPTS[2], "-device CPU -inference_mode SEQUENCE")
    assert result.returncode == 0, result.stderr
    assert "--device=cpu" in arguments
    assert Path(arguments[0]).name == "dlssnr_generate_video.py"


def test_lora_wrapper_preserves_temporal_args_and_width_maps(dataset):
    ranks = '{"32": 2, "64": 4, "128": 8, "256": 8, "512": 16, "1024": 32}'
    parameters = (
        f"{training_inputs(dataset)} -training_mode temporal -sequence_length 4 -burn_in 2 -tbptt_length 2 "
        f"-loss_temporal 0.1 -lora_profile multiscale -rank_by_width {ps_quote(ranks)} "
        "-optimizer_args @('weight_decay=0.0','betas=(0.8, 0.99)') -save_state -development_smoke"
    )
    result, arguments = capture_script(SCRIPTS[0], parameters)
    assert result.returncode == 0, result.stderr
    parsed = setup_parser(lora=True).parse_args(arguments[1:])
    config = build_train_config(parsed, lora=True)
    assert parsed.network_dim is None and parsed.network_alpha is None
    assert parsed.save_state and parsed.development_smoke
    assert config["loss"]["temporal"] == 0.1
    assert config["lora"]["rank_by_width"]["1024"] == 32
    assert "betas=(0.8, 0.99)" in config["optimizer"]["args"]


@pytest.mark.parametrize(
    "mode, entry, manifest_flag",
    [
        ("image", "dlssnr_generate_image.py", "--sample_manifest="),
        ("sequence", "dlssnr_generate_video.py", "--sequence_manifest="),
    ],
)
def test_generation_wrapper_selects_the_matching_manifest(mode, entry, manifest_flag):
    result, arguments = capture_script(
        SCRIPTS[2],
        f"-inference_mode {mode} -sample_manifest 'still pairs.jsonl' -sequence_manifest 'clip pairs.jsonl'",
    )
    assert result.returncode == 0, result.stderr
    assert Path(arguments[0]).name == entry
    manifest = "still pairs.jsonl" if mode == "image" else "clip pairs.jsonl"
    assert f"{manifest_flag}{manifest}" in arguments
    assert not any(arg.startswith("--prompt") or arg.startswith("--lora_weight") for arg in arguments)
    assert "-m" not in arguments


def test_training_wrapper_propagates_native_failure(dataset, tmp_path):
    if not PWSH:
        pytest.skip("PowerShell 7 is unavailable")
    executable = tmp_path / ("failing-python.cmd" if sys.platform == "win32" else "failing-python")
    executable.write_text("@exit /b 7\n" if sys.platform == "win32" else "#!/bin/sh\nexit 7\n", encoding="utf-8")
    executable.chmod(0o755)
    result = subprocess.run(
        [
            PWSH,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(ROOT / SCRIPTS[0]),
            "-dataset_config",
            str(dataset),
            "-python_executable",
            str(executable),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 7, result.stderr
    assert "exit code: 7" in result.stderr


@pytest.mark.parametrize(
    "parameters, message",
    [
        ("-model_dir ''", "model_dir"),
        ("-optimizer_args @('--development_smoke')", "optimizer_args"),
    ],
)
def test_training_wrapper_rejects_missing_model_or_option_injection(dataset, parameters, message):
    result, arguments = capture_script(SCRIPTS[0], f"-dataset_config {ps_quote(dataset)} {parameters}")
    assert result.returncode != 0
    assert arguments is None
    assert message in result.stderr


def test_dataset_templates_are_dataset_only_and_enable_shared_buckets():
    for name in ("qinglong_dlssnr_single.toml", "qinglong_dlssnr_temporal.toml"):
        data = load_dataset_config(ROOT / "toml" / name)
        assert data["bucket_size"] == [1024, 1024]
        assert data["enable_bucket"] and data["bucket_no_upscale"]
        if name == "qinglong_dlssnr_single.toml":
            assert Path(data["image_directory"]).name == "target"
            assert Path(data["control_directory"]).name == "input"
            assert data["fixed_controls"]["nr_auto_mask"] is True
        else:
            assert data["train_manifest"].endswith(".jsonl")


def test_dlssnr_wrappers_are_native_gui_workflows():
    assert set(SCRIPTS).issubset(script_coverage_manifest.NATIVE_GUI)
