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
from musubi_tuner.training.dlssnr_parser import setup_parser  # noqa: E402
from utils import script_coverage_manifest  # noqa: E402
from utils.command_builder import build_train_job  # noqa: E402

SCRIPTS = ("3.12dlssnr_train_lora.ps1", "3.12.1dlssnr_train_db.ps1", "5.12dlssnr_generate.ps1")
PWSH = shutil.which("pwsh")


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


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
def test_training_wrappers_match_gui_effective_backend_config(script, mode, entry, dataset):
    result, arguments = capture_script(script, f"-dataset_config {ps_quote(dataset)}", culture="fr-FR")
    assert result.returncode == 0, result.stderr
    assert Path(arguments[0]).name == entry
    parsed = setup_parser(lora=mode == "lora").parse_args(arguments[1:])
    assert parsed.development_smoke is False
    assert parsed.mixed_precision == "no"
    assert "NR_PRECISION=no" in result.stdout
    assert "NR_RESTORED=bf16" in result.stdout
    gui_job = build_train_job({"arch": "DLSS-NR", "train_mode": mode, "nr_dataset_config": str(dataset)}, ROOT, {})
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


def test_generation_wrapper_normalizes_case_insensitive_device():
    result, arguments = capture_script(SCRIPTS[2], "-device CPU -inference_mode SEQUENCE")
    assert result.returncode == 0, result.stderr
    assert "--device=cpu" in arguments
    assert Path(arguments[0]).name == "dlssnr_generate_video.py"


def test_lora_wrapper_preserves_temporal_args_and_width_maps(dataset):
    ranks = '{"32": 2, "64": 4, "128": 8, "256": 8, "512": 16, "1024": 32}'
    parameters = (
        f"-dataset_config {ps_quote(dataset)} -training_mode temporal -sequence_length 4 -burn_in 2 -tbptt_length 2 "
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
        assert data["bucket_size"] == [512, 512]
        assert data["enable_bucket"] and data["bucket_no_upscale"]
        assert data["train_manifest"].endswith(".jsonl")


def test_dlssnr_wrappers_are_native_gui_workflows():
    assert set(SCRIPTS).issubset(script_coverage_manifest.NATIVE_GUI)
