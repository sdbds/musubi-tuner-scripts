"""Exercise the wrappers, replacing only the expensive Python process boundary."""

import importlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "musubi-tuner/src/musubi_tuner"
sys.path.insert(0, str(SOURCE.parent))
PWSH = shutil.which("pwsh")
ARCHES = (("qwen_image_21", "13"), ("moss_image", "14"))


def script(architecture, number, task):
    prefix, suffix = {
        "cache": (f"2.{number}", "cache_latent_and_text_encoder"),
        "lora": (f"3.{number}", "train_lora"),
        "full": (f"3.{number}.1", "train_db"),
        "generate": (f"5.{number}", "generate"),
    }[task]
    return ROOT / f"{prefix}{architecture}_{suffix}.ps1"


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def capture(path, parameters="", exit_code=0, culture="en-US"):
    if not PWSH:
        pytest.skip("PowerShell 7 is unavailable")
    assert path.is_file(), f"Missing workflow: {path.name}"
    command = (
        "$ErrorActionPreference='Stop'; "
        f"[System.Threading.Thread]::CurrentThread.CurrentCulture='{culture}'; "
        "function global:python { "
        "[Console]::WriteLine('IMAGE_ARGV=' + (ConvertTo-Json -Compress -InputObject @($args))); "
        "[Console]::WriteLine('IMAGE_PYTHONPATH=' + $env:PYTHONPATH); "
        f"$global:LASTEXITCODE={exit_code}; "
        "}; "
        f"& {quote(path)} {parameters}; if ($LASTEXITCODE) {{ exit $LASTEXITCODE }}"
    )
    result = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )
    commands = [json.loads(line[11:]) for line in result.stdout.splitlines() if line.startswith("IMAGE_ARGV=")]
    return result, commands


def cli_parser(source):
    """Use upstream parsers without loading checkpoints or launching training."""
    module_name = source.removesuffix(".py").replace("/", ".")
    return importlib.import_module(f"musubi_tuner.{module_name}").setup_parser()


def module_args(command, architecture, suffix, training=False):
    module = f"musubi_tuner.{architecture}_{suffix}"
    if training:
        assert command[:3] == ["-m", "accelerate.commands.launch", "--num_processes=1"]
        assert command[command.index("--module") + 1] == module
        return command[command.index(module) + 1 :]
    assert command[:2] == ["-m", module]
    return command[2:]


@pytest.mark.parametrize("architecture,number", ARCHES)
def test_cache_runs_both_native_modules_and_preserves_dataset_paths(architecture, number):
    result, commands = capture(
        script(architecture, number, "cache"), "-dataset_config 'data set.toml' -skip_existing -batch_size 2"
    )
    assert result.returncode == 0, result.stderr
    assert len(commands) == 2
    for command, suffix in zip(commands, ("cache_latents", "cache_text_encoder_outputs")):
        values = module_args(command, architecture, suffix)
        parsed = cli_parser(f"{architecture}_{suffix}.py").parse_args(values)
        assert parsed.dataset_config == "data set.toml" and parsed.batch_size == 2 and parsed.skip_existing
    if architecture == "moss_image":
        assert "--vae_dtype=float32" in commands[0]
        assert "--moss_max_token_length=4096" in commands[1]
    assert str(ROOT / "musubi-tuner/src") in result.stdout


@pytest.mark.parametrize("architecture,number", ARCHES)
@pytest.mark.parametrize("mode,suffix", (("lora", "train_network"), ("full", "train")))
def test_training_argv_uses_native_mode_and_safe_defaults(architecture, number, mode, suffix):
    result, commands = capture(script(architecture, number, mode), culture="fr-FR")
    assert result.returncode == 0, result.stderr
    values = module_args(commands[0], architecture, suffix, training=True)
    parsed = cli_parser(f"{architecture}_{suffix}.py").parse_args(values)
    assert parsed.mixed_precision == "bf16" and parsed.sdpa and parsed.gradient_checkpointing
    assert parsed.learning_rate == (1e-4 if mode == "lora" else (5e-6 if architecture == "qwen_image_21" else 1e-5))
    assert parsed.discrete_flow_shift == (3.0 if architecture == "qwen_image_21" else 1.0)
    assert parsed.sample_prompts is None
    assert not any(value.startswith(("--vae=", "--text_encoder=", "--model_version", "--edit")) for value in values)
    if mode == "lora":
        assert parsed.network_module == f"networks.lora_{architecture}"
        assert parsed.network_dim == (16 if architecture == "qwen_image_21" else 32)
    else:
        assert not any(value.startswith("--network_") or value.startswith("--fp8") for value in values)
        assert parsed.mem_eff_save and parsed.save_state_on_train_end


@pytest.mark.parametrize("architecture,number", ARCHES)
@pytest.mark.parametrize("mode", ("lora", "full"))
def test_training_sampling_resume_and_list_arguments_are_forwarded(architecture, number, mode):
    result, commands = capture(
        script(architecture, number, mode),
        "-enable_sample -sample_at_first -sample_prompts 'sample prompts.txt' -sample_every_n_steps 12 "
        "-resume 'state directory' -optimizer_type Adafactor "
        "-optimizer_args @('relative_step=False','scale_parameter=False') -learning_rate 0.00003",
        culture="fr-FR",
    )
    assert result.returncode == 0, result.stderr
    values = commands[0]
    assert "--sample_prompts=sample prompts.txt" in values and "--sample_at_first" in values
    assert "--resume=state directory" in values and "--learning_rate=3E-05" in values
    assert values[values.index("--optimizer_args") + 1 : values.index("--optimizer_args") + 3] == [
        "relative_step=False",
        "scale_parameter=False",
    ]
    assert any(value.startswith("--vae=") for value in values) and any(value.startswith("--text_encoder=") for value in values)


@pytest.mark.parametrize("architecture,number", ARCHES)
def test_generation_uses_architecture_specific_size_and_sampler_flags(architecture, number):
    result, commands = capture(script(architecture, number, "generate"), "-prompt 'a red cup' -width 768 -height 1024")
    assert result.returncode == 0, result.stderr
    values = module_args(commands[0], architecture, "generate_image")
    parsed = cli_parser(f"{architecture}_generate_image.py").parse_args(values)
    assert parsed.prompt == "a red cup" and parsed.seed == 42
    assert not any(value.startswith("--lora_weight") for value in values)
    if architecture == "qwen_image_21":
        assert parsed.image_size == [1024, 768] and parsed.infer_steps == 40 and parsed.guidance_scale == 1
        assert parsed.flow_shift is None and parsed.attn_mode == "sdpa"
    else:
        assert parsed.width == 768 and parsed.height == 1024 and parsed.infer_steps == 50
        assert parsed.guidance_scale == 3 and parsed.flow_shift == 5 and parsed.uncond_skip_layers == "22"
        assert parsed.attn_mode == "torch"


def test_qwen_generation_preserves_ordered_controls_mask_and_lora_lists():
    result, commands = capture(
        script("qwen_image_21", "13", "generate"),
        "-control_image_path @('first ref.png','second.png') -mask_path 'mask.png' -resize_control_to_image_size "
        "-lora_weight @('adapter one.safetensors','adapter two.safetensors') -lora_multiplier @(0.7,1.2)",
    )
    assert result.returncode == 0, result.stderr
    parsed = cli_parser("qwen_image_21_generate_image.py").parse_args(commands[0][2:])
    assert parsed.control_image_path == ["first ref.png", "second.png"] and parsed.mask_path == "mask.png"
    assert parsed.resize_control_to_image_size and parsed.lora_multiplier == [0.7, 1.2]


@pytest.mark.parametrize("architecture,number", ARCHES)
@pytest.mark.parametrize("task", ("cache", "lora", "full", "generate"))
def test_native_failure_propagates_without_continuing(architecture, number, task):
    result, commands = capture(script(architecture, number, task), exit_code=7)
    assert result.returncode == 7, result.stderr
    assert len(commands) == 1 and "exit code: 7" in result.stderr


@pytest.mark.parametrize(
    "architecture,number,task,parameters",
    (
        ("qwen_image_21", "13", "lora", "-blocks_to_swap 32"),
        ("qwen_image_21", "13", "full", "-fused_backward_pass"),
        ("moss_image", "14", "full", "-blocks_to_swap 4 -gradient_accumulation_steps 2"),
        ("moss_image", "14", "generate", "-width 1000"),
        ("qwen_image_21", "13", "generate", "-mask_path 'mask.png'"),
        ("qwen_image_21", "13", "generate", "-lora_multiplier @(0.5)"),
        ("moss_image", "14", "generate", "-lora_weight @('a','b') -lora_multiplier @(0.5)"),
        ("qwen_image_21", "13", "full", "-fp8_base"),
        ("moss_image", "14", "lora", "-fp8_base"),
        ("moss_image", "14", "full", "-network_dim 32"),
        ("qwen_image_21", "13", "lora", "-fp8_scaled"),
    ),
)
def test_invalid_combinations_fail_before_loading_models(architecture, number, task, parameters):
    result, commands = capture(script(architecture, number, task), parameters)
    assert result.returncode != 0 and not commands


@pytest.mark.parametrize("architecture,number", ARCHES)
def test_full_swap_and_fused_backward_forward_only_valid_full_options(architecture, number):
    result, commands = capture(
        script(architecture, number, "full"),
        "-blocks_to_swap 4 -optimizer_type Adafactor -fused_backward_pass -max_grad_norm 0 -full_bf16",
    )
    assert result.returncode == 0, result.stderr
    assert all(flag in commands[0] for flag in ("--blocks_to_swap=4", "--fused_backward_pass", "--full_bf16"))
    assert not any(value.startswith("--network_") for value in commands[0])


def test_qwen_cfg_can_use_the_explicit_empty_unconditional_prompt():
    result, commands = capture(script("qwen_image_21", "13", "generate"), "-guidance_scale 3")
    assert result.returncode == 0, result.stderr
    parsed = cli_parser("qwen_image_21_generate_image.py").parse_args(commands[0][2:])
    assert parsed.guidance_scale == 3 and parsed.negative_prompt == ""


def test_qwen_cpu_caching_uses_float32_text_without_quantization():
    result, commands = capture(script("qwen_image_21", "13", "cache"), "-device cpu")
    assert result.returncode == 0, result.stderr
    assert "--text_encoder_dtype=float32" in commands[1] and "--fp8_vl" not in commands[1]


def test_moss_cache_generation_replaces_text_encoder_and_can_disable_skipping():
    result, commands = capture(
        script("moss_image", "14", "generate"), "-text_cache 'caption cache.safetensors' -uncond_skip_layers ''"
    )
    assert result.returncode == 0, result.stderr
    parsed = cli_parser("moss_image_generate_image.py").parse_args(commands[0][2:])
    assert parsed.text_cache == "caption cache.safetensors" and parsed.uncond_skip_layers == ""
    assert parsed.prompt is None and parsed.text_encoder is None


def test_real_native_failure_preserves_exit_code_when_script_is_invoked_directly(tmp_path):
    if not PWSH:
        pytest.skip("PowerShell 7 is unavailable")
    executable = tmp_path / ("failing-python.cmd" if sys.platform == "win32" else "failing-python")
    executable.write_text("@exit /b 9\n" if sys.platform == "win32" else "#!/bin/sh\nexit 9\n", encoding="utf-8")
    executable.chmod(0o755)
    result = subprocess.run(
        [
            PWSH,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script("moss_image", "14", "cache")),
            "-python_executable",
            str(executable),
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )
    assert result.returncode == 9 and "exit code: 9" in result.stderr


def test_moss_full_optimizer_can_change_without_leaking_adafactor_arguments():
    result, commands = capture(script("moss_image", "14", "full"), "-optimizer_type AdamW")
    assert result.returncode == 0, result.stderr
    parsed = cli_parser("moss_image_train.py").parse_args(module_args(commands[0], "moss_image", "train", training=True))
    assert parsed.optimizer_type == "AdamW" and not parsed.optimizer_args


@pytest.mark.parametrize("architecture,number", ARCHES)
@pytest.mark.parametrize("mode", ("lora", "full"))
def test_ordinary_training_forwards_custom_optimizer_without_restriction(architecture, number, mode):
    result, commands = capture(script(architecture, number, mode), "-optimizer_type custom.optimizer.Class")
    assert result.returncode == 0, result.stderr
    assert "--optimizer_type=custom.optimizer.Class" in commands[0]


@pytest.mark.parametrize("architecture,number", ARCHES)
def test_full_swap_accepts_native_torch_adamw_class_path(architecture, number):
    result, commands = capture(script(architecture, number, "full"), "-optimizer_type torch.optim.AdamW -blocks_to_swap 2")
    assert result.returncode == 0, result.stderr
    assert "--optimizer_type=torch.optim.AdamW" in commands[0]
    assert "--block_swap_optimizer_patch_params" in commands[0]


def test_missing_executable_fails_instead_of_reporting_a_successful_cache(tmp_path):
    if not PWSH:
        pytest.skip("PowerShell 7 is unavailable")
    result = subprocess.run(
        [
            PWSH,
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script("moss_image", "14", "cache")),
            "-python_executable",
            str(tmp_path / "missing-python"),
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )
    assert result.returncode != 0


@pytest.mark.parametrize("architecture,number", ARCHES)
def test_templates_keep_architecture_caches_separate_and_sample_lines_usable(architecture, number):
    path = ROOT / "toml" / f"qinglong-{architecture.replace('_', '-')}-datasets.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["general"]["resolution"] == 1024 and data["general"]["batch_size"] == 1
    assert data["datasets"][0]["cache_directory"] == f"./train/cache_{architecture}"
    assert "control_directory" not in data["datasets"][0]
    prompts = (ROOT / "toml" / f"qinglong_{architecture}.txt").read_text(encoding="utf-8")
    assert any(line.strip() and not line.startswith("#") for line in prompts.splitlines())
