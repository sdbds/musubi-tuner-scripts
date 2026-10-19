import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from nicegui import ui

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "musubi-tuner" / "src"))

from test_dlssnr_scripts import PWSH, capture_script  # noqa: E402
from utils import dlssnr_commands, script_coverage_manifest  # noqa: E402
from utils.command_builder import CommandBuildError  # noqa: E402

UNPACK_SCRIPT = "4.12dlssnr_unpack.ps1"
PACK_SCRIPT = "4.12.1dlssnr_pack.ps1"


@pytest.fixture
def inputs(tmp_path):
    (tmp_path / "original DLL.dll").write_bytes(b"test fixture, not a real DLL")
    (tmp_path / "model weights").mkdir()
    (tmp_path / "adapter weights.safetensors").write_bytes(b"test fixture")
    return {
        "nr_source_dll": "original DLL.dll",
        "nr_template_dll": "original DLL.dll",
        "nr_model_dir": "model weights",
        "nr_output_dir": "new weights",
        "nr_output_dll": "export/new DLL.dll",
    }


def test_unpack_wrapper_forwards_paths_and_normalizes_choices():
    result, argv = capture_script(
        UNPACK_SCRIPT,
        "-source_dll 'original DLL.dll' -output_dir 'new weights' -weights cc -cc_format CANONICAL -save_precision FP32",
    )
    assert result.returncode == 0 and argv is not None, result.stderr
    assert Path(argv[0]).name == "dlssnr_unpack_model.py"
    assert argv[1:] == [
        "--source_dll=original DLL.dll",
        "--output_dir=new weights",
        "--weights=CC",
        "--cc_format=canonical",
        "--save_precision=fp32",
    ]
    assert "NR_PRECISION=bf16" in result.stdout


def test_unpack_wrapper_defaults_to_trainable_canonical_weights():
    result, argv = capture_script(UNPACK_SCRIPT)
    assert result.returncode == 0 and argv is not None, result.stderr
    assert "--weights=all" in argv
    assert "--cc_format=canonical" in argv
    assert "--save_precision=native" in argv


def test_pack_wrapper_forwards_optional_lora_and_invariant_numbers():
    result, argv = capture_script(
        PACK_SCRIPT,
        "-template_dll 'original DLL.dll' -model_dir 'model weights' -output_dll 'new DLL.dll' "
        "-weights cc -mix 0.75 -strength 0.9 -merge_lora 'adapter weights.safetensors' -lora_multiplier -0.5",
        culture="fr-FR",
    )
    assert result.returncode == 0 and argv is not None, result.stderr
    assert Path(argv[0]).name == "dlssnr_pack_model.py"
    assert set(argv[1:]) == {
        "--template_dll=original DLL.dll",
        "--model_dir=model weights",
        "--output_dll=new DLL.dll",
        "--weights=CC",
        "--mix=0.75",
        "--strength=0.9",
        "--merge_lora=adapter weights.safetensors",
        "--lora_multiplier=-0.5",
    }
    from musubi_tuner.dlssnr_pack_model import setup_parser

    parsed = setup_parser().parse_args(argv[1:])
    assert parsed.mix == 0.75 and parsed.lora_multiplier == -0.5


def test_pack_wrapper_omits_automatic_resource_and_empty_adapter():
    result, argv = capture_script(PACK_SCRIPT)
    assert result.returncode == 0 and argv is not None, result.stderr
    assert not any(arg.startswith(("--weights=", "--merge_lora=")) for arg in argv)


@pytest.mark.parametrize(
    "script,parameters",
    [
        (UNPACK_SCRIPT, "-source_dll ' '"),
        (UNPACK_SCRIPT, "-output_dir ''"),
        (PACK_SCRIPT, "-template_dll ''"),
        (PACK_SCRIPT, "-model_dir ' '"),
        (PACK_SCRIPT, "-output_dll ''"),
    ],
)
def test_dll_wrappers_reject_blank_required_paths(script, parameters):
    result, argv = capture_script(script, parameters)
    assert result.returncode != 0 and argv is None
    assert "required" in result.stderr


@pytest.mark.parametrize("script", [UNPACK_SCRIPT, PACK_SCRIPT])
def test_dll_wrappers_propagate_python_failure(script, tmp_path):
    if not PWSH:
        pytest.skip("PowerShell 7 is unavailable")
    executable = tmp_path / ("failing-python.cmd" if sys.platform == "win32" else "failing-python")
    executable.write_text("@exit /b 7\n" if sys.platform == "win32" else "#!/bin/sh\nexit 7\n", encoding="utf-8")
    executable.chmod(0o755)
    result = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-File", str(ROOT / script), "-python_executable", str(executable)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 7
    assert "exit code: 7" in result.stderr


def test_unpack_job_resolves_paths_and_does_not_require_training_inputs(inputs, tmp_path):
    job = dlssnr_commands.build_dlssnr_unpack_job(inputs, tmp_path)
    assert job.script_key == "musubi_tuner.dlssnr_unpack_model"
    assert job.runner_kwargs == {"use_accelerate": False}
    assert set(job.args) == {
        f"--source_dll={tmp_path / 'original DLL.dll'}",
        f"--output_dir={tmp_path / 'new weights'}",
        "--weights=all",
        "--cc_format=canonical",
        "--save_precision=native",
    }


def test_pack_job_normalizes_choices_and_keeps_negative_lora_multiplier(inputs, tmp_path):
    job = dlssnr_commands.build_dlssnr_pack_job(
        {
            **inputs,
            "nr_weights": "cc",
            "nr_mix": "0.75",
            "nr_strength": 0.9,
            "nr_merge_lora": "adapter weights.safetensors",
            "nr_lora_multiplier": -0.5,
        },
        tmp_path,
    )
    assert job.script_key == "musubi_tuner.dlssnr_pack_model"
    assert job.runner_kwargs == {"use_accelerate": False}
    assert set(job.args) == {
        f"--template_dll={tmp_path / 'original DLL.dll'}",
        f"--model_dir={tmp_path / 'model weights'}",
        f"--output_dll={tmp_path / 'export/new DLL.dll'}",
        "--weights=CC",
        "--mix=0.75",
        "--strength=0.9",
        f"--merge_lora={tmp_path / 'adapter weights.safetensors'}",
        "--lora_multiplier=-0.5",
    }


def test_pack_job_omits_unset_optional_arguments(inputs, tmp_path):
    job = dlssnr_commands.build_dlssnr_pack_job(inputs, tmp_path)
    assert not any(arg.startswith(("--weights=", "--merge_lora=")) for arg in job.args)


@pytest.mark.parametrize(
    "operation,changes,message",
    [
        ("unpack", {"nr_source_dll": ""}, "source_dll"),
        ("unpack", {"nr_source_dll": "missing.dll"}, "source_dll"),
        ("unpack", {"nr_output_dir": "model weights"}, "output_dir"),
        ("unpack", {"nr_weights": "bad"}, "weights"),
        ("unpack", {"nr_cc_format": "bad"}, "cc_format"),
        ("unpack", {"nr_save_precision": "fp16"}, "save_precision"),
        ("pack", {"nr_template_dll": "model weights"}, "template_dll"),
        ("pack", {"nr_model_dir": "missing"}, "model_dir"),
        ("pack", {"nr_output_dll": "original DLL.dll"}, "output_dll"),
        ("pack", {"nr_merge_lora": "missing.safetensors"}, "merge_lora"),
        ("pack", {"nr_weights": "all"}, "weights"),
        ("pack", {"nr_mix": 1.1}, "mix"),
        ("pack", {"nr_mix": -0.1}, "mix"),
        ("pack", {"nr_strength": float("nan")}, "strength"),
        ("pack", {"nr_lora_multiplier": 0.5}, "merge_lora"),
    ],
)
def test_dll_jobs_reject_invalid_inputs_before_launch(operation, changes, message, inputs, tmp_path):
    builder = getattr(dlssnr_commands, f"build_dlssnr_{operation}_job")
    with pytest.raises(CommandBuildError, match=message):
        builder({**inputs, **changes}, tmp_path)


def test_pack_job_protects_existing_report(inputs, tmp_path):
    report = tmp_path / "export/new DLL.dll.report.json"
    report.parent.mkdir()
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(CommandBuildError, match="report"):
        dlssnr_commands.build_dlssnr_pack_job(inputs, tmp_path)


def test_dll_tools_dispatch_rendered_fields_through_shared_execution(inputs, tmp_path, monkeypatch):
    from components.dlssnr_dll_tools import DLSSNRDLLTools

    with ui.column() as container:
        tools = DLSSNRDLLTools(project_dir=tmp_path)
    submitted = []

    async def capture(**kwargs):
        submitted.append(kwargs)

    monkeypatch.setattr(tools.exec_panel, "run_job", capture)
    try:
        for name in ("nr_source_dll", "nr_output_dir"):
            tools.unpack_controls[name].value = inputs[name]
        tools.unpack_controls["nr_cc_format"].set_value("canonical")
        asyncio.run(tools.run("unpack"))
        assert submitted[-1]["script_key"] == "musubi_tuner.dlssnr_unpack_model"
        assert "--cc_format=canonical" in submitted[-1]["args"]
        for name in ("nr_template_dll", "nr_model_dir", "nr_output_dll"):
            tools.pack_controls[name].value = inputs[name]
        tools.pack_controls["nr_mix"].set_value(0.5)
        asyncio.run(tools.run("pack"))
        assert submitted[-1]["script_key"] == "musubi_tuner.dlssnr_pack_model"
        assert "--mix=0.5" in submitted[-1]["args"]
        assert submitted[-1]["runner_kwargs"] == {"use_accelerate": False}
        tools.exec_panel._set_running(True)
        assert not tools.unpack_button.enabled and not tools.pack_button.enabled
        tools.exec_panel._set_running(False)
        assert tools.unpack_button.enabled and tools.pack_button.enabled
    finally:
        container.delete()


def test_dll_tools_validation_does_not_submit_invalid_job(tmp_path, monkeypatch):
    from components.dlssnr_dll_tools import DLSSNRDLLTools

    with ui.column() as container:
        tools = DLSSNRDLLTools(project_dir=tmp_path)
    notifications = []
    monkeypatch.setattr(ui, "notify", lambda message, **kwargs: notifications.append(message))

    async def unexpected_job(**kwargs):
        pytest.fail("Invalid paths must not launch a job")

    monkeypatch.setattr(tools.exec_panel, "run_job", unexpected_job)
    try:
        tools.unpack_controls["nr_source_dll"].value = ""
        asyncio.run(tools.run("unpack"))
        assert notifications and "source_dll" in notifications[-1]
    finally:
        container.delete()


@pytest.mark.parametrize(
    "name,value",
    [
        ("nr_mix", "0.375"),
        ("nr_strength", "-3.125"),
        ("nr_lora_multiplier", "4.25"),
    ],
)
def test_pack_sliders_forward_precise_pending_edits(name, value, inputs, tmp_path, monkeypatch):
    from components.dlssnr_dll_tools import DLSSNRDLLTools

    with ui.column() as container:
        tools = DLSSNRDLLTools(project_dir=tmp_path)
    submitted = []

    async def capture(**kwargs):
        submitted.append(kwargs)

    monkeypatch.setattr(tools.exec_panel, "run_job", capture)
    monkeypatch.setattr(ui, "run_javascript", Mock())
    try:
        for field in ("nr_template_dll", "nr_model_dir", "nr_output_dll"):
            tools.pack_controls[field].value = inputs[field]
        tools.pack_controls["nr_merge_lora"].value = "adapter weights.safetensors"
        slider = tools.pack_controls[name]
        assert isinstance(slider, ui.slider)
        assert slider.get_bound_value() == 1.0
        elements = list(slider.parent_slot.parent.descendants())
        button = next(element for element in elements if isinstance(element, ui.button))
        editor = next(element for element in elements if isinstance(element, ui.input))
        click = next(listener.handler for listener in button._event_listeners.values() if listener.type == "click")
        click(None)
        editor.set_value(value)
        asyncio.run(tools.run("pack"))
        assert f"--{name.removeprefix('nr_')}={value}" in submitted[-1]["args"]
    finally:
        container.delete()


@pytest.mark.parametrize(
    "name,value",
    [
        ("nr_mix", "1.1"),
        ("nr_mix", "-0.1"),
        ("nr_strength", "nan"),
        ("nr_lora_multiplier", "not a number"),
    ],
)
def test_pack_sliders_do_not_submit_previous_values_after_invalid_edits(name, value, inputs, tmp_path, monkeypatch):
    from components.dlssnr_dll_tools import DLSSNRDLLTools

    with ui.column() as container:
        tools = DLSSNRDLLTools(project_dir=tmp_path)
    notifications = []
    monkeypatch.setattr(ui, "notify", lambda message, **kwargs: notifications.append(message))
    monkeypatch.setattr(ui, "run_javascript", Mock())

    async def unexpected_job(**kwargs):
        pytest.fail("Invalid numeric edits must not launch a job with the previous value")

    monkeypatch.setattr(tools.exec_panel, "run_job", unexpected_job)
    try:
        for field in ("nr_template_dll", "nr_model_dir", "nr_output_dll"):
            tools.pack_controls[field].value = inputs[field]
        slider = tools.pack_controls[name]
        assert isinstance(slider, ui.slider)
        elements = list(slider.parent_slot.parent.descendants())
        button = next(element for element in elements if isinstance(element, ui.button))
        editor = next(element for element in elements if isinstance(element, ui.input))
        click = next(listener.handler for listener in button._event_listeners.values() if listener.type == "click")
        click(None)
        editor.set_value(value)
        asyncio.run(tools.run("pack"))
        assert notifications and name.removeprefix("nr_") in notifications[-1]
    finally:
        container.delete()


def test_unpack_destination_picker_accepts_a_new_directory(inputs, tmp_path, monkeypatch):
    import tkinter
    from tkinter import filedialog

    from components.dlssnr_dll_tools import DLSSNRDLLTools

    destination = str(tmp_path / "new extraction")
    monkeypatch.setattr(tkinter, "Tk", Mock())
    monkeypatch.setattr(filedialog, "askdirectory", lambda: str(tmp_path))
    monkeypatch.setattr(filedialog, "asksaveasfilename", lambda: destination)
    with ui.column() as container:
        tools = DLSSNRDLLTools(project_dir=tmp_path)
    try:
        output = tools.unpack_controls["nr_output_dir"]
        asyncio.run(output._pick_path())
        job = dlssnr_commands.build_dlssnr_unpack_job({**inputs, "nr_output_dir": output.value}, tmp_path)
        assert f"--output_dir={destination}" in job.args
        assert not Path(destination).exists()
    finally:
        container.delete()


def test_dll_scripts_are_classified_as_native_gui_workflows():
    assert {UNPACK_SCRIPT, PACK_SCRIPT} <= set(script_coverage_manifest.NATIVE_GUI)
