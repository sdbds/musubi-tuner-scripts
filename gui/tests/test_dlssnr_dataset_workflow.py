import copy
import sys
from pathlib import Path

import pytest
from nicegui import ui

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "musubi-tuner/src"))

from musubi_tuner.dlssnr.config import load_dataset_config  # noqa: E402
from utils.dataset_config import export_dataset_config, load_dataset_config_import  # noqa: E402
from wizard.step1_tagging import DatasetStep  # noqa: E402


def dataset_step(project):
    step = DatasetStep.__new__(DatasetStep)
    step.project_config = project
    step.dataset_row_states = []
    step.dataset_row_controls = []
    step.dataset_rows_container = None
    step._refresh_dataset_row_states()
    return step


def test_nr_dataset_editor_roundtrips_directories_and_fixed_conditions(tmp_path):
    project = {
        "dataset": {
            "general": {"resolution": [64, 48]},
            "datasets": [
                {
                    "image_directory": "targets",
                    "control_directory": "inputs",
                    "nr_controls_mode": "fixed",
                    "nr_style": 2,
                    "nr_tone": 0.5,
                    "nr_structure": 0.25,
                    "nr_skin": -1.0,
                    "nr_auto_mask": True,
                }
            ],
        }
    }
    step = dataset_step(project)
    assert step.dataset_row_states[0]["dataset_template"] == "dlssnr"
    with ui.column() as container:
        step.dataset_rows_container = ui.column()
        step._render_dataset_rows()
    try:
        controls = step.dataset_row_controls[0]
        assert {
            "image_directory",
            "control_directory",
            "nr_style",
            "nr_tone",
            "nr_structure",
            "nr_skin",
            "nr_auto_mask",
        } <= controls.keys()
        assert "cache_directory" not in controls
        controls["nr_auto_mask"].set_toggle_value(False)
        datasets, extras, templates = step._collect_dataset_rows()
        assert templates == ["dlssnr"]
        project["dataset"]["datasets"] = datasets
        project["interop"] = {"dataset_extra": {"datasets": extras}}
        path = export_dataset_config(project, tmp_path / "dataset.toml")
        imported = load_dataset_config_import(path)
        assert imported["dataset"]["datasets"][0]["nr_auto_mask"] is False
        config = load_dataset_config(path)
        assert config["control_directory"] == str(tmp_path / "inputs")
        assert config["fixed_controls"]["nr_style"] == 2
        assert config["fixed_controls"]["nr_auto_mask"] is False
        assert config["fixed_controls"]["nr_structure"] == 0.25
    finally:
        container.delete()


def test_legacy_nr_manifest_is_editable_and_not_misclassified_as_text_to_image():
    step = dataset_step({"dataset": {"general": {}, "datasets": [{"train_manifest": "train.jsonl"}]}})
    state = step.dataset_row_states[0]
    assert state["dataset_template"] == "dlssnr"
    assert state["dataset_source"] == "jsonl"
    assert state["nr_controls_mode"] == "files"
    assert step._collect_dataset_rows()[0] == [{"train_manifest": "train.jsonl", "nr_controls_mode": "files"}]


def test_imported_nr_paths_keep_their_base_when_exported_elsewhere(tmp_path):
    folder = tmp_path / "presets"
    folder.mkdir()
    original = folder / "nr.toml"
    original.write_text(
        '[[datasets]]\nimage_directory="targets"\ncontrol_directory="inputs"\n\nnr_controls_mode="fixed"\nnr_auto_mask=false\n',
        encoding="utf-8",
    )
    imported = load_dataset_config_import(original)
    path = export_dataset_config(imported, tmp_path / "dataset_config.toml")
    parsed = load_dataset_config(path)
    assert parsed["image_directory"] == str(folder / "targets")
    assert parsed["control_directory"] == str(folder / "inputs")
    assert parsed["fixed_controls"]["nr_auto_mask"] is False


def test_synthetic_temporal_dataset_controls_roundtrip(tmp_path):
    project = {
        "dataset": {
            "general": {"resolution": [64, 48]},
            "datasets": [{"image_directory": "targets", "control_directory": "inputs", "nr_controls_mode": "fixed"}],
        }
    }
    step = dataset_step(project)
    with ui.column() as container:
        step.dataset_rows_container = ui.column()
        step._render_dataset_rows()
    try:
        controls = step.dataset_row_controls[0]
        assert {"synthetic_temporal", "synthetic_max_shift_px"} <= controls.keys()
        controls["synthetic_temporal"].set_toggle_value(True)
        step.dataset_row_controls[0]["synthetic_max_shift_px"].set_value("0.75")
        project["dataset"]["datasets"] = step._collect_dataset_rows()[0]
        path = export_dataset_config(project, tmp_path / "dataset.toml")
        data = load_dataset_config(path)
        assert data["synthetic_temporal"] == {"schema": "dlssnr_synthetic_temporal_v1", "max_shift_px": 0.75}
        assert "loss_temporal" not in project["dataset"]["datasets"][0]
        step.dataset_row_controls[0]["synthetic_temporal"].set_toggle_value(False)
        disabled = step._collect_dataset_rows()[0][0]
        assert not disabled.get("synthetic_temporal") and "synthetic_max_shift_px" not in disabled
    finally:
        container.delete()


def test_synthetic_general_defaults_are_materialized_as_editable_dataset_values(tmp_path):
    path = tmp_path / "nr.toml"
    path.write_text(
        "[general]\nsynthetic_temporal=true\nsynthetic_max_shift_px=0.75\n"
        '[[datasets]]\ntrain_manifest="one.jsonl"\n'
        '[[datasets]]\ntrain_manifest="two.jsonl"\nsynthetic_temporal=false\n',
        encoding="utf-8",
    )
    original = load_dataset_config(path)
    imported = load_dataset_config_import(path)
    assert "synthetic_temporal" not in imported["dataset"]["general"]
    assert "synthetic_max_shift_px" not in imported["interop"]["dataset_extra"]["general"]
    step = dataset_step(imported)
    assert step.dataset_row_states[0]["synthetic_temporal"] is True
    assert step.dataset_row_states[0]["synthetic_max_shift_px"] == 0.75
    assert step.dataset_row_states[1]["synthetic_temporal"] is False
    imported["dataset"]["datasets"] = step._collect_dataset_rows()[0]
    exported = export_dataset_config(imported, tmp_path / "export.toml")
    assert load_dataset_config(exported) == original


@pytest.mark.parametrize("legacy_extra", [False, True])
def test_saved_project_inherited_synthesis_survives_read_and_can_be_disabled(tmp_path, monkeypatch, legacy_extra):
    inherited = {"synthetic_temporal": True, "synthetic_max_shift_px": 0.75}
    project = {
        "dataset": {"general": {}, "datasets": [{"train_manifest": "one.jsonl"}]},
        "interop": {"dataset_extra": {"general": {}}},
    }
    general = project["interop"]["dataset_extra"]["general"] if legacy_extra else project["dataset"]["general"]
    general.update(inherited)
    original = copy.deepcopy(project)
    step = dataset_step(project)
    assert step.dataset_row_states[0]["synthetic_temporal"] is True
    assert step.dataset_row_states[0]["synthetic_max_shift_px"] == 0.75
    assert general == inherited
    step.dataset_row_states[0]["synthetic_temporal"] = False
    step.project_dir = tmp_path
    monkeypatch.setattr(step, "_collect_general_state", lambda: {})
    monkeypatch.setattr("wizard.step1_tagging.config_manager.load_project_config", lambda _: copy.deepcopy(original))
    step._persist_dataset_to_project_config()
    assert "synthetic_temporal" not in step.project_config["interop"]["dataset_extra"]["general"]
    exported = export_dataset_config(step.project_config, tmp_path / "dataset.toml")
    assert load_dataset_config(exported).get("synthetic_temporal") is None


@pytest.mark.parametrize("value", ["nan", "-0.1", "1.1", True])
def test_synthetic_temporal_rejects_invalid_shift(value):
    step = dataset_step({"dataset": {"datasets": [{"train_manifest": "one.jsonl", "synthetic_temporal": True}]}})
    step.dataset_row_states[0]["synthetic_max_shift_px"] = value
    with pytest.raises(ValueError, match="synthetic_max_shift_px"):
        step._collect_dataset_rows()


def test_switching_image_edit_to_nr_does_not_export_hidden_multiple_targets():
    step = dataset_step(
        {
            "dataset": {
                "general": {},
                "datasets": [
                    {
                        "image_directory": "targets",
                        "control_directory": "inputs",
                        "multiple_target": True,
                    }
                ],
            }
        }
    )
    step._set_dataset_row_mode(0, "dataset_template", "dlssnr")
    dataset = step._collect_dataset_rows()[0][0]
    assert "multiple_target" not in dataset


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch_size", "1.5"),
        ("num_repeats", "0"),
        ("resolution_w", "bad"),
        ("nr_tone", True),
        ("nr_structure", False),
        ("nr_skin", True),
    ],
)
def test_nr_dataset_numeric_edits_are_not_silently_dropped(key, value):
    step = dataset_step(
        {
            "dataset": {
                "general": {},
                "datasets": [
                    {
                        "image_directory": "targets",
                        "control_directory": "inputs",
                        "nr_controls_mode": "fixed",
                    }
                ],
            }
        }
    )
    step.dataset_row_states[0][key] = value
    with pytest.raises(ValueError, match=key):
        step._collect_dataset_rows()
