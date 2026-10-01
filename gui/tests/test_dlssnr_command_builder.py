import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "musubi-tuner" / "src"))

from musubi_tuner.dlssnr.config import build_train_config  # noqa: E402
from musubi_tuner.training.dlssnr_parser import setup_parser  # noqa: E402
from utils import model_catalog  # noqa: E402
from utils.command_builder import CommandBuildError, build_cache_jobs, build_generate_job, build_train_job  # noqa: E402


@pytest.fixture
def project(tmp_path):
    dataset = tmp_path / "paired data" / "dataset.toml"
    dataset.parent.mkdir()
    dataset.write_text(
        "[general]\nresolution = [512, 512]\nbatch_size = 2\nenable_bucket = true\nbucket_no_upscale = true\n"
        '\n[[datasets]]\ntrain_manifest = "train pairs.jsonl"\nvalidation_manifest = "validation.jsonl"\n',
        encoding="utf-8",
    )
    return tmp_path, {
        "arch": "DLSS-NR",
        "train_mode": "lora",
        "nr_dataset_config": "paired data/dataset.toml",
        "nr_model_dir": "canonical model",
    }


def effective(job, *, lora=True):
    parsed = setup_parser(lora=lora).parse_args(job.args)
    return parsed, build_train_config(parsed, lora=lora)


def test_lora_job_uses_dataset_only_interface_without_exporting_diffusion_data(project):
    root, state = project
    unrelated = root / "dataset_config.toml"
    unrelated.write_text("# user-owned diffusion dataset\n", encoding="utf-8")
    job = build_train_job(
        {
            **state,
            "mixed_precision": "bf16",
            "fp8_base": True,
            "gradient_checkpointing": True,
            "network_module": "lycoris.kohya",
            "learning_rate": 0.5,
            "optimizer_type": "Prodigy",
            "multi_gpu": True,
            "enable_sample": True,
        },
        root,
        {},
    )
    assert job.script_key == "musubi_tuner.dlssnr_train_network"
    assert job.runner_kwargs["use_accelerate"] is False
    assert job.runner_kwargs["env_vars"]["ACCELERATE_MIXED_PRECISION"] == "no"
    parsed, config = effective(job)
    assert parsed.dataset_config == root / "paired data" / "dataset.toml"
    assert parsed.model_dir == root / "canonical model"
    assert parsed.mixed_precision == "no"
    assert parsed.optimizer_type == "AdamW"
    assert parsed.learning_rate == 1e-4
    assert config["training"]["batch_size"] == 2
    assert config["data"]["enable_bucket"] is True
    assert config["data"]["train_manifest"] == str(root / "paired data" / "train pairs.jsonl")
    assert config["lora"]["rank"] == config["lora"]["alpha"] == 16
    assert unrelated.read_text(encoding="utf-8") == "# user-owned diffusion dataset\n"


def test_full_job_uses_full_entrypoint_and_its_learning_rate(project):
    root, state = project
    job = build_train_job({**state, "train_mode": "finetune", "nr_prior_lr_multiplier": 0.2, "nr_network_dim": 32}, root, {})
    assert job.script_key == "musubi_tuner.dlssnr_train"
    parsed, config = effective(job, lora=False)
    assert parsed.learning_rate == 1e-5
    assert config["parameter_groups"]["prior_lr_multiplier"] == 0.2
    assert "lora" not in config


def test_temporal_training_passes_lengths_losses_and_optimizer_tokens(project):
    root, state = project
    job = build_train_job(
        {
            **state,
            "nr_training_mode": "temporal",
            "nr_sequence_length": 4,
            "nr_burn_in": 2,
            "nr_tbptt_length": 2,
            "nr_loss_temporal": 0.1,
            "nr_optimizer_args": "weight_decay=0.01\nbetas=(0.8, 0.99)",
            "nr_gradient_accumulation_steps": 3,
            "nr_max_train_steps": 12,
        },
        root,
        {},
    )
    parsed, config = effective(job)
    assert (parsed.sequence_length, parsed.burn_in, parsed.tbptt_length) == (4, 2, 2)
    assert config["training"]["gradient_accumulation_steps"] == 3
    assert config["training"]["max_train_steps"] == 12
    assert config["loss"]["temporal"] == 0.1
    assert "betas=(0.8, 0.99)" in config["optimizer"]["args"]


def test_smoke_mode_is_never_implicitly_enabled(project):
    root, state = project
    job = build_train_job(state, root, {})
    parsed, _ = effective(job)
    assert parsed.development_smoke is False
    with pytest.raises(CommandBuildError, match="model_dir"):
        build_train_job({**state, "nr_model_dir": ""}, root, {})
    with pytest.raises(CommandBuildError, match="model_dir"):
        build_train_job({**state, "nr_model_dir": "", "nr_development_smoke": True}, root, {})


def test_resume_and_negative_baseline_switch(project):
    root, state = project
    job = build_train_job(
        {
            **state,
            "nr_resume": "output/run/state-step000100",
            "nr_save_state": True,
            "nr_compare_baseline": False,
            "nr_sample_every_n_steps": 5,
        },
        root,
        {},
    )
    parsed, config = effective(job)
    assert parsed.resume == root / "output/run/state-step000100"
    assert parsed.forward_validation_report is None
    assert parsed.save_state is True
    assert config["evaluation"]["compare_baseline"] is False
    assert config["evaluation"]["sample_every_n_steps"] == 5


def test_blank_alpha_follows_custom_vit_rank(project):
    root, state = project
    job = build_train_job({**state, "nr_network_dim": 8, "nr_network_alpha": ""}, root, {})
    _, config = effective(job)
    assert config["lora"]["rank"] == config["lora"]["alpha"] == 8


def test_large_text_seed_is_preserved_exactly(project):
    root, state = project
    job = build_train_job({**state, "nr_seed": " 9007199254740993 "}, root, {})
    parsed, _ = effective(job)
    assert parsed.seed == 9007199254740993


def test_large_fractional_step_count_is_not_rounded_to_an_integer(project):
    root, state = project
    with pytest.raises(CommandBuildError, match="max_train_steps"):
        build_train_job({**state, "nr_max_train_steps": "9007199254740992.5"}, root, {})


@pytest.mark.parametrize(
    "name, expected, token",
    [
        ("AdamW_adv", "adv_optm.AdamW_adv", "betas=(0.95, 0.98)"),
        ("Lion", "pytorch_optimizer.Lion", "cautious=True"),
        ("SOAP", "pytorch_optimizer.SOAP", None),
        ("adafactor", "Adafactor", "relative_step=False"),
        ("PagedAdamW8bit", "bitsandbytes.optim.PagedAdamW8bit", None),
        ("Lion8bit", "bitsandbytes.optim.Lion8bit", "weight_decay=0.01"),
        ("AdEMAMix8bit", "bitsandbytes.optim.AdEMAMix8bit", "weight_decay=0.01"),
        ("DAdaptAdam", "pytorch_optimizer.DAdaptAdam", None),
        ("adamg", "pytorch_optimizer.AdamG", "weight_decay=0.1"),
        ("optimi.AdamW", "optimi.AdamW", None),
    ],
)
def test_nr_optimizer_aliases_use_shared_templates(project, name, expected, token):
    root, state = project
    job = build_train_job({**state, "nr_optimizer_type": name}, root, {})
    parsed, config = effective(job)
    assert parsed.optimizer_type == expected
    if token:
        assert token in config["optimizer"]["args"]


def test_nr_custom_optimizer_args_replace_template_and_keep_explicit_rate(project):
    root, state = project
    job = build_train_job(
        {**state, "nr_optimizer_type": "Prodigy_adv", "nr_optimizer_args": "d_coef=0.7", "nr_learning_rate": 0.2}, root, {}
    )
    parsed, config = effective(job)
    assert parsed.optimizer_type == "adv_optm.Prodigy_adv"
    assert parsed.learning_rate == 0.2
    assert config["optimizer"]["args"] == ["d_coef=0.7"]


def test_optimizer_arguments_keep_grouped_literals_and_ignore_comments(project):
    root, state = project
    _, config = effective(
        build_train_job(
            {
                **state,
                "nr_optimizer_args": "betas=(0.8, 0.99) weight_decay=0.01 # custom\n",
            },
            root,
            {},
        )
    )
    assert config["optimizer"]["args"] == ["betas=(0.8, 0.99)", "weight_decay=0.01"]


@pytest.mark.parametrize(
    "name",
    [
        "LBFGS",
        "torch.optim.LBFGS",
        "AdamWScheduleFree",
        "schedulefree.SGDScheduleFree",
        "adammini",
        "Sophia",
        "pytorch_optimizer.SophiaH",
    ],
)
def test_nr_rejects_known_incompatible_optimizer_protocols(project, name):
    root, state = project
    with pytest.raises(CommandBuildError, match="optimizer"):
        build_train_job({**state, "nr_optimizer_type": name}, root, {})


def test_removed_gui_gate_options_cannot_turn_on_random_initialization(project):
    root, state = project
    legacy = {
        **state,
        "nr_forward_validation_report": "missing.json",
        "nr_development_smoke": True,
        "nr_deployment_target": "native_roundtrip",
    }
    parsed, _ = effective(build_train_job(legacy, root, {}))
    assert parsed.forward_validation_report is None
    assert parsed.development_smoke is False
    assert parsed.deployment_target == "float_runtime"
    with pytest.raises(CommandBuildError, match="model_dir"):
        build_train_job({**legacy, "nr_model_dir": ""}, root, {})


@pytest.mark.parametrize("optimizer", ["LoRARite", "pytorch_optimizer.LoRARite"])
def test_lorarite_cannot_treat_full_model_weights_as_lora_pairs(project, optimizer):
    root, state = project
    with pytest.raises(CommandBuildError, match="optimizer"):
        build_train_job({**state, "train_mode": "finetune", "nr_optimizer_type": optimizer}, root, {})


@pytest.mark.parametrize("name", ["AdamW_adv", "Simplified_AdEMAMix", "Lion", "SOAP", "SGD"])
def test_optimizer_selection_constructs_a_real_shared_backend_optimizer(project, name):
    import torch
    from musubi_tuner.training.dlssnr_services import create_nr_optimizer

    pytest.importorskip("adv_optm" if name.endswith("_adv") or name == "Simplified_AdEMAMix" else "pytorch_optimizer")
    root, state = project
    parsed, config = effective(build_train_job({**state, "nr_optimizer_type": name}, root, {}))
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = create_nr_optimizer([parameter], config["optimizer"])
    assert optimizer.param_groups[0]["params"][0] is parameter
    assert optimizer.param_groups[0]["lr"] == parsed.learning_rate


def test_multiscale_profile_omits_scalar_rank_and_preserves_width_tables(project):
    root, state = project
    ranks = '{"32": 2, "64": 4, "128": 8, "256": 8, "512": 16, "1024": 32}'
    job = build_train_job(
        {**state, "nr_lora_profile": "multiscale", "nr_network_dim": 16, "nr_network_alpha": 16, "nr_rank_by_width": ranks},
        root,
        {},
    )
    parsed, config = effective(job)
    assert parsed.network_dim is None and parsed.network_alpha is None
    assert config["lora"]["profile"] == "multiscale"
    assert config["lora"]["rank_by_width"]["1024"] == 32


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"nr_training_mode": "temporal", "nr_sequence_length": 4, "nr_burn_in": 1, "nr_tbptt_length": 2}, "sequence_length"),
        ({"nr_loss_temporal": 0.1}, "single-frame"),
        (
            {
                "nr_training_mode": "temporal",
                "nr_sequence_length": 2,
                "nr_burn_in": 1,
                "nr_tbptt_length": 1,
                "nr_loss_temporal": 0.1,
            },
            "two supervised",
        ),
        ({"nr_network_dim": 0}, "rank"),
        ({"nr_network_dropout": 1.0}, "dropout"),
        ({"nr_max_train_steps": 1.5}, "max_train_steps"),
        ({"nr_learning_rate": "nan"}, "learning_rate"),
        ({"nr_optimizer_args": "--mixed_precision=bf16"}, "optimizer_args"),
        ({"nr_optimizer_args": "lr=0.1"}, "learning_rate"),
        ({"nr_optimizer_args": "weight_decay=0.0\nweight_decay=0.1"}, "unique"),
        ({"nr_lora_profile": "multiscale", "nr_rank_by_width": '{"32": 2}'}, "widths"),
        ({"nr_mixed_precision": "bf16"}, "FP32"),
        ({"nr_lr_scheduler": "cosine"}, "constant"),
        ({"nr_output_name": "../escape"}, "filename"),
        ({"train_mode": "unknown"}, "train_mode"),
    ],
)
def test_invalid_training_state_is_rejected_before_launch(project, overrides, message):
    root, state = project
    with pytest.raises(CommandBuildError, match=message):
        build_train_job({**state, **overrides}, root, {})


def test_legacy_all_in_one_dataset_is_rejected_without_changing_it(project):
    root, state = project
    dataset = root / state["nr_dataset_config"]
    original = dataset.read_text(encoding="utf-8") + "\n[training]\nmax_train_steps = 1\n"
    dataset.write_text(original, encoding="utf-8")
    with pytest.raises(CommandBuildError, match="command-line"):
        build_train_job(state, root, {})
    assert dataset.read_text(encoding="utf-8") == original


def test_eval_interval_requires_a_validation_manifest(project):
    root, state = project
    dataset = root / state["nr_dataset_config"]
    dataset.write_text(
        dataset.read_text(encoding="utf-8").replace('validation_manifest = "validation.jsonl"\n', ""), encoding="utf-8"
    )
    with pytest.raises(CommandBuildError, match="validation_manifest"):
        build_train_job({**state, "nr_sample_every_n_steps": 1}, root, {})


def test_dataset_config_is_required_and_cannot_fall_back_to_diffusion_export(project):
    root, state = project
    with pytest.raises(CommandBuildError, match="dataset_config"):
        build_train_job({**state, "nr_dataset_config": "missing.toml"}, root, {})
    assert not (root / "dataset_config.toml").exists()


def test_dlssnr_is_excluded_from_cache_selector_and_rejected_before_export(project):
    root, _ = project
    assert "DLSS-NR" in model_catalog.get_architecture_names(page_key="train")
    assert "DLSS-NR" in model_catalog.get_architecture_names(page_key="generate")
    assert "DLSS-NR" not in model_catalog.get_architecture_names(page_key="cache")
    with pytest.raises(CommandBuildError, match="cache"):
        build_cache_jobs({"arch": "DLSS-NR"}, root, {})
    assert not (root / "dataset_config.toml").exists()


@pytest.mark.parametrize(
    "mode, module, flag",
    [
        ("image", "dlssnr_generate_image", "--sample_manifest"),
        ("sequence", "dlssnr_generate_video", "--sequence_manifest"),
    ],
)
def test_inference_uses_manifest_not_prompts_and_passes_real_cli(mode, module, flag, tmp_path):
    state = {
        "arch": "DLSS-NR",
        "nr_inference_mode": mode,
        "nr_model_dir": "models/canonical",
        "nr_sample_manifest": "single rows.jsonl",
        "nr_sequence_manifest": "sequence rows.jsonl",
        "nr_bucket_width": 512,
        "nr_bucket_height": 512,
        "nr_output_dir": "previews",
        "nr_seed": 7,
        "prompt": "This diffusion setting must not be passed",
        "lora_weight": "adapter.safetensors",
        "infer_steps": 20,
        "attn_mode": "sageattn",
        "save_path": "wrong.mp4",
    }
    job = build_generate_job(state, tmp_path)
    assert job.script_key == f"musubi_tuner.{module}"
    assert job.runner_kwargs["use_accelerate"] is False
    assert any(arg.startswith(f"{flag}=") for arg in job.args)
    entry = importlib.import_module(f"musubi_tuner.{module}")
    worker = "generate_stills" if mode == "image" else "generate_sequence"
    with patch.object(entry, "load_model", return_value="loaded"), patch.object(entry, worker, return_value=[]) as infer:
        with patch.object(sys, "argv", [module, *job.args]):
            entry.main()
    manifest = "single rows.jsonl" if mode == "image" else "sequence rows.jsonl"
    infer.assert_called_once_with("loaded", str(tmp_path / manifest), 512, 512, str(tmp_path / "previews"), 7)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"nr_model_dir": ""}, "model_dir"),
        ({"nr_sample_manifest": ""}, "sample_manifest"),
        ({"nr_inference_mode": "sequence", "nr_sequence_manifest": ""}, "sequence_manifest"),
        ({"nr_inference_mode": "video"}, "inference_mode"),
        ({"nr_bucket_width": 32}, "size"),
        ({"nr_bucket_height": 511.5}, "bucket_height"),
        ({"nr_device": "mps"}, "device"),
    ],
)
def test_invalid_inference_state_is_rejected(tmp_path, overrides, message):
    state = {"arch": "DLSS-NR", "nr_model_dir": "canonical", "nr_sample_manifest": "single.jsonl"}
    with pytest.raises(CommandBuildError, match=message):
        build_generate_job({**state, **overrides}, tmp_path)
