# DLSS-NR Adapter Verification

Date: 2026-10-01 (Asia/Taipei).
Backend: parent repository pins published `musubi-tuner` commit `97f6f85`, including DLSS-NR CLI/bucketing change `9585e7f`.
The local checkout is `d4d1251`, whose only additional change is an unrelated design document.
Its source and tests match the published pin; that local document commit is not included or pushed.
No actual model training, model download, or native DLL validation was performed.

## Results

- `python -m pytest gui/tests/test_dlssnr_command_builder.py gui/tests/test_dlssnr_gui.py gui/tests/test_dlssnr_scripts.py -q`: **60 passed**.
- `python -m pytest gui/tests -q --tb=line`: **487 passed, 5 failed**. The same five failures existed before these changes (baseline: 427 passed, 5 failed).
- `python -m compileall -q gui`: passed.
- Ruff on the five new Python files: passed. Undefined-name/syntax checks on changed production Python files: passed.
- `git diff --check`: passed; Git reported only local LF/CRLF conversion notices.
- Independent read-only review: no remaining blocking findings. The reported PowerShell enum-casing issue was reproduced and fixed with three regression cases.
- Pre-commit review also found lossy float conversion of large integer inputs. Two failing regression cases were reproduced, then passed after exact decimal validation replaced the float conversion.
- Playwright: exercised the actual NiceGUI training/generation pages at desktop 1440x1000 and mobile 390x844, including temporal mode and PNG sequence selection. No horizontal overflow at either width; the inspected fields remained inside the viewport. Browser console had no errors or warnings; GUI server stderr was empty.
- The test browser is closed. Temporary `.playwright-cli/` snapshots remain local because environment policy blocked cleanup; they are not source changes.

The tests use the real DLSS-NR argument parser and effective configuration builder.
PowerShell workflows are executed to capture their actual argv; a real failing
native command checks preservation of exit code 7. GPU/model execution is replaced
only at the external inference boundary. No validation report is manufactured.

## Existing GUI Failures

These tests were left unchanged because they concern existing H3 sample settings
or assertions about a different parent-tree submodule commit:

```text
gui/tests/test_dataset_page_refactor.py::TestDatasetPageRefactor::test_minimax_h3_image_examples_match_one_frame_contract
gui/tests/test_minimax_h3_command_builder.py::TestMiniMaxH3CommandBuilder::test_h3_default_contract_matches_indexed_upstream_parsers
gui/tests/test_minimax_h3_command_builder.py::TestMiniMaxH3CommandBuilder::test_h3_specific_upstream_flags_are_mapped_or_explicitly_deferred
gui/tests/test_minimax_h3_command_builder.py::TestMiniMaxH3CommandBuilder::test_h3_submodule_head_matches_parent_gitlink_and_is_clean
gui/tests/test_minimax_h3_command_builder.py::TestMiniMaxH3CommandBuilder::test_parent_tree_pins_final_h3_best_of_k_commit
```

The image example uses 28 steps while its test expects 30. The other tests read
the gitlink in the root repository's committed `HEAD` tree or require the exact
unrelated H3 commit `9a087963...`. The original tree pinned `49dda50...`; this change advances it
to the published DLSS-NR backend `97f6f85...`, which leaves the existing H3 parser
mismatches and exact-commit assertions unresolved. Updating unrelated H3 samples
or assertions is outside this adapter change. The local submodule checkout is
left untouched rather than including its unpublished documentation commit.

## Root-Wide Collection

`python -m pytest -q --tb=short` was also attempted. It collects auxiliary
checkouts and vendored libraries, not just this project's GUI tests, and stopped
with **25 collection errors** (duplicate test module names and unavailable
auxiliary packages). This is not reported as a passing root-wide test run.

```text
musubi-tuner-dopsd-zimage/tests/test_dopsd_train_utils.py
musubi-tuner-minimax-h3/tests/test_grad_metrics.py
musubi-tuner-minimax-h3/tests/test_ideogram4_autoencoder.py
musubi-tuner-minimax-h3/tests/test_ideogram4_fp8_loading.py
musubi-tuner-minimax-h3/tests/test_ideogram4_lora_sampling.py
musubi-tuner-minimax-h3/tests/test_ideogram4_synthetic.py
musubi-tuner-minimax-h3/tests/test_ideogram4_te_fp8_loading.py
musubi-tuner-minimax-h3/tests/test_ideogram4_timesteps.py
musubi-tuner-minimax-h3/tests/test_krea2_gather_valid_text.py
musubi-tuner-minimax-h3/tests/test_krea2_timesteps.py
musubi-tuner-minimax-h3/tests/test_lora_dtype_bridging.py
musubi-tuner-minimax-h3/tests/test_sai_model_spec.py
musubi-tuner-minimax-h3/tests/test_save_precision.py
musubi-tuner-minimax-h3/tests/test_top_level_entrypoints.py
qinglong-captions/tests
video_controlnet_aux/src/custom_controlnet_aux/leres/pix2pix/options/test_options.py
video_controlnet_aux/src/custom_controlnet_aux/metric3d/mono/configs/HourglassDecoder/test_kitti_convlarge.0.3_150.py
video_controlnet_aux/src/custom_controlnet_aux/metric3d/mono/configs/HourglassDecoder/test_nyu_convlarge.0.3_150.py
video_controlnet_aux/src/custom_controlnet_aux/metric3d/mono/tools/test_scale_cano.py
video_controlnet_aux/src/custom_controlnet_aux/metric3d/mono/utils/do_test.py
video_controlnet_aux/src/custom_controlnet_aux/tests/test_processor.py
video_controlnet_aux/src/custom_controlnet_aux/tests/test_processor_pytest.py
video_controlnet_aux/src/custom_detectron2/modeling/test_time_augmentation.py
video_controlnet_aux/src/custom_mmpkg/custom_mmseg/datasets/pipelines/test_time_aug.py
video_controlnet_aux/src/custom_timm/models/layers/test_time_pool.py
```

## Acceptance Boundary

Interface/argv/configuration compatibility is verified, not image quality or DLL
equivalence. Normal training retains the backend's authentic forward-validation
gate; experiments still require an explicit `development_smoke` choice. LoRA
inference requires merging the adapter with the existing backend merge tool.
