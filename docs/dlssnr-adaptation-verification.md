# DLSS-NR Adapter Verification

Date: 2026-10-01 (Asia/Taipei).
Parent rework base: `7abe90e`, previously pinning `musubi-tuner` commit `97f6f85`.
Backend rework: `cf09716`, pushed to `DLSSNR` before merging into `qinglong` as `9c37b52`.
The merged backend passed its full suite before `qinglong` was pushed; the parent gitlink now targets published `9c37b52`.
The existing local documentation commit `d4d1251` is preserved only on `codex/qinglong-local-runtime-design`, with no upstream tracking branch.
It is not an ancestor of either published branch. Unrelated parent working-tree changes are excluded from this commit.
No user-dataset or real-checkpoint NR training, model download, or native DLL validation was performed.

## Rework Scope

- NR now uses the existing top-level training/generation tabs and shared path, slider, toggle, and optimizer controls. Diffusion-only sections are hidden; full training hides the LoRA network section.
- Optimizer choices and class-path aliases share one catalog. NR keeps its backend-specific defaults and supports literal tuple/string arguments without splitting them at spaces. Coefficient fields and argument text stay synchronized.
- NR rejects LBFGS, ScheduleFree, AdamMini, and Sophia because their required training-loop behavior is not implemented. LoRARite is available only for LoRA, not full training. These checks also cover explicit class paths.
- Numeric presets and pending edits are validated without silently truncating integers, clamping invalid values, or exporting an earlier valid value. Invalid preset loads/saves display errors. String booleans such as `false` remain false.
- Forward-report, development-smoke, and deployment-target controls are removed from the GUI and built-in training presets. GUI training requires canonical source weights and never silently enables smoke mode.
- Normal backend training no longer requires a forward-validation report. Canonical schema/profile/conversion provenance checks remain mandatory. Explicitly supplied reports remain strictly validated, even in developer smoke mode.
- Training records source-forward validation separately from trained-output compatibility. No float, temporal, or native/DLL acceptance flag is promoted by removing the entry gate.

## Results

- NR command-builder/GUI tests: **93 passed**. PowerShell workflow tests: **13 passed**. A combined rerun against the merged backend passed all **106 tests** in 78.10 seconds.
- `python -m pytest gui/tests -q --tb=line`: **533 passed, 5 failed**, in 156.04 seconds during final review. The same five H3 failures existed before this rework (previous adapter run: 487 passed, 5 failed).
- All `test_dlssnr_*.py` files on the `DLSSNR` branch before its push: **227 passed, 5 skipped**, in 257.25 seconds.
- Merged `qinglong` backend `python -m pytest -q --tb=short -ra`: **1372 passed, 5 skipped, 50 warnings**, in 477.06 seconds. Four NR CUDA weight-smoke cases require an explicit opt-in; one packing test requires unavailable external NR weights.
- Independent backend rerun of `test_dlssnr_artifacts.py`, `test_dlssnr_config.py`, `test_dlssnr_training.py`, and `test_dlssnr_buckets.py`: **143 passed**.
- `python -m compileall -q gui`: passed.
- Ruff check/format on six focused GUI Python files and eight modified backend Python files: passed. Undefined-name/syntax checks on all changed GUI production Python files: passed.
- `git diff --check` in parent and backend: passed; parent Git reported only local LF/CRLF conversion notices.
- Independent read-only review: no remaining blocking findings after correcting Sophia/LoRARite eligibility, the Simplified AdEMAMix class name, numeric/boolean preset validation, and invalid pending-edit export. Regression cases were run failing before their fixes and passing afterward.
- Final pre-publication review found no new P0/P1/P2 issue. It rechecked shared optimizer controls, legacy non-NR argument semantics, preset event ordering, architecture switching, and pending-edit validation. Backend callers and evidence/metadata tests were reviewed separately.
- Windows safetensors test fixtures clone mapped tensors before overwriting their files; production tensor loading was not changed for this test-only file-lock fix.

## Browser Verification

Playwright exercised the actual NiceGUI pages at desktop 1440x1000 and mobile 390x844:

- Training: flat top-level sections, full-width paths, optimizer selection and argument synchronization, rejection of fractional step counts, and temporal controls.
- Presets: saved a temporary full-temporal user preset, checked its effective config through the actual backend parser, reloaded it from the default diffusion architecture, and deleted only that temporary preset through the GUI. It restored full mode, temporal 4/2/2, learning rate `1e-5`, and no smoke/report flags.
- Save dialog: the previous 400px minimum caused mobile overflow. The responsive dialog now measures x=24..366 inside the 390px viewport.
- Generation: model/manifest/output paths, single-frame and closed-loop PNG-sequence modes, and editable 33x37 geometry without unwanted clamping. Diffusion-only tabs remain hidden.
- No horizontal page overflow at either tested width; inspected inputs remained inside the viewport. Browser console reported zero errors/warnings, and the final preview server stderr was empty.

The test browser is closed. Playwright snapshots/screenshots are local QA artifacts, not staged source changes.
The preview remains available at `http://127.0.0.1:7790/train`.

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

The image example uses 28 steps while its test expects 30. Other failures concern
H3 parser options and assertions requiring the unrelated H3 commit `9a087963...`
or a clean submodule matching the parent gitlink. The full GUI run read the
pre-publication parent pin `97f6f85...`; this commit advances it to the published
DLSS-NR integration `9c37b52...`, not the unrelated hard-coded H3 revision.
Updating H3 samples, options, or exact-commit assertions is outside this task;
those tests were not weakened to obtain a pass.

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
equivalence. Normal training requires valid canonical conversion provenance but
not forward evidence. An explicitly supplied report must still match its source
weights, implementation, numerics, and required checks. Developer-only random
initialization still requires explicit CLI `development_smoke`; the GUI does not
expose or infer it. Older training-state implementation identities are not
silently accepted for exact resume after the implementation change. LoRA inference
requires merging the adapter with the existing backend merge tool.

Catalog availability is not proof that every optional optimizer dependency works
on this machine: the installed `pytorch_optimizer` lacks `EmoNeco`/`EmoZeal`, and
the installed bitsandbytes lacks a CUDA 13.0 binary. No dependency installation or
upgrade was performed, and no claim is made that every listed optimizer was
validated in real GPU training.
