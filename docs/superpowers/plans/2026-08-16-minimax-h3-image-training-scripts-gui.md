# MiniMax-H3 Image Training Scripts And GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose upstream MiniMax-H3 one-frame T2VA image LoRA training through the repository's PowerShell workflows and native GUI without changing existing video workflows.

**Architecture:** Pin the submodule gitlink first so clean checkouts own the three-stage `--one_frame` contract. Add explicit one-frame state to the cache and train command paths, then layer task-aware GUI controls and complete presets over that contract. Keep dataset ownership on Step 1 and store the H3 row-template identity only in project interop metadata.

**Tech Stack:** Python 3, unittest/pytest, NiceGUI, TOML, PowerShell, Git submodules

## Global Constraints

- Pin the `musubi-tuner` gitlink to `b462291`; do not edit source files inside the submodule.
- MiniMax-H3 image mode is exactly `version=fl2va`, `task=t2va`, and `one_frame=true`.
- Existing T2VA, FL2VA, and Ref2VA video presets must build commands without `--one_frame` after any prior preset state.
- The image train preset uses `video_only=true`, guidance scale `4.0`, guidance sigma minimum `0.15`, and 50 warmup steps.
- Empty optional arguments are omitted; numeric zero is preserved.
- Do not infer `one_frame` from dataset contents because upstream permits mixed image/video training.
- Preserve unrelated dirty-worktree changes and stage only files named by the current task.

---

## File Map

- `musubi-tuner`: committed gitlink containing upstream `ca221b1`.
- `gui/utils/command_builder.py`: H3 state validation and exact cache/train CLI translation.
- `2.11minimax_h3_cache_latent_and_text_encoder.ps1`: editable one-frame and unconditional-cache workflow.
- `3.11minimax_h3_train_lora.ps1`: editable one-frame, video-only, audio, and guidance workflow.
- `gui/utils/model_catalog.py`: declares H3 cache/train `one_frame` capability.
- `gui/wizard/step2_cache.py`: task-aware cache toggle and callback synchronization.
- `gui/wizard/step3_train.py`: task-aware train toggle and callback synchronization.
- `gui/wizard/step1_tagging.py`: H3 dataset template, validation, cleanup, preview, and interop metadata.
- `gui/utils/i18n.py`: four-language H3 image-mode and target-index copy.
- `gui/presets/cache/minimax_h3*.toml`: image preset plus explicit video resets.
- `gui/presets/train/minimax_h3*.toml`: image preset plus explicit video resets.
- `toml/qinglong_minimax_h3_image.toml`: standalone Step 1 dataset example.
- `toml/qinglong_minimaxh3_image.txt`: one-frame training sample prompt.
- `gui/tests/test_minimax_h3_command_builder.py`: indexed parser ownership and CLI behavior.
- `gui/tests/test_minimax_h3_scripts.py`: PowerShell contract.
- `gui/tests/test_minimax_h3_gui_contract.py`: GUI rendering and task-state behavior.
- `gui/tests/test_preset_scope_and_defaults.py`: preset reset directions and recommended defaults.
- `gui/tests/test_dataset_page_refactor.py`: dataset template and persistence behavior.

### Task 1: Pin And Classify The Upstream Parser Contract

**Files:**
- Modify: `musubi-tuner`
- Modify: `gui/tests/test_minimax_h3_command_builder.py`

**Interfaces:**
- Consumes: indexed submodule gitlink and the four H3 parser files.
- Produces: `_indexed_submodule_source()`-based parser flag classification and a gitlink whose three training parsers expose `--one_frame`.

- [ ] **Step 1: Write the failing indexed-gitlink tests**

Extend `expected_by_parser` so all three training stages require `--one_frame`. Replace the working-tree coverage scan with indexed sources and explicit per-parser supported/deferred sets. The deferred map must contain exactly:

```python
H3_DEFERRED_FLAGS_BY_PARSER = {
    "minimax_h3_cache_latents.py": set(),
    "minimax_h3_cache_text_encoder_outputs.py": {"--teacher_conditions"},
    "minimax_h3_train_network.py": {
        "--h3_teacher_matching",
        "--h3_teacher_conditions",
        "--h3_teacher_condition_sigma_max",
        "--h3_teacher_loss_dc_weight",
        "--h3_teacher_loss_mag_weight",
        "--h3_teacher_preservation_weight",
        "--h3_timestep_focus_min",
        "--h3_timestep_focus_max",
        "--h3_timestep_focus_prob",
        "--h3_video_best_of_k",
    },
    "minimax_h3_generate_video.py": {
        "--interactive",
        "--ref",
        "--trajectory_dir",
        "--trajectory_stride",
        "--lora_runtime_attach",
        "--one_frame",
        "--from_file",
        "--latent_path",
        "--bell",
    },
}
```

Assert parser classification per filename rather than against repository-wide literals.

- [ ] **Step 2: Run the indexed parser tests and confirm RED**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py::TestMiniMaxH3CommandBuilder::test_h3_gui_flags_are_supported_by_indexed_submodule_parsers -q
```

Expected: FAIL because the indexed gitlink is still `29aee45` and its three parsers do not all declare `--one_frame`.

- [ ] **Step 3: Pin the existing clean submodule checkout**

Record `musubi-tuner` at `b462291`. Do not change the submodule branch or any source file.

- [ ] **Step 4: Run the parser contract tests and confirm GREEN**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py -q
```

Expected: the indexed parser tests pass; functional one-frame tests added in Task 2 may still be absent.

- [ ] **Step 5: Commit the parser boundary**

```powershell
git add musubi-tuner gui/tests/test_minimax_h3_command_builder.py
git commit -m "chore: pin MiniMax-H3 image training parser support"
```

### Task 2: Carry One-Frame State Through Scripts And Command Builders

**Files:**
- Modify: `2.11minimax_h3_cache_latent_and_text_encoder.ps1`
- Modify: `3.11minimax_h3_train_lora.ps1`
- Modify: `gui/utils/command_builder.py`
- Modify: `gui/tests/test_minimax_h3_scripts.py`
- Modify: `gui/tests/test_minimax_h3_command_builder.py`

**Interfaces:**
- Consumes: `state["one_frame"]`, H3 task/version state, and optional guidance values.
- Produces: two cache jobs and one train job that emit the exact upstream flags or fail with `CommandBuildError` before launch.

- [ ] **Step 1: Write failing command-builder tests**

Add real job assertions equivalent to:

```python
image_state = {
    "arch": "MiniMax-H3",
    "version": "fl2va",
    "task": "t2va",
    "one_frame": True,
    **PATHS,
}
jobs = build_cache_jobs(image_state, tmp, IMAGE_PROJECT_CONFIG)
assert "--one_frame" in jobs[0].args
assert "--one_frame" in jobs[1].args

train = build_train_job(
    {
        **image_state,
        "train_mode": "lora",
        "mixed_precision": "bf16",
        "dit_dtype": "bfloat16",
        "video_only": True,
        "audio_loss_weight": 0,
        "h3_guidance_loss_scale": 4.0,
        "h3_guidance_loss_scale_audio": 0,
        "h3_guidance_loss_sigma_min": 0.15,
        "h3_guidance_loss_uncond_cache": "cache/minimax_h3_image_uncond.safetensors",
    },
    tmp,
    IMAGE_PROJECT_CONFIG,
)
assert "--one_frame" in train.args
assert "--video_only" in train.args
assert "--audio_loss_weight=0" in train.args
assert "--h3_guidance_loss_scale_audio=0" in train.args
```

Also assert errors for `one_frame` with FL2VA/Ref2VA tasks, enabled raw `teacher_conditions`, enabled raw `h3_teacher_matching`, a positive guidance scale without a cache, negative audio/guidance values, and sigma values outside `[0, 1]`.

- [ ] **Step 2: Write failing PowerShell contract tests**

Require the cache script to declare `$one_frame`, `$uncond_output`, and `$uncond_text`, append `--one_frame` to both argument lists, and omit empty unconditional text. Require the train script to declare every field from the design and preserve zero with explicit empty-string checks rather than truthiness.

- [ ] **Step 3: Run focused tests and confirm RED**

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py -q
```

Expected: FAIL because H3 whitelists and scripts do not yet carry one-frame state.

- [ ] **Step 4: Implement the minimal cache translation**

Add `one_frame` to `CACHE_LATENT_ARCH_BOOL_KEYS[MINIMAX_H3_ARCH]`. Add it to the text-cache bool map and H3 text whitelist so both jobs emit it. Reject H3 teacher-condition state in `_validate_minimax_h3_cache_state`.

In the PowerShell cache script, use one conditional to add `--one_frame` to both `$latent_args` and `$text_args`. Add unconditional arguments only when `$uncond_output` is nonempty; add `--uncond_text` only when its value is nonempty.

- [ ] **Step 5: Implement the minimal train translation**

Add `one_frame` to `TRAIN_ARCH_BOOL_KEYS[MINIMAX_H3_ARCH]`. Extend `_validate_minimax_h3_train_state` with the task predicate, teacher-matching rejection, numeric ranges, and existing guidance-cache dependency.

In the PowerShell train script, add explicit checks such as:

```powershell
if ($h3_guidance_loss_scale_audio -ne "") {
    [void]$ext_args.Add("--h3_guidance_loss_scale_audio=$h3_guidance_loss_scale_audio")
}
```

This preserves numeric zero. Add `--one_frame`, `--video_only`, audio weight, guidance fields, and the same validation rules as the builder.

- [ ] **Step 6: Run focused and regression tests**

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py gui/tests/test_command_builder.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the command contract**

```powershell
git add 2.11minimax_h3_cache_latent_and_text_encoder.ps1 3.11minimax_h3_train_lora.ps1 gui/utils/command_builder.py gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py
git commit -m "feat: add MiniMax-H3 image training commands"
```

### Task 3: Add Task-Aware GUI Controls And Complete Presets

**Files:**
- Modify: `gui/utils/model_catalog.py`
- Modify: `gui/wizard/step2_cache.py`
- Modify: `gui/wizard/step3_train.py`
- Modify: `gui/utils/i18n.py`
- Modify: `gui/presets/cache/minimax_h3.toml`
- Modify: `gui/presets/cache/minimax_h3_fl2va.toml`
- Modify: `gui/presets/cache/minimax_h3_ref2va.toml`
- Create: `gui/presets/cache/minimax_h3_image.toml`
- Modify: `gui/presets/train/minimax_h3.toml`
- Modify: `gui/presets/train/minimax_h3_fl2va.toml`
- Modify: `gui/presets/train/minimax_h3_ref2va.toml`
- Create: `gui/presets/train/minimax_h3_image.toml`
- Modify: `gui/tests/test_minimax_h3_gui_contract.py`
- Modify: `gui/tests/test_preset_scope_and_defaults.py`

**Interfaces:**
- Consumes: ModelSelector task callbacks and partial preset merges.
- Produces: visible one-frame toggles only for H3 T2VA and deterministic preset transitions in both directions.

- [ ] **Step 1: Write failing GUI and preset tests**

Assert the model catalog lists `one_frame` on H3 cache/train pages. Render both H3 cards and assert a bound `one_frame` control exists. Drive the real same-architecture task callback from `t2va` to `fl2va/ref2va` and assert the control is hidden or disabled and its collected value is false.

Load every H3 preset. Assert all six existing cache/train video presets set `one_frame=false`. Assert the image cache preset sets both cache stages true, selectors to FL2VA/T2VA, `uncond_output`, and `uncond_text=""`. Assert the image train preset resets selectors and optional guidance audio, enables sampling, and uses the recommended values and a unique output name.

For partial merges, test both directions:

```python
image_then_video = {**image_preset, **video_preset}
assert image_then_video["one_frame"] is False

custom_then_image = {**custom_ref2va_state, **image_preset}
assert custom_then_image["version"] == "fl2va"
assert custom_then_image["task"] == "t2va"
assert custom_then_image["uncond_text"] == ""
```

- [ ] **Step 2: Run GUI/preset tests and confirm RED**

```powershell
python -m pytest gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_preset_scope_and_defaults.py gui/tests/test_model_catalog.py -q
```

Expected: FAIL for missing controls, flags, and preset files.

- [ ] **Step 3: Add model catalog and GUI controls**

Add `one_frame` to H3 cache/train page flags. Render an H3-specific experimental image-training toggle on both pages with translated label and tooltip. Keep a reference to the row/control so the H3 synchronizers can update visibility and value.

Change CacheStep's same-architecture/version early-return path to call the H3 synchronizer before returning. Extend the existing TrainStep synchronizer to apply the same T2VA predicate.

- [ ] **Step 4: Add complete image and reset presets**

Set `one_frame=false` in every existing H3 cache/train preset. Create the two image presets with the complete reset state from the specification. Do not make either preset change the active Step 1 dataset.

- [ ] **Step 5: Add four-language image-mode copy**

Add nonempty English, Simplified Chinese, Japanese, and Korean values for the H3 image-mode label and tooltip. Keep copy concise and state that FL2VA/T2VA is required.

- [ ] **Step 6: Run GUI/preset tests and confirm GREEN**

```powershell
python -m pytest gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_preset_scope_and_defaults.py gui/tests/test_model_catalog.py gui/tests/test_form_state.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit GUI and presets**

```powershell
git add gui/utils/model_catalog.py gui/wizard/step2_cache.py gui/wizard/step3_train.py gui/utils/i18n.py gui/presets/cache gui/presets/train gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_preset_scope_and_defaults.py
git commit -m "feat: add MiniMax-H3 image training presets"
```

### Task 4: Add The H3 Image Dataset Template And Examples

**Files:**
- Modify: `gui/wizard/step1_tagging.py`
- Modify: `gui/utils/i18n.py`
- Modify: `gui/tests/test_dataset_page_refactor.py`
- Create: `toml/qinglong_minimax_h3_image.toml`
- Create: `toml/qinglong_minimaxh3_image.txt`

**Interfaces:**
- Consumes: image dataset row state and `interop.dataset_templates` metadata.
- Produces: standard upstream TOML with optional `fp_1f_target_index`, plus GUI-only template identity that never reaches the exported dataset file.

- [ ] **Step 1: Write failing dataset state tests**

Cover these concrete cases:

```python
assert step._infer_dataset_row_template(
    "image", {"fp_1f_target_index": 0}
) == "minimax_h3_one_frame"  # when import source is qinglong_minimax_h3_image.toml
```

Also test manual template metadata persistence, preview classification, explicit zero export, empty omission, directory and JSONL sources, negative input error, nonnumeric input error, and switching from FramePack/image-edit state clears every unsupported hidden field.

- [ ] **Step 2: Run dataset tests and confirm RED**

```powershell
python -m pytest gui/tests/test_dataset_page_refactor.py -q
```

Expected: FAIL because the H3 template and validation do not exist.

- [ ] **Step 3: Implement template inference and persistence**

Add `minimax_h3_one_frame` to image template options. In import inference, check the `minimax_h3` source marker and key membership for `fp_1f_target_index` before FramePack truthiness checks. Persist the selected template list under project `interop` metadata aligned by dataset row index, and consult it in preview detection.

- [ ] **Step 4: Implement cleanup and strict target-index validation**

When switching to the H3 template, reset control paths/sizing, `fp_latent_window_size`, `fp_1f_clean_indices`, `fp_1f_no_post`, `multiple_target`, and `no_resize_control`. Render only the standard image controls and H3 target index.

Add a parser that distinguishes empty from invalid input:

```python
def _parse_optional_nonnegative_int(raw_value: Any, label: str) -> int | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a nonnegative integer") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return parsed
```

Use the existing save error path to show the message and block persistence.

- [ ] **Step 5: Add examples and translations**

Create a 1024 by 1024 image dataset example with batch size 1, bucketing, a separate image cache directory, and `fp_1f_target_index=0`. Create a prompt line with `--w 1024 --h 1024 --f 1 --s 30` and a fixed seed. Add four-language template, field, and tooltip translations explaining the zero-based 24 fps index.

- [ ] **Step 6: Run dataset and end-to-end tests**

```powershell
python -m pytest gui/tests/test_dataset_page_refactor.py gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_preset_scope_and_defaults.py -q
```

Expected: PASS, including import dataset -> image cache preset -> image train preset -> built command flow.

- [ ] **Step 7: Commit dataset support**

```powershell
git add gui/wizard/step1_tagging.py gui/utils/i18n.py gui/tests/test_dataset_page_refactor.py toml/qinglong_minimax_h3_image.toml toml/qinglong_minimaxh3_image.txt
git commit -m "feat: add MiniMax-H3 image dataset workflow"
```

### Task 5: Full Verification And Visual QA

**Files:**
- Modify only if a test or visual defect requires a scoped fix.

**Interfaces:**
- Consumes: the completed image workflow.
- Produces: passing repository tests and verified desktop/narrow GUI layouts.

- [ ] **Step 1: Run focused static and behavioral tests**

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_preset_scope_and_defaults.py gui/tests/test_dataset_page_refactor.py gui/tests/test_model_catalog.py -q
```

- [ ] **Step 2: Run the complete GUI suite**

```powershell
python -m pytest gui/tests -q
```

- [ ] **Step 3: Run source checks**

```powershell
python -m compileall -q gui
git diff --check
git status --short
```

- [ ] **Step 4: Launch the GUI and inspect both changed pages**

Start `python gui/launch.py` on an unused local port. Use Playwright at a desktop viewport and a narrow mobile viewport to open the cache, train, and dataset pages. Capture screenshots and verify the toggles, H3 target-index field, and adjacent controls do not overlap, overflow, or resize their rows unexpectedly.

- [ ] **Step 5: Fix only verified defects and rerun affected checks**

For any failure, add or refine the failing test before changing production code. Rerun the focused test, then the full GUI suite.

- [ ] **Step 6: Review final scope**

Confirm the final diff contains only the gitlink, H3 scripts/GUI/presets/tests, examples, and approved documentation. Do not stage unrelated user files.
