# MiniMax-H3 FL2VA One-Frame Image Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned MiniMax-H3 FL2VA one-frame image-edit and inbetween training workflows usable end to end from built-in dataset, cache, and train presets, PowerShell scripts, and the GUI.

**Architecture:** Keep the backend gitlink unchanged because it already contains the upstream implementation. Extend the parent task matrix from T2VA-only to T2VA-or-FL2VA at every launch boundary, then teach the existing single H3 dataset editor to round-trip zero, one, or two control images without exposing backend-incompatible resize fields. Add paired presets for the two controlled workflows while preserving the existing plain-image path and Best-of-K defaults.

**Tech Stack:** Python 3, pytest/unittest, NiceGUI, TOML, PowerShell 5.1/7, Git submodules, Playwright.

## Global Constraints

- The parent gitlink remains the full commit `c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`.
- The gitlink commit must contain upstream `dev` commit `2f7677f6f5e2c5dca22c70dd6376f1487d6626cb`.
- One-frame mode is supported only for MiniMax-H3 version `fl2va` with task `t2va` or `fl2va`; Ref2VA remains unsupported.
- Use `control_count` for the number of dataset control images; reserve `h3_best_of_k` for candidate-noise search.
- H3 control indices are one or two nonnegative integers when present.
- Directory datasets require `control_directory` and `fp_1f_clean_indices` together; controlled datasets also require `fp_1f_target_index`.
- H3 dataset UI and export must not expose `no_resize_control`, `control_resolution`, or `multiple_target`.
- Built-in controlled-image sampling stays disabled until the user supplies real condition files.
- Every new train preset stores integer `h3_best_of_k = 1` and string `h3_best_of_k_stream = "video"`.
- Unpublished removed Best-of-K aliases receive no compatibility, migration, warning, display, or dedicated test logic.
- Preserve and never stage unrelated user-owned working-tree changes.
- Run the full GUI suite and desktop/mobile visual verification from a clean worktree before integration.

---

### Task 1: Extend The One-Frame Launch Matrix

**Files:**
- Modify: `gui/tests/test_minimax_h3_command_builder.py`
- Modify: `gui/tests/test_minimax_h3_scripts.py`
- Modify: `gui/utils/command_builder.py`
- Modify: `2.11minimax_h3_cache_latent_and_text_encoder.ps1`
- Modify: `3.11minimax_h3_train_lora.ps1`

**Interfaces:**
- Consumes: MiniMax-H3 state fields `version: str`, `task: str`, and `one_frame: bool`.
- Produces: cache/train launch validation that accepts `(fl2va, t2va, true)` and `(fl2va, fl2va, true)`, rejects Ref2VA one-frame state before launch, and emits `--one_frame` exactly once per job.

- [ ] **Step 1: Write failing builder tests for the exact task matrix**

Replace the T2VA-only cache expectation with explicit supported and rejected states:

```python
def test_one_frame_cache_accepts_t2va_and_fl2va_but_rejects_ref2va(self):
    valid = {**CACHE_STATE, "version": "fl2va", "one_frame": True}
    for task in ("t2va", "fl2va"):
        jobs = build_cache_jobs({**valid, "task": task})
        self.assertEqual(len(jobs), 2)
        for job in jobs:
            self.assertEqual(job.args.count("--one_frame"), 1)
            self.assertEqual(job.args.count(f"--task={task}"), 1)

    with self.assertRaisesRegex(ValueError, "one_frame.*t2va.*fl2va"):
        build_cache_jobs({**valid, "version": "ref2va", "task": "ref2va"})
```

Add the equivalent train contract:

```python
def test_one_frame_train_accepts_t2va_and_fl2va_but_rejects_ref2va(self):
    valid = {**TRAIN_STATE, "version": "fl2va", "one_frame": True}
    for task in ("t2va", "fl2va"):
        job = build_train_job({**valid, "task": task})
        self.assertEqual(job.args.count("--one_frame"), 1)
        self.assertEqual(job.args.count(f"--task={task}"), 1)

    with self.assertRaisesRegex(ValueError, "one_frame.*t2va.*fl2va"):
        build_train_job({**valid, "version": "ref2va", "task": "ref2va"})
```

- [ ] **Step 2: Add failing PowerShell source-contract assertions**

Assert both launch scripts contain a two-task membership check rather than a T2VA equality check:

```python
for source in (cache, train):
    self.assertIn('$task -notin @("t2va", "fl2va")', source)
    self.assertNotIn('$task -ne "t2va"', source)
```

- [ ] **Step 3: Run focused tests and verify the current T2VA-only behavior fails**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py -k "one_frame_cache or one_frame_train" -q
python -m pytest gui/tests/test_minimax_h3_scripts.py -k "one_frame" -q
```

Expected: FL2VA one-frame builder cases raise the stale T2VA-only validation error, and both PowerShell source checks fail.

- [ ] **Step 4: Implement the shared two-task validation contract**

In both Python validators, use the same membership rule and deterministic message:

```python
if one_frame and (version != "fl2va" or task not in {"t2va", "fl2va"}):
    raise ValueError("MiniMax-H3 one_frame requires version=fl2va and task=t2va or task=fl2va")
```

In both PowerShell scripts, validate before environment activation or process launch:

```powershell
if ($one_frame -and ($version -ne "fl2va" -or $task -notin @("t2va", "fl2va"))) {
    throw "MiniMax-H3 one_frame requires version=fl2va and task=t2va or task=fl2va."
}
```

- [ ] **Step 5: Run focused tests and commit the launch-boundary change**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py -q
git add -- 2.11minimax_h3_cache_latent_and_text_encoder.ps1 3.11minimax_h3_train_lora.ps1 gui/utils/command_builder.py gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py
git commit -m "feat: allow H3 FL2VA one-frame launches"
```

Expected: both test modules pass and the commit contains only the five listed files.

---

### Task 2: Preserve FL2VA One-Frame State In Cache And Train UI

**Files:**
- Modify: `gui/tests/test_minimax_h3_gui_contract.py`
- Modify: `gui/wizard/step2_cache.py`
- Modify: `gui/wizard/step3_train.py`

**Interfaces:**
- Consumes: the selected architecture, H3 version, task, and current `one_frame` value.
- Produces: `_sync_minimax_h3_cache_ui() -> None` and `_sync_minimax_h3_train_ui() -> None` visibility/state transitions with FL2VA T2VA and FL2VA both available, while Ref2VA clears stale state.

- [ ] **Step 1: Rewrite the callback test as a full transition test**

For both cache and train steps, assert this sequence:

```python
step._write_control_value(step.version, "fl2va")
step._write_control_value(step.task, "t2va")
step._write_control_value(step.one_frame, True)
step._sync_minimax_h3_cache_ui()  # use _sync_minimax_h3_train_ui for train
self.assertTrue(step._h3_one_frame_row.visible)
self.assertTrue(step.config["one_frame"])

step._write_control_value(step.task, "fl2va")
step._sync_minimax_h3_cache_ui()
self.assertTrue(step._h3_one_frame_row.visible)
self.assertTrue(step.config["one_frame"])

step._write_control_value(step.version, "ref2va")
step._write_control_value(step.task, "ref2va")
step._sync_minimax_h3_cache_ui()
self.assertFalse(step._h3_one_frame_row.visible)
self.assertFalse(step.config["one_frame"])

step._write_control_value(step.version, "fl2va")
step._write_control_value(step.task, "fl2va")
step._sync_minimax_h3_cache_ui()
self.assertTrue(step._h3_one_frame_row.visible)
self.assertFalse(step.config["one_frame"])
```

- [ ] **Step 2: Run the callback test and verify FL2VA currently hides the row**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_gui_contract.py -k "task_callbacks" -q
```

Expected: the FL2VA task assertion reports that the one-frame row is hidden or its state was cleared.

- [ ] **Step 3: Update both synchronizers**

Use the identical availability expression in cache and train pages:

```python
one_frame_available = (
    is_h3
    and version == "fl2va"
    and task in {"t2va", "fl2va"}
)
```

Retain the existing branch that clears `config["one_frame"]` and its bound control only when `one_frame_available` is false.

- [ ] **Step 4: Run GUI contract tests and commit the UI state change**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_gui_contract.py -q
git add -- gui/wizard/step2_cache.py gui/wizard/step3_train.py gui/tests/test_minimax_h3_gui_contract.py
git commit -m "feat: expose H3 FL2VA one-frame mode"
```

Expected: the complete GUI contract module passes.

---

### Task 3: Add Controlled H3 Dataset Editing And Validation

**Files:**
- Modify: `gui/tests/test_dataset_page_refactor.py`
- Modify: `gui/wizard/step1_tagging.py`
- Modify: `gui/utils/i18n.py`

**Interfaces:**
- Consumes: dataset row fields `source_mode`, `image_directory` or `image_jsonl_file`, optional `control_directory`, optional `fp_1f_clean_indices`, and optional `fp_1f_target_index`.
- Produces: `_parse_h3_control_indices(raw_value: Any, label: str) -> list[int]`, H3-specific translated labels, and canonical dataset TOML export for zero, one, or two controls.

- [ ] **Step 1: Add failing parser and field-pairing tests**

Cover strict normalization and rejection:

```python
def test_h3_control_indices_accept_one_or_two_nonnegative_integers(self):
    self.assertEqual(_parse_h3_control_indices("0", "H3 control frame indices"), [0])
    self.assertEqual(_parse_h3_control_indices("0, 48", "H3 control frame indices"), [0, 48])

def test_h3_control_indices_reject_invalid_shapes(self):
    for value in ("-1", "", "0, 24, 48", "0, nope", "1.5"):
        with self.subTest(value=value), self.assertRaises(ValueError):
            _parse_h3_control_indices(value, "H3 control frame indices")
```

Keep the empty-string case accepted only by the optional editor path, not by the strict parser itself. Add directory export tests that require control directory and indices together, require a target for controlled rows, and preserve `[0]` or `[0, 48]` exactly.

- [ ] **Step 2: Add failing render and round-trip tests**

Update the H3 template contract to require these controls:

```python
self.assertIn("control_directory", rendered_fields)
self.assertIn("fp_1f_clean_indices", rendered_fields)
self.assertIn("fp_1f_target_index", rendered_fields)
self.assertTrue({"control_resolution", "no_resize_control", "multiple_target"}.isdisjoint(rendered_fields))
```

Add preset-shaped directory states for edit and inbetween, collect them, serialize them, re-import them, and assert exact control directory, clean-index list, and target index values after the round trip.

- [ ] **Step 3: Run the focused dataset tests and verify the current editor fails**

Run:

```powershell
python -m pytest gui/tests/test_dataset_page_refactor.py -k "h3 or minimax" -q
```

Expected: the H3 template lacks the control fields and cannot round-trip the controlled states.

- [ ] **Step 4: Implement strict H3 control-index parsing**

Add a module-level helper next to `_parse_optional_nonnegative_int`:

```python
def _parse_h3_control_indices(raw_value: Any, label: str) -> list[int]:
    if isinstance(raw_value, bool):
        raise ValueError(f"{label} must contain one or two nonnegative integers")
    if isinstance(raw_value, str):
        tokens = [token.strip() for token in raw_value.split(",")]
        if not raw_value.strip() or any(not token for token in tokens):
            raise ValueError(f"{label} must contain one or two nonnegative integers")
    elif isinstance(raw_value, (list, tuple)):
        tokens = list(raw_value)
    else:
        tokens = [raw_value]
    if len(tokens) not in {1, 2}:
        raise ValueError(f"{label} must contain one or two nonnegative integers")
    values = []
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) and not (
            isinstance(token, str) and token.isdecimal()
        ):
            raise ValueError(f"{label} must contain one or two nonnegative integers")
        value = int(token)
        if value < 0:
            raise ValueError(f"{label} must contain one or two nonnegative integers")
        values.append(value)
    return values
```

- [ ] **Step 5: Render and collect canonical controlled H3 fields**

Render `control_directory` for directory-mode `minimax_h3_one_frame` rows and render `fp_1f_clean_indices` for both directory and JSONL modes. Keep the resize/no-resize row limited to the existing non-H3 templates.

In `_collect_dataset_rows`, use these rules:

```python
control_directory = self._string_value(state.get("control_directory"))
raw_indices = self._string_value(state.get("fp_1f_clean_indices"))
has_indices = bool(raw_indices.strip())
if source_mode == "directory" and bool(control_directory) != has_indices:
    raise ValueError("MiniMax-H3 control directory and control frame indices must be set together")
indices = _parse_h3_control_indices(raw_indices, t("minimax_h3_control_indices")) if has_indices else []
target_index = _parse_optional_nonnegative_int(
    state.get("fp_1f_target_index"), t("minimax_h3_target_index")
)
if (control_directory or indices) and target_index is None:
    raise ValueError("MiniMax-H3 controlled datasets require a target frame index")
```

Export `control_directory` only for directory rows, `fp_1f_clean_indices` only when present, and `fp_1f_target_index` only when present. Continue clearing FramePack-only settings when switching into the H3 template; applying a preset will then restore canonical values.

- [ ] **Step 6: Add translations for the H3-specific index field**

Register `minimax_h3_control_indices` and `minimax_h3_control_indices_tooltip` in English, Simplified Chinese, Japanese, and Korean. The English copy is:

```text
H3 Control Frame Indices
One or two comma-separated source frame indices at 24 fps, for example 0 or 0, 48.
```

Use native translations in the other three locale blocks and leave the existing FramePack label untouched.

- [ ] **Step 7: Run dataset/i18n tests and commit the editor change**

Run:

```powershell
python -m pytest gui/tests/test_dataset_page_refactor.py gui/tests/test_minimax_h3_gui_contract.py -q
git add -- gui/wizard/step1_tagging.py gui/utils/i18n.py gui/tests/test_dataset_page_refactor.py gui/tests/test_minimax_h3_gui_contract.py
git commit -m "feat: edit H3 controlled image datasets"
```

Expected: the dataset and GUI contract modules pass with exact one-control and two-control round trips.

---

### Task 4: Add Complete Edit And Inbetween Preset Families

**Files:**
- Create: `toml/qinglong_minimax_h3_image_edit.toml`
- Create: `toml/qinglong_minimax_h3_image_inbetween.toml`
- Create: `toml/qinglong_minimaxh3_image_edit.txt`
- Create: `toml/qinglong_minimaxh3_image_inbetween.txt`
- Create: `gui/presets/cache/minimax_h3_image_edit.toml`
- Create: `gui/presets/cache/minimax_h3_image_inbetween.toml`
- Create: `gui/presets/train/minimax_h3_image_edit.toml`
- Create: `gui/presets/train/minimax_h3_image_inbetween.toml`
- Modify: `gui/tests/test_dataset_page_refactor.py`
- Modify: `gui/tests/test_preset_scope_and_defaults.py`

**Interfaces:**
- Consumes: existing plain H3 image preset defaults and the Task 1/Task 3 FL2VA contracts.
- Produces: three discoverable image workflow families with exact control layouts `[]`, `[0]`, and `[0, 48]`, plus inactive sampling templates for the controlled workflows.

- [ ] **Step 1: Add failing preset discovery and exact-value tests**

Define the expected workflow matrix in tests:

```python
workflows = {
    "minimax_h3_image": {"task": "t2va", "controls": None, "target": 0},
    "minimax_h3_image_edit": {"task": "fl2va", "controls": [0], "target": 24},
    "minimax_h3_image_inbetween": {"task": "fl2va", "controls": [0, 48], "target": 24},
}
```

For the two new names, assert dataset control/cache directories are distinct, cache presets set `version="fl2va"`, `task="fl2va"`, `one_frame=true`, and train presets set all required one-frame recommendations. Assert `h3_best_of_k` is an integer equal to `1`, the stream is `video`, `enable_sample=false`, and `sample_at_first=false`.

- [ ] **Step 2: Run preset tests and verify all new paths are absent**

Run:

```powershell
python -m pytest gui/tests/test_preset_scope_and_defaults.py gui/tests/test_dataset_page_refactor.py -k "minimax_h3_image" -q
```

Expected: discovery or file-open assertions fail for the new edit and inbetween paths.

- [ ] **Step 3: Create the two dataset presets**

Follow the existing plain 1024 by 1024 image dataset layout. The edit dataset table must contain:

```toml
control_directory = "./dataset/minimax_h3_image_edit/control"
cache_directory = "./dataset/minimax_h3_image_edit/cache"
fp_1f_clean_indices = [0]
fp_1f_target_index = 24
```

The inbetween dataset table must contain:

```toml
control_directory = "./dataset/minimax_h3_image_inbetween/control"
cache_directory = "./dataset/minimax_h3_image_inbetween/cache"
fp_1f_clean_indices = [0, 48]
fp_1f_target_index = 24
```

Both use their own target `image_directory`, `batch_size = 1`, `[general] resolution = [1024, 1024]`, and omit `multiple_target`, `no_resize_control`, and `control_resolution`.

- [ ] **Step 4: Create the two cache presets**

Copy the maintained H3 image cache defaults, change only the preset identity/dataset path, and set:

```toml
arch = "MiniMax-H3"
version = "fl2va"
task = "fl2va"
one_frame = true
cache_latents = true
cache_text_encoder = true
uncond_cache_output = "./cache/minimax_h3_image_uncond.safetensors"
```

Retain the existing quantized text encoder, video/audio VAE, and block-swap values from `gui/presets/cache/minimax_h3_image.toml`.

- [ ] **Step 5: Create the train presets and inactive prompt templates**

Copy the maintained plain H3 image train defaults, use each new dataset path and a distinct output name, then set:

```toml
arch = "MiniMax-H3"
version = "fl2va"
task = "fl2va"
one_frame = true
video_only = true
h3_teacher_matching = false
h3_guidance_loss_scale = 4.0
h3_guidance_loss_sigma_min = 0.15
h3_guidance_loss_uncond_cache = "./cache/minimax_h3_image_uncond.safetensors"
lr_warmup_steps = 50
h3_best_of_k = 1
h3_best_of_k_stream = "video"
enable_sample = false
sample_at_first = false
```

Point `sample_prompts` at the matching text file. The edit template demonstrates one `--i` condition with its `--ei`; the inbetween template demonstrates two ordered `--i` conditions with matching `--ei` values and includes `--of 1`. Use placeholder paths only because sampling is disabled by default.

- [ ] **Step 6: Run preset, dataset-flow, and command-builder tests**

Run:

```powershell
python -m pytest gui/tests/test_preset_scope_and_defaults.py gui/tests/test_dataset_page_refactor.py gui/tests/test_minimax_h3_command_builder.py -q
```

Expected: all three modules pass, and applying either new train preset builds a single FL2VA one-frame train job without reading the inactive sample paths.

- [ ] **Step 7: Commit the preset families**

Run:

```powershell
git add -- toml/qinglong_minimax_h3_image_edit.toml toml/qinglong_minimax_h3_image_inbetween.toml toml/qinglong_minimaxh3_image_edit.txt toml/qinglong_minimaxh3_image_inbetween.txt gui/presets/cache/minimax_h3_image_edit.toml gui/presets/cache/minimax_h3_image_inbetween.toml gui/presets/train/minimax_h3_image_edit.toml gui/presets/train/minimax_h3_image_inbetween.toml gui/tests/test_dataset_page_refactor.py gui/tests/test_preset_scope_and_defaults.py
git commit -m "feat: add H3 FL2VA image presets"
```

Expected: only the ten listed feature/test paths enter the commit.

---

### Task 5: Verify Clean Reproduction And GUI Behavior

**Files:**
- Inspect: parent `musubi-tuner` gitlink and submodule checkout
- Inspect: all files changed by Tasks 1 through 4

**Interfaces:**
- Consumes: committed feature branch and clean initialized submodule.
- Produces: reproducible test, syntax, source-contract, and visual evidence suitable for fast-forward integration.

- [ ] **Step 1: Verify the exact gitlink and upstream ancestry**

Run:

```powershell
$gitlink = (git ls-tree HEAD -- musubi-tuner).Split()[2]
$submoduleHead = git -C musubi-tuner rev-parse HEAD
if ($gitlink -ne "c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993") { throw "Unexpected gitlink: $gitlink" }
if ($submoduleHead -ne $gitlink) { throw "Submodule HEAD does not match parent gitlink" }
git -C musubi-tuner diff --quiet --exit-code
git -C musubi-tuner diff --cached --quiet --exit-code
git -C musubi-tuner merge-base --is-ancestor 2f7677f6f5e2c5dca22c70dd6376f1487d6626cb HEAD
```

Expected: every command exits zero and the submodule is clean at the full parent-tree SHA.

- [ ] **Step 2: Verify PowerShell syntax and the full GUI suite**

Run:

```powershell
@(
  "2.11minimax_h3_cache_latent_and_text_encoder.ps1",
  "3.11minimax_h3_train_lora.ps1"
) | ForEach-Object {
  $tokens = $null
  $errors = $null
  [void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $_), [ref]$tokens, [ref]$errors)
  if ($errors.Count) { throw ($errors | Out-String) }
}
python -m pytest gui/tests -q
```

Expected: both scripts parse without errors and the full GUI test suite passes with zero failures.

- [ ] **Step 3: Start the GUI and verify desktop and mobile layouts with Playwright**

Start the repository's GUI entry point on an unused local port. At 1440 by 1000 and 390 by 844, navigate through dataset, cache, and train pages; apply plain, edit, and inbetween presets; record screenshots and assert:

```text
Edit:       task=fl2va, one_frame=true, control indices=0
Inbetween:  task=fl2va, one_frame=true, control indices=0, 48
Both:       no clipping, no overlapping controls, no new console error/warning
```

Stop the browser session and GUI server after the screenshots are inspected.

- [ ] **Step 4: Review the staged scope and feature diff**

Run:

```powershell
git status --short
git diff --check main...HEAD
git diff --stat main...HEAD
git log --oneline main..HEAD
```

Expected: no uncommitted feature changes, no whitespace errors, no submodule delta, and no user-owned files in the branch diff.

- [ ] **Step 5: Fast-forward main, push, and remove the feature branch**

After removing the disposable worktree, run from the primary checkout:

```powershell
git switch main
git merge --ff-only codex/minimax-h3-fl2va-image-presets
git push origin main
git branch -d codex/minimax-h3-fl2va-image-presets
```

If the feature branch exists remotely, delete it only after `origin/main` contains the verified head:

```powershell
git push origin --delete codex/minimax-h3-fl2va-image-presets
```

Finally verify `main`, `origin/main`, and the pushed remote SHA match, while the primary checkout's pre-existing user changes and untracked files remain untouched.
