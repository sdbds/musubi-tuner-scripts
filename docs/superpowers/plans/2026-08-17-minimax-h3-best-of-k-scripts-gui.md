# MiniMax-H3 Best-Of-K Scripts And GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the finalized MiniMax-H3 image/video best-of-K contract through the pinned submodule, PowerShell training script, native GUI, command builder, and built-in H3 presets, with K disabled by default at integer 1.

**Architecture:** The parent repository pins `musubi-tuner` commit `c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`. The GUI and script own two canonical fields, `h3_best_of_k` and `h3_best_of_k_stream`; builder and script boundary helpers normalize them, reject duplicate canonical CLI injection, and emit each canonical option exactly once. H3-only UI state remains dormant across non-H3 architecture switches, while non-H3 builders continue to exclude it.

**Tech Stack:** Python 3, pytest/unittest, NiceGUI, TOML, PowerShell 5.1/7, Git submodules, Playwright.

## Global Constraints

- The exact submodule target is `c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`; short SHAs are display-only.
- `h3_best_of_k` defaults to integer `1`; no separate enable toggle is added.
- `h3_best_of_k_stream` defaults to `video` and accepts only `video` or `audio`.
- One-frame image batches always search image/video noise; the stream selector affects only multi-frame batches.
- `K>1`, stream `audio`, and `video_only=true` are rejected before launch.
- Structured H3 state is the only source of `--h3_best_of_k` and `--h3_best_of_k_stream`; raw injection of either canonical option or `--xm_best_of_k` is rejected in both `--name value` and `--name=value` forms.
- Unpublished experimental option spellings receive no compatibility, migration, warning, display, or dedicated test logic.
- Missing K defaults to 1; explicit empty, boolean, string, fractional, non-finite, zero, and negative builder values fail. Integer-valued finite GUI floats normalize to `int`.
- Built-in H3 presets explicitly reset K and stream to `1` and `video`.
- Do not modify files inside the `musubi-tuner` submodule; update only the parent gitlink.
- Preserve and never stage unrelated user-owned working-tree changes.

---

### Task 1: Pin The Parent Gitlink And Update The Parser Contract

**Files:**
- Modify: `musubi-tuner` gitlink
- Modify: `gui/tests/test_minimax_h3_command_builder.py`

**Interfaces:**
- Consumes: Git parent tree and the H3 parser at `src/musubi_tuner/minimax_h3_train_network.py`.
- Produces: `H3_SUBMODULE_TARGET_SHA` and `_parent_tree_submodule_commit(treeish: str = "HEAD") -> str` in the test module; parser support classification containing the two canonical H3 options.

- [ ] **Step 1: Write the failing parent-tree and parser tests**

Add the exact target and tree-object reader. Keep `_indexed_submodule_source` for development coverage, but make gitlink acceptance independent of the dirty submodule checkout.

```python
H3_SUBMODULE_TARGET_SHA = "c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993"


def _parent_tree_submodule_commit(treeish: str = "HEAD") -> str:
    entry = subprocess.run(
        ["git", "ls-tree", treeish, "--", "musubi-tuner"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    mode, object_type, commit, path = entry.split(maxsplit=3)
    if (mode, object_type, path) != ("160000", "commit", "musubi-tuner"):
        raise AssertionError(f"Unexpected musubi-tuner tree entry: {entry}")
    return commit


def test_parent_tree_pins_final_h3_best_of_k_commit(self):
    self.assertEqual(_parent_tree_submodule_commit(), H3_SUBMODULE_TARGET_SHA)
```

Add `--h3_best_of_k` and `--h3_best_of_k_stream` to the supported trainer set. Remove the stale `--h3_video_best_of_k` entry from `H3_DEFERRED_FLAGS_BY_PARSER`; add no migration or removed-alias test.

- [ ] **Step 2: Run the focused test and verify it fails on the old parent gitlink**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py -k "parent_tree or specific_upstream_flags" -q
```

Expected: the parent-tree assertion reports `b462291a6e1bd25180ce1d1298db72982c8ed27a` instead of the target, and parser classification is not yet updated.

- [ ] **Step 3: Stage the exact submodule commit and the test contract**

Verify the checked-out submodule is clean and exact before staging:

```powershell
git -C musubi-tuner rev-parse HEAD
git -C musubi-tuner status --porcelain
git add -- musubi-tuner gui/tests/test_minimax_h3_command_builder.py
```

Expected: HEAD prints `c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`; status output is empty; only the gitlink and the H3 test are staged.

- [ ] **Step 4: Commit the parent-tree change, then rerun the contract**

The test intentionally reads `HEAD`, so commit is required before its green run.

```powershell
git commit -m "chore: pin H3 best-of-k submodule"
python -m pytest gui/tests/test_minimax_h3_command_builder.py -k "parent_tree or specific_upstream_flags" -q
```

Expected: PASS, and `git ls-tree HEAD musubi-tuner` contains the full target SHA.

---

### Task 2: Add Canonical Builder State, Validation, And Reserved-Argument Ownership

**Files:**
- Modify: `gui/utils/command_builder.py`
- Modify: `gui/tests/test_minimax_h3_command_builder.py`
- Modify: `gui/tests/test_command_builder.py`

**Interfaces:**
- Consumes: `build_train_job(state, project_dir, project_config)` and `_parse_optimizer_args_text(value)`.
- Produces: `_normalize_minimax_h3_best_of_k_count(value: Any) -> int`, `_normalize_minimax_h3_best_of_k_stream(value: Any) -> str`, `_validate_minimax_h3_reserved_cli_text(value: Any) -> None`, and `_validate_minimax_h3_best_of_k_argv(args: list[str]) -> None`.

- [ ] **Step 1: Write failing default, modality, and type-boundary tests**

Add a helper that constructs the complete valid H3 boundary state and test the public builder only.

```python
def best_of_k_h3_state(**overrides):
    state = {
        "arch": "MiniMax-H3",
        "version": "fl2va",
        "task": "t2va",
        **PATHS,
        "train_mode": "lora",
        "mixed_precision": "bf16",
        "dit_dtype": "bfloat16",
        "timestep_sampling": "uniform",
        "weighting_scheme": "none",
        "discrete_flow_shift": 1.0,
        "video_only": False,
        "audio_loss_weight": 1.0,
        "enable_sample": False,
        "optimizer_type": "AdamW_adv",
    }
    state.update(overrides)
    return state


def test_h3_best_of_k_defaults_are_emitted_canonically(self):
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(best_of_k_h3_state(), tmp, PROJECT_CONFIG)
    self.assertIn("--h3_best_of_k=1", job.args)
    self.assertIn("--h3_best_of_k_stream=video", job.args)


def test_h3_best_of_k_accepts_integer_valued_gui_float(self):
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(
            best_of_k_h3_state(h3_best_of_k=2.0, h3_best_of_k_stream="VIDEO"),
            tmp,
            PROJECT_CONFIG,
        )
    self.assertIn("--h3_best_of_k=2", job.args)
    self.assertIn("--h3_best_of_k_stream=video", job.args)
```

Use subtests to require `CommandBuildError` for `1.5`, `0`, `-1`, `True`, `""`, `"1"`, `"1e0"`, `"word"`, `float("inf")`, and `float("nan")`. Add cases for invalid stream `""`, `"music"`, and `1`, plus the active `K=2/audio/video_only` conflict. Add successful image and mixed-dataset cases proving the builder does not reject `one_frame=true` or mixed data when stream is `video`.

- [ ] **Step 2: Write failing reserved-argument and non-H3 isolation tests**

```python
def test_h3_best_of_k_reserved_options_cannot_escape_optimizer_args(self):
    cases = (
        "--h3_best_of_k 8",
        "--h3_best_of_k=8",
        "--h3_best_of_k_stream audio",
        "--h3_best_of_k_stream=audio",
        "--xm_best_of_k 8",
        "--xm_best_of_k=8",
    )
    for extra in cases:
        with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(CommandBuildError, "reserved"):
                build_train_job(
                    best_of_k_h3_state(optimizer_extra_args=extra),
                    tmp,
                    PROJECT_CONFIG,
                )


def test_non_h3_builder_never_emits_h3_best_of_k_state(self):
    state = {
        "arch": "FLUX.2",
        "version": "klein-base-4b",
        "dit_path": "ckpts/flux2.safetensors",
        "vae_path": "ckpts/ae.safetensors",
        "text_encoder_path": "ckpts/qwen3.safetensors",
        "h3_best_of_k": 8,
        "h3_best_of_k_stream": "audio",
    }
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(state, tmp, PROJECT_CONFIG)
    self.assertFalse(any(arg.startswith("--h3_best_of_k") for arg in job.args))
```

Also assert an H3 structured `xm_best_of_k` key fails instead of being ignored.

- [ ] **Step 3: Run the new builder tests and verify they fail**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_command_builder.py -k "best_of_k" -q
```

Expected: FAIL because the new fields are not mapped or validated.

- [ ] **Step 4: Implement canonical defaults and strict normalization**

Add the fields to `TRAIN_SCALARS` and the MiniMax-H3 branch of `TRAIN_ARCH_SCALAR_KEYS`. Default by key absence, not truthiness, so explicit empty values remain invalid.

```python
def _normalize_minimax_h3_best_of_k_count(value: Any) -> int:
    if isinstance(value, bool):
        raise CommandBuildError("MiniMax-H3 h3_best_of_k must be an integer of at least 1.")
    if isinstance(value, int):
        count = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        count = int(value)
    else:
        raise CommandBuildError("MiniMax-H3 h3_best_of_k must be an integer of at least 1.")
    if count < 1:
        raise CommandBuildError("MiniMax-H3 h3_best_of_k must be an integer of at least 1.")
    return count


def _normalize_minimax_h3_best_of_k_stream(value: Any) -> str:
    if not isinstance(value, str):
        raise CommandBuildError("MiniMax-H3 h3_best_of_k_stream must be video or audio.")
    stream = value.strip().lower()
    if stream not in {"video", "audio"}:
        raise CommandBuildError("MiniMax-H3 h3_best_of_k_stream must be video or audio.")
    return stream
```

In `_with_minimax_h3_defaults`, insert missing defaults, normalize both values, and write the canonical values into the copied `resolved` mapping before mapped arguments are emitted. Extend `_validate_minimax_h3_train_state` with the audio/video-only conflict and structured `xm_best_of_k` rejection.

- [ ] **Step 5: Implement raw text and final argv invariants**

Use only the three published/reserved names.

```python
H3_BEST_OF_K_RESERVED_CLI_OPTIONS = frozenset(
    {"--h3_best_of_k", "--h3_best_of_k_stream", "--xm_best_of_k"}
)


def _cli_option_name(token: Any) -> str:
    text = str(token).strip()
    return text.split("=", 1)[0].split(None, 1)[0]


def _validate_minimax_h3_reserved_cli_text(value: Any) -> None:
    for token in _parse_optimizer_args_text(value):
        option = _cli_option_name(token)
        if option in H3_BEST_OF_K_RESERVED_CLI_OPTIONS:
            raise CommandBuildError(
                f"MiniMax-H3 option {option} is reserved; use the structured Best-of-K controls."
            )


def _validate_minimax_h3_best_of_k_argv(args: list[str]) -> None:
    names = [_cli_option_name(token) for token in args]
    for option in ("--h3_best_of_k", "--h3_best_of_k_stream"):
        if names.count(option) != 1:
            raise CommandBuildError(
                f"MiniMax-H3 option {option} must occur exactly once from structured state."
            )
    if "--xm_best_of_k" in names:
        raise CommandBuildError(
            "MiniMax-H3 option --xm_best_of_k is not enabled; use h3_best_of_k."
        )
```

Scan `_parse_optimizer_args_text(state.get("optimizer_extra_args"))` before argv assembly. After `_add_train_optimizer_args`, count canonical occurrences in the completed H3 argv, require exactly one of each, and require zero `--xm_best_of_k`. Raise `CommandBuildError` before returning the `CommandJob`.

- [ ] **Step 6: Run focused and shared builder tests**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_command_builder.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the builder contract**

```powershell
git add -- gui/utils/command_builder.py gui/tests/test_minimax_h3_command_builder.py gui/tests/test_command_builder.py
git commit -m "feat: add H3 best-of-k command contract"
```

---

### Task 3: Add Testable PowerShell Best-Of-K Validation

**Files:**
- Create: `powershell/minimax_h3_best_of_k.ps1`
- Modify: `3.11minimax_h3_train_lora.ps1`
- Modify: `gui/tests/test_minimax_h3_scripts.py`

**Interfaces:**
- Consumes: `$optimizer_args`, `$ext_args`, `$video_only` in the existing H3 training script.
- Produces: `Resolve-H3BestOfKCount`, `Resolve-H3BestOfKStream`, `Assert-NoH3BestOfKReservedArguments`, and `Assert-H3BestOfKArgumentInvariant` PowerShell functions.

- [ ] **Step 1: Write failing source and executable helper tests**

Extend the source contract to require these defaults and exact arguments:

```python
self.assertIn("$h3_best_of_k = 1", train)
self.assertIn('$h3_best_of_k_stream = "video"', train)
self.assertIn('--h3_best_of_k=$h3_best_of_k', train)
self.assertIn('--h3_best_of_k_stream=$h3_best_of_k_stream', train)
self.assertIn("powershell/minimax_h3_best_of_k.ps1", train)
```

Add a subprocess helper that dot-sources only `powershell/minimax_h3_best_of_k.ps1`. Require count success for `[int]1`, `[long]2`, and `[string]'3'`; require failure for `[double]1.0`, `[double]1.5`, `0`, `-1`, `$true`, `''`, `'1e0'`, and `'word'`. Require stream normalization of `VIDEO` to `video` and reject empty/non-video/audio values. Test reserved raw strings in separated and equals forms.

- [ ] **Step 2: Run the script tests and verify they fail**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_scripts.py -k "best_of_k" -q
```

Expected: FAIL because the helper and script variables do not exist.

- [ ] **Step 3: Implement the PowerShell helper without truncating casts**

Implement count parsing with type checks plus `Int64.TryParse` and invariant culture. Floating CLR types and booleans must fail before conversion.

```powershell
function Resolve-H3BestOfKCount {
    param([Parameter(Mandatory = $true)][object]$Value)

    if ($Value -is [bool] -or $Value -is [single] -or $Value -is [double] -or $Value -is [decimal]) {
        throw "MiniMax-H3 h3_best_of_k must be a base-10 integer of at least 1."
    }
    $text = [Convert]::ToString($Value, [Globalization.CultureInfo]::InvariantCulture)
    [long]$count = 0
    $style = [Globalization.NumberStyles]::Integer
    if (-not [long]::TryParse($text, $style, [Globalization.CultureInfo]::InvariantCulture, [ref]$count) -or $count -lt 1) {
        throw "MiniMax-H3 h3_best_of_k must be a base-10 integer of at least 1."
    }
    return $count
}

function Resolve-H3BestOfKStream {
    param([Parameter(Mandatory = $true)][object]$Value)

    if ($Value -isnot [string]) {
        throw "MiniMax-H3 h3_best_of_k_stream must be video or audio."
    }
    $stream = $Value.Trim().ToLowerInvariant()
    if ($stream -notin @("video", "audio")) {
        throw "MiniMax-H3 h3_best_of_k_stream must be video or audio."
    }
    return $stream
}

function Assert-NoH3BestOfKReservedArguments {
    param([AllowNull()][object]$Arguments)

    $text = if ($Arguments -is [string]) {
        [string]$Arguments
    }
    else {
        (@($Arguments) | ForEach-Object { [string]$_ }) -join "`n"
    }
    $pattern = '(?m)(?<!\S)(--(?:h3_best_of_k(?:_stream)?|xm_best_of_k))(?=$|[\s=])'
    $match = [regex]::Match($text, $pattern)
    if ($match.Success) {
        throw "MiniMax-H3 option $($match.Groups[1].Value) is reserved; use the structured Best-of-K variables."
    }
}

function Assert-H3BestOfKArgumentInvariant {
    param([Parameter(Mandatory = $true)][object[]]$Arguments)

    $options = foreach ($argument in $Arguments) {
        $match = [regex]::Match(
            [string]$argument,
            '^\s*(--(?:h3_best_of_k(?:_stream)?|xm_best_of_k))(?=$|[\s=])'
        )
        if ($match.Success) { $match.Groups[1].Value }
    }
    foreach ($canonical in @('--h3_best_of_k', '--h3_best_of_k_stream')) {
        if (@($options | Where-Object { $_ -eq $canonical }).Count -ne 1) {
            throw "MiniMax-H3 option $canonical must occur exactly once from structured state."
        }
    }
    if ($options -contains '--xm_best_of_k') {
        throw "MiniMax-H3 option --xm_best_of_k is not enabled; use h3_best_of_k."
    }
}
```

The reserved set contains only `--h3_best_of_k`, `--h3_best_of_k_stream`, and `--xm_best_of_k`. Regex boundaries must recognize whitespace and equals separators. The final invariant requires one occurrence of each canonical option and no XM occurrence.

- [ ] **Step 4: Wire the helper into the training script**

Dot-source the helper next to `powershell/native_command.ps1`. Resolve K and stream before environment activation, reject active audio search with `video_only`, scan `$optimizer_args`, append the two canonical equals-form tokens once, and call the final invariant immediately before printing/launching the completed args.

- [ ] **Step 5: Run PowerShell helper, AST, and source tests**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_scripts.py gui/tests/test_powershell_failure_propagation.py -q
```

Expected: PASS under the available PowerShell executable.

- [ ] **Step 6: Commit the script contract**

```powershell
git add -- powershell/minimax_h3_best_of_k.ps1 3.11minimax_h3_train_lora.ps1 gui/tests/test_minimax_h3_scripts.py
git commit -m "feat: expose H3 best-of-k in PowerShell"
```

---

### Task 4: Add Native GUI Controls, Localization, And Architecture State Transitions

**Files:**
- Modify: `gui/wizard/step3_train.py`
- Modify: `gui/utils/model_catalog.py`
- Modify: `gui/utils/i18n.py`
- Modify: `gui/tests/test_minimax_h3_gui_contract.py`
- Modify: `gui/tests/test_model_catalog.py`

**Interfaces:**
- Consumes: canonical builder fields `h3_best_of_k: int` and `h3_best_of_k_stream: str`.
- Produces: `TrainStep.h3_best_of_k` NiceGUI number control, `TrainStep.h3_best_of_k_stream` select control, `_canonical_h3_best_of_k_ui_value(value: Any) -> Any`, and backing-config updates that survive dynamic scope disposal.

- [ ] **Step 1: Write failing render, localization, and catalog tests**

Add these required translation keys for every language:

```python
{
    "h3_best_of_k",
    "h3_best_of_k_tooltip",
    "h3_best_of_k_stream",
    "h3_best_of_k_stream_tooltip",
    "h3_best_of_k_stream_video",
    "h3_best_of_k_stream_audio",
}
```

Render the H3 train controls and assert K defaults to integer 1, its props contain `min=1` and `step=1`, stream defaults to `video`, and the model catalog lists both field names. Add round-trip assertions that setting K to `3.0` produces collected integer `3` and TOML serialization contains `h3_best_of_k = 3` rather than `3.0` or a quoted value.

- [ ] **Step 2: Write the failing H3 to non-H3 to H3 state-machine test**

```python
def test_h3_best_of_k_state_survives_architecture_round_trip(self):
    step = TrainStep()
    with ui.column() as container:
        step._model_path_container = ui.column()
        step._on_arch_change("MiniMax-H3", get_arch_info("MiniMax-H3"))
        step._write_control_value(step.h3_best_of_k, 4.0)
        step._write_control_value(step.h3_best_of_k_stream, "audio")
        step._on_arch_change("FLUX.2", get_arch_info("FLUX.2"))
        self.assertFalse(hasattr(step, "h3_best_of_k"))
        self.assertEqual(step.config["h3_best_of_k"], 4)
        self.assertEqual(step.config["h3_best_of_k_stream"], "audio")
        step._on_arch_change("MiniMax-H3", get_arch_info("MiniMax-H3"))
        self.assertEqual(step.h3_best_of_k.value, 4)
        self.assertEqual(step.h3_best_of_k_stream.value, "audio")
    step._clear_control_scope("model_paths")
    container.delete()
```

Add a separate `_apply_config` test proving canonical H3 fields loaded with a non-H3 architecture remain dormant in `step.config`.

- [ ] **Step 3: Run GUI contract tests and verify they fail**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_model_catalog.py -k "best_of_k" -q
```

Expected: FAIL because controls, keys, and catalog flags are missing.

- [ ] **Step 4: Implement the K number control and translated stream select**

Use `ui.number`, not an unbounded slider, and keep both controls inside the existing H3 model-path card.

```python
self.config.setdefault("h3_best_of_k", 1)
self._set_control(
    "h3_best_of_k",
    ui.number(
        label=t("h3_best_of_k"),
        value=self.config["h3_best_of_k"],
        min=1,
        step=1,
        on_change=lambda e: self._store_h3_best_of_k_value(e.value),
    ).classes("flex-1"),
    scope="model_paths",
)
```

Create the stream select with translated option labels and an `on_change` callback that writes the canonical string into `self.config`. Bind labels and option refresh through the existing dynamic translation binding helpers. Tooltips must state K=1 is off, K>1 increases compute, and one-frame batches always use image/video search.

- [ ] **Step 5: Preserve and normalize state across collection and architecture changes**

Implement the UI-only canonicalizer and backing-config callbacks as follows. Invalid fractions remain unchanged for the builder to reject rather than being rounded.

```python
@staticmethod
def _canonical_h3_best_of_k_ui_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _store_h3_best_of_k_value(self, value: Any) -> None:
    self.config["h3_best_of_k"] = self._canonical_h3_best_of_k_ui_value(value)


def _store_h3_best_of_k_stream(self, value: Any) -> None:
    self.config["h3_best_of_k_stream"] = (
        value.strip().lower() if isinstance(value, str) else value
    )
```

In `_get_config`, normalize the collected K again and mirror it into `self.config` before returning:

```python
def _get_config(self) -> Dict[str, Any]:
    state = self._collect_form_state()
    if "h3_best_of_k" in state:
        state["h3_best_of_k"] = self._canonical_h3_best_of_k_ui_value(
            state["h3_best_of_k"]
        )
        self.config["h3_best_of_k"] = state["h3_best_of_k"]
    return state
```

At the beginning of `_apply_config`, copy the two canonical H3 fields into `self.config` even when the loaded architecture is non-H3, normalizing K with the UI helper. In `_apply_minimax_h3_train_defaults`, initialize them separately with `self.config.setdefault("h3_best_of_k", 1)` and `self.config.setdefault("h3_best_of_k_stream", "video")`; do not include them in the bulk defaults that reset on architecture entry.

- [ ] **Step 6: Add four-language copy and catalog flags**

Add these exact English, Simplified Chinese, Japanese, and Korean entries, following each locale's existing dictionary placement:

```python
# en
'h3_best_of_k': 'Best of K',
'h3_best_of_k_tooltip': '1 disables candidate search. Values above 1 increase compute; one-frame image batches always search image/video noise.',
'h3_best_of_k_stream': 'Best-of-K Multi-Frame Stream',
'h3_best_of_k_stream_tooltip': 'Selects the ranked noise stream for multi-frame batches only.',
'h3_best_of_k_stream_video': 'Video',
'h3_best_of_k_stream_audio': 'Audio',

# zh
'h3_best_of_k': 'Best of K 候选数',
'h3_best_of_k_tooltip': '1 表示关闭候选搜索。大于 1 会增加计算量；单帧图像批次始终搜索图像/视频噪声。',
'h3_best_of_k_stream': 'Best of K 多帧流',
'h3_best_of_k_stream_tooltip': '仅选择多帧批次用于候选排序的噪声流。',
'h3_best_of_k_stream_video': '视频',
'h3_best_of_k_stream_audio': '音频',

# ja
'h3_best_of_k': 'Best of K 候補数',
'h3_best_of_k_tooltip': '1 で候補探索を無効化します。1 より大きい値は計算量が増え、1 フレーム画像バッチは常に画像/動画ノイズを探索します。',
'h3_best_of_k_stream': 'Best of K マルチフレームストリーム',
'h3_best_of_k_stream_tooltip': 'マルチフレームバッチで候補順位に使うノイズストリームのみを選択します。',
'h3_best_of_k_stream_video': '動画',
'h3_best_of_k_stream_audio': '音声',

# ko
'h3_best_of_k': 'Best of K 후보 수',
'h3_best_of_k_tooltip': '1은 후보 탐색을 끕니다. 1보다 큰 값은 연산량을 늘리며, 단일 프레임 이미지 배치는 항상 이미지/비디오 노이즈를 탐색합니다.',
'h3_best_of_k_stream': 'Best of K 다중 프레임 스트림',
'h3_best_of_k_stream_tooltip': '다중 프레임 배치에서 후보 순위에 사용할 노이즈 스트림만 선택합니다.',
'h3_best_of_k_stream_video': '비디오',
'h3_best_of_k_stream_audio': '오디오',
```

Add `h3_best_of_k` and `h3_best_of_k_stream` to the MiniMax-H3 train flag list only; do not add them to cache or generation pages.

- [ ] **Step 7: Run GUI, form-state, and catalog tests**

Run:

```powershell
python -m pytest gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_model_catalog.py gui/tests/test_form_state.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the GUI contract**

```powershell
git add -- gui/wizard/step3_train.py gui/utils/model_catalog.py gui/utils/i18n.py gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_model_catalog.py
git commit -m "feat: add H3 best-of-k GUI controls"
```

---

### Task 5: Reset Best-Of-K In Every Built-In H3 Training Preset

**Files:**
- Modify: `gui/presets/train/minimax_h3.toml`
- Modify: `gui/presets/train/minimax_h3_fl2va.toml`
- Modify: `gui/presets/train/minimax_h3_ref2va.toml`
- Modify: `gui/presets/train/minimax_h3_image.toml`
- Modify: `gui/tests/test_preset_scope_and_defaults.py`

**Interfaces:**
- Consumes: partial preset merge behavior.
- Produces: explicit integer `h3_best_of_k = 1` and enum `h3_best_of_k_stream = "video"` in every built-in H3 train preset.

- [ ] **Step 1: Write the failing preset reset test**

```python
def test_minimax_h3_train_presets_disable_best_of_k_by_default(self):
    manager = self.config_manager_module.ConfigManager()
    names = ("minimax_h3", "minimax_h3_fl2va", "minimax_h3_ref2va", "minimax_h3_image")
    for name in names:
        with self.subTest(preset=name):
            preset = manager.load_config("train", name)
            self.assertIs(type(preset["h3_best_of_k"]), int)
            self.assertEqual(preset["h3_best_of_k"], 1)
            self.assertEqual(preset["h3_best_of_k_stream"], "video")
            merged = {"h3_best_of_k": 8, "h3_best_of_k_stream": "audio", **preset}
            self.assertEqual(merged["h3_best_of_k"], 1)
            self.assertEqual(merged["h3_best_of_k_stream"], "video")
```

- [ ] **Step 2: Run the preset test and verify it fails**

Run:

```powershell
python -m pytest gui/tests/test_preset_scope_and_defaults.py -k "minimax_h3 and best_of_k" -q
```

Expected: FAIL with missing preset keys.

- [ ] **Step 3: Add the exact defaults to all four TOML presets**

Place these beside the H3 flow/loss fields in each file:

```toml
h3_best_of_k = 1
h3_best_of_k_stream = "video"
```

- [ ] **Step 4: Run preset and command round-trip tests**

Run:

```powershell
python -m pytest gui/tests/test_preset_scope_and_defaults.py gui/tests/test_minimax_h3_command_builder.py -q
```

Expected: PASS, with every preset-built H3 command containing K=1 and stream=video once.

- [ ] **Step 5: Commit the preset resets**

```powershell
git add -- gui/presets/train/minimax_h3.toml gui/presets/train/minimax_h3_fl2va.toml gui/presets/train/minimax_h3_ref2va.toml gui/presets/train/minimax_h3_image.toml gui/tests/test_preset_scope_and_defaults.py
git commit -m "feat: default H3 best-of-k to one"
```

---

### Task 6: Verify Focused Tests, Full Suite, Clean Checkout, And Responsive GUI

**Files:**
- Verify only; do not modify unrelated user files to make baseline failures disappear.

**Interfaces:**
- Consumes: the committed feature branch and full target gitlink.
- Produces: focused/full test evidence, clean-checkout gitlink evidence, and desktop/mobile screenshots confirming the controls are usable.

- [ ] **Step 1: Run focused H3 coverage**

```powershell
python -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_preset_scope_and_defaults.py gui/tests/test_model_catalog.py gui/tests/test_form_state.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete GUI suite and classify unrelated failures**

```powershell
python -m pytest gui/tests -q
```

Expected: all feature-owned tests pass. If existing failures point only to user-owned untracked scripts or modified TOML/prompt files, record them verbatim and do not edit those files.

- [ ] **Step 3: Verify the parent tree and disposable clean checkout**

Resolve an absolute temporary path, confirm it is under `$env:TEMP`, then create a detached worktree and initialize submodules.

```powershell
$clean = Join-Path $env:TEMP ("musubi-h3-best-of-k-" + [guid]::NewGuid().ToString("N"))
$resolvedTemp = [IO.Path]::GetFullPath($env:TEMP)
$resolvedClean = [IO.Path]::GetFullPath($clean)
if (-not $resolvedClean.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe clean-checkout path" }
git worktree add --detach $resolvedClean HEAD
git -C $resolvedClean submodule update --init --recursive
git -C $resolvedClean ls-tree HEAD musubi-tuner
git -C (Join-Path $resolvedClean "musubi-tuner") rev-parse HEAD
git -C (Join-Path $resolvedClean "musubi-tuner") status --porcelain
```

Expected: both SHAs equal `c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`, submodule status is empty, and the clean checkout's trainer source contains both canonical `parser.add_argument` declarations. Remove the verified worktree with `git worktree remove $resolvedClean` only after rechecking the resolved path remains under `$env:TEMP`.

- [ ] **Step 4: Start the GUI on an unused local port**

```powershell
powershell -ExecutionPolicy Bypass -File .\1.6.GUI.ps1 -Port 7791 -BindHost 127.0.0.1 -NoBrowser -NoPause
```

Keep the process running until browser verification finishes. If 7791 is occupied, increment to an unused port and record the selected URL.

- [ ] **Step 5: Verify desktop and mobile layouts with Playwright**

Open the local GUI, navigate to the training step, select MiniMax-H3, and inspect at 1440x1000 and 390x844. Assert the K stepper and stream selector are visible, K starts at 1, changing K to 2 persists, labels do not overlap, and no horizontal overflow appears. Save screenshots outside the repository or in an ignored temporary directory.

- [ ] **Step 6: Run final diff and ownership checks**

```powershell
git diff --check main...HEAD
git status --short
git log --oneline --decorate main..HEAD
```

Expected: feature commits contain only the design/plan, gitlink, H3 script/helper, GUI/builder/catalog/i18n, four H3 presets, and their tests. User-owned dirty files remain unstaged and unchanged.
