# MiniMax-H3 Default CLI Omission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep MiniMax-H3 preset and GUI state explicit while omitting train CLI arguments whose normalized values equal defaults in the pinned `musubi-tuner` parser.

**Architecture:** Build the existing canonical MiniMax-H3 argument list, then run one H3-only typed serialization pass before the final Best-of-K invariant check. PowerShell keeps readable variables and uses conditional argument construction plus a small numeric comparison helper; source-lock tests read parser defaults from the submodule commit stored in the parent repository's committed `HEAD` tree without importing training dependencies.

**Tech Stack:** Python 3.11, `unittest`/`pytest`, Python `ast` and `decimal.Decimal`, PowerShell 5+/7, Git submodules.

## Global Constraints

- The parent `musubi-tuner` gitlink must remain `c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`.
- GUI/TOML state remains explicit; omission happens only while serializing MiniMax-H3 train commands.
- Cache, generation, and every non-H3 train command remain unchanged.
- Only valid non-default values reappear; `timestep_sampling`, `weighting_scheme`, and `discrete_flow_shift` remain fixed H3 validation errors when changed.
- `task`, required paths, output paths, `dit_dtype`, `mixed_precision`, and the selected attention backend remain explicit.
- Best-of-K accepts only two output shapes: neither canonical option for `1/video`, or one occurrence of both canonical options for a non-default pair.
- Structured Best-of-K options remain reserved in optimizer extra arguments; `--xm_best_of_k` remains disabled for H3.
- Removed unpublished aliases are neither translated nor reintroduced.
- Existing user-modified and untracked files must not be staged or rewritten.
- Implementation follows red-green-refactor and commits each independently reviewable task.

## File Structure

- Modify `gui/utils/command_builder.py`: own the typed MiniMax-H3 default table, H3-only CLI compaction, and state-aware Best-of-K invariant.
- Modify `gui/tests/test_minimax_h3_command_builder.py`: cover typed omission, valid non-default restoration, every built-in H3 preset, parent-tree parser defaults, and gitlink cleanliness.
- Create `powershell/minimax_h3_train_defaults.ps1`: provide invariant-culture numeric default comparison for the standalone H3 train script.
- Modify `powershell/minimax_h3_best_of_k.ps1`: accept either an absent canonical pair or one complete pair.
- Modify `3.11minimax_h3_train_lora.ps1`: move default-sensitive direct arguments into conditional argument construction.
- Modify `gui/tests/test_minimax_h3_scripts.py`: execute PowerShell helper contracts and verify the train script has no default-sensitive direct arguments.
- Do not modify any `gui/presets/train/minimax_h3*.toml` file; tests prove their explicit values remain intact.

---

### Task 1: Omit MiniMax-H3-Specific Defaults In The Python Builder

**Files:**
- Modify: `gui/utils/command_builder.py:27-40`
- Modify: `gui/utils/command_builder.py:903-983`
- Modify: `gui/utils/command_builder.py:1909-1932`
- Modify: `gui/utils/command_builder.py:2555-2570`
- Test: `gui/tests/test_minimax_h3_command_builder.py:322-468`
- Test: `gui/tests/test_minimax_h3_command_builder.py:636-715`

**Interfaces:**
- Consumes: normalized H3 state returned by `_with_minimax_h3_defaults(state) -> dict[str, Any]` and canonical `--name=value` tokens produced by existing builders.
- Produces: `MINIMAX_H3_TRAIN_DEFAULTS: dict[str, tuple[str, str]]`, `MINIMAX_H3_BEST_OF_K_DEFAULT: tuple[int, str]`, `_omit_minimax_h3_train_default_args(args: list[str], state: Mapping[str, Any]) -> None`, and `_validate_minimax_h3_best_of_k_argv(args: Iterable[str], state: Mapping[str, Any]) -> None`.
- Invariant: the omission helper mutates only the local `args` list and never mutates `state`.

- [ ] **Step 1: Change Best-of-K and H3-specific tests to the desired behavior**

Replace `test_h3_best_of_k_defaults_are_emitted_once` and add focused helpers/tests:

```python
def _arguments_for_option(arguments: list[str], option: str) -> list[str]:
    return [
        argument
        for argument in arguments
        if argument == option or argument.startswith(f"{option}=")
    ]


def test_h3_best_of_k_defaults_are_omitted(self):
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(_h3_train_state(), tmp, PROJECT_CONFIG)

    self.assertEqual(_arguments_for_option(job.args, "--h3_best_of_k"), [])
    self.assertEqual(
        _arguments_for_option(job.args, "--h3_best_of_k_stream"),
        [],
    )


def test_h3_specific_parser_defaults_are_omitted(self):
    state = _h3_train_state(
        timestep_sampling=" UNIFORM ",
        weighting_scheme=" UNIFORM ",
        h3_shift_video=12,
        h3_shift_audio=3.0,
        h3_visual_cond_clean=0.999,
        h3_audio_cond_clean=1,
        audio_loss_weight=1.0,
        convrot_int8_bwd=" BF16 ",
        h3_guidance_loss_scale=0,
        h3_guidance_loss_sigma_min=0.0,
        enable_sample=True,
        sample_prompts="toml/qinglong_minimaxh3.txt",
        text_encoder_blocks_to_swap=0,
        text_encoder_attn_mode="flash_attention_2",
        attn_mode="sdpa",
    )
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(state, tmp, PROJECT_CONFIG)

    omitted = {
        "--network_module",
        "--timestep_sampling",
        "--weighting_scheme",
        "--discrete_flow_shift",
        "--h3_shift_video",
        "--h3_shift_audio",
        "--h3_visual_cond_clean",
        "--h3_audio_cond_clean",
        "--audio_loss_weight",
        "--convrot_int8_bwd",
        "--h3_guidance_loss_scale",
        "--h3_guidance_loss_sigma_min",
        "--text_encoder_blocks_to_swap",
    }
    for option in omitted:
        with self.subTest(option=option):
            self.assertEqual(_arguments_for_option(job.args, option), [])

    for retained in (
        "--task=t2va",
        "--dit_dtype=bfloat16",
        "--mixed_precision=bf16",
        "--sdpa",
        "--text_encoder_attn_mode=flash_attention_2",
    ):
        self.assertIn(retained, job.args)
```

Add one test proving valid non-default H3 values reappear:

```python
def test_h3_valid_nondefault_values_are_emitted(self):
    state = _h3_train_state(
        h3_best_of_k=2,
        h3_best_of_k_stream="audio",
        h3_shift_video=11.5,
        h3_shift_audio=2.5,
        h3_visual_cond_clean=0.998,
        h3_audio_cond_clean=0.9,
        audio_loss_weight=0.75,
        h3_guidance_loss_scale=4.0,
        h3_guidance_loss_sigma_min=0.15,
        h3_guidance_loss_uncond_cache="cache/h3_uncond.safetensors",
        enable_sample=True,
        sample_prompts="toml/qinglong_minimaxh3.txt",
        text_encoder_blocks_to_swap=1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(state, tmp, PROJECT_CONFIG)

    expected = (
        "--h3_best_of_k=2",
        "--h3_best_of_k_stream=audio",
        "--h3_shift_video=11.5",
        "--h3_shift_audio=2.5",
        "--h3_visual_cond_clean=0.998",
        "--h3_audio_cond_clean=0.9",
        "--audio_loss_weight=0.75",
        "--h3_guidance_loss_scale=4.0",
        "--h3_guidance_loss_sigma_min=0.15",
        "--text_encoder_blocks_to_swap=1",
    )
    for argument in expected:
        self.assertEqual(job.args.count(argument), 1, argument)
```

In `test_sampled_train_adds_future_joint_av_sampling_dependencies`, move these
default arguments out of the expected-present tuple and assert their options
are absent:

```python
for option in (
    "--network_module",
    "--h3_shift_video",
    "--h3_shift_audio",
    "--h3_visual_cond_clean",
    "--h3_audio_cond_clean",
    "--convrot_int8_bwd",
):
    self.assertEqual(_arguments_for_option(job.args, option), [])
```

Keep `--audio_loss_weight=0.75`, `--block_swap_ring_size=2`, and every other
non-default dependency in the expected-present tuple during Task 1. Task 2
adds the shared ring-size default and changes that expectation.

- [ ] **Step 2: Run the new Python tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_command_builder.py -k "best_of_k_defaults or specific_parser_defaults or valid_nondefault_values or sampled_train" -q
```

Expected: FAIL because the current builder emits the default network module,
H3 scalar values, default Best-of-K pair, default text-encoder block count,
and canonical fixed timestep/weighting values.

- [ ] **Step 3: Add the typed H3-specific default contract and comparison helper**

Near the existing H3 constants, add:

```python
MINIMAX_H3_BEST_OF_K_DEFAULT = (1, "video")
MINIMAX_H3_TRAIN_DEFAULTS: dict[str, tuple[str, str]] = {
    "--network_module": ("string", "networks.lora_minimax_h3"),
    "--timestep_sampling": ("string", "uniform"),
    "--weighting_scheme": ("string", "none"),
    "--discrete_flow_shift": ("number", "1.0"),
    "--h3_shift_video": ("number", "12.0"),
    "--h3_shift_audio": ("number", "3.0"),
    "--h3_visual_cond_clean": ("number", "0.999"),
    "--h3_audio_cond_clean": ("number", "1.0"),
    "--audio_loss_weight": ("number", "1.0"),
    "--convrot_int8_bwd": ("string", "bf16"),
    "--h3_guidance_loss_scale": ("number", "0.0"),
    "--h3_guidance_loss_sigma_min": ("number", "0.0"),
    "--text_encoder_blocks_to_swap": ("int", "0"),
}
```

Add the exact typed comparison and compaction functions near the existing H3
normalizers:

```python
def _minimax_h3_cli_value_matches_default(
    value: str,
    kind: str,
    expected: str,
) -> bool:
    text = value.strip()
    if kind == "string":
        return text == expected
    if kind == "int":
        digits = text[1:] if text[:1] in {"+", "-"} else text
        return bool(digits) and digits.isdigit() and int(text) == int(expected)
    if kind == "number":
        try:
            actual = Decimal(text)
            default = Decimal(expected)
        except (InvalidOperation, ValueError):
            return False
        return actual.is_finite() and actual == default
    raise ValueError(f"Unsupported MiniMax-H3 CLI default kind: {kind}")


def _omit_minimax_h3_train_default_args(
    args: list[str],
    state: Mapping[str, Any],
) -> None:
    default_best_of_k = (
        state["h3_best_of_k"],
        state["h3_best_of_k_stream"],
    ) == MINIMAX_H3_BEST_OF_K_DEFAULT
    compacted: list[str] = []
    for argument in args:
        option, separator, value = str(argument).partition("=")
        if default_best_of_k and option in {
            "--h3_best_of_k",
            "--h3_best_of_k_stream",
        }:
            continue
        default = MINIMAX_H3_TRAIN_DEFAULTS.get(option)
        if (
            separator
            and default is not None
            and _minimax_h3_cli_value_matches_default(value, *default)
        ):
            continue
        compacted.append(argument)
    args[:] = compacted
```

Extend `_with_minimax_h3_defaults` so known H3 enum strings are canonical
before validation and serialization:

```python
for key in ("timestep_sampling", "weighting_scheme", "convrot_int8_bwd"):
    if _has_value(resolved.get(key)):
        resolved[key] = str(resolved[key]).strip().lower()
```

- [ ] **Step 4: Invoke compaction before a state-aware Best-of-K invariant**

At the end of `build_train_job`, replace the current H3 invariant call with:

```python
if arch_name == MINIMAX_H3_ARCH:
    _omit_minimax_h3_train_default_args(args, state)
    _validate_minimax_h3_best_of_k_argv(args, state)
```

Change the invariant to distinguish the default absent pair from a custom
complete pair:

```python
def _validate_minimax_h3_best_of_k_argv(
    args: Iterable[str],
    state: Mapping[str, Any],
) -> None:
    option_counts = {option: 0 for option in MINIMAX_H3_BEST_OF_K_RESERVED_OPTIONS}
    for token in args:
        option = str(token).split("=", 1)[0]
        if option in option_counts:
            option_counts[option] += 1

    if option_counts["--xm_best_of_k"]:
        raise CommandBuildError("MiniMax-H3 must not emit --xm_best_of_k.")

    default_pair = (
        state["h3_best_of_k"],
        state["h3_best_of_k_stream"],
    ) == MINIMAX_H3_BEST_OF_K_DEFAULT
    expected_count = 0 if default_pair else 1
    for option in ("--h3_best_of_k", "--h3_best_of_k_stream"):
        if option_counts[option] != expected_count:
            shape = "be omitted" if default_pair else "occur exactly once"
            raise CommandBuildError(f"MiniMax-H3 option {option} must {shape}.")
```

- [ ] **Step 5: Run H3 builder tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_command_builder.py -q
```

Expected: PASS. In particular, invalid Best-of-K values and reserved raw
options must still fail before command construction completes.

- [ ] **Step 6: Commit the Python H3-specific behavior**

```powershell
git add -- gui/utils/command_builder.py gui/tests/test_minimax_h3_command_builder.py
git commit -m "feat: omit default H3 train arguments"
```

---

### Task 2: Cover Shared Defaults, Built-In Presets, And Indexed Upstream Sources

**Files:**
- Modify: `gui/utils/command_builder.py:31-55`
- Modify: `gui/tests/test_minimax_h3_command_builder.py:75-150`
- Modify: `gui/tests/test_minimax_h3_command_builder.py:269-322`
- Modify: `gui/tests/test_minimax_h3_command_builder.py:636-715`

**Interfaces:**
- Consumes: `MINIMAX_H3_TRAIN_DEFAULTS` and `_omit_minimax_h3_train_default_args` from Task 1.
- Produces: a complete shared-default extension of the same table, test-only `_parser_defaults(source: str) -> dict[str, Any]`, and `_parent_tree_submodule_source(relative_path: str) -> str`.
- Source contract: parser defaults are read from the full gitlink commit in the parent `HEAD` tree, not from the index or an arbitrary submodule working-tree file.

- [ ] **Step 1: Add failing shared-default serialization tests**

Add:

```python
def test_h3_shared_parser_defaults_are_omitted(self):
    state = _h3_train_state(
        max_train_steps=1600,
        max_data_loader_n_workers=8.0,
        gradient_accumulation_steps=1.0,
        guidance_scale=1.0,
        learning_rate="2e-6",
        max_grad_norm=1,
        lr_scheduler="constant",
        lr_warmup_steps=0,
        lr_decay_steps=0,
        lr_scheduler_num_cycles=1.0,
        lr_scheduler_power=1.0,
        network_dim=4,
        network_alpha=1,
        gradient_checkpointing=True,
        blocks_to_swap=1,
        block_swap_h2d_only=True,
        block_swap_ring_size=2.0,
        compile=True,
        compile_backend="inductor",
        compile_mode="default",
    )
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(state, tmp, PROJECT_CONFIG)

    for option in (
        "--max_train_steps",
        "--max_data_loader_n_workers",
        "--gradient_accumulation_steps",
        "--guidance_scale",
        "--learning_rate",
        "--max_grad_norm",
        "--lr_scheduler",
        "--lr_warmup_steps",
        "--lr_decay_steps",
        "--lr_scheduler_num_cycles",
        "--lr_scheduler_power",
        "--network_alpha",
        "--block_swap_ring_size",
        "--compile_backend",
        "--compile_mode",
    ):
        with self.subTest(option=option):
            self.assertEqual(_arguments_for_option(job.args, option), [])

    for retained in (
        "--compile",
        "--gradient_checkpointing",
        "--blocks_to_swap=1",
        "--block_swap_h2d_only",
        "--network_dim=4",
    ):
        self.assertIn(retained, job.args)


def test_h3_integer_defaults_reject_fractional_and_boolean_values(self):
    invalid = (
        ("max_train_steps", 1600.5),
        ("max_data_loader_n_workers", True),
        ("gradient_accumulation_steps", 1.5),
        ("lr_scheduler_num_cycles", 1.5),
        ("block_swap_ring_size", 2.5),
    )
    for key, value in invalid:
        with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
            state = _h3_train_state(**{key: value})
            if key == "block_swap_ring_size":
                state.update(
                    gradient_checkpointing=True,
                    blocks_to_swap=1,
                    block_swap_h2d_only=True,
                )
            with self.assertRaisesRegex(CommandBuildError, rf"{key}.*integer"):
                build_train_job(state, tmp, PROJECT_CONFIG)


def test_h3_shared_nondefault_values_are_emitted(self):
    state = _h3_train_state(
        max_train_steps=1601,
        max_data_loader_n_workers=4,
        gradient_accumulation_steps=2,
        guidance_scale=1.25,
        learning_rate="1e-4",
        max_grad_norm=0.5,
        lr_scheduler="polynomial",
        lr_warmup_steps=10,
        lr_decay_steps=0.2,
        lr_scheduler_num_cycles=2,
        lr_scheduler_power=2.0,
        network_dim=4,
        network_alpha=2,
        gradient_checkpointing=True,
        blocks_to_swap=1,
        block_swap_h2d_only=True,
        block_swap_ring_size=1,
        compile=True,
        compile_backend="aot_eager",
        compile_mode="reduce-overhead",
    )
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(state, tmp, PROJECT_CONFIG)

    expected_options = {
        "--max_train_steps",
        "--max_data_loader_n_workers",
        "--gradient_accumulation_steps",
        "--guidance_scale",
        "--learning_rate",
        "--max_grad_norm",
        "--lr_scheduler",
        "--lr_warmup_steps",
        "--lr_decay_steps",
        "--lr_scheduler_num_cycles",
        "--lr_scheduler_power",
        "--network_alpha",
        "--block_swap_ring_size",
        "--compile_backend",
        "--compile_mode",
    }
    for option in expected_options:
        with self.subTest(option=option):
            self.assertEqual(len(_arguments_for_option(job.args, option)), 1)


def test_shared_default_omission_is_h3_only(self):
    state = {
        "arch": "FLUX.2",
        "version": "klein-base-4b",
        "dit_path": "ckpts/flux2.safetensors",
        "vae_path": "ckpts/ae.safetensors",
        "text_encoder_path": "ckpts/qwen3.safetensors",
        "train_mode": "lora",
        "mixed_precision": "bf16",
        "max_train_steps": 1600,
        "max_data_loader_n_workers": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": "2e-6",
        "max_grad_norm": 1,
        "lr_scheduler": "constant",
        "lr_scheduler_num_cycles": 1,
        "network_dim": 4,
        "network_alpha": 1,
        "optimizer_type": "AdamW8bit",
    }
    with tempfile.TemporaryDirectory() as tmp:
        job = build_train_job(state, tmp, PROJECT_CONFIG)

    for option in (
        "--max_train_steps",
        "--max_data_loader_n_workers",
        "--gradient_accumulation_steps",
        "--learning_rate",
        "--max_grad_norm",
        "--lr_scheduler",
        "--lr_scheduler_num_cycles",
        "--network_alpha",
    ):
        with self.subTest(option=option):
            self.assertEqual(len(_arguments_for_option(job.args, option)), 1)
```

- [ ] **Step 2: Run the shared-default tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_command_builder.py -k "shared_parser_defaults or shared_nondefault_values or integer_defaults" -q
```

Expected: the default test FAILS because the builder still emits at least
`--max_train_steps=1600`, `--max_data_loader_n_workers=8`,
`--gradient_accumulation_steps=1`, `--learning_rate=2e-6`,
`--max_grad_norm=1`, `--lr_scheduler=constant`, `--network_alpha=1`,
`--block_swap_ring_size=2`, `--compile_backend=inductor`, and
`--compile_mode=default`. Fractional/boolean integer tests also FAIL because
generic integer builders currently coerce some invalid values.

Also update `test_sampled_train_adds_future_joint_av_sampling_dependencies`
at this stage: remove `--block_swap_ring_size=2` from the expected-present
tuple and assert `_arguments_for_option(job.args, "--block_swap_ring_size")`
is empty.

- [ ] **Step 3: Extend the typed table with the exact shared parser defaults**

Add these entries to `MINIMAX_H3_TRAIN_DEFAULTS`:

```python
    "--max_train_steps": ("int", "1600"),
    "--max_data_loader_n_workers": ("int", "8"),
    "--gradient_accumulation_steps": ("int", "1"),
    "--guidance_scale": ("number", "1.0"),
    "--learning_rate": ("number", "2e-6"),
    "--max_grad_norm": ("number", "1.0"),
    "--lr_scheduler": ("string", "constant"),
    "--lr_warmup_steps": ("number", "0"),
    "--lr_decay_steps": ("number", "0"),
    "--lr_scheduler_num_cycles": ("int", "1"),
    "--lr_scheduler_power": ("number", "1.0"),
    "--network_alpha": ("number", "1.0"),
    "--block_swap_ring_size": ("int", "2"),
    "--compile_backend": ("string", "inductor"),
    "--compile_mode": ("string", "default"),
```

Normalize shared integer values on the copied H3 state inside
`_with_minimax_h3_defaults`, before a generic helper can truncate them:

```python
for key in (
    "max_train_steps",
    "max_data_loader_n_workers",
    "gradient_accumulation_steps",
    "lr_scheduler_num_cycles",
    "block_swap_ring_size",
):
    if _has_value(resolved.get(key)):
        resolved[key] = _minimax_h3_integer(resolved[key], key, 0)
```

Do not add `dit_dtype`, `mixed_precision`, attention flags, task, paths,
output/logging arguments, optimizer type, `network_dim`, or feature-enabling
booleans.

- [ ] **Step 4: Add AST source-lock and clean-submodule tests**

Import `MINIMAX_H3_BEST_OF_K_DEFAULT` and
`MINIMAX_H3_TRAIN_DEFAULTS` in the test module. Replace the current
`_indexed_submodule_source` helper and all its call sites with a helper that
starts from the committed parent tree:

```python
def _parent_tree_submodule_source(relative_path: str) -> str:
    commit = _parent_tree_submodule_commit()
    return subprocess.run(
        [
            "git",
            "-C",
            str(ROOT / "musubi-tuner"),
            "show",
            f"{commit}:{relative_path}",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
```

Add a source parser that
handles both `parser.add_argument(..., default=...)` and
`parser.set_defaults(...)`:

```python
def _parser_defaults(source: str) -> dict[str, object]:
    defaults: dict[str, object] = {}
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "set_defaults":
            for keyword in node.keywords:
                if keyword.arg is not None:
                    defaults[f"--{keyword.arg}"] = ast.literal_eval(keyword.value)
            continue
        if node.func.attr != "add_argument":
            continue
        options = [
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and argument.value.startswith("--")
        ]
        default_node = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "default"),
            None,
        )
        if not options or default_node is None:
            continue
        try:
            default = ast.literal_eval(default_node)
        except (ValueError, TypeError):
            continue
        for option in options:
            defaults[option] = default
    return defaults
```

Add exact source ownership and typed assertions:

```python
def test_h3_default_contract_matches_indexed_upstream_parsers(self):
    source_options = {
        "src/musubi_tuner/minimax_h3_train_network.py": {
            "--network_module",
            "--timestep_sampling",
            "--weighting_scheme",
            "--discrete_flow_shift",
            "--h3_shift_video",
            "--h3_shift_audio",
            "--h3_visual_cond_clean",
            "--h3_audio_cond_clean",
            "--convrot_int8_bwd",
            "--h3_guidance_loss_scale",
            "--h3_guidance_loss_sigma_min",
            "--text_encoder_blocks_to_swap",
        },
        "src/musubi_tuner/training/audio_loss.py": {
            "--audio_loss_weight",
        },
        "src/musubi_tuner/training/parser_common.py": {
            "--max_train_steps",
            "--max_data_loader_n_workers",
            "--gradient_accumulation_steps",
            "--guidance_scale",
            "--learning_rate",
            "--max_grad_norm",
            "--lr_scheduler",
            "--lr_warmup_steps",
            "--lr_decay_steps",
            "--lr_scheduler_num_cycles",
            "--lr_scheduler_power",
            "--network_alpha",
            "--block_swap_ring_size",
            "--compile_backend",
            "--compile_mode",
        },
    }
    for source_path, options in source_options.items():
        defaults = _parser_defaults(_parent_tree_submodule_source(source_path))
        for option in options:
            with self.subTest(source=source_path, option=option):
                kind, expected = MINIMAX_H3_TRAIN_DEFAULTS[option]
                actual = defaults[option]
                if kind == "int":
                    self.assertIs(type(actual), int)
                    self.assertEqual(actual, int(expected))
                elif kind == "number":
                    self.assertEqual(Decimal(str(actual)), Decimal(expected))
                else:
                    self.assertEqual(actual, expected)

    h3_defaults = _parser_defaults(
        _parent_tree_submodule_source(
            "src/musubi_tuner/minimax_h3_train_network.py"
        )
    )
    self.assertEqual(
        (
            h3_defaults["--h3_best_of_k"],
            h3_defaults["--h3_best_of_k_stream"],
        ),
        MINIMAX_H3_BEST_OF_K_DEFAULT,
    )


def test_h3_submodule_head_matches_parent_gitlink_and_is_clean(self):
    head = subprocess.run(
        ["git", "-C", str(ROOT / "musubi-tuner"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(ROOT / "musubi-tuner"), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    self.assertEqual(_parent_tree_submodule_commit(), H3_SUBMODULE_TARGET_SHA)
    self.assertEqual(head, H3_SUBMODULE_TARGET_SHA)
    self.assertEqual(status, "")
```

Add `from decimal import Decimal` to the test imports.

- [ ] **Step 5: Add one command test for every built-in H3 train preset**

Add:

```python
def test_all_h3_train_presets_omit_values_equal_to_upstream_defaults(self):
    preset_dir = ROOT / "gui" / "presets" / "train"
    preset_paths = sorted(preset_dir.glob("minimax_h3*.toml"))
    self.assertEqual(len(preset_paths), 6)

    state_defaults = (
        ("h3_shift_video", "--h3_shift_video", 12.0),
        ("h3_shift_audio", "--h3_shift_audio", 3.0),
        ("h3_visual_cond_clean", "--h3_visual_cond_clean", 0.999),
        ("h3_audio_cond_clean", "--h3_audio_cond_clean", 1.0),
        ("audio_loss_weight", "--audio_loss_weight", 1.0),
        ("convrot_int8_bwd", "--convrot_int8_bwd", "bf16"),
        ("h3_guidance_loss_scale", "--h3_guidance_loss_scale", 0.0),
        ("h3_guidance_loss_sigma_min", "--h3_guidance_loss_sigma_min", 0.0),
        ("gradient_accumulation_steps", "--gradient_accumulation_steps", 1),
        ("max_data_loader_n_workers", "--max_data_loader_n_workers", 8),
        ("max_grad_norm", "--max_grad_norm", 1.0),
        ("lr_scheduler_num_cycles", "--lr_scheduler_num_cycles", 1),
        ("network_dropout", "--network_dropout", 0),
        ("text_encoder_blocks_to_swap", "--text_encoder_blocks_to_swap", 0),
    )

    for preset_path in preset_paths:
        with preset_path.open("rb") as handle:
            preset = tomllib.load(handle)
        for key, _, _ in state_defaults:
            self.assertIn(key, preset, f"{preset_path.name}: {key}")
        project_config = IMAGE_PROJECT_CONFIG if preset.get("one_frame") else PROJECT_CONFIG
        with self.subTest(preset=preset_path.name), tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(preset, tmp, project_config)
            self.assertEqual(
                _arguments_for_option(job.args, "--network_module"),
                [],
            )
            self.assertEqual(
                _arguments_for_option(job.args, "--timestep_sampling"),
                [],
            )
            self.assertEqual(
                _arguments_for_option(job.args, "--weighting_scheme"),
                [],
            )
            self.assertEqual(
                _arguments_for_option(job.args, "--h3_best_of_k"),
                [],
            )
            self.assertEqual(
                _arguments_for_option(job.args, "--h3_best_of_k_stream"),
                [],
            )
            for removed in (
                "--h3_video_best_of_k",
                "--h3_audio_best_of_k",
                "--h3_image_best_of_k",
                "--xm_best_of_k",
            ):
                self.assertEqual(_arguments_for_option(job.args, removed), [])
            for key, option, default in state_defaults:
                if preset.get(key) == default:
                    self.assertEqual(_arguments_for_option(job.args, option), [])
                elif key in preset and key != "network_dropout":
                    self.assertEqual(len(_arguments_for_option(job.args, option)), 1)
```

This test reads presets only. Do not edit them to make the test pass.

- [ ] **Step 6: Run the complete H3 builder test file**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_command_builder.py -q
```

Expected: PASS, including the full 40-character gitlink, indexed AST defaults,
clean submodule, all six H3 presets, default omission, and non-default
restoration.

- [ ] **Step 7: Commit shared defaults and source-lock coverage**

```powershell
git add -- gui/utils/command_builder.py gui/tests/test_minimax_h3_command_builder.py
git commit -m "test: lock H3 CLI defaults to upstream"
```

---

### Task 3: Synchronize The PowerShell Train Script

**Files:**
- Create: `powershell/minimax_h3_train_defaults.ps1`
- Modify: `powershell/minimax_h3_best_of_k.ps1:65-94`
- Modify: `3.11minimax_h3_train_lora.ps1:98-363`
- Modify: `gui/tests/test_minimax_h3_scripts.py:10-18`
- Modify: `gui/tests/test_minimax_h3_scripts.py:106-285`
- Modify: `gui/tests/test_minimax_h3_scripts.py:347-365`

**Interfaces:**
- Produces: `Test-H3NumericDefault -Value <object> -DefaultValue <decimal> -> bool`.
- Consumes: normalized values from `Resolve-H3BestOfKCount` and `Resolve-H3BestOfKStream`.
- Best-of-K invariant accepts `@()` or one complete canonical pair and rejects every other shape.
- The train script still launches the same upstream module with the same effective default configuration.

- [ ] **Step 1: Add failing PowerShell helper and script contract tests**

Add the new helper path and include it in `test_scripts_exist` and the
PowerShell AST test:

```python
TRAIN_DEFAULTS_HELPER = ROOT / "powershell" / "minimax_h3_train_defaults.ps1"
```

Add an executable helper test:

```python
def test_train_default_helper_compares_numeric_values_exactly(self):
    for expression in ("12", "[double]12.0", "'12.000'"):
        with self.subTest(expression=expression):
            result = self.run_train_defaults_helper(
                f"Test-H3NumericDefault ({expression}) ([decimal]12)"
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "True")

    nearby = self.run_train_defaults_helper(
        "Test-H3NumericDefault ([decimal]0.9991) ([decimal]0.999)"
    )
    self.assertEqual(nearby.returncode, 0, nearby.stderr)
    self.assertEqual(nearby.stdout.strip(), "False")

    for expression in ("$true", "''", "'word'", "'NaN'"):
        with self.subTest(expression=expression):
            result = self.run_train_defaults_helper(
                f"Test-H3NumericDefault ({expression}) ([decimal]1)"
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("finite number", result.stderr)
```

Extend the Best-of-K invariant test so both valid shapes are required:

```python
for arguments in (
    "@()",
    "@('--h3_best_of_k=2', '--h3_best_of_k_stream=video')",
):
    with self.subTest(arguments=arguments):
        result = self.run_best_of_k_helper(
            f"Assert-H3BestOfKArgumentInvariant {arguments}"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
```

Add a train-script guard test:

```python
def test_training_guards_default_sensitive_arguments(self):
    train = self.read_script(self.TRAIN)
    guarded_fragments = (
        'if ($h3_best_of_k -ne 1 -or $h3_best_of_k_stream -cne "video")',
        'if ($network_module -cne "networks.lora_minimax_h3")',
        'Test-H3NumericDefault $h3_shift_video ([decimal]12',
        'Test-H3NumericDefault $h3_shift_audio ([decimal]3',
        'Test-H3NumericDefault $h3_visual_cond_clean ([decimal]0.999',
        'Test-H3NumericDefault $h3_audio_cond_clean ([decimal]1',
        'Test-H3NumericDefault $audio_loss_weight ([decimal]1',
        'Test-H3NumericDefault $h3_guidance_loss_scale ([decimal]0',
        'Test-H3NumericDefault $h3_guidance_loss_sigma_min ([decimal]0',
        'Test-H3NumericDefault $lr ([decimal]0.000002',
        'Test-H3NumericDefault $max_train_steps ([decimal]1600',
        'Test-H3NumericDefault $network_alpha ([decimal]1',
        'Test-H3NumericDefault $gradient_accumulation_steps ([decimal]1',
        'Test-H3NumericDefault $lr_scheduler_num_cycles ([decimal]1',
        'Test-H3NumericDefault $max_data_loader_n_workers ([decimal]8',
        'Test-H3NumericDefault $max_grad_norm ([decimal]1',
    )
    for fragment in guarded_fragments:
        self.assertIn(fragment, train)

    for removed in (
        "--h3_video_best_of_k",
        "--h3_audio_best_of_k",
        "--h3_image_best_of_k",
    ):
        self.assertNotIn(removed, train)

    invocation = train.split(
        'python -m accelerate.commands.launch', 1
    )[1].split('Assert-NativeCommandSucceeded', 1)[0]
    for option in (
        "--network_module=",
        "--timestep_sampling=",
        "--discrete_flow_shift=",
        "--weighting_scheme=",
        "--h3_shift_video=",
        "--h3_shift_audio=",
        "--h3_visual_cond_clean=",
        "--h3_audio_cond_clean=",
        "--learning_rate=",
    ):
        self.assertNotIn(option, invocation)
```

Add:

```python
def run_train_defaults_helper(self, body: str) -> subprocess.CompletedProcess[str]:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        self.skipTest("PowerShell is unavailable")
    helper = str(self.TRAIN_DEFAULTS_HELPER).replace("'", "''")
    return subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f". '{helper}'; {body}",
        ],
        capture_output=True,
        encoding="utf-8",
    )
```

- [ ] **Step 2: Run PowerShell contract tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_scripts.py -q
```

Expected: FAIL because the numeric helper does not exist, an empty Best-of-K
argument list is rejected, and default-sensitive H3 values are still direct
arguments in the final Python invocation.

- [ ] **Step 3: Implement invariant-culture numeric comparison**

Create `powershell/minimax_h3_train_defaults.ps1`:

```powershell
function Test-H3NumericDefault {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [AllowEmptyString()]
        [object]$Value,

        [Parameter(Mandatory = $true)]
        [decimal]$DefaultValue
    )

    $errorMessage = "MiniMax-H3 default-sensitive value must be a finite number."
    if ($null -eq $Value -or $Value -is [bool]) {
        throw $errorMessage
    }

    $text = [Convert]::ToString(
        $Value,
        [Globalization.CultureInfo]::InvariantCulture
    )
    [decimal]$number = 0
    if (-not [decimal]::TryParse(
        $text,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$number
    )) {
        throw $errorMessage
    }

    return $number -eq $DefaultValue
}
```

- [ ] **Step 4: Permit the two valid PowerShell Best-of-K shapes**

First add `[AllowEmptyCollection()]` immediately above the
`[object[]]$Arguments` parameter so `@()` binds as the intentional default
shape. Then replace the canonical-count loop in
`Assert-H3BestOfKArgumentInvariant` with:

```powershell
$countOccurrences = @(
    $options | Where-Object { $_ -eq '--h3_best_of_k' }
).Count
$streamOccurrences = @(
    $options | Where-Object { $_ -eq '--h3_best_of_k_stream' }
).Count

if ($options -contains '--xm_best_of_k') {
    throw "MiniMax-H3 option --xm_best_of_k is not enabled; use h3_best_of_k."
}
if (
    ($countOccurrences -eq 0 -and $streamOccurrences -eq 0) -or
    ($countOccurrences -eq 1 -and $streamOccurrences -eq 1)
) {
    return
}
throw "MiniMax-H3 Best-of-K arguments must be absent or occur once as a complete pair."
```

- [ ] **Step 5: Build default-sensitive PowerShell arguments conditionally**

Dot-source the new helper beside the existing helpers:

```powershell
. (Join-Path $PSScriptRoot "powershell/minimax_h3_train_defaults.ps1")
```

Replace unconditional Best-of-K and H3 scalar additions with:

```powershell
if ($h3_best_of_k -ne 1 -or $h3_best_of_k_stream -cne "video") {
    [void]$ext_args.Add("--h3_best_of_k=$h3_best_of_k")
    [void]$ext_args.Add("--h3_best_of_k_stream=$h3_best_of_k_stream")
}
if ($network_module -cne "networks.lora_minimax_h3") {
    [void]$ext_args.Add("--network_module=$network_module")
}
if (-not (Test-H3NumericDefault $h3_shift_video ([decimal]12))) {
    [void]$ext_args.Add("--h3_shift_video=$h3_shift_video")
}
if (-not (Test-H3NumericDefault $h3_shift_audio ([decimal]3))) {
    [void]$ext_args.Add("--h3_shift_audio=$h3_shift_audio")
}
if (-not (Test-H3NumericDefault $h3_visual_cond_clean ([decimal]0.999))) {
    [void]$ext_args.Add("--h3_visual_cond_clean=$h3_visual_cond_clean")
}
if (-not (Test-H3NumericDefault $h3_audio_cond_clean ([decimal]1))) {
    [void]$ext_args.Add("--h3_audio_cond_clean=$h3_audio_cond_clean")
}
if (-not (Test-H3NumericDefault $audio_loss_weight ([decimal]1))) {
    [void]$ext_args.Add("--audio_loss_weight=$audio_loss_weight")
}
if (-not (Test-H3NumericDefault $h3_guidance_loss_scale ([decimal]0))) {
    [void]$ext_args.Add("--h3_guidance_loss_scale=$h3_guidance_loss_scale")
}
if (-not (Test-H3NumericDefault $h3_guidance_loss_sigma_min ([decimal]0))) {
    [void]$ext_args.Add("--h3_guidance_loss_sigma_min=$h3_guidance_loss_sigma_min")
}
if (-not (Test-H3NumericDefault $lr ([decimal]0.000002))) {
    [void]$ext_args.Add("--learning_rate=$lr")
}
```

Keep the existing conditional `h3_guidance_loss_scale_audio` and uncond-cache
arguments. Change these shared conditions:

```powershell
if ($max_train_steps) {
    if (-not (Test-H3NumericDefault $max_train_steps ([decimal]1600))) {
        [void]$ext_args.Add("--max_train_steps=$max_train_steps")
    }
}
elseif ($max_train_epochs) {
    [void]$ext_args.Add("--max_train_epochs=$max_train_epochs")
}

if (-not (Test-H3NumericDefault $network_alpha ([decimal]1))) {
    [void]$ext_args.Add("--network_alpha=$network_alpha")
}

if ($lr_scheduler -and $lr_scheduler -ine "constant") {
    [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}

if ($optimizer_type) {
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}
```

Replace the existing accumulation, scheduler-cycle, worker-count, and
gradient-norm comparisons with the same numeric helper:

```powershell
if (-not (Test-H3NumericDefault $gradient_accumulation_steps ([decimal]1))) {
    [void]$ext_args.Add("--gradient_accumulation_steps=$gradient_accumulation_steps")
}
if (-not (Test-H3NumericDefault $lr_scheduler_num_cycles ([decimal]1))) {
    [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}
if (-not (Test-H3NumericDefault $max_data_loader_n_workers ([decimal]8))) {
    [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}
if (-not (Test-H3NumericDefault $max_grad_norm ([decimal]1))) {
    [void]$ext_args.Add("--max_grad_norm=$max_grad_norm")
}
```

Remove `--network_module`, `--timestep_sampling`,
`--discrete_flow_shift`, `--weighting_scheme`, all four H3 shift/clean
arguments, and `--learning_rate` from the final direct Python invocation.
Keep task, dataset/model paths, `dit_dtype`, `mixed_precision`, seed,
output/logging paths, and attention selection.

- [ ] **Step 6: Run the PowerShell tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_scripts.py -q
```

Expected: PASS, including actual PowerShell execution of both helper files and
AST parsing of all MiniMax-H3 scripts.

- [ ] **Step 7: Run the Python/PowerShell cross-contract tests together**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py -q
```

Expected: PASS. The default Python builder and default PowerShell script omit
the same owned values, while custom Best-of-K remains a complete pair.

- [ ] **Step 8: Commit the PowerShell synchronization**

```powershell
git add -- 3.11minimax_h3_train_lora.ps1 powershell/minimax_h3_best_of_k.ps1 powershell/minimax_h3_train_defaults.ps1 gui/tests/test_minimax_h3_scripts.py
git commit -m "feat: omit default H3 PowerShell arguments"
```

---

### Task 4: Verify The Integrated Change And Deliver Main

**Files:**
- Verify only: `docs/superpowers/specs/2026-08-17-minimax-h3-default-cli-omission-design.md`
- Verify only: every file committed by Tasks 1-3
- Preserve: all pre-existing dirty and untracked user files

**Interfaces:**
- Consumes: the three implementation commits from Tasks 1-3.
- Produces: tested `main` with unchanged gitlink and no uncommitted feature-owned files.

- [ ] **Step 1: Synchronize submodules to the parent tree**

Run:

```powershell
git submodule update --init --recursive
```

Expected: exit code 0. No submodule working tree becomes dirty.

- [ ] **Step 2: Verify the complete gitlink contract**

Run each command separately:

```powershell
git ls-tree HEAD -- musubi-tuner
git -C musubi-tuner rev-parse HEAD
git -C musubi-tuner status --porcelain
```

Expected:

```text
160000 commit c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993	musubi-tuner
c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993
```

The status command must print nothing.

- [ ] **Step 3: Run focused H3 and preset regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_minimax_h3_command_builder.py gui/tests/test_minimax_h3_scripts.py gui/tests/test_minimax_h3_gui_contract.py gui/tests/test_preset_scope_and_defaults.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the broader GUI suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests -q
```

Expected: all feature-related tests PASS. If the known dirty personal prompt
or script fixtures still cause unrelated failures, record the exact failing
tests and confirm they reproduce without changing feature-owned files.

- [ ] **Step 5: Check whitespace, scope, and staged ownership**

Run each command separately:

```powershell
git diff --check origin/main...HEAD
git diff --stat origin/main...HEAD
git status --short
```

Expected: the diff contains only the two approved design commits, this plan,
and Tasks 1-3 implementation files. Existing unrelated dirty/untracked files
remain visible but unstaged. No H3 train preset TOML file is changed.

- [ ] **Step 6: Push main**

Run:

```powershell
git push origin main
```

Expected: `origin/main` advances to the verified local `main`. No feature
branch was created for this phase, so there is no branch to delete.
