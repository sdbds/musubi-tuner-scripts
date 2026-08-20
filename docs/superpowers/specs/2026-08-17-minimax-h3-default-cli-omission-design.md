# MiniMax-H3 Default CLI Omission Design

## Status

Approved section by section in conversation on 2026-08-17. This document
records the implementation contract before production code changes begin.

## Objective

Keep every built-in MiniMax-H3 training preset explicit and editable in TOML
and GUI state, while omitting CLI arguments whose normalized values equal the
defaults defined by the pinned `musubi-tuner` parser.

The GUI command builder and `3.11minimax_h3_train_lora.ps1` must follow the
same omission behavior. A valid non-default value must still be emitted.
Values that MiniMax-H3 requires to remain fixed continue to fail validation.

## Upstream Authority

The parent repository pins `musubi-tuner` at the full commit:

```text
c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993
```

Defaults come from the parser sources stored at that gitlink, primarily:

- `src/musubi_tuner/minimax_h3_train_network.py`;
- `src/musubi_tuner/training/parser_common.py`;
- `src/musubi_tuner/training/audio_loss.py`.

Runtime importing of the upstream parser is out of scope. It would pull heavy
training dependencies into GUI command construction and make command preview
depend on the training environment. Tests instead read parser source from the
gitlink stored in the parent repository's committed `HEAD` tree and compare
its AST defaults with the local omission contract.

## Non-Goals

- Do not remove default-valued fields from built-in TOML presets.
- Do not hide or clear default-valued GUI controls.
- Do not change cache or generation command behavior.
- Do not change command behavior for non-H3 architectures.
- Do not infer arbitrary semantic defaults beyond an explicit allowlist.
- Do not reintroduce unpublished removed Best-of-K aliases.
- Do not implement the eventual all-architecture default policy in this phase.

## State And Command Flow

The canonical project state remains explicit. Applying a built-in preset keeps
values such as:

```toml
h3_visual_cond_clean = 0.999
h3_best_of_k = 1
h3_best_of_k_stream = "video"
text_encoder_blocks_to_swap = 0
```

The train builder follows this order:

1. Resolve architecture and copy/normalize MiniMax-H3 state.
2. Validate the complete state before process launch.
3. Build the existing canonical argument list.
4. For MiniMax-H3 only, remove allowlisted arguments equal to pinned defaults.
5. Validate the post-compaction Best-of-K invariant.
6. Return the command job without mutating the caller's state.

This keeps GUI state as the source of user intent while treating CLI omission
as a serialization concern.

## Explicit Omission Contract

### MiniMax-H3 parser defaults

| CLI option | Default |
| --- | --- |
| `--network_module` | `networks.lora_minimax_h3` |
| `--timestep_sampling` | `uniform` |
| `--weighting_scheme` | `none` |
| `--discrete_flow_shift` | `1.0` |
| `--h3_shift_video` | `12.0` |
| `--h3_shift_audio` | `3.0` |
| `--h3_visual_cond_clean` | `0.999` |
| `--h3_audio_cond_clean` | `1.0` |
| `--audio_loss_weight` | `1.0` |
| `--convrot_int8_bwd` | `bf16` |
| `--h3_guidance_loss_scale` | `0.0` |
| `--h3_guidance_loss_sigma_min` | `0.0` |
| `--text_encoder_blocks_to_swap` | `0` |

Optional `None` values and false boolean flags already remain absent through
normal command construction. Examples include
`--h3_guidance_loss_scale_audio`, `--video_only`, `--convrot_int8`,
`--prune_adaln`, and `--nvfp4_scaled_mm`.

### Shared parser defaults reachable from H3 training

| CLI option | Default |
| --- | --- |
| `--max_train_steps` | `1600` |
| `--max_data_loader_n_workers` | `8` |
| `--gradient_accumulation_steps` | `1` |
| `--guidance_scale` | `1.0` |
| `--learning_rate` | `2e-6` |
| `--max_grad_norm` | `1.0` |
| `--lr_scheduler` | `constant` |
| `--lr_warmup_steps` | `0` |
| `--lr_decay_steps` | `0` |
| `--lr_scheduler_num_cycles` | `1` |
| `--lr_scheduler_power` | `1.0` |
| `--network_alpha` | `1.0` |
| `--block_swap_ring_size` | `2` |
| `--compile_backend` | `inductor` |
| `--compile_mode` | `default` |

Several of these are already suppressed by existing helpers. They remain in
the contract and tests so H3 behavior is explicit rather than dependent on
incidental helper ordering.

`network_dropout = 0` remains absent because upstream explicitly defines zero
and `None` as the same no-dropout behavior. No broader equivalence inference
is permitted.

### Arguments that remain explicit

The omission allowlist must not remove:

- `--dataset_config`, `--task`, and required model paths;
- `--output_name`, `--output_dir`, and configured logging paths;
- `--dit_dtype` and `--mixed_precision`;
- the selected attention backend;
- non-default seed, epoch/step limit, optimizer, scheduler, or LoRA settings;
- feature-enabling boolean flags and their non-default dependent values;
- sampling paths and sampling frequency.

Although MiniMax-H3 later resolves a missing `dit_dtype` to BF16, the parser
default is `None` and the value participates in the launch contract. It is
therefore intentionally not treated as omittable.

## Typed Comparison Rules

The Python contract records both the canonical option and its expected type.
Comparison occurs only after normal command builders have produced canonical
`--name=value` tokens.

- Integer defaults require a canonical integer token.
- Numeric defaults use exact numeric equality, so `12` equals `12.0`.
- Floating-point comparison does not use a tolerance; a deliberate nearby
  value must not disappear.
- Known enum values are compared after their existing normalization.
- Parse or validation failure never causes omission.
- Unknown options are retained.

The compactor is scoped to MiniMax-H3. It must not add default-aware branches
to generic helpers used by other architectures in this phase.

## Best-of-K Atomic Rule

`--h3_best_of_k` and `--h3_best_of_k_stream` are one logical parameter pair:

- resolved `1/video` omits both options;
- if either resolved value differs, both canonical options are emitted exactly
  once;
- a partial pair or duplicate remains an internal command-build error;
- `--xm_best_of_k` remains forbidden for MiniMax-H3;
- structured Best-of-K options remain reserved in generic extra arguments so
  extra arguments cannot override GUI state.

The PowerShell invariant helper must accept exactly two valid shapes: neither
canonical option, or one occurrence of each. It must reject a partial pair,
duplicates, or `--xm_best_of_k`.

## PowerShell Contract

`3.11minimax_h3_train_lora.ps1` retains readable variables at the top of the
script. Default-sensitive direct arguments move into conditional argument
construction where needed.

The default script therefore omits the same H3 and shared defaults as the GUI
builder. Changing a variable to a non-default value restores its canonical
argument when that value is valid for MiniMax-H3. In particular:

- default Best-of-K emits neither option;
- custom Best-of-K emits both options;
- flow shifts and condition-clean coefficients emit only when changed;
- generic timestep, weighting, and discrete-flow defaults are never emitted
  for a valid run; attempted changes remain validation errors;
- guidance and audio-loss defaults emit only when changed;
- the default network module is omitted;
- existing conditional omissions for accumulation, workers, scheduler cycles,
  and gradient norm remain aligned with the contract.

Validation still happens before environment activation and before the Python
process starts.

## Error Behavior

Existing MiniMax-H3 validation remains authoritative. Default omission cannot
turn an invalid value into a valid default. Reserved structured arguments are
still rejected before process launch. No new warning or UI message is added
for successfully omitted defaults.

Removed unpublished Best-of-K aliases remain outside parent state and UI.
This work neither translates them nor adds compatibility behavior.

## Testing

Implementation follows red-green-refactor.

Builder tests cover:

- all built-in MiniMax-H3 train presets retain explicit TOML values;
- each preset command omits every applicable allowlisted default;
- representative non-default integer, float, enum, and string values reappear;
- numeric `12` and `12.0` compare equal without accepting nearby values;
- default Best-of-K omits both options;
- changing either Best-of-K field emits both exactly once;
- invalid and conflicting structured values fail before process launch;
- non-H3 command snapshots remain unchanged.

PowerShell tests cover:

- syntax parsing;
- default-sensitive conditions for every script-owned contract value;
- the two accepted Best-of-K argument shapes;
- rejection of partial, duplicate, and XM Best-of-K arguments;
- non-default values remain expressible.

Source-lock tests cover:

- parent tree gitlink equals the full target SHA;
- submodule HEAD equals that SHA and is clean;
- AST defaults in the parent-tree H3, common, and audio parser sources equal
  the local omission table;
- both canonical Best-of-K parser options exist;
- removed old Best-of-K options are not emitted by parent builders or scripts.

Targeted H3 command-builder, GUI-contract, preset, and PowerShell tests run
first. The broader GUI test suite runs afterward, with unrelated pre-existing
failures reported separately rather than hidden.

## Delivery

Only files owned by this change are staged. Existing user modifications and
untracked personal files remain untouched. After implementation and clean
verification, the completed change is committed to `main` and pushed, matching
the previously approved repository workflow.
