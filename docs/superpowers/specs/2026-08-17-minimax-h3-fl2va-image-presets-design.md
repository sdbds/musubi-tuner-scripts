# MiniMax-H3 FL2VA One-Frame Image Presets Design

## Status

Approved in conversation on 2026-08-17. This document records the exact
implementation contract before work begins.

## Upstream Baseline

The parent repository pins `musubi-tuner` at the full commit:

```text
c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993
```

That commit already contains upstream `dev` commit:

```text
2f7677f6f5e2c5dca22c70dd6376f1487d6626cb
```

The upstream commit adds experimental MiniMax-H3 one-frame FL2VA training
with one or two time-annotated control images. The parent gitlink therefore
does not need to move for this work. Acceptance still verifies that the
gitlink, submodule HEAD, and indexed parser sources remain consistent.

## Problem

The GUI currently exposes only the earlier plain T2VA one-frame image
workflow. The pinned backend can also train:

- FL2VA image editing with one control image;
- FL2VA inbetween training with two control images.

The parent repository is not yet able to use those paths end to end:

- the PowerShell cache and train scripts reject `one_frame + fl2va`;
- the GUI command builder rejects `one_frame + fl2va`;
- the cache and train pages hide and clear one-frame mode for FL2VA;
- the dataset editor clears MiniMax-H3 control fields and exports only the
  target index;
- no built-in dataset, cache, or train presets cover the new workflows.

Adding TOML files alone would create visible presets that fail before the
backend starts. The complete parent-side contract must be updated together.

## Terminology

This design never calls the number of control images "K" in parent-facing UI
or tests. It uses:

- `control_count = 0`, `1`, or `2` for one-frame conditioning layout;
- `h3_best_of_k` only for candidate-noise search.

This prevents the upstream commit title's `K=1/2` control count from being
confused with Best-of-K candidate count.

## Supported Workflow Matrix

| Workflow | Version | Task | One frame | Controls | Control indices | Target index |
| --- | --- | --- | --- | ---: | --- | ---: |
| Plain image LoRA | `fl2va` | `t2va` | yes | 0 | absent | `0` |
| Image edit | `fl2va` | `fl2va` | yes | 1 | `[0]` | `24` |
| Inbetween | `fl2va` | `fl2va` | yes | 2 | `[0, 48]` | `24` |

One-frame Ref2VA training remains unsupported. Cache and train builders,
PowerShell validation, and GUI visibility accept one-frame mode only when the
version is `fl2va` and the task is either `t2va` or `fl2va`.

## Built-In Presets

### Dataset presets

Keep the existing plain preset:

```text
toml/qinglong_minimax_h3_image.toml
```

Add:

```text
toml/qinglong_minimax_h3_image_edit.toml
toml/qinglong_minimax_h3_image_inbetween.toml
```

The edit preset uses one control directory, `fp_1f_clean_indices = [0]`, and
`fp_1f_target_index = 24`. The inbetween preset uses two controls per target,
`fp_1f_clean_indices = [0, 48]`, and `fp_1f_target_index = 24`.

Both presets use 1024 by 1024 buckets, batch size 1, separate target, control,
and cache directories, and `multiple_target = false` by omission. They do not
emit `no_resize_control` or `control_resolution`, which the H3 backend rejects.

### Cache presets

Keep the existing plain cache preset and add matching built-ins:

```text
gui/presets/cache/minimax_h3_image_edit.toml
gui/presets/cache/minimax_h3_image_inbetween.toml
```

Both new presets explicitly select:

```toml
arch = "MiniMax-H3"
version = "fl2va"
task = "fl2va"
one_frame = true
```

They run both cache phases and retain the current recommended quantized text
encoder and block-swap settings. All three image workflows reuse the same
single-space unconditional embedding output because it is independent of the
dataset control layout:

```text
./cache/minimax_h3_image_uncond.safetensors
```

### Train presets

Keep the existing plain train preset and add:

```text
gui/presets/train/minimax_h3_image_edit.toml
gui/presets/train/minimax_h3_image_inbetween.toml
```

The new presets explicitly select `task = "fl2va"`, `one_frame = true`, and
the FL2VA transformer. They retain the upstream one-frame recommendations:

```toml
video_only = true
h3_teacher_matching = false
h3_guidance_loss_scale = 4.0
h3_guidance_loss_sigma_min = 0.15
h3_guidance_loss_uncond_cache = "./cache/minimax_h3_image_uncond.safetensors"
lr_warmup_steps = 50
h3_best_of_k = 1
h3_best_of_k_stream = "video"
```

Best-of-K remains disabled until the user changes the integer count.

Control-image sampling needs user-owned condition files. The edit and
inbetween train presets therefore set `enable_sample = false` and
`sample_at_first = false`, preventing placeholder paths from aborting a
training launch. They reference separate editable prompt templates:

```text
toml/qinglong_minimaxh3_image_edit.txt
toml/qinglong_minimaxh3_image_inbetween.txt
```

The template lines demonstrate the correct `--i`, `--ei`, and `--of` layout,
but remain inactive until the user enables sampling and supplies real images.

## Dataset Editor Contract

The existing `minimax_h3_one_frame` dataset template remains the single H3
one-frame editor. It covers plain and controlled datasets without multiplying
template modes.

For a directory source it exposes:

- target image directory;
- optional control image directory;
- cache directory;
- H3 control frame indices;
- H3 target frame index.

For a JSONL source, control paths remain inside the JSONL while the same index
fields stay available.

H3 control indices normalize to a list of one or two nonnegative integers.
When controls are configured, the target index is mandatory. A directory
source requires the control directory and control indices together. Plain
T2VA may omit both control fields and may omit the target index, which leaves
the upstream default of zero in effect.

The editor must not expose or export H3-incompatible control resizing fields.
Switching into the H3 one-frame template clears FramePack-only settings and
starts without controls. Applying a dataset preset restores its complete
canonical values.

## Cache And Train UI State

On the MiniMax-H3 `fl2va` model version:

- task `t2va`: show one-frame mode;
- task `fl2va`: show one-frame mode;
- task `ref2va`: unavailable by the catalog contract.

On the `ref2va` version, one-frame mode is hidden and cleared. Switching from
an active FL2VA one-frame preset to Ref2VA must not leak `--one_frame` into the
command. Switching back does not silently re-enable it unless a preset or the
user selects it again, matching the existing safety behavior.

## Command And Script Contract

The GUI cache and train builders accept `--one_frame` for the two supported
tasks and continue rejecting Ref2VA one-frame state before process launch.
Both cache jobs and the train job emit `--task=fl2va --one_frame` exactly once
for controlled-image presets.

The PowerShell cache and train scripts use the same task matrix. No new CLI
argument is needed because control paths and indices are dataset TOML fields.
Validation remains before environment activation and process launch.

The removed unpublished Best-of-K aliases remain ignored by parent state and
are not reintroduced by this work.

## Testing

Implementation follows red-green-refactor. Tests first prove the current
failure for FL2VA one-frame before production changes.

Automated coverage includes:

- upstream commit ancestry and unchanged full gitlink SHA;
- cache and train builder acceptance for FL2VA one-frame;
- continued Ref2VA rejection;
- both cache jobs and the train job emit the correct task and one-frame flags;
- PowerShell syntax and task-matrix contract;
- discovery and exact values of all three dataset/cache/train workflows;
- dataset import, GUI state, export, and round-trip of one and two controls;
- rejection of missing paired fields, negative indices, zero controls, and
  more than two controls;
- H3 template exclusion of control-resize and multiple-target fields;
- cache/train task transitions and stale one-frame clearing;
- Best-of-K remains integer `1` with stream `video` in every train preset;
- non-H3 preset and dataset behavior remains unchanged.

The full GUI test suite runs from a clean checkout after implementation.

## Visual Acceptance

Run the GUI locally and verify with Playwright at desktop and mobile widths:

- the three preset labels are discoverable in the relevant pages;
- applying each preset updates architecture, task, and one-frame controls;
- H3 control-directory and index inputs fit without clipping or overlap;
- the one-control and two-control dataset presets display their exact indices;
- browser console has no errors or warnings caused by the changes.

## Delivery

Work is isolated on `codex/minimax-h3-fl2va-image-presets`. Only files owned by
this feature are staged. Existing user modifications and untracked files are
left untouched. After clean verification, the feature is fast-forwarded to
`main`, pushed, and the feature branch is deleted as previously requested.
