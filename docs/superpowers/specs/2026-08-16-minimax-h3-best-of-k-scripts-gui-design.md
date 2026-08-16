# MiniMax-H3 Best-Of-K Scripts And GUI Design

## Context

The `musubi-tuner` submodule currently checked out at `c5df233` adds the final
MiniMax-H3 best-of-K command contract. The parent repository still pins
`b462291`, so a clean checkout cannot use the new options even though they are
present in the local submodule working copy. This change must update the
gitlink and expose the same contract through the PowerShell training script and
the native GUI.

The upstream interface is intentionally unified across image and video
training. It provides one candidate count and one selector for multi-frame
batches. It does not provide separate image, video, and audio counts.

## Goals

- Pin the `musubi-tuner` gitlink to `c5df233`, which contains the finalized H3
  best-of-K implementation and its boundary fixes.
- Add native script and GUI support for `--h3_best_of_k` and
  `--h3_best_of_k_stream`.
- Keep best-of-K disabled by default with `K=1`; users opt in by manually
  setting K above 1.
- Support video-only, one-frame image, and mixed image/video H3 datasets with
  the semantics defined upstream.
- Validate invalid combinations before starting a training process.
- Preserve existing H3 preset behavior and prevent stale best-of-K state from
  leaking across partially merged presets.

## Non-Goals

- Do not add a separate enable toggle. The upstream count already defines the
  state: `K=1` is disabled and `K>1` is enabled.
- Do not invent separate image, video, or audio K values.
- Do not accept or translate the removed `--h3_video_best_of_k`,
  `--h3_audio_best_of_k`, or `--h3_image_best_of_k` spellings.
- Do not edit implementation files inside the submodule. The parent repository
  only advances the gitlink.
- Do not enable the common `--xm_best_of_k` path for MiniMax-H3.

## Upstream Contract

- `--h3_best_of_k K` accepts an integer of at least 1 and defaults to 1.
- `K=1` uses the ordinary training path without extra candidates, metrics, or
  RNG consumption.
- `--h3_best_of_k_stream` accepts `video` or `audio` and defaults to `video`.
- On multi-frame batches, `video` varies video noise and ranks raw video MSE;
  `audio` varies audio noise and ranks raw audio MSE.
- One-frame image batches always vary video noise and rank image/video MSE,
  irrespective of the configured multi-frame stream.
- Mixed image/video runs use the same K for both batch kinds. The stream
  selector affects only their multi-frame batches.
- Active audio search, meaning `K>1` with stream `audio`, is incompatible with
  `--video_only`.
- Audio search with `audio_loss_weight=0` is accepted upstream but falls back
  to an ordinary forward for those batches and emits a warning. The outer
  scripts and GUI preserve this upstream behavior rather than rejecting it.
- H3 rejects `--xm_best_of_k` above 1 and rejects active best-of-K with teacher
  matching. Teacher matching is already unsupported by this GUI workflow.
- The approximate operation-count multiplier is `(K+3)/3`, so users should
  expect a meaningful speed cost as K increases.

## State And Defaults

The training state gains two H3-only keys:

```toml
h3_best_of_k = 1
h3_best_of_k_stream = "video"
```

The command builder supplies those defaults for old projects that do not yet
contain the keys. It emits both arguments explicitly for reproducibility,
including at K=1. Built-in presets also write both fields because preset
application is a partial merge and must clear values left by a previously
applied custom or exploratory preset.

## PowerShell Script Design

`3.11minimax_h3_train_lora.ps1` gains editable variables with the same defaults
as upstream. Before process launch it validates that K parses as an integer of
at least 1 and that the stream is exactly `video` or `audio`, case-insensitively
normalizing the script value for the command. It rejects active audio search
with `video_only`.

The script always appends the normalized values as
`--h3_best_of_k=<K>` and `--h3_best_of_k_stream=<stream>`. Existing one-frame
and video behavior is unchanged at K=1.

## GUI And Command Design

### Train Page

The MiniMax-H3 model settings add:

- A numeric stepper for K, with minimum 1, step 1, and default 1. There is no
  artificial maximum because upstream defines none.
- A translated select control for the multi-frame search stream, with `video`
  selected by default and `audio` as the other option.

The controls remain visible for one-frame mode. The GUI cannot infer whether a
dataset is image-only or mixed, and a mixed dataset still needs a stream choice
for video batches. The tooltip states that K=1 is off, that K>1 increases
training cost, and that one-frame image batches always search image/video
noise. No extra enable switch is shown.

The architecture synchronizer initializes both keys when MiniMax-H3 is
selected. Applying another H3 task or preset keeps the controls bound to the
canonical state values.

### Command Builder

The H3 scalar mapping registers both keys. `_with_minimax_h3_defaults` fills
missing values with `1` and `video`, allowing old saved projects to build a
deterministic command without migration.

H3 training validation:

- parses K with the existing strict integer helper and requires K to be at
  least 1;
- normalizes the stream and requires `video` or `audio`;
- rejects `K>1`, stream `audio`, and `video_only=true` together;
- continues to reject teacher matching through the existing H3 boundary;
- preserves upstream's warning/fallback behavior for audio search with zero
  audio loss weight.

Both normalized values are written back into the resolved state before mapped
arguments are emitted, so case variants and legacy missing keys produce the
canonical CLI form.

### Catalog And Localization

The model catalog lists the two new H3 train flags. Labels, option names, and
tooltips are supplied in English, Simplified Chinese, Japanese, and Korean,
matching the GUI's existing translation contract.

## Presets And Compatibility

All four built-in H3 training presets explicitly contain:

```toml
h3_best_of_k = 1
h3_best_of_k_stream = "video"
```

This includes T2VA, FL2VA, Ref2VA, and one-frame image presets. Best-of-K is
therefore opt-in everywhere, as requested. Applying any built-in H3 preset
after a custom K value reliably turns exploration back off.

Old saved projects without the new fields remain valid through builder and GUI
defaults. Removed experimental flag names remain classified as unsupported;
they are not silently converted because doing so could conceal an incorrect
image/audio mental model.

## Data Flow

1. A user selects an H3 training preset or opens an older H3 project. The GUI
   resolves K to 1 and the multi-frame stream to video.
2. The user may manually set K above 1 and optionally choose audio for
   multi-frame batches.
3. GUI validation rejects invalid values and active audio search combined with
   video-only training.
4. The command builder emits the two canonical upstream arguments.
5. The upstream trainer selects the runtime kind per batch: image for
   one-frame batches, otherwise the configured video or audio stream.

## Test Strategy

- Update the committed-submodule parser contract to read `c5df233`, recognize
  both canonical options, and remove the former video option from the deferred
  set. Verify the removed spellings do not become supported aliases.
- Add PowerShell contract tests for defaults, exact emitted arguments, strict K
  validation, stream validation, and the video-only/audio conflict.
- Add command-builder tests for default K=1 emission, video search, audio
  search, one-frame image state, mixed-compatible state, missing legacy state,
  noninteger/zero K, invalid stream, and the active audio/video-only conflict.
- Add GUI contract tests for rendering, defaults, translation keys, form-state
  round trips, control type and bounds, catalog registration, and preset
  application.
- Add preset tests proving every built-in H3 training preset resets K to 1 and
  stream to video.
- Run focused H3 tests first, then the complete GUI suite while distinguishing
  any failures caused by unrelated user-owned files already present in the
  dirty worktree.
- Start the GUI and inspect desktop and narrow-viewport Playwright screenshots
  for overflow, overlap, readable labels, and stable control dimensions.
