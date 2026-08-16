# MiniMax-H3 Best-Of-K Scripts And GUI Design

## Context

The `musubi-tuner` submodule currently checked out at commit
`c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993` adds the final MiniMax-H3
best-of-K command contract. The parent repository still pins
`b462291a6e1bd25180ce1d1298db72982c8ed27a`, so a clean checkout cannot use
the new options even though they are present in the local submodule working
copy. This change must update the gitlink and expose the same contract through
the PowerShell training script and the native GUI. Short SHAs may be used in
human-facing status output only; specifications, plans, and acceptance checks
use the full 40-character target SHA.

The upstream interface is intentionally unified across image and video
training. It provides one candidate count and one selector for multi-frame
batches. It does not provide separate image, video, and audio counts.

## Goals

- Pin the `musubi-tuner` gitlink to
  `c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`, which contains the finalized
  H3 best-of-K implementation and its boundary fixes.
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

Missing and explicitly invalid values are different. A missing count defaults
to integer `1`, while an explicitly empty count is rejected. At the GUI and
builder boundary, an integer-valued finite float such as `1.0` is accepted only
to accommodate numeric-control output and is immediately normalized to Python
`int`. Booleans, fractional floats, all strings (including empty, numeric, and
scientific-notation strings), non-finite numbers, zero, and negative values are
rejected. The canonical collected project state and saved TOML therefore always
contain an integer such as `h3_best_of_k = 1`, never `1.0` or `"1"`.

The stream must be a string. It is trimmed and lowercased, then must equal
`video` or `audio`; a missing stream defaults to `video`, while an explicitly
empty or non-string stream is rejected.

## Reserved CLI Arguments

For MiniMax-H3 training, structured state is the only source of the canonical
best-of-K values. The following option names are reserved and may not be
supplied through any free-form argument channel:

- `--h3_best_of_k`
- `--h3_best_of_k_stream`
- `--h3_video_best_of_k`
- `--h3_audio_best_of_k`
- `--h3_image_best_of_k`
- `--xm_best_of_k`

The current raw token surfaces are the GUI's `optimizer_extra_args` and the
PowerShell script's `$optimizer_args`. Although they are labeled for optimizer
configuration, a token beginning with `--` can escape an argparse `nargs`
value list and become a top-level option. They therefore receive the same
reserved-option validation as any future general `extra_args` channel.

Validation recognizes both `--name value` and `--name=value`, rejects the
conflict before process launch, and does not rely on argparse ordering or its
last-value behavior. The final H3 argv is also checked as an invariant:
`--h3_best_of_k` and `--h3_best_of_k_stream` each occur exactly once from the
structured mapping, while the three removed names and `--xm_best_of_k` occur
zero times.

The three removed H3 state keys are rejected whenever they occur in loaded
training state or a preset, even if their value is empty or 1. The error names
the canonical replacement. An `xm_best_of_k` state key is likewise rejected
for H3 with guidance to use `h3_best_of_k`. None of these fields is ignored or
migrated silently.

## PowerShell Script Design

`3.11minimax_h3_train_lora.ps1` gains editable variables with the same defaults
as upstream. Before process launch it validates K without a truncating numeric
cast. Integer CLR values and base-10 integer strings are accepted; floating
point values, including `1.0`, booleans, empty strings, decimal fractions, and
scientific notation are rejected. The normalized integer must be at least 1.
The stream is normalized case-insensitively and must be exactly `video` or
`audio`. The script rejects active audio search with `video_only`.

The script scans `$optimizer_args` before adding it to `$ext_args` and rejects
every reserved option in either separated or equals form. It then verifies the
same occurrence invariant over the completed training argv. This mirrors the
GUI builder policy and catches manual script extensions without depending on
argument order.

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

The K control normalizes any integer-valued float emitted by NiceGUI to `int`
on change and again when form state is collected. This prevents a transient
`1.0` UI representation from being serialized into project or preset TOML.

The controls remain visible for one-frame mode. The GUI cannot infer whether a
dataset is image-only or mixed, and a mixed dataset still needs a stream choice
for video batches. The tooltip states that K=1 is off, that K>1 increases
training cost, and that one-frame image batches always search image/video
noise. No extra enable switch is shown.

The architecture transition rules are deterministic:

- Leaving MiniMax-H3 immediately removes the H3-only controls from the rendered
  card but preserves their canonical values in form state.
- A non-H3 command builder ignores both canonical H3 fields unconditionally,
  even when they are present in a loaded project.
- Returning to MiniMax-H3 restores the preserved canonical values. Defaults of
  `1` and `video` are applied only when a key is absent, not on every
  architecture transition.
- Loading non-H3 state that contains the two canonical H3 fields preserves them
  as dormant H3-owned state while keeping the controls hidden and excluding the
  options from non-H3 commands.
- Loading any of the three removed H3 fields fails with migration guidance
  instead of silently dropping unknown state.
- Applying any built-in H3 preset explicitly resets both values to `1` and
  `video`.

Both controls update the backing config as their values change, before the
dynamic H3 control scope can be disposed. The architecture synchronizer only
initializes missing keys; it does not overwrite a value saved before an
H3 to non-H3 to H3 round trip.

### Command Builder

The H3 argument mapping registers the integer count and enum stream fields.
`_with_minimax_h3_defaults` fills only missing values with `1` and `video`,
allowing old saved projects to build a deterministic command without migration.

The train-builder entry rejects the three removed H3 state keys regardless of
the currently selected architecture, so switching to a non-H3 model cannot
turn stale retired state into a silent no-op. H3-specific validation then:

- normalizes K with a dedicated best-of-K boundary helper, accepts `int` and
  integer-valued finite `float`, rejects booleans, strings, fractional or
  non-finite floats, and requires the normalized integer to be at least 1;
- normalizes the stream and requires `video` or `audio`;
- rejects `K>1`, stream `audio`, and `video_only=true` together;
- rejects an H3 `xm_best_of_k` state key with specific migration guidance;
- rejects reserved options found in raw argument text or the final argv;
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

## Gitlink Acceptance

The target gitlink is exactly
`c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993`. Final acceptance reads the
`musubi-tuner` entry from the parent commit tree with `git ls-tree HEAD`, not
from the index alone and not from `git -C musubi-tuner rev-parse HEAD` in the
developer's existing checkout.

After the implementation commit, a disposable clean checkout of that exact
parent commit runs `git submodule update --init --recursive`. Acceptance then
requires:

- the parent tree's `musubi-tuner` gitlink equals the full target SHA;
- the clean checkout's submodule HEAD equals the same SHA;
- `git -C musubi-tuner status --porcelain` is empty;
- the committed trainer parser declares the two canonical options;
- each removed spelling is rejected with its upstream migration error and is
  not classified as a supported functional option.

The last condition deliberately tests rejection rather than absence from
argparse's internal option registry. Commit
`c5df233bd14e5ed1fb9fe00ff7b98f054e5e1993` registers hidden tombstone actions
for the three removed spellings so users receive migration guidance. Requiring
those option strings to be literally absent would contradict the target
upstream implementation.

## Test Strategy

The parser classification matrix is explicit:

```text
Supported:
  --h3_best_of_k
  --h3_best_of_k_stream

Removed / unsupported:
  --h3_video_best_of_k
  --h3_audio_best_of_k
  --h3_image_best_of_k

Not enabled for H3:
  --xm_best_of_k
```

- Update the parent-tree submodule contract to use the full target SHA and test
  each matrix entry directly instead of inferring support from membership in a
  deferred set.
- Add clean-checkout acceptance for gitlink equality, submodule HEAD, clean
  status, canonical parser declarations, and removed-option rejection.
- Add PowerShell contract tests for defaults, exact emitted arguments,
  non-truncating K validation, stream validation, reserved arguments in both
  CLI forms, the final argv occurrence invariant, and the video-only/audio
  conflict. Cover integer 1, floating 1.0, 1.5, 0, -1, boolean, empty, numeric
  text, scientific notation, and nonnumeric text according to the
  PowerShell-specific rules above.
- Add command-builder tests for default K=1 emission, video search, audio
  search, one-frame image state, mixed-compatible state, and missing legacy
  state. Cover integer 1, integer-valued float 1.0, 1.5, 0, -1, boolean, empty,
  numeric string, scientific-notation string, nonnumeric text, invalid stream,
  removed state keys, and the active audio/video-only conflict.
- Test every reserved option through `optimizer_extra_args` using both
  `--name value` and `--name=value`. Assert failure occurs while building the
  job and before any process runner is invoked.
- Add GUI contract tests for rendering, defaults, translation keys, integer
  normalization, TOML-compatible collected state, form-state round trips,
  control type and bounds, catalog registration, and preset application.
- Add an H3 to non-H3 to H3 state-machine test proving controls hide, canonical
  values remain dormant, non-H3 argv contains no H3 best-of-K option, and the
  values are restored when H3 is reselected. Also test loading canonical H3
  fields into non-H3 state and rejecting removed fields.
- Add preset tests proving every built-in H3 training preset resets K to 1 and
  stream to video.
- Run focused H3 tests first, then the complete GUI suite while distinguishing
  any failures caused by unrelated user-owned files already present in the
  dirty worktree.
- Start the GUI and inspect desktop and narrow-viewport Playwright screenshots
  for overflow, overlap, readable labels, and stable control dimensions.
