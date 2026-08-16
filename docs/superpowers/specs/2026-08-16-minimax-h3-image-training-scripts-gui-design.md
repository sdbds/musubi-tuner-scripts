# MiniMax-H3 Image Training Scripts And GUI Design

## Context

The `musubi-tuner` submodule now includes upstream commit `ca221b1`, which adds
experimental one-frame T2VA LoRA training for MiniMax-H3. The feature accepts
standard image datasets by representing each image as a single video latent
token with a silent audio placeholder. It is enabled independently on the
latent cache command, text-encoder cache command, and trainer with
`--one_frame`.

The outer repository already exposes MiniMax-H3 video cache and LoRA workflows,
but its PowerShell scripts, command builder, GUI controls, and built-in presets
do not carry the new flag. The existing T2VA presets must remain video presets.

## Goals

- Add a complete MiniMax-H3 one-frame T2VA path through the PowerShell scripts
  and native GUI.
- Add separate image cache and image LoRA presets without changing existing
  T2VA, FL2VA, or Ref2VA video preset behavior.
- Apply the upstream-recommended image-training defaults in the new preset:
  `video_only=true`, guidance loss scale `4.0`, guidance sigma minimum `0.15`,
  and 50 warmup steps.
- Create the unconditional text cache required by the recommended guidance
  loss workflow.
- Support standard image-directory and image-JSONL datasets, including the
  optional nonnegative `fp_1f_target_index` field.
- Keep the upstream experimental image-and-video mixed dataset behavior
  available.

## Non-Goals

- Do not add FL2VA or Ref2VA image training. Upstream currently supports
  one-frame training only with `--task t2va`.
- Do not infer one-frame mode from the dataset contents. Mixed image and video
  datasets are valid when the user enables the mode explicitly.
- Do not make the recommended guidance settings mandatory in the generic
  command builder. Upstream permits other experimental values.
- Do not modify the `musubi-tuner` submodule implementation.
- Do not change MiniMax-H3 one-frame generation controls except for supplying a
  one-frame training sample prompt.

## Upstream Contract

One-frame training has the following command contract:

- `minimax_h3_cache_latents.py` requires `--one_frame` to accept image datasets.
- `minimax_h3_cache_text_encoder_outputs.py` requires the same flag so image
  captions are encoded as plain T2VA presentations.
- `minimax_h3_train_network.py` requires the same flag to accept the resulting
  single-token latent caches.
- One-frame mode currently requires `--task t2va` and is incompatible with H3
  teacher matching.
- `--video_only` is recommended for image-only runs because the cached audio is
  a silence placeholder excluded from audio supervision.
- Guidance loss scale `4.0` with sigma minimum `0.15` and an unconditional text
  cache is the upstream-tested recommendation. A short warmup of 50 steps is
  also recommended.
- Image and video datasets may be mixed in one run. Their cache directories must
  remain distinct.
- `fp_1f_target_index` is an optional nonnegative 24 fps pixel-frame index. Its
  default is `0`.
- Training sample lines use `--f 1` and may use `--of target_index=N`; one-frame
  samples are written as PNG files without audio decoding.

## Script Design

### Cache Script

`2.11minimax_h3_cache_latent_and_text_encoder.ps1` gains editable variables for
one-frame mode and unconditional-cache output. When one-frame mode is enabled,
the script appends `--one_frame` to both the latent and text-encoder cache
argument lists. It rejects tasks other than `t2va` before launching Python.

The unconditional-cache variables map to `--uncond_output` and
`--uncond_text`. Empty values preserve the current video workflow. The image
preset supplies a shared unconditional-cache path used later by the trainer.

### Training Script

`3.11minimax_h3_train_lora.ps1` gains editable variables and argument handling
for:

- `one_frame`
- `video_only`
- `audio_loss_weight`
- `h3_guidance_loss_scale`
- `h3_guidance_loss_scale_audio`
- `h3_guidance_loss_sigma_min`
- `h3_guidance_loss_uncond_cache`

The script rejects `one_frame` with a task other than `t2va`. A positive
guidance loss scale requires a nonempty unconditional-cache path. Defaults keep
the existing video behavior; the separate image preset carries the recommended
values.

## GUI And Command Design

### Command Builder

The cache builder passes `--one_frame` to both MiniMax-H3 cache jobs. The train
builder includes it in the MiniMax-H3 boolean argument set. Validation rejects
one-frame mode unless the selected version and task resolve to FL2VA/T2VA.

No validation requires an image-only project. This preserves mixed dataset
training. Existing guidance-cache validation remains responsible for rejecting
a positive guidance scale without a cache path.

### Cache Page

The MiniMax-H3 cache card adds an experimental image-training toggle. It is
available only for the `t2va` task. Selecting `fl2va` or `ref2va` hides or
disables the control and clears its value so a stale flag cannot reach the
command builder.

The existing unconditional-cache output and text controls remain the source of
the guidance cache. The new image cache preset enables one-frame mode and fills
the output path.

### Train Page

The MiniMax-H3 train card adds the matching experimental image-training toggle.
It follows the same task-dependent visibility and reset behavior as the cache
page. Applying the image preset sets the toggle, video-only mode, guidance loss,
guidance cache path, warmup, and one-frame sample prompt together.

The GUI does not silently rewrite those values when a user manually changes the
toggle. Presets establish recommended starting values; user edits remain
explicit.

### Dataset Page

The image dataset template list gains a MiniMax-H3 one-frame template. It uses
the existing standard image fields and adds only `fp_1f_target_index`. It does
not expose FramePack clean indices, post controls, control directories, or
`multiple_target`, because those fields are unsupported by H3 one-frame
training.

Import inference selects the H3 template when the imported source path contains
the `minimax_h3` marker and the dataset contains `fp_1f_target_index`. Imports
without that marker retain the existing FramePack inference, because the TOML
field alone does not identify which architecture will consume it. A
nonnegative integer is exported when supplied; an empty value omits the field
and lets upstream use `0`.

## Presets And Examples

Add these built-in artifacts:

- `gui/presets/cache/minimax_h3_image.toml`
- `gui/presets/train/minimax_h3_image.toml`
- `toml/qinglong_minimax_h3_image.toml`
- `toml/qinglong_minimaxh3_image.txt`

The cache and train presets share the same dataset config and unconditional
cache paths. Their labels identify them as Image T2VA presets. Existing
`minimax_h3.toml` files retain their video defaults and labels.

The sample dataset uses a standard image directory, a separate image cache
directory, 1024 by 1024 resolution, batch size 1, bucketing, and target index 0.
The sample prompt uses `--f 1`, valid H3 dimensions, and an output seed suitable
for a smoke test.

## Data Flow

1. The user creates or imports a standard image dataset and selects the
   MiniMax-H3 one-frame template if a custom target index is needed.
2. The image cache preset exports the project dataset TOML and launches both H3
   cache stages with `--task t2va --one_frame`.
3. The text stage also writes the configured unconditional embedding cache.
4. The image train preset launches the H3 trainer with `--task t2va
   --one_frame`, the recommended video-only and guidance settings, and the same
   unconditional-cache path.
5. Training-time sampling reads a prompt containing `--f 1` and writes PNG
   samples.

## Errors And Compatibility

- PowerShell and GUI command validation fail before process launch for
  one-frame FL2VA or Ref2VA requests.
- A positive guidance scale without an unconditional-cache path fails before
  process launch.
- Existing video presets continue to omit `--one_frame` and retain their
  current dataset, audio, and sampling behavior.
- Preset application does not alter user-maintained TOML files.
- The implementation must work with the current dirty worktree and must not
  revert unrelated local changes.

## Test Strategy

- Extend the MiniMax-H3 PowerShell contract tests to assert that one-frame and
  guidance variables map to the exact upstream flags and that both cache stages
  receive `--one_frame`.
- Add command-builder tests for the two one-frame cache jobs, the one-frame
  training job, invalid tasks, missing guidance cache, and mixed-dataset
  acceptance.
- Add GUI contract and state tests for the new controls, task switching, preset
  application, and translated labels/tooltips.
- Add preset tests proving the new image presets contain the upstream-recommended
  defaults while all existing video presets keep `one_frame=false` or omit it.
- Add dataset page tests for H3 template rendering, target-index validation,
  import inference, and TOML round trips.
- Run the focused MiniMax-H3, preset, dataset, and script tests, followed by the
  complete GUI test suite.
- Launch the GUI and use browser screenshots at desktop and narrow viewports to
  verify that the added controls do not overlap or overflow.
