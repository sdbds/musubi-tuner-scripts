# MiniMax-H3 Image Training Scripts And GUI Design

## Context

Upstream `musubi-tuner` commit `ca221b1` adds experimental one-frame T2VA LoRA
training for MiniMax-H3. The current submodule working copy contains it through
`b462291`, but the parent repository gitlink still points to `29aee45`. A clean
clone therefore lacks the feature even though the local dirty checkout exposes
it. The outer-repository change must pin the gitlink to `b462291` together with
the script and GUI changes.

The upstream feature accepts standard image datasets by representing each image
as a single video latent token with a silent audio placeholder. It is enabled
independently on the latent cache command, text-encoder cache command, and
trainer with `--one_frame`.

The outer repository already exposes MiniMax-H3 video cache and LoRA workflows,
but its PowerShell scripts, command builder, GUI controls, and built-in presets
do not carry the new flag. The existing T2VA presets must remain video presets.

## Goals

- Add a complete MiniMax-H3 one-frame T2VA path through the PowerShell scripts
  and native GUI.
- Update the `musubi-tuner` gitlink to `b462291`, a configured-origin commit
  containing upstream `ca221b1`, so clean checkouts expose the same CLI contract.
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
- Do not edit source files inside the `musubi-tuner` submodule. Updating the
  parent repository gitlink is required.
- Do not change MiniMax-H3 one-frame generation controls except for supplying a
  one-frame training sample prompt.

## Upstream Contract

One-frame training has the following command contract:

- `minimax_h3_cache_latents.py` requires `--one_frame` to accept image datasets.
- `minimax_h3_cache_text_encoder_outputs.py` requires the same flag so image
  captions are encoded as plain T2VA presentations.
- `minimax_h3_train_network.py` requires the same flag to accept the resulting
  single-token latent caches.
- The latent cache rejects one-frame mode unless `--task t2va` is selected.
- The text cache rejects one-frame mode unless `--task t2va` is selected and
  rejects `--one_frame` together with `--teacher_conditions`.
- The trainer rejects one-frame mode unless `--task t2va` is selected and
  rejects `--one_frame` together with `--h3_teacher_matching`.
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
`--uncond_text`. An empty output path omits both arguments. A nonempty output
path with empty text omits `--uncond_text`, preserving upstream's tested
single-space unconditional probe. The image preset supplies a shared
unconditional-cache path used later by the trainer.

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
guidance loss scale requires a nonempty unconditional-cache path.
`audio_loss_weight` and both guidance scale values must be nonnegative;
`h3_guidance_loss_sigma_min` must be in `[0.0, 1.0]`. Empty optional numeric
values are omitted, while numeric zero is passed through. Defaults keep the
existing video behavior; the separate image preset carries the recommended
values.

## GUI And Command Design

### Command Builder

The cache builder passes `--one_frame` to both MiniMax-H3 cache jobs. This
requires explicit registration in the H3 latent-cache boolean whitelist and the
H3 text-cache boolean mapping. The train builder registers it in the H3 training
boolean whitelist. The model catalog lists it for the H3 cache and train pages.
Validation rejects one-frame mode unless the selected version and task resolve
to `version=fl2va` and `task=t2va`.

No validation requires an image-only project. This preserves mixed dataset
training. Existing guidance-cache validation remains responsible for rejecting
a positive guidance scale without a cache path.

Teacher matching remains upstream-only in this integration. The GUI and
PowerShell workflows do not emit `--teacher_conditions` or
`--h3_teacher_matching`; command building rejects enabled raw state for either
feature instead of silently dropping it. The remaining teacher-loss shaping
arguments are explicitly deferred with the other unrelated H3 experimental
flags introduced between gitlinks.

### Cache Page

The MiniMax-H3 cache card adds an experimental image-training toggle. It is
available only for the `t2va` task. Selecting `fl2va` or `ref2va` hides or
disables the control and clears its value so a stale flag cannot reach the
command builder.

`CacheStep._on_arch_change` must execute the H3 task-state synchronizer even
when architecture and version are unchanged. ModelSelector sends task changes
through the same callback, and the current early return would otherwise retain
the old one-frame value.

The existing unconditional-cache output and text controls remain the source of
the guidance cache. The new image cache preset enables one-frame mode and fills
the output path.

### Train Page

The MiniMax-H3 train card adds the matching experimental image-training toggle.
It follows the same task-dependent visibility and reset behavior as the cache
page. Applying the image preset sets the toggle, video-only mode, guidance loss,
guidance cache path, warmup, and one-frame sample prompt together.

The existing same-architecture task callback continues to call the H3 train
synchronizer. The synchronizer clears one-frame mode before a non-T2VA command
can be collected.

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
the `minimax_h3` marker and the dataset contains the `fp_1f_target_index` key,
including the explicit value `0`. This branch runs before FramePack inference.
Imports without that marker retain the existing FramePack behavior because the
TOML field alone does not identify which architecture will consume it. The GUI
stores the selected row template in project `interop` metadata so a manually
created H3 row remains identifiable without adding unsupported keys to the
exported dataset TOML. Dataset preview uses the import marker or that GUI-only
metadata and reports the H3 one-frame template.

Switching a row to the H3 template clears all hidden unsupported fields,
including control paths and sizing, FramePack clean/post fields,
`multiple_target`, and `no_resize_control`. A supplied target index must parse
as an integer greater than or equal to zero. Negative and nonnumeric values show
a validation error and block saving instead of being silently dropped. An empty
value omits the field and lets upstream use `0`. These rules apply to directory
and JSONL image sources.

The template, target-index field, and tooltip use H3-specific translation keys
in English, Simplified Chinese, Japanese, and Korean. The tooltip states that
the exported key is `fp_1f_target_index` and that its value is a zero-based
24 fps pixel-frame index; it does not reuse the FramePack label.

## Presets And Examples

Add these built-in artifacts:

- `gui/presets/cache/minimax_h3_image.toml`
- `gui/presets/train/minimax_h3_image.toml`
- `toml/qinglong_minimax_h3_image.toml`
- `toml/qinglong_minimaxh3_image.txt`

The cache and train image presets share the same unconditional-cache path, but
they do not own the active dataset. Native GUI commands always export the Step 1
project dataset to the project's generated `dataset_config.toml`. The sample
`toml/qinglong_minimax_h3_image.toml` is a separate Step 1 import artifact; the
user imports it or creates an equivalent project dataset before caching.

Preset application is a partial state merge, so each preset must write every
mode bit that needs resetting. All three existing MiniMax-H3 cache presets and
all three existing MiniMax-H3 train presets explicitly set `one_frame=false`.
The image cache preset sets `one_frame=true`,
`cache_latents_enabled=true`, and `cache_text_encoder_enabled=true`, then fills
the unconditional output path and resets `uncond_text=""`. Both image presets
explicitly set `arch="MiniMax-H3"`, `version="fl2va"`, and `task="t2va"` so
they override previously applied FL2VA or Ref2VA state. The image train preset
sets `one_frame=true`, `video_only=true`, guidance scale `4.0`, guidance sigma
minimum `0.15`, warmup steps `50`, `enable_sample=true`,
`sample_at_first=true`, the one-frame prompt path, and an image-specific
`output_name`. It resets `h3_guidance_loss_scale_audio=""` and any deferred
teacher-matching enable bit so stale custom state cannot change the recommended
loss. Existing preset labels and other video defaults stay unchanged.

The sample dataset uses a standard image directory, a separate image cache
directory, 1024 by 1024 resolution, batch size 1, bucketing, and target index 0.
The sample prompt uses `--f 1`, valid H3 dimensions, and an output seed suitable
for a smoke test.

## Data Flow

1. The user imports `toml/qinglong_minimax_h3_image.toml` on Step 1 or creates a
   standard image dataset and selects the MiniMax-H3 one-frame template.
2. The image cache preset enables both cache stages. The cache command builder
   exports the active project dataset and launches both H3 cache stages with
   `--task t2va --one_frame`.
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
- Commands built from existing video presets continue to omit `--one_frame` and
  retain their current dataset, audio, and sampling behavior. The preset files
  explicitly reset `one_frame=false` to achieve this after partial state merges.
- Preset application does not alter user-maintained TOML files.
- The committed submodule gitlink resolves to code that declares `--one_frame`
  on both cache parsers and the trainer.

## Test Strategy

- Extend the MiniMax-H3 PowerShell contract tests to assert that one-frame and
  guidance variables map to the exact upstream flags and that both cache stages
  receive `--one_frame`. Cover omitted empty values and preserved numeric zero.
- Add a clean-checkout submodule contract test that reads the committed gitlink
  and verifies all three upstream parsers declare `--one_frame`.
- Change the existing H3 upstream-flag coverage test to read every parser from
  the indexed gitlink rather than the submodule working tree. Maintain an
  explicit per-parser deferred map for unrelated `b462291` additions:
  - text cache: `--teacher_conditions`
  - trainer: `--h3_teacher_matching`, `--h3_teacher_conditions`,
    `--h3_teacher_condition_sigma_max`, `--h3_teacher_loss_dc_weight`,
    `--h3_teacher_loss_mag_weight`, `--h3_teacher_preservation_weight`,
    `--h3_timestep_focus_min`, `--h3_timestep_focus_max`,
    `--h3_timestep_focus_prob`, and `--h3_video_best_of_k`
  - generation: `--interactive`, `--ref`, `--trajectory_dir`,
    `--trajectory_stride`, `--lora_runtime_attach`, generation's
    `--one_frame`, `--from_file`, `--latent_path`, and `--bell`
  Coverage is parser-specific: a flag counts as supported only when the
  MiniMax-H3 builder for that parser can emit or validate it. Repository-wide
  string literals from other architectures do not count. All nondeferred parser
  flags must be present in the corresponding H3 builder contract. This feature
  supports training and cache `--one_frame`; it does not claim H3 generation
  `--one_frame` or the other deferred features.
- Add command-builder tests for the two one-frame cache jobs, the one-frame
  training job, invalid tasks, text-cache teacher-condition conflicts,
  trainer teacher-matching conflicts, missing guidance cache, numeric boundary
  behavior, and mixed-dataset acceptance.
- Add GUI contract and state tests for the new controls, task switching, preset
  application, translated labels/tooltips, and real ModelSelector task-change
  callbacks.
- Add preset tests proving the image presets contain the complete recommended
  state. Apply the image preset followed by each existing video preset and
  assert the final built command omits `--one_frame`. Also verify disabled cache
  stages are re-enabled by the image cache preset and image sampling uses a
  unique output name and `--f 1` prompt.
- Start from Ref2VA/FL2VA and custom unconditional/guidance state, then apply
  each image preset and assert that version, task, unconditional text, optional
  audio guidance, cache stages, and final command all match the image contract.
- Add dataset page tests for H3 template rendering, target-index validation,
  explicit zero, negative and nonnumeric input, JSONL input, template-switch
  cleanup, preview classification, GUI-only template metadata, and TOML round
  trips.
- Add an end-to-end GUI-state test that imports the standalone image dataset,
  applies the cache preset, applies the train preset, and builds the expected
  cache and training commands from the exported project dataset.
- Run the focused MiniMax-H3, preset, dataset, and script tests, followed by the
  complete GUI test suite.
- Launch the GUI and use browser screenshots at desktop and narrow viewports to
  verify that the added controls do not overlap or overflow.
