# Mage-Flow Scripts and GUI Integration Design

Date: 2026-07-26
Status: Approved for implementation planning

## Summary

Add Mage-Flow as a first-class model family in the PowerShell workflow and
NiceGUI wizard. The integration covers text-to-image and image editing, supports
Standard and Turbo BF16 checkpoints, and keeps the three workflow stages
consistent through one explicit `is_edit` mode value.

This is a native integration rather than a compatibility wrapper. Cache and
training can reuse the existing shared command-building structure where the CLI
contracts match. Generation receives a dedicated command builder because
Mage-Flow uses a file output, explicit dimensions, and repeated control-image
arguments that do not match the generic generation contract.

## Upstream Baseline

The design targets the checked-in `musubi-tuner` submodule at commit `3962a5a`
on its `qinglong` branch.

The current Mage-Flow implementation has these relevant characteristics:

- Processor, tokenizer, chat-template, and image-processor assets are resolved
  automatically from the pinned `microsoft/Mage-Flow/text_encoder` repository.
- The Mage-Flow CLIs do not accept `--processor` or `--tokenizer`.
- Text-to-image and Edit modes are explicit through `--is_edit`.
- Edit generation accepts one to three repeated `--control_image` arguments.
- LoRA training is supported; full fine-tuning is not.
- Training is BF16-oriented and supports scaled FP8 base weights, checkpointing,
  block swapping, compile, SDPA, and optional FlashAttention 2 within the
  upstream validation rules.
- Native packing, sharded model directories, inferred mode identity,
  SageAttention, xformers, FlashAttention 3/4, watermarking, and ComfyUI INT8
  ConvRot checkpoints are outside the supported surface.
- Released-weight parity is still described upstream as experimental, so this
  integration must not claim end-to-end validation against real model weights.

## Goals

- Add PowerShell entry points for latent/text caching, LoRA training, and
  generation.
- Support text-to-image and Edit workflows with one consistent mode identity.
- Support Standard and Turbo BF16 DiT variants from
  `Comfy-Org/Mage-Flow`.
- Support one to three ordered reference images for Edit generation.
- Add Mage-Flow to the GUI model catalog, wizard pages, presets, command
  construction, documentation, and script coverage metadata.
- Add installer choices for supported BF16 checkpoints and shared components.
- Reject invalid or unsupported combinations before starting a native process.
- Preserve all existing model-family behavior.

## Non-Goals

- Downloading large model files during automated tests.
- Claiming real-checkpoint inference or training parity.
- Supporting full-model fine-tuning.
- Supporting ComfyUI INT8 ConvRot checkpoints.
- Reintroducing processor or tokenizer fields removed by upstream.
- Adding Base checkpoints that are not present in the selected Comfy repository.
- Generalizing every model family behind a new command-builder abstraction.
- Supporting unsupported attention backends or generation features.

## Model Repository Mapping

Use `https://huggingface.co/Comfy-Org/Mage-Flow` as the user-facing download
source. Expose only the supported BF16 files.

| Mode | Variant | Repository file | Local target |
| --- | --- | --- | --- |
| T2I | Standard | `diffusion_models/mage_flow_bf16.safetensors` | `ckpts/diffusion_models/mage_flow_bf16.safetensors` |
| T2I | Turbo | `diffusion_models/mage_flow_turbo_bf16.safetensors` | `ckpts/diffusion_models/mage_flow_turbo_bf16.safetensors` |
| Edit | Standard | `diffusion_models/mage_flow_edit_bf16.safetensors` | `ckpts/diffusion_models/mage_flow_edit_bf16.safetensors` |
| Edit | Turbo | `diffusion_models/mage_flow_edit_turbo_bf16.safetensors` | `ckpts/diffusion_models/mage_flow_edit_turbo_bf16.safetensors` |
| Shared | VAE | `vae/mage_flow_vae_bf16.safetensors` | `ckpts/vae/mage_flow_vae_bf16.safetensors` |
| Shared | Text encoder | `text_encoders/qwen3vl_4b_bf16.safetensors` | `ckpts/text_encoder/qwen3vl_4b_bf16.safetensors` |

The text encoder is intentionally placed in the repository's existing local
`text_encoder` convention even though the Hugging Face source directory is
plural.

The installer adds an interactive Mage-Flow section using the existing
`DownloadHfFile` helper. Selecting any DiT downloads the shared VAE and text
encoder once. The installer prompt should allow selecting individual variants
or all four, and skipping remains the default low-cost path.

The following repository files remain hidden and unsupported:

- `mage_flow_int8_convrot.safetensors`
- `mage_flow_turbo_int8_convrot.safetensors`
- `mage_flow_edit_int8_convrot.safetensors`
- `mage_flow_edit_turbo_int8_convrot.safetensors`

## Workflow Identity And Defaults

The GUI stores the mode as T2I or Edit and serializes it to a boolean
`is_edit`. Variant is stored independently as Standard or Turbo.

| Mode | Variant | Default steps | Default CFG | Control images |
| --- | --- | ---: | ---: | --- |
| T2I | Standard | 20 | 5.0 | Forbidden |
| T2I | Turbo | 4 | 1.0 | Forbidden |
| Edit | Standard | 30 | 5.0 | Required, 1-3 |
| Edit | Turbo | 4 | 1.0 | Required, 1-3 |

Changing mode or variant loads the matching recommended generation defaults.
The fields remain editable after the recommendation is applied. Mode changes
also switch the recommended DiT path to the corresponding checkpoint.

Cache and training do not infer the mode from the checkpoint filename. Their
`is_edit` value must be explicit and must match each other. Generation also
passes the explicit mode and validates the selected inputs.

## PowerShell Scripts

### `2.10mage_flow_cache_latent_and_text_encoder.ps1`

The script follows the repository's native PowerShell wrapper conventions:

- Activate `.venv`.
- Source `powershell/native_command.ps1`.
- Store editable values above the `DO NOT MODIFY` marker so GUI preset
  extraction can parse them.
- Build argument arrays rather than concatenated command strings.
- Check each native process through `Assert-NativeCommandSucceeded`.

It runs these modules in order:

1. `musubi_tuner.mage_flow_cache_latents`
2. `musubi_tuner.mage_flow_cache_text_encoder_outputs`

Both commands receive the same `--is_edit` decision. The latent command also
receives `--seed`; text caching receives the required text-encoder path. VAE
and text-encoder dtype defaults are BF16. There is no processor parameter.

### `3.10mage_flow_train_lora.ps1`

The script invokes `musubi_tuner.mage_flow_train_network` and fixes the network
module to:

`musubi_tuner.networks.lora_mage_flow`

Defaults align with the upstream trainer:

- `mixed_precision=bf16`
- `timestep_sampling=shift`
- `discrete_flow_shift=6`
- `weighting_scheme=none`
- VAE dtype BF16
- attention mode SDPA unless FlashAttention 2 is explicitly selected

The script exposes mode, DiT, VAE, optional text encoder for sampling, LoRA
dimension/alpha, optimizer and schedule settings, block swapping, compile,
checkpointing, scaled FP8, sampling, resume, and output settings that the
upstream trainer actually accepts.

It rejects these combinations before launch:

- Network modules other than the fixed Mage-Flow LoRA module.
- Full-model or LyCORIS training.
- Attention modes other than SDPA and FlashAttention 2.
- `blocks_to_swap` outside 0 through 10.
- `fp8_base` without `fp8_scaled`.
- Fullgraph compilation; Mage-Flow supports compile only without fullgraph.
- `dim_from_weights` without network weights.

`--allow_mage_architecture_mismatch` is available as an explicit advanced
escape hatch, not enabled by default.

### `5.10mage_flow_generate.ps1`

The script invokes `musubi_tuner.mage_flow_generate_image` with a dedicated
parameter surface:

- Required DiT, VAE, text encoder, and prompt.
- Optional negative prompt, defaulting to a single space as upstream does.
- Output file through `--output`, not a save directory.
- Explicit `--is_edit`.
- Repeated `--control_image` for ordered Edit references.
- Optional width and height as a pair.
- Edit-only `--max_size`.
- Steps, CFG scale, flow shift, seed, device, dtype, and attention mode.
- Optional CFG renormalization.
- Zero or more LoRA weights and matching optional multipliers.
- Optional architecture-mismatch override.

The script validates that Edit has one to three reference images, T2I has none,
width and height are supplied together, steps and flow shift are positive, and
LoRA multipliers never outnumber LoRA weights.

It does not expose generic-only options such as `--save_path`,
`--output_type`, prompt files, latent decode, block swap, compile, or FP8
generation flags.

## GUI Integration

### Model Catalog

Add a `Mage-Flow` entry with architecture ID `mage_flow`. Its required paths are
DiT, VAE, and text encoder. The catalog provides the four supported default DiT
paths and the two shared component paths.

### Cache Page

Add an architecture-specific Mage-Flow section containing:

- A T2I/Edit segmented mode control.
- Qwen3-VL text-encoder path.
- Latent cache seed.

The VAE remains in the common model-path section. Saving a preset records the
mode explicitly. Loading either built-in preset restores the correct mode.

### Training Page

Add Mage-Flow-specific controls for mode and supported checkpoint variant. The
text encoder is presented as optional unless sampling is enabled.

For Mage-Flow:

- The network module is fixed and not user-editable.
- LyCORIS and unsupported target-pattern controls are hidden or disabled.
- Attention choices are limited to SDPA and FlashAttention 2.
- The block-swap range is limited to 0 through 10.
- Invalid FP8, compile, weight-derived-dimension, and sampling combinations
  produce actionable validation errors.

### Generation Page

Add architecture-specific controls for:

- T2I/Edit mode.
- Standard/Turbo variant.
- Output file.
- Ordered control-image paths.
- Width and height.
- Edit maximum size.
- Dtype.
- CFG renormalization.
- Architecture-mismatch override.

The mode and variant controls update the recommended DiT path, step count, and
CFG value using the workflow table above. Unsupported generic controls are
hidden or disabled for Mage-Flow so the visible UI matches the command that
will run.

### Command Construction

Cache and training extend the existing shared builders only where their
contracts match existing workflows.

Generation branches early to `_build_mage_flow_generate_job`. This builder:

- Emits `--output` instead of `--save_path`.
- Emits `--width` and `--height` only as a pair.
- Converts ordered control-image entries into repeated arguments.
- Omits every unsupported generic generation option.
- Applies all mode-specific and numerical validation before producing a job.

No command in any stage emits `--processor` or `--tokenizer`.

## Presets And Metadata

Add built-in presets:

- Cache: T2I and Edit.
- Training: T2I and Edit.
- Generation: T2I Standard, T2I Turbo, Edit Standard, and Edit Turbo.

Register all three scripts in the script preset catalog and native GUI coverage
manifest. Extend parser key mappings only for real editable script parameters,
including mode, variant, cache seed, output file, ordered control images,
maximum size, dtype, CFG renormalization, and architecture mismatch.

Update model lists and Mage-Flow parameter guidance in the root README, GUI
README translations, and parameter reference. Documentation must label the
integration experimental and link to both the Comfy checkpoint repository and
the checked-in upstream Mage-Flow documentation.

## Error Handling

Validation errors must describe the invalid field and the allowed correction.
Failures from the native processes continue to use the repository's common
native-command error path, preserving the command name and exit code.

The GUI must reject invalid Mage-Flow state without constructing a partial job.
It must not silently drop user-supplied control images, mismatched dimensions,
or unsupported attention choices.

## Verification Strategy

Automated verification will include:

- Parse every new PowerShell script with the PowerShell AST parser without
  executing model code.
- Verify model catalog paths, GUI defaults, preset loading, script sources, and
  coverage-manifest entries.
- Verify cache T2I and Edit jobs pass the same mode flag to both cache commands.
- Verify training fixes the network module and rejects unsupported combinations.
- Verify every generation mode/variant combination selects the expected DiT,
  steps, and CFG recommendation.
- Verify Edit emits one to three ordered repeated control-image arguments and
  T2I rejects them.
- Verify Mage-Flow generation uses `--output` and never emits generic
  `--save_path`.
- Verify no Mage-Flow command emits `--processor` or `--tokenizer`.
- Run the complete GUI unit test suite.
- Run relevant lightweight Mage-Flow tests in the checked-in submodule.
- Perform NiceGUI desktop and mobile visual smoke checks to catch clipped,
  overlapping, or stale architecture-specific controls.

Large checkpoint downloads and real-weight inference or training are explicitly
excluded from automated verification. The final report must state this residual
risk instead of implying otherwise.

## Acceptance Criteria

The work is acceptable when:

- Users can install any supported BF16 Mage-Flow checkpoint combination.
- Users can configure and launch cache, LoRA train, and generation workflows
  from both PowerShell scripts and the GUI.
- T2I/Edit identity remains explicit and consistent across all stages.
- Edit generation accepts exactly one to three ordered references.
- Standard/Turbo recommendations match the workflow table while remaining
  editable.
- Unsupported upstream features cannot leak into generated commands.
- Existing model-family tests remain green.
- Documentation accurately states the experimental and unverified-real-weight
  limitations.
