# MiniMax-H3 Text Encoder Download Submenu Design

Date: 2026-08-05

Status: Approved for implementation

## Goal

Move MiniMax-H3 text-encoder selection into a second-level installer menu so
users can choose the precision and source independently from the selected DiT.

## Scope

- Keep the existing first-level MiniMax-H3 DiT menu unchanged.
- Show the text-encoder menu only after at least one DiT is selected.
- Offer official BF16, official INT8 ConvRot, and Ultra-Heretic H3 INT8
  ConvRot conditioning encoders.
- Treat an empty text-encoder response as option `2`, the official INT8
  ConvRot encoder.
- Keep `n` as an explicit way to skip text-encoder download.
- Continue downloading the shared video and audio VAEs once for every valid
  DiT selection.

## Menu Contract

The second-level menu is `[1/2/3/n]`:

| Choice | Repository | Source file | Local target |
| --- | --- | --- | --- |
| `1` | `Comfy-Org/MiniMax-H3` | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | `text_encoder/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| `2` or empty | `Comfy-Org/MiniMax-H3` | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | `text_encoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `3` | `ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot` | `qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` | `text_encoder/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` |
| `n` | none | none | none |

The third-party repository's optional
`qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors` is not
downloaded. It is a ComfyUI prompt-enhancement component, not an H3
conditioning requirement.

## Download Data Flow

The menu resolves the selected text encoder into one optional hashtable with
`RepoId`, `FilePath`, and `TargetPath`. `DownloadMiniMaxH3Model` accepts that
descriptor alongside `DiffusionFiles` and downloads, in order:

1. Selected DiT files.
2. The selected text encoder, when present.
3. The shared video and audio VAEs.

Keeping repository and path data in the descriptor avoids global menu state
inside the download function and supports the external repository without a
special-case branch in the downloader.

## Compatibility And Errors

- The menu identifies ConvRot entries as download options and preserves the
  current MiniMax-H3 loader compatibility warning.
- An unknown text-encoder response falls back to option `2`, matching the
  documented default.
- Explicit `n` skips only the text encoder; selected DiTs and VAEs still
  download.
- Existing `DownloadModelComponent` failure handling remains authoritative for
  all repositories and target paths.

## Tests

Installer contract tests verify:

- The second-level prompt is nested under a valid DiT selection.
- Empty and unknown input resolve to official INT8 ConvRot.
- Choices `1`, `2`, and `3` contain the exact repository, source, and target
  paths.
- Choice `n` yields no text-encoder descriptor.
- `DownloadMiniMaxH3Model` receives the descriptor and downloads shared VAEs
  only once.
- The optional generation tail is absent from the installer.

PowerShell parsing and scoped whitespace checks complete verification.

## Non-Goals

This change does not switch GUI, training-script, or preset runtime defaults
to ConvRot INT8. Runtime integration remains separate from installer download
selection.
