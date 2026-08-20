# Joint video and audio generation script for MiniMax-H3

$task = "t2va"

# Released model components
$fl2va_dit = "./ckpts/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
$ref2va_dit = "./ckpts/diffusion_models/minimax_h3_ref2va_bf16.safetensors"
$dit = if ($task -ieq "ref2va") { $ref2va_dit } else { $fl2va_dit }
$video_vae = "./ckpts/vae/minimax_h3_video_vae_fp16.safetensors"
$audio_vae = "./ckpts/vae/minimax_h3_audio_vae_fp32.safetensors"
$text_encoder = "./ckpts/text_encoder/qwen3vl_32b_minimax_h3_bf16.safetensors"
$text_cache = ""

# Task inputs
$prompt = "A singer performs under stage lights."
$first_frame = ""
$last_frame = ""
$reference_jsonl = ""
$reference_index = 0

# Geometry and sampling
$width = 768
$height = 1344
$frame_count = 124
$allow_experimental_duration = $False
$steps = 30
$seed = 42
$h3_shift_video = 12
$h3_shift_audio = 3
$h3_visual_cond_clean = 0.999
$h3_audio_cond_clean = 1.0
$output = "./output_dir/minimax_h3.mp4"

# Attention and memory
$attn_mode = "sdpa"
$split_attn = $False
$blocks_to_swap = 48
$use_pinned_memory_for_block_swap = $True
$device = ""
$disable_numpy_memmap = $False

# LoRA
$lora_weight = ""
$lora_multiplier = "1.0"

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")

if ($task -notin @("t2va", "fl2va", "ref2va")) {
    throw "MiniMax-H3 task must be t2va, fl2va, or ref2va."
}
if (($width -le 0) -or ($height -le 0) -or ($width % 32) -or ($height % 32)) {
    throw "MiniMax-H3 width and height must be positive multiples of 32."
}
if ($frame_count -le 0) {
    throw "MiniMax-H3 frame_count must be positive."
}
if (($frame_count - 5) % 17 -ne 0) {
    throw "MiniMax-H3 frame_count must satisfy 17*n+5."
}
$duration_seconds = $frame_count / 24.0
if (-not $allow_experimental_duration -and (($duration_seconds -lt 5) -or ($duration_seconds -gt 15))) {
    throw "MiniMax-H3 duration must remain in the released 5-15 second range unless experimental duration is enabled."
}
if (($h3_shift_video -le 0) -or ($h3_shift_audio -le 0)) {
    throw "MiniMax-H3 video and audio shifts must be positive."
}
if (($blocks_to_swap -lt 0) -or ($blocks_to_swap -gt 48)) {
    throw "MiniMax-H3 blocks_to_swap must be 0 through 48."
}
if ([System.IO.Path]::GetExtension($output).ToLowerInvariant() -notin @(".mp4", ".mkv", ".mov")) {
    throw "MiniMax-H3 output must use .mp4, .mkv, or .mov."
}

if ($env:OS -ilike "*windows*") {
    if (Test-Path "./venv/Scripts/activate") {
        ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:VSLANG = "1033"

$ext_args = [System.Collections.ArrayList]::new()
$script = "minimax_h3_generate_video.py"

if ($task -ieq "fl2va") {
    if (-not $prompt -or -not $first_frame -or -not $last_frame) {
        throw "MiniMax-H3 FL2VA requires prompt, first_frame, and last_frame."
    }
    [void]$ext_args.Add("--prompt=$prompt")
    [void]$ext_args.Add("--first_frame=$first_frame")
    [void]$ext_args.Add("--last_frame=$last_frame")
}
elseif ($task -ieq "ref2va") {
    if (-not $reference_jsonl) {
        throw "MiniMax-H3 Ref2VA requires reference_jsonl."
    }
    if ($reference_index -lt 0) {
        throw "MiniMax-H3 reference_index must be nonnegative."
    }
    [void]$ext_args.Add("--reference_jsonl=$reference_jsonl")
    [void]$ext_args.Add("--reference_index=$reference_index")
    if ($prompt) {
        [void]$ext_args.Add("--prompt=$prompt")
    }
}
else {
    if (-not $prompt) {
        throw "MiniMax-H3 T2VA requires a prompt."
    }
    [void]$ext_args.Add("--prompt=$prompt")
}

if ($text_cache) {
    if ($task -ieq "fl2va") {
        throw "MiniMax-H3 FL2VA cannot use a dataset text cache."
    }
    [void]$ext_args.Add("--text_cache=$text_cache")
}
else {
    [void]$ext_args.Add("--text_encoder=$text_encoder")
}

if ($allow_experimental_duration) {
    [void]$ext_args.Add("--allow_experimental_duration")
}
if ($device) {
    [void]$ext_args.Add("--device=$device")
}
if ($attn_mode -ine "torch") {
    [void]$ext_args.Add("--attn_mode=$attn_mode")
}
if ($split_attn) {
    [void]$ext_args.Add("--split_attn")
}
if ($blocks_to_swap) {
    [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
    if ($use_pinned_memory_for_block_swap) {
        [void]$ext_args.Add("--use_pinned_memory_for_block_swap")
    }
}
if ($disable_numpy_memmap) {
    [void]$ext_args.Add("--disable_numpy_memmap")
}

$loraWeights = @(
    $lora_weight -split "[`r`n;]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
$loraMultipliers = @(
    $lora_multiplier -split "[`r`n; ]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($loraWeights.Count -gt 0) {
    if ($loraMultipliers.Count -gt $loraWeights.Count) {
        throw "MiniMax-H3 has more LoRA multipliers than weights."
    }
    [void]$ext_args.Add("--lora_weight")
    foreach ($weight in $loraWeights) {
        [void]$ext_args.Add($weight)
    }
    if ($loraMultipliers.Count -gt 0) {
        [void]$ext_args.Add("--lora_multiplier")
        foreach ($multiplier in $loraMultipliers) {
            [void]$ext_args.Add($multiplier)
        }
    }
}

Write-Output "Extended arguments:"
$ext_args | ForEach-Object { Write-Output "  $_" }

# Legacy generators use --vae=$vae and --save_path=$save_path; H3 uses dual VAEs and --output.
python "./musubi-tuner/$script" `
    --task=$task `
    --dit=$dit `
    --video_vae=$video_vae `
    --audio_vae=$audio_vae `
    --width=$width `
    --height=$height `
    --frame_count=$frame_count `
    --steps=$steps `
    --seed=$seed `
    --h3_shift_video=$h3_shift_video `
    --h3_shift_audio=$h3_shift_audio `
    --h3_visual_cond_clean=$h3_visual_cond_clean `
    --h3_audio_cond_clean=$h3_audio_cond_clean `
    --output=$output $ext_args
Assert-NativeCommandSucceeded "Command failed: 5.11minimax_h3_generate.ps1"

Write-Output "MiniMax-H3 generation finished"
Read-Host | Out-Null
