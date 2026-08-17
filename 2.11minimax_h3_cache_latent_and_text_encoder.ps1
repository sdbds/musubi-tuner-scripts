# Cache script for MiniMax-H3

$task = "t2va"
$dataset_config = "./toml/qinglong-video-datasets.toml"

# Released model components
$video_vae = "./ckpts/vae/minimax_h3_video_vae_fp16.safetensors"
$audio_vae = "./ckpts/vae/minimax_h3_audio_vae_fp32.safetensors"
$text_encoder = "./ckpts/text_encoder/qwen3vl_32b_minimax_h3_bf16.safetensors"

# Cache settings
$cache_seed = 0
$batch_size = ""
$num_workers = 0
$skip_existing = $False
$keep_cache = $False
$disable_mmap = $False
$text_cache_dtype = "bf16"

# One-frame image training
$one_frame = $False
$uncond_output = ""
$uncond_text = ""

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")

if ($task -notin @("t2va", "fl2va", "ref2va")) {
    throw "MiniMax-H3 task must be t2va, fl2va, or ref2va."
}
if ($one_frame -and $task -notin @("t2va", "fl2va")) {
    throw "MiniMax-H3 one-frame cache mode requires task=t2va or task=fl2va."
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

$latent_args = [System.Collections.ArrayList]::new()
$text_args = [System.Collections.ArrayList]::new()

if ($batch_size) {
    [void]$latent_args.Add("--batch_size=$batch_size")
    [void]$text_args.Add("--batch_size=$batch_size")
}
if ($num_workers -ne 0) {
    [void]$latent_args.Add("--num_workers=$num_workers")
    [void]$text_args.Add("--num_workers=$num_workers")
}
if ($skip_existing) {
    [void]$latent_args.Add("--skip_existing")
    [void]$text_args.Add("--skip_existing")
}
if ($keep_cache) {
    [void]$latent_args.Add("--keep_cache")
    [void]$text_args.Add("--keep_cache")
}
if ($disable_mmap) {
    [void]$latent_args.Add("--disable_mmap")
    [void]$text_args.Add("--disable_mmap")
}
if ($one_frame) {
    [void]$latent_args.Add("--one_frame")
    [void]$text_args.Add("--one_frame")
}
if ($uncond_output -ne "") {
    [void]$text_args.Add("--uncond_output=$uncond_output")
    if ($uncond_text -ne "") {
        [void]$text_args.Add("--uncond_text=$uncond_text")
    }
}

# MiniMax-H3 uses --video_vae/--audio_vae instead of the shared --vae= interface.
python "./musubi-tuner/minimax_h3_cache_latents.py" `
    --dataset_config=$dataset_config `
    --task=$task `
    --video_vae=$video_vae `
    --audio_vae=$audio_vae `
    --cache_seed=$cache_seed $latent_args
Assert-NativeCommandSucceeded "Command failed: 2.11minimax_h3_cache_latent_and_text_encoder.ps1 latent cache"

python "./musubi-tuner/minimax_h3_cache_text_encoder_outputs.py" `
    --dataset_config=$dataset_config `
    --task=$task `
    --text_encoder=$text_encoder `
    --text_cache_dtype=$text_cache_dtype $text_args
Assert-NativeCommandSucceeded "Command failed: 2.11minimax_h3_cache_latent_and_text_encoder.ps1 text cache"

Write-Output "MiniMax-H3 cache finished"
Read-Host | Out-Null
