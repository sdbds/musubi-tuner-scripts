# Cache script for Mage-Flow

$cache_mode = "mage_flow"
$dataset_config = "./toml/qinglong-qwen-image-datasets.toml"

# Latent cache
$vae = "./ckpts/vae/mage_flow_vae_bf16.safetensors"
$vae_dtype = "bfloat16"
$batch_size = ""
$device = ""
$num_workers = 0
$skip_existing = $False
$cache_seed = 0

# Text encoder cache
$text_encoder = "./ckpts/text_encoder/qwen_3_VL_4b.safetensors"
$text_encoder_dtype = "bfloat16"
$text_encoder_batch_size = 16
$text_encoder_device = ""
$text_encoder_num_workers = 0
$text_encoder_skip_existing = $False

# Mage-Flow mode
$is_edit = $False
$model_variant = "standard"

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")

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

if ($is_edit) {
    [void]$latent_args.Add("--is_edit")
    [void]$text_args.Add("--is_edit")
}
[void]$latent_args.Add("--seed=$cache_seed")
[void]$text_args.Add("--text_encoder=$text_encoder")
[void]$text_args.Add("--text_encoder_dtype=$text_encoder_dtype")

if ($batch_size) {
    [void]$latent_args.Add("--batch_size=$batch_size")
}
if ($device) {
    [void]$latent_args.Add("--device=$device")
}
if ($num_workers -ne 0) {
    [void]$latent_args.Add("--num_workers=$num_workers")
}
if ($skip_existing) {
    [void]$latent_args.Add("--skip_existing")
}

if ($text_encoder_batch_size) {
    [void]$text_args.Add("--batch_size=$text_encoder_batch_size")
}
if ($text_encoder_device) {
    [void]$text_args.Add("--device=$text_encoder_device")
}
if ($text_encoder_num_workers -ne 0) {
    [void]$text_args.Add("--num_workers=$text_encoder_num_workers")
}
if ($text_encoder_skip_existing) {
    [void]$text_args.Add("--skip_existing")
}

python "./musubi-tuner/mage_flow_cache_latents.py" `
    --dataset_config=$dataset_config `
    --vae=$vae `
    --vae_dtype=$vae_dtype $latent_args
Assert-NativeCommandSucceeded "Command failed: 2.10mage_flow_cache_latent_and_text_encoder.ps1"

python "./musubi-tuner/mage_flow_cache_text_encoder_outputs.py" `
    --dataset_config=$dataset_config $text_args
Assert-NativeCommandSucceeded "Command failed: 2.10mage_flow_cache_latent_and_text_encoder.ps1"

Write-Output "Mage-Flow cache finished"
Read-Host | Out-Null
