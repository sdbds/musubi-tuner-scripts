# Image generation script for Mage-Flow

$generate_mode = "mage_flow"
$is_edit = $False
$model_variant = "standard"

# Model
$dit = "./ckpts/diffusion_models/mage_flow_bf16.safetensors"
$vae = "./ckpts/vae/mage_flow_vae_bf16.safetensors"
$text_encoder = "./ckpts/text_encoder/qwen_3_VL_4b.safetensors"

# Prompt and output
$prompt = "A cinematic portrait with natural light, intricate detail"
$negative_prompt = " "
$mage_output_path = "./output_dir/mage_flow.png"

# Mage-Flow mode and sampler
$mage_control_images = ""
$mage_width = 1024
$mage_height = 1024
$mage_max_size = $null
$mage_steps = 20
$mage_cfg_scale = 5.0
$mage_flow_shift = 6.0
$mage_seed = 42
$mage_device = ""
$mage_dtype = "bfloat16"
$mage_attn_mode = "sdpa"
$mage_renormalize_cfg = $False

# LoRA
$mage_allow_architecture_mismatch = $False
$mage_lora_weights = ""
$mage_lora_multipliers = ""

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
$Env:VSLANG = "1033"

$controlImages = @(
    $mage_control_images -split "[`r`n;]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($is_edit -and (($controlImages.Count -lt 1) -or ($controlImages.Count -gt 3))) {
    throw "Mage-Flow Edit requires one to three ordered control images."
}
if (-not $is_edit -and $controlImages.Count -gt 0) {
    throw "Mage-Flow T2I does not accept control images."
}
if (($null -eq $mage_width) -xor ($null -eq $mage_height)) {
    throw "Mage-Flow width and height must be supplied together."
}
if (($null -ne $mage_width) -and (($mage_width -le 0) -or ($mage_height -le 0))) {
    throw "Mage-Flow width and height must be positive."
}
if (-not $is_edit -and $null -ne $mage_max_size) {
    throw "Mage-Flow max_size is Edit-only."
}
if (($null -ne $mage_max_size) -and ($mage_max_size -le 0)) {
    throw "Mage-Flow max_size must be positive."
}
if ($mage_steps -le 0) {
    throw "Mage-Flow steps must be positive."
}
if ($mage_flow_shift -le 0) {
    throw "Mage-Flow flow_shift must be positive."
}
if ($mage_dtype -notin @("bfloat16", "float16", "float32")) {
    throw "Mage-Flow dtype must be bfloat16, float16, or float32."
}
if ($mage_attn_mode -notin @("sdpa", "flash2")) {
    throw "Mage-Flow generation supports sdpa or flash2 attention."
}
if ($model_variant -notin @("standard", "turbo")) {
    throw "Mage-Flow model_variant must be standard or turbo."
}
if (-not $prompt) {
    throw "Mage-Flow prompt is required."
}
if (-not $mage_output_path) {
    throw "Mage-Flow output path is required."
}

$loraWeights = @(
    $mage_lora_weights -split "[`r`n;]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
$loraMultipliers = @(
    $mage_lora_multipliers -split "[`r`n;]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($loraMultipliers.Count -gt $loraWeights.Count) {
    throw "Mage-Flow LoRA multipliers cannot outnumber LoRA weights."
}

$script = "mage_flow_generate_image.py"
$ext_args = [System.Collections.ArrayList]::new()

if ($is_edit) {
    [void]$ext_args.Add("--is_edit")
}
foreach ($controlImage in $controlImages) {
    [void]$ext_args.Add("--control_image=$controlImage")
}
if ($null -ne $mage_width) {
    [void]$ext_args.Add("--width=$mage_width")
    [void]$ext_args.Add("--height=$mage_height")
}
if ($null -ne $mage_max_size) {
    [void]$ext_args.Add("--max_size=$mage_max_size")
}
if ($mage_device) {
    [void]$ext_args.Add("--device=$mage_device")
}
if ($mage_renormalize_cfg) {
    [void]$ext_args.Add("--renormalize_cfg")
}
if ($mage_allow_architecture_mismatch) {
    [void]$ext_args.Add("--allow_mage_architecture_mismatch")
}
if ($loraWeights.Count -gt 0) {
    [void]$ext_args.Add("--lora_weight")
    foreach ($loraWeight in $loraWeights) {
        [void]$ext_args.Add($loraWeight)
    }
}
if ($loraMultipliers.Count -gt 0) {
    [void]$ext_args.Add("--lora_multiplier")
    foreach ($loraMultiplier in $loraMultipliers) {
        [void]$ext_args.Add($loraMultiplier)
    }
}

Write-Output "Extended arguments:"
$ext_args | ForEach-Object { Write-Output "  $_" }

python "./musubi-tuner/$script" `
    --dit=$dit `
    --vae=$vae `
    --text_encoder=$text_encoder `
    --prompt=$prompt `
    --negative_prompt=$negative_prompt `
    --output=$mage_output_path `
    --steps=$mage_steps `
    --cfg_scale=$mage_cfg_scale `
    --flow_shift=$mage_flow_shift `
    --seed=$mage_seed `
    --dtype=$mage_dtype `
    --attn_mode=$mage_attn_mode $ext_args
Assert-NativeCommandSucceeded "Command failed: 5.10mage_flow_generate.ps1"

Write-Output "Mage-Flow generation finished"
Read-Host | Out-Null
