#Generate images script by @bdsqlsz

#Generate Mode (FLUX.2)
$generate_mode = "flux2"

# Parameters from flux_2_generate_image.py
$dit = "./ckpts/diffusion_models/flux2-dev.safetensors"                          # DiT directory | DiT路径
$vae = "./ckpts/vae/ae.safetensors"                                              # VAE directory | VAE路径

# FLUX.2 specific parameters
$model_version = "dev"                                                           # model version: dev | klein-4b | klein-base-4b | klein-9b | klein-base-9b
$text_encoder = "./ckpts/text_encoder/mistral3_model.safetensors"                # Text Encoder (Mistral 3 or Qwen 3) directory
$fp8_text_encoder = $false                                                       # use fp8 for Text Encoder model

# LoRA
$lora_weight = ""                                                                # LoRA weight path
$lora_multiplier = "1.0"                                                         # LoRA multiplier
$include_patterns = ""                                                           # LoRA module include patterns
$exclude_patterns = ""                                                           # LoRA module exclude patterns
$save_merged_model = $false                                                      # save merged model. If specified, no inference will be performed

$prompt = "A beautiful anime girl with long silver hair, blue eyes, wearing a white dress, masterpiece, best quality"
$negative_prompt = ""                                                            # negative prompt
$from_file = ""                                                                  # Read prompts from a file
$image_size = "1024 1024"                                                        # image size (height width)
$infer_steps = 50                                                                # number of inference steps, default is 50 for FLUX.2 dev
$save_path = "./output_dir"                                                      # path to save generated image(s)
$seed = 1026                                                                     # Seed for evaluation.
$guidance_scale = 4.0                                                            # guidance scale for CFG. Default 4.0 for FLUX.2 dev
$embedded_cfg_scale = 4.0                                                        # Embedded CFG scale (distilled CFG Scale), default is 4.0

# Control Image for image editing
$control_image_path = ""                                                         # path to control (reference) image(s) for Flux 2 image edit
$no_resize_control = $false                                                      # do not resize control image

# Flow Matching
$flow_shift = $null                                                              # Shift factor for flow matching schedulers. Default is None (FLUX.2 default)
$fp8 = $false                                                                    # use fp8 for DiT model
$fp8_scaled = $true                                                              # use fp8 scaled for DiT model

$device = ""                                                                     # device to use for inference
$attn_mode = "sageattn"                                                          # attention mode (torch, sdpa, xformers, sageattn, flash)
$blocks_to_swap = 0                                                              # number of blocks to swap in the model
$use_pinned_memory_for_block_swap = $true                                        # use pinned memory for block swapping
$output_type = "images"                                                          # output type (images, latent, latent_images)
$no_metadata = $false                                                            # do not save metadata
$latent_path = ""                                                                # path to latent for decode. no inference
$lycoris = $false

$compile = $false                                                                # Enable torch.compile
$compile_backend = "inductor"
$compile_mode = "default"
$compile_fullgraph = $false
$compile_dynamic = $true
$compile_cache_size_limit = 32
$enable_tf32 = $true

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
    if ($compile) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        $vsPath = & $vswhere -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        & (Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1") -Arch amd64
        Set-Location $PSScriptRoot
    }
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
if ($enable_tf32) {
    $Env:NVIDIA_TF32_OVERRIDE = "1"
}
else {
    Remove-Item Env:NVIDIA_TF32_OVERRIDE -ErrorAction SilentlyContinue
}
$Env:VSLANG = "1033"
$ext_args = [System.Collections.ArrayList]::new()
$script = "flux_2_generate_image.py"

# FLUX.2 specific arguments
[void]$ext_args.Add("--model_version=$model_version")
[void]$ext_args.Add("--text_encoder=$text_encoder")

if ($fp8_text_encoder) {
    [void]$ext_args.Add("--fp8_text_encoder")
}

if ($fp8_scaled) {
    [void]$ext_args.Add("--fp8_scaled")
}
elseif ($fp8) {
    [void]$ext_args.Add("--fp8")
}

if ($device) {
    [void]$ext_args.Add("--device=$device")
}

if ($attn_mode -ine "torch") {
    [void]$ext_args.Add("--attn_mode=$attn_mode")
}

if ($blocks_to_swap -ne 0) {
    [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
    if ($use_pinned_memory_for_block_swap) {
        [void]$ext_args.Add("--use_pinned_memory_for_block_swap")
    }
}

if ($output_type -ne "images") {
    [void]$ext_args.Add("--output_type=$output_type")
}

if ($no_metadata) {
    [void]$ext_args.Add("--no_metadata")
}

if ($latent_path) {
    [void]$ext_args.Add("--latent_path=$latent_path")
}

if ($seed) {
    [void]$ext_args.Add("--seed=$seed")
}

if ($flow_shift -ne $null) {
    [void]$ext_args.Add("--flow_shift=$flow_shift")
}

if ($guidance_scale -ne 4.0) {
    [void]$ext_args.Add("--guidance_scale=$guidance_scale")
}

if ($embedded_cfg_scale -ne 4.0) {
    [void]$ext_args.Add("--embedded_cfg_scale=$embedded_cfg_scale")
}

if ($negative_prompt) {
    [void]$ext_args.Add("--negative_prompt=$negative_prompt")
}

# Control image parameters
if ($control_image_path) {
    [void]$ext_args.Add("--control_image_path")
    foreach ($ci in $control_image_path.Split(" ")) {
        [void]$ext_args.Add($ci)
    }
    if ($no_resize_control) {
        [void]$ext_args.Add("--no_resize_control")
    }
}

if ($compile) {
    [void]$ext_args.Add("--compile")
    if ($compile_backend) {
        [void]$ext_args.Add("--compile_backend=$compile_backend")
    }
    if ($compile_mode) {
        [void]$ext_args.Add("--compile_mode=$compile_mode")
    }
    if ($compile_fullgraph) {
        [void]$ext_args.Add("--compile_fullgraph")
    }
    if ($compile_dynamic) {
        [void]$ext_args.Add("--compile_dynamic=$compile_dynamic")
    }
    if ($compile_cache_size_limit) {
        [void]$ext_args.Add("--compile_cache_size_limit=$compile_cache_size_limit")
    }
}

if ($lora_weight) {
    [void]$ext_args.Add("--lora_weight")
    foreach ($lw in $lora_weight.Split(" ")) {
        [void]$ext_args.Add($lw)
    }
    [void]$ext_args.Add("--lora_multiplier")
    foreach ($lm in $lora_multiplier.Split(" ")) {
        [void]$ext_args.Add($lm)
    }
    if ($include_patterns) {
        [void]$ext_args.Add("--include_patterns=$include_patterns")
    }
    if ($exclude_patterns) {
        [void]$ext_args.Add("--exclude_patterns=$exclude_patterns")
    }
    if ($save_merged_model) {
        [void]$ext_args.Add("--save_merged_model=$save_merged_model")
    }
}

if ($from_file) {
    [void]$ext_args.Add("--from_file=$from_file")
}
elseif ($prompt) {
    [void]$ext_args.Add("--prompt=$prompt")
}

if ($image_size) {
    [void]$ext_args.Add("--image_size")
    foreach ($is in $image_size.Split(" ")) {
        [void]$ext_args.Add($is)
    }
}

if ($infer_steps -ne 50) {
    [void]$ext_args.Add("--infer_steps=$infer_steps")
}

if ($lycoris) {
    [void]$ext_args.Add("--lycoris")
}

Write-Output "Extended arguments:"
$ext_args | ForEach-Object { Write-Output "  $_" }

# run Generate
python -m accelerate.commands.launch "./musubi-tuner/$script" --dit=$dit `
    --vae=$vae `
    --save_path=$save_path `
    $ext_args

Write-Output "Generate finished"
Read-Host | Out-Null
