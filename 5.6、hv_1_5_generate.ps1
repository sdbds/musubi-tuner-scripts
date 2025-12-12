#Generate videos script by @bdsqlsz

#Generate Mode (HunyuanVideo_1_5)
$generate_mode = "HunyuanVideo_1_5"

# Parameters from hv_1_5_generate_video.py
$dit = "./ckpts/hunyuan-video-1.5/transformer/720p_t2v/diffusion_pytorch_model.safetensors"        # DiT directory | DiT路径 (T2V: 720p_t2v, I2V: 720p_i2v)
$vae = "./ckpts/hunyuan-video-1.5/vae/pytorch_model.pt"                          # VAE directory | VAE路径
$vae_dtype = ""                                                                   # data type for VAE, default is float16

# HunyuanVideo 1.5 specific parameters
$text_encoder = "./ckpts/text_encoder/qwen_2.5_vl_7b.safetensors"                # Text Encoder (Qwen2.5-VL) directory
$text_encoder_cpu = $false                                                        # Load text encoder on CPU to save GPU memory
$byt5 = "./ckpts/text_encoder/byt5_model.safetensors"                            # BYT5 text encoder directory
$image_encoder = "./ckpts/hunyuan-video-1.5/sigclip_vision_patch14_384.safetensors"      # Image Encoder for I2V

# VAE specific
$vae_sample_size = 128                                                            # VAE sample size (height/width). Default 128; set 256 if VRAM is sufficient
$vae_enable_patch_conv = $false                                                   # Enable patch-based convolution in VAE for memory optimization

# Attention and memory settings
$split_attn = $true                                                               # use split attention
$img_in_txt_in_offloading = $true                                                # offload img_in and txt_in to cpu

# I2V
$image_path = ""                                                                  # image path for I2V. If specified, I2V mode is used

# LoRA
$lora_weight = ""                                                                 # LoRA weight path
$lora_multiplier = "1.0"                                                          # LoRA multiplier
$save_merged_model = $false                                                       # save merged model. If specified, no inference will be performed

$prompt = "A beautiful landscape with mountains and a lake, cinematic quality, 4K"
$negative_prompt = ""                                                             # negative prompt
$from_file = ""                                                                   # Read prompts from a file
$video_size = "544 960"                                                           # video size (height width)
$video_length = 45                                                                # video length (frames)
$fps = 24                                                                         # video fps, Default is 24
$infer_steps = 50                                                                 # number of inference steps
$save_path = "./output_dir"                                                       # path to save generated video
$seed = 1026                                                                      # Seed for evaluation.
$guidance_scale = 6.0                                                             # guidance scale for CFG

# Flow Matching
$flow_shift = 7.0                                                                 # Shift factor for flow matching schedulers (default 7.0 for HV1.5)
$fp8 = $true                                                                      # use fp8 for DiT model
$fp8_scaled = $true                                                               # use fp8 scaled for DiT model
$cpu_noise = $false                                                               # use cpu noise (compatible with ComfyUI)

$device = ""                                                                      # device to use for inference
$attn_mode = "sageattn"                                                           # attention mode (torch, sdpa, xformers, sageattn, flash2, flash, flash3)
$blocks_to_swap = 0                                                               # number of blocks to swap in the model
$use_pinned_memory_for_block_swap = $true                                         # use pinned memory for block swapping
$output_type = "video"                                                            # output type (video, images, latent, both, latent_images)
$no_metadata = $false                                                             # do not save metadata
$latent_path = ""                                                                 # path to latent for decode. no inference
$lycoris = $false

$compile = $false                                                                 # Enable torch.compile
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
$script = "hv_1_5_generate_video.py"

# HunyuanVideo 1.5 specific arguments
[void]$ext_args.Add("--text_encoder=$text_encoder")
[void]$ext_args.Add("--byt5=$byt5")

if ($text_encoder_cpu) {
    [void]$ext_args.Add("--text_encoder_cpu")
}

if ($image_path) {
    [void]$ext_args.Add("--image_path=$image_path")
    [void]$ext_args.Add("--image_encoder=$image_encoder")
}

if ($vae_dtype) {
    [void]$ext_args.Add("--vae_dtype=$vae_dtype")
}

if ($vae_sample_size -ne 128) {
    [void]$ext_args.Add("--vae_sample_size=$vae_sample_size")
}

if ($vae_enable_patch_conv) {
    [void]$ext_args.Add("--vae_enable_patch_conv")
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

if ($output_type -ne "video") {
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

if ($flow_shift -ne 7.0) {
    [void]$ext_args.Add("--flow_shift=$flow_shift")
}

if ($fps -ne 24) {
    [void]$ext_args.Add("--fps=$fps")
}

if ($guidance_scale -ne 6.0) {
    [void]$ext_args.Add("--guidance_scale=$guidance_scale")
}

if ($negative_prompt) {
    [void]$ext_args.Add("--negative_prompt=$negative_prompt")
}

if ($cpu_noise) {
    [void]$ext_args.Add("--cpu_noise")
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

if ($video_size) {
    [void]$ext_args.Add("--video_size")
    foreach ($vs in $video_size.Split(" ")) {
        [void]$ext_args.Add($vs)
    }
}

if ($video_length) {
    [void]$ext_args.Add("--video_length=$video_length")
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
