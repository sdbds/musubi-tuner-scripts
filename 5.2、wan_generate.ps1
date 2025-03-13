#Generate videos script by @bdsqlsz

#Generate Mode (HunyuanVideo/Wan)
$generate_mode = "Wan"

#Parameters from hv_generate_video.py
# $dit = "./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" # DiT checkpoint path or directory
# $vae = "./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt" # VAE checkpoint path or directory
$dit = "./ckpts/wan/split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors"   # DiT directory | DiT路径
$vae = "./ckpts/vae/Wan2.1_VAE.pth"                                                 # VAE directory | VAE路径
$vae_dtype = "" # data type for VAE, default is float16

# HunyuanVideo specific parameters
$text_encoder1 = "./ckpts/text_encoder/llava_llama3_fp16.safetensors" # Text Encoder 1 directory
$text_encoder2 = "./ckpts/text_encoder_2/clip_l.safetensors" # Text Encoder 2 directory
$vae_chunk_size = 32 # chunk size for CausalConv3d in VAE
$vae_spatial_tile_sample_min_size = 128 # spatial tile sample min size for VAE, default 256
$fp8_llm = $false # use fp8 for Text Encoder 1 (LLM)
$fp8_fast = $true # Enable fast FP8 arthimetic(RTX 4XXX+)
$compile = $true # Enable torch.compile
$split_attn = $true # use split attention
$embedded_cfg_scale = 7.0 # Embeded classifier free guidance scale.
$img_in_txt_in_offloading = $true # offload img_in and txt_in to cpu

# WAN specific parameters
$task = "t2v-14B" # one of t2v-1.3B, t2v-14B, i2v-14B
$t5 = "./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth" # T5 model path
$fp8_t5 = $false # use fp8 for T5 model
$fp8_scaled = $true # use fp8 scaled for T5 model
$negative_prompt = "" # negative prompt, if omitted, the default negative prompt is used
$guidance_scale = 3.0 # guidance scale for classifier free guidance, wan is 3.0 for 480P,5.0 for 720P  (default 5.0)
$vae_cache_cpu = $true # enable VAE cache in main memory

# LoRA
$lora_weight = "./output_dir/wan_qinglong.safetensors" # LoRA weight path
$lora_multiplier = "1.0" # LoRA multiplier

$prompt = """1girl, solo, long hair, looking at viewer, open mouth, blue eyes, simple background, hair ornament, animal ears, hair between eyes, bare shoulders, medium breasts, yellow eyes, short sleeves, :d, detached sleeves, green hair, black gloves, virtual youtuber, puffy sleeves, hand up, midriff, hair flower, fingerless gloves, crop top, puffy short sleeves, hand on own hip, v, fake animal ears, heterochromia, black background, gem, green skirt, brooch, green bow, yellow flower, green shirt, mini crown, tilted headwear, blue gemstone, reindeer antlers, yellow rose, puffy detached sleeves, deer ears, green choker, deer antlers, lace gloves, deer girl
"""
$video_size = "832 480" # video size
$video_length = 85 # video length
$fps = 16
$infer_steps = 20 # number of inference steps
$save_path = "./output_dir/output.mp4" # path to save generated video
$seed = 1026 # Seed for evaluation.

# Flow Matching
$flow_shift = 5.0 # Shift factor for flow matching schedulers (default 3.0 for I2V with 480p, 5.0 for others)
$fp8 = $true # use fp8 for DiT model
$device = "" # device to use for inference. If None, use CUDA if available, otherwise use CPU
$attn_mode = "sageattn" # attention mode (torch, sdpa, xformers, sageattn, flash2, flash, flash3)
$blocks_to_swap = 8 # number of blocks to swap in the model (max 39 for 14B, 29 for 1.3B)
$output_type = "both" # output type
$no_metadata = $false # do not save metadata
$latent_path = "" # path to latent for decode. no inference
$lycoris = $false

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
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
$ext_args = [System.Collections.ArrayList]::new()
$script = "hv_generate_video.py" 

if ($vae_dtype) {
    [void]$ext_args.Add("--vae_dtype=$vae_dtype")
}

if ($fp8) {
    [void]$ext_args.Add("--fp8")
}

if ($device) {
    [void]$ext_args.Add("--device=$device")
}

if ($attn_mode -ine "torch") {
    [void]$ext_args.Add("--attn_mode=$attn_mode")
    if ($attn_mode -eq "sageattn" -and $split_attn -and $generate_mode -ine "Wan") {
        [void]$ext_args.Add("--split_attn")
    }
}

if ($blocks_to_swap -ne 0) {
    [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
}

if ($output_type) {
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

if ($flow_shift -ne 3.0) {
    [void]$ext_args.Add("--flow_shift=$flow_shift")
}

if ($fps -ne 16) {
    [void]$ext_args.Add("--fps=$fps")
}

# WAN specific parameters
if ($generate_mode -ieq "Wan") {
    $script = "wan_generate_video.py"
    if ($task) {
        [void]$ext_args.Add("--task=$task")
    }
    
    if ($t5) {
        [void]$ext_args.Add("--t5=$t5")
    }

    if ($fp8_t5) {
        [void]$ext_args.Add("--fp8_t5")
    }

    if ($fp8_scaled -and $fp8) {
        [void]$ext_args.Add("--fp8_scaled")
    }

    if ($negative_prompt) {
        [void]$ext_args.Add("--negative_prompt=$negative_prompt")
    }

    if ($guidance_scale -ne 5.0) {
        [void]$ext_args.Add("--guidance_scale=$guidance_scale")
    }

    if ($vae_cache_cpu) {
        [void]$ext_args.Add("--vae_cache_cpu")
    }
}
else {
    [void]$ext_args.Add("--text_encoder1=$text_encoder1")
    [void]$ext_args.Add("--text_encoder2=$text_encoder2")
    if ($vae_chunk_size) {
        [void]$ext_args.Add("--vae_chunk_size=$vae_chunk_size")
    }
    
    if ($vae_spatial_tile_sample_min_size -ne 256) {
        [void]$ext_args.Add("--vae_spatial_tile_sample_min_size=$vae_spatial_tile_sample_min_size")
    }
    if ($fp8_llm) {
        [void]$ext_args.Add("--fp8_llm")
    }
    if ($fp8_fast -and $fp8) {
        [void]$ext_args.Add("--fp8_fast")
    }
    if ($compile) {
        [void]$ext_args.Add("--compile")
    }
    if ($embedded_cfg_scale -ne 6.0) {
        [void]$ext_args.Add("--embedded_cfg_scale=$embedded_cfg_scale")
    }
    if ($img_in_txt_in_offloading) {
        [void]$ext_args.Add("--img_in_txt_in_offloading")
    }
}

if ($lora_weight) {
    [void]$ext_args.Add("--lora_weight")
    foreach ($lora_weight in $lora_weight.Split(" ")) {
        [void]$ext_args.Add($lora_weight)
    }
    [void]$ext_args.Add("--lora_multiplier")
    foreach ($lora_multiplier in $lora_multiplier.Split(" ")) {
        [void]$ext_args.Add($lora_multiplier)
    }
}

if ($prompt) {
    [void]$ext_args.Add("--prompt=$prompt")
}

if ($video_size) {
    [void]$ext_args.Add("--video_size")
    foreach ($video_size in $video_size.Split(" ")) {
        [void]$ext_args.Add($video_size)
    }
}

if ($video_length -ne 129) {
    [void]$ext_args.Add("--video_length=$video_length")
}

if ($infer_steps -ne 50) {
    [void]$ext_args.Add("--infer_steps=$infer_steps")
}

if ($lycoris) {
    [void]$ext_args.Add("--lycoris")
}

# run Cache
python "./musubi-tuner/$script" --dit=$dit `
    --vae=$vae `
    --prompt=$prompt `
    --save_path=$save_path `
    $ext_args

Write-Output "Cache finished"
Read-Host | Out-Null