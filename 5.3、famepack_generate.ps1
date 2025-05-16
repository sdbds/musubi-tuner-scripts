#Generate videos script by @bdsqlsz

#Generate Mode (HunyuanVideo/Wan/FramePack)
$generate_mode = "FramePack"

# Parameters from hv_generate_video.py
# $dit = "./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" # DiT checkpoint path or directory
# $vae = "./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt" # VAE checkpoint path or directory
# $dit = "./ckpts/wan/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors"   # DiT directory | DiT路径
# $vae = "./ckpts/vae/Wan2.1_VAE.pth"                                                 # VAE directory | VAE路径
$dit = "./ckpts/framepack/FramePackI2V_HY_bf16.safetensors"   # DiT directory | DiT路径
$vae = "./ckpts/framepack/hunyuan_video_vae_fp32.safetensors" # VAE directory | VAE路径
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
$embedded_cfg_scale = 10.0 # Embeded classifier free guidance scale.
$img_in_txt_in_offloading = $true # offload img_in and txt_in to cpu

# WAN specific parameters
$task = "t2v-14B" # one of t2v-1.3B, t2v-14B, i2v-14B, t2v-1.3B-FC, t2v-14B-FC, i2v-14B-FC
$t5 = "./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth" # T5 model path
$fp8_t5 = $false # use fp8 for T5 model
$negative_prompt = "" # negative prompt, if omitted, the default negative prompt is used
$guidance_scale = 5.0 # guidance scale for classifier free guidance, wan is 3.0 for 480P,5.0 for 720P  (default 5.0)
$vae_cache_cpu = $true # enable VAE cache in main memory
$trim_tail_frames = 0 # number of frames to trim from the tail of the video
$exclude_patterns = "" # Specify the values as a list. For example, "exclude_patterns=[blocks_[23]\d_]".
$include_patterns = "" # Specify the values as a list. For example, "include_patterns=[blocks_\d{2}_]".
$cpu_noise = $true # use cpu noise, get same result with comfyui
$cfg_skip_mode = "late" # cfg skip mode, ["early", "late", "middle", "early_late", "alternate", "none"]
$cfg_apply_ratio = 0.3 # The ratio of steps to apply CFG (0.0 to 1.0). Default is None (apply all steps).
$slg_layers = "" # Skip block (layer) indices for SLG (Skip Layer Guidance), comma separated
$slg_scale = 3.0 # scale for SLG classifier free guidance. Default is 3.0. Ignored if slg_mode is None or uncond
$slg_start = 0.0 # start ratio for inference steps for SLG. Default is 0.0.
$slg_end = 0.3 # end ratio for inference steps for SLG. Default is 0.3.
$slg_mode = "uncond" # SLG mode. original: same as SD3, uncond: replace uncond pred with SLG pred


# I2V
#$video_path = "" # video path
$image_path = "toml/1.jpg" # image path
$end_image_path = "" # end image path

# WAN FUN
$control_path = "" #  control video path

# FramePack specific parameters
$image_encoder = "./ckpts/framepack/sigclip_vision_patch14_384.safetensors" # Image encoder directory | 图像编码器路径
$latent_window_size = 9
$bulk_decode = $false
$video_seconds = 4
$f1=$false

# LoRA
$lora_weight = "./output_dir/framepack_qinglong.safetensors" # LoRA weight path
$lora_multiplier = "1.0" # LoRA multiplier
$save_merged_model = $false # save merged model.If specified, no inference will be performed

$prompt = """Ichika Anon from BanG Dream! MyGO!!!!!, an anime girl with long, straight pink hair cascading past her shoulders and bright blue eyes, stands bathed in stage lighting. She's holding a light seafoam green electric guitar, which looks like an ESP model (the logo 'ESP' is visible on the black strap and the headstock). She wears a stylish blue zip-up jacket with ruffled details over a dark top, paired with a short white pleated skirt and a simple black choker. A confident, slightly playful smile is on her face as she looks directly forward. The camera angle is slightly low, emphasizing her presence on what appears to be a dimly lit stage, with strong spotlights creating lens flares and highlighting her figure against the dark background. She lightly strums the guitar as the shot holds.
"""
$from_file = "" # Read prompts from a file
$video_size = "480 832" # video size
$video_length = 81 # video length
$fps = 30          # Framepack default is 30
$infer_steps = 28 # number of inference steps
$save_path = "./output_dir/output.mp4" # path to save generated video
$seed = 1026 # Seed for evaluation.
$sample_solver = "vanilla" # sample solver ["unipc", "dpm++", "vanilla"]

# Flow Matching
$flow_shift = 3.0 # Shift factor for flow matching schedulers (default 3.0 for I2V with 480p, 5.0 for others)
$fp8 = $true # use fp8 for DiT model
$fp8_scaled = $true # use fp8 scaled for DiT model
$device = "" # device to use for inference. If None, use CUDA if available, otherwise use CPU
$attn_mode = "sageattn" # attention mode (torch, sdpa, xformers, sageattn, flash2, flash, flash3)
$blocks_to_swap = 0 # number of blocks to swap in the model (max 39 for 14B, 29 for 1.3B)
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

if ($flow_shift -ne 3.0) {
    [void]$ext_args.Add("--flow_shift=$flow_shift")
}

if ($fps -ne 30) {
    [void]$ext_args.Add("--fps=$fps")
}

# WAN specific parameters
if ($generate_mode -ieq "Wan") {
    $script = "wan_generate_video.py"
    [void]$ext_args.Add("--task=$task")
    if ($task -ilike "t2v*") {
        $video_path = "" # video path
        $image_path = "" # image path
        $end_image_path = "" # end image path
    }
    elseif ($task -ilike "i2v*") {
        [void]$ext_args.Add("--image_path=$image_path")
        if ($end_image_path) {
            [void]$ext_args.Add("--end_image_path=$end_image_path")
            $trim_tail_frames = 3
        }
    }
    if ($task -ilike "*fc") {
        [void]$ext_args.Add("--control_path=$control_path")
    }
    if ($t5) {
        [void]$ext_args.Add("--t5=$t5")
    }
    if ($fp8_t5) {
        [void]$ext_args.Add("--fp8_t5")
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

    if ($trim_tail_frames -ne 0) {
        [void]$ext_args.Add("--trim_tail_frames=$trim_tail_frames")
    }

    if ($exclude_patterns) {
        [void]$ext_args.Add("--exclude_patterns=$exclude_patterns")
    }

    if ($include_patterns) {
        [void]$ext_args.Add("--include_patterns=$include_patterns")
    }

    if ($cpu_noise) {
        [void]$ext_args.Add("--cpu_noise")
    }

    if ($cfg_skip_mode) {
        [void]$ext_args.Add("--cfg_skip_mode=$cfg_skip_mode")
    }

    if ($cfg_apply_ratio -ne 0.0) {
        [void]$ext_args.Add("--cfg_apply_ratio=$cfg_apply_ratio")
    }

    if ($slg_layers) {
        [void]$ext_args.Add("--slg_layers=$slg_layers")
    }
    elseif ($slg_mode) {
        [void]$ext_args.Add("--slg_mode=$slg_mode")
    }

    if ($slg_scale -ne 3.0 -and $slg_mode -ieq "original") {
        [void]$ext_args.Add("--slg_scale=$slg_scale")
    }

    if ($slg_start -ne 0.0) {
        [void]$ext_args.Add("--slg_start=$slg_start")
    }

    if ($slg_end -ne 0.3) {
        [void]$ext_args.Add("--slg_end=$slg_end")
    }
    if ($sample_solver -ieq "unipc"){
        [void]$ext_args.Add("--sample_solver=$sample_solver")
    }
}
else {
    if ($generate_mode -ieq "FramePack") {
        $script = "fpack_generate_video.py"
        [void]$ext_args.Add("--image_encoder=$image_encoder")
        if ($bulk_decode) {
            [void]$ext_args.Add("--bulk_decode")
        }
        if ($latent_window_size -ne 9) {
            [void]$ext_args.Add("--latent_window_size=$latent_window_size")
        }
        if ($f1) {
            [void]$ext_args.Add("--f1")
        }
        if ($sample_solver -ieq "unipc"){
            [void]$ext_args.Add("--sample_solver=$sample_solver")
        }
    }
    else {
        if ($attn_mode -eq "sageattn" -and $split_attn) {
            [void]$ext_args.Add("--split_attn")
        }
        if ($compile) {
            [void]$ext_args.Add("--compile")
        }
        if ($img_in_txt_in_offloading) {
            [void]$ext_args.Add("--img_in_txt_in_offloading")
        }
        if ($fp8_fast -and $fp8) {
            [void]$ext_args.Add("--fp8_fast")
        }
    }
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
    if ($embedded_cfg_scale -ne 10.0) {
        [void]$ext_args.Add("--embedded_cfg_scale=$embedded_cfg_scale")
    }
    if ($image_path) {
        [void]$ext_args.Add("--image_path=$image_path")
    }
    if ($end_image_path) {
        [void]$ext_args.Add("--end_image_path=$end_image_path")
        $trim_tail_frames = 3
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
    foreach ($video_size in $video_size.Split(" ")) {
        [void]$ext_args.Add($video_size)
    }
}

if ($video_length -ne 129 -and $generate_mode -ine "FramePack") {
    [void]$ext_args.Add("--video_length=$video_length")
}

if ($video_seconds -ne 5) {
    [void]$ext_args.Add("--video_seconds=$video_seconds")
}

if ($infer_steps -ne 25) {
    [void]$ext_args.Add("--infer_steps=$infer_steps")
}

if ($lycoris) {
    [void]$ext_args.Add("--lycoris")
}

Write-Output "Extended arguments:"
$ext_args | ForEach-Object { Write-Output "  $_" }

# run Cache
python "./musubi-tuner/$script" --dit=$dit `
    --vae=$vae `
    --save_path=$save_path `
    $ext_args

Write-Output "Cache finished"
Read-Host | Out-Null