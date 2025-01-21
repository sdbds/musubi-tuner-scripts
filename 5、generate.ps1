#Generate videos script by @bdsqlsz

#Parameters from hv_generate_video.py
$dit = "./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" # DiT checkpoint path or directory
$vae = "./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt" # VAE checkpoint path or directory
$vae_dtype = "" # data type for VAE, default is float16
$text_encoder1 = "./ckpts/text_encoder/llava_llama3_fp16.safetensors" # Text Encoder 1 directory
$text_encoder2 = "./ckpts/text_encoder_2/clip_l.safetensors" # Text Encoder 2 directory

# LoRA
$lora_weight = "./output_dir/hyvideo-qinglong.safetensors" # LoRA weight path
$lora_multiplier = "1.0" # LoRA multiplier

$prompt = """ a girl with long, flowing green hair adorned with a hair
ornament, a yellow flower, and a yellow rose. Her hair falls between her
eyes, and she has heterochromia, with one eye being blue and the other brown
or yellow. She is looking directly at the viewer with her mouth slightly
open, then laughting. Her attire consists of a green crop top
with puffy short sleeves, which are detached, revealing her collarbone and
bare shoulders. The top is complemented by a green skirt, and she wears a
green choker around her neck. Adding to her unique appearance, she has deer
ears and reindeer antlers, and a mini crown rests atop her head. A brooch and
a green bow further accentuate her outfit. The background is simple and
black, ensuring that the focus remains solely on the a girl.
"""
$video_size = "512 512" # video size
$video_length = 129 # video length
$infer_steps = 50 # number of inference steps
$save_path = "./output_dir" # path to save generated video
$seed = 1026 # Seed for evaluation.
$embedded_cfg_scale = 6.0 # Embeded classifier free guidance scale.

# Flow Matching
$flow_shift = 7.0 # Shift factor for flow matching schedulers.

$fp8 = $true # use fp8 for DiT model
$fp8_llm = $false # use fp8 for Text Encoder 1 (LLM)
$device = "" # device to use for inference. If None, use CUDA if available, otherwise use CPU
$attn_mode = "sageattn" # attention mode
$split_attn = $true # use split attention
$vae_chunk_size = 32 # chunk size for CausalConv3d in VAE
$vae_spatial_tile_sample_min_size = 128 # spatial tile sample min size for VAE, default 256
$blocks_to_swap = 0 # number of blocks to swap in the model
$img_in_txt_in_offloading = $true # offload img_in and txt_in to cpu
$output_type = "video" # output type
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

if ($vae_dtype) {
    [void]$ext_args.Add("--vae_dtype=$vae_dtype")
}

if ($fp8) {
    [void]$ext_args.Add("--fp8")
}

if ($fp8_llm) {
    [void]$ext_args.Add("--fp8_llm")
}

if ($device) {
    [void]$ext_args.Add("--device=$device")
}

if ($attn_mode) {
    [void]$ext_args.Add("--attn_mode=$attn_mode")
    if ($attn_mode -eq "sageattn" -and $split_attn) {
        [void]$ext_args.Add("--split_attn")
    }
}

if ($vae_chunk_size) {
    [void]$ext_args.Add("--vae_chunk_size=$vae_chunk_size")
}

if ($vae_spatial_tile_sample_min_size -ne 256) {
    [void]$ext_args.Add("--vae_spatial_tile_sample_min_size=$vae_spatial_tile_sample_min_size")
}

if ($blocks_to_swap -ne 0) {
    [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
}

if ($img_in_txt_in_offloading) {
    [void]$ext_args.Add("--img_in_txt_in_offloading")
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

if ($embedded_cfg_scale -ne 6.0) {
    [void]$ext_args.Add("--embedded_cfg_scale=$embedded_cfg_scale")
}

if ($flow_shift -ne 7.0) {
    [void]$ext_args.Add("--flow_shift=$flow_shift")
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
python "./musubi-tuner/hv_generate_video.py" --dit=$dit `
    --vae=$vae `
    --text_encoder1=$text_encoder1 `
    --text_encoder2=$text_encoder2 `
    --prompt=$prompt `
    --save_path=$save_path `
    $ext_args

Write-Output "Cache finished"
Read-Host | Out-Null