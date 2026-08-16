# LoRA training script for MiniMax-H3

$task = "t2va"

# Model and dataset
$dataset_config = "./toml/qinglong-video-datasets.toml"
$fl2va_dit = "./ckpts/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
$ref2va_dit = "./ckpts/diffusion_models/minimax_h3_ref2va_bf16.safetensors"
$dit = if ($task -ieq "ref2va") { $ref2va_dit } else { $fl2va_dit }
$video_vae = "./ckpts/vae/minimax_h3_video_vae_fp16.safetensors"
$audio_vae = "./ckpts/vae/minimax_h3_audio_vae_fp32.safetensors"
$text_encoder = "./ckpts/text_encoder/qwen3vl_32b_minimax_h3_bf16.safetensors"
$resume = ""
$network_weights = ""

# MiniMax-H3 flow matching
$h3_shift_video = 12
$h3_shift_audio = 3
$h3_visual_cond_clean = 0.999
$h3_audio_cond_clean = 1.0
$timestep_sampling = "uniform"
$discrete_flow_shift = 1.0
$weighting_scheme = "none"

# One-frame image training and guidance loss
$one_frame = $False
$video_only = $False
$audio_loss_weight = 1.0
$h3_guidance_loss_scale = 0.0
$h3_guidance_loss_scale_audio = ""
$h3_guidance_loss_sigma_min = 0.0
$h3_guidance_loss_uncond_cache = ""

# Training
$max_train_steps = ""
$max_train_epochs = 16
$gradient_checkpointing = $True
$gradient_checkpointing_cpu_offload = $False
$gradient_accumulation_steps = 1
$seed = 42

# Learning rate and optimizer
$lr = "1e-4"
$lr_scheduler = "cosine_with_min_lr"
$lr_warmup_steps = 0
$lr_scheduler_num_cycles = 1
$lr_scheduler_min_lr_ratio = 0.1
$optimizer_type = "AdamW8bit"
$optimizer_args = ""
$max_grad_norm = 1.0

# LoRA
$network_module = "networks.lora_minimax_h3"
$network_dim = 16
$network_alpha = 16
$network_dropout = 0
$dim_from_weights = $False

# Precision and memory
$mixed_precision = "bf16"
$dit_dtype = "bfloat16"
$full_bf16 = $False
$attn_mode = "sdpa"
$blocks_to_swap = 48
$block_swap_h2d_only = $False
$use_pinned_memory_for_block_swap = $True
$max_data_loader_n_workers = 8
$persistent_data_loader_workers = $True

# Output and logging
$output_name = "minimax_h3_lora"
$output_dir = "./output_dir"
$logging_dir = "./logs"
$save_every_n_epochs = 1
$save_every_n_steps = ""
$save_last_n_epochs = ""
$save_last_n_steps = ""
$save_state = $False
$save_state_on_train_end = $False
$wandb_api_key = ""

# Sampling
$enable_sample = $True
$sample_at_first = $True
$sample_prompts = "./toml/qinglong_minimaxh3.txt"
$sample_every_n_epochs = 1
$sample_every_n_steps = 0

# Distributed training
$multi_gpu = $False
$ddp_timeout = 120
$ddp_gradient_as_bucket_view = $True
$ddp_static_graph = $True

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")

if ($task -notin @("t2va", "fl2va", "ref2va")) {
    throw "MiniMax-H3 task must be t2va, fl2va, or ref2va."
}
if ($one_frame -and $task -ine "t2va") {
    throw "MiniMax-H3 one-frame training requires task=t2va."
}
if ($mixed_precision -ine "bf16" -or $dit_dtype -notin @("bf16", "bfloat16")) {
    throw "MiniMax-H3 training requires a BF16 transformer and bf16 mixed precision."
}
if ($timestep_sampling -ine "uniform" -or $weighting_scheme -ine "none") {
    throw "MiniMax-H3 training requires uniform timestep sampling without generic loss weighting."
}
if ($discrete_flow_shift -ne 1.0) {
    throw "MiniMax-H3 generic discrete_flow_shift must remain 1.0; use the H3 modality shifts."
}
if (($h3_shift_video -le 0) -or ($h3_shift_audio -le 0)) {
    throw "MiniMax-H3 video and audio shifts must be positive."
}
if ($audio_loss_weight -lt 0) {
    throw "MiniMax-H3 audio_loss_weight must be nonnegative."
}
if ($h3_guidance_loss_scale -lt 0) {
    throw "MiniMax-H3 h3_guidance_loss_scale must be nonnegative."
}
if (($h3_guidance_loss_scale_audio -ne "") -and ($h3_guidance_loss_scale_audio -lt 0)) {
    throw "MiniMax-H3 h3_guidance_loss_scale_audio must be nonnegative."
}
if (($h3_guidance_loss_sigma_min -lt 0) -or ($h3_guidance_loss_sigma_min -gt 1)) {
    throw "MiniMax-H3 h3_guidance_loss_sigma_min must be between 0.0 and 1.0."
}
if (($h3_guidance_loss_scale -gt 0) -and [string]::IsNullOrWhiteSpace($h3_guidance_loss_uncond_cache)) {
    throw "MiniMax-H3 positive guidance loss requires an unconditional-cache path."
}
if (($blocks_to_swap -lt 0) -or ($blocks_to_swap -gt 48)) {
    throw "MiniMax-H3 blocks_to_swap must be 0 through 48."
}
if ($block_swap_h2d_only -and $blocks_to_swap -and -not $gradient_checkpointing) {
    throw "MiniMax-H3 block_swap_h2d_only requires gradient checkpointing."
}
if ($dim_from_weights -and -not $network_weights) {
    throw "dim_from_weights requires network_weights."
}
if ($enable_sample -and -not $sample_prompts) {
    throw "MiniMax-H3 sampling requires a prompt file."
}
if ($enable_sample -and (-not $video_vae -or -not $audio_vae -or -not $text_encoder)) {
    throw "MiniMax-H3 sampling requires video VAE, audio VAE, and text encoder paths."
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
$launch_args = [System.Collections.ArrayList]::new()

if ($one_frame) {
    [void]$ext_args.Add("--one_frame")
}
if ($video_only) {
    [void]$ext_args.Add("--video_only")
}
[void]$ext_args.Add("--audio_loss_weight=$audio_loss_weight")
[void]$ext_args.Add("--h3_guidance_loss_scale=$h3_guidance_loss_scale")
if ($h3_guidance_loss_scale_audio -ne "") {
    [void]$ext_args.Add("--h3_guidance_loss_scale_audio=$h3_guidance_loss_scale_audio")
}
[void]$ext_args.Add("--h3_guidance_loss_sigma_min=$h3_guidance_loss_sigma_min")
if ($h3_guidance_loss_uncond_cache -ne "") {
    [void]$ext_args.Add("--h3_guidance_loss_uncond_cache=$h3_guidance_loss_uncond_cache")
}

if ($attn_mode -ieq "sdpa") {
    [void]$ext_args.Add("--sdpa")
}
elseif ($attn_mode -in @("flash", "flash2")) {
    [void]$ext_args.Add("--flash_attn")
}
else {
    throw "MiniMax-H3 training supports sdpa, flash, or flash2 attention."
}

if ($multi_gpu) {
    [void]$launch_args.Add("--multi_gpu")
    [void]$ext_args.Add("--ddp_timeout=$ddp_timeout")
    if ($ddp_gradient_as_bucket_view) {
        [void]$ext_args.Add("--ddp_gradient_as_bucket_view")
    }
    if ($ddp_static_graph) {
        [void]$ext_args.Add("--ddp_static_graph")
    }
}
[void]$launch_args.Add("--mixed_precision=$mixed_precision")
[void]$launch_args.Add("--downcast_bf16")

if ($max_train_steps) {
    [void]$ext_args.Add("--max_train_steps=$max_train_steps")
}
elseif ($max_train_epochs) {
    [void]$ext_args.Add("--max_train_epochs=$max_train_epochs")
}
if ($gradient_checkpointing) {
    [void]$ext_args.Add("--gradient_checkpointing")
    if ($gradient_checkpointing_cpu_offload) {
        [void]$ext_args.Add("--gradient_checkpointing_cpu_offload")
    }
}
if ($gradient_accumulation_steps -ne 1) {
    [void]$ext_args.Add("--gradient_accumulation_steps=$gradient_accumulation_steps")
}

if ($network_weights) {
    [void]$ext_args.Add("--network_weights=$network_weights")
    if ($dim_from_weights) {
        [void]$ext_args.Add("--dim_from_weights")
    }
}
if ($network_dim) {
    [void]$ext_args.Add("--network_dim=$network_dim")
}
if ($network_alpha) {
    [void]$ext_args.Add("--network_alpha=$network_alpha")
}
if ($network_dropout) {
    [void]$ext_args.Add("--network_dropout=$network_dropout")
}

if ($lr_scheduler) {
    [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}
if ($lr_warmup_steps) {
    [void]$ext_args.Add("--lr_warmup_steps=$lr_warmup_steps")
}
if ($lr_scheduler_num_cycles -ne 1) {
    [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}
if ($lr_scheduler_min_lr_ratio) {
    [void]$ext_args.Add("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio")
}

if ($full_bf16) {
    [void]$ext_args.Add("--full_bf16")
}
if ($max_data_loader_n_workers -ne 8) {
    [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}
if ($persistent_data_loader_workers) {
    [void]$ext_args.Add("--persistent_data_loader_workers")
}
if ($blocks_to_swap) {
    [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
    if ($block_swap_h2d_only) {
        [void]$ext_args.Add("--block_swap_h2d_only")
    }
    if ($use_pinned_memory_for_block_swap) {
        [void]$ext_args.Add("--use_pinned_memory_for_block_swap")
    }
}

[void]$ext_args.Add("--optimizer_type=$optimizer_type")
if ($optimizer_args) {
    [void]$ext_args.Add("--optimizer_args")
    foreach ($optimizerArg in ($optimizer_args -split "[`r`n;]+")) {
        if ($optimizerArg.Trim()) {
            [void]$ext_args.Add($optimizerArg.Trim())
        }
    }
}
if ($max_grad_norm -ne 1.0) {
    [void]$ext_args.Add("--max_grad_norm=$max_grad_norm")
}

if ($save_every_n_steps) {
    [void]$ext_args.Add("--save_every_n_steps=$save_every_n_steps")
}
elseif ($save_every_n_epochs) {
    [void]$ext_args.Add("--save_every_n_epochs=$save_every_n_epochs")
}
if ($save_last_n_epochs) {
    [void]$ext_args.Add("--save_last_n_epochs=$save_last_n_epochs")
}
if ($save_last_n_steps) {
    [void]$ext_args.Add("--save_last_n_steps=$save_last_n_steps")
}
if ($save_state_on_train_end) {
    [void]$ext_args.Add("--save_state_on_train_end")
}
elseif ($save_state) {
    [void]$ext_args.Add("--save_state")
}
if ($resume) {
    [void]$ext_args.Add("--resume=$resume")
}
if ($wandb_api_key) {
    [void]$ext_args.Add("--wandb_api_key=$wandb_api_key")
    [void]$ext_args.Add("--log_with=wandb")
    [void]$ext_args.Add("--log_tracker_name=$output_name")
}

if ($enable_sample) {
    if ($sample_at_first) {
        [void]$ext_args.Add("--sample_at_first")
    }
    if ($sample_every_n_steps) {
        [void]$ext_args.Add("--sample_every_n_steps=$sample_every_n_steps")
    }
    else {
        [void]$ext_args.Add("--sample_every_n_epochs=$sample_every_n_epochs")
    }
    [void]$ext_args.Add("--sample_prompts=$sample_prompts")
    [void]$ext_args.Add("--video_vae=$video_vae")
    [void]$ext_args.Add("--audio_vae=$audio_vae")
    [void]$ext_args.Add("--text_encoder=$text_encoder")
}

# Metadata is supplied by the shared trainer when configured through the GUI.

Write-Output "Extended arguments:"
$ext_args | ForEach-Object { Write-Output "  $_" }

# H3 sampling uses dual VAEs; the legacy --vae=$vae training argument does not apply.
python -m accelerate.commands.launch $launch_args "./musubi-tuner/minimax_h3_train_network.py" `
    --dataset_config=$dataset_config `
    --task=$task `
    --dit=$dit `
    --dit_dtype=$dit_dtype `
    --network_module=$network_module `
    --mixed_precision=$mixed_precision `
    --timestep_sampling=$timestep_sampling `
    --discrete_flow_shift=$discrete_flow_shift `
    --weighting_scheme=$weighting_scheme `
    --h3_shift_video=$h3_shift_video `
    --h3_shift_audio=$h3_shift_audio `
    --h3_visual_cond_clean=$h3_visual_cond_clean `
    --h3_audio_cond_clean=$h3_audio_cond_clean `
    --seed=$seed `
    --learning_rate=$lr `
    --output_name=$output_name `
    --output_dir=$output_dir `
    --logging_dir=$logging_dir $ext_args
Assert-NativeCommandSucceeded "Command failed: 3.11minimax_h3_train_lora.ps1"

Write-Output "MiniMax-H3 training finished"
Read-Host | Out-Null
