# LoRA training script for Mage-Flow

$train_mode = "mage_flow_lora"
$is_edit = $False
$model_variant = "standard"

# Model and dataset
$dataset_config = "./toml/qinglong-qwen-image-datasets.toml"
$dit = "./ckpts/diffusion_models/mage_flow_bf16.safetensors"
$vae = "./ckpts/vae/mage_flow_vae_bf16.safetensors"
$text_encoder = "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors"
$resume = ""
$network_weights = ""

# Training
$max_train_steps = ""
$max_train_epochs = 20
$gradient_checkpointing = $True
$gradient_checkpointing_cpu_offload = $False
$gradient_accumulation_steps = 1
$seed = 42
$timestep_sampling = "shift"
$discrete_flow_shift = 6.0
$weighting_scheme = "none"
$min_timestep = 0
$max_timestep = 1000

# Learning rate
$lr = "1e-4"
$lr_scheduler = "cosine_with_min_lr"
$lr_warmup_steps = 0
$lr_decay_steps = 0
$lr_scheduler_num_cycles = 1
$lr_scheduler_power = 1
$lr_scheduler_min_lr_ratio = 0.1

# LoRA
$network_dim = 32
$network_alpha = 16
$network_dropout = 0
$dim_from_weights = $False
$scale_weight_norms = 0
$enable_lora_plus = $False
$loraplus_lr_ratio = 4
$allow_mage_architecture_mismatch = $False

# Precision and memory
$attn_mode = "sdpa"
$mixed_precision = "bf16"
$full_bf16 = $False
$vae_dtype = "bfloat16"
$fp8_base = $False
$fp8_scaled = $False
$blocks_to_swap = 0
$use_pinned_memory_for_block_swap = $True
$max_data_loader_n_workers = 8
$persistent_data_loader_workers = $True

# torch.compile
$compile = $False
$compile_backend = "inductor"
$compile_mode = "max-autotune-no-cudagraphs"
$compile_fullgraph = $False
$compile_dynamic = "auto"
$compile_cache_size_limit = 32
$cuda_allow_tf32 = $True
$cuda_cudnn_benchmark = $True

# Optimizer
$optimizer_type = "AdamW8bit"
$optimizer_args = ""
$max_grad_norm = 1.0

# Output and logging
$output_name = "mage_flow_lora"
$output_dir = "./output_dir"
$logging_dir = "./logs"
$save_every_n_epochs = 2
$save_every_n_steps = ""
$save_last_n_epochs = ""
$save_last_n_steps = ""
$save_state = $False
$save_state_on_train_end = $False
$save_last_n_epochs_state = ""
$save_last_n_steps_state = ""
$wandb_api_key = ""

# Sampling
$enable_sample = $False
$sample_at_first = $False
$sample_prompts = "./toml/qinglong_qwen_image.txt"
$sample_every_n_epochs = 1
$sample_every_n_steps = 0

# Metadata and Hugging Face
$training_comment = ""
$metadata_title = ""
$metadata_author = ""
$metadata_description = ""
$metadata_license = ""
$metadata_tags = ""
$async_upload = $False
$huggingface_repo_id = ""
$huggingface_repo_type = ""
$huggingface_path_in_repo = ""
$huggingface_token = ""
$huggingface_repo_visibility = ""
$save_state_to_huggingface = $False
$resume_from_huggingface = $False

# DDP
$multi_gpu = $False
$ddp_timeout = 120
$ddp_gradient_as_bucket_view = $True
$ddp_static_graph = $True

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")

if ($mixed_precision -ine "bf16") {
    throw "Mage-Flow training requires bf16 mixed precision."
}
if ($fp8_base -and -not $fp8_scaled) {
    throw "Mage-Flow fp8_base requires fp8_scaled."
}
if (($blocks_to_swap -lt 0) -or ($blocks_to_swap -gt 10)) {
    throw "Mage-Flow blocks_to_swap must be 0 through 10."
}
if ($compile_fullgraph) {
    throw "Mage-Flow does not support compile_fullgraph."
}
if ($attn_mode -notin @("sdpa", "flash", "flash2")) {
    throw "Mage-Flow supports SDPA or FlashAttention 2 only."
}
if ($dim_from_weights -and -not $network_weights) {
    throw "dim_from_weights requires network_weights."
}
if ($model_variant -notin @("standard", "turbo")) {
    throw "Mage-Flow model_variant must be standard or turbo."
}
if ($enable_sample -and -not $sample_prompts) {
    throw "Mage-Flow sampling requires a prompt file."
}
if ($enable_sample -and (-not $vae -or -not $text_encoder)) {
    throw "Mage-Flow sampling requires both VAE and text encoder paths."
}

if ($env:OS -ilike "*windows*") {
    if ($compile) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
        if (Test-Path $vswhere) {
            $vsPath = & $vswhere -latest -products * `
                -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
                -property installationPath
            if ($vsPath) {
                & (Join-Path $vsPath "Common7\Tools\Launch-VsDevShell.ps1") -Arch amd64
                Set-Location $PSScriptRoot
            }
        }
    }
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
if ($cuda_allow_tf32) {
    $Env:NVIDIA_TF32_OVERRIDE = "1"
}
else {
    Remove-Item Env:NVIDIA_TF32_OVERRIDE -ErrorAction SilentlyContinue
}

$network_module = "musubi_tuner.networks.lora_mage_flow"
$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()

if ($is_edit) {
    [void]$ext_args.Add("--is_edit")
}
if ($fp8_base) {
    [void]$ext_args.Add("--fp8_base")
}
if ($fp8_scaled) {
    [void]$ext_args.Add("--fp8_scaled")
}
if ($allow_mage_architecture_mismatch) {
    [void]$ext_args.Add("--allow_mage_architecture_mismatch")
}
if ($attn_mode -in @("flash", "flash2")) {
    [void]$ext_args.Add("--flash_attn")
}
else {
    [void]$ext_args.Add("--sdpa")
}

if ($multi_gpu) {
    [void]$launch_args.Add("--multi_gpu")
    if ($ddp_timeout) {
        [void]$ext_args.Add("--ddp_timeout=$ddp_timeout")
    }
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
if ($scale_weight_norms) {
    [void]$ext_args.Add("--scale_weight_norms=$scale_weight_norms")
}
if ($enable_lora_plus) {
    [void]$ext_args.Add("--network_args")
    [void]$ext_args.Add("loraplus_lr_ratio=$loraplus_lr_ratio")
}

if ($lr_scheduler) {
    [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}
if ($lr_warmup_steps) {
    [void]$ext_args.Add("--lr_warmup_steps=$lr_warmup_steps")
}
if ($lr_decay_steps) {
    [void]$ext_args.Add("--lr_decay_steps=$lr_decay_steps")
}
if ($lr_scheduler_num_cycles -ne 1) {
    [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}
if ($lr_scheduler_power -ne 1) {
    [void]$ext_args.Add("--lr_scheduler_power=$lr_scheduler_power")
}
if ($lr_scheduler_min_lr_ratio) {
    [void]$ext_args.Add("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio")
}

if ($full_bf16) {
    [void]$ext_args.Add("--full_bf16")
}
if ($vae_dtype) {
    [void]$ext_args.Add("--vae_dtype=$vae_dtype")
}
if ($max_data_loader_n_workers -ne 8) {
    [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}
if ($persistent_data_loader_workers) {
    [void]$ext_args.Add("--persistent_data_loader_workers")
}
if ($blocks_to_swap) {
    [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
    if ($use_pinned_memory_for_block_swap) {
        [void]$ext_args.Add("--use_pinned_memory_for_block_swap")
    }
}
if ($compile) {
    [void]$ext_args.Add("--compile")
    [void]$ext_args.Add("--compile_backend=$compile_backend")
    [void]$ext_args.Add("--compile_mode=$compile_mode")
    [void]$ext_args.Add("--compile_dynamic=$compile_dynamic")
    [void]$ext_args.Add("--compile_cache_size_limit=$compile_cache_size_limit")
}
if ($cuda_cudnn_benchmark) {
    [void]$ext_args.Add("--cuda_cudnn_benchmark")
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
    if ($save_last_n_epochs_state) {
        [void]$ext_args.Add("--save_last_n_epochs_state=$save_last_n_epochs_state")
    }
    if ($save_last_n_steps_state) {
        [void]$ext_args.Add("--save_last_n_steps_state=$save_last_n_steps_state")
    }
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
}

foreach ($metadata in @(
    @("training_comment", $training_comment),
    @("metadata_title", $metadata_title),
    @("metadata_author", $metadata_author),
    @("metadata_description", $metadata_description),
    @("metadata_license", $metadata_license),
    @("metadata_tags", $metadata_tags)
)) {
    if ($metadata[1]) {
        [void]$ext_args.Add("--$($metadata[0])=$($metadata[1])")
    }
}

if ($async_upload) {
    [void]$ext_args.Add("--async_upload")
    foreach ($hubSetting in @(
        @("huggingface_repo_id", $huggingface_repo_id),
        @("huggingface_repo_type", $huggingface_repo_type),
        @("huggingface_path_in_repo", $huggingface_path_in_repo),
        @("huggingface_token", $huggingface_token),
        @("huggingface_repo_visibility", $huggingface_repo_visibility)
    )) {
        if ($hubSetting[1]) {
            [void]$ext_args.Add("--$($hubSetting[0])=$($hubSetting[1])")
        }
    }
    if ($save_state_to_huggingface) {
        [void]$ext_args.Add("--save_state_to_huggingface")
    }
    if ($resume_from_huggingface) {
        [void]$ext_args.Add("--resume_from_huggingface")
    }
}

Write-Output "Extended arguments:"
$ext_args | ForEach-Object { Write-Output "  $_" }

python -m accelerate.commands.launch $launch_args "./musubi-tuner/mage_flow_train_network.py" `
    --dataset_config=$dataset_config `
    --dit=$dit `
    --vae=$vae `
    --text_encoder=$text_encoder `
    --network_module=$network_module `
    --mixed_precision=$mixed_precision `
    --timestep_sampling=$timestep_sampling `
    --discrete_flow_shift=$discrete_flow_shift `
    --weighting_scheme=$weighting_scheme `
    --min_timestep=$min_timestep `
    --max_timestep=$max_timestep `
    --seed=$seed `
    --learning_rate=$lr `
    --output_name=$output_name `
    --output_dir=$output_dir `
    --logging_dir=$logging_dir $ext_args
Assert-NativeCommandSucceeded "Command failed: 3.10mage_flow_train_lora.ps1"

Write-Output "Mage-Flow training finished"
Read-Host | Out-Null
