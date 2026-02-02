# Train script by @bdsqlsz

#训练模式(Z-Image_db)
$train_mode = "zimage_db"

# model_path
$dataset_config = "./toml/qinglong-qwen-image-datasets-cosplay.toml"                         # path to dataset config .toml file | 数据集配置文件路径
$dit = "./ckpts/diffusion_models/Z-image-base.safetensors"                         # DiT directory | DiT路径
$vae = "./ckpts/vae/ae.safetensors"                                      # VAE directory | VAE路径

# Z-Image Model
$text_encoder = "./ckpts/text_encoder/qwen3_model.safetensors"                   # Text Encoder (Qwen3) directory | 文本编码器路径

# Z-Image specific
$fp8_llm = $false                                                                # use fp8 for Text Encoder model
$fp8_scaled = $True                                                              # use scaled fp8 for DiT
$use_32bit_attention = $True                                                    # use 32-bit precision for attention computations
$mem_eff_save = $False                                                           # Memory efficient checkpoint saving
$fused_backward_pass = $False                                                    # Use fused backward pass (Adafactor)

$resume = ""                                                                     # resume from state | 从某个状态文件夹中恢复训练

#COPY machine | 差异炼丹法
$base_weights = "" #指定合并到底模basemodel中的模型路径，多个用空格隔开。默认为空，不使用。
$base_weights_multiplier = "1.0" #指定合并模型的权重，多个用空格隔开，默认为1.0。

#train config | 训练配置
$max_train_steps = ""                                                            # max train steps | 最大训练步数
$max_train_epochs = 20                                                           # max train epochs | 最大训练轮数
$gradient_checkpointing = $true                                                  # 梯度检查，开启后可节约显存，但是速度变慢
$gradient_checkpointing_cpu_offload = $false                                     # 梯度检查cpu offload，开启后可节约显存，但是速度变慢
$gradient_accumulation_steps = 1                                                 # 梯度累加数量，变相放大batchsize的倍数
$guidance_scale = 0.0                                                            # Z-Image Turbo model doesn't use CFG
$seed = 1026 # reproducable seed | 设置跑测试用的种子

#timestep sampling
$timestep_sampling = "sigmoid" # 时间步采样方法
$discrete_flow_shift = 3.0 # Euler 离散调度器的离散流位移，默认3.0
$sigmoid_scale = 1.0 # sigmoid 采样的缩放因子

$weighting_scheme = ""# sigma_sqrt, logit_normal, mode, cosmap, uniform, none
$logit_mean = -6.0           # logit mean
$logit_std = 2.0            # logit std
$mode_scale = 1.29          # mode scale
$min_timestep = 0           #最小时序
$max_timestep = 1000        #最大时间步
$show_timesteps = ""        #是否显示timesteps

# Learning rate | 学习率
$lr = "1e-5"
$lr_scheduler = "warmup_stable_decay"
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 新增cosine_with_min_lr(适合训练lora)、warmup_stable_decay(适合训练db)、inverse_sqrt
$lr_warmup_steps = 0
$lr_decay_steps = 0.2
$lr_scheduler_num_cycles = 1
$lr_scheduler_power = 1
$lr_scheduler_timescale = 0
$lr_scheduler_min_lr_ratio = 0.1

#precision and accelerate/save memory
$attn_mode = "flash"                                                             # "flash", "xformers", "sdpa", "sageattn"
$split_attn = $True
$mixed_precision = "bf16"
$full_bf16 = $False

# Compile parameters
$compile = $True
$compile_backend = "inductor"
$compile_mode = "max-autotune-no-cudagraphs"
$compile_fullgraph = $False
$compile_dynamic = "auto"
$compile_cache_size_limit = 32
# TF32 parameters
$cuda_allow_tf32 = $True
$cuda_cudnn_benchmark = $True

$vae_dtype = ""                                                                  # Z-Image VAE always uses float32
$fp8_base = $True
$max_data_loader_n_workers = 8
$persistent_data_loader_workers = $True

$blocks_to_swap = 0
$use_pinned_memory_for_block_swap = $True 
$img_in_txt_in_offloading = $True

#optimizer
$optimizer_type = "BCOS"
# adamw8bit | adamw32bit | adamw16bit | adafactor | Lion | Lion8bit | 
# PagedLion8bit | AdamW | AdamW8bit | PagedAdamW8bit | AdEMAMix8bit | PagedAdEMAMix8bit
# DAdaptAdam | DAdaptLion | DAdaptAdan | DAdaptSGD | Sophia | Prodigy
# Adv series：AdamW_adv | Prodigy_adv | Adopt_adv | Simplified_AdEMAMix | Lion_adv | Lion_Prodigy_adv
$max_grad_norm = 1.0

$d_coef = "0.5"
$d0 = "1e-5"

# wandb log
$wandb_api_key = "9c3747c46705bd779c58799295e6bb6d3da5dc98"

# save and load settings | 保存和输出设置
$output_name = "zimage_db_qinglong"
$save_every_n_epochs = "2"
$save_every_n_steps = ""
$save_last_n_epochs = ""
$save_last_n_steps = ""

# save state | 保存训练状态
$save_state = $False
$save_state_on_train_end = $False
$save_last_n_epochs_state = ""
$save_last_n_steps_state = ""

#sample | 输出采样图片
$enable_sample = $true
$sample_at_first = 0
$sample_prompts = "./toml/qinglong_qwen_image_cosplay.txt"
$sample_every_n_epochs = 2
$sample_every_n_steps = 1000

#metadata
$training_comment = "" # training_comment | 训练介绍，DB训练不使用
$metadata_title = ""
$metadata_author = ""
$metadata_description = ""
$metadata_license = ""
$metadata_tags = ""

#huggingface settings
$async_upload = $False
$huggingface_repo_id = ""
$huggingface_repo_type = ""
$huggingface_path_in_repo = ""
$huggingface_token = ""
$huggingface_repo_visibility = ""
$save_state_to_huggingface = $False
$resume_from_huggingface = $False

#DDP | 多卡设置
$multi_gpu = $False
$ddp_timeout = 120
$ddp_gradient_as_bucket_view = 1
$ddp_static_graph = 1

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
#$env:CUDA_VISIBLE_DEVICES = "1"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:VSLANG = "1033"
$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()
$laungh_script = "zimage_train"

# Z-Image specific arguments
[void]$ext_args.Add("--text_encoder=$text_encoder")

if ($fp8_llm) {
    [void]$ext_args.Add("--fp8_llm")
}

if ($fp8_scaled) {
    [void]$ext_args.Add("--fp8_scaled")
}

if ($use_32bit_attention) {
    [void]$ext_args.Add("--use_32bit_attention")
}

if ($mem_eff_save) {
    [void]$ext_args.Add("--mem_eff_save")
}

if ($fused_backward_pass) {
    [void]$ext_args.Add("--fused_backward_pass")
}

if ($full_bf16) {
    [void]$ext_args.Add("--full_bf16")
}

if ($attn_mode -ieq "flash") {
    [void]$ext_args.Add("--flash_attn")
}
elseif ($attn_mode -ieq "flash3") {
    [void]$ext_args.Add("--flash3")
}
elseif ($attn_mode -ieq "xformers") {
    [void]$ext_args.Add("--xformers")
    $split_attn = $True
}
elseif ($attn_mode -ieq "sageattn") {
    [void]$ext_args.Add("--attn_mode=sageattn")
}
else {
    [void]$ext_args.Add("--sdpa")
}

if ($split_attn) {
    [void]$ext_args.Add("--split_attn")
}

if ($multi_gpu) {
    [void]$launch_args.Add("--multi_gpu")
    [void]$launch_args.Add("--rdzv_backend=c10d")
    if ($ddp_timeout -ne 0) {
        [void]$ext_args.Add("--ddp_timeout=$ddp_timeout")
    }
    if ($ddp_gradient_as_bucket_view -ne 0) {
        [void]$ext_args.Add("--ddp_gradient_as_bucket_view")
    }
    if ($ddp_static_graph -ne 0) {
        [void]$ext_args.Add("--ddp_static_graph")
    }
}

if ($timestep_sampling -ine "sigma") {
    [void]$ext_args.Add("--timestep_sampling=$timestep_sampling")
    if ($timestep_sampling -ieq "sigmoid" -or $timestep_sampling -ieq "shift") {
        if ($discrete_flow_shift -ne 1.0 -and $timestep_sampling -ieq "shift") {
            [void]$ext_args.Add("--discrete_flow_shift=$discrete_flow_shift")
        }
        if ($sigmoid_scale -ne 1.0) {
            [void]$ext_args.Add("--sigmoid_scale=$sigmoid_scale")
        }
    }
}

if ($guidance_scale -ne 0.0) {
    [void]$ext_args.Add("--guidance_scale=$guidance_scale")
}

if ($weighting_scheme) {
    [void]$ext_args.Add("--weighting_scheme=$weighting_scheme")
    if ($weighting_scheme -ieq "logit_normal") {
        if ($logit_mean -ne 0.0) {
            [void]$ext_args.Add("--logit_mean=$logit_mean")
        }
        if ($logit_std -ne 1.0) {
            [void]$ext_args.Add("--logit_std=$logit_std")
        }
    }
    elseif ($weighting_scheme -ieq "mode") {
        if ($mode_scale -ne 1.29) {
            [void]$ext_args.Add("--mode_scale=$mode_scale")
        }
    }
}

if ($min_timestep -ne 0) {
    [void]$ext_args.Add("--min_timestep=$min_timestep")
}

if ($max_timestep -ne 1000) {
    [void]$ext_args.Add("--max_timestep=$max_timestep")
}

if ($show_timesteps) {
    [void]$ext_args.Add("--show_timesteps=$show_timesteps")
}

if ($max_train_steps) {
    [void]$ext_args.Add("--max_train_steps=$max_train_steps")
}
if ($max_train_epochs) {
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

if ($base_weights) {
    [void]$ext_args.Add("--base_weights")
    foreach ($base_weight in $base_weights.Split(" ")) {
        [void]$ext_args.Add($base_weight)
    }
    [void]$ext_args.Add("--base_weights_multiplier")
    foreach ($ratio in $base_weights_multiplier.Split(" ")) {
        [void]$ext_args.Add([float]$ratio)
    }
}

if ($lr_scheduler) {
    [void]$ext_args.Add("--lr_scheduler=$lr_scheduler")
}

if ($lr_scheduler_num_cycles) {
    [void]$ext_args.Add("--lr_scheduler_num_cycles=$lr_scheduler_num_cycles")
}

if ($lr_warmup_steps) {
    [void]$ext_args.Add("--lr_warmup_steps=$lr_warmup_steps")
}

if ($lr_decay_steps) {
    [void]$ext_args.Add("--lr_decay_steps=$lr_decay_steps")
}

if ($lr_scheduler_power -ne 1) {
    [void]$ext_args.Add("--lr_scheduler_power=$lr_scheduler_power")
}

if ($lr_scheduler_timescale) {
    [void]$ext_args.Add("--lr_scheduler_timescale=$lr_scheduler_timescale")
}

if ($lr_scheduler_min_lr_ratio) {
    [void]$ext_args.Add("--lr_scheduler_min_lr_ratio=$lr_scheduler_min_lr_ratio")
}

if ($full_bf16) {
    [void]$ext_args.Add("--full_bf16")
    $mixed_precision = "bf16"
}

if ($cuda_allow_tf32) {
    [void]$ext_args.Add("--cuda_allow_tf32")
}
if ($cuda_cudnn_benchmark) {
    [void]$ext_args.Add("--cuda_cudnn_benchmark")
}

if ($mixed_precision) {
    [void]$launch_args.Add("--mixed_precision=$mixed_precision")
    if ($mixed_precision -ieq "bf16" -or $mixed_precision -ieq "bfloat16") {
        [void]$launch_args.Add("--downcast_bf16")
    }
    [void]$ext_args.Add("--mixed_precision=$mixed_precision")
}

# Note: Z-Image VAE always uses float32, vae_dtype is ignored
if ($vae_dtype) {
    Write-Output "Note: vae_dtype is ignored for Z-Image (always uses float32)"
}

if ($fp8_base) {
    [void]$ext_args.Add("--fp8_base")
}

if ($max_data_loader_n_workers -ne 8) {
    [void]$ext_args.Add("--max_data_loader_n_workers=$max_data_loader_n_workers")
}

if ($persistent_data_loader_workers) {
    [void]$ext_args.Add("--persistent_data_loader_workers")
}

if ($blocks_to_swap -ne 0) {
    [void]$ext_args.Add("--blocks_to_swap=$blocks_to_swap")
    if ($use_pinned_memory_for_block_swap) {
        [void]$ext_args.Add("--use_pinned_memory_for_block_swap")
    }
}

# Add dynamo parameters
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
if ($img_in_txt_in_offloading) {
    [void]$ext_args.Add("--img_in_txt_in_offloading")
}

# Optimizer settings
if ($optimizer_type -ieq "Adam") {
    [void]$ext_args.Add("--optimizer_type=optimi.Adam")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("betas=.95,.98")
    if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
        [void]$ext_args.Add("decouple_lr=True")
    }
}

if ($optimizer_type -ieq "AdamW") {
    [void]$ext_args.Add("--optimizer_type=optimi.AdamW")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("betas=.95,.98")
    if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
        [void]$ext_args.Add("decouple_lr=True")
    }
}

if ($optimizer_type -ieq "adafactor") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.AdaFactor")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("scale_parameter=False")
    [void]$ext_args.Add("warmup_init=False")
    [void]$ext_args.Add("relative_step=False")
    [void]$ext_args.Add("cautious=True")
}

if ($optimizer_type -ieq "AdamW_adv") {
    [void]$ext_args.Add("--optimizer_type=adv_optm.AdamW_adv")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("grams_moment=True")
}

if ($optimizer_type -ieq "PagedAdamW8bit" -or $optimizer_type -ieq "AdamW" -or $optimizer_type -ieq "AdamW8bit") {
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}

if ($optimizer_type -ieq "Lion") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Lion")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("cautious=True")
}

if ($optimizer_type -ieq "Lion8bit" -or $optimizer_type -ieq "PagedLion8bit") {
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
    [void]$ext_args.Add("betas=.95,.98")
}

if ($optimizer_type -ieq "ademamix") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.AdEMAMix")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("alpha=10")
    [void]$ext_args.Add("cautious=True")
}

if ($optimizer_type -ieq "Sophia") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SophiaH")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "Prodigy") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Prodigy")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
    [void]$ext_args.Add("betas=.9,.99")
    [void]$ext_args.Add("decouple=True")
    [void]$ext_args.Add("use_bias_correction=True")
    [void]$ext_args.Add("d_coef=$d_coef")
    if ($lr_warmup_steps) {
        [void]$ext_args.Add("safeguard_warmup=True")
    }
    if ($d0) {
        [void]$ext_args.Add("d0=$d0")
    }
    $lr = "1"
    if ($unet_lr) {
        $unet_lr = $lr
    }
    if ($text_encoder_lr) {
        $text_encoder_lr = $lr
    }
}

if ($optimizer_type -ieq "Ranger") {
    [void]$ext_args.Add("--optimizer_type=optimi.Ranger")
    if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
        [void]$ext_args.Add("--optimizer_args")
        [void]$ext_args.Add("decouple_lr=True")
    }
}

if ($optimizer_type -ieq "Adan") {
    [void]$ext_args.Add("--optimizer_type=optimi.Adan")
    if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
        [void]$ext_args.Add("--optimizer_args")
        [void]$ext_args.Add("decouple_lr=True")
    }
}

if ($optimizer_type -ieq "StableAdamW") {
    [void]$ext_args.Add("--optimizer_type=optimi.StableAdamW")
    if (-not($train_unet_only -or $train_text_encoder_only) -or $train_text_encoder) {
        [void]$ext_args.Add("--optimizer_args")
        [void]$ext_args.Add("decouple_lr=True")
    }
}

if ($optimizer_type -ieq "Tiger") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Tiger")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ilike "*ScheduleFree") {
    $lr_scheduler = ""
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.08")
    [void]$ext_args.Add("weight_lr_power=0.001")
}

if ($optimizer_type -ieq "adammini") {
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "adamg") {
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.1")
    [void]$ext_args.Add("weight_decouple=True")
}

if ($optimizer_type -ieq "came") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.CAME")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "SOAP") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SOAP")
}

if ($optimizer_type -ieq "sgdsai") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.SGDSaI")
}

if ($optimizer_type -ieq "adopt") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.ADOPT")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("cautious=True")
}

if ($optimizer_type -ieq "Fira") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Fira")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
    [void]$ext_args.Add("rank=" + $network_dim)
    [void]$ext_args.Add("update_proj_gap=50")
    [void]$ext_args.Add("scale=1")
    [void]$ext_args.Add("projection_type='std'")
}

if ($optimizer_type -ieq "EmoNavi") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.EmoNavi")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "EmoFact") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.EmoFact")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "EmoLynx") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.EmoLynx")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "EmoNeco") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.EmoNeco")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
}

if ($optimizer_type -ieq "EmoZeal") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.EmoZeal")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
    [void]$ext_args.Add("shadow_weight=0.1")
}

if ($optimizer_type -ieq "SimplifiedAdEMAMix") {
    [void]$ext_args.Add("--optimizer_type=adv_optm.SimplifiedAdEMAMix")
    [void]$ext_args.Add("--optimizer_args")
    # [void]$ext_args.Add("nnmf_factor=True")
    if ($compile) {
        [void]$ext_args.Add("compiled_optimizer=True")
    }
}

if ($optimizer_type -ieq "AdaMuon") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.AdaMuon")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("weight_decay=0.01")
    [void]$ext_args.Add("adamw_lr=2e-4")
    [void]$ext_args.Add("adamw_betas=.9,.95")
}


if ($optimizer_type -ieq "AdamW_adv") {
    [void]$ext_args.Add("--optimizer_type=adv_optm.AdamW_adv")
    [void]$ext_args.Add("--optimizer_args")
    # [void]$ext_args.Add("use_atan2=True")
    [void]$ext_args.Add("grams_moment=True")
    # [void]$ext_args.Add("nnmf_factor=True")
    if ($compile) {
        [void]$ext_args.Add("compiled_optimizer=True")
    }
}

if ($optimizer_type -ieq "Adopt_adv") {
    [void]$ext_args.Add("--optimizer_type=adv_optm.Adopt_adv")
    [void]$ext_args.Add("--optimizer_args")
    # [void]$ext_args.Add("use_atan2=True")
    [void]$ext_args.Add("grams_moment=True")
    if ($compile) {
        [void]$ext_args.Add("compiled_optimizer=True")
    }
}

if ($optimizer_type -ieq "Prodigy_adv") {
    [void]$ext_args.Add("--optimizer_type=adv_optm.Prodigy_adv")
    [void]$ext_args.Add("--optimizer_args")
    # [void]$ext_args.Add("use_atan2=True")
    [void]$ext_args.Add("grams_moment=True")
    [void]$ext_args.Add("d_coef=$d_coef")
    if ($compile) {
        [void]$ext_args.Add("compiled_optimizer=True")
    }
    if ($lr_warmup_steps) {
        [void]$ext_args.Add("growth_rate=1.02")
    }
    if ($d0) {
        [void]$ext_args.Add("d0=$d0")
    }
    $lr = "1"
    if ($unet_lr) {
        $unet_lr = $lr
    }
    if ($text_encoder_lr) {
        $text_encoder_lr = $lr
    }
}

if ($optimizer_type -ieq "Lion_adv") {
    [void]$ext_args.Add("--optimizer_type=adv_optm.Lion_adv")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("cautious_mask=True")
    if ($compile) {
        [void]$ext_args.Add("compiled_optimizer=True")
    }
}

if ($optimizer_type -ieq "Lion_Prodigy_adv") {
    [void]$ext_args.Add("--optimizer_type=adv_optm.Lion_Prodigy_adv")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("grams_moment=True")
    [void]$ext_args.Add("d_coef=$d_coef")
    if ($compile) {
        [void]$ext_args.Add("compiled_optimizer=True")
    }
    if ($lr_warmup_steps) {
        [void]$ext_args.Add("growth_rate=1.02")
    }
    if ($d0) {
        [void]$ext_args.Add("d0=$d0")
    }
    $lr = "1"
    if ($unet_lr) {
        $unet_lr = $lr
    }
    if ($text_encoder_lr) {
        $text_encoder_lr = $lr
    }
}

if ($optimizer_type -ieq "BCOS") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.BCOS")
    [void]$ext_args.Add("--optimizer_args")
    [void]$ext_args.Add("simple_cond=True")
}

if ($optimizer_type -ieq "Ano") {
    [void]$ext_args.Add("--optimizer_type=pytorch_optimizer.Ano")
}

if ($optimizer_type -ilike "pytorch_optimizer.*") {
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}

if ($optimizer_type -ilike "adv_optm.*") {
    [void]$ext_args.Add("--optimizer_type=$optimizer_type")
}

if ($max_grad_norm -ne 1.0) {
    [void]$ext_args.Add("--max_grad_norm=$max_grad_norm")
}

if ($save_every_n_steps) {
    [void]$ext_args.Add("--save_every_n_steps=$save_every_n_steps")
}
else {
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
    [void]$ext_args.Add("--log_tracker_name=" + $output_name)
}

if ($enable_sample) {
    if ($sample_at_first) {
        [void]$ext_args.Add("--sample_at_first")
    }
    if ($sample_every_n_steps -ne 0) {
        [void]$ext_args.Add("--sample_every_n_steps=$sample_every_n_steps")
    }
    else {
        [void]$ext_args.Add("--sample_every_n_epochs=$sample_every_n_epochs")
    }
    [void]$ext_args.Add("--sample_prompts=$sample_prompts")
}

if ($training_comment) {
    [void]$ext_args.Add("--training_comment=$training_comment")
}

if ($metadata_title) {
    [void]$ext_args.Add("--metadata_title=$metadata_title")
}

if ($metadata_description) {
    [void]$ext_args.Add("--metadata_description=$metadata_description")
}

if ($metadata_author) {
    [void]$ext_args.Add("--metadata_author=$metadata_author")
}

if ($metadata_license) {
    [void]$ext_args.Add("--metadata_license=$metadata_license")
}

if ($metadata_tags) {
    [void]$ext_args.Add("--metadata_tags=$metadata_tags")
}

if ($async_upload) {
    [void]$ext_args.Add("--async_upload")
    if ($huggingface_token) {
        [void]$ext_args.Add("--huggingface_token=$huggingface_token")
    }
    if ($huggingface_repo_id) {
        [void]$ext_args.Add("--huggingface_repo_id=$huggingface_repo_id")
    }
    if ($huggingface_repo_type) {
        [void]$ext_args.Add("--huggingface_repo_type=$huggingface_repo_type")
    }
    if ($huggingface_path_in_repo) {
        [void]$ext_args.Add("--huggingface_path_in_repo=$huggingface_path_in_repo")
    }
    if ($huggingface_repo_visibility) {
        [void]$ext_args.Add("--huggingface_repo_visibility=$huggingface_repo_visibility")
    }
    if ($save_state_to_huggingface) {
        [void]$ext_args.Add("--save_state_to_huggingface=$save_state_to_huggingface")
    }
    if ($resume_from_huggingface) {
        [void]$ext_args.Add("--resume_from_huggingface=$resume_from_huggingface")
    }
}

Write-Output "Extended arguments:"
$ext_args | ForEach-Object { Write-Output "  $_" }

# run Training
python -m accelerate.commands.launch --num_cpu_threads_per_process=8 $launch_args "./musubi-tuner/$laungh_script.py" `
    --dataset_config="$dataset_config" `
    --dit=$dit `
    --vae=$vae `
    --seed=$seed  `
    --learning_rate=$lr `
    --output_name=$output_name `
    --output_dir="./output_dir" `
    --logging_dir="./logs" `
    $ext_args

Write-Output "Training finished"
Read-Host | Out-Null ;
