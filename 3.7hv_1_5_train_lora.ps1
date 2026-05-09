# Train script by @bdsqlsz

#训练模式(HunyuanVideo_1_5_Lora)
$train_mode = "HunyuanVideo_1_5_Lora"

# model_path
$dataset_config = "./toml/qinglong-hv-1-5-datasets.toml"                        # path to dataset config .toml file | 数据集配置文件路径                                             # VAE directory | VAE路径
$dit = "./ckpts/hunyuan-video-1.5/transformer/720p_t2v/diffusion_pytorch_model.safetensors"       # DiT directory | DiT路径 (T2V: 720p_t2v, I2V: 720p_i2v)
$vae = "./ckpts/hunyuan-video-1.5/vae/pytorch_model.pt"                         # VAE directory | VAE路径

# HunyuanVideo 1.5 Model
$text_encoder = "./ckpts/text_encoder/qwen_2.5_vl_7b.safetensors"               # Text Encoder (Qwen2.5-VL) directory | 文本编码器路径
$byt5 = "./ckpts/text_encoder/byt5_model.safetensors"                           # BYT5 text encoder directory | BYT5编码器路径
$image_encoder = "./ckpts/hunyuan-video-1.5/sigclip_vision_patch14_384.safetensors"     # Image encoder directory (for I2V) | 图像编码器路径

# HunyuanVideo 1.5 specific
$task = "t2v"                                                                    # training task type: text-to-video (t2v) or image-to-video (i2v)
$dit_dtype = ""                                                                  # data type for DiT, default is bfloat16 (auto-detected from checkpoint)
$fp8_vl = $false                                                                 # use fp8 for Text Encoder model
$fp8_scaled = $True                                                              # use scaled fp8 for DiT
$vae_sample_size = 128                                                           # VAE sample size (height/width). Default 128; set 256 if VRAM is sufficient
$vae_enable_patch_conv = $False                                                  # Enable patch-based convolution in VAE for memory optimization

$resume = ""                                                                     # resume from state | 从某个状态文件夹中恢复训练
$network_weights = ""                                                            # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。

#COPY machine | 差异炼丹法
$base_weights = "" #指定合并到底模basemodel中的模型路径，多个用空格隔开。默认为空，不使用。
$base_weights_multiplier = "1.0" #指定合并模型的权重，多个用空格隔开，默认为1.0。

#train config | 训练配置
$max_train_steps = ""                                                            # max train steps | 最大训练步数
$max_train_epochs = 20                                                           # max train epochs | 最大训练轮数
$gradient_checkpointing = $true                                                  # 梯度检查，开启后可节约显存，但是速度变慢
$gradient_checkpointing_cpu_offload = $false                                     # 梯度检查cpu offload，开启后可节约显存，但是速度变慢
$gradient_accumulation_steps = 1                                                 # 梯度累加数量，变相放大batchsize的倍数
$guidance_scale = 6.0                                                            # HunyuanVideo 1.5 default guidance scale
$seed = 1026 # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

#timestep sampling
$timestep_sampling = "sigmoid" # 时间步采样方法，可选 sd3用"sigma"、普通DDPM用"uniform" 或 flux用"sigmoid" 或者 "flux_shift". shift需要修改discarete_flow_shift的参数
$discrete_flow_shift = 7.0 # Euler 离散调度器的离散流位移，HV1.5默认7.0
$sigmoid_scale = 1.0 # sigmoid 采样的缩放因子，默认为 1.0。较大的值会使采样更加均匀

$weighting_scheme = ""# sigma_sqrt, logit_normal, mode, cosmap, uniform, none
$logit_mean = 0.0           # logit mean | logit 均值 默认0.0 只在logit_normal下生效
$logit_std = 1.0            # logit std | logit 标准差 默认1.0 只在logit_normal下生效
$mode_scale = 1.29          # mode scale | mode 缩放 默认1.29 只在mode下生效
$min_timestep = 0           #最小时序，默认值0
$max_timestep = 1000        #最大时间步 默认1000
$show_timesteps = ""        #是否显示timesteps， console/image

# Learning rate | 学习率
$lr = "2e-4"
# $unet_lr = "5e-4"
# $text_encoder_lr = "2e-5"
$lr_scheduler = "cosine_with_min_lr"
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 新增cosine_with_min_lr(适合训练lora)、warmup_stable_decay(适合训练db)、inverse_sqrt
$lr_warmup_steps = 0 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
$lr_decay_steps = 0.2 # decay steps | 学习率衰减步数，仅在 lr_scheduler 为warmup_stable_decay时 需要填写，一般是10%总步数
$lr_scheduler_num_cycles = 1 # restarts nums | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时需要填写这个值
$lr_scheduler_power = 1     #Polynomial power for polynomial scheduler |余弦退火power
$lr_scheduler_timescale = 0 #times scale |时间缩放，仅在 lr_scheduler 为 inverse_sqrt 时需要填写这个值，默认同lr_warmup_steps
$lr_scheduler_min_lr_ratio = 0.1 #min lr ratio |最小学习率比率，仅在 lr_scheduler 为 cosine_with_min_lr、、warmup_stable_decay 时需要填写这个值，默认0

#network settings
$network_dim = 32 # network dim | 常用 4~128，不是越大越好
$network_alpha = 16 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。
$network_dropout = 0 # network dropout | 常用 0~0.3
$dim_from_weights = $False # use dim from weights | 从已有的 LoRA 模型上继续训练时，自动获取 dim
$scale_weight_norms = 0 # scale weight norms (1 is a good starting point)| scale weight norms (1 is a good starting point)

#precision and accelerate/save memory
$attn_mode = "flash"                                                             # "flash", "xformers", "sdpa"
$split_attn = $True                                                              # split attention | split attention
$mixed_precision = "bf16"                                                        # fp16 |bf16 default: bf16
$full_bf16 = $False                                                              # Enable full BF16 training

# Compile parameters
$compile = $True
$compile_backend = "inductor"
$compile_mode = "max-autotune-no-cudagraphs"                                     # "default", "reduce-overhead", "max-autotune-no-cudagraphs"
$compile_fullgraph = $False                                                      # use fullgraph mode for dynamo
$compile_dynamic = "auto"                                                        # use dynamic mode for dynamo
$compile_cache_size_limit = 32
# TF32 parameters
$cuda_allow_tf32 = $True
$cuda_cudnn_benchmark = $True

$vae_dtype = ""                                                                  # fp16 | fp32 |bf16 default: fp16
$fp8_base = $True                                                                # fp8
$max_data_loader_n_workers = 8                                                   # max data loader n workers | 最大数据加载线程数
$persistent_data_loader_workers = $True                                          # save every n epochs | 每多少轮保存一次

$blocks_to_swap = 0                                                              # 交换的块数
$use_pinned_memory_for_block_swap = $True 
$img_in_txt_in_offloading = $True                                                # img in txt in offloading

#optimizer
$optimizer_type = "AdamW_adv"                                                    
# adamw8bit | adamw32bit | adamw16bit | adafactor | Lion | Lion8bit | 
# PagedLion8bit | AdamW | AdamW8bit | PagedAdamW8bit | AdEMAMix8bit | PagedAdEMAMix8bit
# DAdaptAdam | DAdaptLion | DAdaptAdan | DAdaptSGD | Sophia | Prodigy
# Adv series：AdamW_adv | Prodigy_adv | Adopt_adv | Simplified_AdEMAMix | Lion_adv | Lion_Prodigy_adv
$max_grad_norm = 1.0 # max grad norm | 最大梯度范数，默认为1.0
$fused_backward_pass = $False                                                    # Use fused backward pass (Adafactor)

$d_coef = "0.5"
$d0 = "1e-4"

# wandb log
$wandb_api_key = ""                                                              # wandbAPI KEY，用于登录

# save and load settings | 保存和输出设置
$output_name = "hv_1_5_lora_qinglong"  # output model name | 模型保存名称
$save_every_n_epochs = "2"            # save every n epochs | 每多少轮保存一次
$save_every_n_steps = ""              # save every n steps | 每多少步保存一次
$save_last_n_epochs = ""              # save last n epochs | 保存最后多少轮
$save_last_n_steps = ""               # save last n steps | 保存最后多少步

# save state | 保存训练状态
$save_state = $False                  # save training state | 保存训练状态
$save_state_on_train_end = $False     # save state on train end |只在训练结束最后保存训练状态
$save_last_n_epochs_state = ""        # save last n epochs state | 保存最后多少轮训练状态
$save_last_n_steps_state = ""         # save last n steps state | 保存最后多少步训练状态

#LORA_PLUS
$enable_lora_plus = $true
$loraplus_lr_ratio = 4                #recommend 4~16

#target blocks
$enable_blocks = $False
$enable_double_blocks_only = $False
$exclude_patterns = "" # Specify the values as a list. For example, "exclude_patterns=[r'.*single_blocks.*', r'.*double_blocks\.[0-9]\..*']".
$include_patterns = "" # Specify the values as a list. For example, "include_patterns=[r'.*single_blocks\.\d{2}\.linear.*']".

#lycoris组件
$enable_lycoris = $False # 开启lycoris
$conv_dim = 0 #卷积 dim，推荐＜32
$conv_alpha = 0 #卷积 alpha，推荐1或者0.3
$algo = "lokr" # algo参数，指定训练lycoris模型种类
$dropout = 0 #lycoris专用dropout
$preset = "attn-mlp" #预设训练模块配置
$factor = 8 #只适用于lokr的因子，-1~8，8为全维度
$decompose_both = $false #适用于lokr的参数，对 LoKr 分解产生的两个矩阵执行 LoRA 分解
$block_size = 4 #适用于dylora,分割块数单位
$use_tucker = $false #适用于除 (IA)^3 和full
$use_scalar = $false #根据不同算法，自动调整初始权重
$train_norm = $false #归一化层
$dora_wd = $true #Dora方法分解，低rank使用
$full_matrix = $false  #全矩阵分解
$bypass_mode = $false #通道模式
$rescaled = 1 #适用于设置缩放，效果等同于OFT
$constrain = $false #设置值为FLOAT，效果等同于COFT

#sample | 输出采样图片
$enable_sample = $false #1开启出图，0禁用
$sample_at_first = 1 #是否在训练开始时就出图
$sample_prompts = "./toml/qinglong_hv_1_5.txt" #prompt文件路径
$sample_every_n_epochs = 1 #每n个epoch出一次图
$sample_every_n_steps = 0 #每n步出一次图

#metadata
$training_comment = "this LoRA model created by bdsqlsz'script" # training_comment | 训练介绍
$metadata_title = "" # metadata title | 元数据标题
$metadata_author = "" # metadata author | 元数据作者
$metadata_description = "" # metadata contact | 元数据联系方式
$metadata_license = "" # metadata license | 元数据许可证
$metadata_tags = "" # metadata tags | 元数据标签

#huggingface settings
$async_upload = $False # push to hub | 推送到huggingface
$huggingface_repo_id = "" # huggingface repo id | huggingface仓库id
$huggingface_repo_type = "" # huggingface repo type | huggingface仓库类型
$huggingface_path_in_repo = "" # huggingface path in repo | huggingface仓库路径
$huggingface_token = "" # huggingface token | huggingface仓库token
$huggingface_repo_visibility = "" # huggingface repo visibility | huggingface仓库可见性
$save_state_to_huggingface = $False # save state to huggingface | 保存训练状态到huggingface
$resume_from_huggingface = $False # resume from huggingface | 从huggingface恢复训练

#DDP | 多卡设置
$multi_gpu = $False                         #multi gpu | 多显卡训练开关
$ddp_timeout = 120 #ddp timeout | ddp超时时间，单位秒
$ddp_gradient_as_bucket_view = 1 #ddp gradient as bucket view
$ddp_static_graph = 1 #ddp static graph

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
$env:CUDA_VISIBLE_DEVICES="0"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:VSLANG = "1033"
$ext_args = [System.Collections.ArrayList]::new()
$launch_args = [System.Collections.ArrayList]::new()
$laungh_script = "hv_1_5_train_network"
$network_module = "networks.lora_hv_1_5"
$has_network_args = $False

# HunyuanVideo 1.5 specific arguments
[void]$ext_args.Add("--text_encoder=$text_encoder")
[void]$ext_args.Add("--byt5=$byt5")

if ($task -ine "t2v") {
  [void]$ext_args.Add("--task=$task")
  [void]$ext_args.Add("--image_encoder=$image_encoder")
}

if ($dit_dtype) {
  [void]$ext_args.Add("--dit_dtype=$dit_dtype")
}

if ($fp8_vl) {
  [void]$ext_args.Add("--fp8_vl")
}

if ($fp8_scaled) {
  [void]$ext_args.Add("--fp8_scaled")
}

if ($vae_sample_size -ne 128) {
  [void]$ext_args.Add("--vae_sample_size=$vae_sample_size")
}

if ($vae_enable_patch_conv) {
  [void]$ext_args.Add("--vae_enable_patch_conv")
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
if ($guidance_scale) {
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

if ($network_weights) {
  [void]$ext_args.Add("--network_weights=$network_weights")
  if ($dim_from_weights) {
    [void]$ext_args.Add("--dim_from_weights")
  }
}

if ($enable_lycoris) {
  $network_module = "lycoris.kohya"
  $network_dropout = "0"
  [void]$ext_args.Add("--network_args")
  [void]$ext_args.Add("algo=$algo")
  if ($algo -ine "ia3" -and $algo -ine "diag-oft") {
    if ($algo -ine "full") {
      if ($conv_dim) {
        [void]$ext_args.Add("conv_dim=$conv_dim")
        if ($conv_alpha) {
          [void]$ext_args.Add("conv_alpha=$conv_alpha")
        }
      }
      if ($use_tucker) {
        [void]$ext_args.Add("use_tucker=True")
      }
      if ($algo -ine "dylora") {
        if ($dora_wd) {
          [void]$ext_args.Add("dora_wd=True")
        }
        if ($bypass_mode) {
          [void]$ext_args.Add("bypass_mode=True")
        }
        else {
          [void]$ext_args.Add("bypass_mode=False")
        }
        if ($use_scalar) {
          [void]$ext_args.Add("use_scalar=True")
        }
      }
    }
    [void]$ext_args.Add("preset=$preset")
  }
  if ($dropout -and $algo -ieq "locon") {
    [void]$ext_args.Add("dropout=$dropout")
  }
  if ($train_norm -and $algo -ine "ia3") {
    [void]$ext_args.Add("train_norm=True")
  }
  if ($algo -ieq "lokr") {
    [void]$ext_args.Add("factor=$factor")
    if ($decompose_both) {
      [void]$ext_args.Add("decompose_both=True")
    }
    if ($full_matrix) {
      [void]$ext_args.Add("full_matrix=True")
    }
  }
  elseif ($algo -ieq "dylora") {
    [void]$ext_args.Add("block_size=$block_size")
  }
  elseif ($algo -ieq "diag-oft") {
    if ($rescaled) {
      [void]$ext_args.Add("rescaled=True")
    }
    if ($constrain) {
      [void]$ext_args.Add("constrain=$constrain")
    }
  }
}
elseif ($enable_lora_plus) {
  [void]$ext_args.Add("--network_args")
  $has_network_args = $True
  if ($loraplus_lr_ratio) {
    [void]$ext_args.Add("loraplus_lr_ratio=$loraplus_lr_ratio")
  }
}
elseif ($enable_blocks) {
  if (!$has_network_args) {
    [void]$ext_args.Add("--network_args")
    $has_network_args = $True
  }
  if ($enable_double_blocks_only) {
    [void]$ext_args.Add("exclude_patterns=[r'.*single_blocks.*']")
    $exclude_patterns = ""
    $include_patterns = ""
  }
  if ($exclude_patterns) {
    [void]$ext_args.Add("exclude_patterns=$exclude_patterns")
  }
  if ($include_patterns) {
    [void]$ext_args.Add("include_patterns=$include_patterns")
  }
}

if ($network_dim) {
  [void]$ext_args.Add("--network_dim=$network_dim")
}

if ($network_alpha) {
  [void]$ext_args.Add("--network_alpha=$network_alpha")
}

if ($network_dropout -ne 0) {
  [void]$ext_args.Add("--network_dropout=$network_dropout")
}

if ($network_module) {
  [void]$ext_args.Add("--network_module=$network_module")
}

if ($scale_weight_norms -ne 0) {
  [void]$ext_args.Add("--scale_weight_norms=$scale_weight_norms")
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

if ($vae_dtype) {
  [void]$ext_args.Add("--vae_dtype=$vae_dtype")
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
  if ($use_pinned_memory_for_block_swap){
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

if ($optimizer_type -ieq "Prodigy") {
  [void]$ext_args.Add("--optimizer_type=$optimizer_type")
  [void]$ext_args.Add("--optimizer_args")
  [void]$ext_args.Add("weight_decay=0.01")
  [void]$ext_args.Add("betas=.9,.99")
  [void]$ext_args.Add("decouple=True")
  [void]$ext_args.Add("use_bias_correction=True")
  $lr = "1"
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
