# musubi-tuner-scripts

original codebase from kohya_ss

https://github.com/kohya-ss/musubi-tuner

## 🔧 Setting up the Environment

  Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Installation

Clone the repo with `--recurse-submodules`:

```
git clone --recurse-submodules https://github.com/sdbds/musubi-tuner-scripts.git
```

# MUST USE --recurse-submodules

Dependencies are managed from `pyproject.toml` with `uv`; requirements files
are no longer the source of truth for new installs.

### Windows
Run the following PowerShell script:
```powershell
./1.install-uv-qinglong.ps1
```

For a dependency-only install without model download prompts:

```powershell
uv sync --extra cu130 --extra gui --extra lycoris --extra attention --index-strategy unsafe-best-match
```

#### VS Studio 2022 for torch compile
Download from Microsoft offical link:
https://aka.ms/vs/17/release/vs_community.exe

Install C++ desktop and language package with English(especially for asian computer)

### Linux
1. First install PowerShell:
```bash
./0install pwsh.sh
```
2. Then run the installation script using PowerShell:
```powershell
sudo pwsh ./1.install-uv-qinglong.ps1
```
use sudo pwsh if you in Linux.

## Usage

edit 2、3、4 script before you run.

## GUI

除了 PowerShell 脚本外，项目还提供基于 NiceGUI 的图形化界面，支持完整的训练工作流程（打标 → 缓存 → 训练 → 推理）。

### 启动 GUI

```powershell
# Windows: 双击或运行
./1.6.GUI.ps1

# 指定端口
./1.6.GUI.ps1 -Port 8888

# 云模式（允许外部访问）
./1.6.GUI.ps1 -Cloud

# 原生窗口模式（不依赖浏览器）
./1.6.GUI.ps1 -Native
```

也可以直接用 Python 启动：

```bash
python gui/launch.py
python gui/launch.py --port 8888
python gui/launch.py --cloud
python gui/launch.py --native
python gui/launch.py --no-browser
```

启动后访问 `http://127.0.0.1:7788`（默认端口）即可使用。

### 支持的模型架构

FLUX.2、FLUX Kontext、Wan2.1、HunyuanVideo、FramePack、Long-CAT、Z-Image、HV 1.5、Qwen Image、Lens、Ideogram-4、HiDream O1、Krea-2

### GUI 文档

详细的使用说明、参数映射、项目结构等文档请参见 [`gui/README.md`](gui/README.md)。

<details>
<summary>

### 2cache_latent_and_text_encoder.ps1</summary>

```
# Cache lantent
$dataset_config = "./toml/qinglong-datasets.toml"            # path to dataset config .toml file | 数据集配置文件路径
$vae = "./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt" # VAE directory | VAE路径
$vae_dtype = ""                                              # fp16 | fp32 |bf16 default: fp16
$vae_chunk_size = 32                                         # chunk size for CausalConv3d in VAE
$vae_tiling = $True                                          # enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled
$vae_spatial_tile_sample_min_size = 256                      # spatial tile sample min size for VAE, default 256
$device = ""                                                 # cuda | cpu
$batch_size = ""                                             # batch size, override dataset config if dataset batch size > this
$num_workers = 0                                             # number of workers for dataset. default is cpu count-1
$skip_existing = $True                                       # skip existing cache files
$debug_mode = ""                                             # image | console
$console_width = $Host.UI.RawUI.WindowSize.Width             # console width
$console_back = "black"                                      # console background color
$console_num_images = 16                                     # number of images to show in console

# Cache text encoder
$text_encoder1 = "./ckpts/text_encoder/llava_llama3_fp16.safetensors"     # Text Encoder 1 directory | 文本编码器路径
$text_encoder2 = "./ckpts/text_encoder_2/clip_l.safetensors"              # Text Encoder 2 directory | 文本编码器路径
$text_encoder_batch_size = "16"                                           # batch size
$text_encoder_device = ""                                                 # cuda | cpu
$text_encoder_dtype = "bf16"                                              # fp16 | fp32 |bf16 default: fp16
$fp8_llm = $False                                                         # enable fp8 for text encoder
$text_encoder_num_workers = 0                                             # number of workers for dataset. default is cpu count-1
$text_encoder_skip_existing = $False                                       # skip existing cache files
```
</details>

<details>
<summary>

### 3、train.ps1
</summary>

```
#训练模式(Lora、db)
$train_mode = "Lora"

# model_path
$dataset_config = "./toml/qinglong-datasets.toml"                                   # path to dataset config .toml file | 数据集配置文件路径
$dit = "./ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" # DiT directory | DiT路径
$vae = "./ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt"                        # VAE directory | VAE路径
$text_encoder1 = "./ckpts/text_encoder/llava_llama3_fp16.safetensors"               # Text Encoder 1 directory | 文本编码器路径
$text_encoder2 = "./ckpts/text_encoder_2/clip_l.safetensors"                        # Text Encoder 2 directory | 文本编码器路径

$resume = ""                                                                        # resume from state | 从某个状态文件夹中恢复训练
$network_weights = ""                                                               # pretrained weights for LoRA network | 若需要从已有的 LoRA 模型上继续训练，请填写 LoRA 模型路径。

#COPY machine | 差异炼丹法
$base_weights = "" #指定合并到底模basemodel中的模型路径，多个用空格隔开。默认为空，不使用。
$base_weights_multiplier = "1.0" #指定合并模型的权重，多个用空格隔开，默认为1.0。

#train config | 训练配置
$max_train_steps = ""                                                                # max train steps | 最大训练步数
$max_train_epochs = 15                                                               # max train epochs | 最大训练轮数
$gradient_checkpointing = 1                                                          # 梯度检查，开启后可节约显存，但是速度变慢
$gradient_accumulation_steps = 4                                                     # 梯度累加数量，变相放大batchsize的倍数
$guidance_scale = 1.0
$seed = 1026 # reproducable seed | 设置跑测试用的种子，输入一个prompt和这个种子大概率得到训练图。可以用来试触发关键词

#timestep sampling
$timestep_sampling = "sigmoid" # 时间步采样方法，可选 sd3用"sigma"、普通DDPM用"uniform" 或 flux用"sigmoid" 或者 "shift". shift需要修改discarete_flow_shift的参数
$discrete_flow_shift = 1.0 # Euler 离散调度器的离散流位移，sd3默认为3.0
$sigmoid_scale = 1.0 # sigmoid 采样的缩放因子，默认为 1.0。较大的值会使采样更加均匀

$weighting_scheme = ""      # sigma_sqrt, logit_normal, mode, cosmap, uniform, none
$logit_mean = 0           # logit mean | logit 均值 默认0.0 只在logit_normal下生效
$logit_std = 1.0            # logit std | logit 标准差 默认1.0 只在logit_normal下生效
$mode_scale = 1.29          # mode scale | mode 缩放 默认1.29 只在mode下生效
$min_timestep = 0           #最小时序，默认值0
$max_timestep = 1000        #最大时间步 默认1000
$show_timesteps = ""        #是否显示timesteps， console/images

# Learning rate | 学习率
$lr = "1e-3"
# $unet_lr = "5e-4"
# $text_encoder_lr = "2e-5"
$lr_scheduler = "cosine_with_min_lr"
# "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup" | PyTorch自带6种动态学习率函数
# constant，常量不变, constant_with_warmup 线性增加后保持常量不变, linear 线性增加线性减少, polynomial 线性增加后平滑衰减, cosine 余弦波曲线, cosine_with_restarts 余弦波硬重启，瞬间最大值。
# 新增cosine_with_min_lr(适合训练lora)、warmup_stable_decay(适合训练db)、inverse_sqrt
$lr_warmup_steps = 0 # warmup steps | 学习率预热步数，lr_scheduler 为 constant 或 adafactor 时该值需要设为0。仅在 lr_scheduler 为 constant_with_warmup 时需要填写这个值
$lr_decay_steps = 0.25 # decay steps | 学习率衰减步数，仅在 lr_scheduler 为warmup_stable_decay时 需要填写，一般是10%总步数
$lr_scheduler_num_cycles = 1 # restarts nums | 余弦退火重启次数，仅在 lr_scheduler 为 cosine_with_restarts 时需要填写这个值
$lr_scheduler_power = 1     #Polynomial power for polynomial scheduler |余弦退火power
$lr_scheduler_timescale = 0 #times scale |时间缩放，仅在 lr_scheduler 为 inverse_sqrt 时需要填写这个值，默认同lr_warmup_steps
$lr_scheduler_min_lr_ratio = 0.1 #min lr ratio |最小学习率比率，仅在 lr_scheduler 为 cosine_with_min_lr、、warmup_stable_decay 时需要填写这个值，默认0

#network settings
$network_dim = 32 # network dim | 常用 4~128，不是越大越好
$network_alpha = 16 # network alpha | 常用与 network_dim 相同的值或者采用较小的值，如 network_dim的一半 防止下溢。默认值为 1，使用较小的 alpha 需要提升学习率。
$network_dropout = 0 # network dropout | 常用 0~0.3
$dim_from_weights = $True # use dim from weights | 从已有的 LoRA 模型上继续训练时，自动获取 dim
$scale_weight_norms = 0 # scale weight norms (1 is a good starting point)| scale weight norms (1 is a good starting point)

# $train_unet_only = 1 # train U-Net only | 仅训练 U-Net，开启这个会牺牲效果大幅减少显存使用。6G显存可以开启
# $train_text_encoder_only = 0 # train Text Encoder only | 仅训练 文本编码器

#precision and accelerate/save memory
$attn_mode = "xformers"                                                                # "flash", "sageattn", "xformers", "sdpa"
$split_attn = $True                                                                 # split attention | split attention
$mixed_precision = "bf16"                                                           # fp16 |bf16 default: bf16
# $full_fp16 = $False
# $full_bf16 = $True
$dit_dtype = ""                                                                     # fp16 | fp32 |bf16 default: bf16

$vae_dtype = ""                                                                     # fp16 | fp32 |bf16 default: fp16
$vae_tiling = $True                                                                 # enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled
$vae_chunk_size = 32                                                                # chunk size for CausalConv3d in VAE
$vae_spatial_tile_sample_min_size = 256                                             # spatial tile sample min size for VAE, default 256

$text_encoder_dtype = ""                                                            # fp16 | fp32 |bf16 default: fp16

$fp8_base = $True                                                                   # fp8
$fp8_llm = $False                                                                   # fp8 for LLM
$max_data_loader_n_workers = 8                                                      # max data loader n workers | 最大数据加载线程数
$persistent_data_loader_workers = $True                                             # save every n epochs | 每多少轮保存一次

$blocks_to_swap = 0                                                                 # 交换的块数
$img_in_txt_in_offloading = $True                                                   # img in txt in offloading

#optimizer
$optimizer_type = "AdamW8bit"                                                       
# adamw8bit | adamw32bit | adamw16bit | adafactor | Lion | Lion8bit | 
# PagedLion8bit | AdamW | AdamW8bit | PagedAdamW8bit | AdEMAMix8bit | PagedAdEMAMix8bit
# DAdaptAdam | DAdaptLion | DAdaptAdan | DAdaptSGD | Sophia | Prodigy
$max_grad_norm = 1.0 # max grad norm | 最大梯度范数，默认为1.0

# wandb log
$wandb_api_key = ""                   # wandbAPI KEY，用于登录

# save and load settings | 保存和输出设置
$output_name = "hyvideo-qinglong"  # output model name | 模型保存名称
$save_every_n_epochs = "10"           # save every n epochs | 每多少轮保存一次
$save_every_n_steps = ""              # save every n steps | 每多少步保存一次
$save_last_n_epochs = ""            # save last n epochs | 保存最后多少轮
$save_last_n_steps = ""               # save last n steps | 保存最后多少步

# save state | 保存训练状态
$save_state = $False                  # save training state | 保存训练状态
$save_state_on_train_end = $False     # save state on train end |只在训练结束最后保存训练状态
$save_last_n_epochs_state = ""        # save last n epochs state | 保存最后多少轮训练状态
$save_last_n_steps_state = ""         # save last n steps state | 保存最后多少步训练状态

#LORA_PLUS
$enable_lora_plus = $True
$loraplus_lr_ratio = 4                #recommend 4~16

#target blocks
$enable_blocks = $False
$enable_double_blocks_only = $False
$exclude_patterns="" # Specify the values as a list. For example, "exclude_patterns=[r'.*single_blocks.*', r'.*double_blocks\.[0-9]\..*']".
$include_patterns="" # Specify the values as a list. For example, "include_patterns=[r'.*single_blocks\.\d{2}\.linear.*']".

#lycoris组件
$enable_lycoris = $False # 开启lycoris
$conv_dim = 0 #卷积 dim，推荐＜32
$conv_alpha = 0 #卷积 alpha，推荐1或者0.3
$algo = "lokr" # algo参数，指定训练lycoris模型种类，
#包括lora(就是locon)、
#loha
#IA3
#lokr
#dylora
#full(DreamBooth先训练然后导出lora)
#diag-oft
#它通过训练适用于各层输出的正交变换来保留超球面能量。
#根据原始论文，它的收敛速度比 LoRA 更快，但仍需进行实验。
#dim 与区块大小相对应：我们在这里固定了区块大小而不是区块数量，以使其与 LoRA 更具可比性。

$dropout = 0 #lycoris专用dropout
$preset = "attn-mlp" #预设训练模块配置
#full: default preset, train all the layers in the UNet and CLIP|默认设置，训练所有Unet和Clip层
#full-lin: full but skip convolutional layers|跳过卷积层
#attn-mlp: train all the transformer block.|kohya配置，训练所有transformer模块
#attn-only：only attention layer will be trained, lot of papers only do training on attn layer.|只有注意力层会被训练，很多论文只对注意力层进行训练。
#unet-transformer-only： as same as kohya_ss/sd_scripts with disabled TE, or, attn-mlp preset with train_unet_only enabled.|和attn-mlp类似，但是关闭te训练
#unet-convblock-only： only ResBlock, UpSample, DownSample will be trained.|只训练卷积模块，包括res、上下采样模块
#./toml/example_lycoris.toml: 也可以直接使用外置配置文件，制定各个层和模块使用不同算法训练，需要输入位置文件路径，参考样例已添加。

$factor = 8 #只适用于lokr的因子，-1~8，8为全维度
$decompose_both = $false #适用于lokr的参数，对 LoKr 分解产生的两个矩阵执行 LoRA 分解（默认情况下只分解较大的矩阵）
$block_size = 4 #适用于dylora,分割块数单位，最小1也最慢。一般4、8、12、16这几个选
$use_tucker = $false #适用于除 (IA)^3 和full
$use_scalar = $false #根据不同算法，自动调整初始权重
$train_norm = $false #归一化层
$dora_wd = 1 #Dora方法分解，低rank使用。适用于LoRA, LoHa, 和LoKr
$full_matrix = $false  #全矩阵分解
$bypass_mode = $false #通道模式，专为 bnb 8 位/4 位线性层设计。(QLyCORIS)适用于LoRA, LoHa, 和LoKr
$rescaled = 1 #适用于设置缩放，效果等同于OFT
$constrain = $false #设置值为FLOAT，效果等同于COFT

#sample | 输出采样图片
$enable_sample = $True #1开启出图，0禁用
$sample_at_first = 1 #是否在训练开始时就出图
$sample_every_n_epochs = 2 #每n个epoch出一次图
$sample_prompts = "./toml/qinglong.txt" #prompt文件路径

#metadata
$training_comment = "this LoRA model created by bdsqlsz'script" # training_comment | 训练介绍，可以写作者名或者使用触发关键词
$metadata_title = "" # metadata title | 元数据标题
$metadata_author = "" # metadata author | 元数据作者
$metadata_description = "" # metadata contact | 元数据联系方式
$metadata_license = "" # metadata license | 元数据许可证
$metadata_tags = "" # metadata tags | 元数据标签

#huggingface settings
$async_upload = $False # push to hub | 推送到huggingface
$huggingface_repo_id = "" # huggingface repo id | huggingface仓库id
$huggingface_repo_type = "dataset" # huggingface repo type | huggingface仓库类型
$huggingface_path_in_repo = "" # huggingface path in repo | huggingface仓库路径
$huggingface_token = "" # huggingface token | huggingface仓库token
$huggingface_repo_visibility = "" # huggingface repo visibility | huggingface仓库可见性
$save_state_to_huggingface = $False # save state to huggingface | 保存训练状态到huggingface
$resume_from_huggingface = $False # resume from huggingface | 从huggingface恢复训练

#DDP | 多卡设置
$multi_gpu = $False                         #multi gpu | 多显卡训练开关，0关1开， 该参数仅限在显卡数 >= 2 使用
# $highvram = 0                            #高显存模式，开启后会尽量使用显存
# $deepspeed = 0                         #deepspeed | 使用deepspeed训练，0关1开， 该参数仅限在显卡数 >= 2 使用
# $zero_stage = 2                        #zero stage | zero stage 0,1,2,3,阶段2用于训练 该参数仅限在显卡数 >= 2 使用
# $offload_optimizer_device = ""      #offload optimizer device | 优化器放置设备，cpu或者nvme, 该参数仅限在显卡数 >= 2 使用
# $fp16_master_weights_and_gradients = 0 #fp16 master weights and gradients | fp16主权重和梯度，0关1开， 该参数仅限在显卡数 >= 2 使用

$ddp_timeout = 120 #ddp timeout | ddp超时时间，单位秒， 该参数仅限在显卡数 >= 2 使用
$ddp_gradient_as_bucket_view = 1 #ddp gradient as bucket view | ddp梯度作为桶视图，0关1开， 该参数仅限在显卡数 >= 2 使用
$ddp_static_graph = 1 #ddp static graph | ddp静态图，0关1开， 该参数仅限在显卡数 >= 2 使用
```
</details>

<details>
<summary>

### 4convert_lora.ps1
</summary>

```
$input_path="./output_dir/hyvideo-qinglong.safetensors"
$output_path="./output_dir/hyvideo-qinglong_comfy.safetensors"
$target="other" # "other" or "default"
```

</details>

<details>
<summary>

### 5generate.ps1
</summary>

```
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
```
</details>
