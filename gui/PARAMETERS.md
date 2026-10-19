# GUI 参数映射文档

本文档记录所有 PowerShell 脚本参数与 GUI 的映射关系。

### 架构专用契约

#### Qwen Image 2.1 / MOSS-Image

`qwen_image_21` 和 `moss_image` 使用各自的缓存、LoRA、全量训练和生成模块；不能复用旧 Qwen Image 缓存。预设中的 `arch` 分别为 `Qwen Image 2.1` / `MOSS-Image`，`version` 分别为 `standard`/`turbo` 和 `base`/`sft`。MOSS 仅文生图，拒绝参考图、蒙版和 FP8 参数。

| GUI 字段 | CLI 参数 / 行为 | 默认与约束 |
| --- | --- | --- |
| `train_mode` | 选择 `*_train_network` / `*_train` | LoRA / 全量；全量不传 LoRA 参数 |
| `text_encoder_path` | `--text_encoder` | Qwen 2.1 用 Qwen3-VL 8B BF16；MOSS 用 Qwen3.5 9B BF16，传单文件或官方首分片，其余分片放同目录 |
| `vae_dtype` | `--vae_dtype` | Qwen 2.1 为 BF16；MOSS 固定 `float32` |
| `network_dim` / `network_alpha` | 同名参数 | Qwen 2.1 默认 16/16；MOSS 默认 32/32；仅 LoRA |
| `learning_rate` / `optimizer_type` | 同名参数 | LoRA 均为 `1e-4`、AdamW；全量 Qwen 2.1 为 `5e-6`、AdamW，MOSS 为 `1e-5`、Adafactor |
| `timestep_sampling` / `discrete_flow_shift` | 同名参数 | Qwen 2.1：`shift` / 3；MOSS：`sigmoid` / 1；权重策略均为 `none` |
| `enable_sample` | 开启时才发送采样提示词和频率 | 默认关闭；开启后需要提示词及对应 VAE/文本编码器 |
| `mem_eff_save` / `save_state_on_train_end` | 同名开关 | 全量预设均开启；状态文件用于恢复优化器等训练状态 |
| `infer_steps` / `guidance_scale` / `flow_shift` | 同名参数 | Qwen 2.1：40/1/动态，Turbo：8/1/动态；MOSS：50/3/5 |
| `uncond_skip_layers` | `--uncond_skip_layers` | MOSS 默认 `22`；逗号分隔层号 |

MOSS 安装器默认下载 `Comfy-Org/Qwen3.5/text_encoders/qwen3.5_9b_bf16.safetensors`（约 19.3 GB），只加载语言模型张量，跳过视觉/输出头。官方 BF16 分片共约 16.8 GB，仍支持同目录自动发现；无需为单文件安装额外分片。各架构缓存应写入不同目录，MOSS VAE 与文本缓存的精度不可沿用其他架构设置。

普通 LoRA 和全量训练保留 GUI 原有的优化器目录、别名解析和自定义类路径，不限于 AdamW/Adafactor。只有全量块交换或融合反向传播按当前模块边界校验；这两种内存优化模式下，AdamW/Adafactor 别名使用模块要求的原生实现，未编辑的参数模板随模式切换同步更新。融合反向传播仍仅支持原生 Adafactor；本次适配不修改模块的优化器兼容性。

MOSS 全量块交换／融合反向传播要求梯度累积为 1，GUI 拒绝多 GPU 开关并显式以单机单进程启动，避免继承本机 Accelerate 的多进程默认值；融合反向传播还要求 `max_grad_norm=0`。普通训练不受这些专用模式约束。

#### DLSS-NR HT / CC

NR 表单使用独立的 `nr_` 字段，构建命令时剥离前缀；其他架构遗留的参数不会传入。训练入口为 `dlssnr_train_network`（LoRA）或 `dlssnr_train`（全量），默认单进程、constant LR。全量微调默认 QAT 开启、训练混合精度 FP16 AMP、EMA `0.99`；LoRA 保留 FP32 训练，QAT 和 EMA 关闭。已有预设中的显式设置保留。进程数大于 1 时通过单机 torchrun DDP 启动，不读取全局 Accelerate 启动配置。

NR 与其他架构共用顶层页签、优化器目录/参数模板及数值编辑控件。独立字段仅用于隔离不同模型的配置，不再使用嵌套的专用训练页。普通训练不要求前向验收报告；GUI 不提供或发送验收报告、部署目标、开发 smoke 开关，旧预设中的这些字段被忽略。模型目录始终必填。

| GUI 字段 | CLI 参数 | 约束 |
| --- | --- | --- |
| 数据集页面保存的项目配置 | `--dataset_config` | 统一导出项目 dataset-only TOML；训练页不再拥有 `nr_dataset_config` 控件 |
| `nr_model_dir` | `--model_dir` | canonical HT/CC 目录；正式训练必填，包含有效 `model_config.json`；不接受 bundle 根目录或 CC 原始记录目录 |
| `nr_profile` | `--profile` | 默认 `auto` 从模型元数据识别；显式可选 `dlss_nr_310_8_0`（HT）或 `dlss_nr_310_9_16_cc`（CC），必须与模型一致 |
| `train_mode` | 入口选择 | `lora` / `finetune`，全量入口不接收 LoRA 参数 |
| `nr_training_mode` | `--training_mode` | `single_frame` / `temporal` |
| `nr_shuffle_dataset` | `--shuffle_dataset` / `--no-shuffle_dataset` | 默认开启；关闭恢复固定数据顺序 |
| `nr_sequence_length` / `nr_burn_in` / `nr_tbptt_length` | 同名参数 | 仅时序；片段长度 = 预热 + 监督帧数，两者至少为 1 |
| `nr_learning_rate` / `nr_optimizer_type` / `nr_optimizer_args` | 同名参数 | 全量默认 `1e-5`，LoRA 默认 `1e-4`；复用公共优化器名称映射/模板，保留 NR AdamW 和 Adafactor 默认语义；不接受 CLI 注入 |
| `nr_d_coef` / `nr_d0` | `--optimizer_args` 中的对应键 | 与参数文本同步；自定义参数替换模板，不静默叠加 |
| `nr_lr_scheduler` | `--lr_scheduler` | 复用项目调度器工厂和下拉目录，默认 constant，可选择 cosine、linear、warmup 等 |
| `nr_lr_warmup_steps` / `nr_lr_decay_steps` | 同名参数 | 整数更新步数或小于 1 的比例；不按 microbatch 或 DDP rank 重复计步 |
| `nr_lr_scheduler_num_cycles` / `nr_lr_scheduler_power` / `nr_lr_scheduler_timescale` / `nr_lr_scheduler_min_lr_ratio` | 同名参数 | 公共调度参数；timescale 留空时遵循后端默认 |
| `nr_numerics_profile` | `--numerics_profile` | 训练默认 `train_experimental`，满足默认 SDPA 的要求；AMP/FP8/替代注意力均要求该模式 |
| `nr_mixed_precision` | `--mixed_precision` | `no` / `fp16` / `bf16`；全量默认 `fp16`，LoRA 默认 `no`；AMP 要求 CUDA，主权重仍为 FP32，与存储精度独立 |
| `nr_gradient_checkpointing` | `--gradient_checkpointing` | 训练默认开启；全量和 LoRA 均可用，该功能本身无需实验模式 |
| `nr_gradient_checkpointing_min_pixels` | `--gradient_checkpointing_min_pixels` | 非负整数，默认 `0`；仅填充后的模块输入特征图 `H * W` 严格大于阈值时重算，不含 batch 维度；正数要求开启梯度检查点，仅用于训练 |
| `nr_native_weight_qat` | `--native_weight_qat` / `--no-native_weight_qat` | 全量默认开启，LoRA 默认关闭；LoRA 要求 dropout 为 0；与 FP8 存储独立 |
| `nr_fp8_base` / `nr_fp8_scaled` | 同名参数 | 训练仅支持 LoRA 冻结基座存储；scaled 依赖 base，不代表 FP8 GEMM |
| `nr_attention_backend` / `nr_attention_scope` | 同名参数 | 训练默认 `sdpa` + `all`；可选 `native` / `sdpa` / `xformers` / `flash_attn`；Flash 需 AMP + global，Sage 仅推理可用 |
| `nr_max_overflow_retries` | `--max_overflow_retries` | 非负整数，默认 16；仅 FP16 显示，耗尽时后端停止 |
| `nr_num_processes` | torchrun `--nproc_per_node` | 正整数，默认 1；多进程只支持单机 DDP，不传给训练 parser |
| `nr_runtime_mode` | 推理覆盖策略 | `inherit` 默认不发送 runtime 参数；`override` 显式发送全部策略，包括关闭 FP8 的负向开关 |
| `nr_loss_pre` / `nr_loss_out` / `nr_loss_edge` / `nr_loss_temporal` | 同名参数 | 非负且至少一项为正；单帧时序损失为 0，时序损失非零时监督帧数至少为 2 |
| `nr_loss_profile` / `nr_loss_lowpass_sigma` | 同名参数 | 默认 `pixel`；`frequency_split` 启用低频目标匹配和输入边缘约束，sigma 默认 6，范围 `(0, 32]`，仅该模式发送 |
| `nr_base_anchor_weight` | `--base_anchor_weight` | 非负，默认 0 关闭；约束输出相对初始基座的偏离，增加参考前向开销 |
| `nr_control_randomization` | `--control_randomization` | 默认关闭；所有训练数据集必须使用固定条件，且 tone/structure 在 FP16 编码后均为正 |
| `nr_control_residual_sigma` | `--control_residual_sigma` | 仅控制随机化开启时发送，默认 6，范围 `(0, 32]` |
| `nr_control_anchor_probability` / `nr_control_corner_probability` | 同名参数 | 仅控制随机化开启时发送，各默认 0.25，各在 `[0, 1]` 内且和不超过 1 |
| `nr_dino_loss_weight` | `--dino_loss_weight` | 非负，默认 0 关闭；开启时加载冻结 DINOv3 特征模型 |
| `nr_dino_loss_model_type` / `nr_dino_loss_layer` / `nr_dino_loss_resize` | 同名参数 | 仅 DINO 权重大于 0 时发送；默认 `small` / `-4` / `224`，模型可选 small/small_plus/base/large，resize 为 `[16, 1024]` 内的 16 倍数 |
| `nr_dino_loss_use_gram` / `nr_dino_loss_use_norm` | 同名正向或 `--no-` 开关 | 仅 DINO 权重大于 0 时发送；均默认开启，显式关闭值不会丢失 |
| `nr_lora_profile` | `--network_args profile=...` | `vit_only` / `multiscale` |
| `nr_network_dim` / `nr_network_alpha` | 同名参数 | 仅 ViT；alpha 留空跟随 rank |
| `nr_rank_by_width` / `nr_alpha_by_width` | `--network_args` 中的字典 | 仅多尺度；宽度键为 `32`、`64`、`128`、`256`、`512`、`1024` 的字符串，不同时传标量 dim/alpha |
| `nr_prior_lr_multiplier` / `nr_scale_lr_multiplier` / `nr_temporal_blend_lr_multiplier` | 同名参数 | 仅全量，默认 `1.0` / `1.0` / `0.1` |
| `nr_compare_baseline` | `--no-compare_baseline` | 默认对比基座，关闭时传负向开关 |
| `nr_sample_every_n_steps` | `--sample_every_n_steps` | 非零要求 TOML 中有验证或时序评估清单 |
| `nr_eval_native` / `nr_eval_detail_diagnostics` / `nr_eval_content_preservation` | 同名参数 | 均默认关闭；任何一项开启均要求验证或时序评估清单；内容保持评估独立于训练 DINO 损失 |
| `nr_save_precision` | `--save_precision` | `native` 默认按原生 FP8/FP16/FP32 布局保存普通全量/EMA 权重，`fp32` 保留未舍入权重；LoRA 增量和精确续训快照始终 FP32 |
| `nr_save_state` / `nr_resume` | `--save_state` / `--resume` | 保存和恢复未舍入主权重及 optimizer/scheduler/RNG 状态；普通 native 权重文件不满足精确续训要求 |
| `nr_ema_decay` | `--ema_decay` / `--no-ema` | 全量默认 `0.99`，LoRA 默认留空；留空显式关闭，启用时严格在 `(0, 1)` 内，增加可训练参数的 FP32 副本；原始和 EMA 产品分开保存 |
| `nr_inference_mode` | 入口选择 | `image` 用 `dlssnr_generate_image`；`sequence` 用 `dlssnr_generate_video`，输出 PNG 序列 |
| `nr_sample_manifest` / `nr_sequence_manifest` | 同名参数 | 只传当前模式的 JSONL 清单，推理不要求 target |
| `nr_bucket_width` / `nr_bucket_height` | 同名参数 | 固定有效尺寸，使用后端几何规则校验，不是推理自动分桶 |

训练命令先经过真实后端 parser，再复用 `build_train_config` 校验；运行策略复用 `validate_runtime_policy`。表单加载前检查字段类型和范围，命令构建时再检查跨字段约束和数据集条件。CLI 路径以项目工作目录为基准，TOML 中的清单路径以该 TOML 的目录为基准。NR 不使用扩散模型的 latent/text-encoder 缓存、caption、提示词或 CFG，不提供 block swap、CPU activation offload、FSDP/ZeRO/DeepSpeed 或多机选项。

两个训练 PowerShell 脚本接受上述后端同名训练参数（不带 `nr_` 前缀），以及运行策略和调度器参数。两者默认 `-profile auto -save_precision native`，继续使用 native 注意力并关闭 checkpointing。全量脚本默认 `train_experimental`、FP16 AMP、QAT 开启、EMA `0.99`；LoRA 脚本保留 FP32 surrogate、QAT 和 EMA 关闭。全量可用 `-native_weight_qat:$false -mixed_precision no -no_ema` 关闭这三项；`-no_ema` 优先于默认衰减，`-ema_decay ''` 也可关闭 EMA。普通布尔参数使用 `$true`/`$false`，例如 `-shuffle_dataset $false`；switch 关闭语法如 `-native_weight_qat:$false`。DINO 的两个可选布尔参数未填写时不发送，避免关闭 DINO 时误传子参数。显式传入不适用的子参数由后端拒绝，不静默丢弃。EMA、QAT、损失和合成协议受后端恢复配置一致性检查约束。

以上全量默认值是前端和脚本策略，直接调用后端 CLI 时 QAT、EMA 仍默认关闭。生成 PowerShell 与 GUI 一样继承模型保存的 runtime，不再强制 `train_surrogate`。TF32 跟随外部环境及 PyTorch 设置。

两个训练脚本的 `-gradient_checkpointing_min_pixels` 默认均为 `0`，不会自动开启梯度检查点。可选示例为 `-gradient_checkpointing -gradient_checkpointing_min_pixels 20000`；更高阈值增加激活显存，示例值不作为默认值或通用推荐。GUI 关闭梯度检查点时将阈值归零；非法预设在修改表单前拒绝，旧预设缺少该字段时补为 `0`。修改阈值会改变恢复配置，须新建训练。当前后端仅接受 `dlssnr_runtime_v3` 格式的 runtime 元数据，前端不自动迁移旧版声明。

数据集页面选择 `DLSS-NR 配对图像` 后，下列参数写入各自的 `[[datasets]]`，不写入训练预设。支持多个数据集，每个数据集保留自己的分辨率、批次大小、重复次数和固定条件。

DLSS-NR 单帧、时序模板及后端未指定分辨率时的默认值为 `1024 × 1024`。模板开启分桶，因此该值是面积预算，显式保存的用户分辨率不会被覆盖。

| TOML 字段 | 含义与约束 |
| --- | --- |
| `image_directory` / `control_directory` | 训练目标图像 / 输入控制图像；复用图像编辑数据集的文件名配对规则，每张目标必须恰有一张输入，像素网格必须对齐 |
| `train_manifest` | 与目录配对互斥的 JSONL 清单；真实时序要求帧序、运动与有效区元数据，启用合成时使用单帧配对清单 |
| `synthetic_temporal` | 默认 false；将配对静止图在线合成为片段，要求训练模式为 temporal，不自动启用任何附加损失 |
| `synthetic_max_shift_px` | 仅合成启用时保存，默认 0.5，有限值且在 `[0, 1]` 内；关闭时保留编辑值但不导出 |
| `nr_controls_mode` | 目录固定为 `fixed`；JSONL 可选 `fixed` 或沿用逐帧 `controls_path` 的 `files` |
| `nr_style` | 非负整数风格 ID，默认 `0`，上限 `16777216` |
| `nr_tone` / `nr_structure` | 色调 / 结构，范围 `[0, 1]`，默认 `1` |
| `nr_skin` | 皮肤结构，范围 `[0, 1]`；默认 `-1` 跟随结构，仅自动蒙版开启时参与编码 |
| `nr_auto_mask` | 默认 `true`；控制模型的五通道条件编码，不生成分割图，也不替代监督损失掩码 |

固定参数在加载时直接编码，不生成临时 JSONL 或 `.npy` 文件。导入 NR TOML 时保留原文件的相对路径基准，之后导出到项目目录不会改变数据位置。

合成选项可从 `[general]` 继承；GUI 导入或读取旧项目时将有效值展开到各行，保存时去除已展开的全局字段。显式关闭的条目可忽略继承的位移，但不能声明本地位移；没有启用条目时，全局位移配置无效。合成条目的 `validation_manifest` 使用静止配对，`sequence_manifest` 始终要求真实运动序列。模板见 `toml/qinglong_dlssnr_synthetic_temporal.toml`。

HT 和 CC 使用各自的模型拓扑，LoRA QKV 布局仅 `fused_head_major`。LBFGS、Schedule-Free、需要模型对象的 AdamMini，以及要求高阶反向图的 SophiaH 不能用于当前 NR 更新循环。LoRARite 仅提供给 LoRA 模式，不能将全量权重误当作成对 LoRA 因子。其他第三方优化器仍需对应依赖已安装。

#### Mage-Flow

| Parameter | Type | Meaning | Constraint |
| --- | --- | --- | --- |
| is_edit | segmented | T2I 或 Edit 模式标识 | 缓存、训练与生成必须一致 |
| version | select | Standard 或 Turbo | 选择对应的 BF16 DiT 推荐权重 |
| cache_seed | integer | 稳定的 Latent 采样种子 | 仅用于 Latent 缓存 |
| mage_control_images | multiline paths | 按顺序排列的 Edit 参考图 | Edit：1–3 张；T2I：不可填写 |
| mage_steps | integer | Euler 采样步数 | Standard T2I 20、Standard Edit 30、Turbo 4 |
| mage_cfg_scale | float | CFG scale | Standard 5.0、Turbo 1.0 |
| mage_width / mage_height | integers | 显式输出尺寸 | 必须成对填写；T2I 默认 1024×1024 |
| mage_max_size | integer | 保持参考图宽高比时的最大尺寸 | 仅 Edit，默认 1024 |
| mage_attn_mode | select | Attention backend | 仅 `sdpa` 或 `flash2` |
| mage_output_path | file path | 生成图片文件 | 传给 `--output`，不是目录 |

GUI 与原生脚本调用
`mage_flow_cache_latents` / `mage_flow_cache_text_encoder_outputs`、
`mage_flow_train_network` 和 `mage_flow_generate_image`。推荐文件路径如下：

| Profile / Component | Local path |
| --- | --- |
| T2I Standard DiT | `./ckpts/diffusion_models/mage_flow_bf16.safetensors` |
| T2I Turbo DiT | `./ckpts/diffusion_models/mage_flow_turbo_bf16.safetensors` |
| Edit Standard DiT | `./ckpts/diffusion_models/mage_flow_edit_bf16.safetensors` |
| Edit Turbo DiT | `./ckpts/diffusion_models/mage_flow_edit_turbo_bf16.safetensors` |
| VAE | `./ckpts/vae/mage_flow_vae_bf16.safetensors` |
| Shared Qwen3-VL 4B Text Encoder | `./ckpts/text_encoder/qwen_3_VL_4b.safetensors` |

训练仅支持 LoRA，混合精度必须为 BF16；默认采用 `shift`、flow shift `6.0`
和 `weighting_scheme=none`。Attention 仅支持 SDPA/FlashAttention 2，
`blocks_to_swap` 范围为 0–10，且不支持 Fullgraph、LyCORIS、全量微调或
自定义 `include_patterns` / `exclude_patterns`。
Processor 资源从固定的 `microsoft/Mage-Flow/text_encoder` 仓库自动解析，
GUI 不提供 processor/tokenizer 路径。安装器不下载 INT8 ConvRot 权重。

模型文件来自 [Comfy-Org/Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow)，
后端契约见 [Mage-Flow upstream guide](https://github.com/sdbds/musubi-tuner/blob/qinglong/docs/mage_flow.md)。
轻量契约测试不下载真实权重，因此真实权重推理与训练的一致性仍属实验性。

## 缓存脚本参数 (step2_cache)

### 基础参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| cache_latents_enabled | checkbox | 启用 Latent 缓存阶段 | checkbox |
| cache_text_encoder_enabled | checkbox | 启用 Text Encoder 缓存阶段 | checkbox |
| dataset_config | path | TOML 配置文件 | path_selector |
| vae | path | VAE 模型路径 | path_selector |
| vae_dtype | select | VAE 数据类型 | select [float32/float16/bfloat16] |
| device | select | 运行设备 | select [cuda/cpu] |
| batch_size | number | 批处理大小 | number_input |
| num_workers | number | 数据加载线程数 | number_input |
| skip_existing | checkbox | 跳过已存在的缓存 | checkbox |

### 架构特定参数

#### HunyuanVideo / HV 1.5 / FramePack / Kontext
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| vae_chunk_size | number | VAE chunk size | number_input |
| vae_tiling | checkbox | 启用 VAE tiling | checkbox |
| vae_spatial_tile_sample_min_size | number | 最小 tile 大小 | number_input |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| vae_cache_cpu | checkbox | VAE 缓存到 CPU | checkbox |
| clip | path | CLIP 模型路径 | path_selector |

#### FramePack
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| image_encoder | path | Image Encoder 路径 | path_selector |
| f1 | checkbox | F1 模式 | checkbox |
| one_frame | checkbox | 单帧模式 | checkbox |
| one_frame_no_2x | checkbox | 禁用 2x 放大 | checkbox |
| one_frame_no_4x | checkbox | 禁用 4x 放大 | checkbox |

### Text Encoder 缓存参数

#### FLUX.2 / Z-Image / Qwen Image
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder | path | Text Encoder 路径 | path_selector |
| fp8_text_encoder | checkbox | 使用 FP8 | checkbox |

#### HunyuanVideo / HV 1.5 / FramePack / Kontext / Long-CAT
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder1 | path | Text Encoder 1 (LLM) | path_selector |
| text_encoder2 | path | Text Encoder 2 (CLIP) | path_selector |
| text_encoder_dtype | select | TE 数据类型 | select |
| fp8_llm | checkbox | 使用 FP8 LLM | checkbox |

#### Lens
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder_dtype | select | Lens TE 运行精度 | select [bfloat16/float16/float32/fp8] |
| text_encoder_cache_precision | select | Lens TE 缓存保存精度 | select [auto/bf16/fp16/fp32/fp8/nvfp4] |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| t5 | path | T5 模型路径 | path_selector |
| fp8_t5 | checkbox | 使用 FP8 T5 | checkbox |

#### 通用 TE 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| batch_size | number | TE 批处理大小 | number_input |
| device | select | TE 运行设备 | select |
| num_workers | number | TE 数据加载线程 | number_input |
| skip_existing | checkbox | 跳过已存在的 TE 缓存 | checkbox |

#### D-OPSD Teacher 缓存参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| dopsd_cache_teacher_outputs | checkbox | 缓存 D-OPSD multimodal teacher embeddings；支持 Z-Image、FLUX.2 Klein | checkbox |
| dopsd_teacher_text_encoder | path | Z-Image / FLUX.2 使用的 Qwen3-VL teacher 权重文件 | path_selector |
| dopsd_teacher_already_reweighted | checkbox | teacher 已离线重加权 | checkbox |
| dopsd_teacher_allow_raw_vlm | checkbox | 允许 raw VLM teacher，仅用于 ablation | checkbox |
| dopsd_teacher_dtype | select | teacher encoder dtype | select |

### 调试参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| debug_mode | select | 调试模式 | select [image/console] |
| console_width | number | 控制台宽度 | number_input |
| console_back | select | 背景颜色 | select |
| console_num_images | number | 显示图片数量 | number_input |

## 训练脚本参数 (step3_train)

### 基础参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| dataset_config | path | TOML 配置文件 | path_selector |
| dit | path | DiT 模型路径 | path_selector |
| vae | path | VAE 模型路径 | path_selector |
| seed | number | 随机种子 | number_input |
| output_name | text | 输出模型名称 | text_input |
| output_dir | path | 输出目录 | path_selector |
| logging_dir | path | 日志目录 | path_selector |

### 架构特定参数

#### FLUX.2
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| model_version | select | 模型版本 | select [dev/klein-4b/klein-base-4b/klein-9b/klein-base-9b] |
| text_encoder | path | Text Encoder 路径 | path_selector |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| task | select | 任务类型 | select |
| t5 | path | T5 模型路径 | path_selector |
| clip | path | CLIP 模型路径 | path_selector |
| fp8_t5 | checkbox | FP8 T5 | checkbox |
| vae_cache_cpu | checkbox | VAE 缓存到 CPU | checkbox |
| preserve_distribution_shape | checkbox | 保持分布形状 | checkbox |

#### HunyuanVideo
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder1 | path | Text Encoder 1 | path_selector |
| text_encoder2 | path | Text Encoder 2 | path_selector |
| fp8_llm | checkbox | FP8 LLM | checkbox |
| vae_tiling | checkbox | VAE Tiling | checkbox |
| vae_chunk_size | number | VAE Chunk Size | number_input |
| vae_spatial_tile_sample_min_size | number | 最小 Tile 大小 | number_input |

#### FramePack
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| image_encoder | path | Image Encoder 路径 | path_selector |
| latent_window_size | number | Latent 窗口大小 | number_input |
| bulk_decode | checkbox | 批量解码 | checkbox |
| vanilla_sampling | checkbox | Vanilla 采样 | checkbox |
| f1 | checkbox | F1 模式 | checkbox |

### 训练配置参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| max_train_steps | number | 最大训练步数 | number_input |
| max_train_epochs | number | 最大训练轮数 | number_input |
| gradient_accumulation_steps | number | 梯度累加步数 | number_input |
| guidance_scale | number | Guidance Scale | number_input |

### 时间步采样参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| timestep_sampling | select | 时间步采样方法 | select [sigma/uniform/sigmoid/shift/flux2_shift] |
| discrete_flow_shift | number | Flow Shift | number_input |
| sigmoid_scale | number | Sigmoid 缩放 | number_input |
| weighting_scheme | select | 加权方案 | select |
| logit_mean | number | Logit 均值 | number_input |
| logit_std | number | Logit 标准差 | number_input |
| mode_scale | number | Mode 缩放 | number_input |
| min_timestep | number | 最小时序 | number_input |
| max_timestep | number | 最大时间步 | number_input |
| show_timesteps | select | 显示时序 | select [console/image] |

### SOAR 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| soar | checkbox | 启用 SOAR 辅助训练 | checkbox |
| soar_lambda_aux | number | SOAR 辅助损失权重 | editable_slider |
| soar_trajectory_length | number | SOAR 轨迹点数量 | editable_slider |
| soar_num_sampling_steps | number | SOAR 采样步数 | editable_slider |
| soar_sigma_upper_ratio | number | SOAR sigma 上界倍率 | editable_slider |
| soar_cfg_scale_sampling | number | SOAR rollout CFG scale | editable_slider |

### D-OPSD 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| dopsd | checkbox | 启用 D-OPSD distillation；支持 Z-Image LoRA/微调、FLUX.2 Klein LoRA | checkbox |
| dopsd_loss_weight | number | D-OPSD distillation loss 权重 | editable_slider |
| dopsd_num_sampling_steps | number | D-OPSD few-step rollout 采样步数 | editable_slider |
| dopsd_ema_decay | number | teacher adapter EMA 衰减 | editable_slider |

### 学习率参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| learning_rate | text | 学习率 | text_input |
| lr_scheduler | select | 学习率调度器 | select |
| lr_warmup_steps | number | Warmup 步数 | number_input |
| lr_decay_steps | number | 衰减步数 | number_input |
| lr_scheduler_num_cycles | number | 重启次数 | number_input |
| lr_scheduler_power | number | 多项式 Power | number_input |
| lr_scheduler_timescale | number | 时间缩放 | number_input |
| lr_scheduler_min_lr_ratio | number | 最小 LR 比率 | number_input |

### 网络结构参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| network_module | text | 网络模块 | text_input (固定值) |
| network_dim | number | Network Dim | number_input |
| network_alpha | number | Network Alpha | number_input |
| network_dropout | number | Dropout | number_input |
| dim_from_weights | checkbox | 从权重获取 Dim | checkbox |
| scale_weight_norms | number | Scale Weight Norms | number_input |

### 内存优化参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| gradient_checkpointing | checkbox | 梯度检查点 | checkbox |
| gradient_checkpointing_cpu_offload | checkbox | CPU Offload | checkbox |
| fp8_base | checkbox | FP8 基础模式 | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| blocks_to_swap | number | Block Swap 数量 | number_input |
| use_pinned_memory_for_block_swap | checkbox | Pinned Memory | checkbox |
| block_swap_h2d_only | checkbox | H2D-only Block Swap（仅 LoRA / 冻结基座训练） | checkbox |
| block_swap_ring_size | number | H2D-only GPU ring buffer 数量 | editable_slider |
| img_in_txt_in_offloading | checkbox | Img/Txt Offloading | checkbox |

### 注意力参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| attn_mode | select | 注意力模式 | select [flash/xformers/sdpa/sageattn/torch] |
| split_attn | checkbox | Split Attention | checkbox |

### 精度参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| mixed_precision | select | 混合精度 | select [bf16/fp16/no] |
| full_bf16 | checkbox | Full BF16 | checkbox |
| vae_dtype | select | VAE 数据类型 | select |
| dit_dtype | select | DiT 数据类型 | select |

### Torch Compile 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| compile | checkbox | 启用 Compile | checkbox |
| compile_backend | select | 编译后端 | select [inductor/eager/aot_eager] |
| compile_mode | select | 编译模式 | select |
| compile_fullgraph | checkbox | Full Graph | checkbox |
| compile_dynamic | select | Dynamic | select [auto/true/false] |
| compile_cache_size_limit | number | 缓存大小限制 | number_input |

### TF32 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| cuda_allow_tf32 | checkbox | 允许 TF32 | checkbox |
| cuda_cudnn_benchmark | checkbox | CuDNN Benchmark | checkbox |

### 数据加载参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| max_data_loader_n_workers | number | 数据加载线程数 | number_input |
| persistent_data_loader_workers | checkbox | 持久化 Workers | checkbox |

### 优化器参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| optimizer_type | select | 优化器类型 | select |
| max_grad_norm | number | 最大梯度范数 | number_input |

### 保存参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| save_every_n_epochs | number | 每 N 轮保存 | number_input |
| save_every_n_steps | number | 每 N 步保存 | number_input |
| save_last_n_epochs | number | 保存最后 N 轮 | number_input |
| save_last_n_steps | number | 保存最后 N 步 | number_input |

### 训练状态参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| save_state | checkbox | 保存训练状态 | checkbox |
| save_state_on_train_end | checkbox | 只在训练结束保存 | checkbox |
| save_last_n_epochs_state | number | 保存最后 N 轮状态 | number_input |
| save_last_n_steps_state | number | 保存最后 N 步状态 | number_input |
| resume | path | 恢复训练路径 | path_selector |

### LoRA+ 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_lora_plus | checkbox | 启用 LoRA+ | checkbox |
| loraplus_lr_ratio | number | LoRA+ LR 比率 | number_input |

### 目标 Block 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_blocks | checkbox | 启用 Block 选择 | checkbox |
| exclude_patterns | text | 排除模式 | text_input |
| include_patterns | text | 包含模式 | text_input |

### LyCORIS 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_lycoris | checkbox | 启用 LyCORIS | checkbox |
| conv_dim | number | 卷积 Dim | number_input |
| conv_alpha | number | 卷积 Alpha | number_input |
| algo | select | 算法 | select [lora/loha/ia3/lokr/dylora/diag-oft/full] |
| dropout | number | Dropout | number_input |
| preset | select | 预设 | select |
| factor | number | 因子 | number_input |
| decompose_both | checkbox | Decompose Both | checkbox |
| block_size | number | Block Size | number_input |
| use_tucker | checkbox | Use Tucker | checkbox |
| use_scalar | checkbox | Use Scalar | checkbox |
| train_norm | checkbox | Train Norm | checkbox |
| dora_wd | checkbox | DoRA WD | checkbox |
| full_matrix | checkbox | Full Matrix | checkbox |
| bypass_mode | checkbox | Bypass Mode | checkbox |
| rescaled | number | Rescaled | number_input |
| constrain | number | Constrain | number_input |

### 采样参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| enable_sample | checkbox | 启用采样 | checkbox |
| sample_at_first | checkbox | 训练开始时采样 | checkbox |
| sample_prompts | path | 采样提示词文件 | path_selector |
| sample_every_n_epochs | number | 每 N 轮采样 | number_input |
| sample_every_n_steps | number | 每 N 步采样 | number_input |

### 元数据参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| training_comment | text | 训练注释 | text_input |
| metadata_title | text | 元数据标题 | text_input |
| metadata_author | text | 元数据作者 | text_input |
| metadata_description | text | 元数据描述 | text_input |
| metadata_license | text | 元数据许可证 | text_input |
| metadata_tags | text | 元数据标签 | text_input |

### HuggingFace 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| async_upload | checkbox | 异步上传 | checkbox |
| huggingface_repo_id | text | Repo ID | text_input |
| huggingface_repo_type | text | Repo Type | text_input |
| huggingface_path_in_repo | text | Path in Repo | text_input |
| huggingface_token | text | Token | text_input |
| huggingface_repo_visibility | select | 可见性 | select [public/private] |
| save_state_to_huggingface | checkbox | 保存状态到 HF | checkbox |
| resume_from_huggingface | checkbox | 从 HF 恢复 | checkbox |

### DDP 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| multi_gpu | checkbox | 多 GPU | checkbox |
| ddp_timeout | number | DDP 超时 | number_input |
| ddp_gradient_as_bucket_view | checkbox | Gradient Bucket View | checkbox |
| ddp_static_graph | checkbox | Static Graph | checkbox |

### 权重参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| network_weights | path | LoRA 预训练权重 | path_selector |
| base_weights | text | 基础权重路径 | text_input |
| base_weights_multiplier | text | 基础权重倍率 | text_input |

### WandB 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| wandb_api_key | text | WandB API Key | text_input |

## 推理脚本参数 (step4_generate)

### 基础参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| dit | path | DiT 模型路径 | path_selector |
| vae | path | VAE 模型路径 | path_selector |
| save_path | path | 保存路径 | path_selector |
| seed | number | 随机种子 | number_input |

### 架构特定参数

#### FLUX.2
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| model_version | select | 模型版本 | select |
| text_encoder | path | Text Encoder 路径 | path_selector |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |
| control_image_path | path | 参考图像路径 | path_selector |
| no_resize_control | checkbox | 不调整控制图像大小 | checkbox |

#### Wan
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| task | select | 任务类型 | select |
| t5 | path | T5 模型路径 | path_selector |
| clip | path | CLIP 模型路径 | path_selector |
| fp8_t5 | checkbox | FP8 T5 | checkbox |
| negative_prompt | text | 负向提示词 | text_input |
| guidance_scale | number | Guidance Scale | number_input |
| vae_cache_cpu | checkbox | VAE 缓存到 CPU | checkbox |
| trim_tail_frames | number | 裁剪尾部帧数 | number_input |
| cpu_noise | checkbox | CPU 噪声 | checkbox |
| cfg_skip_mode | select | CFG Skip 模式 | select |
| cfg_apply_ratio | number | CFG 应用比率 | number_input |
| slg_layers | text | SLG 层 | text_input |
| slg_scale | number | SLG Scale | number_input |
| slg_start | number | SLG 开始 | number_input |
| slg_end | number | SLG 结束 | number_input |
| slg_mode | select | SLG 模式 | select |
| offload_inactive_dit | checkbox | 卸载非活跃 DiT | checkbox |
| timestep_boundary | text | 时序边界 | text_input |
| image_path | path | 图像路径 | path_selector |
| end_image_path | path | 结束图像路径 | path_selector |
| control_path | path | 控制视频路径 | path_selector |
| dit_high_noise | path | 高噪声 DiT | path_selector |

#### HunyuanVideo
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| text_encoder1 | path | Text Encoder 1 | path_selector |
| text_encoder2 | path | Text Encoder 2 | path_selector |
| fp8_llm | checkbox | FP8 LLM | checkbox |
| fp8_fast | checkbox | FP8 Fast | checkbox |
| vae_chunk_size | number | VAE Chunk Size | number_input |
| vae_spatial_tile_sample_min_size | number | 最小 Tile 大小 | number_input |
| split_attn | checkbox | Split Attention | checkbox |
| embedded_cfg_scale | number | Embedded CFG Scale | number_input |
| img_in_txt_in_offloading | checkbox | Img/Txt Offloading | checkbox |
| image_path | path | 图像路径 | path_selector |
| end_image_path | path | 结束图像路径 | path_selector |
| control_image_path | path | 控制图像路径 | path_selector |

#### FramePack
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| image_encoder | path | Image Encoder 路径 | path_selector |
| latent_window_size | number | Latent 窗口大小 | number_input |
| bulk_decode | checkbox | 批量解码 | checkbox |
| video_seconds | number | 视频秒数 | number_input |
| video_sections | text | 视频段落 | text_input |
| f1 | checkbox | F1 模式 | checkbox |
| one_frame_inference | text | 单帧推理 | text_input |
| image_mask_path | path | 图像掩码路径 | path_selector |
| end_image_mask_path | path | 结束图像掩码路径 | path_selector |
| magcache_calibration | checkbox | MegaCache 校准 | checkbox |
| enable_megacache | checkbox | 启用 MegaCache | checkbox |
| magcache_mag_ratios | text | MegaCache Mag Ratios | text_input |
| magcache_retention_ratio | number | MegaCache 保留比率 | number_input |
| magcache_threshold | number | MegaCache 阈值 | number_input |
| magcache_k | number | MegaCache K | number_input |

### 生成参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| prompt | text | 提示词 | textarea |
| negative_prompt | text | 负向提示词 | textarea |
| from_file | path | 提示词文件 | path_selector |
| image_size | text | 图像尺寸 | text_input |
| video_size | text | 视频尺寸 | text_input |
| video_length | number | 视频长度 | number_input |
| fps | number | FPS | number_input |
| infer_steps | number | 推理步数 | number_input |
| guidance_scale | number | Guidance Scale | number_input |
| embedded_cfg_scale | number | Embedded CFG Scale | number_input |
| flow_shift | number | Flow Shift | number_input |

### 内存优化参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| fp8 | checkbox | FP8 | checkbox |
| fp8_scaled | checkbox | FP8 Scaled | checkbox |
| fp8_text_encoder | checkbox | FP8 Text Encoder | checkbox |
| device | select | 设备 | select |
| attn_mode | select | 注意力模式 | select |
| blocks_to_swap | number | Block Swap 数量 | number_input |
| use_pinned_memory_for_block_swap | checkbox | Pinned Memory | checkbox |

### 输出参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| output_type | select | 输出类型 | select [images/video/both/latent] |
| no_metadata | checkbox | 不包含元数据 | checkbox |
| latent_path | path | Latent 路径 | path_selector |

### LoRA 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| lora_weight | text | LoRA 权重路径 | textarea |
| lora_multiplier | text | LoRA 倍率 | text_input |
| lora_weight_high_noise | text | 高噪声 LoRA 权重 | textarea |
| lora_multiplier_high_noise | text | 高噪声 LoRA 倍率 | text_input |
| include_patterns | text | 包含模式 | text_input |
| exclude_patterns | text | 排除模式 | text_input |
| save_merged_model | checkbox | 保存合并模型 | checkbox |
| lycoris | checkbox | LyCORIS | checkbox |

### Torch Compile 参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| compile | checkbox | 启用 Compile | checkbox |
| compile_backend | select | 编译后端 | select |
| compile_mode | select | 编译模式 | select |
| compile_fullgraph | checkbox | Full Graph | checkbox |
| compile_dynamic | select | Dynamic | select |
| compile_cache_size_limit | number | 缓存大小限制 | number_input |

### 采样器参数
| 参数名 | 类型 | 说明 | GUI 组件 |
|--------|------|------|----------|
| sample_solver | select | 采样求解器 | select [unipc/dpm++/vanilla] |
