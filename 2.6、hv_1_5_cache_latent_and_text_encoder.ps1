# Cache script by @bdsqlsz

# Cache mode(HunyuanVideo_1_5)
$cache_mode = "HunyuanVideo_1_5" # Cache mode | 缓存模式

# Cache lantent
$dataset_config = "./toml/qinglong-hv-1-5-datasets.toml"               # path to dataset config .toml file | 数据集配置文件路径
$vae = "./ckpts/hunyuan-video-1.5/vae/pytorch_model.pt"                # VAE directory | VAE路径
$vae_dtype = ""                                                        # fp16 | fp32 |bf16 default: fp16
$device = ""                                                           # cuda | cpu
$batch_size = ""                                                       # batch size, override dataset config if dataset batch size > this
$num_workers = 0                                                       # number of workers for dataset. default is cpu count-1
$skip_existing = $False                                                # skip existing cache files
$debug_mode = ""                                                       # image | console
$console_width = $Host.UI.RawUI.WindowSize.Width                       # console width
$console_back = "black"                                                # console background color
$console_num_images = 16                                               # number of images to show in console

# HunyuanVideo 1.5 VAE specific
$vae_sample_size = 128                                                 # VAE sample size (height/width). Default 128; set 256 if VRAM is sufficient for better quality; set 0 to disable tiling.
$vae_enable_patch_conv = $False                                        # Enable patch-based convolution in VAE for memory optimization

# I2V specific
$i2v = $False                                                          # Cache image features and conditional latents for I2V training/inference
$image_encoder = "./ckpts/hunyuan-video-1.5/sigclip_vision_patch14_384.safetensors" # SigLIP Image Encoder (required if --i2v is set)

# Cache text encoder
$text_encoder_batch_size = "16"                                        # batch size
$text_encoder_device = ""                                              # cuda | cpu
$text_encoder_num_workers = 0                                          # number of workers for dataset. default is cpu count-1
$text_encoder_skip_existing = $False                                   # skip existing cache files

# HunyuanVideo 1.5 Text Encoder
$text_encoder = "./ckpts/text_encoder/qwen_2.5_vl_7b.safetensors"      # Qwen2.5-VL text encoder checkpoint path
$byt5 = "./ckpts/text_encoder/byt5_model.safetensors"                   # BYT5 text encoder checkpoint path
$fp8_vl = $False                                                       # use fp8 for Qwen2.5-VL model

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
$ext2_args = [System.Collections.ArrayList]::new()

# VAE arguments
if ($vae_sample_size -ne 128) {
  [void]$ext_args.Add("--vae_sample_size=$vae_sample_size")
}
if ($vae_enable_patch_conv) {
  [void]$ext_args.Add("--vae_enable_patch_conv")
}

# I2V arguments
if ($i2v) {
  [void]$ext_args.Add("--i2v")
  [void]$ext_args.Add("--image_encoder=$image_encoder")
}

if ($vae_dtype) {
  [void]$ext_args.Add("--vae_dtype=$vae_dtype")
}

if ($device) {
  [void]$ext_args.Add("--device=$device")
}

if ($batch_size) {
  [void]$ext_args.Add("--batch_size=$batch_size")
}

if ($num_workers -ne 0) {
  [void]$ext_args.Add("--num_workers=$num_workers")
}

if ($skip_existing) {
  [void]$ext_args.Add("--skip_existing")
}

if ($debug_mode) {
  [void]$ext_args.Add("--debug_mode=$debug_mode")
  if ($debug_mode -eq "console") {
    if ($console_width) {
      [void]$ext_args.Add("--console_width=$console_width")
    }
    
    if ($console_back) {
      [void]$ext_args.Add("--console_back=$console_back")
    }
    
    if ($console_num_images) {
      [void]$ext_args.Add("--console_num_images=$console_num_images")
    }
  }
}

# Text encoder arguments
[void]$ext2_args.Add("--text_encoder=$text_encoder")
[void]$ext2_args.Add("--byt5=$byt5")

if ($fp8_vl) {
  [void]$ext2_args.Add("--fp8_vl")
}

if ($text_encoder_batch_size) {
  [void]$ext2_args.Add("--batch_size=$text_encoder_batch_size")
}

if ($text_encoder_device) {
  [void]$ext2_args.Add("--device=$text_encoder_device")
}

if ($text_encoder_num_workers -ne 0) {
  [void]$ext2_args.Add("--num_workers=$text_encoder_num_workers")
}

if ($text_encoder_skip_existing) {
  [void]$ext2_args.Add("--skip_existing")
}

# run Cache
python -m accelerate.commands.launch "./musubi-tuner/hv_1_5_cache_latents.py" `
  --dataset_config=$dataset_config `
  --vae=$vae $ext_args

python -m accelerate.commands.launch "./musubi-tuner/hv_1_5_cache_text_encoder_outputs.py" `
  --dataset_config=$dataset_config $ext2_args

Write-Output "Cache finished"
Read-Host | Out-Null ;
