# Cache script by @bdsqlsz

# Cache mode(FLUX.2)
$cache_mode = "flux2" # Cache mode | 缓存模式

# Cache lantent
$dataset_config = "./toml/qinglong-flux2-datasets.toml"               # path to dataset config .toml file | 数据集配置文件路径
$vae = "./ckpts/vae/ae.safetensors"                                   # VAE (AutoEncoder) directory | VAE路径
$vae_dtype = ""                                                       # fp16 | fp32 | bf16, default: fp32
$device = ""                                                          # cuda | cpu
$batch_size = ""                                                      # batch size, override dataset config if dataset batch size > this
$num_workers = 0                                                      # number of workers for dataset. default is cpu count-1
$skip_existing = $False                                               # skip existing cache files
$debug_mode = ""                                                      # image | console
$console_width = $Host.UI.RawUI.WindowSize.Width                      # console width
$console_back = "black"                                               # console background color
$console_num_images = 16                                              # number of images to show in console

# Cache text encoder
$text_encoder_batch_size = "16"                                       # batch size
$text_encoder_device = ""                                             # cuda | cpu
$text_encoder_num_workers = 0                                         # number of workers for dataset. default is cpu count-1
$text_encoder_skip_existing = $False                                  # skip existing cache files

# FLUX.2 Model
$model_version = "dev"                                                # model version: dev | klein-4b | klein-base-4b | klein-9b | klein-base-9b
$text_encoder = "./ckpts/text_encoder/mistral3_model.safetensors"     # Text Encoder (Mistral 3 or Qwen 3) checkpoint path
$fp8_text_encoder = $False                                            # use fp8 for Text Encoder model

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

# Model version
[void]$ext_args.Add("--model_version=$model_version")
[void]$ext2_args.Add("--model_version=$model_version")

# VAE arguments
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

if ($fp8_text_encoder) {
    [void]$ext2_args.Add("--fp8_text_encoder")
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
python -m accelerate.commands.launch "./musubi-tuner/flux_2_cache_latents.py" `
    --dataset_config=$dataset_config `
    --vae=$vae $ext_args

python -m accelerate.commands.launch "./musubi-tuner/flux_2_cache_text_encoder_outputs.py" `
    --dataset_config=$dataset_config $ext2_args

Write-Output "Cache finished"
Read-Host | Out-Null ;
