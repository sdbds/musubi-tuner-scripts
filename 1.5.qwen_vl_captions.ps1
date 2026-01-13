# Caption script by @bdsqlsz

# Qwen2.5-VL Caption parameters | 图像打标参数（Qwen2.5-VL）
# Refer to: musubi-tuner/docs/tools.md -> Image Captioning with Qwen2.5-VL
$image_dir = "./train/image"                                # directory of images to caption | 需要打标的图片目录（必填）
$model_path = "./ckpts/text_encoder/qwen_2.5_vl_7b.safetensors" # Qwen2.5-VL model path | 模型路径（必填）
$output_format = "text"                                        # jsonl | text
$output_file = "./train/caption.json"                            # required if output_format == jsonl | 当 output_format 为 jsonl 时必填
$max_new_tokens = 1024                                           # max tokens per caption | 单条caption最大生成token数
$max_size = 1280                                                 # image max size used by VLM | 输入给VLM的最大边长
$prompt = ""                                                    # custom prompt; empty to use script default | 自定义提示词；留空使用脚本默认
$fp8_vl = $false                                                 # use fp8 for Qwen2.5-VL | 是否使用fp8精度

# Strict error and concise progress
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

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

# Build caption arguments
$caption_args = [System.Collections.ArrayList]::new()
[void]$caption_args.Add("--image_dir=$image_dir")
[void]$caption_args.Add("--model_path=$model_path")
if ($output_format) { [void]$caption_args.Add("--output_format=$output_format") }
if ($output_format -ieq "jsonl" -and $output_file) {
    $outDir = Split-Path -Parent $output_file
    if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }
    [void]$caption_args.Add("--output_file=$output_file")
}
if ($max_new_tokens -ne 1024) { [void]$caption_args.Add("--max_new_tokens=$max_new_tokens") }
if ($max_size -ne 1280) { [void]$caption_args.Add("--max_size=$max_size") }
if ($prompt) { [void]$caption_args.Add("--prompt=$prompt") }
if ($fp8_vl) { [void]$caption_args.Add("--fp8_vl") }


# run Caption
python -m accelerate.commands.launch "./musubi-tuner/src/musubi_tuner/caption_images_by_qwen_vl.py" `
    $caption_args

Write-Output "Caption finished"
Read-Host | Out-Null ;
