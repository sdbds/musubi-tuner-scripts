﻿# Install script by @bdsqlsz

Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
#$Env:PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
#$Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = 1
$Env:UV_NO_CACHE = 0
$Env:UV_LINK_MODE = "symlink"
$Env:GIT_LFS_SKIP_SMUDGE = 1
$Env:CUDA_HOME = "${env:CUDA_PATH}"

function InstallFail {
    Write-Output "Install failed|安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

try {
    ~/.local/bin/uv --version
    Write-Output "uv installed|UV模块已安装."
}
catch {
    Write-Output "Installing uv|安装uv模块中..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        Check "uv install failed|安装uv模块失败。"
    }
    else {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        Check "uv install failed|安装uv模块失败。"
    }
}

if ($env:OS -ilike "*windows*") {
    #chcp 65001
    # First check if UV cache directory already exists
    if (Test-Path -Path "${env:LOCALAPPDATA}/uv/cache") {
        Write-Host "UV cache directory already exists, skipping disk space check"
    }
    else {
        # Check C drive free space with error handling
        try {
            $CDrive = Get-WmiObject Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction Stop
            if ($CDrive) {
                $FreeSpaceGB = [math]::Round($CDrive.FreeSpace / 1GB, 2)
                Write-Host "C: drive free space: ${FreeSpaceGB}GB"
                
                # $Env:UV cache directory based on available space
                if ($FreeSpaceGB -lt 20) {
                    Write-Host "Low disk space detected. Using local .cache directory"
                    $Env:UV_CACHE_DIR = ".cache"
                } 
            }
            else {
                Write-Warning "C: drive not found. Using local .cache directory"
                $Env:UV_CACHE_DIR = ".cache"
            }
        }
        catch {
            Write-Warning "Failed to check disk space: $_. Using local .cache directory"
            $Env:UV_CACHE_DIR = ".cache"
        }
    }
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        . ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        . ./.venv/Scripts/activate
    }
    else {
        Write-Output "Create .venv"
        ~/.local/bin/uv venv -p 3.10
        . ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    . ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    . ./.venv/bin/activate.ps1
}
else {
    Write-Output "Create .venv"
    ~/.local/bin/uv venv -p 3.10
    . ./.venv/bin/activate.ps1
}

Write-Output "Installing main requirements"

~/.local/bin/uv pip install --upgrade setuptools wheel

if ($env:OS -ilike "*windows*") {
    ~/.local/bin/uv pip sync requirements-uv-windows.txt --index-strategy unsafe-best-match
    Check "Install main requirements failed"
}
else {
    ~/.local/bin/uv pip sync requirements-uv-linux.txt --index-strategy unsafe-best-match
    Check "Install main requirements failed"
}

~/.local/bin/uv pip install -U --pre lycoris_lora torch==2.5.1+cu124

~/.local/bin/uv pip install -U typing-extensions --index-strategy unsafe-best-match

$download_hy = Read-Host "是否下载HunyuanVideo模型? 若需要下载模型选择 y，若不需要选择 n。[y/n] (默认为 n)`nDo you want to download the HunyuanVideo model? Choose y to download, n to skip. [y/n] (default is n)"
if ($download_hy -eq "y" -or $download_hy -eq "Y") {
    huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts --exclude "*mp_rank_00_model_states_fp8*"

    if (-not (Test-Path "./ckpts/text_encoder/llava_llama3_fp16.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/llava_llama3_fp16.safetensors --local-dir ./ckpts/text_encoder

        Move-Item -Path ./ckpts/text_encoder/split_files/text_encoders/llava_llama3_fp16.safetensors -Destination ./ckpts/text_encoder/llava_llama3_fp16.safetensors
    }

    if (-not (Test-Path "./ckpts/text_encoder_2/clip_l.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/clip_l.safetensors --local-dir ./ckpts/text_encoder_2

        Move-Item -Path ./ckpts/text_encoder_2/split_files/text_encoders/clip_l.safetensors -Destination ./ckpts/text_encoder_2/clip_l.safetensors
    }
}

$download_wan = Read-Host "请选择要下载的Wan模型 [1/2/3/4/n] (默认为 n)
1: 下载 T2V-1.3B 模型
2: 下载 T2V-14B 模型
3: 下载 I2V-480P 模型
4: 下载 I2V-720P 模型
n: 不下载
Please select which Wan model to download [1/2/3/4/n] (default is n)
1: Download T2V-1.3B model
2: Download T2V-14B model
3: Download I2V-480P model
4: Download I2V-720P model
n: Skip download"

if ($download_wan -eq "1") {
    Write-Output "正在下载 Wan T2V-1.3B 模型 / Downloading Wan T2V-1.3B model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --local-dir ./ckpts/wan

} elseif ($download_wan -eq "2") {
    Write-Output "正在下载 Wan T2V-14B 模型 / Downloading Wan T2V-14B model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors --local-dir ./ckpts/wan

} elseif ($download_wan -eq "3") {
    Write-Output "正在下载 Wan I2V-480P 模型 / Downloading Wan I2V-480P model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors --local-dir ./ckpts/wan

} elseif ($download_wan -eq "4") {
    Write-Output "正在下载 Wan I2V-720P 模型 / Downloading Wan I2V-720P model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged  split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors --local-dir ./ckpts/wan
}

if ($download_wan -in @("1", "2", "3", "4")) {
    if (-not (Test-Path "./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth")) {
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth --local-dir ./ckpts/text_encoder
        #Move-Item -Path ./ckpts/text_encoder/split_files/text_encoders/llava_llama3_fp16.safetensors -Destination ./ckpts/text_encoder/llava_llama3_fp16.safetensors
    }

    if (-not (Test-Path "./ckpts/vae/Wan2.1_VAE.pth")) {
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B Wan2.1_VAE.pth --local-dir ./ckpts/vae
        #Move-Item -Path ./ckpts/vae/split_files/text_encoders/clip_l.safetensors -Destination ./ckpts/vae/clip_l.safetensors
    }
}

Write-Output "Install finished"
Read-Host | Out-Null ;
