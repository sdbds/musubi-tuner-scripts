# Install script by @bdsqlsz

Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
#$Env:PIP_INDEX_URL="https://pypi.mirrors.ustc.edu.cn/simple"
#$Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
$Env:UV_CACHE_DIR = "${env:LOCALAPPDATA}/uv/cache"
$Env:UV_NO_BUILD_ISOLATION = 1
$Env:UV_NO_CACHE = 0
$Env:UV_LINK_MODE = "symlink"
$Env:GIT_LFS_SKIP_SMUDGE = 1
$Env:CUDA_HOME = "${env:CUDA_PATH}"
$Env:HF_HUB_ENABLE_HF_TRANSFER = 0

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
        ~/.local/bin/uv venv -p 3.11 --seed
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
    ~/.local/bin/uv venv -p 3.11 --seed
    . ./.venv/bin/activate.ps1
}

Write-Output "Installing main requirements"

~/.local/bin/uv pip install -U hatchling editables torch==2.9.0

if ($env:OS -ilike "*windows*") {
    ~/.local/bin/uv pip sync requirements-uv-windows.txt --index-strategy unsafe-best-match
    Check "Install main requirements failed"
}
else {
    ~/.local/bin/uv pip sync requirements-uv-linux.txt --index-strategy unsafe-best-match
    Check "Install main requirements failed"
}

~/.local/bin/uv pip install git+https://github.com/sdbds/LyCORIS@dev torch==2.9.0

~/.local/bin/uv pip install -U typing-extensions --index-strategy unsafe-best-match

$download_hy = Read-Host "请选择要下载的HunyuanVideo模型 [1/2/n] (默认为 n)
1: 下载 T2V 模型
2: 下载 I2V 模型
n: 不下载
Please select which HunyuanVideo model to download [1/2/n] (default is n)
1: Download T2V model
2: Download I2V model
n: Skip download"
if ($download_hy -eq "1") {
    Write-Output "正在下载 HunyuanVideo T2V 模型 / Downloading HunyuanVideo T2V model..."
    huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts --exclude "*mp_rank_00_model_states_fp8*"
}
elseif ($download_hy -eq "2") {
    Write-Output "正在下载 HunyuanVideo I2V 模型 / Downloading HunyuanVideo I2V model..."
    huggingface-cli download tencent/HunyuanVideo-I2V --local-dir ./ckpts --include "*hunyuan-video-i2v-720p*"
}

if ($download_hy -in @("1", "2")) {
    if (-not (Test-Path "./ckpts/text_encoder/llava_llama3_fp16.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/llava_llama3_fp16.safetensors --local-dir ./ckpts/text_encoder

        Move-Item -Path ./ckpts/text_encoder/split_files/text_encoders/llava_llama3_fp16.safetensors -Destination ./ckpts/text_encoder/llava_llama3_fp16.safetensors
    }

    if (-not (Test-Path "./ckpts/text_encoder_2/clip_l.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/clip_l.safetensors --local-dir ./ckpts/text_encoder_2

        Move-Item -Path ./ckpts/text_encoder_2/split_files/text_encoders/clip_l.safetensors -Destination ./ckpts/text_encoder_2/clip_l.safetensors
    }
}

$download_wan = Read-Host "请选择要下载的Wan2.1模型 [1/2/3/4/5/6/n] (默认为 n)
1: 下载 T2V-1.3B 模型
2: 下载 T2V-14B 模型
3: 下载 I2V-480P 模型
4: 下载 I2V-720P 模型
5: 下载 1.3B-FC 模型
6: 下载 14B-FC 模型
n: 不下载
Please select which Wan2.1 model to download [1/2/3/4/5/6/n] (default is n)
1: Download T2V-1.3B model
2: Download T2V-14B model
3: Download I2V-480P model
4: Download I2V-720P model
5: Download 1.3B-FC model
6: Download 14B-FC model
n: Skip download"

if ($download_wan -eq "1") {
    Write-Output "正在下载 Wan T2V-1.3B 模型 / Downloading Wan T2V-1.3B model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors --local-dir ./ckpts/wan

}
elseif ($download_wan -eq "2") {
    Write-Output "正在下载 Wan T2V-14B 模型 / Downloading Wan T2V-14B model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_14B_fp16.safetensors --local-dir ./ckpts/wan

}
elseif ($download_wan -eq "3") {
    Write-Output "正在下载 Wan I2V-480P 模型 / Downloading Wan I2V-480P model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_i2v_480p_14B_fp16.safetensors --local-dir ./ckpts/wan

}
elseif ($download_wan -eq "4") {
    Write-Output "正在下载 Wan I2V-720P 模型 / Downloading Wan I2V-720P model..."
    huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged  split_files/diffusion_models/wan2.1_i2v_720p_14B_fp16.safetensors --local-dir ./ckpts/wan
}
elseif ($download_wan -eq "5") {
    Write-Output "正在下载 Wan T2V-1.3B-FC 模型 / Downloading Wan T2V-1.3B-FC model..."
    huggingface-cli download alibaba-pai/Wan2.1-Fun-1.3B-Control diffusion_pytorch_model.safetensors --local-dir ./ckpts/wan-1.3B-FC
}
elseif ($download_wan -eq "6") {
    Write-Output "正在下载 Wan T2V-14B-FC 模型 / Downloading Wan T2V-14B-FC model..."
    huggingface-cli download alibaba-pai/Wan2.1-Fun-14B-Control diffusion_pytorch_model.safetensors --local-dir ./ckpts/wan-14B-FC
}

if ($download_wan -in @("1", "2", "3", "4", "5", "6")) {
    if ($download_wan -in @("3", "4", "5", "6")) {
        if (-not (Test-Path "./ckpts/text_encoder_2/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")) {
            huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./ckpts/text_encoder_2
        }
    }

    if (-not (Test-Path "./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth")) {
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth --local-dir ./ckpts/text_encoder
        #Move-Item -Path ./ckpts/text_encoder/split_files/text_encoders/llava_llama3_fp16.safetensors -Destination ./ckpts/text_encoder/llava_llama3_fp16.safetensors
    }

    if (-not (Test-Path "./ckpts/vae/Wan2.1_VAE.pth")) {
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B Wan2.1_VAE.pth --local-dir ./ckpts/vae
        #Move-Item -Path ./ckpts/vae/split_files/text_encoders/clip_l.safetensors -Destination ./ckpts/vae/clip_l.safetensors
    }
}

$download_fp = Read-Host "请选择要下载的FramePack模型 [1/2/n] (默认为 n)
Please select which FramePack model to download [1/2/n] (default is n)
1: 下载 FramePack 模型
2: 下载 FramePack F1 模型
n: 不下载
Please select which HunyuanVideo model to download [1/2/n] (default is n)
1: Download FramePack model
2: Download FramePack F1 model
n: Skip download"
if ($download_fp -eq "1") {
    Write-Output "正在下载 FramePack 模型 / Downloading FramePack model..."
    huggingface-cli download Kijai/HunyuanVideo_comfy FramePackI2V_HY_bf16.safetensors --local-dir ./ckpts/framepack
}
elseif ($download_fp -eq "2") {
    Write-Output "正在下载 FramePack F1 模型 / Downloading FramePack F1 model..."
    huggingface-cli download kabachuha/FramePack_F1_I2V_HY_20250503_comfy FramePack_F1_I2V_HY_20250503.safetensors --local-dir ./ckpts/framepack
}

if ($download_fp -in @("1", "2")) {
    Write-Output "正在下载 hunyuan_video_vae_fp32 模型 / Downloading hunyuan_video_vae_fp32 model..."
    huggingface-cli download Kijai/HunyuanVideo_comfy hunyuan_video_vae_fp32.safetensors --local-dir ./ckpts/framepack

    Write-Output "正在下载 llava_llama3_fp16 模型 / Downloading llava_llama3_fp16 model..."
    if (-not (Test-Path "./ckpts/text_encoder/llava_llama3_fp16.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/llava_llama3_fp16.safetensors --local-dir ./ckpts/text_encoder

        Move-Item -Path ./ckpts/text_encoder/split_files/text_encoders/llava_llama3_fp16.safetensors -Destination ./ckpts/text_encoder/llava_llama3_fp16.safetensors
    }

    Write-Output "正在下载 clip_l 模型 / Downloading clip_l model..."
    if (-not (Test-Path "./ckpts/text_encoder_2/clip_l.safetensors")) {
        huggingface-cli download Comfy-Org/HunyuanVideo_repackaged split_files/text_encoders/clip_l.safetensors --local-dir ./ckpts/text_encoder_2

        Move-Item -Path ./ckpts/text_encoder_2/split_files/text_encoders/clip_l.safetensors -Destination ./ckpts/text_encoder_2/clip_l.safetensors
    }

    Write-Output "正在下载 sigclip_vision_patch14_384 模型 / Downloading sigclip_vision_patch14_384 model..."
    huggingface-cli download Comfy-Org/sigclip_vision_384 sigclip_vision_patch14_384.safetensors --local-dir ./ckpts/framepack
}

$download_wan = Read-Host "请选择要下载的Wan2.2模型 [1/2/3/4/5/6/n] (默认为 n)
1: 下载 T2V-14B-low-noise 模型
2: 下载 T2V-14B-Hight-noise 模型
3: 下载 I2V-14B-low-noise 模型
4: 下载 I2V-14B-Hight-noise 模型
5: 下载 TI2V-5B 模型
n: 不下载
Please select which Wan2.2 model to download [1/2/3/4/5/6/n] (default is n)
1: Download T2V-14B-low-noise model
2: Download T2V-14B-Hight-noise model
3: Download I2V-14B-low-noise model
4: Download I2V-14B-Hight-noise model
5: Download TI2V-5B model
n: Skip download"

if ($download_wan -eq "1") {
    Write-Output "正在下载 wan2.2_t2v_low_noise_14B 模型 / Downloading wan2.2_t2v_low_noise_14B model..."
    huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir ./ckpts/wan
}
elseif ($download_wan -eq "2") {
    Write-Output "正在下载 wan2.2_t2v_high_noise_14B 模型 / Downloading wan2.2_t2v_high_noise_14B model..."
    huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --local-dir ./ckpts/wan
}
elseif ($download_wan -eq "3") {
    Write-Output "正在下载 wan2.2_i2v_720p 模型 / Downloading wan2.2_i2v_720p model..."
    huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors --local-dir ./ckpts/wan
}
elseif ($download_wan -eq "4") {
    Write-Output "正在下载 wan2.2_i2v_high_noise 模型 / Downloading wan2.2_i2v_high_noise model..."
    huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_repackaged  split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors --local-dir ./ckpts/wan
}
elseif ($download_wan -eq "5") {
    Write-Output "正在下载 wan2.2_ti2v_5B 模型 / Downloading wan2.2_ti2v_5B model..."
    huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_repackaged split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors --local-dir ./ckpts/wan
}

if ($download_wan -in @("1", "2", "3", "4", "5")) {
    if ($download_wan -in @("3", "4", "5")) {
        if (-not (Test-Path "./ckpts/text_encoder_2/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")) {
            huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir ./ckpts/text_encoder_2
        }
    }

    if (-not (Test-Path "./ckpts/text_encoder/models_t5_umt5-xxl-enc-bf16.pth")) {
        huggingface-cli download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth --local-dir ./ckpts/text_encoder
    }

    if ($download_wan -in @("1", "2", "3", "4")) {
        if (-not (Test-Path "./ckpts/vae/Wan2.1_VAE.pth")) {
            huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir ./ckpts/vae
            Move-Item -Path ./ckpts/vae/split_files/vae/wan_2.1_vae.safetensors -Destination ./ckpts/vae/wan_2.1_vae.safetensors
        }
    }
    elseif ($download_wan -in @("5")) {
        if (-not (Test-Path "./ckpts/vae/Wan2.2_VAE.pth")) {
            huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/vae/wan2.2_vae.safetensors --local-dir ./ckpts/vae
            Move-Item -Path ./ckpts/vae/split_files/vae/wan2.2_vae.safetensors -Destination ./ckpts/vae/wan2.2_vae.safetensors
        }
    }
}

$download_hy = Read-Host "请选择要下载的qwen_image模型 [1/2/3/n] (默认为 n)
1: 下载 qwen_image 模型
2: 下载 qwen_image-edit 模型
3: 下载 qwen_image-edit-plus 模型
n: 不下载
Please select which qwen_image model to download [1/2/3/n] (default is n)
1: Download qwen_image model
2: Download qwen_image-edit model
3: Download qwen_image-edit-plus model
n: Skip download"
if ($download_hy -eq "1") {
    Write-Output "正在下载 qwen_image 模型 / Downloading qwen_image model..."
    if (-not (Test-Path "./ckpts/diffusion_models/qwen_image_bf16.safetensors")) {
        hf download Comfy-Org/Qwen-Image_ComfyUI split_files/diffusion_models/qwen_image_bf16.safetensors --local-dir ./ckpts
        Move-Item -Path ./ckpts/split_files/diffusion_models/qwen_image_bf16.safetensors -Destination ./ckpts/diffusion_models/qwen_image_bf16.safetensors
    }
}
elseif ($download_hy -eq "2") {
    Write-Output "正在下载 qwen_image-edit 模型 / Downloading qwen_image-edit model..."
    if (-not (Test-Path "./ckpts/diffusion_models/qwen_image_edit_bf16.safetensors")) {
        hf download Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_bf16.safetensors --local-dir ./ckpts
        Move-Item -Path ./ckpts/split_files/diffusion_models/qwen_image_edit_bf16.safetensors -Destination ./ckpts/diffusion_models/qwen_image_edit_bf16.safetensors
    }
}
elseif ($download_hy -eq "3") {
    Write-Output "正在下载 qwen_image-edit-plus 模型 / Downloading qwen_image-edit-plus model..."
    if (-not (Test-Path "./ckpts/diffusion_models/qwen_image_edit_2509_bf16.safetensors")) {
        hf download Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors --local-dir ./ckpts
        Move-Item -Path ./ckpts/split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors -Destination ./ckpts/diffusion_models/qwen_image_edit_2509_bf16.safetensors
    }
}

if ($download_hy -in @("1", "2", "3")) {
    if (-not (Test-Path "./ckpts/text_encoder/qwen_2.5_vl_7b.safetensors")) {
        hf download Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b.safetensors --local-dir ./ckpts
        Move-Item -Path ./ckpts/split_files/text_encoders/qwen_2.5_vl_7b.safetensors -Destination ./ckpts/text_encoder/qwen_2.5_vl_7b.safetensors
    }

    if (-not (Test-Path "./ckpts/vae/qwen_image_vae.safetensors")) {
        hf download Qwen/Qwen-Image vae/diffusion_pytorch_model.safetensors --local-dir ./ckpts
        Move-Item -Path ./ckpts/vae/diffusion_pytorch_model.safetensors -Destination ./ckpts/vae/qwen_image_vae.safetensors
    }
}

Write-Output "Install finished"
Read-Host | Out-Null ;
