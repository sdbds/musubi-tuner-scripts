# MOSS is T2I-only. Its VAE always uses FP32 and its text encoder uses BF16.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$dataset_config = './toml/qinglong-moss-image-datasets.toml',
    [string]$vae = './ckpts/vae/flux2-vae.safetensors',
    [string]$text_encoder = './ckpts/text_encoder/qwen3.5_9b_bf16.safetensors',
    [string]$tokenizer_path = '',
    [int]$moss_max_token_length = 4096,
    [int]$batch_size = 1,
    [int]$num_workers = 0,
    [string]$device = '',
    [switch]$skip_existing,
    [switch]$keep_cache,
    [string]$python_executable = 'python'
)
. (Join-Path $PSScriptRoot 'powershell/image_architecture_workflow.ps1')
Invoke-ImageArchitectureCache 'moss_image' (Get-ImageWorkflowOptions $MyInvocation.MyCommand.Parameters.Keys)
