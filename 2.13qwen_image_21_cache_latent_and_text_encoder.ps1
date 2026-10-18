# Dedicated Qwen-Image 2.1 caches; earlier Qwen-Image caches are incompatible.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$dataset_config = './toml/qinglong-qwen-image-21-datasets.toml',
    [string]$vae = './ckpts/vae/qwen_image_2.1_vae_bf16.safetensors',
    [string]$text_encoder = './ckpts/text_encoder/qwen3vl_8b_bf16.safetensors',
    [int]$batch_size = 1,
    [int]$num_workers = 0,
    [string]$device = '',
    [bool]$vae_tiling = $true,
    [switch]$fp8_vl,
    [switch]$skip_existing,
    [switch]$keep_cache,
    [string]$python_executable = 'python'
)
. (Join-Path $PSScriptRoot 'powershell/image_architecture_workflow.ps1')
Invoke-ImageArchitectureCache 'qwen_image_21' (Get-ImageWorkflowOptions $MyInvocation.MyCommand.Parameters.Keys)
