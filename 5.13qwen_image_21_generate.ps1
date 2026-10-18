# image_size is emitted in native height/width order; empty flow_shift uses dynamic shifting.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$dit = './ckpts/diffusion_models/qwen_image_2.1_bf16.safetensors',
    [string]$dit_config = '',
    [string]$vae = './ckpts/vae/qwen_image_2.1_vae_bf16.safetensors',
    [string]$text_encoder = './ckpts/text_encoder/qwen3vl_8b_bf16.safetensors',
    [string]$prompt = 'A studio photograph of a red ceramic cup',
    [string]$negative_prompt = '',
    [int]$width = 1024,
    [int]$height = 1024,
    [int]$infer_steps = 40,
    [double]$guidance_scale = 1,
    [string]$flow_shift = '',
    [int]$seed = 42,
    [string]$save_path = './output_dir/qwen_image_21/images',
    [string]$output_type = 'images',
    [string[]]$control_image_path = @(),
    [string]$mask_path = '',
    [switch]$resize_control_to_image_size,
    [string[]]$lora_weight = @(),
    [double[]]$lora_multiplier = @(),
    [string]$attn_mode = 'sdpa',
    [string]$dtype = 'bfloat16',
    [string]$device = '',
    [int]$blocks_to_swap = 0,
    [switch]$use_pinned_memory_for_block_swap,
    [switch]$text_encoder_cpu,
    [switch]$fp8_vl,
    [switch]$fp8_scaled,
    [bool]$vae_tiling = $true,
    [switch]$no_metadata,
    [string]$python_executable = 'python'
)
. (Join-Path $PSScriptRoot 'powershell/image_architecture_workflow.ps1')
Invoke-ImageArchitectureGeneration 'qwen_image_21' (Get-ImageWorkflowOptions $MyInvocation.MyCommand.Parameters.Keys)
