# torch is this native generator's name for PyTorch SDPA. No FP8 or reference-image inputs.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$dit = './ckpts/diffusion_models/moss_image_base_bf16.safetensors',
    [string]$vae = './ckpts/vae/flux2-vae.safetensors',
    [string]$text_encoder = './ckpts/text_encoder/qwen3.5_9b_bf16.safetensors',
    [string]$tokenizer_path = '',
    [int]$moss_max_token_length = 4096,
    [string]$text_cache = '',
    [string]$prompt = 'A studio photograph of a red ceramic cup',
    [string]$negative_prompt = '',
    [int]$width = 1024,
    [int]$height = 1024,
    [int]$infer_steps = 50,
    [double]$guidance_scale = 3,
    [double]$flow_shift = 5,
    [string]$uncond_skip_layers = '22',
    [int]$seed = 42,
    [string]$output_dir = './output_dir/moss_image/images',
    [string]$output_name = '',
    [string[]]$lora_weight = @(),
    [double[]]$lora_multiplier = @(),
    [string]$attn_mode = 'torch',
    [string]$dtype = 'bfloat16',
    [string]$device = '',
    [int]$blocks_to_swap = 0,
    [switch]$split_attn,
    [switch]$overwrite,
    [string]$python_executable = 'python'
)
. (Join-Path $PSScriptRoot 'powershell/image_architecture_workflow.ps1')
Invoke-ImageArchitectureGeneration 'moss_image' (Get-ImageWorkflowOptions $MyInvocation.MyCommand.Parameters.Keys)
