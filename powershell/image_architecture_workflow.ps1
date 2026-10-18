. (Join-Path $PSScriptRoot 'native_command.ps1')

function Get-ImageWorkflowOptions {
    param([string[]]$ParameterNames)
    $options = @{}
    foreach ($name in $ParameterNames) {
        if ($name -in [System.Management.Automation.PSCmdlet]::CommonParameters) { continue }
        $options[$name] = Get-Variable -Name $name -Scope 1 -ValueOnly
    }
    return $options
}

function Initialize-ImageWorkflow {
    $root = Split-Path $PSScriptRoot -Parent
    Set-Location $root
    foreach ($venv in @('venv', '.venv')) {
        $activate = if ($env:OS -ilike '*windows*') { "$venv/Scripts/Activate.ps1" } else { "$venv/bin/Activate.ps1" }
        if (Test-Path $activate) {
            . $activate
            break
        }
    }
    $src = Join-Path $root 'musubi-tuner/src'
    $env:PYTHONPATH = if ($env:PYTHONPATH) { "$src$([IO.Path]::PathSeparator)$env:PYTHONPATH" } else { $src }
    $env:HF_HOME = 'huggingface'
    $env:XFORMERS_FORCE_DISABLE_TRITON = '1'
}

function ConvertTo-ImageWorkflowArguments {
    param([hashtable]$Options, [string[]]$Exclude = @(), [string[]]$KeepEmpty = @())
    foreach ($name in ($Options.Keys | Sort-Object)) {
        if ($name -in $Exclude) { continue }
        $value = $Options[$name]
        if ($value -is [bool] -or $value -is [System.Management.Automation.SwitchParameter]) {
            if ($value) { "--$name" }
        }
        elseif ($value -is [array]) {
            if ($value.Count) {
                "--$name"
                foreach ($item in $value) {
                    if ($item -is [System.IFormattable]) { $item.ToString($null, [cultureinfo]::InvariantCulture) }
                    else { [string]$item }
                }
            }
        }
        elseif ($null -ne $value -and ([string]$value -ne '' -or $name -in $KeepEmpty)) {
            $text = if ($value -is [System.IFormattable]) { $value.ToString($null, [cultureinfo]::InvariantCulture) } else { [string]$value }
            "--${name}=$text"
        }
    }
}

function Invoke-ImageWorkflowPython {
    param([string]$Executable, [string[]]$Arguments, [string]$Description)
    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $false
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        $nativeExitCode = $LASTEXITCODE
        # Write-Error can throw under a caller's Stop preference; keep the native exit code.
        try { Assert-NativeCommandSucceeded $Description -ErrorAction Continue }
        finally { exit $nativeExitCode }
    }
}

function Assert-ImageWorkflowSwap {
    param([string]$Architecture, [int]$Blocks)
    $maximum = if ($Architecture -eq 'qwen_image_21') { 31 } else { 40 }
    if ($Blocks -lt 0 -or $Blocks -gt $maximum) { throw "blocks_to_swap must be between 0 and $maximum for $Architecture." }
}

function Invoke-ImageArchitectureCache {
    param([string]$Architecture, [hashtable]$Options)
    if ($Options.batch_size -lt 1 -or $Options.num_workers -lt 0) { throw 'batch_size must be positive and num_workers nonnegative.' }
    if ($Architecture -eq 'moss_image' -and ($Options.moss_max_token_length -lt 1 -or $Options.moss_max_token_length -gt 4096)) {
        throw 'moss_max_token_length must be between 1 and 4096.'
    }
    Initialize-ImageWorkflow
    $common = @{}
    foreach ($name in @('dataset_config', 'batch_size', 'num_workers', 'device', 'skip_existing', 'keep_cache')) {
        $common[$name] = $Options[$name]
    }
    $latent = $common.Clone()
    $latent.vae = $Options.vae
    $text = $common.Clone()
    $text.text_encoder = $Options.text_encoder
    if ($Architecture -eq 'qwen_image_21') {
        $latent.vae_tiling = $Options.vae_tiling
        $text.fp8_vl = $Options.fp8_vl
        $text.text_encoder_dtype = if ($Options.device -eq 'cpu') { 'float32' } else { 'bfloat16' }
    }
    else {
        $latent.vae_dtype = 'float32'
        $text.tokenizer_path = $Options.tokenizer_path
        $text.moss_max_token_length = $Options.moss_max_token_length
    }
    $latentArgs = @('-m', "musubi_tuner.${Architecture}_cache_latents") + @(ConvertTo-ImageWorkflowArguments $latent)
    Invoke-ImageWorkflowPython $Options.python_executable $latentArgs "$Architecture latent caching failed"
    $textArgs = @('-m', "musubi_tuner.${Architecture}_cache_text_encoder_outputs") + @(ConvertTo-ImageWorkflowArguments $text)
    Invoke-ImageWorkflowPython $Options.python_executable $textArgs "$Architecture text caching failed"
}

function Invoke-ImageArchitectureTraining {
    param([string]$Architecture, [bool]$FullModel, [hashtable]$Options)
    Assert-ImageWorkflowSwap $Architecture $Options.blocks_to_swap
    if ($Options.fp8_scaled -and -not $Options.fp8_base) { throw 'fp8_scaled training requires fp8_base.' }
    if ($Options.gradient_checkpointing_cpu_offload -and (-not $Options.gradient_checkpointing -or $Options.blocks_to_swap)) {
        throw 'gradient_checkpointing_cpu_offload requires gradient_checkpointing and cannot be combined with block swap.'
    }
    foreach ($name in @('optimizer_args', 'network_args')) {
        foreach ($value in $Options[$name]) {
            if ($value -like '--*') { throw "$name cannot contain command-line options." }
        }
    }
    if ($Options.optimizer_type -ieq 'Adafactor' -and -not $Options.optimizer_args.Count) {
        $Options.optimizer_args = @('relative_step=False', 'scale_parameter=False', 'warmup_init=False')
    }
    if ($FullModel) {
        if ($Options.fused_backward_pass -and ($Options.optimizer_type -ine 'Adafactor' -or $Options.gradient_accumulation_steps -ne 1)) {
            throw 'fused_backward_pass requires Adafactor and gradient_accumulation_steps 1.'
        }
        if ($Architecture -eq 'moss_image' -and $Options.fused_backward_pass -and $Options.max_grad_norm -ne 0) {
            throw 'MOSS fused_backward_pass requires max_grad_norm 0.'
        }
        if ($Options.blocks_to_swap) {
            if ($Options.optimizer_type -inotmatch '^(AdamW|torch\.optim\.AdamW|Adafactor)$') { throw 'Full-model block swap supports AdamW or Adafactor.' }
            if ($Architecture -eq 'moss_image' -and $Options.gradient_accumulation_steps -ne 1) {
                throw 'MOSS full-model block swap requires gradient_accumulation_steps 1.'
            }
            if (-not $Options.fused_backward_pass) { $Options.block_swap_optimizer_patch_params = $true }
        }
    }
    elseif ($Options.block_swap_h2d_only -and (-not $Options.blocks_to_swap -or -not $Options.gradient_checkpointing)) {
        throw 'block_swap_h2d_only requires blocks_to_swap and gradient_checkpointing.'
    }
    $wrapperOnly = @('python_executable', 'enable_sample')
    if (-not $Options.enable_sample) {
        $wrapperOnly += @('sample_prompts', 'sample_every_n_steps', 'sample_at_first', 'vae', 'text_encoder', 'vae_tiling', 'fp8_vl', 'tokenizer_path', 'sample_uncond_skip_layers')
    }
    Initialize-ImageWorkflow
    $suffix = if ($FullModel) { 'train' } else { 'train_network' }
    $arguments = @('-m', 'accelerate.commands.launch', '--num_processes=1', '--num_machines=1', '--num_cpu_threads_per_process=1', '--mixed_precision=bf16', '--module', "musubi_tuner.${Architecture}_$suffix")
    $arguments += '--mixed_precision=bf16'
    $arguments += '--sdpa'
    $arguments += @(ConvertTo-ImageWorkflowArguments $Options -Exclude $wrapperOnly -KeepEmpty @('sample_uncond_skip_layers'))
    Invoke-ImageWorkflowPython $Options.python_executable $arguments "$Architecture training failed"
}

function Invoke-ImageArchitectureGeneration {
    param([string]$Architecture, [hashtable]$Options)
    Assert-ImageWorkflowSwap $Architecture $Options.blocks_to_swap
    $multiple = if ($Architecture -eq 'qwen_image_21') { 32 } else { 16 }
    if ($Options.width -le 0 -or $Options.height -le 0 -or $Options.width % $multiple -or $Options.height % $multiple) {
        throw "width and height must be positive multiples of $multiple."
    }
    if ($Options.lora_multiplier.Count -and (-not $Options.lora_weight.Count -or ($Options.lora_multiplier.Count -ne 1 -and $Options.lora_multiplier.Count -ne $Options.lora_weight.Count))) {
        throw 'lora_multiplier requires LoRA weights and one value or one value per weight.'
    }
    if ($Architecture -eq 'moss_image' -and $Options.lora_multiplier.Count -and $Options.lora_multiplier.Count -ne $Options.lora_weight.Count) {
        throw 'MOSS requires one lora_multiplier per lora_weight.'
    }
    $exclude = @('python_executable')
    $keepEmpty = @('uncond_skip_layers')
    if ($Architecture -eq 'qwen_image_21') {
        if ($Options.guidance_scale -gt 1) { $keepEmpty += 'negative_prompt' }
        if ($Options.mask_path -and (-not $Options.control_image_path.Count -or -not $Options.resize_control_to_image_size)) {
            throw 'mask_path requires a control_image_path and resize_control_to_image_size.'
        }
        $Options.image_size = @($Options.height, $Options.width)
        $exclude += @('height', 'width')
    }
    elseif ($Options.text_cache) {
        $exclude += @('text_encoder', 'tokenizer_path', 'prompt', 'negative_prompt')
    }
    Initialize-ImageWorkflow
    $arguments = @('-m', "musubi_tuner.${Architecture}_generate_image") + @(ConvertTo-ImageWorkflowArguments $Options -Exclude $exclude -KeepEmpty $keepEmpty)
    Invoke-ImageWorkflowPython $Options.python_executable $arguments "$Architecture generation failed"
}
