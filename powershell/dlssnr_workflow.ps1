function Get-DLSSNRScriptSettings {
    param([string[]]$Names)

    $settings = @{}
    # Include parameter defaults as well as caller overrides; PSBoundParameters omits defaults.
    foreach ($name in $Names) {
        $settings[$name] = Get-Variable -Name $name -Scope 1 -ValueOnly -ErrorAction SilentlyContinue
    }
    return $settings
}

function ConvertTo-DLSSNRScalar {
    param($Value)

    if ($Value -is [System.IFormattable]) {
        return $Value.ToString($null, [System.Globalization.CultureInfo]::InvariantCulture)
    }
    return [string]$Value
}

function New-DLSSNRTrainArguments {
    param(
        [hashtable]$Settings,
        [ValidateSet("lora", "finetune")][string]$TrainMode
    )

    if (-not $Settings.model_dir -and -not $Settings.development_smoke) {
        throw "DLSS-NR model_dir is required unless development_smoke is explicitly enabled."
    }
    if (-not $Settings.output_dir) {
        throw "DLSS-NR output_dir is required."
    }
    $trainingMode = $Settings.training_mode.ToLowerInvariant()
    if (-not $Settings.dataset_config) {
        $Settings.dataset_config = if ($trainingMode -eq "temporal") {
            "./toml/qinglong_dlssnr_temporal.toml"
        } else {
            "./toml/qinglong_dlssnr_single.toml"
        }
    }
    $arguments = [System.Collections.Generic.List[string]]::new()
    foreach ($name in @(
        "dataset_config", "training_mode", "device", "deployment_target", "seed", "max_train_steps",
        "gradient_accumulation_steps", "optimizer_type", "learning_rate", "max_grad_norm",
        "loss_pre", "loss_out", "loss_edge", "loss_temporal", "sample_every_n_steps",
        "min_sequence_frames", "output_dir", "output_name", "save_every_n_steps",
        "numerics_profile", "mixed_precision", "attention_backend", "attention_scope", "max_overflow_retries",
        "lr_scheduler", "lr_warmup_steps", "lr_decay_steps", "lr_scheduler_num_cycles", "lr_scheduler_power",
        "lr_scheduler_min_lr_ratio", "loss_profile", "base_anchor_weight", "dino_loss_weight", "profile", "save_precision"
    )) {
        $value = if ($name -eq "training_mode") {
            $trainingMode
        } elseif ($name -in @("device", "deployment_target", "numerics_profile", "mixed_precision", "attention_backend", "attention_scope", "lr_scheduler", "loss_profile", "profile", "save_precision")) {
            $Settings[$name].ToLowerInvariant()
        } else {
            ConvertTo-DLSSNRScalar $Settings[$name]
        }
        $arguments.Add("--${name}=$value")
    }
    foreach ($name in @(
        "model_dir", "forward_validation_report", "resume", "lr_scheduler_timescale", "lr_scheduler_type",
        "loss_lowpass_sigma", "control_residual_sigma", "control_anchor_probability", "control_corner_probability",
        "dino_loss_model_type", "dino_loss_layer", "dino_loss_resize"
    )) {
        $value = ConvertTo-DLSSNRScalar $Settings[$name]
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            if ($name -eq "dino_loss_model_type") { $value = $value.ToLowerInvariant() }
            $arguments.Add("--${name}=$value")
        }
    }
    $emaDecay = ConvertTo-DLSSNRScalar $Settings.ema_decay
    if ($Settings.no_ema -or [string]::IsNullOrWhiteSpace($emaDecay)) {
        $arguments.Add("--no-ema")
    } else {
        $arguments.Add("--ema_decay=$emaDecay")
    }
    foreach ($name in @(
        "development_smoke", "save_state", "gradient_checkpointing", "fp8_base", "fp8_scaled",
        "control_randomization", "eval_native", "eval_detail_diagnostics", "eval_content_preservation"
    )) {
        if ($Settings[$name]) { $arguments.Add("--$name") }
    }
    foreach ($name in @("native_weight_qat", "compare_baseline", "shuffle_dataset", "dino_loss_use_gram", "dino_loss_use_norm")) {
        if ($null -ne $Settings[$name]) {
            $prefix = if ($Settings[$name]) { "--" } else { "--no-" }
            $arguments.Add("${prefix}${name}")
        }
    }
    if ($trainingMode -eq "temporal") {
        foreach ($name in @("sequence_length", "burn_in", "tbptt_length")) {
            $arguments.Add("--${name}=$(ConvertTo-DLSSNRScalar $Settings[$name])")
        }
    }
    if ($TrainMode -eq "lora") {
        $arguments.Add("--network_dropout=$(ConvertTo-DLSSNRScalar $Settings.network_dropout)")
        $profile = $Settings.lora_profile.ToLowerInvariant()
        if ($profile -eq "vit_only") {
            $arguments.Add("--network_dim=$(ConvertTo-DLSSNRScalar $Settings.network_dim)")
            if ($Settings.network_alpha -ne "") {
                $arguments.Add("--network_alpha=$(ConvertTo-DLSSNRScalar $Settings.network_alpha)")
            }
        }
        $arguments.Add("--network_args")
        $arguments.Add("profile=$profile")
        if ($profile -eq "multiscale") {
            foreach ($name in @("rank_by_width", "alpha_by_width")) {
                if ($Settings[$name]) { $arguments.Add("${name}=$($Settings[$name])") }
            }
        }
    } else {
        foreach ($name in @("prior_lr_multiplier", "scale_lr_multiplier", "temporal_blend_lr_multiplier")) {
            $arguments.Add("--${name}=$(ConvertTo-DLSSNRScalar $Settings[$name])")
        }
    }
    if ($Settings.optimizer_args.Count -gt 0) {
        $arguments.Add("--optimizer_args")
        foreach ($token in $Settings.optimizer_args) {
            if ($token -notmatch '^[A-Za-z_]\w*=.+$') {
                throw "DLSS-NR optimizer_args requires key=value tokens, not CLI options."
            }
            $arguments.Add($token)
        }
    }
    if ($Settings.lr_scheduler_args.Count -gt 0) {
        $arguments.Add("--lr_scheduler_args")
        foreach ($token in $Settings.lr_scheduler_args) {
            if ($token -notmatch '^[A-Za-z_]\w*=.+$') {
                throw "DLSS-NR lr_scheduler_args requires key=value tokens, not CLI options."
            }
            $arguments.Add($token)
        }
    }
    return ,$arguments.ToArray()
}

function Invoke-DLSSNRScript {
    param(
        [string]$ProjectRoot,
        [string]$EntryPoint,
        [string[]]$Arguments,
        [string]$PythonExecutable = "",
        [switch]$Training,
        [ValidateSet("no", "fp16", "bf16")][string]$MixedPrecision = "no"
    )

    if (-not $PythonExecutable) {
        foreach ($candidate in @(".venv/Scripts/python.exe", ".venv/bin/python", "venv/Scripts/python.exe", "venv/bin/python")) {
            $path = Join-Path $ProjectRoot $candidate
            if (Test-Path -LiteralPath $path) {
                $PythonExecutable = $path
                break
            }
        }
        if (-not $PythonExecutable) { $PythonExecutable = "python" }
    }
    $oldPrecision = $env:ACCELERATE_MIXED_PRECISION
    $PSNativeCommandUseErrorActionPreference = $false
    Push-Location -LiteralPath $ProjectRoot
    try {
        if ($Training) { $env:ACCELERATE_MIXED_PRECISION = $MixedPrecision }
        & $PythonExecutable (Join-Path $ProjectRoot "musubi-tuner/$EntryPoint") @Arguments
        Assert-NativeCommandSucceeded "DLSS-NR command failed: $EntryPoint" -ErrorAction Continue
    } finally {
        if ($Training) { $env:ACCELERATE_MIXED_PRECISION = $oldPrecision }
        Pop-Location
    }
}
