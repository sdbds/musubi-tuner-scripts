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
        "min_sequence_frames", "output_dir", "output_name", "save_every_n_steps"
    )) {
        $value = if ($name -eq "training_mode") {
            $trainingMode
        } elseif ($name -in @("device", "deployment_target")) {
            $Settings[$name].ToLowerInvariant()
        } else {
            ConvertTo-DLSSNRScalar $Settings[$name]
        }
        $arguments.Add("--${name}=$value")
    }
    foreach ($name in @("model_dir", "forward_validation_report", "resume")) {
        if ($Settings[$name]) {
            $arguments.Add("--${name}=$($Settings[$name])")
        }
    }
    $arguments.Add("--profile=dlss_nr_310_8_0")
    $arguments.Add("--numerics_profile=train_surrogate")
    $arguments.Add("--mixed_precision=no")
    $arguments.Add("--lr_scheduler=constant")
    if ($Settings.development_smoke) { $arguments.Add("--development_smoke") }
    if ($Settings.save_state) { $arguments.Add("--save_state") }
    if (-not $Settings.compare_baseline) { $arguments.Add("--no-compare_baseline") }
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
    return ,$arguments.ToArray()
}

function Invoke-DLSSNRScript {
    param(
        [string]$ProjectRoot,
        [string]$EntryPoint,
        [string[]]$Arguments,
        [string]$PythonExecutable = "",
        [switch]$Training
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
        if ($Training) { $env:ACCELERATE_MIXED_PRECISION = "no" }
        & $PythonExecutable (Join-Path $ProjectRoot "musubi-tuner/$EntryPoint") @Arguments
        Assert-NativeCommandSucceeded "DLSS-NR command failed: $EntryPoint" -ErrorAction Continue
    } finally {
        if ($Training) { $env:ACCELERATE_MIXED_PRECISION = $oldPrecision }
        Pop-Location
    }
}
