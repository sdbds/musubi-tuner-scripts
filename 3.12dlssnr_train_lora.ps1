# DLSS-NR supervised LoRA training. No diffusion caches or Accelerate launcher.
[CmdletBinding()]
param(
    [string]$dataset_config = "",
    [string]$model_dir = "./ckpts/dlssnr/canonical",
    [string]$forward_validation_report = "",
    [switch]$development_smoke,
    [ValidateSet("auto", "cpu", "cuda")][string]$device = "auto",
    [ValidateSet("float_runtime", "native_roundtrip")][string]$deployment_target = "float_runtime",
    [ValidateSet("single_frame", "temporal")][string]$training_mode = "single_frame",
    [string]$sequence_length = "4",
    [string]$burn_in = "2",
    [string]$tbptt_length = "2",
    [string]$seed = "42",
    [string]$max_train_steps = "1000",
    [string]$gradient_accumulation_steps = "1",
    [string]$optimizer_type = "AdamW",
    [string[]]$optimizer_args = @("weight_decay=0.0"),
    [string]$learning_rate = "1e-4",
    [string]$max_grad_norm = "0.0",
    [string]$loss_pre = "1.0",
    [string]$loss_out = "1.0",
    [string]$loss_edge = "0.05",
    [string]$loss_temporal = "0.0",
    [ValidateSet("vit_only", "multiscale")][string]$lora_profile = "vit_only",
    [string]$network_dim = "16",
    [string]$network_alpha = "",
    [string]$network_dropout = "0.0",
    [string]$rank_by_width = "",
    [string]$alpha_by_width = "",
    [string]$sample_every_n_steps = "0",
    [string]$min_sequence_frames = "64",
    [bool]$compare_baseline = $true,
    [string]$output_dir = "./output_dir/dlssnr",
    [string]$output_name = "dlssnr_lora",
    [string]$save_every_n_steps = "100",
    [switch]$save_state,
    [string]$resume = "",
    [string]$python_executable = ""
)

# ============= DO NOT MODIFY CONTENTS BELOW =====================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")
. (Join-Path $PSScriptRoot "powershell/dlssnr_workflow.ps1")
$settings = Get-DLSSNRScriptSettings -Names @($MyInvocation.MyCommand.Parameters.Keys)
$cliArguments = New-DLSSNRTrainArguments -Settings $settings -TrainMode "lora"
Invoke-DLSSNRScript -ProjectRoot $PSScriptRoot -EntryPoint "dlssnr_train_network.py" `
    -Arguments $cliArguments -PythonExecutable $python_executable -Training
