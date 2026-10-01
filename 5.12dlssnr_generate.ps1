# DLSS-NR stills or closed-loop PNG sequences, not text-to-image/video generation.
[CmdletBinding()]
param(
    [ValidateSet("image", "sequence")][string]$inference_mode = "image",
    [string]$model_dir = "./ckpts/dlssnr/canonical",
    [string]$sample_manifest = "./data/dlssnr/inference_single.jsonl",
    [string]$sequence_manifest = "./data/dlssnr/inference_sequence.jsonl",
    [string]$bucket_width = "512",
    [string]$bucket_height = "512",
    [string]$seed = "0",
    [ValidateSet("auto", "cpu", "cuda")][string]$device = "auto",
    [string]$output_dir = "./output_dir/dlssnr_preview",
    [string]$python_executable = ""
)

# ============= DO NOT MODIFY CONTENTS BELOW =====================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")
. (Join-Path $PSScriptRoot "powershell/dlssnr_workflow.ps1")
if (-not $model_dir -or -not $output_dir) { throw "DLSS-NR model_dir and output_dir are required." }
$sequence = $inference_mode -ieq "sequence"
$manifestName = if ($sequence) { "sequence_manifest" } else { "sample_manifest" }
$manifest = if ($sequence) { $sequence_manifest } else { $sample_manifest }
if (-not $manifest) { throw "DLSS-NR $manifestName is required." }
$cliArguments = @(
    "--model_dir=$model_dir", "--${manifestName}=$manifest", "--output_dir=$output_dir",
    "--bucket_width=$bucket_width", "--bucket_height=$bucket_height", "--seed=$seed",
    "--device=$($device.ToLowerInvariant())", "--numerics_profile=train_surrogate"
)
$entryPoint = if ($sequence) { "dlssnr_generate_video.py" } else { "dlssnr_generate_image.py" }
Invoke-DLSSNRScript -ProjectRoot $PSScriptRoot -EntryPoint $entryPoint `
    -Arguments $cliArguments -PythonExecutable $python_executable
