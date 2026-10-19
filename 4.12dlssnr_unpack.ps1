# Extract audited DLSS-NR weights from an original DLL. Paths are relative to this project.
[CmdletBinding()]
param(
    [string]$source_dll = "./ckpts/dlssnr/nvngx_dlssnr.dll",
    # Must be a new directory. Multiple resources create HT/CC subdirectories.
    [string]$output_dir = "./ckpts/dlssnr/unpacked",
    [ValidateSet("all", "HT", "CC")][string]$weights = "all",
    # records = lossless CC records; canonical = CC weights for training/inference.
    [ValidateSet("records", "canonical")][string]$cc_format = "canonical",
    [ValidateSet("native", "fp32")][string]$save_precision = "native",
    [string]$python_executable = ""
)

# ============= DO NOT MODIFY CONTENTS BELOW =====================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")
. (Join-Path $PSScriptRoot "powershell/dlssnr_workflow.ps1")
foreach ($name in @("source_dll", "output_dir")) {
    if ([string]::IsNullOrWhiteSpace((Get-Variable -Name $name -ValueOnly))) {
        throw "DLSS-NR $name is required."
    }
}
$selectedWeights = if ($weights -ieq "all") { "all" } else { $weights.ToUpperInvariant() }
$cliArguments = @(
    "--source_dll=$source_dll", "--output_dir=$output_dir", "--weights=$selectedWeights",
    "--cc_format=$($cc_format.ToLowerInvariant())", "--save_precision=$($save_precision.ToLowerInvariant())"
)
Invoke-DLSSNRScript -ProjectRoot $PSScriptRoot -EntryPoint "dlssnr_unpack_model.py" `
    -Arguments $cliArguments -PythonExecutable $python_executable
