# Pack one DLSS-NR resource into a new DLL; the original template is never modified.
# Native loading is unverified; PE checksums and signatures are not repaired.
[CmdletBinding()]
param(
    [string]$template_dll = "./ckpts/dlssnr/nvngx_dlssnr.dll",
    # Canonical checkpoint or CC records. For a multi-model root, also set weights.
    [string]$model_dir = "./ckpts/dlssnr/canonical",
    [string]$output_dll = "./output_dir/dlssnr_export/nvngx_dlssnr.dll",
    # Empty = infer from a single model directory; select HT/CC for a bundle root.
    [ValidateSet("", "HT", "CC")][string]$weights = "",
    # Canonical only: 0 = template weights, 1 = trained/merged weights.
    [ValidateRange(0.0, 1.0)][double]$mix = 1.0,
    # Scales ALL numeric weights after mixing, NOT visual effect strength.
    [double]$strength = 1.0,
    [string]$merge_lora = "",
    [double]$lora_multiplier = 1.0,
    [string]$python_executable = ""
)

# ============= DO NOT MODIFY CONTENTS BELOW =====================
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "powershell/native_command.ps1")
. (Join-Path $PSScriptRoot "powershell/dlssnr_workflow.ps1")
foreach ($name in @("template_dll", "model_dir", "output_dll")) {
    if ([string]::IsNullOrWhiteSpace((Get-Variable -Name $name -ValueOnly))) {
        throw "DLSS-NR $name is required."
    }
}
$cliArguments = @(
    "--template_dll=$template_dll", "--model_dir=$model_dir", "--output_dll=$output_dll",
    "--mix=$(ConvertTo-DLSSNRScalar $mix)", "--strength=$(ConvertTo-DLSSNRScalar $strength)",
    "--lora_multiplier=$(ConvertTo-DLSSNRScalar $lora_multiplier)"
)
if ($weights) { $cliArguments += "--weights=$($weights.ToUpperInvariant())" }
if (-not [string]::IsNullOrWhiteSpace($merge_lora)) { $cliArguments += "--merge_lora=$merge_lora" }
Invoke-DLSSNRScript -ProjectRoot $PSScriptRoot -EntryPoint "dlssnr_pack_model.py" `
    -Arguments $cliArguments -PythonExecutable $python_executable
