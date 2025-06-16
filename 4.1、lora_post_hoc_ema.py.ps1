# Script to run LoRA Post Hoc EMA by @bdsqlsz

# --- User Configurable Parameters ---
# Adjust these paths and values as needed
$lora_input_files = @(
    "lora_epoch_001.safetensors",
    "lora_epoch_002.safetensors",
    "lora_epoch_003.safetensors"
) # List of input LoRA files (provide paths if not in CWD)

$output_file_path = "lora_power_ema_merged.safetensors" # Output file path
$method = "power" # "constant", "linear", "power"
$beta = 0.90 #0.90~0.95
$beta2 = 0.95 #0.90~0.95
$sigma_rel_value = 0.2 #0.15~0.25

# Path to the python script, relative to this PowerShell script's location
$python_script_path = "./musubi_tuner/lora_post_hoc_ema.py"
# --- End User Configurable Parameters ---

# ============= DO NOT MODIFY CONTENTS BELOW (unless you know what you're doing) =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Activating Windows venv from ./venv/Scripts/activate"
        ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Activating Windows .venv from ./.venv/Scripts/activate"
        ./.venv/Scripts/activate
    }
    else {
        Write-Warning "Could not find a Python virtual environment to activate."
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Activating Linux venv from ./venv/bin/activate"
    # For PowerShell Core on Linux, source the activate script if direct execution doesn't set env vars
    # This might need adjustment based on the specific shell and venv setup
    . ./venv/bin/activate
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Activating Linux .venv from ./.venv/bin/activate"
    . ./.venv/bin/activate
}
else {
    Write-Warning "Could not find a Python virtual environment to activate."
}

$Env:HF_HOME = "huggingface"
#$Env:HF_ENDPOINT = "https://hf-mirror.com" # Uncomment if using a mirror
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$ext_args = [System.Collections.ArrayList]::new()

if ($method -ieq "constant") {
    if ($beta -ne 0.95) {
        [void]$ext_args.Add("--beta=$beta")
    }
}

elseif ($method -ieq "linear") {
    if ($beta) {
        [void]$ext_args.Add("--beta=$beta")
    }
    if ($beta2) {
        [void]$ext_args.Add("--beta2=$beta2")
    }
}
elseif ($method -ieq "power") {
    if ($sigma_rel_value) {
        [void]$ext_args.Add("--sigma_rel=$sigma_rel_value")
    }
}

if ($beta) {
    [void]$ext_args.Add("--beta=$beta")
}

if ($beta2) {
    [void]$ext_args.Add("--beta2=$beta2")
}

if ($sigma_rel_value) {
    [void]$ext_args.Add("--sigma_rel=$sigma_rel_value")
}

# Run LoRA Post Hoc EMA script
Write-Output "Running LoRA Post Hoc EMA script..."
Write-Output "Command: python $python_script_path $lora_input_files --output_file $output_file_path $ext_args"

python -m accelerate.commands.launch $python_script_path $lora_input_files --output_file $output_file_path $ext_args

Write-Output "LoRA Post Hoc EMA finished"
Read-Host | Out-Null ;