[CmdletBinding()]
param(
    [int]$Port = 7788,
    [string]$BindHost = "127.0.0.1",
    [switch]$Cloud,
    [switch]$Native,
    [switch]$NoBrowser,
    [switch]$NoPause
)

$ErrorActionPreference = "Stop"
$GuiDir = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $GuiDir "..")).Path
Set-Location $GuiDir

function Exit-WithError {
    param([string]$Message)

    Write-Host $Message -ForegroundColor Red
    if (-not $NoPause) {
        Read-Host "Press Enter to exit" | Out-Null
    }
    exit 1
}

function Resolve-ProjectPython {
    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot "venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".venv/bin/python"),
        (Join-Path $ProjectRoot "venv/bin/python")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    if ($env:VIRTUAL_ENV) {
        $activeCandidates = @(
            (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"),
            (Join-Path $env:VIRTUAL_ENV "bin/python")
        )

        foreach ($candidate in $activeCandidates) {
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }
        }
    }

    return $null
}

$PythonExe = Resolve-ProjectPython
if (-not $PythonExe) {
    Exit-WithError "Project virtual environment was not found. From $ProjectRoot run: uv sync --extra cu130 --extra gui --extra lycoris --extra attention --index-strategy unsafe-best-match"
}

$PythonDir = Split-Path -Parent $PythonExe
$VenvDir = Split-Path -Parent $PythonDir
$Env:VIRTUAL_ENV = $VenvDir
$Env:PATH = "$PythonDir$([System.IO.Path]::PathSeparator)$Env:PATH"

Write-Host "Using project Python: $PythonExe" -ForegroundColor Cyan
Write-Host "Virtual environment: $VenvDir" -ForegroundColor Cyan
$Env:MUSUBI_GUI_PORT = [string]$Port

try {
    & $PythonExe -c "import nicegui; import sys; print(sys.version); print(sys.executable)"
    if ($LASTEXITCODE -ne 0) {
        throw "nicegui import failed"
    }
}
catch {
    Exit-WithError "Project virtual environment is missing NiceGUI. From $ProjectRoot run: uv sync --extra cu130 --extra gui --extra lycoris --extra attention --index-strategy unsafe-best-match"
}

$launchArgs = @(".\launch.py", "--port=$Port")
if ($Cloud) {
    $launchArgs += "--cloud"
}
elseif ($BindHost -and $BindHost -ne "0.0.0.0") {
    $launchArgs += "--host=$BindHost"
}
if ($Native) {
    $launchArgs += "--native"
}
if ($NoBrowser) {
    $launchArgs += "--no-browser"
}

Write-Host "Starting NiceGUI launcher (preferred port: $Port)..." -ForegroundColor Green
if ($Native) {
    Write-Host "Mode: Native window" -ForegroundColor Cyan
}
else {
    Write-Host "Mode: Web browser" -ForegroundColor Cyan
}
Write-Host "Launch args: $($launchArgs -join ' ')" -ForegroundColor DarkGray
& $PythonExe @launchArgs
$exitCode = $LASTEXITCODE

if (-not $NoPause) {
    Read-Host "Press Enter to exit" | Out-Null
}

exit $exitCode
