function Assert-NativeCommandSucceeded {
    param (
        [Parameter(Mandatory = $true)]
        [string]$ErrorInfo
    )

    $ExitCode = $LASTEXITCODE
    if ($ExitCode -eq 0) {
        return
    }

    Write-Error "$ErrorInfo (exit code: $ExitCode)"
    exit $ExitCode
}
