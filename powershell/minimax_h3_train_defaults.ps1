function Test-H3NumericDefault {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [AllowEmptyString()]
        [object]$Value,

        [Parameter(Mandatory = $true)]
        [decimal]$DefaultValue
    )

    $errorMessage = "MiniMax-H3 default-sensitive value must be a finite number."
    if ($null -eq $Value -or $Value -is [bool]) {
        throw $errorMessage
    }

    $text = [Convert]::ToString(
        $Value,
        [Globalization.CultureInfo]::InvariantCulture
    )
    [decimal]$number = 0
    if (-not [decimal]::TryParse(
        $text,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$number
    )) {
        throw $errorMessage
    }

    return $number -eq $DefaultValue
}
