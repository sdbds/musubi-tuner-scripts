function Resolve-H3BestOfKCount {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [AllowEmptyString()]
        [object]$Value
    )

    $errorMessage = "MiniMax-H3 h3_best_of_k must be a base-10 integer of at least 1."
    if (
        $null -eq $Value -or
        $Value -is [bool] -or
        $Value -is [single] -or
        $Value -is [double] -or
        $Value -is [decimal]
    ) {
        throw $errorMessage
    }

    $text = [Convert]::ToString($Value, [Globalization.CultureInfo]::InvariantCulture)
    [long]$count = 0
    $style = [Globalization.NumberStyles]::Integer
    if (-not [long]::TryParse(
        $text,
        $style,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$count
    ) -or $count -lt 1) {
        throw $errorMessage
    }

    return $count
}

function Resolve-H3BestOfKStream {
    param(
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [AllowEmptyString()]
        [object]$Value
    )

    if ($Value -isnot [string]) {
        throw "MiniMax-H3 h3_best_of_k_stream must be video or audio."
    }
    $stream = $Value.Trim().ToLowerInvariant()
    if ($stream -notin @("video", "audio")) {
        throw "MiniMax-H3 h3_best_of_k_stream must be video or audio."
    }

    return $stream
}

function Assert-NoH3BestOfKReservedArguments {
    param(
        [AllowNull()]
        [object]$Arguments
    )

    $text = if ($Arguments -is [string]) {
        [string]$Arguments
    }
    else {
        (@($Arguments) | ForEach-Object { [string]$_ }) -join "`n"
    }
    $pattern = '(?m)(?<!\S)(--(?:h3_best_of_k(?:_stream)?|xm_best_of_k))(?=$|[\s=])'
    $match = [regex]::Match($text, $pattern)
    if ($match.Success) {
        throw "MiniMax-H3 option $($match.Groups[1].Value) is reserved; use the structured Best-of-K variables."
    }
}

function Assert-H3BestOfKArgumentInvariant {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [object[]]$Arguments
    )

    $options = foreach ($argument in $Arguments) {
        $match = [regex]::Match(
            [string]$argument,
            '^\s*(--(?:h3_best_of_k(?:_stream)?|xm_best_of_k))(?=$|[\s=])'
        )
        if ($match.Success) {
            $match.Groups[1].Value
        }
    }

    $countOccurrences = @(
        $options | Where-Object { $_ -eq '--h3_best_of_k' }
    ).Count
    $streamOccurrences = @(
        $options | Where-Object { $_ -eq '--h3_best_of_k_stream' }
    ).Count

    if ($options -contains '--xm_best_of_k') {
        throw "MiniMax-H3 option --xm_best_of_k is not enabled; use h3_best_of_k."
    }
    if (
        ($countOccurrences -eq 0 -and $streamOccurrences -eq 0) -or
        ($countOccurrences -eq 1 -and $streamOccurrences -eq 1)
    ) {
        return
    }
    throw "MiniMax-H3 Best-of-K arguments must be absent or occur once as a complete pair."
}
