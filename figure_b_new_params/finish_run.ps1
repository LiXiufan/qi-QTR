$ErrorActionPreference = "Stop"

$fixedProcessId = 5964
$ascendingProcessId = 32944
$stageDirectory = $PSScriptRoot
$repositoryDirectory = Split-Path -Parent $stageDirectory
$dataDirectory = Join-Path $repositoryDirectory "data_and_figures"
$statusLog = Join-Path $stageDirectory "completion_status.log"

function Get-StartedRunCount {
    param(
        [string]$LogPath,
        [string]$Prefix
    )
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return 0
    }
    return @(
        Get-Content -LiteralPath $LogPath |
            Where-Object { $_.StartsWith($Prefix) }
    ).Count
}

function Test-ProcessAlive {
    param([int]$ProcessId)
    return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

try {
    "monitor_started=$(Get-Date -Format o)" |
        Set-Content -LiteralPath $statusLog

    while (
        (Test-ProcessAlive -ProcessId $fixedProcessId) -or
        (Test-ProcessAlive -ProcessId $ascendingProcessId)
    ) {
        $fixedCount = Get-StartedRunCount `
            -LogPath (Join-Path $stageDirectory "fixed_5000.stdout.log") `
            -Prefix "[fixed]"
        $ascendingCount = Get-StartedRunCount `
            -LogPath (Join-Path $stageDirectory "ascending_5000.stdout.log") `
            -Prefix "[ascending]"
        "progress=$(Get-Date -Format o) fixed=$fixedCount/300 ascending=$ascendingCount/325" |
            Add-Content -LiteralPath $statusLog
        Start-Sleep -Seconds 60
    }

    $fixedErrorLog = Join-Path $stageDirectory "fixed_5000.stderr.log"
    $ascendingErrorLog = Join-Path $stageDirectory "ascending_5000.stderr.log"
    if (
        (Get-Item -LiteralPath $fixedErrorLog).Length -ne 0 -or
        (Get-Item -LiteralPath $ascendingErrorLog).Length -ne 0
    ) {
        throw "At least one experiment wrote to stderr."
    }

    $expectedRows = @{
        "fixed_gamma_all_restarts(shot5000).csv" = 300
        "fixed_gamma_restart_avg(shot5000).csv" = 60
        "fixed_gamma_shot_5000.csv" = 12
        "schedule_gamma_all_restarts(shots5000).csv" = 325
        "schedule_gamma_restart_avg(shots5000).csv" = 65
        "schedule_gamma_restart_group(shots5000)all.csv" = 13
    }
    foreach ($entry in $expectedRows.GetEnumerator()) {
        $sourcePath = Join-Path $stageDirectory $entry.Key
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "Missing expected output: $sourcePath"
        }
        $actualRows = @(Import-Csv -LiteralPath $sourcePath).Count
        if ($actualRows -ne $entry.Value) {
            throw (
                "Unexpected row count for {0}: expected {1}, found {2}" -f
                $entry.Key,
                $entry.Value,
                $actualRows
            )
        }
    }

    New-Item -ItemType Directory -Path $dataDirectory -Force | Out-Null
    foreach ($fileName in $expectedRows.Keys) {
        Copy-Item `
            -LiteralPath (Join-Path $stageDirectory $fileName) `
            -Destination (Join-Path $dataDirectory $fileName) `
            -Force
    }

    $python = Join-Path $repositoryDirectory ".venv\Scripts\python.exe"
    $plotScript = Join-Path $repositoryDirectory "ascending_tilt.py"
    & $python $plotScript `
        --fixed-csv (Join-Path $dataDirectory "fixed_gamma_shot_5000.csv") `
        --ascending-csv (
            Join-Path $dataDirectory `
                "schedule_gamma_restart_group(shots5000)all.csv"
        ) `
        --output-png (
            Join-Path $dataDirectory "schedule_gamma_expquad_fit.png"
        ) `
        --output-pdf (
            Join-Path $dataDirectory "schedule_gamma_expquad_fit.pdf"
        ) `
        --fit-summary-csv (
            Join-Path $dataDirectory `
                "schedule_gamma_expquad_fit_summary.csv"
        )
    if ($LASTEXITCODE -ne 0) {
        throw "Figure (b) plotting failed with exit code $LASTEXITCODE."
    }

    "complete=$(Get-Date -Format o)" |
        Add-Content -LiteralPath $statusLog
}
catch {
    "failed=$(Get-Date -Format o) message=$($_.Exception.Message)" |
        Add-Content -LiteralPath $statusLog
    exit 1
}
