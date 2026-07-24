[CmdletBinding()]
param(
    [string]$SourceArchive = '',
    [string]$LocalArchive = "$PSScriptRoot\..\data\downloads\nuScenes-v1.0-trainval-keyframes-kaggle.zip",
    [string]$DataRoot = "$PSScriptRoot\..\data\nuscenes-trainval-keyframes",
    [string]$Python = "$PSScriptRoot\..\.venv-analysis\Scripts\python.exe",
    [switch]$Extract,
    [switch]$Validate
)

$ErrorActionPreference = 'Stop'
$ExpectedArchiveBytes = 33489285238L
$ExpectedExtractedBytes = 55GB
$ArchiveName = 'nuScenes-v1.0-trainval-keyframes-kaggle.zip'

function Get-FreeBytes([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetPathRoot($fullPath)
    return (Get-PSDrive -Name $root.TrimEnd('\').TrimEnd(':')).Free
}

if (-not $SourceArchive) {
    $SourceArchive = Get-ChildItem -LiteralPath 'G:\' -Directory | ForEach-Object {
        Join-Path $_.FullName "NuScenes DDI Research\01_Dataset\$ArchiveName"
    } | Where-Object {
        Test-Path -LiteralPath $_ -PathType Leaf
    } | Select-Object -First 1
}

if (-not $SourceArchive -or -not (Test-Path -LiteralPath $SourceArchive -PathType Leaf)) {
    throw "Source archive not found: $SourceArchive"
}

$source = Get-Item -LiteralPath $SourceArchive
if ($source.Length -ne $ExpectedArchiveBytes) {
    throw "Unexpected source size: $($source.Length) bytes; expected $ExpectedArchiveBytes bytes. Wait for the Drive upload to finish or replace the source."
}

$localArchiveFull = [System.IO.Path]::GetFullPath($LocalArchive)
$localArchiveDirectory = Split-Path -Parent $localArchiveFull
New-Item -ItemType Directory -Force -Path $localArchiveDirectory | Out-Null

$localIsComplete = (Test-Path -LiteralPath $localArchiveFull) -and
    ((Get-Item -LiteralPath $localArchiveFull).Length -eq $ExpectedArchiveBytes)

$requiredBytes = if ($localIsComplete) { 0L } else { $ExpectedArchiveBytes }
if ($Extract) {
    $requiredBytes += $ExpectedExtractedBytes
}

$freeBytes = Get-FreeBytes $localArchiveFull
if ($freeBytes -lt $requiredBytes) {
    throw "Insufficient local space. Need about $([math]::Ceiling($requiredBytes / 1GB)) GiB, available $([math]::Floor($freeBytes / 1GB)) GiB."
}

if (-not $localIsComplete) {
    Write-Host "Materializing archive from Google Drive..."
    $sourceDirectory = Split-Path -Parent $SourceArchive
    $sourceName = Split-Path -Leaf $SourceArchive
    & robocopy.exe $sourceDirectory $localArchiveDirectory $sourceName /J /Z /R:5 /W:10 /NP
    if ($LASTEXITCODE -gt 7) {
        throw "Robocopy failed with exit code $LASTEXITCODE."
    }
}

$local = Get-Item -LiteralPath $localArchiveFull
if ($local.Length -ne $ExpectedArchiveBytes) {
    throw "Local archive is incomplete: $($local.Length) bytes; expected $ExpectedArchiveBytes bytes."
}

Write-Host "Archive ready: $localArchiveFull ($($local.Length) bytes)"

if ($Extract) {
    $dataRootFull = [System.IO.Path]::GetFullPath($DataRoot)
    New-Item -ItemType Directory -Force -Path $dataRootFull | Out-Null
    Write-Host "Extracting keyframes to $dataRootFull..."
    & tar.exe -xf $localArchiveFull -C $dataRootFull
    if ($LASTEXITCODE -ne 0) {
        throw "Archive extraction failed with exit code $LASTEXITCODE."
    }
}

if ($Validate) {
    if (-not $Extract -and -not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
        throw "Data root does not exist. Use -Extract or provide -DataRoot."
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python environment not found: $Python"
    }

    $validationOutput = Join-Path ([System.IO.Path]::GetFullPath($DataRoot)) 'keyframe_validation.json'
    & $Python "$PSScriptRoot\validate_nuscenes_keyframes.py" `
        --dataroot ([System.IO.Path]::GetFullPath($DataRoot)) `
        --version v1.0-trainval `
        --output $validationOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Keyframe validation failed with exit code $LASTEXITCODE."
    }
    Write-Host "Validation report: $validationOutput"
}
