[CmdletBinding()]
param(
    [string]$SourceArchive = '',
    [string]$LocalArchive = '',
    [string]$DataRoot = '',
    [string]$Python = '',
    [switch]$Extract,
    [switch]$Validate
)

$ErrorActionPreference = 'Stop'
$ExpectedArchiveBytes = 33489285238L
$ExpectedExtractedBytes = 55GB
$ArchiveName = 'nuScenes-v1.0-trainval-keyframes-kaggle.zip'
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

if (-not $LocalArchive) {
    $LocalArchive = Join-Path $ProjectRoot "data\downloads\$ArchiveName"
}
if (-not $DataRoot) {
    $DataRoot = Join-Path $ProjectRoot 'data\nuscenes-trainval-keyframes'
}
if (-not $Python) {
    $Python = Join-Path $ProjectRoot '.venv-analysis\Scripts\python.exe'
}

function Get-FreeBytes([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $root = [System.IO.Path]::GetPathRoot($fullPath)
    return (Get-PSDrive -Name $root.TrimEnd('\').TrimEnd(':')).Free
}

function Test-ZipReadable([string]$Path) {
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & tar.exe -tf $Path 2>$null | Out-Null
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    return $exitCode -eq 0
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

$localIsComplete = $false
if ((Test-Path -LiteralPath $localArchiveFull) -and
    ((Get-Item -LiteralPath $localArchiveFull).Length -eq $ExpectedArchiveBytes)) {
    Write-Host 'Checking the existing local ZIP table...'
    $localIsComplete = Test-ZipReadable $localArchiveFull
    if (-not $localIsComplete) {
        Write-Host 'Removing an unreadable local archive before retrying...'
        [System.IO.File]::Delete($localArchiveFull)
    }
}

$requiredBytes = if ($localIsComplete) { 0L } else { $ExpectedArchiveBytes }
if ($Extract) {
    $requiredBytes += $ExpectedExtractedBytes
}

$freeBytes = Get-FreeBytes $localArchiveFull
if ($freeBytes -lt $requiredBytes) {
    throw "Insufficient local space. Need about $([math]::Ceiling($requiredBytes / 1GB)) GiB, available $([math]::Floor($freeBytes / 1GB)) GiB."
}

if (-not $localIsComplete) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python environment not found: $Python"
    }
    Write-Host "Materializing archive from Google Drive with resumable streaming..."
    & $Python "$PSScriptRoot\stream_copy_file.py" `
        --source $SourceArchive `
        --destination $localArchiveFull `
        --expected-bytes $ExpectedArchiveBytes
    if ($LASTEXITCODE -ne 0) {
        throw "Streaming copy failed with exit code $LASTEXITCODE."
    }
}

$local = Get-Item -LiteralPath $localArchiveFull
if ($local.Length -ne $ExpectedArchiveBytes) {
    throw "Local archive is incomplete: $($local.Length) bytes; expected $ExpectedArchiveBytes bytes."
}

Write-Host "Archive ready: $localArchiveFull ($($local.Length) bytes)"

Write-Host 'Checking the completed local ZIP table...'
if (-not (Test-ZipReadable $localArchiveFull)) {
    throw 'The completed local ZIP is unreadable.'
}

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

    $validationRoot = [System.IO.Path]::GetFullPath($DataRoot)
    $nestedRoot = Join-Path $validationRoot 'v1.0-trainval'
    if (Test-Path -LiteralPath (Join-Path $nestedRoot 'v1.0-trainval\category.json')) {
        $validationRoot = $nestedRoot
    }
    $validationOutput = Join-Path ([System.IO.Path]::GetFullPath($DataRoot)) 'keyframe_validation.json'
    & $Python "$PSScriptRoot\validate_nuscenes_keyframes.py" `
        --dataroot $validationRoot `
        --version v1.0-trainval `
        --output $validationOutput
    if ($LASTEXITCODE -ne 0) {
        throw "Keyframe validation failed with exit code $LASTEXITCODE."
    }
    Write-Host "Validation report: $validationOutput"
}
