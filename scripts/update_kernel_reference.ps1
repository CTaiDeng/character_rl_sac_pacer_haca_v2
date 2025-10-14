# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

Param(
    [string]$RepoUrl = "https://github.com/CTaiDeng/open_meta_mathematical_theory.git",
    [string]$Branch = "master"
)

$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    if ($PSScriptRoot) {
        return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    } else {
        return (Get-Location).Path
    }
}

function Ensure-Git {
    try {
        git --version | Out-Null
    } catch {
        throw "未检测到 Git，请先安装 Git 并确保可在 PATH 中调用。"
    }
}

function Clear-Directory([string]$PathToClear) {
    if (-not (Test-Path -LiteralPath $PathToClear)) {
        New-Item -ItemType Directory -Path $PathToClear -Force | Out-Null
        return
    }
    Get-ChildItem -LiteralPath $PathToClear -Force | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction Stop
    }
}

function Copy-DirectoryContent([string]$FromDir, [string]$ToDir) {
    if (-not (Test-Path -LiteralPath $FromDir)) {
        throw "源目录不存在：$FromDir"
    }
    if (-not (Test-Path -LiteralPath $ToDir)) {
        New-Item -ItemType Directory -Path $ToDir -Force | Out-Null
    }
    Get-ChildItem -LiteralPath $FromDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $ToDir -Recurse -Force -ErrorAction Stop
    }
}

function Main {
    $root = Resolve-RepoRoot
    $destDir = Join-Path $root "docs/kernel_reference"
    $workDir = Join-Path $root "out/kernel_reference_only"

    Ensure-Git

    if (Test-Path -LiteralPath $workDir) {
        Remove-Item -LiteralPath $workDir -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath (Split-Path -Parent $workDir))) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $workDir) -Force | Out-Null
    }

    if (-not (Test-Path -LiteralPath (Split-Path -Parent $destDir))) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $destDir) -Force | Out-Null
    }

    if (-not (Test-Path -LiteralPath $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    Clear-Directory -PathToClear $destDir

    git clone --sparse --filter=blob:none --branch $Branch --single-branch $RepoUrl $workDir

    & git -C $workDir sparse-checkout set src/kernel_reference

    $srcDir = Join-Path $workDir "src/kernel_reference"
    if (-not (Test-Path -LiteralPath $srcDir)) {
        throw "稀疏签出后未找到目录：$srcDir"
    }

    Copy-DirectoryContent -FromDir $srcDir -ToDir $destDir

    if (Test-Path -LiteralPath $workDir) {
        Remove-Item -LiteralPath $workDir -Recurse -Force
    }

    Write-Host "已更新 docs/kernel_reference 并清理临时目录。"
}

Main

