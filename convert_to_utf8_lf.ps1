# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

param(
    [string]$ConfigPath = "convert_to_utf8_lf_config_whitelist.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$p) {
    if ([string]::IsNullOrWhiteSpace($p)) { return $null }
    $root = (Get-Location).Path
    return [IO.Path]::GetFullPath((Join-Path $root $p))
}

function Read-JsonFile([string]$path) {
    if (-not (Test-Path $path)) { throw "配置文件未找到: $path" }
    $raw = Get-Content -Raw -Encoding UTF8 $path
    return $raw | ConvertFrom-Json
}

function Is-BinaryByExt([string]$path, [string[]]$binExts) {
    $ext = [IO.Path]::GetExtension($path).ToLowerInvariant()
    return $binExts -contains $ext
}

function Is-TextByExt([string]$path, [string[]]$txtExts) {
    $ext = [IO.Path]::GetExtension($path).ToLowerInvariant()
    return $txtExts -contains $ext
}

function Has-BOM([byte[]]$bytes) {
    return ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF)
}

function Convert-ToUtf8Lf([string]$file) {
    $bytes = [IO.File]::ReadAllBytes($file)
    $hasBom = Has-BOM $bytes
    $utf8 = [Text.UTF8Encoding]::new($false)
    $text = [Text.Encoding]::UTF8.GetString($bytes)
    $orig = $text
    $crlfCount = ([regex]::Matches($text, "\r\n")).Count
    $text = $text -replace "\r\n", "`n"
    $text = $text -replace "\r(?!\n)", "`n"
    $changed = $hasBom -or ($crlfCount -gt 0)
    if (-not $changed -and $orig -eq $text) {
        return [pscustomobject]@{ Changed=$false; CrLfToLf=0; BomRemoved=$false }
    }
    [IO.File]::WriteAllText($file, $text, $utf8)
    return [pscustomobject]@{ Changed=$true; CrLfToLf=$crlfCount; BomRemoved=$hasBom }
}

Write-Host "[convert_to_utf8_lf] 读取配置: $ConfigPath"
$cfg = Read-JsonFile -path $ConfigPath
$includeDirs  = @() + ($cfg.include  | ForEach-Object { $_.ToString() })
$includeFiles = @() + ($cfg.include_files | ForEach-Object { $_.ToString() })
$textExts = if ($cfg.PSObject.Properties.Name -contains 'text_extensions') { @() + $cfg.text_extensions } else { @() }
$binExts  = if ($cfg.PSObject.Properties.Name -contains 'binary_extensions') { @() + $cfg.binary_extensions } else { @() }

# 规范化路径并去重
$fileSet = New-Object System.Collections.Generic.HashSet[string]
foreach ($f in $includeFiles) {
    $pp = Resolve-RepoPath $f
    if ($pp -and (Test-Path $pp)) { [void]$fileSet.Add($pp) }
}
foreach ($d in $includeDirs) {
    $pd = Resolve-RepoPath $d
    if (-not $pd -or -not (Test-Path $pd)) { continue }
    Get-ChildItem -Path $pd -Recurse -File | ForEach-Object {
        $full = $_.FullName
        if ($full -match "\\\.git\\" ) { return }
        if ((Is-BinaryByExt $full $binExts)) { return }
        if ($textExts.Count -gt 0) {
            if (-not (Is-TextByExt $full $textExts)) { return }
        }
        [void]$fileSet.Add($full)
    }
}

$total = 0; $changed = 0; $skipped = 0
foreach ($p in $fileSet) {
    try {
        $total++
        $res = Convert-ToUtf8Lf -file $p
        if ($res.Changed) {
            $changed++
            Write-Host ("[changed] " + $p + " crlf->lf=" + $res.CrLfToLf + " bom_removed=" + $res.BomRemoved)
        } else {
            $skipped++
        }
    } catch {
        Write-Warning ("[error] 转换失败: " + $p + " => " + $_.Exception.Message)
    }
}

Write-Host ("[summary] total=" + $total + " changed=" + $changed + " skipped=" + $skipped)

