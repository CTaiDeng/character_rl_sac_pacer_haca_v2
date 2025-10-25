#!/usr/bin/env pwsh
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

<#
Checks staged files (or all tracked files) for UTF-8 (no BOM) + LF line endings.
Skips files marked binary or -text via .gitattributes and the read-only docs/kernel_reference/.

Usage:
  pwsh ./scripts/check_utf8_nobom_lf.ps1              # check staged only
  pwsh ./scripts/check_utf8_nobom_lf.ps1 -All         # check all tracked files

Environment:
  SKIP_UTF8LF_CHECK=1   # to bypass pre-commit check (temporary escape hatch)
#>

[CmdletBinding()]
param(
  [switch]$All
)

$ErrorActionPreference = 'Stop'

function Resolve-GitRoot {
  $root = (git rev-parse --show-toplevel 2>$null)
  if (-not $root) { throw "Not a git repository." }
  return $root
}

function Get-StagedOrTrackedPaths {
  param([switch]$All)
  if ($All) {
    $list = git ls-files
  } else {
    $list = git diff --cached --name-only --diff-filter=ACMRT
  }
  return @($list | Where-Object { $_ -and (Test-Path $_) })
}

function Get-GitAttrMap {
  param([string]$Path)
  # Query only attributes we care about
  $attrs = git check-attr binary text eol working-tree-encoding -- "$Path" 2>$null
  $map = @{}
  foreach ($line in $attrs) {
    # Format: <path>: <attr>: <value>
    $parts = $line -split ":\s+", 3
    if ($parts.Count -ge 3) { $map[$parts[1].Trim()] = $parts[2].Trim() }
  }
  return $map
}

function Test-IsTextByAttr {
  param($attrMap)
  if ($attrMap.ContainsKey('binary') -and $attrMap['binary'] -eq 'set') { return $false }
  if ($attrMap.ContainsKey('text')) {
    switch ($attrMap['text']) {
      'set' { return $true }
      'unset' { return $false }
      default { return $true } # unspecified -> treat as text
    }
  }
  return $true
}

function Test-FileUtf8NoBomLf {
  param([string]$Path)
  $bytes = [System.IO.File]::ReadAllBytes($Path)
  # BOM check: EF BB BF
  if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
    return @{ Ok=$false; Reason='Has UTF-8 BOM' }
  }
  # CRLF check
  for ($i=0; $i -lt $bytes.Length-1; $i++) {
    if ($bytes[$i] -eq 0x0D -and $bytes[$i+1] -eq 0x0A) {
      return @{ Ok=$false; Reason='Contains CRLF (\r\n)' }
    }
  }
  # Encoding check
  try {
    $utf8 = New-Object System.Text.UTF8Encoding($true)
    [void]$utf8.GetString($bytes)
  } catch {
    return @{ Ok=$false; Reason='Not valid UTF-8' }
  }
  return @{ Ok=$true; Reason='' }
}

if ($env:SKIP_UTF8LF_CHECK) {
  Write-Host "[check_utf8_nobom_lf] Skipped due to SKIP_UTF8LF_CHECK." -ForegroundColor Yellow
  exit 0
}

$root = Resolve-GitRoot
Set-Location $root

$paths = Get-StagedOrTrackedPaths -All:$All
if (-not $paths -or $paths.Count -eq 0) {
  Write-Host "[check_utf8_nobom_lf] No files to check." -ForegroundColor DarkGray
  exit 0
}

$violations = @()
foreach ($p in $paths) {
  $pNorm = $p -replace '\\','/'
  if ($pNorm.StartsWith('docs/kernel_reference/')) { continue }
  $attr = Get-GitAttrMap -Path $p
  if (-not (Test-IsTextByAttr -attrMap $attr)) { continue }

  # If .gitattributes explicitly sets eol=crlf for this path, we still report
  # a violation since the project enforces LF globally.
  $res = Test-FileUtf8NoBomLf -Path $p
  if (-not $res.Ok) {
    $violations += [pscustomobject]@{ Path=$p; Reason=$res.Reason }
  }
}

if ($violations.Count -gt 0) {
  Write-Host "✖ UTF-8 (no BOM) + LF check failed for:" -ForegroundColor Red
  foreach ($v in $violations) {
    Write-Host (" - {0}  => {1}" -f $v.Path, $v.Reason) -ForegroundColor Red
  }
  Write-Host "" 
  Write-Host "Fix tips:" -ForegroundColor Yellow
  Write-Host "  - Ensure your editor respects .editorconfig (UTF-8 + LF)."
  Write-Host "  - Convert files to LF and remove BOM, then re-add to index:"
  Write-Host "      git rm --cached -r ." -ForegroundColor DarkGray
  Write-Host "      git add --renormalize ." -ForegroundColor DarkGray
  Write-Host "      git add <files>" -ForegroundColor DarkGray
  exit 1
}

Write-Host "✔ UTF-8 (no BOM) + LF check passed." -ForegroundColor Green
exit 0
