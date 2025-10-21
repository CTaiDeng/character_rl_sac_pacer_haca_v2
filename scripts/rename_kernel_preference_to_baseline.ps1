# Copyright (C) 2025 GaoZheng
# SPDX-License-Identifier: GPL-3.0-only
# This file is part of this project.
# Licensed under the GNU General Public License version 3.
# See https://www.gnu.org/licenses/gpl-3.0.html for details.

[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [string]$Root = 'src/kernel_reference',
  # 默认仅处理 Markdown，可传入 '*' 处理所有扩展名
  [string[]]$Extensions = @('.md'),
  [switch]$Recurse = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

function Get-Rel([string]$p){
  try {
    $abs = (Resolve-Path -LiteralPath $p).Path
    $root = (Resolve-Path '.').Path
    return ($abs.Substring($root.Length).TrimStart([char]92,'/') -replace '\\','/')
  } catch { return $p }
}

$resolvedRoot = Resolve-Path -LiteralPath $Root -ErrorAction Stop
$files = if ($Recurse) {
  Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -ErrorAction Stop
} else {
  Get-ChildItem -LiteralPath $resolvedRoot -File -ErrorAction Stop
}

$excludeExact = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
@('INDEX.md','KERNEL_REFERENCE_README.md','LICENSE.md','LICENSE') | ForEach-Object { [void]$excludeExact.Add($_) }

$extSet = $null
if ($Extensions -and $Extensions.Count -gt 0 -and -not ($Extensions -contains '*')){
  $extSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach($e in $Extensions){ [void]$extSet.Add($e) }
}

$scanned=0; $candidates=0; $renamed=0; $skipped=0; $conflicts=0
foreach($f in $files){
  $scanned++
  if ($excludeExact.Contains($f.Name)) { $skipped++; continue }
  if ($extSet -ne $null) {
    $ext = [IO.Path]::GetExtension($f.Name)
    if (-not $extSet.Contains($ext)) { continue }
  }
  if ($f.Name -notlike '*偏好*') { continue }

  $candidates++
  $newName = ($f.Name -replace '偏好','基准')
  if ($newName -eq $f.Name) { continue }
  $target = Join-Path $f.DirectoryName $newName
  if (Test-Path -LiteralPath $target) {
    Write-Warning ("目标已存在，跳过：{0} -> {1}" -f (Get-Rel $f.FullName), (Get-Rel $target))
    $conflicts++; continue
  }

  $rel = Get-Rel $f.FullName
  if ($PSCmdlet.ShouldProcess($rel, "Rename to '$newName'")){
    Rename-Item -LiteralPath $f.FullName -NewName $newName -ErrorAction Stop
    Write-Host ("Renamed: {0} -> {1}" -f $rel, (Get-Rel $target))
    $renamed++
  }
}

Write-Host ("Summary => scanned={0} candidates={1} renamed={2} conflicts={3} skipped={4}" -f $scanned,$candidates,$renamed,$conflicts,$skipped)

# 使用示例：
# 干跑预览：
#   pwsh -NoLogo -File script/rename_kernel_preference_to_baseline.ps1 -WhatIf
# 实际执行：
#   pwsh -NoLogo -File script/rename_kernel_preference_to_baseline.ps1
# 处理所有扩展名：
#   pwsh -NoLogo -File script/rename_kernel_preference_to_baseline.ps1 -Extensions '*'
# 重建索引（推荐在重命名后执行）：
#   pwsh -NoLogo -File script/kernel_reference_build_index.ps1 -MaxChars 500

