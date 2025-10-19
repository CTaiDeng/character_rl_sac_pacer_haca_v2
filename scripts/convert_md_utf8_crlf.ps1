#!/usr/bin/env pwsh
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

<#
.SYNOPSIS
  将目标目录（含子目录）内所有 *.md 文档规范为 UTF-8 编码与 CRLF 行尾。

.DESCRIPTION
  - 仅处理扩展名为 .md 的文本文件；自动跳过可疑二进制（含 NUL）。
  - 默认跳过规则：
      - 任意名为 INDEX.md 的文件（避免批量影响自动索引）
      - LICENSE、src/docs/LICENSE.md、src/kernel_reference/LICENSE.md、src/full_reference/LICENSE.md
      - src/kernel_reference/KERNEL_REFERENCE_README.md
    但以下路径作为“强制包含”不受上述跳过规则限制：
      - src/kernel_reference/INDEX.md
      - src/kernel_reference/KERNEL_REFERENCE_README.md
      - src/kernel_reference/LICENSE.md
      - src/sub_projects_docs/LICENSE.md
      - src/sub_projects_docs/README.md
  - 若文件已满足 UTF-8 + CRLF，将跳过不改动，避免无意义写入。

.PARAMETER Path
  目标根目录（默认：当前目录）。

.PARAMETER DryRun
  干跑模式，仅打印将执行的操作，不改动文件。

.EXAMPLE
  pwsh -NoLogo -File script/convert_md_utf8_crlf.ps1 -Path .

.EXAMPLE
  pwsh -NoLogo -File script/convert_md_utf8_crlf.ps1 -Path docs -DryRun
#>

[CmdletBinding()]
param(
  [Parameter(Position=0)][string]$Path = '.',
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

try {
  $root = (Resolve-Path -LiteralPath $Path).Path
} catch {
  Write-Error "路径不存在：$Path"; exit 1
}

# 可选：探测 Git 仓库根，便于与强制包含列表匹配仓库相对路径
$repoTop = $null
try {
  $gitTop = & git rev-parse --show-toplevel 2>$null
  if ($LASTEXITCODE -eq 0 -and $gitTop) { $repoTop = ($gitTop.Trim()) }
} catch { $repoTop = $null }

# 注册代码页（以便 GBK/GB18030 等回退解码可用）
[System.Text.Encoding]::RegisterProvider([System.Text.CodePagesEncodingProvider]::Instance) | Out-Null

$utf8NoBom   = New-Object System.Text.UTF8Encoding($false)
$utf8Strict  = New-Object System.Text.UTF8Encoding($false, $true)

function Get-RelativePath([string]$base, [string]$full) {
  try { return [System.IO.Path]::GetRelativePath($base, $full) } catch { return $full }
}

$skipByName = @('INDEX.md')
$skipExact  = @(
  'LICENSE',
  'src/docs/LICENSE.md',
  'src/kernel_reference/LICENSE.md',
  'src/full_reference/LICENSE.md',
  'src/kernel_reference/KERNEL_REFERENCE_README.md'
)

$includeOverride = @(
  'src/kernel_reference/INDEX.md',
  'src/kernel_reference/KERNEL_REFERENCE_README.md',
  'src/kernel_reference/LICENSE.md',
  'src/sub_projects_docs/LICENSE.md',
  'src/sub_projects_docs/README.md'
)

$total=0; $changed=0; $skipped=0; $errors=0

Write-Host "[md2utf8crlf] root: $root" -ForegroundColor Cyan

$files = Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.md' -ErrorAction SilentlyContinue
foreach ($f in $files) {
  $total++
  $rel = (Get-RelativePath $root $f.FullName).Replace('\\','/')

  $relRepo = $null
  if ($repoTop) {
    try { $relRepo = ([System.IO.Path]::GetRelativePath($repoTop, $f.FullName)).Replace('\\','/') } catch { $relRepo = $null }
  }

  $shouldSkip = $false
  $reason = $null
  $absUnix = ([System.IO.Path]::GetFullPath($f.FullName)).Replace('\\','/')
  $absLower = $absUnix.ToLowerInvariant()
  $inOverride = (
      ($includeOverride -contains $rel) -or
      ($relRepo -and ($includeOverride -contains $relRepo)) -or
      ($absLower -like '*/src/kernel_reference/index.md') -or
      ($absLower -like '*/src/kernel_reference/kernel_reference_readme.md') -or
      ($absLower -like '*/src/kernel_reference/license.md') -or
      ($absLower -like '*/src/sub_projects_docs/license.md') -or
      ($absLower -like '*/src/sub_projects_docs/readme.md') -or
      ( ($root.Replace('\\','/').ToLowerInvariant() -like '*/src/kernel_reference*') -and ($f.Name -eq 'INDEX.md') )
    )
  if ((($skipExact -contains $rel) -or ($relRepo -and ($skipExact -contains $relRepo))) -and -not $inOverride) {
    $shouldSkip = $true; $reason = 'policy'
  } elseif (($skipByName -contains $f.Name) -and -not $inOverride) {
    $shouldSkip = $true; $reason = 'name rule'
  }
  if ($shouldSkip) { $skipped++; Write-Host "[skip] $rel ($reason)"; continue }

  try {
    $bytes = [System.IO.File]::ReadAllBytes($f.FullName)
  } catch {
    $errors++; Write-Warning "读取失败：$rel"; continue
  }

  # 粗略二进制判断：包含 NUL 则跳过
  if (0 -in $bytes) { $skipped++; Write-Host "[skip] $rel (binary-like)"; continue }

  $text = $null
  $validUtf8 = $true
  try {
    $null = $utf8Strict.GetString($bytes)  # 仅用于校验
  } catch { $validUtf8 = $false }

  if ($validUtf8) {
    $text = [System.Text.Encoding]::UTF8.GetString($bytes)
  } else {
    # 宽容回退：优先 GB18030，再退系统默认
    try {
      $gb = [System.Text.Encoding]::GetEncoding('GB18030')
      $text = $gb.GetString($bytes)
    } catch {
      $text = [System.Text.Encoding]::Default.GetString($bytes)
    }
  }

  # 统一行为：先归一化到 LF，再转 CRLF
  $norm = $text -replace "`r`n", "`n"
  $norm = $norm -replace "`r", ""
  $norm = $norm -replace "`n", "`r`n"

  $new = $utf8NoBom.GetBytes($norm)

  # 若内容无差异则跳过写入
  $same = $false
  if ($new.Length -eq $bytes.Length) {
    $same = $true
    for ($i=0; $i -lt $new.Length; $i++) { if ($new[$i] -ne $bytes[$i]) { $same=$false; break } }
  }

  if ($same) {
    $skipped++; Write-Host "[keep] $rel"; continue
  }

  if ($DryRun) {
    Write-Host "[plan] $rel -> UTF-8 + CRLF" -ForegroundColor Yellow
    continue
  }

  try {
    [System.IO.File]::WriteAllBytes($f.FullName, $new)
    $changed++
    Write-Host "[fix]  $rel" -ForegroundColor Green
  } catch {
    $errors++; Write-Warning "写入失败：$rel"
  }
}

Write-Host "[summary] total=$total, changed=$changed, skipped=$skipped, errors=$errors" -ForegroundColor Cyan
