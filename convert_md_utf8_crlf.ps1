#!/usr/bin/env pwsh
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

<#
.SYNOPSIS
  将 Markdown 规范化为 UTF-8+CRLF，或扫描未被 .gitignore 排除的源码/文档，检测“中文乱码”并导出 CSV。

.DESCRIPTION
  模式一（ConvertMd，默认）：
    - 递归处理目标目录下扩展名为 .md 的文本文件；
    - 自动规整换行为 CRLF，重写为 UTF-8（无 BOM）；
    - 跳过二进制-like（含 NUL）文件。

  模式二（ScanGarbled）：
    - 通过 git 列举“未被 .gitignore 排除”的文件（含已跟踪与未跟踪但未忽略）；
    - 仅扫描指定扩展（默认：源码 .py 与文档 .md/.txt/.rst）；
    - 以多重启发式检测“中文乱码”（无效 UTF-8、替换符、典型 UTF‑8→Latin-1/GBK 乱码片段等）；
    - 在仓库根目录导出 CSV 报告，列出相对路径与命中原因。

.PARAMETER Path
  目标目录（ConvertMd 模式用）；默认当前目录。

.PARAMETER Mode
  处理模式：ConvertMd | ScanGarbled（默认：ConvertMd）。

.PARAMETER CodeExts
  源码后缀（ScanGarbled 用），默认：.py。

.PARAMETER DocExts
  文档后缀（ScanGarbled 用），默认：.md, .txt, .rst。

.PARAMETER OutputCsv
  CSV 输出路径（ScanGarbled 用）；默认写入仓库根：garbled_files.csv。

.PARAMETER DryRun
  演示模式（仅 ConvertMd 模式有效），只打印计划，不写文件。

.EXAMPLE
  pwsh -NoLogo -File scripts/convert_md_utf8_crlf.ps1 -Path docs

.EXAMPLE
  pwsh -NoLogo -File scripts/convert_md_utf8_crlf.ps1 -Mode ScanGarbled
#>

[CmdletBinding()]
param(
  [Parameter(Position=0)][string]$Path = '.',
  [ValidateSet('ConvertMd','ScanGarbled','ConvertText','FixGarbled')][string]$Mode = 'ConvertMd',
  [string[]]$CodeExts = @('.py'),
  [string[]]$DocExts  = @('.md', '.txt', '.rst'),
  [string]$OutputCsv = '',
  [string]$InputCsv = '',
  [switch]$IncludeDocs,
  [switch]$Backup,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# 注册附加编码（GBK/GB18030 等）
[System.Text.Encoding]::RegisterProvider([System.Text.CodePagesEncodingProvider]::Instance) | Out-Null

$utf8NoBom   = New-Object System.Text.UTF8Encoding($false)
$utf8Strict  = New-Object System.Text.UTF8Encoding($false, $true)

function Get-RelativePath([string]$base, [string]$full) {
  try { return [System.IO.Path]::GetRelativePath($base, $full) } catch { return $full }
}

function Get-RepoRoot() {
  try {
    $top = & git rev-parse --show-toplevel 2>$null
    if ($LASTEXITCODE -eq 0 -and $top) { return $top.Trim() }
  } catch { }
  return $null
}

function Normalize-CRLF([string]$text) {
  $norm = $text -replace "`r`n", "`n"
  $norm = $norm -replace "`r", ""
  $norm = $norm -replace "`n", "`r`n"
  return $norm
}

function Test-ValidUtf8([byte[]]$bytes) {
  try { $null = $utf8Strict.GetString($bytes); return $true } catch { return $false }
}

function Detect-GarbledChinese([string]$text) {
  # 检测 U+FFFD（替换符）或常见 UTF-8→Latin-1/GBK 乱码模式
  $reasons = New-Object System.Collections.Generic.List[string]

  # 更稳健的替换符检测（直接按字符统计）
  $rep = [string][char]0xFFFD
  $repHits = [regex]::Matches($text, [regex]::Escape($rep)).Count
  if ($repHits -gt 0) { $reasons.Add("ReplacementChar:$repHits") }

  # 典型 Latin-1（Windows-1252）误解码产生的片段（如：Ã, Â, â€˜/â€™/â€œ/â€ 等）
  $latin1Pattern = '(Ã.|Â.|â€[˜™œžšº¹º“”•]|â€“|â€”|â€¦|ï¼|ï¿)'
  $latin1Hits = [regex]::Matches($text, $latin1Pattern).Count
  if ($latin1Hits -ge 3) { $reasons.Add("MojibakeLatin1:$latin1Hits") }

  # 典型 GBK/GB2312 相关的错位常见字（扩充更常见的错误字集）
  $gbkPattern = '(锟|烫|浣|鈥|鏂|纭|绛|涓|绯|绁|鎵|鍙|闂|鑷|灏|骞|鍔|鍗|鍏|鍚|锛|脳)'
  $gbkHits = [regex]::Matches($text, $gbkPattern).Count
  if ($gbkHits -ge 2) { $reasons.Add("MojibakeGBK:$gbkHits") }

  # 更广义的 CJK 错位热点（MojibakeCJK），降低阈值防漏报
  $cjkHotPattern = '(鎵|鏍|鐢|鐧|鐙|閫|鎸|缁|缂|缃|缇|缈|缍|缎|缐)'
  $cjkHotHits = [regex]::Matches($text, $cjkHotPattern).Count
  if ($cjkHotHits -ge 2) { $reasons.Add("MojibakeCJK:$cjkHotHits") }

  # 高频西欧重音字母（äåæçèéêëìíîïñòóôõöøùúûüýþÿ）重复出现也多为乱码迹象
  $accentPattern = '[äåæçèéêëìíîïñòóôõöøùúûüýþÿ]'
  $accentHits = [regex]::Matches($text, $accentPattern).Count
  if ($accentHits -ge 6) { $reasons.Add("AccentedBurst:$accentHits") }

  return $reasons
}

function Detect-GarbledChinese2([string]$text) {
  # 编码无关的启发式：避免把具体非 ASCII 字符写进正则，降低脚本自体编码造成的失真。
  $reasons = New-Object System.Collections.Generic.List[string]
  if (-not $text) { return $reasons }

  # 1) Unicode 替换符（U+FFFD）
  $rep = [string][char]0xFFFD
  $repHits = [regex]::Matches($text, [regex]::Escape($rep)).Count
  if ($repHits -gt 0) { $reasons.Add("ReplacementChar:$repHits") }

  # 2) Latin-1/Windows-1252 典型乱码：可疑码点（C2/C3/E2/EF/A0/AD）数量
  $latinSuspects = 0
  foreach ($ch in $text.ToCharArray()) {
    $code = [int][char]$ch
    if ($code -in 0x00C2,0x00C3,0x00E2,0x00EF,0x00A0,0x00AD) { $latinSuspects++ }
  }
  if ($latinSuspects -ge 3) { $reasons.Add("Latin1Burst:$latinSuspects") }

  # 3) UTF-8→GBK 常见“错字集”命中数（以码点列举，避免直接嵌字）
  #    锛(951B) 锟(951F) 链(94FE) 銆(92C6)
  #    绛(7EDB) 绯(7EEF) 绮(7EEE) 绱(7EF1) 绾(7EFE) 缁(7F01) 缂(7F02)
  #    鎵(9395) 鏍(93CD/93D6/93D9 常见)
  $utf8AsGbkCodes = @(
    0x951B,0x951F,0x94FE,0x92C6,
    0x7EDB,0x7EEF,0x7EEE,0x7EF1,0x7EFE,0x7F01,0x7F02,
    0x9395,0x93CD,0x93D6,0x93D9
  )
  $utf8AsGbkHits = 0
  foreach ($ch in $text.ToCharArray()) {
    if ($utf8AsGbkCodes -contains ([int][char]$ch)) { $utf8AsGbkHits++ }
  }
  if ($utf8AsGbkHits -ge 6) { $reasons.Add("UTF8asGBKCommon:$utf8AsGbkHits") }

  # 4) 重音拉丁字母爆发（Latin-1 Supplement 区间 0x00C0–0x00FF）
  $accentHits = 0
  foreach ($ch in $text.ToCharArray()) {
    $code = [int][char]$ch
    if ($code -ge 0x00C0 -and $code -le 0x00FF) { $accentHits++ }
  }
  if ($accentHits -ge 6) { $reasons.Add("AccentedBurst:$accentHits") }

  return $reasons
}

function Get-CjkRatio([string]$text) {
  if (-not $text) { return 0.0 }
  $total = $text.Length
  $cjk = 0
  foreach ($ch in $text.ToCharArray()) {
    $code = [int][char]$ch
    # CJK Unified Ideographs ranges + 常见中文全角标点
    if ((($code -ge 0x4E00 -and $code -le 0x9FFF) -or ($code -ge 0x3400 -and $code -le 0x4DBF) -or ($code -ge 0x20000 -and $code -le 0x2A6DF) -or ($code -ge 0x2A700 -and $code -le 0x2B73F) -or ($code -ge 0x2B740 -and $code -le 0x2B81F) -or ($code -ge 0x2B820 -and $code -le 0x2CEAF)) -or ($code -in 0x3001,0x3002,0xFF0C,0xFF1A,0xFF1B,0xFF1F,0xFF01)) {
      $cjk++
    }
  }
  return [double]$cjk / [double][math]::Max(1,$total)
}

function Get-MojibakeScore([string]$text) {
  $reasons = Detect-GarbledChinese2 $text
  $score = 0
  foreach ($r in $reasons) {
    if ($r -match ':(\d+)$') { $score += [int]$matches[1] } else { $score += 10 }
  }
  return [pscustomobject]@{
    Score    = $score
    Reasons  = $reasons
    CjkRatio = (Get-CjkRatio $text)
  }
}

switch ($Mode) {
  'ConvertMd' {
    try {
      $root = (Resolve-Path -LiteralPath $Path).Path
    } catch {
      Write-Error "路径不存在: $Path"; exit 1
    }

    $repoTop = Get-RepoRoot
    $total=0; $changed=0; $skipped=0; $errors=0
    Write-Host "[md2utf8crlf] root: $root" -ForegroundColor Cyan

    $files = Get-ChildItem -LiteralPath $root -Recurse -File -Filter '*.md' -ErrorAction SilentlyContinue
    foreach ($f in $files) {
      $total++
      $rel = (Get-RelativePath $root $f.FullName).Replace('\\','/')

      try { $bytes = [System.IO.File]::ReadAllBytes($f.FullName) } catch { $errors++; Write-Warning "读取失败: $rel"; continue }
      if (0 -in $bytes) { $skipped++; Write-Host "[skip] $rel (binary-like)"; continue }

      $validUtf8 = Test-ValidUtf8 $bytes
      if ($validUtf8) {
        $text = [System.Text.Encoding]::UTF8.GetString($bytes)
      } else {
        try { $text = [System.Text.Encoding]::GetEncoding('GB18030').GetString($bytes) } catch { $text = [System.Text.Encoding]::Default.GetString($bytes) }
      }

      $norm = Normalize-CRLF $text
      $new  = $utf8NoBom.GetBytes($norm)

      $same = ($new.Length -eq $bytes.Length)
      if ($same) {
        for ($i=0; $i -lt $new.Length; $i++) { if ($new[$i] -ne $bytes[$i]) { $same=$false; break } }
      }
      if ($same) { $skipped++; Write-Host "[keep] $rel"; continue }

      if ($DryRun) { Write-Host "[plan] $rel -> UTF-8 + CRLF" -ForegroundColor Yellow; continue }

      try { [System.IO.File]::WriteAllBytes($f.FullName, $new); $changed++; Write-Host "[fix]  $rel" -ForegroundColor Green }
      catch { $errors++; Write-Warning "写入失败: $rel" }
    }

    Write-Host "[summary] total=$total, changed=$changed, skipped=$skipped, errors=$errors" -ForegroundColor Cyan
  }

  'ScanGarbled' {
    $repoTop = Get-RepoRoot
    if (-not $repoTop) { Write-Error '未检测到 Git 仓库，无法依据 .gitignore 过滤。'; exit 1 }

    # 统一小写扩展集合
    $scanSet = New-Object System.Collections.Generic.HashSet[string] ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($e in $CodeExts + $DocExts) {
      if (-not $e) { continue }
      $ext = $e.StartsWith('.') ? $e : ('.' + $e)
      $null = $scanSet.Add($ext)
    }

    # 列出未被 .gitignore 排除的文件（含已跟踪与未忽略的未跟踪）
    $paths = & git -C $repoTop ls-files -co --exclude-standard -z 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Error 'git ls-files 执行失败'; exit 1 }
    $files = @()
    if ($paths) {
      $parts = $paths -split "`0"
      foreach ($p in $parts) { if ($p) { $full = Join-Path $repoTop $p; if (Test-Path -LiteralPath $full) { $files += $full } } }
    }

    $results = New-Object System.Collections.Generic.List[object]
    $scanned=0; $flagged=0; $skipped=0

    foreach ($f in $files) {
      $ext = [System.IO.Path]::GetExtension($f)
      if (-not $scanSet.Contains($ext)) { continue }

      $scanned++
      $rel = [System.IO.Path]::GetRelativePath($repoTop, $f).Replace('\\','/')

      try { $bytes = [System.IO.File]::ReadAllBytes($f) } catch { $skipped++; Write-Host "[skip] $rel (read-failed)"; continue }
      if (0 -in $bytes) { $skipped++; continue } # binary-like

      $isUtf8 = Test-ValidUtf8 $bytes
      $text = $null
      if ($isUtf8) {
        $text = [System.Text.Encoding]::UTF8.GetString($bytes)
      } else {
        # 非 UTF-8 直接记为问题，同时也尝试解码以便后续模式匹配
        $text = try { [System.Text.Encoding]::GetEncoding('GB18030').GetString($bytes) } catch { [System.Text.Encoding]::Default.GetString($bytes) }
      }

      $reasons = New-Object System.Collections.Generic.List[string]
      if (-not $isUtf8) { $reasons.Add('InvalidUTF8') }
      $more = Detect-GarbledChinese2 $text
      foreach ($r in $more) { $reasons.Add($r) }

      if ($reasons.Count -gt 0) {
        $flagged++
        $results.Add([pscustomobject]@{
          Path    = $rel
          Reasons = ($reasons -join '|')
        })
        Write-Host "[hit]  $rel -> $($reasons -join ',')" -ForegroundColor Yellow
      }
    }

    if (-not $OutputCsv) { $OutputCsv = Join-Path $repoTop 'garbled_files.csv' }
    $dir = Split-Path -Parent $OutputCsv
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

    # 导出 CSV（UTF-8，CRLF 由平台与 .gitattributes 保证）
    if (Test-Path -LiteralPath $OutputCsv) {
      try { Remove-Item -LiteralPath $OutputCsv -Force -ErrorAction SilentlyContinue } catch { }
    }
    $results | Sort-Object Path -Unique | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding utf8
    Write-Host "[scan-summary] scanned=$scanned, flagged=$flagged, skipped=$skipped" -ForegroundColor Cyan
    Write-Host "[csv] $OutputCsv" -ForegroundColor Green
  }

  'ConvertText' {
    $repoTop = Get-RepoRoot
    if (-not $repoTop) { Write-Error '未检测到 Git 仓库，无法依据 .gitignore 过滤。'; exit 1 }

    # 需要处理的扩展集合（默认仅源码；传入 -IncludeDocs 时合并文档扩展）
    $extSet = New-Object System.Collections.Generic.HashSet[string] ([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($e in $CodeExts) { if ($e) { $ext = $e.StartsWith('.') ? $e : ('.' + $e); $null = $extSet.Add($ext) } }
    if ($IncludeDocs) { foreach ($e in $DocExts) { if ($e) { $ext = $e.StartsWith('.') ? $e : ('.' + $e); $null = $extSet.Add($ext) } } }

    $paths = & git -C $repoTop ls-files -co --exclude-standard -z 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Error 'git ls-files 执行失败'; exit 1 }
    $files = @()
    if ($paths) {
      $parts = $paths -split "`0"
      foreach ($p in $parts) { if ($p) { $full = Join-Path $repoTop $p; if (Test-Path -LiteralPath $full) { $files += $full } } }
    }

    $total=0; $changed=0; $skipped=0; $errors=0
    foreach ($f in $files) {
      $ext = [System.IO.Path]::GetExtension($f)
      if (-not $extSet.Contains($ext)) { continue }
      $total++
      $rel = [System.IO.Path]::GetRelativePath($repoTop, $f).Replace('\\','/')

      try { $bytes = [System.IO.File]::ReadAllBytes($f) } catch { $errors++; Write-Warning "读取失败: $rel"; continue }
      if (0 -in $bytes) { $skipped++; Write-Host "[skip] $rel (binary-like)"; continue }

      $isUtf8 = Test-ValidUtf8 $bytes
      $text = $null
      if ($isUtf8) {
        $text = [System.Text.Encoding]::UTF8.GetString($bytes)
      } else {
        try { $text = [System.Text.Encoding]::GetEncoding('GB18030').GetString($bytes) } catch { $text = [System.Text.Encoding]::Default.GetString($bytes) }
      }

      $norm = Normalize-CRLF $text
      $new  = $utf8NoBom.GetBytes($norm)

      $same = ($new.Length -eq $bytes.Length)
      if ($same) {
        for ($i=0; $i -lt $new.Length; $i++) { if ($new[$i] -ne $bytes[$i]) { $same=$false; break } }
      }
      if ($same) { $skipped++; Write-Host "[keep] $rel"; continue }

      if ($DryRun) { Write-Host "[plan] $rel -> UTF-8 + CRLF" -ForegroundColor Yellow; continue }

      try { [System.IO.File]::WriteAllBytes($f, $new); $changed++; Write-Host "[fix]  $rel" -ForegroundColor Green }
      catch { $errors++; Write-Warning "写入失败: $rel" }
    }

    Write-Host "[text-summary] total=$total, changed=$changed, skipped=$skipped, errors=$errors" -ForegroundColor Cyan
  }

  'FixGarbled' {
    $repoTop = Get-RepoRoot
    if (-not $repoTop) { Write-Error '未检测到 Git 仓库，无法依据 .gitignore 过滤。'; exit 1 }

    if (-not $InputCsv) {
      # 默认优先使用 out/garbled_scan.csv，不存在则退回根目录 garbled_files.csv
      $cand1 = Join-Path $repoTop 'out/garbled_scan.csv'
      $cand2 = Join-Path $repoTop 'garbled_files.csv'
      if (Test-Path -LiteralPath $cand1) { $InputCsv = $cand1 }
      elseif (Test-Path -LiteralPath $cand2) { $InputCsv = $cand2 }
      else { Write-Error '未提供 -InputCsv，且缺少默认扫描结果：out/garbled_scan.csv 与 garbled_files.csv'; exit 1 }
    }

    if (-not (Test-Path -LiteralPath $InputCsv)) { Write-Error "CSV 不存在: $InputCsv"; exit 1 }

    $rows = Import-Csv -Path $InputCsv
    if (-not $rows) { Write-Host '[fix] CSV 为空，无需处理。'; return }

    $fixed=0; $kept=0; $failed=0

    foreach ($row in $rows) {
      $rel = [string]$row.Path
      if (-not $rel) { continue }
      $full = Join-Path $repoTop $rel
      if (-not (Test-Path -LiteralPath $full)) { Write-Warning "缺失: $rel"; continue }

      try { $bytes = [System.IO.File]::ReadAllBytes($full) } catch { $failed++; Write-Warning "读取失败: $rel"; continue }
      if (0 -in $bytes) { $kept++; Write-Host "[skip] $rel (binary-like)"; continue }

      $isUtf8 = Test-ValidUtf8 $bytes
      $orig = $isUtf8 ? [System.Text.Encoding]::UTF8.GetString($bytes) : ([System.Text.Encoding]::GetEncoding('GB18030').GetString($bytes))

      # 候选修复：按常见误解码路径尝试回转
      $candidates = @()
      $candidates += [pscustomobject]@{ Name='orig'; Text=$orig }

      try {
        $latinBytes = [System.Text.Encoding]::GetEncoding(1252).GetBytes($orig)
        $latinFixed = $utf8Strict.GetString($latinBytes)
        $candidates += [pscustomobject]@{ Name='Latin1->UTF8'; Text=$latinFixed }
      } catch { }

      try {
        $gbkBytes = [System.Text.Encoding]::GetEncoding('GB18030').GetBytes($orig)
        $gbkFixed = $utf8Strict.GetString($gbkBytes)
        $candidates += [pscustomobject]@{ Name='GB18030->UTF8'; Text=$gbkFixed }
      } catch { }

      try {
        $gbkBytes936 = [System.Text.Encoding]::GetEncoding(936).GetBytes($orig)
        $gbkFixed936 = $utf8Strict.GetString($gbkBytes936)
        $candidates += [pscustomobject]@{ Name='GBK936->UTF8'; Text=$gbkFixed936 }
      } catch { }

      # 可选：再尝试双重回转（在部分链式误码时有用）
      try {
        $tmp = $utf8Strict.GetString([System.Text.Encoding]::GetEncoding(1252).GetBytes($orig))
        $gbkBytes2 = [System.Text.Encoding]::GetEncoding('GB18030').GetBytes($tmp)
        $twice = $utf8Strict.GetString($gbkBytes2)
        $candidates += [pscustomobject]@{ Name='Latin1->UTF8->GB18030->UTF8'; Text=$twice }
      } catch { }

      # 反向尝试：将当前 UTF-8 字符串的字节按 GB 编码解释后再转回
      try {
        $origUtf8Bytes = [System.Text.Encoding]::UTF8.GetBytes($orig)
        $utf8ToGbk = [System.Text.Encoding]::GetEncoding('GB18030').GetString($origUtf8Bytes)
        $candidates += [pscustomobject]@{ Name='UTF8bytes->GB18030(decode)'; Text=$utf8ToGbk }
      } catch { }

      try {
        $origUtf8Bytes2 = [System.Text.Encoding]::UTF8.GetBytes($orig)
        $utf8ToGbk936 = [System.Text.Encoding]::GetEncoding(936).GetString($origUtf8Bytes2)
        $candidates += [pscustomobject]@{ Name='UTF8bytes->GBK936(decode)'; Text=$utf8ToGbk936 }
      } catch { }

      # 评分选择
      $best = $null
      $bestScore = $null
      foreach ($c in $candidates) {
        $s = Get-MojibakeScore $c.Text
        if (-not $best) { $best=$c; $bestScore=$s; continue }
        # 先比 Mojibake 分数，低者优；分数相等比中文比率，高者优
        if ($s.Score -lt $bestScore.Score -or ($s.Score -eq $bestScore.Score -and $s.CjkRatio -gt $bestScore.CjkRatio)) {
          $best=$c; $bestScore=$s
        }
      }

      $origScore = Get-MojibakeScore $orig
      $improved = ($bestScore.Score -lt $origScore.Score) -or (($bestScore.Score -eq $origScore.Score) -and ($bestScore.CjkRatio -gt $origScore.CjkRatio + 0.02))

      if (-not $improved -or $best.Name -eq 'orig') {
        $kept++; Write-Host "[keep] $rel (no-better-fix)"; continue
      }

      $final = Normalize-CRLF $best.Text
      $outBytes = $utf8NoBom.GetBytes($final)

      if ($DryRun) {
        Write-Host "[plan-fix] $rel via $($best.Name) | score $($origScore.Score)->$($bestScore.Score), cjk $([math]::Round($origScore.CjkRatio,3))->$([math]::Round($bestScore.CjkRatio,3))" -ForegroundColor Yellow
        continue
      }

      try {
        if ($Backup) {
          $bak = "$full.bak"
          if (Test-Path -LiteralPath $bak) {
            $ts = Get-Date -Format 'yyyyMMddHHmmss'
            $bak = "$full.$ts.bak"
          }
          Copy-Item -LiteralPath $full -Destination $bak -Force
        }
        [System.IO.File]::WriteAllBytes($full, $outBytes)
        $fixed++
        Write-Host "[fix]  $rel via $($best.Name) (score $($origScore.Score)->$($bestScore.Score), cjk $([math]::Round($origScore.CjkRatio,3))->$([math]::Round($bestScore.CjkRatio,3)))" -ForegroundColor Green
      } catch {
        $failed++
        Write-Warning "写入失败: $rel"
      }
    }

    Write-Host "[fix-summary] fixed=$fixed, kept=$kept, failed=$failed" -ForegroundColor Cyan
  }
}
