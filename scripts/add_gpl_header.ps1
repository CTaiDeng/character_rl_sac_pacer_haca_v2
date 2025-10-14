<#
SPDX-License-Identifier: GPL-3.0-only
Copyright (C) 2025 GaoZheng

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
#>

[CmdletBinding()]
param(
  [string]$Root = ".",
  [switch]$DryRun,
  [string]$Author = "GaoZheng",
  [int]$Year = 2025,
  [ValidateSet("GPL-3.0-only","GPL-3.0-or-later")]
  [string]$LicenseId = "GPL-3.0-only"
)

$ErrorActionPreference = "Stop"

# 需要处理的扩展名及注释风格（不含 Markdown）
$ExtensionStyle = @{
  '.py'='hash'; '.sh'='hash'; '.bash'='hash'; '.zsh'='hash'; '.ps1'='hash'; '.rb'='hash'; '.pl'='hash'; '.r'='hash'; '.jl'='hash';
  '.c'='cblock'; '.h'='cblock'; '.cpp'='cblock'; '.cc'='cblock'; '.hpp'='cblock'; '.cs'='cblock'; '.java'='cblock'; '.go'='cblock';
  '.js'='cblock'; '.mjs'='cblock'; '.cjs'='cblock'; '.ts'='cblock'; '.tsx'='cblock'; '.jsx'='cblock'; '.swift'='cblock'; '.kt'='cblock'; '.kts'='cblock'; '.scala'='cblock'; '.rs'='cblock';
  '.php'='php';
  '.lua'='dashdash'; '.sql'='dashdash';
  '.html'='xml'; '.htm'='xml'; '.xml'='xml'; '.svg'='xml'
}

# 显式跳过的文本/标记类扩展名（保证不改 Markdown）
$SkipExts = @('.md', '.mdx', '.markdown', '.rst', '.adoc', '.txt')

# 跳过的目录（含只读目录 docs/kernel_reference/）
$SkipDirs = @(
  ".git", ".githooks", ".idea", ".vscode",
  "node_modules", "dist", "build", "out", "target", "bin", "obj",
  "__pycache__", ".pytest_cache", ".mypy_cache", ".cache",
  "venv", ".venv",
  "docs/kernel_reference"
)

function Set-FileContentUtf8BomLf {
  param([string]$Path, [string]$Text)
  $enc = [System.Text.UTF8Encoding]::new($true) # 带 BOM
  $normalized = $Text -replace "`r`n", "`n" -replace "`r", "`n"
  [System.IO.File]::WriteAllText($Path, $normalized, $enc)
}

function New-LicenseHeader {
  param([string]$Style, [int]$Year, [string]$Author, [string]$LicenseId)

  $lines = @(
    "SPDX-License-Identifier: $LicenseId",
    "Copyright (C) $Year $Author",
    "",
    "This program is free software: you can redistribute it and/or modify",
    "it under the terms of the GNU General Public License as published by",
    "the Free Software Foundation, version 3.",
    "",
    "This program is distributed in the hope that it will be useful,",
    "but WITHOUT ANY WARRANTY; without even the implied warranty of",
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the",
    "GNU General Public License for more details.",
    "",
    "You should have received a copy of the GNU General Public License",
    "along with this program. If not, see https://www.gnu.org/licenses/."
  )

  switch ($Style) {
    'hash'     { $commented = $lines | ForEach-Object { "# $_" }; return (($commented -join "`n") + "`n`n") }
    'dashdash' { $commented = $lines | ForEach-Object { "-- $_" }; return (($commented -join "`n") + "`n`n") }
    'cblock'   {
      $inner = $lines | ForEach-Object { " * $_" }
      return ("/*`n" + ($inner -join "`n") + "`n */`n`n")
    }
    'xml'      {
      $inner = $lines | ForEach-Object { " $_" }
      return ("<!--`n" + ($inner -join "`n") + "`n-->`n`n")
    }
    'php'      {
      $inner = $lines | ForEach-Object { " * $_" }
      return ("/*`n" + ($inner -join "`n") + "`n */`n`n")
    }
    default    { throw "未知注释风格: $Style" }
  }
}

function Test-HasHeader {
  param([string]$Content)
  $lines = $Content -split "`r?`n", 0
  if ($lines.Length -eq 0) { return $false }
  $take = [Math]::Min(100, $lines.Length)
  $head = ($lines[0..($take-1)] -join "`n")
  return ($head -match 'SPDX-License-Identifier:\s*GPL-3\.0' -or $head -match 'GNU General Public License')
}

function Get-PreludeCount {
  param([string[]]$Lines, [string]$Ext)
  $i = 0
  if ($Lines.Length -gt 0 -and $Lines[0] -match '^\#\!') { $i++ } # shebang

  if ($Ext -eq ".py") {
    if ($Lines.Length -gt $i -and $Lines[$i] -match '^\s*#.*coding\s*[:=]\s*[-\w\.]+') { $i++ } # PEP 263
  }

  if ($Ext -eq ".php") {
    # 仅对纯脚本：shebang 后紧随 <?php，避免破坏 HTML 输出
    $j = $i
    if ($Lines.Length -gt $j -and $Lines[$j] -match '^\s*<\?php') { return ($j + 1) }
    return -1
  }

  if ($Ext -in @(".xml",".svg",".html",".htm")) {
    if ($Lines.Length -gt 0 -and $Lines[0] -match '^\s*<\?xml\b.*\?>') { $i++ }
  }
  return $i
}

$rootPath = (Resolve-Path -LiteralPath $Root).Path
$includeGlobs = $ExtensionStyle.Keys | ForEach-Object { "*$_" }

# 规范化跳过目录前缀
$skipPrefixes = @()
foreach ($sd in $SkipDirs) {
  $full = [System.IO.Path]::GetFullPath((Join-Path $rootPath $sd))
  $sep = [IO.Path]::DirectorySeparatorChar
  if (-not $full.EndsWith($sep)) { $full = "$full$sep" }
  $skipPrefixes += $full
}

$files = Get-ChildItem -Path (Join-Path $rootPath '*') -Recurse -File -Include $includeGlobs -Force -ErrorAction SilentlyContinue

$updated = 0; $already = 0; $skipped = 0; $dry = 0; $errors = 0

foreach ($file in $files) {
  try {
    $full = [System.IO.Path]::GetFullPath($file.FullName)
    $shouldSkip = $false
    foreach ($sp in $skipPrefixes) {
      if ($full.StartsWith($sp, [StringComparison]::OrdinalIgnoreCase)) { $shouldSkip = $true; break }
    }
    if ($shouldSkip) { $skipped++; continue }

    $ext = $file.Extension.ToLowerInvariant()
    if ($SkipExts -contains $ext) { $skipped++; continue }
    if (-not $ExtensionStyle.ContainsKey($ext)) { $skipped++; continue }

    $content = [System.IO.File]::ReadAllText($full)
    if (Test-HasHeader -Content $content) { $already++; continue }

    $lines = $content -split "`r?`n", 0
    $preludeCount = Get-PreludeCount -Lines $lines -Ext $ext
    if ($preludeCount -eq -1) {
      Write-Host "跳过（PHP 非纯脚本或未以 <?php 开头）: $full"
      $skipped++; continue
    }

    $header = New-LicenseHeader -Style $ExtensionStyle[$ext] -Year $Year -Author $Author -LicenseId $LicenseId

    $prefix = ""
    if ($preludeCount -gt 0) {
      $prefix = ($lines[0..($preludeCount-1)] -join "`n") + "`n"
    }
    $rest = if ($preludeCount -lt $lines.Length) { ($lines[$preludeCount..($lines.Length-1)] -join "`n") } else { "" }
    $newText = $prefix + $header + $rest

    if ($DryRun) {
      Write-Host "[DRY] 将添加头部: $full"
      $dry++
    } else {
      Set-FileContentUtf8BomLf -Path $full -Text $newText
      $updated++
    }
  } catch {
    Write-Warning "处理失败: $($file.FullName) -> $($_.Exception.Message)"
    $errors++
  }
}

if ($DryRun) {
  Write-Host "DryRun 完成：将更新 $dry 个文件；已存在头部 $already；跳过 $skipped；错误 $errors。"
} else {
  Write-Host "完成：已更新 $updated 个文件；已存在头部 $already；跳过 $skipped；错误 $errors。"
}
