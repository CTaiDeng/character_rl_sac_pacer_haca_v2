# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025 GaoZheng

function Write-TextUtf8Lf {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][string]$Content
  )
  $noBom = New-Object System.Text.UTF8Encoding($false)
  $normalized = $Content -replace "`r`n","`n" -replace "`r","`n"
  [System.IO.File]::WriteAllText($Path, $normalized, $noBom)
}

function Append-TextUtf8Lf {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][string]$Content
  )
  $noBom = New-Object System.Text.UTF8Encoding($false)
  $normalized = $Content -replace "`r`n","`n" -replace "`r","`n"
  $dir = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($Path))
  if ($dir -and -not (Test-Path $dir)) { [System.IO.Directory]::CreateDirectory($dir) | Out-Null }
  $bytes = $noBom.GetBytes($normalized)
  $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
  try {
    $fs.Write($bytes, 0, $bytes.Length)
  } finally {
    $fs.Dispose()
  }
}

Export-ModuleMember -Function Write-TextUtf8Lf, Append-TextUtf8Lf

