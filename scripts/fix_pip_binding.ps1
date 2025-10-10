<#
  修复 pip 启动器绑定到错误 Python 的问题（Windows/PowerShell）

  功能概述：
  - 自动发现并使用仓库 .venv\Scripts\python.exe（或通过 -PythonPath/-VenvRoot 指定）
  - 通过 `python -m ensurepip --upgrade` + `python -m pip install --upgrade [--force-reinstall] pip` 重建/修复 pip 启动器
  - 可选同时升级 setuptools、wheel
  - 可选使用自定义 PyPI 索引（如清华镜像）
  - 绑定一致性校验：检查 pip-script.py 的 shebang 是否指向目标 Python，必要时清理旧的 pip 启动器后重建

  用法示例：
    # 最常用：修复当前仓库 .venv 的 pip 绑定
    powershell -ExecutionPolicy Bypass -File scripts\fix_pip_binding.ps1

    # 指定 Python 路径，强制重装 pip，并同步升级 setuptools/wheel
    powershell -ExecutionPolicy Bypass -File scripts\fix_pip_binding.ps1 -PythonPath .\.venv\Scripts\python.exe -ForceReinstall -WithSetuptoolsWheel

    # 使用镜像
    powershell -ExecutionPolicy Bypass -File scripts\fix_pip_binding.ps1 -IndexUrl https://pypi.tuna.tsinghua.edu.cn/simple
#>

param(
  [string]$VenvRoot,
  [string]$PythonPath,
  [switch]$ForceReinstall,
  [switch]$WithSetuptoolsWheel,
  [string]$IndexUrl,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

function Resolve-Root {
  $here = $PSScriptRoot
  if (-not $here) { $here = Split-Path -Parent $MyInvocation.MyCommand.Path }
  return (Split-Path $here -Parent)
}

function Find-Python {
  param([string]$Prefer, [string]$Venv)
  if ($Prefer -and (Test-Path $Prefer)) { return (Resolve-Path $Prefer).Path }
  if ($Venv) {
    $cand = Join-Path $Venv 'Scripts\python.exe'
    if (Test-Path $cand) { return (Resolve-Path $cand).Path }
  }
  $root = Resolve-Root
  $venvPy = Join-Path $root '.venv\Scripts\python.exe'
  if (Test-Path $venvPy) { return (Resolve-Path $venvPy).Path }
  $py = (Get-Command py -ErrorAction SilentlyContinue)?.Source
  if ($py) { return $py }
  $py = (Get-Command python -ErrorAction SilentlyContinue)?.Source
  if ($py) { return $py }
  return $null
}

function Get-PipLaunchers {
  param([string]$Venv)
  $dir = if ($Venv) { Join-Path $Venv 'Scripts' } else { (Split-Path (Find-Python) -Parent) }
  return [pscustomobject]@{
    Dir = $dir
    PipExe = Join-Path $dir 'pip.exe'
    Pip3Exe = Join-Path $dir 'pip3.exe'
    PipScript = Join-Path $dir 'pip-script.py'
    Pip3Script = Join-Path $dir 'pip3-script.py'
  }
}

function Read-Shebang {
  param([string]$Path)
  try {
    if (Test-Path $Path) { return (Get-Content $Path -TotalCount 1 -ErrorAction Stop) }
  } catch {}
  return $null
}

function Test-BindingMismatch {
  param([string]$ExpectedPython, $Launchers)
  $exp = (Resolve-Path $ExpectedPython).Path
  foreach ($p in @($Launchers.PipScript, $Launchers.Pip3Script)) {
    $line = Read-Shebang -Path $p
    if ($line -and ($line -match '^#!(.+)$')) {
      $sb = $Matches[1]
      if (-not ($sb -ieq $exp)) { return $true }
    }
  }
  return $false
}

function Invoke-Fix {
  param([string]$Py, [switch]$DoForce, [switch]$DoSetuptoolsWheel, [string]$Idx, [switch]$Noop)
  # 1) ensurepip 升级
  $ensure = @($Py,'-m','ensurepip','--upgrade')
  Write-Host ('> ' + ($ensure -join ' ')) -ForegroundColor Cyan
  if (-not $Noop) { & $ensure[0] $ensure[1..($ensure.Length-1)] }

  # 2) pip 升级（可选强制 + 组件 + 镜像）
  $pipArgs = @('install','--upgrade')
  if ($DoForce) { $pipArgs += '--force-reinstall' }
  $pipArgs += 'pip'
  if ($DoSetuptoolsWheel) { $pipArgs += @('setuptools','wheel') }
  if ($Idx) { $pipArgs += @('--index-url', $Idx) }
  $pipCmd = @($Py,'-m','pip') + $pipArgs
  Write-Host ('> ' + ($pipCmd -join ' ')) -ForegroundColor Cyan
  if (-not $Noop) { & $pipCmd[0] $pipCmd[1..($pipCmd.Length-1)] }
}

function Remove-OldLaunchers {
  param($Launchers)
  foreach ($p in @($Launchers.PipExe,$Launchers.Pip3Exe,$Launchers.PipScript,$Launchers.Pip3Script)) {
    try { if (Test-Path $p) { Remove-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue } } catch {}
  }
}

# --- main ---
$ROOT = Resolve-Root
$pyExe = Find-Python -Prefer $PythonPath -Venv $VenvRoot
if (-not $pyExe) { throw '未找到 Python。请指定 -PythonPath 或在仓库创建 .venv' }

Write-Host ("使用 Python: {0}" -f $pyExe) -ForegroundColor Green
$launchers = Get-PipLaunchers -Venv $VenvRoot
Write-Host ("pip 目录: {0}" -f $launchers.Dir)

# 先尝试查询当前状态
Write-Host '当前状态自检:' -ForegroundColor Yellow
try { & $pyExe -m pip --version } catch { Write-Warning ('python -m pip 异常: ' + $_.Exception.Message) }
try { if (Test-Path $launchers.PipExe) { & $launchers.PipExe --version } } catch { Write-Warning ('pip.exe 异常: ' + $_.Exception.Message) }
$mismatch = Test-BindingMismatch -ExpectedPython $pyExe -Launchers $launchers
if ($mismatch) { Write-Warning '检测到 pip-script.py 绑定与当前 Python 不一致，将清理并重建。' }

# 提示可手动执行的强制修复命令（黄字显示）
$explicitVenvPy = Join-Path $ROOT '.venv\Scripts\python.exe'
if (Test-Path $explicitVenvPy) {
  Write-Host '.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall pip' -ForegroundColor Yellow
}

if ($mismatch -and -not $DryRun) { Remove-OldLaunchers -Launchers $launchers }

Invoke-Fix -Py $pyExe -DoForce:$ForceReinstall -DoSetuptoolsWheel:$WithSetuptoolsWheel -Idx $IndexUrl -Noop:$DryRun

# 若用户显式要求 ForceReinstall，且本仓库存在 .venv，则再执行一次明确的强制重装命令
if ($ForceReinstall -and (Test-Path $explicitVenvPy) -and -not $DryRun) {
  Write-Host '> .\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall pip' -ForegroundColor Cyan
  & $explicitVenvPy -m pip install --upgrade --force-reinstall pip
}

Write-Host ''
Write-Host '修复后校验:' -ForegroundColor Yellow
& $pyExe -m pip --version
if (Test-Path $launchers.PipExe) { & $launchers.PipExe --version }
foreach ($p in @($launchers.PipScript,$launchers.Pip3Script)) {
  $sb = Read-Shebang -Path $p
  if ($sb) { Write-Host ("{0} -> {1}" -f (Split-Path $p -Leaf), $sb) }
}

Write-Host '完成。如仍异常，可加 -ForceReinstall 或删除 .venv 后重建。' -ForegroundColor Green
