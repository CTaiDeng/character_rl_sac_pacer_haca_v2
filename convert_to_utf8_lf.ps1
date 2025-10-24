# encoding: UTF-8 (no BOM)
# eol: LF
# 说明：本脚本会递归遍历指定目录下的文件，打印文件编码与换行符类型，
# 并将非 UTF-8 + LF 的文本文件转换为 UTF-8 + LF（默认移除 BOM）。
# 原因：脚本用于统一为 LF，以便跨平台工具和 CI 更一致；为遵循仓库“例外需声明”的约定，这里显式标注换行规则为 LF。
# 同时：
# - 跳过 Git 忽略（.gitignore 等）文件/目录
# - 跳过根目录 partial_clone_exclude_whitelist.json 中声明要排除的路径
# - 跳过常见二进制与过大文件

[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [string]$Path = ".",

    # 仅打印，不执行写回
    [switch]$DryRun,

    # 写出时是否包含 UTF-8 BOM（默认不包含）
    [switch]$WithBom,

    # 最大处理文件大小（字节），超过将跳过（默认 10MB）
    [long]$MaxFileBytes = 10MB,

    # 额外排除的目录名（相对/包含匹配），默认会排除常见生成目录
    [string[]]$ExcludeDirs = @('.git', '.svn', '.hg', '.idea', '.vs', 'node_modules', 'dist', 'build', 'out', 'bin', 'obj'),

    # 额外排除的文件通配符（匹配 Name），如 '*.png'
    [string[]]$ExcludePatterns = @('*.png','*.jpg','*.jpeg','*.gif','*.bmp','*.ico','*.pdf','*.zip','*.7z','*.rar','*.gz','*.tar','*.jar','*.dll','*.exe','*.bin','*.mp3','*.mp4','*.mov','*.avi','*.wav','*.woff','*.woff2','*.ttf','*.otf','*.cmd','*.bat')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-RelativePath {
    param(
        [string]$FullPath,
        [string]$BasePath
    )
    try {
        $uriFull = [System.Uri]([System.IO.Path]::GetFullPath($FullPath))
        $uriBase = [System.Uri]([System.IO.Path]::GetFullPath($BasePath))
        if ($uriBase.IsBaseOf($uriFull)) {
            return [System.Uri]::UnescapeDataString($uriBase.MakeRelativeUri($uriFull).ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
        }
    } catch {}
    return $FullPath
}

function Test-PathExcluded {
    param(
        [System.IO.FileSystemInfo]$Item,
        [string[]]$ExcludeDirs,
        [string[]]$ExcludePatterns
    )
    # 目录排除（匹配路径片段，避免宽泛包含误伤）
    foreach ($ex in $ExcludeDirs) {
        if ([string]::IsNullOrWhiteSpace($ex)) { continue }
        $sep = [System.IO.Path]::DirectorySeparatorChar
        $full = $Item.FullName
        if ($full -like ("*{0}{1}{0}*" -f $sep, $ex)) { return $true }
        if ($full -like ("*{0}{1}" -f $sep, $ex)) { return $true } # 末段恰为该目录名
    }
    # 文件名通配符排除
    foreach ($pat in $ExcludePatterns) {
        if ([string]::IsNullOrWhiteSpace($pat)) { continue }
        if ($Item.Name -like $pat) { return $true }
    }
    return $false
}

function Read-TextAuto {
    param(
        [string]$FilePath
    )
    $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    $encInfo = Detect-FileEncoding -Bytes $bytes
    $offset = 0
    switch ($encInfo.Kind) {
        'utf8'     { if ($encInfo.HasBOM) { $offset = 3 } }
        'utf32le'  { $offset = 4 }
        'utf32be'  { $offset = 4 }
        'utf16le'  { $offset = 2 }
        'utf16be'  { $offset = 2 }
        default    { $offset = 0 }
    }
    $text = $encInfo.Encoding.GetString($bytes, $offset, $bytes.Length - $offset)
    [pscustomobject]@{ Text=$text; Encoding=$encInfo }
}

function Load-PartialCloneExcludeList {
    param(
        [string]$Root
    )
    $file = Join-Path $Root 'partial_clone_exclude_whitelist.json'
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { return @() }
    try {
        $data = Read-TextAuto -FilePath $file
        $obj = $data.Text | ConvertFrom-Json -ErrorAction Stop

        $paths = @()
        if ($obj -is [System.Collections.IEnumerable] -and -not ($obj -is [string])) {
            foreach ($i in $obj) { if ($i -is [string]) { $paths += $i } }
        } elseif ($obj -is [pscustomobject]) {
            foreach ($key in @('exclude','exclude_files','excludeFiles','whitelist','paths','items')) {
                if ($obj.PSObject.Properties.Name -contains $key) {
                    $val = $obj.$key
                    if ($val -is [string]) { $paths += $val }
                    elseif ($val -is [System.Collections.IEnumerable]) {
                        foreach ($i in $val) { if ($i -is [string]) { $paths += $i } }
                    }
                }
            }
        }

        # 归一化为通配符或前缀匹配单元
        $norm = @()
        foreach ($p in $paths) {
            if ([string]::IsNullOrWhiteSpace($p)) { continue }
            $pat = $p.Trim().TrimStart('./').Replace('/', [System.IO.Path]::DirectorySeparatorChar)
            $norm += $pat
        }
        return $norm
    } catch {
        Write-Warning "无法解析 partial_clone_exclude_whitelist.json：$($_.Exception.Message)"
        return @()
    }
}

function Test-PartialCloneExcluded {
    param(
        [string]$RelativePath,
        [string[]]$ExcludeList
    )
    if (-not $ExcludeList -or $ExcludeList.Count -eq 0) { return $false }
    $rel = $RelativePath.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    foreach ($pat in $ExcludeList) {
        if ($pat -match '[\*\?\[]') {
            if ($rel -like $pat) { return $true }
        } else {
            if ($rel -eq $pat) { return $true }
            if ($rel -like ("{0}{1}*" -f $pat, [System.IO.Path]::DirectorySeparatorChar)) { return $true }
        }
    }
    return $false
}

function Test-GitIgnored {
    param(
        [string]$RepoRoot,
        [string]$RelativePath
    )
    try {
        $null = & git -C $RepoRoot rev-parse --is-inside-work-tree 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
    } catch { return $false }

    $null = & git -C $RepoRoot check-ignore -q -- $RelativePath 2>$null
    $code = $LASTEXITCODE
    if ($code -eq 0) { return $true }
    if ($code -eq 1) { return $false }
    return $false
}

function Detect-FileEncoding {
    param(
        [byte[]]$Bytes
    )
    # 返回： [pscustomobject] @{ Encoding = [System.Text.Encoding]; Name = string; HasBOM = bool; Kind = string }

    $len = $Bytes.Length
    $has = {
        param([int[]]$sig)
        if ($Bytes.Length -lt $sig.Length) { return $false }
        for ($i=0; $i -lt $sig.Length; $i++) {
            if ($Bytes[$i] -ne $sig[$i]) { return $false }
        }
        return $true
    }

    # BOM 检测
    if (& $has 0xEF,0xBB,0xBF) {
        return [pscustomobject]@{ Encoding = [System.Text.Encoding]::UTF8; Name = 'UTF-8 (BOM)'; HasBOM = $true; Kind='utf8' }
    }
    if (& $has 0xFF,0xFE,0x00,0x00) {
        return [pscustomobject]@{ Encoding = [System.Text.Encoding]::GetEncoding('utf-32'); Name = 'UTF-32 LE (BOM)'; HasBOM = $true; Kind='utf32le' }
    }
    if (& $has 0x00,0x00,0xFE,0xFF) {
        return [pscustomobject]@{ Encoding = [System.Text.Encoding]::GetEncoding('utf-32BE'); Name = 'UTF-32 BE (BOM)'; HasBOM = $true; Kind='utf32be' }
    }
    if (& $has 0xFF,0xFE) {
        return [pscustomobject]@{ Encoding = [System.Text.Encoding]::Unicode; Name = 'UTF-16 LE (BOM)'; HasBOM = $true; Kind='utf16le' }
    }
    if (& $has 0xFE,0xFF) {
        return [pscustomobject]@{ Encoding = [System.Text.Encoding]::BigEndianUnicode; Name = 'UTF-16 BE (BOM)'; HasBOM = $true; Kind='utf16be' }
    }
    if (& $has 0x2B,0x2F,0x76) {
        try {
            $enc = [System.Text.Encoding]::GetEncoding('utf-7')
        } catch { $enc = [System.Text.Encoding]::ASCII }
        return [pscustomobject]@{ Encoding = $enc; Name = 'UTF-7 (BOM)'; HasBOM = $true; Kind='utf7' }
    }

    # 无 BOM：尝试严格 UTF-8 解码验证
    try {
        $strictUtf8 = New-Object System.Text.UTF8Encoding($false, $true)
        [void]$strictUtf8.GetCharCount($Bytes)
        return [pscustomobject]@{ Encoding = [System.Text.Encoding]::UTF8; Name = 'UTF-8'; HasBOM = $false; Kind='utf8' }
    } catch {}

    # 回退到系统默认代码页（ANSI）
    $def = [System.Text.Encoding]::Default
    return [pscustomobject]@{ Encoding = $def; Name = "ANSI ($($def.WebName))"; HasBOM = $false; Kind='ansi' }
}

function Detect-LineEndings {
    param(
        [string]$Text
    )
    $crlf = [regex]::Matches($Text, "`r`n").Count
    $lf   = [regex]::Matches($Text, "(?<!`r)`n").Count
    $cr   = [regex]::Matches($Text, "`r(?!`n)").Count

    $type = if ($crlf -gt 0 -and $lf -eq 0 -and $cr -eq 0) { 'CRLF' }
            elseif ($lf -gt 0 -and $crlf -eq 0 -and $cr -eq 0) { 'LF' }
            elseif ($cr -gt 0 -and $crlf -eq 0 -and $lf -eq 0) { 'CR' }
            elseif ($crlf -eq 0 -and $lf -eq 0 -and $cr -eq 0) { 'None' }
            else { 'Mixed' }

    [pscustomobject]@{
        Type = $type
        CRLF = $crlf
        LF   = $lf
        CR   = $cr
    }
}

function Test-BinaryBytes {
    param(
        [byte[]]$Bytes
    )
    if ($Bytes.Length -eq 0) { return $false }
    # 含有 NUL 字节（且不是 UTF-16/32 情况）通常视为二进制
    $nulCount = 0
    for ($i=0; $i -lt [Math]::Min($Bytes.Length, 4096); $i++) {
        if ($Bytes[$i] -eq 0) { $nulCount++ }
    }
    if ($nulCount -gt 0) { return $true }
    return $false
}

function Convert-ToUtf8Lf {
    param(
        [string]$FilePath,
        [switch]$DryRun,
        [switch]$WithBom
    )

    $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    $encInfo = Detect-FileEncoding -Bytes $bytes

    # 初步二进制判断（对 UTF-16/32 仍按文本处理）
    $maybeBinary = Test-BinaryBytes -Bytes $bytes

    # 以检测出的编码解码
    $offset = 0
    switch ($encInfo.Kind) {
        'utf8'     { if ($encInfo.HasBOM) { $offset = 3 } }
        'utf32le'  { $offset = 4 }
        'utf32be'  { $offset = 4 }
        'utf16le'  { $offset = 2 }
        'utf16be'  { $offset = 2 }
        default    { $offset = 0 }
    }

    $text = $encInfo.Encoding.GetString($bytes, $offset, $bytes.Length - $offset)
    $eol = Detect-LineEndings -Text $text

    # 判断是否需要转换
    $isUtf8 = ($encInfo.Encoding.WebName -eq 'utf-8')
    $hasBom = [bool]$encInfo.HasBOM
    $isLf   = ($eol.Type -eq 'LF')

    # 默认移除 UTF-8 BOM（除非显式要求 WithBom）
    $needStripBom = (-not $WithBom) -and $hasBom
    $needsEncoding = -not $isUtf8 -or ($WithBom -and -not $hasBom) -or $needStripBom
    $needsEol = -not $isLf
    $needsChange = $needsEncoding -or $needsEol

    # 输出状态行（文件 | 编码 | 行尾 | 动作）
    $action = if ($maybeBinary) { '跳过(二进制)' } elseif ($needsChange) { if ($DryRun) { '转换(试运行)' } else { '转换' } } else { '已是目标' }
    $rel = Get-RelativePath -FullPath $FilePath -BasePath $Path
    $fg = if ($maybeBinary) { 'Red' } elseif ($needsChange) { 'Red' } else { 'Green' }
    Write-Host ("{0} | {1} | {2} | {3}" -f $rel, $encInfo.Name, $eol.Type, $action) -ForegroundColor $fg

    if ($maybeBinary) { return [pscustomobject]@{ Changed=$false; Skipped=$true } }
    if (-not $needsChange -or $DryRun) { return [pscustomobject]@{ Changed=$false; Skipped=$false } }

    # 规范化为 LF，并在需要时确保结尾换行（处理 EOL=None 场景）
    if ($needsEol) {
        $text = $text -replace "`r`n", "`n"
        $text = $text -replace "`r", "`n"
        if (-not $text.EndsWith("`n")) { $text += "`n" }
    }

    # 写回 UTF-8（默认无 BOM）
    $utf8 = New-Object System.Text.UTF8Encoding($WithBom.IsPresent, $false)
    $dir = [System.IO.Path]::GetDirectoryName($FilePath)
    $tmp = [System.IO.Path]::Combine($dir, [System.IO.Path]::GetRandomFileName())
    try {
        $sw = New-Object System.IO.StreamWriter($tmp, $false, $utf8)
        try {
            $sw.Write($text)
        } finally {
            $sw.Close()
            $sw.Dispose()
        }
        [System.IO.File]::Copy($tmp, $FilePath, $true)
    } finally {
        if (Test-Path $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
    }

    return [pscustomobject]@{ Changed=$true; Skipped=$false }
}

# 主流程
$root = Resolve-Path -LiteralPath $Path | Select-Object -First 1 -ExpandProperty Path
if (-not (Test-Path -LiteralPath $root -PathType Container)) {
    throw "Path not found or not a directory: $Path"
}

$partialExcludeList = Load-PartialCloneExcludeList -Root $root

# 以“目录级别”直接跳过：
# - 被 Git 忽略的目录
# - 在 partial_clone_exclude_whitelist.json 中声明的目录
# - 在参数 ExcludeDirs 中命中的目录
function Get-FilesWithPrune {
    param(
        [string]$Root,
        [string[]]$ExcludeDirs,
        [string[]]$ExcludePatterns,
        [string[]]$PartialExcludeList
    )

    $stack = New-Object System.Collections.Generic.Stack[System.IO.DirectoryInfo]
    $stack.Push((Get-Item -LiteralPath $Root))

    $prunedByName = 0
    $prunedByGit  = 0
    $prunedByList = 0

    while ($stack.Count -gt 0) {
        $dir = $stack.Pop()
        if (-not (Test-Path -LiteralPath $dir.FullName -PathType Container)) { continue }

        $relDir = Get-RelativePath -FullPath $dir.FullName -BasePath $Root
        if ($relDir -eq '.' -or $relDir -eq $dir.FullName) { $relDir = '' }

        $shouldPrune = $false
        $pruneReason = ''
        if ($relDir -ne '') {
            # 目录名/路径规则排除
            if (Test-PathExcluded -Item $dir -ExcludeDirs $ExcludeDirs -ExcludePatterns @()) { $shouldPrune = $true; $pruneReason = '名称规则'; $prunedByName++ }

            # Git 忽略目录
            if (-not $shouldPrune) {
                if (Test-GitIgnored -RepoRoot $Root -RelativePath $relDir) { $shouldPrune = $true; $pruneReason = 'Git 忽略'; $prunedByGit++ }
            }

            # partial_clone 白名单中的排除目录
            if (-not $shouldPrune) {
                if (Test-PartialCloneExcluded -RelativePath $relDir -ExcludeList $PartialExcludeList) { $shouldPrune = $true; $pruneReason = '部分克隆白名单'; $prunedByList++ }
            }
        }

        if ($shouldPrune) { continue }

        # 当前目录文件（仍应用文件级排除与通配）
        Get-ChildItem -LiteralPath $dir.FullName -File -Force -ErrorAction SilentlyContinue |
            Where-Object { -not (Test-PathExcluded -Item $_ -ExcludeDirs $ExcludeDirs -ExcludePatterns $ExcludePatterns) }

        # 下探子目录（延后再判断是否需要 prune）
        Get-ChildItem -LiteralPath $dir.FullName -Directory -Force -ErrorAction SilentlyContinue |
            ForEach-Object { $stack.Push($_) }
    }
    # 不打印任何被跳过目录的信息，按要求静默处理
}

$all = @(Get-FilesWithPrune -Root $root -ExcludeDirs $ExcludeDirs -ExcludePatterns $ExcludePatterns -PartialExcludeList $partialExcludeList)

$total = 0
$converted = 0
$ok = 0
$tooLarge = 0
$gitIgnored = 0
$partialExcluded = 0
$binary = 0
$errors = 0

foreach ($file in $all) {
    # 额外的安全校验：单文件维度再判一次（以防目录未被 prune 的散落文件）
    $relp = Get-RelativePath -FullPath $file.FullName -BasePath $root

    if (Test-GitIgnored -RepoRoot $root -RelativePath $relp) {
        # 不再逐条打印，避免噪声；直接计数
        $gitIgnored++
        continue
    }
    if (Test-PartialCloneExcluded -RelativePath $relp -ExcludeList $partialExcludeList) {
        $partialExcluded++
        continue
    }
    if ($file.Length -gt $MaxFileBytes) {
        Write-Host ("{0} | 过大({1} 字节) | 跳过" -f $relp, $file.Length) -ForegroundColor Red
        $tooLarge++
        continue
    }

    $total++
    try {
        $res = Convert-ToUtf8Lf -FilePath $file.FullName -DryRun:$DryRun -WithBom:$WithBom
        if ($res.Skipped) { $binary++ }
        elseif ($res.Changed) { $converted++ } else { $ok++ }
    } catch {
        Write-Warning "错误：$relp => $($_.Exception.Message)"
        $errors++
    }
}

Write-Host ("---- 汇总 ----") -ForegroundColor Cyan
Write-Host ("根目录: {0}" -f $root) -ForegroundColor DarkCyan
Write-Host ("处理文件数: {0}" -f $total) -ForegroundColor DarkCyan

# 转换为 0 时显示绿色，否则红色
$fgConverted = if ($converted -eq 0) { 'Green' } else { 'Red' }
Write-Host ("转换: {0}" -f $converted) -ForegroundColor $fgConverted
Write-Host ("已是目标: {0}" -f $ok) -ForegroundColor Green

# 0 显示为绿色，否则红色
$fgBinary          = if ($binary -eq 0) { 'Green' } else { 'Red' }
$fgTooLarge        = if ($tooLarge -eq 0) { 'Green' } else { 'Red' }
$fgGitIgnored      = if ($gitIgnored -eq 0) { 'Green' } else { 'Red' }
$fgPartialExcluded = if ($partialExcluded -eq 0) { 'Green' } else { 'Red' }
$fgErrors          = if ($errors -eq 0) { 'Green' } else { 'Red' }

Write-Host ("跳过-二进制: {0}" -f $binary) -ForegroundColor $fgBinary
Write-Host ("跳过-过大: {0}" -f $tooLarge) -ForegroundColor $fgTooLarge
Write-Host ("跳过-Git忽略: {0}" -f $gitIgnored) -ForegroundColor $fgGitIgnored
Write-Host ("跳过-部分克隆排除: {0}" -f $partialExcluded) -ForegroundColor $fgPartialExcluded
Write-Host ("错误: {0}" -f $errors) -ForegroundColor $fgErrors
