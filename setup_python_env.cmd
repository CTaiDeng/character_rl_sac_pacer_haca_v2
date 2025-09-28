@echo off
setlocal ENABLEDELAYEDEXPANSION

rem -----------------------------------------------
rem Setup Python virtualenv and install dependencies
rem Default: envdir=.venv, install NLP (jieba, ltp) and AI (google-generativeai)
rem Flags:
rem   --envdir <dir>           Target virtualenv directory (default .venv)
rem   --requirements <path>    Install from requirements.txt first
rem   --no-nlp                 Do not install jieba, ltp
rem   --no-ai                  Do not install google-generativeai
rem   --trace                  Print commands before running
rem -----------------------------------------------

set "ENVDIR=.venv"
set "REQUIREMENTS="
set "NLP=1"
set "AI=1"
set "TRACE=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--envdir" (
  set "ENVDIR=%~2"
  shift & shift & goto parse_args
)
if /I "%~1"=="--requirements" (
  set "REQUIREMENTS=%~2"
  shift & shift & goto parse_args
)
if /I "%~1"=="--no-nlp" (
  set "NLP=0"
  shift & goto parse_args
)
if /I "%~1"=="--no-ai" (
  set "AI=0"
  shift & goto parse_args
)
if /I "%~1"=="--trace" (
  set "TRACE=1"
  shift & goto parse_args
)
echo Unknown argument: %~1
exit /b 1

:args_done

if "%TRACE%"=="1" echo [debug] trace enabled

rem Resolve Python (prefer py -3.10, then py -3, then python)
set "PY_CMD="
py -3.10 -c "import sys" >NUL 2>&1 && set "PY_CMD=py -3.10"
if not defined PY_CMD py -3 -c "import sys" >NUL 2>&1 && set "PY_CMD=py -3"
if not defined PY_CMD python -c "import sys" >NUL 2>&1 && set "PY_CMD=python"
if not defined PY_CMD (
  echo [error] Python 3.10+ not found. Please install Python or the Windows Python Launcher.
  exit /b 1
)

if "%TRACE%"=="1" echo [exec] %PY_CMD% -m venv "%ENVDIR%"
%PY_CMD% -m venv "%ENVDIR%"
if errorlevel 1 goto error

set "VENVPY=%ENVDIR%\Scripts\python.exe"
if not exist "%VENVPY%" (
  echo [error] virtualenv python not found: "%VENVPY%"
  goto error
)

call :run "%VENVPY%" -m pip install --upgrade pip setuptools wheel || goto error

if defined REQUIREMENTS (
  if "%TRACE%"=="1" echo [info] installing from requirements: "%REQUIREMENTS%"
  call :run "%VENVPY%" -m pip install --upgrade -r "%REQUIREMENTS%" || goto error
)

rem numpy + CPU torch
call :run "%VENVPY%" -m pip install --upgrade numpy || goto error
rem Verify numpy import; retry with binary wheels if needed
"%VENVPY%" -c "import numpy as _; print('numpy ok')" >NUL 2>&1
if errorlevel 1 (
  echo [warn] numpy import failed, retrying with binary wheels and version floor...
  call :run "%VENVPY%" -m pip install --upgrade --only-binary=:all: "numpy>=2.0" || goto error
  call :run "%VENVPY%" -c "import numpy as _; print('numpy ok')" || goto error
)

call :run "%VENVPY%" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu
set "TORCH_STATUS=%ERRORLEVEL%"

if "%NLP%"=="1" (
  call :run "%VENVPY%" -m pip install --upgrade jieba ltp || goto error
)

if "%AI%"=="1" (
  call :run "%VENVPY%" -m pip install --upgrade google-generativeai || goto error
)

echo [ok] versions:
call :run "%VENVPY%" -c "import sys,platform; print('Python:', sys.version); import numpy as np; print('numpy:', np.__version__)" || goto error

if "%TORCH_STATUS%"=="0" (
  call :run "%VENVPY%" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" || echo [warn] torch import failed unexpectedly
) else (
  echo [warn] torch installation may be unsupported on this Python; skipped reporting
)
if "%NLP%"=="1" call :run "%VENVPY%" -c "import jieba; print('jieba:', getattr(jieba, '__version__', 'unknown'))" >NUL 2>&1
if "%AI%"=="1"  call :run "%VENVPY%" -c "import google.generativeai as g; print('google-generativeai:', getattr(g, '__version__', 'unknown'))" >NUL 2>&1

echo.
echo [done] Activate your environment:
echo    %ENVDIR%\Scripts\activate
exit /b 0

:run
if "%TRACE%"=="1" echo [exec] %*
%*
exit /b %ERRORLEVEL%

:error
echo [fail] setup failed (errorlevel=%ERRORLEVEL%)
exit /b 1
