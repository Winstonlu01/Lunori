@echo off
setlocal enableextensions
title Lunori – Dev Runner

rem --- Resolve project root (folder where this .bat lives) ---
set "ROOT=%~dp0"
pushd "%ROOT%"
set "ROOT=%CD%\"
popd

rem --- Ensure logs dir + timestamped logfile ---
set "LOGDIR=%ROOT%logs"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f %%i in ('powershell -NoProfile -Command "(Get-Date -Format yyyyMMdd-HHmmss)"') do set "STAMP=%%i"
set "LOGFILE=%LOGDIR%\run-%STAMP%.txt"

echo [Lunori] Starting (log: %LOGFILE%)
echo ==== Lunori run %DATE% %TIME% ====>> "%LOGFILE%"

rem --- Prefer bundled ffmpeg first on PATH ---
set "FFBIN=%ROOT%vendor\ffmpeg\win64\bin"
if exist "%FFBIN%\ffmpeg.exe" set "PATH=%FFBIN%;%PATH%"

rem --- Activate Python venv ---
set "VENV_ACT=%ROOT%.venv\Scripts\activate.bat"
if exist "%VENV_ACT%" (
  echo [+] Activating Python venv...
  call "%VENV_ACT%"
) else (
  echo [!] Could not find venv at "%VENV_ACT%".
  echo     Create it first:  python -m venv .venv  &&  .venv\Scripts\pip install -r requirements.txt
  echo [!] Missing venv noted. >> "%LOGFILE%"
  pause
  exit /b 1
)

rem --- Check Node & npm on PATH ---
where node >nul 2>nul || ( echo [!] Node.js not found on PATH. Install Node 18+ and retry.& echo Node missing.>>"%LOGFILE%" & pause & exit /b 1 )
where npm  >nul 2>nul || ( echo [!] npm not found on PATH.& echo npm missing.>>"%LOGFILE%" & pause & exit /b 1 )

rem --- Start the Electron dev app (which also starts FastAPI via npm scripts) ---
pushd "%ROOT%electron" || ( echo [!] Can't find "%ROOT%electron".& echo electron folder missing.>>"%LOGFILE%" & pause & exit /b 1 )
if not exist package.json ( echo [!] No package.json in /electron.& echo missing package.json.>>"%LOGFILE%" & popd & pause & exit /b 1 )

echo [+] Running npm dev (logging to %LOGFILE%)...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Continue'; npm run dev 2>&1 | Tee-Object -FilePath '%LOGFILE%' -Append"

set "EXITCODE=%ERRORLEVEL%"
popd

echo.
echo Finished. Exit code: %EXITCODE%
echo ==== Exit %DATE% %TIME% (code %EXITCODE%) ====>> "%LOGFILE%"
echo Log saved to: %LOGFILE%
pause
endlocal
