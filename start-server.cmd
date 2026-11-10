@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul

set "serverPython=%~dp0.venv-server\Scripts\python.exe"
set "serverConfig=%AUTOYKT_SERVER_CONFIG%"
if not defined serverConfig set "serverConfig=%LOCALAPPDATA%\AutoYKT\server\config.server.yaml"
set "serverAction=%~1"
if not defined serverAction set "serverAction=run"
set "serverExit=2"

if /i "%serverAction%"=="--help" goto help
if not exist "%serverPython%" (
  echo The Web answer environment is missing. See docs\server.md for Windows setup.
  goto finish
)
if /i "%serverAction%"=="init" goto initialize
for %%A in (check login probe run health) do (
  if /i "%serverAction%"=="%%A" (
    set "serverAction=%%A"
    goto execute
  )
)
goto usage_error

:initialize
"%serverPython%" -X utf8 -u -m autoykt.server init --path "%serverConfig%"
set "serverExit=%ERRORLEVEL%"
goto finish

:execute
if not exist "%serverConfig%" (
  echo Config "%serverConfig%" was not found. Run start-server.cmd init.
  goto finish
)
"%serverPython%" -X utf8 -u -m autoykt.server --config "%serverConfig%" %serverAction%
set "serverExit=%ERRORLEVEL%"
goto finish

:help
set "serverExit=0"
:usage_error
echo Usage: start-server.cmd [init ^| check ^| login ^| probe ^| run ^| health]
echo With no command, the Web answer service runs in this window.

:finish
if not "%serverExit%"=="0" if "%~1"=="" pause
exit /b %serverExit%
