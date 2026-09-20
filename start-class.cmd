@echo off
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo AutoYKT environment is missing. Install Python 3.12 and follow README.md for setup.
  exit /b 2
)
pushd "%~dp0"
"%~dp0.venv\Scripts\python.exe" -X utf8 -u -m autoykt session %*
set "quizExit=%ERRORLEVEL%"
popd
exit /b %quizExit%
