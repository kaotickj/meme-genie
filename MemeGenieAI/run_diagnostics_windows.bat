@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" diagnose_engine.py %*
) else (
  py -3 diagnose_engine.py %*
)
echo.
echo The diagnostic report is under %%LOCALAPPDATA%%\MemeGenieAI\diagnostics
pause
endlocal
