@echo off
rem One-time setup: creates .venv next to this file and installs the packages.
cd /d "%~dp0"
py -3.12 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Setup done. Start the app with run_bench.bat
pause
