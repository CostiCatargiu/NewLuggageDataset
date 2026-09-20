@echo off
REM Build the standalone AI_DupeHunter.exe (see trek_gui.spec).
REM Run from this folder with the project's .venv present.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Create it first:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

echo [1/3] Installing/verifying build dependencies...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || exit /b 1

echo [2/3] Cleaning previous build...
if exist "build\trek_gui" rmdir /s /q "build\trek_gui"
if exist "dist\AI_DupeHunter.exe" del /q "dist\AI_DupeHunter.exe"

echo [3/3] Building (this takes a few minutes)...
".venv\Scripts\pyinstaller.exe" --noconfirm --clean trek_gui.spec || exit /b 1

REM Ship the (commented-out) data-folder config next to the exe so the
REM shared-cache option is visible without launching the app first.
".venv\Scripts\python.exe" -c "import trek_paths,pathlib;p=pathlib.Path('dist/trek_data_dir.txt');p.write_text(trek_paths.CONFIG_TEMPLATE,encoding='utf-8') if not p.exists() else None"

echo.
echo Done -^> dist\AI_DupeHunter.exe
echo        dist\trek_data_dir.txt   (edit to share data with colleagues)
echo.
echo Ship BOTH files. Data (cache, log, projects) is written NEXT TO the exe.
endlocal
