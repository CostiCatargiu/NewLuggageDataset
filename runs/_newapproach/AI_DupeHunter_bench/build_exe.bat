@echo off
REM Build AI_DupeHunter.
REM   build_exe.bat          -> FOLDER build  dist\AI_DupeHunter\AI_DupeHunter.exe  (starts instantly)
REM   build_exe.bat onefile  -> single file   dist\AI_DupeHunter.exe   (unpacks ~80 MB on every launch)

setlocal
cd /d "%~dp0"

set "SPEC=trek_gui_onedir.spec"
set "MODE=folder"
if /i "%~1"=="onefile" ( set "SPEC=trek_gui.spec" & set "MODE=single file" )

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found. Create it first:
    echo     py -3.12 -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

echo [1/3] Installing/verifying build dependencies...
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || exit /b 1

echo [2/3] Cleaning previous build (%MODE%)...
if exist "build\trek_gui" rmdir /s /q "build\trek_gui"
if exist "dist\AI_DupeHunter.exe" del /q "dist\AI_DupeHunter.exe"
if exist "dist\AI_DupeHunter" rmdir /s /q "dist\AI_DupeHunter"

echo [3/3] Building %MODE% (this takes a few minutes)...
".venv\Scripts\pyinstaller.exe" --noconfirm --clean "%SPEC%" || exit /b 1

REM Ship the (commented-out) data-folder config next to the exe so the
REM shared-cache option is visible without launching the app first.
if /i "%MODE%"=="folder" (
    ".venv\Scripts\python.exe" -c "import trek_paths,pathlib;p=pathlib.Path('dist/AI_DupeHunter/trek_data_dir.txt');p.write_text(trek_paths.CONFIG_TEMPLATE,encoding='utf-8') if not p.exists() else None"
) else (
    ".venv\Scripts\python.exe" -c "import trek_paths,pathlib;p=pathlib.Path('dist/trek_data_dir.txt');p.write_text(trek_paths.CONFIG_TEMPLATE,encoding='utf-8') if not p.exists() else None"
)

echo.
if /i "%MODE%"=="folder" (
    echo Done -^> dist\AI_DupeHunter\AI_DupeHunter.exe
    echo        Ship the WHOLE dist\AI_DupeHunter folder ^(zip it^). It starts
    echo        immediately -- nothing is unpacked at launch.
) else (
    echo Done -^> dist\AI_DupeHunter.exe   ^(single file; unpacks ~80 MB on every start^)
)
echo        Data ^(cache, log, projects^) is written NEXT TO the exe.
endlocal
