# PyInstaller spec for AI DupeHunter -- FOLDER build (fast start).
#   build:  .venv\Scripts\pyinstaller --noconfirm trek_gui_onedir.spec
#   output: dist\AI_DupeHunter\AI_DupeHunter.exe  (+ its files)
#
# Why a folder: the single-file build packs ~80 MB into the .exe and
# UNPACKS it to a temp folder on EVERY launch -- that is the pause between
# double-clicking and the "Connecting to cache database..." splash. A
# folder build starts straight away (nothing to unpack); ship the folder
# (or a zip of it) instead of one .exe.
#
# PORTABLE by default: the cache, activity log, and project list are
# written NEXT TO the exe (see trek_paths.py data_dir()), so the whole
# folder can be copied to another PC or a USB stick and everything
# travels with it. Override via trek_data_dir.txt or TREK_DATA_DIR env.

block_cipher = None

a = Analysis(
    ["trek_gui.py"],
    pathex=["."],
    binaries=[],
    datas=[
        # Loaded at runtime by file path via importlib (see trek_gui._API_PATH),
        # so PyInstaller cannot discover it as an import -- ship it explicitly.
        ("tal/KeywordDrivenBase/Addons/WorkspaceUpdate/TrekExportLinksAPI.py",
         "tal/KeywordDrivenBase/Addons/WorkspaceUpdate"),
        # Window / taskbar icon, loaded at runtime via trek_paths.resource_file.
        ("dupehunter_icon.ico", "."),
    ],
    hiddenimports=[
        "PySide6.QtCharts",          # imported inside a try/except -> not auto-detected
        # Windows SSO auth chain. pywintypes pulls win32timezone in lazily at
        # first use, so PyInstaller's static analysis misses it and the app
        # only fails once it actually authenticates against TREK.
        "requests_negotiate_sspi",
        "win32timezone",
        "pywintypes",
        "sspi",
        "sspicon",
        "win32security",
        "win32api",
        "win32con",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim the Qt modules this app never touches -- keeps the exe far smaller.
    excludes=[
        "tkinter", "matplotlib", "pandas", "scipy", "PIL", "pytest",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
        "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtMultimedia",
        "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtDesigner",
        "PySide6.QtBluetooth", "PySide6.QtNetworkAuth", "PySide6.QtSensors",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # <- binaries go next to the exe, not inside it
    name="AI_DupeHunter",
    icon="dupehunter_icon.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                  # GUI app -- no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="AI_DupeHunter",
)
