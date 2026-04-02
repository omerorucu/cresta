"""
build_zip.py  —  Packages CRESTA into a QGIS-installable ZIP.

Usage:
    python build_zip.py

Output:
    CRESTA_v1.0.0.zip  (ready to install via QGIS Plugin Manager)
"""

import os
import zipfile
import shutil

PLUGIN_NAME = "CRESTA"
VERSION     = "1.0.0"
OUT_ZIP     = f"{PLUGIN_NAME}_v{VERSION}.zip"

# Files/dirs to include (relative to this script's directory)
INCLUDE = [
    "__init__.py",
    "cresta.py",
    "main_dialog.py",
    "analysis_engine.py",
    "metadata.txt",
    "resources",      # directory — all contents included
]

# Always exclude these
EXCLUDE_EXT  = {".pyc", ".pyo"}
EXCLUDE_DIRS = {"__pycache__"}


def _add(zf: zipfile.ZipFile, src: str, arcname: str):
    if os.path.isdir(src):
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if os.path.splitext(f)[1] in EXCLUDE_EXT:
                    continue
                fp   = os.path.join(root, f)
                rel  = os.path.relpath(fp, start=os.path.dirname(src))
                zf.write(fp, os.path.join(PLUGIN_NAME, rel))
    else:
        zf.write(src, os.path.join(PLUGIN_NAME, arcname))


base = os.path.dirname(os.path.abspath(__file__))
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in INCLUDE:
        src = os.path.join(base, item)
        if os.path.exists(src):
            _add(zf, src, item)
        else:
            print(f"  [SKIP] not found: {item}")

print(f"✅ Created {OUT_ZIP}")
