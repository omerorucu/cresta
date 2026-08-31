"""
build_zip.py  —  Packages CRESTA into a QGIS-installable ZIP.

Usage:
    python build_zip.py

The version is read from metadata.txt so it can only be defined in one place
(v1.0's copy hard-coded it separately and drifted).  LICENSE is included
because GPL-3.0 requires the licence text to be distributed with the work.
"""

import os
import re
import zipfile

PLUGIN_NAME = "CRESTA"
BASE = os.path.dirname(os.path.abspath(__file__))


def read_version():
    with open(os.path.join(BASE, "metadata.txt"), encoding="utf-8") as f:
        m = re.search(r"^version=(.+)$", f.read(), re.MULTILINE)
    if not m:
        raise SystemExit("version= not found in metadata.txt")
    return m.group(1).strip()


VERSION = read_version()
OUT_ZIP = f"{PLUGIN_NAME}_v{VERSION}.zip"

INCLUDE = [
    "__init__.py",
    "cresta.py",
    "main_dialog.py",
    "analysis_engine.py",
    "metadata.txt",
    "requirements.txt",
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "test_engine.py",
    "resources",      # directory — all contents included
]

EXCLUDE_EXT  = {".pyc", ".pyo"}
EXCLUDE_DIRS = {"__pycache__"}


def _add(zf, src, arcname):
    if os.path.isdir(src):
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                if os.path.splitext(f)[1] in EXCLUDE_EXT:
                    continue
                fp  = os.path.join(root, f)
                rel = os.path.relpath(fp, start=os.path.dirname(src))
                zf.write(fp, os.path.join(PLUGIN_NAME, rel))
    else:
        zf.write(src, os.path.join(PLUGIN_NAME, arcname))


missing = []
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for item in INCLUDE:
        src = os.path.join(BASE, item)
        if os.path.exists(src):
            _add(zf, src, item)
        else:
            missing.append(item)
            print(f"  [SKIP] not found: {item}")

if "LICENSE" in missing:
    raise SystemExit("LICENSE is mandatory for a GPL-3.0 distribution.")

print(f"Created {OUT_ZIP}  (version {VERSION})")
