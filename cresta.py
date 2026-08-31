"""
CRESTA — Climate Resilience Ensemble Score & Topographic Analysis
==================================================================
QGIS Plugin  v2.0.0
Main plugin class: integrates into the QGIS menu, checks that the scientific
stack is present, and opens the analysis dialog.

Author : Ömer K. Örücü
License: GPL-3.0
"""

import os
import sys

from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon


# --------------------------------------------------------------------------
#  Dependency check
#
#  scikit-learn is NOT part of a standard QGIS install (verified on QGIS
#  3.40 LTR and 4.2 on Windows).  v1.0's README claimed otherwise, so a user
#  without it got a raw ImportError traceback in the Python console instead
#  of an explanation.
# --------------------------------------------------------------------------

REQUIRED = [
    ("numpy",   "1.21"),
    ("scipy",   "1.7"),
    ("sklearn", "1.0"),
]


def missing_dependencies():
    """Return the list of required packages that cannot be imported."""
    missing = []
    for mod, minver in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            missing.append((mod, minver))
    return missing


def _install_hint(missing):
    names = {"sklearn": "scikit-learn"}
    pkgs = " ".join(names.get(m, m) + ">=" + v for m, v in missing)
    exe = sys.executable or "python"
    return (
        "CRESTA needs the following Python packages, which are not present in "
        "this QGIS installation:\n\n"
        "    " + ", ".join(names.get(m, m) for m, _ in missing) + "\n\n"
        "Install them into the QGIS Python environment, then restart QGIS.\n\n"
        "Windows — open the OSGeo4W Shell as administrator and run:\n"
        "    python -m pip install " + pkgs + "\n\n"
        "Linux / macOS:\n"
        "    \"" + exe + "\" -m pip install " + pkgs + "\n\n"
        "matplotlib is optional; without it the Charts tab is hidden."
    )


class CrestaPlugin:

    def __init__(self, iface):
        self.iface      = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions    = []
        self.menu       = "&CRESTA"
        self.toolbar    = self.iface.addToolBar("CRESTA")
        self.toolbar.setObjectName("CrestaToolbar")

    def add_action(self, icon_path, text, callback, parent=None):
        icon   = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        self.toolbar.addAction(action)
        self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "resources", "icon.png")
        self.add_action(
            icon_path,
            text="CRESTA — Bioclimatic Niche Analyzer",
            callback=self.run,
            parent=self.iface.mainWindow(),
        )

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu("&CRESTA", action)
            self.iface.removeToolBarIcon(action)
        del self.toolbar

    def run(self):
        missing = missing_dependencies()
        if missing:
            QMessageBox.critical(self.iface.mainWindow(),
                                 "CRESTA — missing dependencies",
                                 _install_hint(missing))
            return

        # Imported lazily so the dependency dialog above can be shown first.
        from .main_dialog import ClimateResilienceDialog
        dialog = ClimateResilienceDialog(self.iface)
        # exec_() was removed in Qt6; exec() works on both Qt5 and Qt6.
        dialog.exec()
