"""
CRESTA — Climate Resilience Ensemble Score & Topographic Analysis
==================================================================
QGIS Plugin  v1.0.0
Main plugin class: integrates into the QGIS menu,
opens the dialog window and triggers analysis.

Author : Ömer K. Örücü  (with Claude AI / Anthropic support)
License: GPL-3.0
"""

import os
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.core import QgsProject

from .main_dialog import ClimateResilienceDialog


class CrestaPlugin:

    def __init__(self, iface):
        self.iface      = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions    = []
        self.menu       = "&CRESTA"
        self.toolbar    = self.iface.addToolBar("CRESTA")
        self.toolbar.setObjectName("CrestaToolbar")

    def add_action(self, icon_path, text, callback, parent=None):
        icon   = QIcon(icon_path)
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
            text="CRESTA — Climate Resilience Analyzer",
            callback=self.run,
            parent=self.iface.mainWindow(),
        )

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu("&CRESTA", action)
            self.iface.removeToolBarIcon(action)
        del self.toolbar

    def run(self):
        dialog = ClimateResilienceDialog(self.iface)
        # exec_() was removed in Qt6; exec() works on both Qt5 and Qt6
        dialog.exec()
