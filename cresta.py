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
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QToolButton
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import Qt, QSettings, QTranslator, QCoreApplication
from qgis.core import QgsProject

from .main_dialog import ClimateResilienceDialog


# ── Qt5/Qt6 enum compatibility ──────────────────────────────────────────────
# Qt6 (PyQt6) requires fully-scoped enums (Qt.ToolButtonStyle.ToolButtonIconOnly).
# Qt5 (PyQt5) accepts both scoped and unscoped (_TB_ICON_ONLY).
def _qt_attr(cls, *names):
    for name in names:
        obj = cls
        try:
            for part in name.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None

_TB_ICON_ONLY = _qt_attr(Qt, 'ToolButtonStyle.ToolButtonIconOnly', 'ToolButtonIconOnly')


class CrestaPlugin:

    def __init__(self, iface):
        self.iface      = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions    = []
        self.menu       = "&CRESTA"
        self.toolbar    = self.iface.addToolBar("CRESTA")
        self.toolbar.setObjectName("CrestaToolbar")
        # Force icon-only display so the action shows just the icon, not text
        if _TB_ICON_ONLY is not None:
            try:
                self.toolbar.setToolButtonStyle(_TB_ICON_ONLY)
            except Exception:
                pass
        self.dialog = None

    def _resolve_icon(self):
        """Find an icon file (svg preferred, png fallback)."""
        for name in ("icon.svg", "icon.png"):
            p = os.path.join(self.plugin_dir, "resources", name)
            if os.path.exists(p):
                return p
        return None

    def add_action(self, icon_path, text, callback, parent=None):
        icon = QIcon(icon_path) if icon_path else QIcon()
        action = QAction(icon, text, parent)
        action.setToolTip(text)
        action.setStatusTip(text)
        action.triggered.connect(callback)
        self.toolbar.addAction(action)
        # Make sure the toolbar button shows icon only (no text label)
        if _TB_ICON_ONLY is not None:
            btn = self.toolbar.widgetForAction(action)
            if isinstance(btn, QToolButton):
                try:
                    btn.setToolButtonStyle(_TB_ICON_ONLY)
                except Exception:
                    pass
        self.iface.addPluginToMenu(self.menu, action)
        self.actions.append(action)
        return action

    def initGui(self):
        icon_path = self._resolve_icon()
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
        if self.dialog is not None:
            try:
                self.dialog.close()
            except Exception:
                pass
            self.dialog = None
        del self.toolbar

    def run(self):
        # Non-modal: keep a single instance and just bring it to the front
        # so the user can keep working on QGIS layers while it's open.
        if self.dialog is None:
            self.dialog = ClimateResilienceDialog(self.iface, self.iface.mainWindow())
            self.dialog.destroyed.connect(self._on_dialog_destroyed)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _on_dialog_destroyed(self, *args):
        self.dialog = None
