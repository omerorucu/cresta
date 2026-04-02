"""
CRESTA — Climate Resilience Ensemble Score & Topographic Analysis
QGIS Plugin  v1.0.0
"""


def classFactory(iface):
    from .cresta import CrestaPlugin
    return CrestaPlugin(iface)
