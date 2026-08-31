"""
CRESTA — Climate Resilience Ensemble Score & Topographic Analysis
Conformal bioclimatic niche analysis for QGIS.  v2.0.0
"""

__version__ = "2.0.0"


def classFactory(iface):
    from .cresta import CrestaPlugin
    return CrestaPlugin(iface)
