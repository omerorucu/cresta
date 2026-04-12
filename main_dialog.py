"""
Climate Resilience Analyzer  v1.0.0  —  Main Dialog
====================================================
QGIS dialog for ML + Multi-Threshold Score system.
"""

import os, json, csv
import numpy as np

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QFileDialog, QProgressBar, QGroupBox,
    QMessageBox, QHeaderView, QCheckBox, QSpinBox, QFrame,
    QButtonGroup, QRadioButton, QStackedWidget, QLineEdit, QSizePolicy,
    QScrollArea, QSplitter
)
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont, QPainter, QPen, QIcon, QPixmap
from qgis.core import (
    QgsProject, QgsMapLayer,
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY,
    QgsField, QgsFields,
    QgsMarkerSymbol, QgsSingleSymbolRenderer,
    QgsCoordinateReferenceSystem,
)
try:
    from qgis.PyQt.QtCore import QVariant
except ImportError:
    from PyQt5.QtCore import QVariant

# ── Qt5/Qt6 compatibility shims ─────────────────────────────────────────────
# Qt6 (PyQt6) requires fully-scoped enums; Qt5 supports both forms.
# These helpers ensure the plugin loads on QGIS 3.x (Qt5) AND QGIS 4.x (Qt6).
def _qt_enum(cls, *names):
    """Return the first resolvable scoped or unscoped enum member."""
    for name in names:
        parts = name.split('.')
        obj = cls
        try:
            for p in parts:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"Cannot resolve {cls.__name__}.{names}")

Qt_AlignCenter          = _qt_enum(Qt, 'AlignmentFlag.AlignCenter', 'AlignCenter')
QHeaderView_Stretch     = _qt_enum(QHeaderView, 'ResizeMode.Stretch',     'Stretch')
QHeaderView_ResizeFit   = _qt_enum(QHeaderView, 'ResizeMode.ResizeToContents', 'ResizeToContents')
QSizePolicy_Expanding   = _qt_enum(QSizePolicy, 'Policy.Expanding', 'Expanding')
QSizePolicy_Fixed       = _qt_enum(QSizePolicy, 'Policy.Fixed',     'Fixed')

# QFrame shape / shadow — Qt6 requires scoped enums (QFrame.Shape.Box, QFrame.Shadow.Raised)
QFrame_Box              = _qt_enum(QFrame, 'Shape.Box',    'Box')
QFrame_Raised           = _qt_enum(QFrame, 'Shadow.Raised','Raised')
QFrame_HLine            = _qt_enum(QFrame, 'Shape.HLine',  'HLine')
QFrame_NoFrame          = _qt_enum(QFrame, 'Shape.NoFrame','NoFrame')

# QFont weight — Qt6 uses QFont.Weight.Bold / QFont.Weight.Normal
QFont_Bold              = _qt_enum(QFont, 'Weight.Bold',   'Bold')
QFont_Normal            = _qt_enum(QFont, 'Weight.Normal', 'Normal')

# QPainter render hint — Qt6 uses QPainter.RenderHint.Antialiasing
QPainter_Antialiasing   = _qt_enum(QPainter, 'RenderHint.Antialiasing', 'Antialiasing')

# QTabWidget tab position — Qt6 uses QTabWidget.TabPosition.West
QTabWidget_West         = _qt_enum(QTabWidget, 'TabPosition.West', 'West')

try:
    QgsMapLayer_VectorLayer = QgsMapLayer.LayerType.VectorLayer   # QGIS 4 / Qt6
except AttributeError:
    QgsMapLayer_VectorLayer = QgsMapLayer.VectorLayer             # QGIS 3 / Qt5 fallback
# ────────────────────────────────────────────────────────────────────────────

try:
    import matplotlib
    # Auto-select Qt backend for QGIS 3 (Qt5) and QGIS 4 (Qt6) compatibility
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        matplotlib.use("QtAgg")  # matplotlib >= 3.5: backend_qtagg works on both Qt5+Qt6
    except ImportError:
        matplotlib.use("Qt5Agg")
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .analysis_engine import (
    ClimateResilienceAnalyzer,
    BIO_NAMES, TOPO_NAMES, ALL_VAR_NAMES, OPT_VAR_NAMES,
    ALL_BIO_COLS, ALL_TOPO_COLS, ALL_OPT_COLS,
    BIO_COLS, TOPO_COLS, ALL_COLS, CRITICAL_BIOS, CRITICAL_OPT,
    W_THRESHOLD_ZONE, W_GMM, W_ISOFOREST, W_OCSVM, W_MAHAL,
)


# ── Data readers ──────────────────────────────────────────────────────────

def _normalize_col(name: str) -> str:
    """Normalises column name for comparison."""
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def _find_col_index(fields_lower: list, col: str):
    """Flexible column matching: case, underscore, BIO_ prefix."""
    targets = [
        _normalize_col(col),
        _normalize_col(col.upper()),
        _normalize_col(col.replace("bio", "BIO_")),
        _normalize_col(f"_{col}"),
        _normalize_col(col.replace("bio", "Bio")),
    ]
    for t in targets:
        if t in fields_lower:
            return fields_lower.index(t)
    return None


_COORD_LAT_NAMES = {"lat","latitude","y","coord_y","northing","ylat","lat_dd","y_coord","enlem"}
_COORD_LON_NAMES = {"lon","longitude","x","coord_x","easting","xlon","lon_dd","x_coord","long","boylam"}
_SPECIES_COL_NAMES = {
    "species","species_name","scientific_name","scientificname",
    "taxon","taxon_name","binomial","binomen",
    "tur","tur_adi","tür","tür_adı","bilimsel_ad","ad","isim",
}


def _detect_latlon(fields_lower: list):
    """Returns (lat_idx, lon_idx) or (None, None) if no coords found."""
    lat_idx = next((i for i,f in enumerate(fields_lower) if f in _COORD_LAT_NAMES), None)
    lon_idx = next((i for i,f in enumerate(fields_lower) if f in _COORD_LON_NAMES), None)
    return lat_idx, lon_idx


def _detect_species_col(fields_lower: list):
    """Returns index of the species-name column or None."""
    for i, f in enumerate(fields_lower):
        if f in _SPECIES_COL_NAMES:
            return i
    return None


def _pick_species_name(raw_values):
    """
    Given a list of species strings from a CSV/layer, return the most common
    non-empty name — or None if none found.  Handles rows with mixed case
    or trailing whitespace gracefully.
    """
    from collections import Counter
    cleaned = []
    for v in raw_values:
        if v is None:
            continue
        try:
            s = str(v).strip()
        except Exception:
            continue
        if s and s.lower() not in ("na", "n/a", "null", "none", ""):
            cleaned.append(s)
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _sanitize_species_name(name: str) -> str:
    """
    Makes a species name safe for use in filenames and layer names.
    Keeps letters, digits, underscores, hyphens; collapses whitespace.
    """
    if not name:
        return ""
    import re
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._\-]", "", s)
    return s[:60]


def _read_csv_layer(layer, expected_cols, skip_null=True):
    """Reads expected columns from a QGIS vector layer into a numpy array.
    Returns ``(data_array, coords_or_None, species_name_or_None)``."""
    fields_raw   = [f.name() for f in layer.fields()]
    fields_lower = [_normalize_col(f) for f in fields_raw]

    col_indices, missing = [], []
    for col in expected_cols:
        idx = _find_col_index(fields_lower, col)
        if idx is None: missing.append(col)
        else:           col_indices.append(idx)

    if missing:
        avail = ", ".join(fields_raw[:15]) + ("..." if len(fields_raw) > 15 else "")
        raise ValueError(f"Missing columns: {', '.join(missing)}\nAvailable: {avail}")

    lat_idx, lon_idx = _detect_latlon(fields_lower)
    sp_idx           = _detect_species_col(fields_lower)

    rows, lats, lons, sp_vals = [], [], [], []
    for feat in layer.getFeatures():
        # Try geometry first, then attribute columns
        geom = feat.geometry()
        if geom and not geom.isNull():
            pt = geom.asPoint()
            _lat, _lon = pt.y(), pt.x()
        else:
            _lat = _lon = None

        attrs = feat.attributes()
        row   = [attrs[i] for i in col_indices]
        if skip_null and any(v is None for v in row): continue
        rows.append([float(v) if v is not None else 0.0 for v in row])

        if _lat is None and lat_idx is not None:
            try: _lat = float(attrs[lat_idx])
            except Exception: pass
        if _lon is None and lon_idx is not None:
            try: _lon = float(attrs[lon_idx])
            except Exception: pass
        lats.append(_lat); lons.append(_lon)

        if sp_idx is not None:
            try: sp_vals.append(attrs[sp_idx])
            except Exception: sp_vals.append(None)

    if not rows: raise ValueError("No valid rows could be extracted from layer.")
    coords = (lats, lons) if any(v is not None for v in lats) else None
    species = _pick_species_name(sp_vals) if sp_vals else None
    return np.array(rows), coords, species


def _read_csv_file(path: str, expected_cols: list, skip_null=True):
    """
    Reads expected columns from a CSV file on disk.
    Returns ``(data_array, coords_or_None, species_name_or_None)``.

    Supported delimiters: comma, semicolon, tab.
    Column names are case-insensitive; spaces/hyphens/underscores flexible.
    Lines starting with '#' are treated as comments and skipped.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect delimiter
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                sample_line = stripped
                break
        else:
            sample_line = ""
    try:
        dialect = csv.Sniffer().sniff(sample_line, delimiters=",;\t")
        sep = dialect.delimiter
    except csv.Error:
        counts = {d: sample_line.count(d) for d in (",", ";", "\t")}
        sep = max(counts, key=counts.get) if max(counts.values()) > 0 else ","

    rows_raw = []
    header   = None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        for line in reader:
            if not line or line[0].startswith("#"):
                continue
            if header is None:
                header = [_normalize_col(h) for h in line]
                continue
            rows_raw.append(line)

    if header is None:
        raise ValueError(f"CSV header row not found: {path}")

    # Column matching
    col_indices, missing = [], []
    for col in expected_cols:
        idx = _find_col_index(header, col)
        if idx is None: missing.append(col)
        else:           col_indices.append(idx)

    if missing:
        avail = ", ".join(header[:15]) + ("..." if len(header) > 15 else "")
        raise ValueError(
            f"Missing columns in CSV: {', '.join(missing)}\n"
            f"File header: {avail}"
        )

    # Detect lat/lon + species columns
    lat_idx, lon_idx = _detect_latlon(header)
    sp_idx           = _detect_species_col(header)

    rows, lats, lons, sp_vals = [], [], [], []
    for line in rows_raw:
        if len(line) <= max(col_indices):
            continue
        try:
            vals = [line[i].strip() for i in col_indices]
            if skip_null and any(v in ("", "NA", "N/A", "null", "NULL", "None") for v in vals):
                continue
            rows.append([float(v) for v in vals])
            try:
                lats.append(float(line[lat_idx]) if lat_idx is not None and lat_idx < len(line) else None)
                lons.append(float(line[lon_idx]) if lon_idx is not None and lon_idx < len(line) else None)
            except (ValueError, TypeError):
                lats.append(None); lons.append(None)
            if sp_idx is not None and sp_idx < len(line):
                sp_vals.append(line[sp_idx])
        except (ValueError, IndexError):
            continue

    if not rows:
        raise ValueError(f"No valid rows could be read from CSV: {path}")

    coords  = (lats, lons) if any(v is not None for v in lats) else None
    species = _pick_species_name(sp_vals) if sp_vals else None
    return np.array(rows), coords, species


# ── Worker ────────────────────────────────────────────────────────────────────
class AnalysisWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)
    log      = pyqtSignal(str)   # streams engine-side progress lines

    def __init__(self, species_data, target_values,
                 bio_cols, topo_cols, opt_cols):
        super().__init__()
        self.species_data  = species_data
        self.target_values = target_values
        self.bio_cols      = bio_cols
        self.topo_cols     = topo_cols
        self.opt_cols      = opt_cols

    def _emit_log(self, msg):
        # Forwarded to the log panel via Qt signal (thread-safe).
        try: self.log.emit(str(msg))
        except Exception: pass

    def run(self):
        # Steps that accept a ``log_callback`` — used to stream intra-step
        # progress into the UI log panel.  Anything not in this set runs
        # silently and simply advances the progress bar.
        steps = [
            ( 8,  "Computing PCA...",                        "compute_pca",                   False),
            (16,  "Kernel PCA...",                           "compute_kernel_pca",            False),
            (26,  "GMM niche model...",                      "compute_gmm",                   False),
            (36,  "Isolation Forest...",                     "compute_isolation_forest",      False),
            (46,  "One-Class SVM...",                        "compute_ocsvm",                 False),
            (54,  "K-Means clustering...",                   "compute_kmeans_niche",          False),
            (61,  "Mahalanobis thresholds...",               "compute_mahalanobis",           False),
            (68,  "Multi-threshold zone voting...",          "compute_threshold_zone",        False),
            (74,  "Percentile analysis (PDI)...",            "compute_percentile_analysis",   False),
            (79,  "Topographic compatibility...",            "compute_topo_score",            False),
            (84,  "Variable importance + risk...",           "compute_variable_importance",   False),
            (88,  "Composite score...",                      "compute_composite_score",       False),
            (90,  "Environmental thinning (spatial AC)...",  "apply_environmental_thinning",  False),
            (92,  "Blocked cross-validation...",             "compute_blocked_cv",            False),
            (94,  "Spatial autocorrelation correction...",   "apply_spatial_correction",      True),
            (97,  "Full bootstrap CI (200× model refit)...", "compute_bootstrap_ci_full",     True),
            (100, "Sensitivity & weight optimization...",    "compute_sensitivity_analysis",  True),
        ]
        try:
            az = ClimateResilienceAnalyzer(
                self.species_data, self.target_values,
                bio_cols=self.bio_cols,
                topo_cols=self.topo_cols,
                opt_cols=self.opt_cols,
            )
            self._emit_log(
                f"[worker] n_species={len(self.species_data)}  "
                f"bio={len(self.bio_cols)}  topo={len(self.topo_cols)}  "
                f"opt={len(self.opt_cols)}")
            for pct, msg, fn_name, use_log in steps:
                self.progress.emit(pct, msg)
                self._emit_log(f"[step {pct:>3}%] {msg}")
                fn = getattr(az, fn_name)
                if use_log:
                    fn(log_callback=self._emit_log)
                else:
                    fn()
            self._emit_log("[worker] analysis complete ✔")
            self.finished.emit(az.results)
        except Exception as e:
            self._emit_log(f"[worker] error: {e}")
            self.error.emit(str(e))


# ── Gauge ─────────────────────────────────────────────────────────────────────
class ScoreGauge(QWidget):
    def __init__(self, score=0, parent=None):
        super().__init__(parent)
        self.score = score
        self.setMinimumSize(190, 190)

    def set_score(self, score):
        self.score = score; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter_Antialiasing)
        w, h  = self.width(), self.height()
        size  = min(w,h)-20; x,y = (w-size)//2, (h-size)//2
        color = (QColor("#27ae60") if self.score>=80 else
                 QColor("#2ecc71") if self.score>=65 else
                 QColor("#f39c12") if self.score>=50 else
                 QColor("#e67e22") if self.score>=35 else
                 QColor("#e74c3c"))
        pen = QPen(QColor("#dfe6e9"), 14)
        p.setPen(pen); p.drawArc(x,y,size,size,225*16,-270*16)
        pen.setColor(color); p.setPen(pen)
        p.drawArc(x,y,size,size,225*16,int(-270*16*self.score/100))
        p.setPen(QPen(QColor("#2d3436")))
        p.setFont(QFont("Arial", int(size*0.18), QFont_Bold))
        p.drawText(x,y,size,size,Qt_AlignCenter,f"{self.score:.1f}")
        p.setFont(QFont("Arial",int(size*0.08)))
        p.setPen(QPen(QColor("#636e72")))
        p.drawText(x,y+int(size*0.35),size,size,Qt_AlignCenter,"/ 100")


# ── Main Dialog ────────────────────────────────────────────────────────────────
class ClimateResilienceDialog(QDialog):

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface   = iface
        self.results = None
        self.worker  = None

        # Make the dialog behave like a top-level window with full window
        # controls (minimize / maximize / close) and non-modal behavior so
        # the user can interact with QGIS layers while it's open.
        # Qt6 requires scoped enums; use the _qt_enum helper for compatibility.
        _Window           = _qt_enum(Qt, 'WindowType.Window',                  'Window')
        _MinHint          = _qt_enum(Qt, 'WindowType.WindowMinimizeButtonHint','WindowMinimizeButtonHint')
        _MaxHint          = _qt_enum(Qt, 'WindowType.WindowMaximizeButtonHint','WindowMaximizeButtonHint')
        _CloseHint        = _qt_enum(Qt, 'WindowType.WindowCloseButtonHint',   'WindowCloseButtonHint')
        _SysMenuHint      = _qt_enum(Qt, 'WindowType.WindowSystemMenuHint',    'WindowSystemMenuHint')
        _TitleHint        = _qt_enum(Qt, 'WindowType.WindowTitleHint',         'WindowTitleHint')
        try:
            self.setWindowFlags(
                _Window | _MinHint | _MaxHint | _CloseHint | _SysMenuHint | _TitleHint
            )
        except Exception:
            pass

        self.setModal(False)
        try:
            self.setWindowModality(_qt_enum(Qt, 'WindowModality.NonModal', 'NonModal'))
        except Exception:
            pass

        # Don't destroy on close — let the plugin reuse the same instance
        try:
            _wa_del = _qt_enum(Qt, 'WidgetAttribute.WA_DeleteOnClose', 'WA_DeleteOnClose')
            self.setAttribute(_wa_del, False)
        except Exception:
            pass

        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(
            "🌿 Climate Resilience Analyzer v1.0.0  |  ML + Multi-Threshold Score"
        )
        # Window icon (svg preferred, png fallback)
        _res_dir = os.path.join(os.path.dirname(__file__), "resources")
        for _name in ("icon.svg", "icon.png"):
            _icon_path = os.path.join(_res_dir, _name)
            if os.path.exists(_icon_path):
                self.setWindowIcon(QIcon(_icon_path))
                break
        self.setMinimumSize(900, 650); self.resize(1150, 780)
        main = QVBoxLayout(self)

        title = QLabel(
            "🌿 Climate Resilience Analyzer  v1.0.0  ·  "
            "GMM · Isolation Forest · One-Class SVM · Mahalanobis · PCA"
        )
        title.setAlignment(Qt_AlignCenter)
        title.setStyleSheet(
            "font-size:14px;font-weight:bold;color:#1a5276;"
            "padding:8px;background:#d4e6f1;border-radius:4px;"
        )
        main.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_input_tab(),          "📂 Data Input")
        self.tabs.addTab(self._build_results_tab(),        "📊 Summary")
        self.tabs.addTab(self._build_ml_tab(),             "🤖 ML Models")
        self.tabs.addTab(self._build_bio_tab(),            "🔬 Bio Detail")
        self.tabs.addTab(self._build_topo_tab(),           "🏔 Topography")
        self.tabs.addTab(self._build_opt_tab(),            "🌤 Optional")
        self.tabs.addTab(self._build_importance_tab(),     "📌 Variable Importance")
        self.tabs.addTab(self._build_risk_detail_tab(),    "⚠️ Risk Details")
        self.tabs.addTab(self._build_ssp_tab(),            "🌍 SSP Scenarios")
        self.tabs.addTab(self._build_calibration_tab(),    "⚙️ Weight Calibration")
        if HAS_MPL:
            self.tabs.addTab(self._build_chart_tab(), "📈 Charts")
        self.tabs.addTab(self._build_report_tab(),   "📄 Report")
        main.addWidget(self.tabs)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color:#636e72;font-size:11px;")
        main.addWidget(self.progress_label)

        self.progress = QProgressBar(); self.progress.setVisible(False)
        main.addWidget(self.progress)

        # ── Live analysis log panel ───────────────────────────────────────
        # Streams engine-side progress (bootstrap iters, sensitivity folds,
        # spatial correction phases) so the user can see what is happening
        # during the slow 95–100 % region instead of a frozen progress bar.
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.setVisible(False)
        self.log_view.setStyleSheet(
            "QTextEdit { background:#1e272e; color:#d2dae2; "
            "font-family:Consolas,'Courier New',monospace; "
            "font-size:10px; border:1px solid #485460; border-radius:3px; }"
        )
        main.addWidget(self.log_view)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setStyleSheet(
            "background:#27ae60;color:white;font-weight:bold;"
            "padding:8px 22px;border-radius:4px;"
        )
        self.btn_run.clicked.connect(self._run_analysis)
        btn_row.addWidget(self.btn_run)

        self.btn_json = QPushButton("💾 JSON"); self.btn_json.setEnabled(False)
        self.btn_json.clicked.connect(self._export_json)
        btn_row.addWidget(self.btn_json)

        self.btn_csv = QPushButton("📋 CSV"); self.btn_csv.setEnabled(False)
        self.btn_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(self.btn_csv)

        self.btn_full_report = QPushButton("📄 Export Report (.txt)")
        self.btn_full_report.setEnabled(False)
        self.btn_full_report.clicked.connect(self._export_full_report)
        btn_row.addWidget(self.btn_full_report)

        btn_close = QPushButton("✖ Close"); btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_close)
        main.addLayout(btn_row)

    # ── Input Tab ─────────────────────────────────────────────────────────────
    def _build_input_tab(self):
        """
        Dual-source mode (QGIS Layer / CSV File) for both datasets.
        Wrapped in QScrollArea so the Analyze button is always reachable.
        """
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QFrame_NoFrame)

        w = QWidget(); lay = QVBoxLayout(w)
        lay.setSpacing(5)
        lay.setContentsMargins(6, 6, 6, 6)

        # ══════════════════════════════════════════════════════════════════
        # 1️⃣  NATIVE RANGE POINTS
        # ══════════════════════════════════════════════════════════════════
        sp_grp = QGroupBox("1️⃣  Species Native Range Points")
        sp_main = QVBoxLayout(sp_grp)

        sp_mode_row = QHBoxLayout()
        self.rb_sp_layer = QRadioButton("QGIS Layer")
        self.rb_sp_file  = QRadioButton("CSV File")
        self.rb_sp_layer.setChecked(True)
        self._bg_sp = QButtonGroup(self)
        self._bg_sp.addButton(self.rb_sp_layer, 0)
        self._bg_sp.addButton(self.rb_sp_file,  1)
        sp_mode_row.addWidget(self.rb_sp_layer); sp_mode_row.addWidget(self.rb_sp_file)
        sp_mode_row.addStretch(); sp_main.addLayout(sp_mode_row)

        self.stack_sp = QStackedWidget()
        pg_sp_lyr = QWidget(); pg_sp_lyr_lay = QHBoxLayout(pg_sp_lyr)
        pg_sp_lyr_lay.setContentsMargins(0,0,0,0)
        pg_sp_lyr_lay.addWidget(QLabel("Layer:"))
        self.cb_sp = QComboBox(); self.cb_sp.setMinimumWidth(300)
        pg_sp_lyr_lay.addWidget(self.cb_sp)
        btn_ref = QPushButton("🔄 Refresh"); btn_ref.setMaximumWidth(80)
        btn_ref.clicked.connect(self._refresh_layers)
        pg_sp_lyr_lay.addWidget(btn_ref); pg_sp_lyr_lay.addStretch()
        self.stack_sp.addWidget(pg_sp_lyr)

        pg_sp_file = QWidget(); pg_sp_file_lay = QHBoxLayout(pg_sp_file)
        pg_sp_file_lay.setContentsMargins(0,0,0,0)
        pg_sp_file_lay.addWidget(QLabel("File:"))
        self.le_sp_path = QLineEdit(); self.le_sp_path.setPlaceholderText("Select CSV file…")
        self.le_sp_path.setReadOnly(True)
        self.le_sp_path.setSizePolicy(QSizePolicy_Expanding, QSizePolicy_Fixed)
        pg_sp_file_lay.addWidget(self.le_sp_path)
        btn_sp_browse = QPushButton("📂 Browse")
        btn_sp_browse.clicked.connect(lambda: self._browse_csv(self.le_sp_path))
        pg_sp_file_lay.addWidget(btn_sp_browse)
        btn_sp_clear = QPushButton("✖"); btn_sp_clear.setMaximumWidth(28)
        btn_sp_clear.clicked.connect(lambda: self.le_sp_path.clear())
        pg_sp_file_lay.addWidget(btn_sp_clear)
        self.stack_sp.addWidget(pg_sp_file)

        sp_main.addWidget(self.stack_sp)
        self._bg_sp.idClicked.connect(self.stack_sp.setCurrentIndex)
        lay.addWidget(sp_grp)

        # ══════════════════════════════════════════════════════════════════
        # 2️⃣  TARGET SITE
        # ══════════════════════════════════════════════════════════════════
        tgt_grp = QGroupBox("2️⃣  Target Site")
        tgt_main = QVBoxLayout(tgt_grp)

        tgt_mode_row = QHBoxLayout()
        self.rb_tgt_layer = QRadioButton("QGIS Layer")
        self.rb_tgt_file  = QRadioButton("CSV File")
        self.rb_tgt_layer.setChecked(True)
        self._bg_tgt = QButtonGroup(self)
        self._bg_tgt.addButton(self.rb_tgt_layer, 0)
        self._bg_tgt.addButton(self.rb_tgt_file,  1)
        tgt_mode_row.addWidget(self.rb_tgt_layer); tgt_mode_row.addWidget(self.rb_tgt_file)
        tgt_mode_row.addStretch(); tgt_main.addLayout(tgt_mode_row)

        self.stack_tgt = QStackedWidget()
        pg_tgt_lyr = QWidget(); pg_tgt_lyr_lay = QHBoxLayout(pg_tgt_lyr)
        pg_tgt_lyr_lay.setContentsMargins(0,0,0,0)
        pg_tgt_lyr_lay.addWidget(QLabel("Layer:"))
        self.cb_tgt = QComboBox(); self.cb_tgt.setMinimumWidth(300)
        pg_tgt_lyr_lay.addWidget(self.cb_tgt); pg_tgt_lyr_lay.addStretch()
        self.stack_tgt.addWidget(pg_tgt_lyr)

        pg_tgt_file = QWidget(); pg_tgt_file_lay = QHBoxLayout(pg_tgt_file)
        pg_tgt_file_lay.setContentsMargins(0,0,0,0)
        pg_tgt_file_lay.addWidget(QLabel("File:"))
        self.le_tgt_path = QLineEdit(); self.le_tgt_path.setPlaceholderText("Select CSV file…")
        self.le_tgt_path.setReadOnly(True)
        self.le_tgt_path.setSizePolicy(QSizePolicy_Expanding, QSizePolicy_Fixed)
        pg_tgt_file_lay.addWidget(self.le_tgt_path)
        btn_tgt_browse = QPushButton("📂 Browse")
        btn_tgt_browse.clicked.connect(lambda: self._browse_csv(self.le_tgt_path))
        pg_tgt_file_lay.addWidget(btn_tgt_browse)
        btn_tgt_clear = QPushButton("✖"); btn_tgt_clear.setMaximumWidth(28)
        btn_tgt_clear.clicked.connect(lambda: self.le_tgt_path.clear())
        pg_tgt_file_lay.addWidget(btn_tgt_clear)
        self.stack_tgt.addWidget(pg_tgt_file)

        tgt_main.addWidget(self.stack_tgt)
        tgt_main.addWidget(QLabel(
            "ℹ️  If multiple rows are present, each is treated as a target; their mean is used."))
        self._bg_tgt.idClicked.connect(self.stack_tgt.setCurrentIndex)
        lay.addWidget(tgt_grp)

        # ══════════════════════════════════════════════════════════════════
        # 3️⃣  BIOCLIMATIC VARIABLE SELECTION
        # ══════════════════════════════════════════════════════════════════
        bio_grp = QGroupBox("3️⃣  Bioclimatic Variables  (select which to include in analysis)")
        bio_outer = QVBoxLayout(bio_grp)

        # Hint
        bio_hint = QLabel(
            "💡  bio4 · bio5 · bio6 · bio14 · bio15 · bio17 are marked as critical "
            "(highlighted) and receive double weight in scoring.")
        bio_hint.setWordWrap(True)
        bio_hint.setStyleSheet(
            "background:#eaf4fb;border:1px solid #aed6f1;"
            "padding:4px 8px;border-radius:3px;font-size:11px;color:#154360;")
        bio_outer.addWidget(bio_hint)

        # Bio checkboxes — flat 2-row layout:
        #   Row 0: 🌡 label + BIO1…BIO11  (temperature)
        #   Row 1: 🌧 label + BIO12…BIO19 (precipitation)
        bio_flat = QGridLayout()
        bio_flat.setSpacing(3)
        bio_flat.setContentsMargins(2, 2, 2, 2)

        self._chk_bio = {}
        TEMP_BIOS = [f"bio{i}" for i in range(1, 12)]
        PREC_BIOS = [f"bio{i}" for i in range(12, 20)]
        CRITICAL_BIO_SET = {"bio4","bio5","bio6","bio14","bio15","bio17"}

        # Row 0 — temperature label then bio1-11
        lbl_temp = QLabel("🌡 Temp:")
        lbl_temp.setStyleSheet("font-size:10px;color:#555;")
        bio_flat.addWidget(lbl_temp, 0, 0)
        for col_idx, bio in enumerate(TEMP_BIOS, start=1):
            chk = QCheckBox(bio.upper())
            chk.setChecked(True)
            chk.setToolTip(BIO_NAMES[bio])
            if bio in CRITICAL_BIO_SET:
                chk.setStyleSheet("color:#1a5276;font-weight:bold;font-size:10px;")
            else:
                chk.setStyleSheet("font-size:10px;")
            self._chk_bio[bio] = chk
            bio_flat.addWidget(chk, 0, col_idx)

        # Row 1 — precipitation label then bio12-19
        lbl_prec = QLabel("🌧 Prec:")
        lbl_prec.setStyleSheet("font-size:10px;color:#555;")
        bio_flat.addWidget(lbl_prec, 1, 0)
        for col_idx, bio in enumerate(PREC_BIOS, start=1):
            chk = QCheckBox(bio.upper())
            chk.setChecked(True)
            chk.setToolTip(BIO_NAMES[bio])
            if bio in CRITICAL_BIO_SET:
                chk.setStyleSheet("color:#1a5276;font-weight:bold;font-size:10px;")
            else:
                chk.setStyleSheet("font-size:10px;")
            self._chk_bio[bio] = chk
            bio_flat.addWidget(chk, 1, col_idx)

        bio_outer.addLayout(bio_flat)

        # Select All / Deselect All / Reset to defaults buttons
        bio_btn_row = QHBoxLayout()
        btn_bio_all = QPushButton("✅ Select All")
        btn_bio_all.clicked.connect(lambda: [c.setChecked(True)  for c in self._chk_bio.values()])
        btn_bio_none = QPushButton("☐ Deselect All")
        btn_bio_none.clicked.connect(lambda: [c.setChecked(False) for c in self._chk_bio.values()])
        btn_bio_crit = QPushButton("★ Critical only")
        btn_bio_crit.clicked.connect(lambda: [
            c.setChecked(b in CRITICAL_BIO_SET) for b,c in self._chk_bio.items()])
        for b in (btn_bio_all, btn_bio_none, btn_bio_crit):
            b.setMaximumWidth(120); bio_btn_row.addWidget(b)
        bio_btn_row.addStretch()
        bio_outer.addLayout(bio_btn_row)
        lay.addWidget(bio_grp)

        # ══════════════════════════════════════════════════════════════════
        # 4️⃣  TOPOGRAPHIC VARIABLE SELECTION
        # ══════════════════════════════════════════════════════════════════
        topo_grp = QGroupBox("4️⃣  Topographic Variables  (select which to include)")
        topo_lay = QHBoxLayout(topo_grp)
        self._chk_topo = {}
        TOPO_LABELS = {
            "elevation_m": "📐 Elevation (m)",
            "slope_pct":   "📏 Slope (%)",
            "aspect_deg":  "🧭 Aspect (°)",
        }
        for col, lbl in TOPO_LABELS.items():
            chk = QCheckBox(lbl); chk.setChecked(True)
            chk.setToolTip(f"Column name: {col}")
            self._chk_topo[col] = chk; topo_lay.addWidget(chk)
        topo_lay.addStretch()
        # Deselect all button
        btn_topo_none = QPushButton("☐ None (skip topography)")
        btn_topo_none.setStyleSheet("color:#7f8c8d;font-size:10px;")
        btn_topo_none.clicked.connect(lambda: [c.setChecked(False) for c in self._chk_topo.values()])
        topo_lay.addWidget(btn_topo_none)
        lay.addWidget(topo_grp)

        # ══════════════════════════════════════════════════════════════════
        # 5️⃣  OPTIONAL EXTRA VARIABLES
        # ══════════════════════════════════════════════════════════════════
        opt_grp = QGroupBox(
            "5️⃣  Optional Extra Variables  "
            "(include if columns exist in your data — enrich the analysis)")
        opt_lay = QHBoxLayout(opt_grp)
        self._chk_opt = {}
        OPT_LABELS = {
            "srad": "☀  Solar Radiation (kJ m⁻² day⁻¹)",
            "wind": "💨  Wind Speed (m s⁻¹)",
            "vapr": "💧  Vapour Pressure (kPa)",
        }
        for col, lbl in OPT_LABELS.items():
            chk = QCheckBox(lbl); chk.setChecked(False)
            chk.setToolTip(f"Column name: {col}")
            self._chk_opt[col] = chk; opt_lay.addWidget(chk)
        opt_lay.addStretch()
        lay.addWidget(opt_grp)

        # ── Options row ───────────────────────────────────────────────────
        misc_grp = QGroupBox("⚙️  Options")
        misc_lay = QHBoxLayout(misc_grp)
        self.chk_null = QCheckBox("Skip null rows"); self.chk_null.setChecked(True)
        misc_lay.addWidget(self.chk_null)
        misc_lay.addWidget(QLabel("  Min. points:"))
        self.spin_min = QSpinBox(); self.spin_min.setRange(10,9999)
        self.spin_min.setValue(15); misc_lay.addWidget(self.spin_min)
        misc_lay.addStretch()
        lay.addWidget(misc_grp)

        # ── Data Preview ───────────────────────────────────────────────────
        prev_grp = QGroupBox("📋  Data Preview  (first 5 rows — native range)")
        prev_lay = QVBoxLayout(prev_grp)
        self.preview_table = QTableWidget(0,5); self.preview_table.setMaximumHeight(130)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView_ResizeFit)
        self.preview_table.verticalHeader().setVisible(False)
        prev_lay.addWidget(self.preview_table)
        btn_prev = QPushButton("🔍 Load Preview")
        btn_prev.clicked.connect(self._update_preview)
        prev_lay.addWidget(btn_prev)
        lay.addWidget(prev_grp)

        lay.addStretch()
        self._refresh_layers()
        _scroll.setWidget(w)
        return _scroll


    # ── Summary ───────────────────────────────────────────────────────────
    def _build_results_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        top = QHBoxLayout()

        # Gauge
        gf = QFrame(); gf.setFrameStyle(QFrame_Box | QFrame_Raised)
        gl = QVBoxLayout(gf)
        self.gauge = ScoreGauge(0); gl.addWidget(self.gauge)
        self.lbl_class = QLabel("—"); self.lbl_class.setAlignment(Qt_AlignCenter)
        self.lbl_class.setStyleSheet("font-size:13px;font-weight:bold;")
        gl.addWidget(self.lbl_class)
        self.lbl_zone = QLabel("—"); self.lbl_zone.setAlignment(Qt_AlignCenter)
        self.lbl_zone.setWordWrap(True); self.lbl_zone.setStyleSheet("font-size:11px;color:#636e72;")
        gl.addWidget(self.lbl_zone)
        gf.setMaximumWidth(230); top.addWidget(gf)

        # Component table
        self.comp_table = QTableWidget(6,3)
        self.comp_table.setHorizontalHeaderLabels(["Method","Score","Weight"])
        self.comp_table.verticalHeader().setVisible(False)
        self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView_Stretch)
        self.comp_table.setMaximumHeight(210); top.addWidget(self.comp_table)
        lay.addLayout(top)

        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Climate:"))
        self.lbl_clim = QLabel("—"); self.lbl_clim.setStyleSheet("font-weight:bold;color:#2980b9;")
        sub_row.addWidget(self.lbl_clim)
        sub_row.addWidget(QLabel("   Topography:"))
        self.lbl_topo_sc = QLabel("—"); self.lbl_topo_sc.setStyleSheet("font-weight:bold;color:#8e44ad;")
        sub_row.addWidget(self.lbl_topo_sc)
        sub_row.addStretch(); lay.addLayout(sub_row)

        # Bootstrap CI row
        ci_row = QHBoxLayout()
        ci_row.addWidget(QLabel("95% CI (Bootstrap):"))
        self.lbl_ci = QLabel("—")
        self.lbl_ci.setStyleSheet(
            "font-weight:bold;color:#1a5276;"
            "background:#eaf4fb;border:1px solid #aed6f1;"
            "border-radius:3px;padding:2px 8px;")
        ci_row.addWidget(self.lbl_ci)
        ci_row.addWidget(QLabel("   Spatial AC:"))
        self.lbl_spatial_ac = QLabel("—")
        self.lbl_spatial_ac.setStyleSheet(
            "font-weight:bold;color:#1a5276;"
            "border:1px solid #aed6f1;border-radius:3px;padding:2px 8px;")
        ci_row.addWidget(self.lbl_spatial_ac)
        ci_row.addStretch(); lay.addLayout(ci_row)

        self.txt_rec = QTextEdit(); self.txt_rec.setMaximumHeight(70)
        self.txt_rec.setReadOnly(True)
        self.txt_rec.setStyleSheet("background:#eafaf1;border:1px solid #27ae60;border-radius:4px;padding:6px;")
        lay.addWidget(QLabel("💡 Recommendation:")); lay.addWidget(self.txt_rec)

        # Map button row
        map_row = QHBoxLayout()
        btn_map = QPushButton("🗺  Show Points on QGIS Map")
        btn_map.setStyleSheet(
            "background:#8e44ad;color:white;font-weight:bold;"
            "padding:5px 16px;border-radius:4px;font-size:11px;")
        btn_map.setToolTip(
            "Adds '<Species> — Native Range' and '<Species> — Target Site' layers to the QGIS map.\n"
            "Layer names use the species name detected from the input CSV.\n"
            "Requires latitude/longitude columns in the input CSV files.")
        btn_map.clicked.connect(self._add_points_to_map)
        map_row.addWidget(btn_map)
        self.lbl_map_status = QLabel("")
        self.lbl_map_status.setStyleSheet("font-size:10px;color:#636e72;")
        map_row.addWidget(self.lbl_map_status)
        map_row.addStretch()
        lay.addLayout(map_row)

        # Risk variables — header bar
        risk_hdr = QHBoxLayout()
        self.lbl_risk = QLabel("🎯 Top Risk Variables")
        self.lbl_risk.setStyleSheet(
            "color:#c0392b;font-size:12px;font-weight:bold;padding:2px 0;")
        risk_hdr.addWidget(self.lbl_risk)
        btn_risk_detail = QPushButton("⚠️  Detailed Risk Analysis")
        btn_risk_detail.setStyleSheet(
            "color:white;background:#c0392b;padding:3px 10px;"
            "border-radius:3px;font-size:11px;")
        btn_risk_detail.clicked.connect(
            lambda: self.tabs.setCurrentIndex(
                next((i for i in range(self.tabs.count())
                      if "Risk" in self.tabs.tabText(i)), 0)))
        risk_hdr.addWidget(btn_risk_detail); risk_hdr.addStretch()
        lay.addLayout(risk_hdr)

        # Risk cards scroll area
        risk_scroll = QScrollArea(); risk_scroll.setWidgetResizable(True)
        risk_scroll.setFixedHeight(200)
        risk_scroll.setStyleSheet("background:transparent;border:none;")
        risk_inner = QWidget()
        self.risk_cards_layout = QHBoxLayout(risk_inner)
        self.risk_cards_layout.setSpacing(6)
        self.risk_cards_layout.setContentsMargins(2,2,2,2)
        self.risk_cards_layout.addStretch()
        risk_scroll.setWidget(risk_inner)
        lay.addWidget(risk_scroll)
        lay.addStretch(); return w

    # ── ML Models ──────────────────────────────────────────────────────────
    def _build_ml_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        sub = QTabWidget()

        # GMM
        gw = QWidget(); gl = QVBoxLayout(gw); self.gmm_text = QTextEdit(); self.gmm_text.setReadOnly(True); gl.addWidget(self.gmm_text); sub.addTab(gw,"Gaussian Mixture")

        # Isolation Forest
        iw = QWidget(); il = QVBoxLayout(iw); self.iso_text = QTextEdit(); self.iso_text.setReadOnly(True); il.addWidget(self.iso_text); sub.addTab(iw,"Isolation Forest")

        # OCSVM
        ow = QWidget(); ol = QVBoxLayout(ow); self.svm_text = QTextEdit(); self.svm_text.setReadOnly(True); ol.addWidget(self.svm_text); sub.addTab(ow,"One-Class SVM")

        # K-Means
        kw = QWidget(); kl = QVBoxLayout(kw); self.km_text = QTextEdit(); self.km_text.setReadOnly(True); kl.addWidget(self.km_text); sub.addTab(kw,"K-Means Clustering")

        # Threshold voting
        tw = QWidget(); tl = QVBoxLayout(tw); self.tz_text = QTextEdit(); self.tz_text.setReadOnly(True); tl.addWidget(self.tz_text); sub.addTab(tw,"Threshold Zone Voting")

        lay.addWidget(sub); return w

    # ── Bio Detail ─────────────────────────────────────────────────────────────
    def _build_bio_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.bio_table = QTableWidget(19, 10)
        self.bio_table.setHorizontalHeaderLabels(
            ["Bio","Name","Target","P5","P25","Median","P75","P95","Score","MW-p"]
        )
        self.bio_table.horizontalHeader().setSectionResizeMode(QHeaderView_ResizeFit)
        lay.addWidget(self.bio_table); return w

    # ── Topography ─────────────────────────────────────────────────────────────
    def _build_topo_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.topo_text = QTextEdit(); self.topo_text.setReadOnly(True)
        self.topo_text.setFont(QFont("Courier",10))
        lay.addWidget(self.topo_text); return w

    # ── Variable Importance ────────────────────────────────────────────────────────
    def _build_importance_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.imp_table = QTableWidget(22, 4)
        self.imp_table.setHorizontalHeaderLabels(
            ["Variable","RF Importance","PDI (Percentile Dev.)","Combined Importance"]
        )
        self.imp_table.horizontalHeader().setSectionResizeMode(QHeaderView_Stretch)
        lay.addWidget(self.imp_table); return w

    # ── Charts ─────────────────────────────────────────────────────────────
    def _build_chart_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        if not HAS_MPL:
            lay.addWidget(QLabel("matplotlib required.")); return w
        self.fig    = Figure(figsize=(14,10))
        self.canvas = FigureCanvas(self.fig)
        lay.addWidget(self.canvas)

        # ── Export button row ───────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_export_charts = QPushButton("💾 Export Charts (300 dpi PNG)")
        self.btn_export_charts.setToolTip(
            "Grafiği 300 dpi çözünürlükte PNG olarak kaydet (bilimsel yayın kalitesi)"
        )
        self.btn_export_charts.setEnabled(False)
        self.btn_export_charts.clicked.connect(self._export_charts_png)
        btn_row.addWidget(self.btn_export_charts)
        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setStyleSheet("color:#27ae60;font-size:10px;")
        btn_row.addWidget(self.lbl_export_status)
        btn_row.addStretch()
        lay.addLayout(btn_row)
        return w

    def _export_charts_png(self):
        """Save the current figure as a 300-dpi PNG (publication quality)."""
        default_name = self._default_export_name("charts", "png")
        start_dir    = self._export_start_dir()

        path, _ = QFileDialog.getSaveFileName(
            self, "Grafikleri Kaydet — 300 dpi PNG",
            os.path.join(start_dir, default_name),
            "PNG Images (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"

        try:
            self.fig.savefig(
                path,
                dpi=300,
                format="png",
                bbox_inches="tight",
                facecolor=self.fig.get_facecolor()
            )
            self.lbl_export_status.setText(f"✅ Kaydedildi: {os.path.basename(path)}")
            QMessageBox.information(
                self, "Grafik Kaydedildi",
                f"Grafik başarıyla kaydedildi (300 dpi):\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Hata", f"Kaydetme başarısız:\n{exc}")

    # ── SSP Scenarios Tab ────────────────────────────────────────────────────
    def _build_ssp_tab(self):
        """
        Future climate scenario analysis.
        The user loads a CSV where each row is a scenario — columns must match
        the active bio/topo/opt columns.  An optional "scenario" text column
        is used as the row label.
        """
        _scroll = QScrollArea(); _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QFrame_NoFrame)
        w = QWidget(); lay = QVBoxLayout(w)

        hdr = QLabel(
            "🌍  Future Climate Scenarios  (SSP / CMIP6 / CHELSA)\n"
            "Load a CSV where every row is a future climate scenario with the same\n"
            "bio/topo/opt columns used in the main analysis.  An optional 'scenario'\n"
            "column is used as the row label (e.g. 'SSP2-4.5 2050').")
        hdr.setWordWrap(True)
        hdr.setStyleSheet(
            "background:#eaf4fb;border:1px solid #aed6f1;"
            "padding:8px;border-radius:4px;font-size:11px;color:#154360;")
        lay.addWidget(hdr)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Scenario data:"))
        self.le_ssp_path = QLineEdit()
        self.le_ssp_path.setPlaceholderText("Select CSV file with future climate values…")
        self.le_ssp_path.setReadOnly(True)
        self.le_ssp_path.setSizePolicy(QSizePolicy_Expanding, QSizePolicy_Fixed)
        src_row.addWidget(self.le_ssp_path)
        btn_ssp_browse = QPushButton("📂 Browse")
        btn_ssp_browse.clicked.connect(lambda: self._browse_csv(self.le_ssp_path))
        src_row.addWidget(btn_ssp_browse)
        lay.addLayout(src_row)

        self.btn_run_ssp = QPushButton("▶  Run SSP Analysis")
        self.btn_run_ssp.setStyleSheet(
            "background:#2980b9;color:white;font-weight:bold;"
            "padding:6px 18px;border-radius:4px;")
        self.btn_run_ssp.clicked.connect(self._run_ssp_analysis)
        lay.addWidget(self.btn_run_ssp)

        # Results table: scenario | score | Δ | trend | zone | class
        self.ssp_table = QTableWidget(0, 6)
        self.ssp_table.setHorizontalHeaderLabels(
            ["Scenario", "Score", "Δ vs Current", "Trend", "Zone", "Class"])
        self.ssp_table.horizontalHeader().setSectionResizeMode(QHeaderView_Stretch)
        self.ssp_table.setMinimumHeight(200)
        lay.addWidget(self.ssp_table)

        self.ssp_text = QTextEdit(); self.ssp_text.setReadOnly(True)
        self.ssp_text.setFont(QFont("Courier", 9))
        self.ssp_text.setPlaceholderText(
            "Run analysis first, then load SSP data and click 'Run SSP Analysis'.")
        lay.addWidget(self.ssp_text)

        _scroll.setWidget(w)
        return _scroll

    def _run_ssp_analysis(self):
        """Read SSP CSV and call run_ssp_scenarios() using the stored analyzer data."""
        if not self.results:
            QMessageBox.warning(self, "No Analysis",
                "Run the main analysis first before loading SSP scenarios."); return
        path = self.le_ssp_path.text().strip()
        if not path:
            QMessageBox.warning(self, "No File", "Select a scenario CSV file first."); return

        bio_cols  = self.results["composite"].get("active_bio_cols",  [])
        topo_cols = self.results["composite"].get("active_topo_cols", [])
        opt_cols  = self.results["composite"].get("active_opt_cols",  [])
        all_cols  = bio_cols + topo_cols + opt_cols

        # Try to read a "scenario" label column too
        try:
            import csv as _csv
            rows_raw, header_raw = [], None
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                for line in _csv.reader(f):
                    if not line or line[0].startswith("#"): continue
                    if header_raw is None: header_raw = [c.strip().lower() for c in line]
                    else: rows_raw.append(line)
        except Exception as e:
            QMessageBox.critical(self, "Read Error", str(e)); return

        # Detect "scenario" name column
        sc_col_idx = None
        for i, h in enumerate(header_raw):
            if h in ("scenario","name","label","ssp","ssps"):
                sc_col_idx = i; break

        def norm(s): return s.lower().strip().replace(" ","_").replace("-","_")
        header_norm = [norm(h) for h in header_raw]

        # Build scenario dict
        scenarios = {}
        for row_i, row in enumerate(rows_raw):
            label = (row[sc_col_idx].strip() if sc_col_idx is not None
                     else f"Scenario {row_i+1}")
            vals = []
            missing = []
            for col in all_cols:
                # flexible column matching
                target_norms = [norm(col), norm(col.upper()),
                                norm(col.replace("bio","BIO_")),
                                norm(f"_{col}")]
                idx = next((header_norm.index(t) for t in target_norms
                            if t in header_norm), None)
                if idx is None: missing.append(col); vals.append(0.0)
                else:
                    try: vals.append(float(row[idx]))
                    except (ValueError, IndexError): vals.append(0.0)
            if missing:
                QMessageBox.warning(
                    self, "Missing Columns",
                    f"Row '{label}' missing: {', '.join(missing)}\n"
                    f"These columns will be set to 0.");
            scenarios[label] = vals

        if not scenarios:
            QMessageBox.warning(self, "No Scenarios", "No valid scenario rows found."); return

        # Re-construct analyzer with original data
        try:
            sp_data  = getattr(self, "_sp_data_cache",  None)
            tgt_vals = getattr(self, "_tgt_vals_cache", None)
            if sp_data is None or tgt_vals is None:
                QMessageBox.warning(self, "No Analysis Data",
                    "Re-run the main analysis to refresh data."); return

            az = ClimateResilienceAnalyzer(
                sp_data, tgt_vals,
                bio_cols=bio_cols, topo_cols=topo_cols, opt_cols=opt_cols)
            az.results = dict(self.results)  # seed with current results
            ssp_r = az.run_ssp_scenarios(scenarios)
            self.results["ssp_scenarios"] = ssp_r
            self._populate_ssp_tab()
        except Exception as e:
            QMessageBox.critical(self, "SSP Error", str(e))

    def _populate_ssp_tab(self):
        """Fill SSP table + text summary from results."""
        ssp = self.results.get("ssp_scenarios")
        if not ssp:
            self.ssp_text.setPlainText(
                "No SSP results yet.  Load a scenario CSV and click 'Run SSP Analysis'.")
            return

        current_sc = ssp.get("current_score", 0)
        scens = ssp.get("scenarios", {})

        self.ssp_table.setRowCount(len(scens) + 1)  # +1 for current baseline row
        def _item(txt, bg=None):
            it = QTableWidgetItem(txt)
            it.setTextAlignment(Qt_AlignCenter)
            if bg: it.setBackground(QColor(bg))
            return it

        # Row 0 — current baseline
        self.ssp_table.setItem(0, 0, _item("▶ Current (Observed)", "#d6eaf8"))
        self.ssp_table.setItem(0, 1, _item(f"{current_sc:.1f}", "#d6eaf8"))
        self.ssp_table.setItem(0, 2, _item("—", "#d6eaf8"))
        self.ssp_table.setItem(0, 3, _item("Baseline", "#d6eaf8"))
        self.ssp_table.setItem(0, 4, _item(
            self.results.get("threshold_zone", {}).get("zone_label","—")[:25]))
        self.ssp_table.setItem(0, 5, _item(
            self.results.get("composite",{}).get("resilience_class","—")))

        color_delta = {"▲ Improving": "#d5f5e3", "▼ Declining": "#fdecea", "◆ Stable": "#fef9e7"}
        lines = [f"FUTURE CLIMATE SCENARIO COMPARISON\n{'═'*55}",
                 f"Current (observed) composite score: {current_sc:.1f}\n"]

        for r_idx, (name, sc_r) in enumerate(scens.items(), start=1):
            if "error" in sc_r:
                self.ssp_table.setItem(r_idx, 0, _item(name))
                self.ssp_table.setItem(r_idx, 1, _item("ERROR"))
                for c in range(2, 6): self.ssp_table.setItem(r_idx, c, _item("—"))
                lines.append(f"{name}: ERROR — {sc_r['error']}")
                continue
            sc   = sc_r["composite_score"]
            delt = sc_r["delta_from_current"]
            trend= sc_r["trend"]
            bg   = color_delta.get(trend, "#ffffff")
            self.ssp_table.setItem(r_idx, 0, _item(name))
            self.ssp_table.setItem(r_idx, 1, _item(f"{sc:.1f}",
                "#d5f5e3" if sc>=65 else "#fef9e7" if sc>=50 else "#fdecea"))
            self.ssp_table.setItem(r_idx, 2, _item(f"{delt:+.1f}", bg))
            self.ssp_table.setItem(r_idx, 3, _item(trend, bg))
            self.ssp_table.setItem(r_idx, 4, _item(sc_r["zone_label"][:25]))
            self.ssp_table.setItem(r_idx, 5, _item(sc_r["resilience_class"]))
            lines.append(
                f"{name:<30}: {sc:.1f}  ({delt:+.1f})  {trend}  "
                f"→  {sc_r['resilience_class']}")

        self.ssp_text.setPlainText("\n".join(lines))

    # ── Weight Calibration Tab ───────────────────────────────────────────────
    def _build_calibration_tab(self):
        """
        Validation-dataset based weight calibration.
        Load a CSV with the same bio/topo/opt columns + a 'label' column (1=suitable / 0=unsuitable).
        """
        _scroll = QScrollArea(); _scroll.setWidgetResizable(True)
        _scroll.setFrameShape(QFrame_NoFrame)
        w = QWidget(); lay = QVBoxLayout(w)

        hdr = QLabel(
            "⚙️  Weight Calibration  (Validation Dataset)\n"
            "Load a labelled CSV (same bio/topo/opt columns + a 'label' column:\n"
            "1 = suitable,  0 = unsuitable).  The optimiser minimises binary\n"
            "cross-entropy to find the best component weights for your region.")
        hdr.setWordWrap(True)
        hdr.setStyleSheet(
            "background:#fef9e7;border:1px solid #f39c12;"
            "padding:8px;border-radius:4px;font-size:11px;color:#7d6608;")
        lay.addWidget(hdr)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Validation data:"))
        self.le_cal_path = QLineEdit()
        self.le_cal_path.setPlaceholderText("Select labelled CSV…")
        self.le_cal_path.setReadOnly(True)
        self.le_cal_path.setSizePolicy(QSizePolicy_Expanding, QSizePolicy_Fixed)
        src_row.addWidget(self.le_cal_path)
        btn_cal_browse = QPushButton("📂 Browse")
        btn_cal_browse.clicked.connect(lambda: self._browse_csv(self.le_cal_path))
        src_row.addWidget(btn_cal_browse)
        lay.addLayout(src_row)

        self.btn_run_cal = QPushButton("▶  Calibrate Weights")
        self.btn_run_cal.setStyleSheet(
            "background:#f39c12;color:white;font-weight:bold;"
            "padding:6px 18px;border-radius:4px;")
        self.btn_run_cal.clicked.connect(self._run_calibration)
        lay.addWidget(self.btn_run_cal)

        # Weight comparison table: method | default | optimized
        self.cal_table = QTableWidget(5, 3)
        self.cal_table.setHorizontalHeaderLabels(
            ["Component", "Default Weight", "Optimized Weight"])
        self.cal_table.horizontalHeader().setSectionResizeMode(QHeaderView_Stretch)
        self.cal_table.setMaximumHeight(200)
        lay.addWidget(self.cal_table)

        self.cal_text = QTextEdit(); self.cal_text.setReadOnly(True)
        self.cal_text.setFont(QFont("Courier", 9))
        self.cal_text.setPlaceholderText(
            "Run analysis first, then load a labelled CSV and click 'Calibrate Weights'.")
        lay.addWidget(self.cal_text)

        _scroll.setWidget(w)
        return _scroll

    def _run_calibration(self):
        """Read labelled CSV and run calibrate_weights()."""
        if not self.results:
            QMessageBox.warning(self, "No Analysis",
                "Run the main analysis first."); return
        path = self.le_cal_path.text().strip()
        if not path:
            QMessageBox.warning(self, "No File", "Select a validation CSV file."); return

        bio_cols  = self.results["composite"].get("active_bio_cols",  [])
        topo_cols = self.results["composite"].get("active_topo_cols", [])
        opt_cols  = self.results["composite"].get("active_opt_cols",  [])
        all_cols  = bio_cols + topo_cols + opt_cols

        try:
            import csv as _csv, numpy as np
            rows_raw, header_raw = [], None
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                for line in _csv.reader(f):
                    if not line or line[0].startswith("#"): continue
                    if header_raw is None: header_raw = [c.strip().lower() for c in line]
                    else: rows_raw.append(line)
        except Exception as e:
            QMessageBox.critical(self, "Read Error", str(e)); return

        def norm(s): return s.lower().strip().replace(" ","_").replace("-","_")
        header_norm = [norm(h) for h in header_raw]

        # Find label column
        lbl_idx = next((header_norm.index(h) for h in header_norm
                        if h in ("label","class","suitable","presence")), None)
        if lbl_idx is None:
            QMessageBox.warning(self, "No Label Column",
                "CSV must have a 'label' column (1=suitable, 0=unsuitable)."); return

        val_data, val_labels = [], []
        for row in rows_raw:
            try:
                lbl = float(row[lbl_idx])
            except (ValueError, IndexError):
                continue
            vals = []
            for col in all_cols:
                target_norms = [norm(col), norm(col.upper())]
                idx = next((header_norm.index(t) for t in target_norms
                            if t in header_norm), None)
                try: vals.append(float(row[idx]) if idx is not None else 0.0)
                except (ValueError, TypeError): vals.append(0.0)
            val_data.append(vals); val_labels.append(int(lbl))

        if len(val_data) < 4:
            QMessageBox.warning(self, "Insufficient Data",
                f"Need at least 4 valid rows — found {len(val_data)}."); return

        try:
            sp_data  = getattr(self, "_sp_data_cache",  None)
            tgt_vals = getattr(self, "_tgt_vals_cache", None)
            if sp_data is None:
                QMessageBox.warning(self, "No Analysis Data",
                    "Re-run the main analysis first."); return

            az = ClimateResilienceAnalyzer(
                sp_data, tgt_vals,
                bio_cols=bio_cols, topo_cols=topo_cols, opt_cols=opt_cols)
            az.results = dict(self.results)
            cal_r = az.calibrate_weights(
                np.array(val_data), np.array(val_labels))
            self.results["calibration"] = cal_r
            self._populate_calibration_tab()
        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", str(e))

    def _populate_calibration_tab(self):
        """Fill calibration table + text from results."""
        cal = self.results.get("calibration")
        if not cal:
            self.cal_text.setPlainText("No calibration results yet.")
            return
        if "error" in cal:
            self.cal_text.setPlainText(f"Calibration error: {cal['error']}")
            return

        w_def = cal.get("default_weights",    {})
        w_opt = cal.get("optimized_weights",  {})
        keys  = ["threshold_zone","gmm","isolation_forest","ocsvm","mahalanobis"]
        labels= ["Threshold Zone","GMM","Isolation Forest","One-Class SVM","Mahalanobis"]

        self.cal_table.setRowCount(len(keys))
        for i, (k, lbl) in enumerate(zip(keys, labels)):
            d = w_def.get(k, 0); o = w_opt.get(k, 0)
            self.cal_table.setItem(i, 0, QTableWidgetItem(lbl))
            di = QTableWidgetItem(f"{d:.4f}"); di.setTextAlignment(Qt_AlignCenter)
            oi = QTableWidgetItem(f"{o:.4f}"); oi.setTextAlignment(Qt_AlignCenter)
            if o > d: oi.setBackground(QColor("#d5f5e3"))
            elif o < d: oi.setBackground(QColor("#fdecea"))
            self.cal_table.setItem(i, 1, di)
            self.cal_table.setItem(i, 2, oi)

        conv_str = "✅ Converged" if cal.get("converged") else "⚠️ Did not converge"
        self.cal_text.setPlainText(
            f"WEIGHT CALIBRATION RESULTS\n{'═'*50}\n"
            f"  Validation points : {cal['n_valid']}  "
            f"({cal['n_suitable']} suitable, {cal['n_unsuitable']} unsuitable)\n"
            f"  Optimiser         : {conv_str}\n\n"
            f"  Default accuracy   : {cal['default_accuracy']:.1%}\n"
            f"  Optimized accuracy : {cal['optimized_accuracy']:.1%}\n"
            f"  Accuracy gain      : {cal['accuracy_gain']:+.1%}\n\n"
            f"  ℹ️  To apply optimized weights, update the W_* constants in\n"
            f"  analysis_engine.py accordingly and re-run the analysis.\n\n"
            f"Optimized weights:\n" +
            "\n".join(f"  {labels[i]:<20}: {w_opt.get(k,0):.4f}" for i,k in enumerate(keys))
        )

    # ── Report ─────────────────────────────────────────────────────────────────
    def _build_report_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.report_text = QTextEdit(); self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier",10))
        lay.addWidget(self.report_text)
        btn = QPushButton("📋 Copy to Clipboard")
        btn.clicked.connect(lambda: self._copy_report())
        lay.addWidget(btn); return w

    # ── Optional Variables Tab ────────────────────────────────────────
    def _build_opt_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        info = QLabel(
            "ℹ️  This tab shows analysis results only for optional variables "
            "(srad / wind / vapr) selected in Data Input.")
        info.setWordWrap(True)
        info.setStyleSheet("background:#eaf4fb;border:1px solid #aed6f1;"
                           "padding:5px;border-radius:3px;font-size:11px;")
        lay.addWidget(info)
        self.opt_text = QTextEdit(); self.opt_text.setReadOnly(True)
        self.opt_text.setFont(QFont("Courier", 9))
        lay.addWidget(self.opt_text)
        return w

    # ── Risk Details Tab ────────────────────────────────────────────────
    def _build_risk_detail_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        hdr = QLabel(
            "⚠️  Top Risk Variables — Detailed Numerical Explanations")
        hdr.setStyleSheet(
            "font-size:13px;font-weight:bold;color:#2c3e50;"
            "padding:6px;background:#fef9e7;border-radius:4px;")
        lay.addWidget(hdr)
        self.risk_subtabs = QTabWidget()
        self.risk_subtabs.setTabPosition(QTabWidget_West)
        self._risk_subtab_widgets = []
        for i in range(7):
            te = QTextEdit(); te.setReadOnly(True)
            te.setFont(QFont("Courier", 9))
            te.setStyleSheet("background:#fffef5;")
            self._risk_subtab_widgets.append(te)
            self.risk_subtabs.addTab(te, f"—")
        lay.addWidget(self.risk_subtabs)
        return w

    # ── Variable selection helpers ──────────────────────────────────
    def _get_active_bio_cols(self) -> list:
        """Returns selected bio column names in canonical order."""
        return [c for c in ALL_BIO_COLS if self._chk_bio.get(c, QCheckBox()).isChecked()]

    def _get_active_topo_cols(self) -> list:
        """Returns selected topo column names in canonical order."""
        return [c for c in ALL_TOPO_COLS if self._chk_topo.get(c, QCheckBox()).isChecked()]

    def _get_active_opt_cols(self) -> list:
        """Returns selected optional column names in canonical order."""
        return [c for c in ALL_OPT_COLS if self._chk_opt.get(c, QCheckBox()).isChecked()]

    # ── Layer & File Management ──────────────────────────────────────────────
    def _refresh_layers(self):
        """Loads vector layers from the QGIS project into ComboBoxes."""
        self.cb_sp.clear(); self.cb_tgt.clear()
        for lyr in QgsProject.instance().mapLayers().values():
            if lyr.type() == QgsMapLayer_VectorLayer:
                self.cb_sp.addItem(lyr.name(),  lyr.id())
                self.cb_tgt.addItem(lyr.name(), lyr.id())

    def _get_layer(self, cb):
        lid = cb.currentData()
        return QgsProject.instance().mapLayer(lid) if lid else None

    def _browse_csv(self, line_edit: "QLineEdit"):
        """Opens a file dialog and writes the selected path to the QLineEdit."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "",
            "CSV Files (*.csv *.txt *.tsv);;All Files (*)"
        )
        if path:
            line_edit.setText(path)

    def _load_data(self, mode_layer: bool, cb, le_path,
                   expected_cols: list, skip_null: bool):
        """
        Loads data according to selected mode.
        Returns (ndarray, coords_or_None, species_name_or_None).
        mode_layer=True  → QGIS layer (_read_csv_layer)
        mode_layer=False → Disk CSV  (_read_csv_file)
        """
        if mode_layer:
            lyr = self._get_layer(cb)
            if lyr is None:
                raise ValueError("No QGIS layer selected or found.")
            return _read_csv_layer(lyr, expected_cols, skip_null)
        else:
            path = le_path.text().strip()
            if not path:
                raise ValueError("No CSV file selected.")
            return _read_csv_file(path, expected_cols, skip_null)

    def _update_preview(self):
        """Loads the first 5 rows from the native range source into the preview table."""
        bio_cols  = self._get_active_bio_cols()
        topo_cols = self._get_active_topo_cols()
        opt_cols  = self._get_active_opt_cols()
        cols = bio_cols + topo_cols + opt_cols
        if not cols:
            QMessageBox.warning(self, "Preview Error", "No variables selected."); return
        try:
            data, _, _ = self._load_data(
                self._bg_sp.checkedId() == 0,
                self.cb_sp, self.le_sp_path, cols, self.chk_null.isChecked()
            )
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", str(e)); return

        n = min(5, len(data)); nc = min(len(cols), data.shape[1])
        self.preview_table.setRowCount(n)
        self.preview_table.setColumnCount(nc)
        self.preview_table.setHorizontalHeaderLabels(cols[:nc])
        for r in range(n):
            for c in range(nc):
                self.preview_table.setItem(r, c, QTableWidgetItem(f"{data[r,c]:.3f}"))


    # ── Start Analysis ────────────────────────────────────────────────────────
    def _run_analysis(self):
        bio_cols  = self._get_active_bio_cols()
        topo_cols = self._get_active_topo_cols()
        opt_cols  = self._get_active_opt_cols()
        cols      = bio_cols + topo_cols + opt_cols
        skip_null = self.chk_null.isChecked()

        if not bio_cols:
            QMessageBox.warning(self, "No Variables",
                "Please select at least one bioclimatic variable."); return

        sp_mode_layer  = (self._bg_sp.checkedId()  == 0)
        tgt_mode_layer = (self._bg_tgt.checkedId() == 0)

        sp_src  = ("Layer: " + self.cb_sp.currentText()
                   if sp_mode_layer else
                   "File: " + os.path.basename(self.le_sp_path.text()))
        tgt_src = ("Layer: " + self.cb_tgt.currentText()
                   if tgt_mode_layer else
                   "File: " + os.path.basename(self.le_tgt_path.text()))
        opt_info = (f"  |  Optional: {', '.join(opt_cols)}" if opt_cols else "")
        topo_info = (f"  Topo: {', '.join(topo_cols)}" if topo_cols else "  No topo")
        self.progress_label.setText(
            f"Loading data…  {sp_src}  |  {tgt_src}  |  "
            f"{len(bio_cols)} bios  {topo_info}{opt_info}")

        try:
            sp_data,  sp_coords,  sp_species  = self._load_data(
                sp_mode_layer,  self.cb_sp,
                self.le_sp_path,  cols, skip_null)
            tgt_data, tgt_coords, tgt_species = self._load_data(
                tgt_mode_layer, self.cb_tgt,
                self.le_tgt_path, cols, skip_null)
        except Exception as e:
            QMessageBox.critical(self, "Data Error", str(e)); return

        if len(sp_data) < self.spin_min.value():
            QMessageBox.warning(self, "Insufficient Points",
                f"At least {self.spin_min.value()} points required for native range."
                f"\nPresent: {len(sp_data)} rows")
            return

        tgt_vals = tgt_data[0] if len(tgt_data) == 1 else tgt_data.mean(axis=0)

        # Store coords for map visualisation (used after analysis completes)
        self._sp_coords  = sp_coords
        self._tgt_coords = tgt_coords
        self._sp_data_cache  = sp_data
        self._tgt_vals_cache = tgt_vals

        # Species name — prefer species CSV over target, fall back cleanly.
        self.species_name        = sp_species or tgt_species or None
        self.species_name_clean  = _sanitize_species_name(self.species_name) \
                                    if self.species_name else ""

        self.btn_run.setEnabled(False)
        [b.setEnabled(False) for b in (self.btn_json, self.btn_csv)]
        self.progress.setVisible(True); self.progress.setValue(0)

        # Reveal and clear the log panel for this run
        self.log_view.clear()
        self.log_view.setVisible(True)
        if self.species_name:
            self._log_append(f"[data] species = {self.species_name}")
        self._log_append(
            f"[data] sp={len(sp_data)} pts  "
            f"bio={len(bio_cols)}  topo={len(topo_cols)}  opt={len(opt_cols)}"
        )

        self.worker = AnalysisWorker(sp_data, tgt_vals,
                                     bio_cols=bio_cols,
                                     topo_cols=topo_cols,
                                     opt_cols=opt_cols)
        self.worker.progress.connect(lambda p, m: (
            self.progress.setValue(p), self.progress_label.setText(m)))
        self.worker.log.connect(self._log_append)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _log_append(self, line: str):
        """Append a line to the live log panel and auto-scroll to bottom."""
        self.log_view.append(line)
        try:
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
        except Exception:
            pass

    def _on_done(self, results):
        self.results = results
        self.progress.setVisible(False)
        msg = "✅ Analysis complete."
        if getattr(self, "species_name", None):
            msg = f"✅ Analysis complete.  Species: {self.species_name}"
        self.progress_label.setText(msg)
        self.btn_run.setEnabled(True)
        [b.setEnabled(True) for b in (self.btn_json, self.btn_csv, self.btn_full_report)]
        self._populate()
        self.tabs.setCurrentIndex(1)
        # Add points to QGIS map canvas (if coordinates available)
        self._add_points_to_map()

    def _on_err(self, msg):
        self.progress.setVisible(False); self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", msg)

    # ── QGIS Map Visualisation ────────────────────────────────────────────────
    def _add_points_to_map(self):
        """
        Creates / refreshes two QGIS memory layers:
          • <Species> — Native Range   (blue dots, one per species occurrence)
          • <Species> — Target Site    (red star)

        Layer names are prefixed with the species name detected from the
        input CSV (e.g. "Cedrus libani — Native Range") so every output for
        that species groups together in the QGIS Layers panel.  Falls back
        to the legacy "CRESTA" prefix if no species was detected.

        Works with WGS-84 (EPSG:4326) coordinates.
        """
        sp_coords  = getattr(self, "_sp_coords",  None)
        tgt_coords = getattr(self, "_tgt_coords", None)

        has_sp_coords  = (sp_coords  is not None and
                          any(v is not None for v in sp_coords[0]))
        has_tgt_coords = (tgt_coords is not None and
                          any(v is not None for v in tgt_coords[0]))

        if not has_sp_coords and not has_tgt_coords:
            # No coordinate columns found — inform the user once
            if self.results:
                self.progress_label.setText(
                    "✅ Analysis complete.  "
                    "ℹ️ Add 'latitude'/'longitude' columns to CSV to see points on map.")
            return

        crs = QgsCoordinateReferenceSystem("EPSG:4326")

        # Species prefix for all layer names — shown in the QGIS Layers panel
        sp_prefix = getattr(self, "species_name", None) or "CRESTA"
        native_layer_name = f"{sp_prefix} — Native Range"
        target_layer_name = f"{sp_prefix} — Target Site"

        # ── Helper: remove old layer of same name ─────────────────────────
        def _remove_old(name):
            for lyr in QgsProject.instance().mapLayersByName(name):
                QgsProject.instance().removeMapLayer(lyr.id())

        composite_sc = (self.results.get("composite", {})
                        .get("composite_score", 0) if self.results else 0)
        class_lbl = (self.results.get("composite", {})
                     .get("resilience_class", "") if self.results else "")
        zone_lbl  = (self.results.get("threshold_zone", {})
                     .get("zone_label", "") if self.results else "")

        # ── Native Range layer ────────────────────────────────────────────
        if has_sp_coords:
            # Also remove the legacy name so old runs don't leave stale layers
            _remove_old("CRESTA — Native Range")
            _remove_old(native_layer_name)
            sp_lyr = QgsVectorLayer("Point?crs=EPSG:4326", native_layer_name, "memory")
            sp_lyr.setCrs(crs)
            pr = sp_lyr.dataProvider()
            pr.addAttributes([
                QgsField("id",      QVariant.Int),
                QgsField("species", QVariant.String),
                QgsField("lat",     QVariant.Double),
                QgsField("lon",     QVariant.Double),
            ])
            sp_lyr.updateFields()
            lats, lons = sp_coords
            feats = []
            sp_label = getattr(self, "species_name", None) or ""
            for i, (lat, lon) in enumerate(zip(lats, lons)):
                if lat is None or lon is None:
                    continue
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(lon), float(lat))))
                f.setAttributes([i + 1, sp_label, float(lat), float(lon)])
                feats.append(f)
            pr.addFeatures(feats)
            sp_lyr.updateExtents()

            # Style: blue filled circle, 3 pt, 60 % opacity
            sym = QgsMarkerSymbol.createSimple({
                "name":           "circle",
                "color":          "31,120,180,180",
                "outline_color":  "255,255,255,200",
                "outline_width":  "0.3",
                "size":           "3",
            })
            sp_lyr.setRenderer(QgsSingleSymbolRenderer(sym))
            sp_lyr.setOpacity(0.75)
            QgsProject.instance().addMapLayer(sp_lyr)

        # ── Target Site layer ─────────────────────────────────────────────
        if has_tgt_coords:
            _remove_old("CRESTA — Target Site")
            _remove_old(target_layer_name)
            tgt_lyr = QgsVectorLayer("Point?crs=EPSG:4326", target_layer_name, "memory")
            tgt_lyr.setCrs(crs)
            pr2 = tgt_lyr.dataProvider()
            pr2.addAttributes([
                QgsField("id",          QVariant.Int),
                QgsField("species",     QVariant.String),
                QgsField("lat",         QVariant.Double),
                QgsField("lon",         QVariant.Double),
                QgsField("score",       QVariant.Double),
                QgsField("class",       QVariant.String),
                QgsField("zone",        QVariant.String),
            ])
            tgt_lyr.updateFields()
            lats_t, lons_t = tgt_coords
            tgt_feats = []
            sp_label = getattr(self, "species_name", None) or ""
            for i, (lat, lon) in enumerate(zip(lats_t, lons_t)):
                if lat is None or lon is None:
                    continue
                f = QgsFeature()
                f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(float(lon), float(lat))))
                f.setAttributes([i + 1, sp_label, float(lat), float(lon),
                                  composite_sc, class_lbl, zone_lbl])
                tgt_feats.append(f)

            # If no tgt coords but only one row → try to reuse sp_coords mean
            if not tgt_feats and has_sp_coords:
                lats_s, lons_s = sp_coords
                valid_sp = [(la, lo) for la, lo in zip(lats_s, lons_s)
                            if la is not None and lo is not None]
                if valid_sp:
                    avg_lat = sum(p[0] for p in valid_sp) / len(valid_sp)
                    avg_lon = sum(p[1] for p in valid_sp) / len(valid_sp)
                    f = QgsFeature()
                    f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(avg_lon, avg_lat)))
                    f.setAttributes([1, sp_label, avg_lat, avg_lon,
                                     composite_sc, class_lbl, zone_lbl])
                    tgt_feats.append(f)

            if tgt_feats:
                pr2.addFeatures(tgt_feats)
            tgt_lyr.updateExtents()

            # Style: red star, 7 pt
            score_color = (
                "39,174,96,255"   if composite_sc >= 80 else
                "46,204,113,255"  if composite_sc >= 65 else
                "243,156,18,255"  if composite_sc >= 50 else
                "230,126,34,255"  if composite_sc >= 35 else
                "231,76,60,255"
            )
            sym2 = QgsMarkerSymbol.createSimple({
                "name":          "star",
                "color":         score_color,
                "outline_color": "255,255,255,255",
                "outline_width": "0.5",
                "size":          "9",
            })
            tgt_lyr.setRenderer(QgsSingleSymbolRenderer(sym2))
            QgsProject.instance().addMapLayer(tgt_lyr)

        # Zoom to native range extent
        try:
            if has_sp_coords and sp_lyr.featureCount() > 0:
                self.iface.mapCanvas().setExtent(sp_lyr.extent())
                self.iface.mapCanvas().zoomByFactor(1.15)
                self.iface.mapCanvas().refresh()
        except Exception:
            pass

        # Update label
        n_sp = len([v for v in (sp_coords[0] if sp_coords else []) if v is not None])
        self.progress_label.setText(
            f"✅ Analysis complete.  🗺 {n_sp} native range points + target site added to map.")

    # ── Populate Results ──────────────────────────────────────────────────────
    def _populate(self):
        r    = self.results
        comp = r["composite"]
        sc   = comp["composite_score"]

        # Gauge + class
        self.gauge.set_score(sc)
        cls = comp["resilience_class"]
        self.lbl_class.setText(cls)
        cm  = {"A":"#27ae60","B":"#2ecc71","C":"#f39c12","D":"#e67e22","E":"#e74c3c"}
        self.lbl_class.setStyleSheet(
            f"font-size:13px;font-weight:bold;color:{cm.get(cls[0],'#7f8c8d')}")
        self.lbl_zone.setText(comp["zone_label"])
        self.lbl_clim.setText(
            f"{comp['climate_score']:.2f}  (%{int(comp['climate_weight']*100)})")
        self.lbl_topo_sc.setText(
            f"{comp['topo_score']:.2f}  (%{int(comp['topo_weight']*100)})")
        self.txt_rec.setPlainText(comp["recommendation"])
        top5 = comp.get("top_risk_variables",[])
        self.lbl_risk.setText(
            f"🎯 Top Risk Variables  "
            f"({len(top5)} variables — full analysis in ⚠️ tab)")
        self._populate_risk_cards(top5)

        # Component table
        rows = [
            ("Multi-Threshold Zone (voting)", comp["component_scores"]["threshold_zone"],   "35%"),
            ("GMM (Gaussian Mixture)",     comp["component_scores"]["gmm"],              "20%"),
            ("Isolation Forest",           comp["component_scores"]["isolation_forest"], "15%"),
            ("One-Class SVM",              comp["component_scores"]["ocsvm"],            "15%"),
            ("Mahalanobis",               comp["component_scores"]["mahalanobis"],       "15%"),
            ("Topography",                 comp["component_scores"]["topo"],
             f"{int(comp['topo_weight']*100)}%"),
        ]
        for i,(lbl,s,wt) in enumerate(rows):
            self.comp_table.setItem(i,0,QTableWidgetItem(lbl))
            si = QTableWidgetItem(f"{s:.1f}"); si.setTextAlignment(Qt_AlignCenter)
            si.setBackground(QColor("#d5f5e3") if s>=65 else
                             QColor("#fef9e7") if s>=50 else QColor("#fdecea"))
            self.comp_table.setItem(i,1,si)
            self.comp_table.setItem(i,2,QTableWidgetItem(wt))

        # Bootstrap CI
        bci = r.get("bootstrap_ci", {})
        if bci.get("n_valid", 0) > 0:
            self.lbl_ci.setText(
                f"{bci['ci_lower']:.1f} – {bci['ci_upper']:.1f}  "
                f"(±{bci['std']:.1f},  n={bci['n_bootstrap']})")
        else:
            self.lbl_ci.setText("—")

        # Spatial autocorrelation (blocked CV)
        bcv = r.get("blocked_cv", {})
        mir = bcv.get("mean_inlier_rate")
        if mir is not None:
            ac_style = ("#eafaf1;border:1px solid #27ae60" if mir >= 0.80 else
                        "#fef9e7;border:1px solid #f39c12" if mir >= 0.60 else
                        "#fdecea;border:1px solid #e74c3c")
            self.lbl_spatial_ac.setText(
                f"{bcv['interpretation']}  "
                f"(inlier rate {mir:.0%},  {bcv['n_blocks']} blocks)")
            self.lbl_spatial_ac.setStyleSheet(
                f"font-weight:bold;color:#2c3e50;"
                f"background:{ac_style};border-radius:3px;padding:2px 8px;")
        else:
            self.lbl_spatial_ac.setText("—")

        self._populate_ml_tabs()
        self._populate_bio_tab()
        self._populate_topo_tab()
        self._populate_opt_tab()
        self._populate_importance_tab()
        self._populate_risk_detail_tab()
        self._populate_ssp_tab()
        if HAS_MPL: self._draw_charts()
        self._build_report()

    def _populate_ml_tabs(self):
        r = self.results
        g = r["gmm"]; iso = r["isolation_forest"]; svm = r["ocsvm"]
        km = r["kmeans"]; tz = r["threshold_zone"]; m = r["mahalanobis_stat"]

        self.gmm_text.setPlainText(
            f"Gaussian Mixture Model\n{'─'*45}\n"
            f"  Optimal components : {g['best_n_components']}\n"
            f"  Target log-prob    : {g['log_probability']}\n"
            f"  Max component prob : {g['max_component_prob']}\n"
            f"  Niche breadth (Levins): {g['niche_breadth']}\n"
            f"  Zone               : {g['zone']}\n"
            f"  Score              : {g['score']:.2f}\n\n"
            f"BIC Values:\n" +
            "\n".join(f"  {k} components: {v:.1f}" for k,v in g['bic_values'].items()) +
            f"\n\nThresholds: Q75≥CORE · Q25-Q75=SUITABLE · Q10-Q25=MARGINAL · <Q10=OUTSIDE"
        )

        self.iso_text.setPlainText(
            f"Isolation Forest\n{'─'*45}\n"
            f"  Anomaly score     : {iso['anomaly_score']} "
            f"(higher = more normal)\n"
            f"  Is normal?        : {'✅ Yes' if iso['is_normal'] else '❌ No'}\n"
            f"  Percentile rank   : {iso['percentile_rank']:.1f}%\n"
            f"  Species pts Q10   : {iso['sp_q10']}\n"
            f"  Species pts Q25   : {iso['sp_q25']}\n"
            f"  Species pts Q75   : {iso['sp_q75']}\n"
            f"  Zone              : {iso['zone']}\n"
            f"  Score             : {iso['score']:.2f}\n\n"
            f"  contamination=0.10 → outermost 10% of\n"
            f"  training points are treated as anomalies."
        )

        self.svm_text.setPlainText(
            f"One-Class SVM  (RBF kernel)\n{'─'*45}\n"
            f"  Decision value     : {svm['decision_value']} "
            f"(>0 = inside boundary)\n"
            f"  Is inside?         : {'✅ Yes' if svm['is_inside'] else '❌ No'}\n"
            f"  Percentile rank    : {svm['percentile_rank']:.1f}%\n"
            f"  nu used            : {svm['nu']}\n"
            f"  Zone               : {svm['zone']}\n"
            f"  Score              : {svm['score']:.2f}\n\n"
            f"  nu: fraction of training points that are\n"
            f"  not support vectors (= boundary flexibility)."
        )

        self.km_text.setPlainText(
            f"K-Means Clustering\n{'─'*45}\n"
            f"  Optimal k              : {km['best_k']}\n"
            f"  Best silhouette score  : {km['best_silhouette']}\n"
            f"  Target cluster         : {km['target_cluster']}\n"
            f"  Cluster size fraction  : {km['cluster_size_frac']:.2%}\n"
            f"  Distance to centre     : {km['dist_to_center']:.4f}\n"
            f"  Score                  : {km['score']:.2f}\n\n"
            f"Silhouette Values:\n" +
            "\n".join(f"  k={k}: {v}" for k,v in km['silhouette'].items())
        )

        vote_str = "\n".join(
            f"    {model:<20}: {zone}"
            for model,zone in tz['votes'].items()
        )
        self.tz_text.setPlainText(
            f"Multi-Threshold Zone Voting\n{'─'*45}\n\n"
            f"Model Votes:\n{vote_str}\n\n"
            f"  Mean vote   : {tz['mean_vote']:.3f}  (0=OUTSIDE, 3=CORE)\n"
            f"  Decision    : {tz['final_zone']}\n"
            f"  Zone label  : {tz['zone_label']}\n"
            f"  Score       : {tz['score']:.2f}\n\n"
            f"Mahalanobis Thresholds:\n"
            f"  D₅₀ = {m['d50_threshold']:.4f}  → CORE boundary (chi2 50%)\n"
            f"  D₉₀ = {m['d90_threshold']:.4f}  → SUITABLE boundary (chi2 90%)\n"
            f"  D₉₅ = {m['d95_threshold']:.4f}  → MARGINAL boundary (chi2 95%)\n"
            f"  Target D = {m['distance']:.4f}  → {m['interpretation']}"
        )

    def _populate_bio_tab(self):
        pd       = self.results["percentile"]["per_bio"]
        bio_cols = self.results["composite"].get("active_bio_cols", list(pd.keys()))
        self.bio_table.setRowCount(len(bio_cols))
        for row, bio in enumerate(bio_cols):
            d  = pd[bio]
            vs = [bio.upper(), d["name"][:28], f"{d['target_value']:.2f}",
                  f"{d['species_p5']:.2f}", f"{d['species_p25']:.2f}",
                  f"{d['species_median']:.2f}", f"{d['species_p75']:.2f}",
                  f"{d['species_p95']:.2f}", f"{d['score']:.1f}",
                  f"{d['mw_pvalue']:.4f}"]
            for col, val in enumerate(vs):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt_AlignCenter)
                if col==8:
                    s = d["score"]
                    item.setBackground(QColor("#d5f5e3") if s>=65 else
                                       QColor("#fef9e7") if s>=40 else QColor("#fdecea"))
                if col==0 and bio in CRITICAL_BIOS:
                    item.setBackground(QColor("#d6eaf8"))
                if col==9 and d["mw_pvalue"] < 0.05:
                    item.setForeground(QColor("#e74c3c"))
                self.bio_table.setItem(row, col, item)

    def _populate_topo_tab(self):
        t = self.results["topo"]
        if not t.get("elevation"):
            self.topo_text.setPlainText("No topographic variables were included in this analysis."); return

        e,s,a = t["elevation"], t["slope"], t["aspect"]
        self.topo_text.setPlainText(
            f"TOPOGRAPHIC COMPATIBILITY REPORT\n{'═'*50}\n\n"
            f"Composite Topography Score : {t['score']:.2f} / 100\n\n"
            f"── ELEVATION ({'Within' if e['in_tolerance_range'] else 'Outside'})\n"
            f"   Target         : {e['target_value']} m\n"
            f"   Species range  : [{e['species_p5']} – {e['species_p95']}] m  "
            f"(median: {e['species_median']} m)\n"
            f"   Percentile     : P{e['percentile']:.0f}\n"
            f"   Score          : {e['score']:.1f}\n\n"
            f"── SLOPE ({'Within' if s['in_tolerance_range'] else 'Outside'})\n"
            f"   Target         : {s['target_value']}%\n"
            f"   Species range  : [{s['species_p5']}% – {s['species_p95']}%]  "
            f"(median: {s['species_median']}%)\n"
            f"   Percentile     : P{s['percentile']:.0f}\n"
            f"   Score          : {s['score']:.1f}\n\n"
            f"── ASPECT (Circular Cosine Method)\n"
            f"   Target aspect  : {a['target_deg']}°  ({a['target_exposure']})\n"
            f"   Species typical: {a['sp_exposure']}\n"
            f"   Target N/E     : {a['target_northness']:.4f} / "
            f"{a['target_eastness']:.4f}\n"
            f"   Species avg N/E: {a['sp_mean_north']:.4f} / {a['sp_mean_east']:.4f}\n"
            f"   Vector length  : {a['vec_length']:.4f}  "
            f"(0=uniform, 1=single direction)\n"
            f"   Cosine similarity: {a['cos_similarity']:.4f}\n"
            f"   Score          : {a['score']:.1f}\n"
        )

    def _populate_importance_tab(self):
        vi   = self.results["variable_importance"]
        ci   = vi["combined_importance"]
        rf   = vi["rf_importance"]
        pdi  = vi["pdi"]
        top5 = vi["top5_variables"]

        all_cols = list(ci.keys())
        self.imp_table.setRowCount(len(all_cols))

        for row, col in enumerate(all_cols):
            comb = ci[col]; rf_v = rf.get(col,0); pdi_v = pdi.get(col,0)
            items = [col.upper(), f"{rf_v:.5f}", f"{pdi_v:.4f}", f"{comb:.5f}"]
            for c,val in enumerate(items):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt_AlignCenter)
                if c==3:
                    # Highlight top five
                    if col in top5: item.setBackground(QColor("#fdebd0"))
                    item.setFont(QFont("Arial", 9,
                                       QFont_Bold if col in top5 else QFont_Normal))
                self.imp_table.setItem(row, c, item)

    def _populate_opt_tab(self):
        """Populates the optional variable (srad/wind/vapr) percentile results."""
        po = self.results.get("percentile", {}).get("per_opt", {})
        if not po:
            self.opt_text.setPlainText(
                "No optional variables included in this analysis.\n"
                "Select srad / wind / vapr in the Data Input tab.")
            return

        lines = ["OPTIONAL VARIABLE ANALYSIS RESULTS", "═"*60, ""]
        for col, d in po.items():
            emoji = d.get("risk_emoji","—")
            lvl   = d.get("risk_level","—")
            lines += [
                f"{emoji}  {d['name']}  [{col}]",
                f"   Score      : {d['score']:.1f} / 100   Risk: {lvl}",
                f"   Target     : {d['target_value']:.4g}",
                f"   P5–P95     : [{d['species_p5']:.4g} – {d['species_p95']:.4g}]  "
                f"Median: {d['species_median']:.4g}",
                f"   Position   : P{d['percentile']:.0f} percentile   "
                f"Deviation {d['dev_pct']:+.1f}% from median  ({d['n_std']:+.2f} std)",
                f"   Dist to P5 : {d['dist_to_p5']:+.4g}   "
                f"Dist to P95: {d['dist_to_p95']:+.4g}",
                "",
                "   " + "─"*50,
                "   Biological Rationale:",
            ]
            for ln in d.get("risk_explanation","").splitlines():
                lines.append("   " + ln)
            lines += ["", ""]
        self.opt_text.setPlainText("\n".join(lines))

    def _populate_risk_cards(self, top_vars: list):
        """
        Adds a narrow card with numerical summary for each
        risk variable to the Summary tab.
        """
        vi = self.results.get("variable_importance", {})
        rd = vi.get("risk_details", {})

        # Clear existing cards (except stretch)
        while self.risk_cards_layout.count() > 1:
            item = self.risk_cards_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        color_bg = {"CRITICAL":"#fdecea","HIGH":"#fef5e7",
                    "MODERATE":"#fef9e7","LOW":"#eafaf1"}
        color_bd = {"CRITICAL":"#e74c3c","HIGH":"#e67e22",
                    "MODERATE":"#f39c12","LOW":"#27ae60"}

        for i, col in enumerate(top_vars[:7]):
            info = rd.get(col, {})
            lvl  = info.get("risk_level", "MODERATE")
            emoji= info.get("risk_emoji", "🟡")
            sc   = info.get("score", 0)
            pct  = info.get("percentile", 0)
            dev  = info.get("dev_pct", 0)
            nstd = info.get("n_std", 0)
            expl = info.get("risk_explanation", "")

            # Extract biological rationale line from explanation
            bio_line = ""
            for ln in expl.splitlines():
                ln = ln.strip()
                if ln and not ln.startswith("─") and not ln.startswith("═") and ":" in ln:
                    if any(k in ln for k in ("°C","mm","kPa","kJ","m s","std","P5","P95","VPD","photosynthesis","stomatal","cambium","transpir","vernaliz")):
                        bio_line = ln[:90]
                        break
            if not bio_line:
                # Fallback: first meaningful line
                for ln in expl.splitlines():
                    ln = ln.strip()
                    if len(ln) > 20 and not ln.startswith("─") and not ln.startswith("═"):
                        bio_line = ln[:90]
                        break

            bg = color_bg.get(lvl, "#fef9e7")
            bd = color_bd.get(lvl, "#f39c12")

            # Kart widget
            card = QFrame()
            card.setFixedWidth(200)
            card.setMinimumHeight(160)
            card.setStyleSheet(
                f"background:{bg};"
                f"border:2px solid {bd};"
                f"border-radius:6px;"
                f"padding:4px;"
            )
            card_lay = QVBoxLayout(card)
            card_lay.setSpacing(3)
            card_lay.setContentsMargins(6,6,6,6)

            # Title
            t_lbl = QLabel(f"{emoji} #{i+1}  {col.upper()}")
            t_lbl.setStyleSheet(
                f"font-weight:bold;font-size:11px;color:{bd};"
                f"border:none;padding:0;")
            card_lay.addWidget(t_lbl)

            # Variable name
            vname = ALL_VAR_NAMES.get(col, col)[:38]
            n_lbl = QLabel(vname)
            n_lbl.setStyleSheet("font-size:9px;color:#555;border:none;padding:0;")
            n_lbl.setWordWrap(True)
            card_lay.addWidget(n_lbl)

            # Numerical rows
            sep = QFrame(); sep.setFrameShape(QFrame_HLine)
            sep.setStyleSheet(f"background:{bd};border:none;max-height:1px;")
            card_lay.addWidget(sep)

            def stat_row(label, value):
                lw = QLabel(f"<b>{label}</b>  {value}")
                lw.setStyleSheet("font-size:10px;color:#2c3e50;border:none;padding:0;")
                return lw

            card_lay.addWidget(stat_row("Score:", f"{sc:.1f} / 100"))
            card_lay.addWidget(stat_row("Percentile:", f"P{pct:.0f}"))
            card_lay.addWidget(stat_row("Deviation:", f"{dev:+.1f}%  /  {nstd:+.2f} std"))
            card_lay.addWidget(stat_row("Risk level:", f"{lvl}"))

            # Biological rationale summary
            if bio_line:
                sep2 = QFrame(); sep2.setFrameShape(QFrame_HLine)
                sep2.setStyleSheet(f"background:{bd};border:none;max-height:1px;")
                card_lay.addWidget(sep2)
                bio_lbl = QLabel(bio_line)
                bio_lbl.setWordWrap(True)
                bio_lbl.setStyleSheet(
                    "font-size:9px;color:#444;border:none;padding:0;font-style:italic;")
                card_lay.addWidget(bio_lbl)

            card_lay.addStretch()
            self.risk_cards_layout.insertWidget(i, card)

    def _populate_risk_detail_tab(self):
        """Populates the sub-tabs with detailed explanations for the top-7 risk variables."""
        vi  = self.results.get("variable_importance", {})
        rd  = vi.get("risk_details", {})
        top = vi.get("top7_variables", vi.get("top5_variables", []))

        color_map = {
            "CRITICAL": "#e74c3c", "HIGH": "#e67e22",
            "MODERATE": "#f39c12", "LOW":  "#27ae60",
        }

        for i in range(7):
            if i < len(top):
                col  = top[i]
                info = rd.get(col, {})
                lvl  = info.get("risk_level", "—")
                emoji= info.get("risk_emoji", "—")
                sc   = info.get("score", 0)
                pct  = info.get("percentile", 0)
                expl = info.get("risk_explanation", "No explanation found.")

                # Tab title
                short = col.upper()[:10]
                self.risk_subtabs.setTabText(i, f"{emoji} {short}")
                color = color_map.get(lvl, "#7f8c8d")
                self.risk_subtabs.setTabToolTip(i, lvl)

                # Content
                hdr = (f"{'═'*60}\n"
                       f"  #{i+1}  {ALL_VAR_NAMES.get(col, col)}  [{col}]\n"
                       f"{'═'*60}\n"
                       f"  Rank   : #{i+1}  |  Combined Importance = "
                       f"{info.get('combined_importance',0):.5f}\n"
                       f"  RF Imp : {info.get('rf_importance',0):.5f}   "
                       f"PDI     : {info.get('pdi',0):.4f}\n"
                       f"  Score  : {sc:.1f} / 100   "
                       f"P{pct:.0f} percentile   "
                       f"Deviation {info.get('dev_pct',0):+.1f}% from median   "
                       f"{info.get('n_std',0):+.2f} std\n"
                       f"{'─'*60}\n")
                self._risk_subtab_widgets[i].setPlainText(hdr + expl)

                # Tab colour (if supported)
                try:
                    from qgis.PyQt.QtGui import QColor as _QC
                    tab_bar = self.risk_subtabs.tabBar()
                    tab_bar.setTabTextColor(i, _QC(color))
                except Exception:
                    pass
            else:
                self.risk_subtabs.setTabText(i, "—")
                self._risk_subtab_widgets[i].setPlainText("")

    # ── Charts ─────────────────────────────────────────────────────────────
    def _draw_charts(self):
        self.fig.clear()
        r = self.results

        ax1 = self.fig.add_subplot(2,3,1)  # PCA
        ax2 = self.fig.add_subplot(2,3,2)  # ML scores
        ax3 = self.fig.add_subplot(2,3,3)  # Bio scores
        ax4 = self.fig.add_subplot(2,3,4)  # Variable importance (top10)
        ax5 = self.fig.add_subplot(2,3,5)  # Topo
        ax6 = self.fig.add_subplot(2,3,6,projection="polar")  # Aspect

        bg = "#f8f9fa"
        for ax in (ax1,ax2,ax3,ax4,ax5): ax.set_facecolor(bg)
        self.fig.patch.set_facecolor("#ffffff")

        # PCA
        pca_d = r["pca"]
        sp_pts = np.array(pca_d["species_pc2"])
        tgt_pt = np.array(pca_d["target_pc2"])
        ev = pca_d["ev_2"]
        ax1.scatter(sp_pts[:,0],sp_pts[:,1],alpha=0.4,s=18,color="#3498db",label="Native range")
        ax1.scatter(tgt_pt[0],tgt_pt[1],s=150,color="#e74c3c",marker="*",zorder=5,label="Target")
        ax1.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)",fontsize=9)
        ax1.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)",fontsize=9)
        ax1.set_title("PCA Climate Space",fontsize=10,fontweight="bold")
        ax1.legend(fontsize=8)

        # ML model scores
        ml_labels = ["Zone","GMM","IsoForest","OCSVM","Mahal.","Topo"]
        ml_scores = [
            r["threshold_zone"]["score"],
            r["gmm"]["score"],
            r["isolation_forest"]["score"],
            r["ocsvm"]["score"],
            r["mahalanobis_stat"]["score"],
            r["topo"]["score"],
        ]
        ml_colors = ["#27ae60" if s>=65 else "#f39c12" if s>=40 else "#e74c3c"
                     for s in ml_scores]
        bars = ax2.bar(ml_labels, ml_scores, color=ml_colors)
        ax2.axhline(50, color="gray", ls="--", lw=0.8)
        ax2.set_ylim(0, 115)
        ax2.set_title("ML Model Scores", fontsize=10, fontweight="bold")
        ax2.tick_params(axis="x", labelsize=7, rotation=20)
        ax2.tick_params(axis="y", labelsize=7)
        for bar, sc in zip(bars, ml_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, sc + 2, f"{sc:.0f}",
                     ha="center", va="bottom", fontsize=7)

        # Bio scores — use only the active cols that exist in results
        _per_bio = r["percentile"]["per_bio"]
        _bio_keys = r["composite"].get("active_bio_cols", list(_per_bio.keys()))
        bio_sc  = [_per_bio[b]["score"] for b in _bio_keys if b in _per_bio]
        _bio_lbl = [b.upper() for b in _bio_keys if b in _per_bio]
        bio_col = ["#27ae60" if s>=65 else "#f39c12" if s>=40 else "#e74c3c"
                   for s in bio_sc]
        ax3.barh(_bio_lbl, bio_sc, color=bio_col)
        ax3.axvline(50, color="gray", ls="--", lw=0.8)
        ax3.set_xlim(0, 100)
        ax3.tick_params(axis="y", labelsize=7)
        ax3.set_title("Bio Variable Scores", fontsize=10, fontweight="bold")

        # Variable importance top-10
        ci   = r["variable_importance"]["combined_importance"]
        top10= list(ci.items())[:10]
        labels_imp = [k for k,v in top10]
        vals_imp   = [v for k,v in top10]
        imp_colors = ["#e74c3c" if k in r["variable_importance"]["top5_variables"]
                      else "#3498db" for k in labels_imp]
        ax4.barh(labels_imp[::-1],vals_imp[::-1],color=imp_colors[::-1])
        ax4.set_title("Variable Importance (Top 10)",fontsize=10,fontweight="bold")
        ax4.tick_params(axis="y",labelsize=8)

        # Topo sub-scores
        t = r["topo"]
        if t.get("elevation"):
            tl = ["Elevation","Slope","Aspect"]
            tv = [t["elevation"]["score"],t["slope"]["score"],t["aspect"]["score"]]
            tc = ["#27ae60" if v>=65 else "#f39c12" if v>=40 else "#e74c3c" for v in tv]
            b5 = ax5.bar(tl,tv,color=tc,width=0.5)
            ax5.axhline(50,color="gray",ls="--",lw=0.8)
            ax5.set_ylim(0,105); ax5.set_title("Topographic Sub-scores",fontsize=10,fontweight="bold")
            for bar,v in zip(b5,tv):
                ax5.text(bar.get_x()+bar.get_width()/2,v+1,f"{v:.0f}",ha="center",fontsize=9)

        # Aspect polar histogram
        try:
            sp_asp_rad = np.deg2rad(self.worker.species_data[:,21])
            ax6.hist(sp_asp_rad,bins=16,color="#3498db",alpha=0.7,density=True)
            tgt_rad = np.deg2rad(t["aspect"]["target_deg"])
            ax6.axvline(tgt_rad,color="#e74c3c",lw=2.5,label="Target")
            ax6.set_theta_zero_location("N"); ax6.set_theta_direction(-1)
            ax6.set_title("Aspect Distribution",fontsize=10,fontweight="bold",pad=15)
            ax6.legend(fontsize=8,loc="lower right")
        except Exception:
            pass

        self.fig.tight_layout(pad=2.5, h_pad=3.0, w_pad=2.5)
        self.canvas.draw()

        # ── Auto-export: save 300 dpi PNG to the output folder ─────────
        self._auto_save_charts_png()
        if hasattr(self, "btn_export_charts"):
            self.btn_export_charts.setEnabled(True)

    # ── Auto-save charts as 300 dpi PNG ────────────────────────────────
    def _auto_save_charts_png(self):
        """Automatically save the figure as a 300-dpi PNG next to the
        species CSV file (or the working directory as fallback)."""
        filename = self._default_export_name("charts", "png")

        sp_path = getattr(self, "le_sp_path", None)
        if sp_path and sp_path.text():
            out_dir = os.path.dirname(sp_path.text())
        else:
            out_dir = os.path.expanduser("~")

        save_path = os.path.join(out_dir, filename)
        try:
            self.fig.savefig(
                save_path,
                dpi=300,
                format="png",
                bbox_inches="tight",
                facecolor=self.fig.get_facecolor()
            )
            if hasattr(self, "lbl_export_status"):
                self.lbl_export_status.setText(
                    f"✅ Otomatik kaydedildi: {filename}"
                )
            try:
                from qgis.core import QgsMessageLog, Qgis
                QgsMessageLog.logMessage(
                    f"[CRESTA] Charts auto-saved (300 dpi): {save_path}",
                    "CRESTA", Qgis.Info
                )
            except Exception:
                pass
        except Exception as exc:
            if hasattr(self, "lbl_export_status"):
                self.lbl_export_status.setText(f"⚠️ Otomatik kayıt başarısız: {exc}")

    # ── Report ─────────────────────────────────────────────────────────────────
    def _build_report(self):
        r    = self.results
        comp = r["composite"]
        tz   = r["threshold_zone"]
        m    = r["mahalanobis_stat"]
        g    = r["gmm"]
        iso  = r["isolation_forest"]
        svm  = r["ocsvm"]
        km   = r["kmeans"]
        p    = r["percentile"]
        t    = r["topo"]
        vi   = r["variable_importance"]

        # Bootstrap CI summary
        bci = r.get("bootstrap_ci", {})
        ci_str = (f"{bci['ci_lower']:.1f} – {bci['ci_upper']:.1f}  "
                  f"(±{bci['std']:.1f},  n={bci['n_bootstrap']})"
                  if bci.get("n_valid", 0) > 0 else "—")
        # Spatial AC / blocked CV summary
        bcv = r.get("blocked_cv", {})
        ac_str = (f"{bcv['interpretation']}  "
                  f"(mean inlier rate {bcv['mean_inlier_rate']:.0%},  "
                  f"{bcv['n_blocks']} blocks)"
                  if bcv.get("mean_inlier_rate") is not None else "—")
        # Thinning summary
        thin = r.get("thinning", {})
        thin_str = (f"{thin['n_before']} → {thin['n_after']} points  "
                    f"(retained {thin['retained_fraction']:.0%},  "
                    f"min PCA dist={thin['min_dist_pca']})"
                    if thin.get("n_after") is not None else "—")

        # Thinned analysis (spatial correction) summary
        ta = r.get("thinned_analysis", {}) or {}
        if ta.get("applied"):
            ta_str = (
                f"APPLIED — raw {ta['raw_composite']:.1f} → "
                f"corrected {ta['thinned_composite']:.1f}  "
                f"(Δ {ta['delta']:+.1f},  "
                f"{ta['n_before']}→{ta['n_after']} pts)"
                + ("  [CLASS CHANGED]" if ta.get("class_changed") else "")
            )
        else:
            ta_str = ta.get("reason", "—")

        # Sensitivity / unsupervised weight optimization
        sens = r.get("sensitivity", {}) or {}
        if "error" not in sens and sens.get("robustness"):
            sens_str = (
                f"{sens['robustness']} robustness — range ±{sens['target_score_range']/2:.1f}, "
                f"class flips {sens['class_flip_fraction']*100:.0f}% of grid"
            )
            opt_str = (
                f"default {sens['target_score_default']:.1f} → "
                f"optimal {sens['target_score_optimal']:.1f}  "
                f"(Δ {sens['target_score_delta']:+.1f},  "
                f"LOO gain {sens['loo_score_gain']:+.1f})"
            )
        else:
            sens_str = sens.get("error", "—")
            opt_str  = "—"

        lines = [
            "╔══════════════════════════════════════════════════════════════════╗",
            "║   Climate Resilience Analyzer v1.0.0  —  ML + Multi-Threshold Score ║",
            "╚══════════════════════════════════════════════════════════════════╝",
            "",
            f"  SPECIES               : {getattr(self, 'species_name', None) or '—'}",
            f"  COMPOSITE RESILIENCE SCORE : {comp['composite_score']:.2f} / 100",
            f"  Resilience Class       : {comp['resilience_class']}",
            f"  Niche Zone             : {comp['zone_label']}",
            f"  Climate Score          : {comp['climate_score']:.2f}  ({int(comp['climate_weight']*100)}%)",
            f"  Topography Score       : {comp['topo_score']:.2f}  ({int(comp['topo_weight']*100)}%)",
            f"  Bootstrap 95% CI       : {ci_str}",
            f"  Bootstrap method       : {bci.get('method', '—')}",
            f"  Spatial AC (Blocked CV): {ac_str}",
            f"  Environmental Thinning : {thin_str}",
            f"  Spatial Correction     : {ta_str}",
            f"  Weight Sensitivity     : {sens_str}",
            f"  Unsupervised Optimum   : {opt_str}",
            "",
            "──────────────────────────────────────────────────────────────────",
            "  ML MODEL RESULTS",
            "──────────────────────────────────────────────────────────────────",
            f"  Multi-Threshold Zone : {tz['final_zone']:<12} score={tz['score']:.2f}",
            f"    GMM              : {g['zone']:<12} log-p={g['log_probability']:.3f}  n_comp={g['best_n_components']}",
            f"    Isolation Forest: {iso['zone']:<12} anom={iso['anomaly_score']:.4f}  normal={iso['is_normal']}",
            f"    One-Class SVM    : {svm['zone']:<12} dec={svm['decision_value']:.4f}  inside={svm['is_inside']}",
            f"    Mahalanobis     : {m['zone']:<12} D={m['distance']:.4f}",
            f"    K-Means          : k={km['best_k']}  sil={km['best_silhouette']:.4f}  cluster={km['target_cluster']}",
            "",
            "──────────────────────────────────────────────────────────────────",
            "  MAHALANOBIS THRESHOLDS",
            "──────────────────────────────────────────────────────────────────",
            f"  D₅₀  (Chi²50%) = {m['d50_threshold']:.4f}  → Core niche boundary",
            f"  D₉₀  (Chi²90%) = {m['d90_threshold']:.4f}  → Suitable niche outer boundary",
            f"  D₉₅  (Chi²95%) = {m['d95_threshold']:.4f}  → Marginal niche outer boundary",
            f"  Target D       = {m['distance']:.4f}  → {m['interpretation']}",
            "",
            "──────────────────────────────────────────────────────────────────",
            "  HIGHEST RISK VARIABLES — NUMERICAL RATIONALE",
            "──────────────────────────────────────────────────────────────────",
        ]
        rd  = vi.get("risk_details", {})
        top = vi.get("top7_variables", vi.get("top5_variables", []))
        for idx, v in enumerate(top, 1):
            rf_v  = vi["rf_importance"].get(v, 0)
            pdi_v = vi["pdi"].get(v, 0)
            info  = rd.get(v, {})
            lvl   = info.get("risk_level", "—")
            emoji = info.get("risk_emoji", "—")
            sc    = info.get("score", 0)
            pct   = info.get("percentile", 0)
            dev   = info.get("dev_pct", 0)
            nstd  = info.get("n_std", 0)
            expl  = info.get("risk_explanation", "")
            lines.append(
                f"  {emoji} #{idx:<2} {v:<16} "
                f"Score={sc:.1f}  P{pct:.0f}  {dev:+.1f}%  {nstd:+.2f}std  "
                f"RF={rf_v:.5f}  PDI={pdi_v:.4f}  [{lvl}]"
            )
            # Append the main biological rationale sentence
            for ln in expl.splitlines():
                ln = ln.strip()
                if (len(ln) > 20 and not ln.startswith("─") and
                        not ln.startswith("═") and
                        any(k in ln for k in ("°C","mm","kPa","kJ","m s⁻",
                                               "std","photosynthesis","stomatal",
                                               "cambium","transpir","vernaliz",
                                               "VPD","PSII","cavitation","radiation",
                                               "wind","fungal","erosion"))):
                    lines.append(f"       → {ln[:100]}")
                    break
            lines.append("")

        lines += [
            "",
            "──────────────────────────────────────────────────────────────────",
            "  BIO VARIABLE SUMMARY",
            "──────────────────────────────────────────────────────────────────",
            f"  Within P25-P75      : {p['bios_in_core_range']} / 19",
            f"  Within P5-P95       : {p['bios_in_tolerance_range']} / 19",
            f"  Outside tolerance   : {p['bios_outside_range']} / 19",
        ]

        if t.get("elevation"):
            e,s,a = t["elevation"],t["slope"],t["aspect"]
            lines += [
                "",
                "──────────────────────────────────────────────────────────────────",
                "  TOPOGRAPHY",
                "──────────────────────────────────────────────────────────────────",
                f"  Elevation : {e['target_value']} m   P{e['percentile']:.0f}   score={e['score']:.1f}",
                f"  Slope     : {s['target_value']}%   P{s['percentile']:.0f}   score={s['score']:.1f}",
                f"  Aspect    : {a['target_deg']}°  ({a['target_exposure']})   cos={a['cos_similarity']:.4f}   score={a['score']:.1f}",
            ]

        # SSP summary in report
        ssp = r.get("ssp_scenarios")
        if ssp:
            lines += [
                "",
                "──────────────────────────────────────────────────────────────────",
                "  FUTURE CLIMATE SCENARIOS",
                "──────────────────────────────────────────────────────────────────",
                f"  Current score : {ssp['current_score']:.1f}",
            ]
            for sname, sr in ssp.get("scenarios",{}).items():
                if "error" in sr:
                    lines.append(f"  {sname:<30}: ERROR")
                else:
                    lines.append(
                        f"  {sname:<30}: {sr['composite_score']:.1f}  "
                        f"({sr['delta_from_current']:+.1f})  {sr['trend']}")

        lines += [
            "",
            "──────────────────────────────────────────────────────────────────",
            "  RECOMMENDATION",
            "──────────────────────────────────────────────────────────────────",
            f"  {comp['recommendation']}",
            "",
            "══════════════════════════════════════════════════════════════════",
        ]

        self.report_text.setPlainText("\n".join(lines))

    def _copy_report(self):
        from qgis.PyQt.QtWidgets import QApplication
        QApplication.clipboard().setText(self.report_text.toPlainText())

    # ── Full Text Report ──────────────────────────────────────────────────────
    def _build_full_report_text(self):
        """
        Builds a comprehensive plain-text report covering every analysis tab:
        Data Input · Summary · ML Models · Bio Detail · Topo · Optional ·
        Variable Importance · Risk Details · Bootstrap CI · Spatial AC ·
        SSP Scenarios (if run) · Weight Calibration (if run).
        Returns the full text as a single string.
        """
        import datetime
        r    = self.results
        comp = r["composite"]
        tz   = r["threshold_zone"]
        m    = r["mahalanobis_stat"]
        g    = r["gmm"]
        iso  = r["isolation_forest"]
        svm  = r["ocsvm"]
        km   = r["kmeans"]
        p    = r["percentile"]
        t    = r["topo"]
        vi   = r["variable_importance"]
        bci  = r.get("bootstrap_ci", {})
        bcv  = r.get("blocked_cv",   {})
        thin = r.get("thinning",     {})

        W  = "═" * 70
        S1 = "─" * 70
        S2 = "· " * 35

        def section(title):
            return ["", W, f"  {title}", W]

        def sub(title):
            return ["", f"  ── {title} {'─'*(60-len(title))}"]

        lines = []

        # ══════════════════════════════════════════════════════════════════════
        # HEADER
        species_name = getattr(self, "species_name", None) or "—"
        lines += [
            W,
            "  CRESTA — Climate Resilience Analyzer  v1.0.0",
            "  ML + Multi-Threshold Score System",
            f"  Species:   {species_name}",
            f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            W,
        ]

        # ══════════════════════════════════════════════════════════════════════
        # 1. DATA INPUT
        lines += section("1. DATA INPUT")

        bio_cols  = comp.get("active_bio_cols",  [])
        topo_cols = comp.get("active_topo_cols", [])
        opt_cols  = comp.get("active_opt_cols",  [])

        sp_src = ""
        if hasattr(self, "le_sp_path") and self.le_sp_path.text():
            sp_src = self.le_sp_path.text()
        elif hasattr(self, "cb_sp") and self.cb_sp.currentText():
            sp_src = f"QGIS Layer: {self.cb_sp.currentText()}"

        tgt_src = ""
        if hasattr(self, "le_tgt_path") and self.le_tgt_path.text():
            tgt_src = self.le_tgt_path.text()
        elif hasattr(self, "cb_tgt") and self.cb_tgt.currentText():
            tgt_src = f"QGIS Layer: {self.cb_tgt.currentText()}"

        sp_data_ref = getattr(self, "_sp_data_cache", None)
        n_sp = len(sp_data_ref) if sp_data_ref is not None else "—"

        lines += [
            f"  Species              : {species_name}",
            f"  Species native range : {sp_src or '—'}",
            f"  Target site          : {tgt_src or '—'}",
            f"  Native range points  : {n_sp}",
            f"  Bioclimatic variables: {len(bio_cols)}  →  {', '.join(bio_cols) or '—'}",
            f"  Topographic variables: {len(topo_cols)}  →  {', '.join(topo_cols) or 'none'}",
            f"  Optional variables   : {len(opt_cols)}  →  {', '.join(opt_cols) or 'none'}",
            f"  Total variables used : {len(bio_cols)+len(topo_cols)+len(opt_cols)}",
        ]
        if thin.get("n_after") is not None:
            lines.append(
                f"  Environmental thinning: {thin['n_before']} → {thin['n_after']} pts  "
                f"(retained {thin['retained_fraction']:.0%}, min PCA dist={thin['min_dist_pca']})")

        # ══════════════════════════════════════════════════════════════════════
        # 2. SUMMARY
        lines += section("2. SUMMARY")

        ci_str = (f"{bci['ci_lower']:.1f} – {bci['ci_upper']:.1f}  "
                  f"(±{bci['std']:.1f},  n={bci['n_bootstrap']})"
                  if bci.get("n_valid", 0) > 0 else "—")
        ac_str = (f"{bcv['interpretation']}  "
                  f"(mean inlier rate {bcv['mean_inlier_rate']:.0%}, {bcv['n_blocks']} blocks)"
                  if bcv.get("mean_inlier_rate") is not None else "—")

        lines += [
            f"  Composite Score      : {comp['composite_score']:.2f} / 100",
            f"  95% Bootstrap CI     : {ci_str}",
            f"  Resilience Class     : {comp['resilience_class']}",
            f"  Niche Zone           : {comp['zone_label']}",
            f"  Climate Score        : {comp['climate_score']:.2f}  (weight {int(comp['climate_weight']*100)}%)",
            f"  Topography Score     : {comp['topo_score']:.2f}  (weight {int(comp['topo_weight']*100)}%)",
            f"  Optional Bonus       : +{comp.get('opt_bonus', 0):.3f}",
            f"  Spatial AC (Blocked CV): {ac_str}",
            "",
            f"  Recommendation:",
            f"  {comp['recommendation']}",
        ]

        # Component scores
        lines += sub("Component Score Breakdown")
        comps = comp.get("component_scores", {})
        w_map = {"threshold_zone": "35%", "gmm": "20%", "isolation_forest": "10%",
                 "ocsvm": "10%", "mahalanobis": "25%"}
        comp_labels = [
            ("threshold_zone",   "Multi-Threshold Zone Voting"),
            ("gmm",              "Gaussian Mixture Model (GMM)"),
            ("isolation_forest", "Isolation Forest"),
            ("ocsvm",            "One-Class SVM"),
            ("mahalanobis",      "Mahalanobis Distance"),
            ("topo",             "Topography"),
        ]
        for key, lbl in comp_labels:
            sc_ = comps.get(key, t["score"] if key == "topo" else 0)
            wt_ = w_map.get(key, f"{int(comp['topo_weight']*100)}%")
            bar = "█" * int(sc_ / 5) + "░" * (20 - int(sc_ / 5))
            lines.append(f"  {lbl:<35}: {sc_:5.1f}  [{bar}]  wt={wt_}")

        # ══════════════════════════════════════════════════════════════════════
        # 3. ML MODELS
        lines += section("3. ML MODELS — DETAILED RESULTS")

        lines += sub("3.1  Gaussian Mixture Model (GMM)")
        lines += [
            f"  Optimal components   : {g['best_n_components']}",
            f"  Target log-prob      : {g['log_probability']:.4f}",
            f"  Max component prob   : {g['max_component_prob']:.4f}",
            f"  Niche breadth (Levins): {g['niche_breadth']:.4f}",
            f"  Zone                 : {g['zone']}",
            f"  Score                : {g['score']:.2f}",
            "  BIC values           :",
        ]
        for k, v in g.get("bic_values", {}).items():
            lines.append(f"    {k} components: {v:.2f}")

        lines += sub("3.2  Isolation Forest")
        lines += [
            f"  Anomaly score        : {iso['anomaly_score']}  (higher = more normal)",
            f"  Is normal?           : {'Yes' if iso['is_normal'] else 'No'}",
            f"  Percentile rank      : {iso['percentile_rank']:.1f}%",
            f"  Species Q10          : {iso['sp_q10']}",
            f"  Species Q25          : {iso['sp_q25']}",
            f"  Species Q75          : {iso['sp_q75']}",
            f"  Zone                 : {iso['zone']}",
            f"  Score                : {iso['score']:.2f}",
        ]

        lines += sub("3.3  One-Class SVM (RBF kernel)")
        lines += [
            f"  Decision value       : {svm['decision_value']}  (>0 = inside)",
            f"  Is inside boundary?  : {'Yes' if svm['is_inside'] else 'No'}",
            f"  Percentile rank      : {svm['percentile_rank']:.1f}%",
            f"  nu parameter         : {svm['nu']}",
            f"  Zone                 : {svm['zone']}",
            f"  Score                : {svm['score']:.2f}",
        ]

        lines += sub("3.4  K-Means Clustering")
        lines += [
            f"  Optimal k            : {km['best_k']}",
            f"  Best silhouette      : {km['best_silhouette']:.4f}",
            f"  Target cluster       : {km['target_cluster']}",
            f"  Cluster size frac.   : {km['cluster_size_frac']:.2%}",
            f"  Distance to centre   : {km['dist_to_center']:.4f}",
            f"  Score                : {km['score']:.2f}",
            "  Silhouette by k      :",
        ]
        for k, v in km.get("silhouette", {}).items():
            lines.append(f"    k={k}: {v}")

        lines += sub("3.5  Multi-Threshold Zone Voting")
        lines += [
            f"  Final zone           : {tz['final_zone']}",
            f"  Zone label           : {tz['zone_label']}",
            f"  Weighted mean vote   : {tz['mean_vote']:.3f}  (0=OUTSIDE, 3=CORE)",
            f"  Score                : {tz['score']:.2f}",
            "  Model votes          :",
        ]
        for mdl, zone_ in tz.get("votes", {}).items():
            lines.append(f"    {mdl:<20}: {zone_}")

        lines += sub("3.6  Mahalanobis Distance")
        lines += [
            f"  Distance D           : {m['distance']:.4f}",
            f"  Chi² p-value         : {m['p_value']:.4f}",
            f"  D₅₀  (Chi² 50%)      : {m['d50_threshold']:.4f}  → Core boundary",
            f"  D₉₀  (Chi² 90%)      : {m['d90_threshold']:.4f}  → Suitable boundary",
            f"  D₉₅  (Chi² 95%)      : {m['d95_threshold']:.4f}  → Marginal boundary",
            f"  D₉₉  (Chi² 99%)      : {m['d99_threshold']:.4f}",
            f"  Zone                 : {m['zone']}",
            f"  Score                : {m['score']:.2f}",
            f"  Interpretation       : {m['interpretation']}",
        ]

        # ══════════════════════════════════════════════════════════════════════
        # 4. BIO DETAIL
        lines += section("4. BIOCLIMATIC VARIABLES — DETAIL")
        lines += [
            f"  {'Variable':<8}  {'Target':>8}  {'P5':>8}  {'P25':>8}  "
            f"{'Median':>8}  {'P75':>8}  {'P95':>8}  {'Pct':>5}  {'Score':>6}  {'MW-p':>7}  Crit",
            "  " + "─"*90,
        ]
        per_bio = p.get("per_bio", {})
        for bio in comp.get("active_bio_cols", list(per_bio.keys())):
            d = per_bio.get(bio)
            if not d:
                continue
            crit_mark = "★" if bio in CRITICAL_BIOS else " "
            mw_flag   = "*" if d["mw_pvalue"] < 0.05 else " "
            lines.append(
                f"  {bio.upper():<8}  {d['target_value']:>8.2f}  {d['species_p5']:>8.2f}  "
                f"{d['species_p25']:>8.2f}  {d['species_median']:>8.2f}  "
                f"{d['species_p75']:>8.2f}  {d['species_p95']:>8.2f}  "
                f"P{d['percentile']:>3.0f}  {d['score']:>6.1f}  "
                f"{d['mw_pvalue']:>7.4f}{mw_flag}  {crit_mark}"
            )
        lines += [
            "",
            f"  ★ = critical variable (double weight)   * = MW-U p < 0.05 (significant difference)",
            f"  Within P25-P75 (core)      : {p['bios_in_core_range']}",
            f"  Within P5-P95 (tolerance)  : {p['bios_in_tolerance_range']}",
            f"  Outside P5-P95             : {p['bios_outside_range']}",
        ]

        # ══════════════════════════════════════════════════════════════════════
        # 5. TOPOGRAPHY
        lines += section("5. TOPOGRAPHY")
        if t.get("elevation"):
            e, s, a = t["elevation"], t.get("slope", {}), t.get("aspect", {})
            lines += [
                f"  Composite topo score : {t['score']:.2f}",
                "",
            ]
            if e:
                lines += [
                    f"  ELEVATION",
                    f"    Target           : {e.get('target_value')} m",
                    f"    Species range    : [{e.get('species_p5')} – {e.get('species_p95')}] m  (median {e.get('species_median')} m)",
                    f"    Percentile       : P{e.get('percentile', 0):.0f}",
                    f"    Score            : {e.get('score', 0):.1f}",
                    f"    In tolerance     : {'Yes' if e.get('in_tolerance_range') else 'No'}",
                ]
            if s:
                lines += [
                    "",
                    f"  SLOPE",
                    f"    Target           : {s.get('target_value')}%",
                    f"    Species range    : [{s.get('species_p5')}% – {s.get('species_p95')}%]  (median {s.get('species_median')}%)",
                    f"    Percentile       : P{s.get('percentile', 0):.0f}",
                    f"    Score            : {s.get('score', 0):.1f}",
                    f"    In tolerance     : {'Yes' if s.get('in_tolerance_range') else 'No'}",
                ]
            if a and a.get("target_deg") is not None:
                lines += [
                    "",
                    f"  ASPECT  (circular cosine method)",
                    f"    Target aspect    : {a.get('target_deg')}°  ({a.get('target_exposure', '')})",
                    f"    Species typical  : {a.get('sp_exposure', '')}",
                    f"    Target N/E       : {a.get('target_northness', 0):.4f} / {a.get('target_eastness', 0):.4f}",
                    f"    Species avg N/E  : {a.get('sp_mean_north', 0):.4f} / {a.get('sp_mean_east', 0):.4f}",
                    f"    Vector length    : {a.get('vec_length', 0):.4f}  (0=uniform, 1=single dir.)",
                    f"    Cosine similarity: {a.get('cos_similarity', 0):.4f}",
                    f"    Score            : {a.get('score', 0):.1f}",
                ]
        else:
            lines.append("  No topographic variables selected.")

        # ══════════════════════════════════════════════════════════════════════
        # 6. OPTIONAL VARIABLES
        lines += section("6. OPTIONAL VARIABLES (srad / wind / vapr)")
        per_opt = p.get("per_opt", {})
        if per_opt:
            for col, d in per_opt.items():
                lines += [
                    f"  {d.get('risk_emoji','')}  {d['name']}  [{col}]",
                    f"    Score          : {d['score']:.1f} / 100   Risk: {d.get('risk_level', '—')}",
                    f"    Target value   : {d['target_value']:.4g}",
                    f"    P5 – P95       : [{d['species_p5']:.4g} – {d['species_p95']:.4g}]  "
                    f"Median: {d['species_median']:.4g}",
                    f"    Percentile     : P{d['percentile']:.0f}",
                    f"    Deviation      : {d['dev_pct']:+.1f}% from median  ({d['n_std']:+.2f} std)",
                    "",
                ]
        else:
            lines.append("  No optional variables included.")

        # ══════════════════════════════════════════════════════════════════════
        # 7. VARIABLE IMPORTANCE
        lines += section("7. VARIABLE IMPORTANCE")
        lines += [
            f"  {'Variable':<18}  {'RF Importance':>14}  {'PDI':>8}  {'Combined':>10}  Top5",
            "  " + "─"*60,
        ]
        ci_imp = vi.get("combined_importance", {})
        rf_imp = vi.get("rf_importance", {})
        pdi_   = vi.get("pdi", {})
        top5   = vi.get("top5_variables", [])
        for col, comb in ci_imp.items():
            mark = "★" if col in top5 else " "
            lines.append(
                f"  {col.upper():<18}  {rf_imp.get(col,0):>14.5f}  "
                f"{pdi_.get(col,0):>8.4f}  {comb:>10.5f}  {mark}"
            )

        # ══════════════════════════════════════════════════════════════════════
        # 8. RISK DETAILS  (full biological rationale for top 7)
        lines += section("8. RISK DETAILS — TOP 7 VARIABLES")
        rd  = vi.get("risk_details", {})
        top7= vi.get("top7_variables", top5)
        for idx, col in enumerate(top7, 1):
            info  = rd.get(col, {})
            expl  = info.get("risk_explanation", "No explanation available.")
            lines += [
                "",
                f"  {'─'*68}",
                f"  #{idx}  {ALL_VAR_NAMES.get(col, col)}  [{col.upper()}]",
                f"  {'─'*68}",
                f"  Rank             : #{idx}",
                f"  Combined Importance: {info.get('combined_importance', 0):.5f}",
                f"  RF Importance    : {info.get('rf_importance', 0):.5f}",
                f"  PDI              : {info.get('pdi', 0):.4f}",
                f"  Score            : {info.get('score', 0):.1f} / 100",
                f"  Percentile       : P{info.get('percentile', 0):.0f}",
                f"  Deviation        : {info.get('dev_pct', 0):+.1f}% from median  "
                f"({info.get('n_std', 0):+.2f} std)",
                f"  Risk Level       : {info.get('risk_level', '—')}",
                "",
                "  Biological Rationale:",
            ]
            for ln in expl.splitlines():
                lines.append(f"    {ln}")

        # ══════════════════════════════════════════════════════════════════════
        # 9. BOOTSTRAP CI
        lines += section("9. BOOTSTRAP CONFIDENCE INTERVAL")
        if bci.get("n_valid", 0) > 0:
            lines += [
                f"  Method           : {bci.get('method', 'Bootstrap resampling')}",
                f"  Iterations       : {bci['n_bootstrap']}  (valid: {bci['n_valid']})",
                f"  Bootstrap mean   : {bci['mean']:.2f}",
                f"  Bootstrap std    : {bci['std']:.2f}",
                f"  95% CI           : [{bci['ci_lower']:.2f},  {bci['ci_upper']:.2f}]",
                f"  CI width         : {bci['ci_width']:.2f}",
                f"  Original score   : {comp['composite_score']:.2f}",
                "",
                "  Interpretation: CI width < 10 → stable estimate; > 20 → high uncertainty",
            ]
        else:
            lines.append(f"  {bci.get('error', 'Bootstrap not available.')}")

        # ══════════════════════════════════════════════════════════════════════
        # 10. SPATIAL AUTOCORRELATION  (Blocked CV + Thinning)
        lines += section("10. SPATIAL AUTOCORRELATION DIAGNOSTICS")
        lines += sub("10.1  Blocked Cross-Validation")
        if bcv.get("mean_inlier_rate") is not None:
            lines += [
                f"  Blocks (k-means)   : {bcv['n_blocks']}",
                f"  Mean inlier rate   : {bcv['mean_inlier_rate']:.3f}  "
                f"(±{bcv.get('std_inlier_rate', 0):.3f})",
                f"  Interpretation     : {bcv['interpretation']}",
                "  Per-block results  :",
            ]
            for blk, br in bcv.get("block_results", {}).items():
                if "error" in br:
                    lines.append(f"    {blk}: ERROR — {br['error']}")
                else:
                    lines.append(
                        f"    {blk}: n_train={br['n_train']}  "
                        f"n_test={br['n_test']}  inlier_rate={br['inlier_rate']:.4f}")
        else:
            lines.append(f"  {bcv.get('error', '—')}")

        lines += sub("10.2  Environmental Thinning")
        if thin.get("n_after") is not None:
            lines += [
                f"  Points before      : {thin['n_before']}",
                f"  Points after       : {thin['n_after']}",
                f"  Retained fraction  : {thin['retained_fraction']:.1%}",
                f"  Min PCA distance   : {thin['min_dist_pca']}",
                f"  Note               : {thin.get('note', '')}",
            ]
        else:
            lines.append("  —")

        # ══════════════════════════════════════════════════════════════════════
        # 11. SSP SCENARIOS  (conditional)
        ssp = r.get("ssp_scenarios")
        if ssp:
            lines += section("11. FUTURE CLIMATE SCENARIOS  (SSP / CMIP6 / CHELSA)")
            lines += [
                f"  Current (observed) score : {ssp['current_score']:.2f}",
                f"  Scenarios evaluated      : {ssp['n_scenarios']}",
                "",
                f"  {'Scenario':<35}  {'Score':>6}  {'Δ':>6}  {'Trend':<14}  Class",
                "  " + "─"*80,
            ]
            for sname, sr in ssp.get("scenarios", {}).items():
                if "error" in sr:
                    lines.append(f"  {sname:<35}: ERROR — {sr['error']}")
                else:
                    lines.append(
                        f"  {sname:<35}  {sr['composite_score']:>6.1f}  "
                        f"{sr['delta_from_current']:>+6.1f}  "
                        f"{sr['trend']:<14}  {sr['resilience_class']}")

            lines.append("")
            lines.append("  Per-scenario details:")
            for sname, sr in ssp.get("scenarios", {}).items():
                if "error" not in sr:
                    lines += [
                        f"  {sname}:",
                        f"    Score : {sr['composite_score']:.2f}  |  "
                        f"Climate: {sr['climate_score']:.2f}  |  "
                        f"Topo: {sr['topo_score']:.2f}",
                        f"    Zone  : {sr['zone_label']}",
                        f"    Class : {sr['resilience_class']}",
                        f"    Trend : {sr['trend']}",
                        "",
                    ]

        # ══════════════════════════════════════════════════════════════════════
        # 12. WEIGHT CALIBRATION  (conditional)
        cal = r.get("calibration")
        if cal and "error" not in cal:
            lines += section("12. WEIGHT CALIBRATION RESULTS")
            lines += [
                f"  Validation points  : {cal['n_valid']}  "
                f"({cal['n_suitable']} suitable / {cal['n_unsuitable']} unsuitable)",
                f"  Converged          : {'Yes' if cal.get('converged') else 'No'}",
                f"  Default accuracy   : {cal['default_accuracy']:.1%}",
                f"  Optimized accuracy : {cal['optimized_accuracy']:.1%}",
                f"  Accuracy gain      : {cal['accuracy_gain']:+.1%}",
                "",
                f"  {'Component':<25}  {'Default':>10}  {'Optimized':>10}  {'Change':>8}",
                "  " + "─"*58,
            ]
            w_def = cal.get("default_weights",   {})
            w_opt = cal.get("optimized_weights",  {})
            cal_keys   = ["threshold_zone","gmm","isolation_forest","ocsvm","mahalanobis"]
            cal_labels = ["Multi-Threshold Zone","GMM","Isolation Forest",
                          "One-Class SVM","Mahalanobis"]
            for k, lbl in zip(cal_keys, cal_labels):
                d_ = w_def.get(k, 0); o_ = w_opt.get(k, 0)
                chg = o_ - d_
                arrow = "▲" if chg > 0.001 else "▼" if chg < -0.001 else "◆"
                lines.append(
                    f"  {lbl:<25}  {d_:>10.4f}  {o_:>10.4f}  "
                    f"{chg:>+7.4f} {arrow}")

        # ══════════════════════════════════════════════════════════════════════
        # 13. SPATIAL AUTOCORRELATION CORRECTION  (thinned re-analysis)
        ta = r.get("thinned_analysis") or {}
        if ta:
            lines += section("13. SPATIAL AUTOCORRELATION CORRECTION")
            if ta.get("applied"):
                lines += [
                    f"  Status             : APPLIED",
                    f"  Reason             : Blocked CV inlier rate "
                    f"{ta['raw_mean_inlier']:.3f} < threshold",
                    f"  Points before/after: {ta['n_before']} → {ta['n_after']}  "
                    f"(retained {ta['retained_fraction']:.1%})",
                    "",
                    f"  {'Metric':<25}  {'Raw':>10}  {'Corrected':>12}  {'Delta':>8}",
                    "  " + "─"*60,
                    f"  {'Composite score':<25}  "
                    f"{ta['raw_composite']:>10.2f}  "
                    f"{ta['thinned_composite']:>12.2f}  "
                    f"{ta['delta']:>+8.2f}",
                    f"  {'Resilience class':<25}  "
                    f"{ta['raw_class'][:10]:>10}  "
                    f"{ta['thinned_class'][:12]:>12}  "
                    f"{'CHANGED' if ta.get('class_changed') else 'same':>8}",
                    f"  {'Blocked-CV inlier rate':<25}  "
                    f"{ta['raw_mean_inlier']:>10.3f}  "
                    f"{(ta.get('thinned_mean_inlier') or 0):>12.3f}",
                    "",
                    f"  Recommended score  : {ta['recommended_score']:.2f}  "
                    f"(use the corrected value in peer-reviewed reporting)",
                    "",
                    f"  Note: {ta.get('note', '')}",
                ]
            else:
                lines += [
                    f"  Status : NOT APPLIED",
                    f"  Reason : {ta.get('reason', '—')}",
                ]

        # ══════════════════════════════════════════════════════════════════════
        # 14. SENSITIVITY ANALYSIS & UNSUPERVISED WEIGHT OPTIMISATION
        sens = r.get("sensitivity") or {}
        if sens:
            lines += section("14. SENSITIVITY & UNSUPERVISED WEIGHT OPTIMISATION")
            if "error" in sens:
                lines += [f"  Error: {sens['error']}"]
            else:
                lines += [
                    f"  Grid configurations : {sens['n_grid_points']}  "
                    f"({len(sens['w_clim_grid'])} W_climate × "
                    f"{sens['n_ml_samples']} Dirichlet ML samples)",
                    f"  K-fold test points  : {sens['n_test_points']}  "
                    f"({sens['n_cv_folds']} folds)",
                    "",
                    f"  ── Target score sensitivity ──",
                    f"  Default weights     : {sens['target_score_default']:.2f}",
                    f"  Optimal weights     : {sens['target_score_optimal']:.2f}  "
                    f"(Δ {sens['target_score_delta']:+.2f})",
                    f"  Score range (min-max): "
                    f"{sens['target_score_min']:.2f} – {sens['target_score_max']:.2f}  "
                    f"(width {sens['target_score_range']:.2f})",
                    f"  Score std over grid : {sens['target_score_std']:.2f}",
                    f"  Class flip fraction : {sens['class_flip_fraction']*100:.1f}%  "
                    f"(default class {sens['default_class']})",
                    "",
                    f"  ── Unsupervised K-fold LOO objective ──",
                    f"  Default LOO score   : {sens['default_loo_score']:.2f}",
                    f"  Optimal LOO score   : {sens['optimal_loo_score']:.2f}  "
                    f"(gain {sens['loo_score_gain']:+.2f})",
                    "",
                    f"  ── Robustness ──",
                    f"  Verdict: {sens['robustness']}",
                    f"  {sens['interpretation']}",
                    "",
                    f"  {'Weight':<22}  {'Default':>10}  {'Optimal':>10}",
                    "  " + "─"*48,
                ]
                w_d = sens.get("default_weights", {})
                w_o = sens.get("optimal_weights", {})
                w_keys_labels = [
                    ("w_climate",        "Climate"),
                    ("w_topo",           "Topography"),
                    ("w_threshold_zone", "  Threshold zone"),
                    ("w_gmm",            "  GMM"),
                    ("w_iso_forest",     "  Isolation Forest"),
                    ("w_ocsvm",          "  One-Class SVM"),
                    ("w_mahalanobis",    "  Mahalanobis"),
                ]
                for k, lbl in w_keys_labels:
                    lines.append(
                        f"  {lbl:<22}  "
                        f"{w_d.get(k, 0):>10.4f}  "
                        f"{w_o.get(k, 0):>10.4f}")

        # ══════════════════════════════════════════════════════════════════════
        # FOOTER
        lines += [
            "",
            W,
            "  End of CRESTA Full Analysis Report",
            f"  Generated by Climate Resilience Analyzer v1.0.0",
            W,
        ]

        return "\n".join(lines)

    def _default_export_name(self, kind: str, ext: str) -> str:
        """
        Build a default export filename of the form
        ``<species>_cresta_<kind>_<timestamp>.<ext>``.

        Species name comes from the CSV/layer column detected during load
        and is placed FIRST so every output (JSON, CSV, TXT) sorts together
        in a directory listing.  Falls back to ``cresta_<kind>_...`` if
        no species name was found.
        """
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sp = getattr(self, "species_name_clean", "") or ""
        if sp:
            return f"{sp}_cresta_{kind}_{ts}.{ext}"
        return f"cresta_{kind}_{ts}.{ext}"

    def _export_start_dir(self) -> str:
        """Prefer the species-CSV directory if available."""
        if hasattr(self, "le_sp_path") and self.le_sp_path.text():
            return os.path.dirname(self.le_sp_path.text())
        return ""

    def _export_full_report(self):
        """Saves the full analysis report to a UTF-8 .txt file."""
        if not self.results:
            QMessageBox.warning(self, "No Results", "Run analysis first."); return

        default_name = self._default_export_name("full_report", "txt")
        start_dir    = self._export_start_dir()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Full Report",
            os.path.join(start_dir, default_name),
            "Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"

        try:
            text = self._build_full_report_text()
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            QMessageBox.information(
                self, "Report Saved",
                f"Full report saved ({len(text.splitlines())} lines):\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ── Export ────────────────────────────────────────────────────────────────
    def _export_json(self):
        if not self.results: return
        default_name = self._default_export_name("results", "json")
        start_dir    = self._export_start_dir()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON",
            os.path.join(start_dir, default_name),
            "JSON (*.json)"
        )
        if not path: return
        if not path.lower().endswith(".json"):
            path += ".json"
        def conv(o):
            if isinstance(o,(np.integer,)): return int(o)
            if isinstance(o,(np.floating,)): return float(o)
            if isinstance(o,np.ndarray): return o.tolist()
            raise TypeError(type(o))
        # Inject species metadata at top of JSON so downstream tools can
        # group by species without re-parsing filenames.
        payload = {
            "species_name": getattr(self, "species_name", None),
            **self.results,
        }
        with open(path,"w",encoding="utf-8") as f:
            json.dump(payload,f,indent=2,default=conv,ensure_ascii=False)
        QMessageBox.information(self,"Saved",f"JSON saved:\n{path}")

    def _export_csv(self):
        if not self.results: return
        default_name = self._default_export_name("per_bio", "csv")
        start_dir    = self._export_start_dir()
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV",
            os.path.join(start_dir, default_name),
            "CSV (*.csv)"
        )
        if not path: return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        r = self.results; comp=r["composite"]; p=r["percentile"]["per_bio"]
        vi=r["variable_importance"]
        lines = [
            "# Climate Resilience Analyzer v1.0.0",
            f"# Composite Score,{comp['composite_score']:.2f}",
            f"# Class,{comp['resilience_class']}",
            f"# Niche Zone,{comp['final_zone']}",
            "",
            "bio,target,p5,p25,median,p75,p95,percentile,score,mw_p,rf_importance,pdi",
        ]
        for bio in list(p.keys()):
            d    = p[bio]
            rf   = vi["rf_importance"].get(bio, 0)
            pdi_ = vi["pdi"].get(bio, 0)
            lines.append(",".join([
                bio, str(d["target_value"]),
                str(d["species_p5"]),str(d["species_p25"]),str(d["species_median"]),
                str(d["species_p75"]),str(d["species_p95"]),
                f"{d['percentile']:.2f}",f"{d['score']:.2f}",
                f"{d['mw_pvalue']:.4f}",f"{rf:.5f}",f"{pdi_:.4f}",
            ]))
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
        QMessageBox.information(self,"Saved",f"CSV saved:\n{path}")
