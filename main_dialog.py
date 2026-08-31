"""
CRESTA v2.0.0  —  Main Dialog
==============================
QGIS dialog for the conformal bioclimatic niche analyser.

The score shown here is a monotone map of a cross-conformal p-value; the
Validation tab reports the calibration actually achieved on the user's data.
See analysis_engine.py for the method and its references.
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
from qgis.core import QgsProject, QgsMapLayer

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
QMB_Yes                 = _qt_enum(QMessageBox, 'StandardButton.Yes', 'Yes')

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
    InsufficientDataError, circular_mean_deg,
    BIO_NAMES, TOPO_NAMES, ALL_VAR_NAMES, OPT_VAR_NAMES,
    ALL_BIO_COLS, ALL_TOPO_COLS, ALL_OPT_COLS,
    BIO_COLS, TOPO_COLS, ALL_COLS, CRITICAL_BIOS, CRITICAL_OPT,
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


def _read_csv_layer(layer, expected_cols, skip_null=True):
    """Reads expected columns from a QGIS vector layer into a numpy array."""
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

    rows = []
    for feat in layer.getFeatures():
        attrs = feat.attributes()
        row   = [attrs[i] for i in col_indices]
        if skip_null and any(v is None for v in row): continue
        rows.append([float(v) if v is not None else 0.0 for v in row])

    if not rows: raise ValueError("No valid rows could be extracted from layer.")
    return np.array(rows)


def _read_csv_file(path: str, expected_cols: list, skip_null=True) -> np.ndarray:
    """
    Reads expected columns from a CSV file on disk.

    Supported delimiters: comma, semicolon, tab.
    Column names are case-insensitive; spaces/hyphens/underscores flexible.
    Lines starting with '#' are treated as comments and skipped.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Auto-detect delimiter
    with open(path, "r", encoding="utf-8-sig") as f:
        # Skip comment lines, read first real line
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                sample_line = stripped
                break
        else:
            sample_line = ""
    # Try Sniffer first; fall back to frequency count
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

    rows = []
    for line in rows_raw:
        if len(line) <= max(col_indices):
            continue  # Short line
        try:
            vals = [line[i].strip() for i in col_indices]
            if skip_null and any(v in ("", "NA", "N/A", "null", "NULL", "None") for v in vals):
                continue
            rows.append([float(v) for v in vals])
        except (ValueError, IndexError):
            continue  # Skip non-convertible row

    if not rows:
        raise ValueError(f"No valid rows could be read from CSV: {path}")
    return np.array(rows)


# ── Worker ────────────────────────────────────────────────────────────────────
class AnalysisWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, species_data, target_values,
                 bio_cols, topo_cols, opt_cols,
                 target_future=None, target_cells=None, coords=None):
        super().__init__()
        self.species_data  = species_data
        self.target_values = target_values
        self.bio_cols      = bio_cols
        self.topo_cols     = topo_cols
        self.opt_cols      = opt_cols
        self.target_future = target_future
        self.target_cells  = target_cells
        self.coords        = coords

    def run(self):
        try:
            az = ClimateResilienceAnalyzer(
                self.species_data, self.target_values,
                bio_cols=self.bio_cols,
                topo_cols=self.topo_cols,
                opt_cols=self.opt_cols,
                target_future=self.target_future,
                target_cells=self.target_cells,
                coords=self.coords,
            )
            az.run_all(progress_cb=lambda pct, msg: self.progress.emit(pct, msg))
            self.finished.emit(az.results)
        except InsufficientDataError as e:
            self.error.emit("INSUFFICIENT DATA\n\n" + str(e))
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


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
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(
            "🌿 CRESTA v2.0.0  |  Conformal Bioclimatic Niche Analysis"
        )
        # Window icon
        _icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
        if os.path.exists(_icon_path):
            self.setWindowIcon(QIcon(_icon_path))
        self.setMinimumSize(900, 650); self.resize(1150, 780)
        main = QVBoxLayout(self)

        title = QLabel(
            "🌿 CRESTA  v2.0.0  ·  cross-conformal ensemble  ·  "
            "Mahalanobis (shrinkage) · GMM · Isolation Forest · One-Class SVM"
        )
        title.setAlignment(Qt_AlignCenter)
        title.setStyleSheet(
            "font-size:14px;font-weight:bold;color:#1a5276;"
            "padding:8px;background:#d4e6f1;border-radius:4px;"
        )
        main.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_input_tab(),       "📂 Data Input")
        self.tabs.addTab(self._build_results_tab(),     "📊 Summary")
        self.tabs.addTab(self._build_ml_tab(),          "🤖 ML Models")
        self.tabs.addTab(self._build_bio_tab(),         "🔬 Bio Detail")
        self.tabs.addTab(self._build_topo_tab(),        "🏔 Topography")
        self.tabs.addTab(self._build_opt_tab(),         "🌤 Optional")
        self.tabs.addTab(self._build_importance_tab(),  "📌 Variable Importance")
        self.tabs.addTab(self._build_risk_detail_tab(), "⚠️ Risk Details")
        if HAS_MPL:
            self.tabs.addTab(self._build_chart_tab(), "📈 Charts")
        self.tabs.addTab(self._build_validation_tab(), "🧪 Validation")
        self.tabs.addTab(self._build_report_tab(),   "📄 Report")
        main.addWidget(self.tabs)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color:#636e72;font-size:11px;")
        main.addWidget(self.progress_label)

        self.progress = QProgressBar(); self.progress.setVisible(False)
        main.addWidget(self.progress)

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
        _tgt_note = QLabel(
            "ℹ️  Multiple rows = the cells of the target area. Every cell is scored "
            "individually and the within-area distribution is reported; the "
            "representative aggregate uses the median for linear variables and the "
            "CIRCULAR mean for aspect (an arithmetic mean would turn 350° and 10° "
            "into 180°).")
        _tgt_note.setWordWrap(True)
        _tgt_note.setStyleSheet("font-size:10px;color:#566573;")
        tgt_main.addWidget(_tgt_note)
        self._bg_tgt.idClicked.connect(self.stack_tgt.setCurrentIndex)
        lay.addWidget(tgt_grp)

        # ══════════════════════════════════════════════════════════════════
        # 2️⃣b  FUTURE CLIMATE FOR THE TARGET SITE  (optional)
        # ══════════════════════════════════════════════════════════════════
        fut_grp = QGroupBox(
            "2️⃣b  Target Site under a FUTURE Climate Scenario  (optional)")
        fut_main = QVBoxLayout(fut_grp)
        self.chk_future = QCheckBox(
            "Enable climate-change response (ΔScore between current and future)")
        self.chk_future.setChecked(False)
        fut_main.addWidget(self.chk_future)

        fut_row = QHBoxLayout()
        fut_row.addWidget(QLabel("Future CSV:"))
        self.le_fut_path = QLineEdit()
        self.le_fut_path.setPlaceholderText(
            "Same columns, e.g. CMIP6 SSP2-4.5 2041-2060 values for the same site…")
        self.le_fut_path.setReadOnly(True)
        self.le_fut_path.setSizePolicy(QSizePolicy_Expanding, QSizePolicy_Fixed)
        fut_row.addWidget(self.le_fut_path)
        btn_fut = QPushButton("📂 Browse")
        btn_fut.clicked.connect(lambda: self._browse_csv(self.le_fut_path))
        fut_row.addWidget(btn_fut)
        btn_fut_clear = QPushButton("✖"); btn_fut_clear.setMaximumWidth(28)
        btn_fut_clear.clicked.connect(lambda: self.le_fut_path.clear())
        fut_row.addWidget(btn_fut_clear)
        fut_main.addLayout(fut_row)

        _fut_note = QLabel(
            "Without this input the analysis is SITE MATCHING under the present "
            "climate — not a climate-change projection, and it is reported as such. "
            "Run several GCM/SSP combinations and compare: one realisation carries "
            "no scenario uncertainty.")
        _fut_note.setWordWrap(True)
        _fut_note.setStyleSheet(
            "background:#fef5e7;border:1px solid #f5cba7;padding:4px 8px;"
            "border-radius:3px;font-size:10px;color:#7e5109;")
        fut_main.addWidget(_fut_note)
        lay.addWidget(fut_grp)

        # ══════════════════════════════════════════════════════════════════
        # 3️⃣  BIOCLIMATIC VARIABLE SELECTION
        # ══════════════════════════════════════════════════════════════════
        bio_grp = QGroupBox("3️⃣  Bioclimatic Variables  (select which to include in analysis)")
        bio_outer = QVBoxLayout(bio_grp)

        # Hint
        bio_hint = QLabel(
            "💡  bio4 · bio5 · bio6 · bio14 · bio15 · bio17 are highlighted as "
            "commonly limiting variables. They are NOT up-weighted in the score: "
            "the composite comes from one joint statistic over all selected "
            "variables. Note that bio1–bio19 are strongly collinear; near-duplicate "
            "variables (|r| ≥ 0.95) are pruned before modelling and listed in the "
            "report.")
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
        self.spin_min = QSpinBox(); self.spin_min.setRange(10, 99999)
        self.spin_min.setValue(70)
        self.spin_min.setToolTip(
            "Recommended minimum is 3 x the number of model variables. "
            "Below n = p + 2 the covariance is singular and the analysis is "
            "refused rather than returning a meaningless number.")
        misc_lay.addWidget(self.spin_min)
        self.lbl_minhint = QLabel("")
        self.lbl_minhint.setStyleSheet("font-size:10px;color:#566573;")
        misc_lay.addWidget(self.lbl_minhint)
        btn_auto_min = QPushButton("Auto")
        btn_auto_min.setMaximumWidth(52)
        btn_auto_min.setToolTip("Set the minimum from the current variable selection.")
        btn_auto_min.clicked.connect(self._auto_min_points)
        misc_lay.addWidget(btn_auto_min)

        self.chk_coords = QCheckBox("Use x/y columns for spatial-block CV")
        self.chk_coords.setChecked(True)
        self.chk_coords.setToolTip(
            "If the occurrence data carry x/y (or lon/lat) columns, the "
            "cross-validation folds become spatial blocks. Random folds let a "
            "record be predicted from its immediate neighbours and make the "
            "reported calibration optimistic (Roberts et al. 2017).")
        misc_lay.addWidget(self.chk_coords)
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

        self.lbl_validity = QLabel("—")
        self.lbl_validity.setWordWrap(True)
        self.lbl_validity.setStyleSheet(
            "background:#f4f6f7;border:1px solid #d5dbdb;border-radius:3px;"
            "padding:4px 8px;font-size:10px;color:#2c3e50;")
        lay.addWidget(self.lbl_validity)

        sub_row = QHBoxLayout()
        sub_row.addWidget(QLabel("Climate:"))
        self.lbl_clim = QLabel("—"); self.lbl_clim.setStyleSheet("font-weight:bold;color:#2980b9;")
        sub_row.addWidget(self.lbl_clim)
        sub_row.addWidget(QLabel("   Topography:"))
        self.lbl_topo_sc = QLabel("—"); self.lbl_topo_sc.setStyleSheet("font-weight:bold;color:#8e44ad;")
        sub_row.addWidget(self.lbl_topo_sc)
        sub_row.addStretch(); lay.addLayout(sub_row)

        self.txt_rec = QTextEdit(); self.txt_rec.setMaximumHeight(70)
        self.txt_rec.setReadOnly(True)
        self.txt_rec.setStyleSheet("background:#eafaf1;border:1px solid #27ae60;border-radius:4px;padding:6px;")
        lay.addWidget(QLabel("💡 Recommendation:")); lay.addWidget(self.txt_rec)

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
            ["Bio","Name","Target","P5","P25","Median","P75","P95","Marginal\nscore","Rank p"]
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
        self.imp_table = QTableWidget(22, 5)
        self.imp_table.setHorizontalHeaderLabels(
            ["Variable","Permutation\n(primary)","RF\n(background)",
             "Marginal\ndeviation","Combined"]
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"cresta_charts_{timestamp}.png"

        # Suggest the directory of the loaded species CSV (if any)
        start_dir = ""
        sp_path = getattr(self, "le_sp_path", None)
        if sp_path and sp_path.text():
            start_dir = os.path.dirname(sp_path.text())

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

    # ── Report ─────────────────────────────────────────────────────────────────
    def _build_validation_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.val_text = QTextEdit(); self.val_text.setReadOnly(True)
        self.val_text.setFont(QFont("Courier", 10))
        lay.addWidget(self.val_text); return w

    def _populate_validation_tab(self):
        r = self.results
        v = r.get("validation", {})
        u = r.get("uncertainty", {})
        e = r.get("extrapolation", {})
        ws = r.get("weight_sensitivity", {})
        cc = r.get("climate_change", {})
        ta = r.get("target_area", {})
        eng = r.get("engine", {})
        col = getattr(self, "_collinearity", None) or r.get("collinearity", {})

        def _n(x, fmt="{:.4f}", dash="—"):
            try:
                val = float(x)
                return dash if val != val else fmt.format(val)
            except (TypeError, ValueError):
                return dash

        L = ["MODEL VALIDATION & UNCERTAINTY", "=" * 66, "",
             "1. CALIBRATION  (does the p-value mean what it claims?)",
             "-" * 66,
             f"   Method            : {v.get('method','—')}",
             f"   Calibration set   : {v.get('n_calibration_records','—')} records",
             f"   Fold type         : {v.get('fold_type','—')}  "
             f"({v.get('n_folds_used','—')} folds)",
             f"   p-value resolution: {v.get('pvalue_resolution','—')}",
             f"   KS uniformity     : D = {_n(v.get('ks_uniformity_statistic'))}, "
             f"p = {_n(v.get('ks_uniformity_pvalue'))}",
             ""]
        rej = v.get("empirical_rejection_rate", {})
        if rej:
            L.append("   Empirical type-I error (should match the nominal level):")
            for k, val in rej.items():
                nominal = float(k.split("_")[1])
                L.append(f"      alpha = {nominal:.2f}  ->  observed "
                         f"{val * 100:5.2f} %   (nominal {nominal * 100:.0f} %)")
            L.append("")
        cp = v.get("self_class_percent", {})
        if cp:
            L += ["   Class distribution of the species' OWN occurrence records",
                  "   (design targets: A = 80 %, A+B = 90 %, E = 1 %):",
                  "      " + "   ".join(f"{k}: {cp.get(k, 0):.1f}%" for k in "ABCDE"),
                  ""]
        L += [f"   Null-model discrimination AUC : "
              f"{_n(v.get('null_model_auc'), '{:.3f}')}",
              "      (permuting each variable independently destroys the "
              "covariance",
              "       structure but preserves every marginal; AUC ~ 0.5 means the",
              "       multivariate model adds nothing over the marginals.)",
              ""]
        for line in v.get("verdict", []):
            L.append(f"   * {line}")

        L += ["", "2. UNCERTAINTY", "-" * 66,
              f"   Point score        : {u.get('point_score','—')}"]
        cb = u.get("calibration_bootstrap", {})
        rb = u.get("refit_bootstrap", {})
        L += [f"   Calibration bootstrap (n={cb.get('n',0)}) 95% CI : "
              f"{cb.get('score_ci95','—')}",
              f"   Model-refit bootstrap (n={rb.get('n',0)}) 95% CI : "
              f"{rb.get('score_ci95','—')}",
              f"   {u.get('note','')}", ""]

        L += ["3. EXTRAPOLATION  (MESS, Elith et al. 2010)", "-" * 66,
              f"   MESS               : {_n(e.get('mess'), '{:.2f}')}",
              f"   Limiting variable  : {e.get('limiting_variable','—')}",
              f"   Variables outside training range : "
              f"{e.get('n_variables_outside_range','—')}",
              f"   {e.get('interpretation','')}", ""]

        L += ["4. SENSITIVITY TO THE ENSEMBLE WEIGHTS", "-" * 66]
        for k, val in ws.get("schemes", {}).items():
            L.append(f"   {k:<24} p={val['p']:.4f}  score={val['score']:6.2f}  "
                     f"{val['zone']}")
        L += [f"   Score spread across weightings : {ws.get('score_spread','—')}",
              f"   {ws.get('note','')}", ""]

        L += ["5. COLLINEARITY / DIMENSIONALITY", "-" * 66]
        if col:
            L += [f"   Nominal dimensions   : {col.get('nominal_dimension','—')}",
                  f"   Effective dimensions : {col.get('effective_dimension','—')}"
                  "   (participation ratio of the correlation eigenvalues)",
                  f"   Condition number     : {col.get('condition_number','—')}",
                  f"   Pruned (|r| >= {col.get('corr_threshold','—')}) : "
                  f"{', '.join(col.get('dropped', [])) or 'none'}"]
            hp = col.get("high_correlation_pairs", [])
            if hp:
                L.append("   Strongest correlations:")
                for d in hp[:8]:
                    L.append(f"      {d['a']:<12} ~ {d['b']:<12} r = {d['r']:+.3f}")
            L.append(f"   {col.get('note','')}")
        L.append("")

        if ta.get("available"):
            L += ["6. TARGET AREA - CELL-WISE DISTRIBUTION", "-" * 66,
                  f"   Cells evaluated    : {ta['n_cells_evaluated']} "
                  f"(of {ta['n_cells_supplied']} supplied)",
                  f"   Aggregate score    : {ta['aggregate_score']}",
                  f"   Cell scores        : median {ta['cell_score_median']}, "
                  f"mean {ta['cell_score_mean']}, "
                  f"P5-P95 [{ta['cell_score_p5']} - {ta['cell_score_p95']}]",
                  f"   Class shares       : " +
                  "  ".join(f"{k}:{val}%" for k, val in ta["class_percent"].items()),
                  f"   Cells not rejected at 5% : "
                  f"{ta['pct_cells_inside_niche_p05']} %",
                  f"   Aggregate minus median cell : {ta['aggregate_bias']}",
                  f"   {ta['interpretation']}", ""]
        else:
            L += ["6. TARGET AREA", "-" * 66,
                  f"   {ta.get('note','—')}", ""]

        if cc.get("available"):
            L += ["7. CLIMATE-CHANGE RESPONSE", "-" * 66,
                  f"   Current : score {cc['current']['score']}  "
                  f"(p = {cc['current']['p']}, {cc['current']['zone']})",
                  f"   Future  : score {cc['future']['score']}  "
                  f"(p = {cc['future']['p']}, {cc['future']['zone']}), "
                  f"MESS = {cc['future']['mess']}",
                  f"   Delta   : {cc['delta_score']:+.2f}  ->  {cc['direction']}"
                  f"   [class {cc['class_change']}]",
                  f"   Largest niche shifts : "
                  f"{', '.join(cc.get('largest_niche_shifts', []))}",
                  f"   {cc['caveat']}", ""]
        else:
            L += ["7. CLIMATE-CHANGE RESPONSE", "-" * 66,
                  f"   {cc.get('note','—')}", ""]

        warns = eng.get("warnings", [])
        L += ["8. WARNINGS", "-" * 66]
        L += [f"   ! {x}" for x in warns] or ["   (none)"]

        self.val_text.setPlainText("\n".join(L))

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
                   expected_cols: list, skip_null: bool) -> np.ndarray:
        """
        Loads data according to selected mode.

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
            data = self._load_data(
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
    def _auto_min_points(self):
        """Set the minimum-n spin box from the current variable selection."""
        nb = len(self._get_active_bio_cols())
        nt = len(self._get_active_topo_cols())
        no = len(self._get_active_opt_cols())
        has_asp = "aspect_deg" in self._get_active_topo_cols()
        rec, hard = ClimateResilienceAnalyzer.recommended_min_records(
            nb, nt, no, has_asp)
        self.spin_min.setValue(rec)
        self.lbl_minhint.setText(
            f"recommended {rec} (3p), refused below {hard}")
        return rec, hard

    def _aggregate_target(self, rows, cols):
        """Representative value for a multi-cell target area.

        Linear variables use the MEDIAN (robust); aspect uses the CIRCULAR
        mean.  v1.0 used a plain arithmetic mean on every column, so two
        north-facing cells at 350 deg and 10 deg averaged to 180 deg (south).
        """
        if len(rows) == 1:
            return rows[0].copy()
        agg = np.median(rows, axis=0)
        if "aspect_deg" in cols:
            j = cols.index("aspect_deg")
            agg[j] = circular_mean_deg(rows[:, j])
        return agg

    def _try_load_coords(self, mode_layer, cb, le_path, skip_null, n_expected):
        """Optional x/y (or lon/lat) columns for spatial-block CV."""
        if not self.chk_coords.isChecked():
            return None
        for pair in (["x", "y"], ["lon", "lat"], ["longitude", "latitude"],
                     ["easting", "northing"]):
            try:
                c = self._load_data(mode_layer, cb, le_path, pair, skip_null)
                if c is not None and len(c) == n_expected:
                    return c
            except Exception:
                continue
        return None

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

        sp_src  = ("Layer: " + self.cb_sp.currentText() if sp_mode_layer else
                   "File: " + os.path.basename(self.le_sp_path.text()))
        tgt_src = ("Layer: " + self.cb_tgt.currentText() if tgt_mode_layer else
                   "File: " + os.path.basename(self.le_tgt_path.text()))
        self.progress_label.setText(
            f"Loading data…  {sp_src}  |  {tgt_src}  |  {len(bio_cols)} bio, "
            f"{len(topo_cols)} topo, {len(opt_cols)} optional")

        try:
            sp_data  = self._load_data(sp_mode_layer,  self.cb_sp,
                                       self.le_sp_path,  cols, skip_null)
            tgt_data = self._load_data(tgt_mode_layer, self.cb_tgt,
                                       self.le_tgt_path, cols, skip_null)
        except Exception as e:
            QMessageBox.critical(self, "Data Error", str(e)); return

        # ---- future climate (optional) ---------------------------------
        target_future = None
        if self.chk_future.isChecked():
            fpath = self.le_fut_path.text().strip()
            if not fpath:
                QMessageBox.warning(
                    self, "Future Climate",
                    "Climate-change mode is enabled but no future CSV was "
                    "selected.\nEither pick a file or untick the box.")
                return
            try:
                fut_rows = _read_csv_file(fpath, cols, skip_null)
                target_future = self._aggregate_target(fut_rows, cols)
            except Exception as e:
                QMessageBox.critical(self, "Future Climate Error", str(e)); return

        # ---- sample-size gate ------------------------------------------
        rec_min, hard_min = ClimateResilienceAnalyzer.recommended_min_records(
            len(bio_cols), len(topo_cols), len(opt_cols),
            "aspect_deg" in topo_cols)
        self.lbl_minhint.setText(f"recommended {rec_min} (3p), refused below {hard_min}")
        n_sp = len(sp_data)
        if n_sp < hard_min:
            QMessageBox.critical(
                self, "Insufficient Points",
                f"{n_sp} occurrence records for {hard_min - 2} model variables.\n\n"
                f"Below n = p + 2 the covariance matrix is singular and every "
                f"multivariate statistic is meaningless. The analysis is refused "
                f"rather than returning a number that looks real.\n\n"
                f"Either supply at least {rec_min} records (3 x the number of "
                f"variables) or select fewer variables.")
            return
        if n_sp < max(self.spin_min.value(), rec_min):
            ok = QMessageBox.question(
                self, "Small Sample",
                f"{n_sp} records for {hard_min - 2} variables.\n\n"
                f"At least {rec_min} are recommended. With this few records the "
                f"finest attainable p-value is {1.0/(n_sp+1):.3f}, so the lowest "
                f"classes may be unreachable and the calibration check will be "
                f"coarse.\n\nRun anyway?")
            if ok != QMB_Yes:
                return

        # ---- target aggregation (circular-aware) + cell-wise set --------
        target_cells = tgt_data if len(tgt_data) > 1 else None
        tgt_vals = self._aggregate_target(tgt_data, cols)

        coords = self._try_load_coords(sp_mode_layer, self.cb_sp,
                                       self.le_sp_path, skip_null, n_sp)

        self.btn_run.setEnabled(False)
        [b.setEnabled(False) for b in (self.btn_json, self.btn_csv)]
        self.progress.setVisible(True); self.progress.setValue(0)

        self.worker = AnalysisWorker(sp_data, tgt_vals,
                                     bio_cols=bio_cols,
                                     topo_cols=topo_cols,
                                     opt_cols=opt_cols,
                                     target_future=target_future,
                                     target_cells=target_cells,
                                     coords=coords)
        self.worker.progress.connect(lambda pct, m: (
            self.progress.setValue(pct), self.progress_label.setText(m)))
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_done(self, results):
        self.results = results
        self.progress.setVisible(False)
        self.progress_label.setText("✅ Analysis complete.")
        self.btn_run.setEnabled(True)
        [b.setEnabled(True) for b in (self.btn_json, self.btn_csv)]
        self._populate(); self.tabs.setCurrentIndex(1)

    def _on_err(self, msg):
        self.progress.setVisible(False); self.btn_run.setEnabled(True)
        self.progress_label.setText("❌ Analysis failed — no score produced.")
        QMessageBox.critical(self, "Analysis Error", msg)

    # ── Populate Results ──────────────────────────────────────────────────────
    def _populate(self):
        r    = self.results
        comp = r["composite"]
        sc   = comp["composite_score"]

        def _n(x, fmt="{:.2f}", dash="—"):
            try:
                v = float(x)
                return dash if v != v else fmt.format(v)
            except (TypeError, ValueError):
                return dash

        # Gauge + class
        self.gauge.set_score(sc if isinstance(sc, (int, float)) else 0)
        cls = comp["resilience_class"]
        self.lbl_class.setText(cls)
        cm = {"A": "#27ae60", "B": "#2ecc71", "C": "#f39c12",
              "D": "#e67e22", "E": "#e74c3c"}
        self.lbl_class.setStyleSheet(
            f"font-size:13px;font-weight:bold;color:{cm.get(cls[0],'#7f8c8d')}")

        ci = comp.get("score_ci95")
        ci_txt = f"   95% CI [{ci[0]:.0f} – {ci[1]:.0f}]" if ci else ""
        self.lbl_zone.setText(
            f"{comp['zone_label']}\nconformal p = {_n(comp.get('conformal_p'), '{:.4f}')}"
            f"{ci_txt}")

        self.lbl_clim.setText(
            f"{_n(comp.get('climate_score'))}  "
            f"({int(round(comp.get('climate_weight', 0) * 100))}% of variables)")
        self.lbl_topo_sc.setText(
            f"{_n(comp.get('topo_score'))}  "
            f"({int(round(comp.get('topo_weight', 0) * 100))}% of variables)")

        self.txt_rec.setPlainText(comp["recommendation"])

        # Method / validity strip
        val = r.get("validation", {})
        eng = r.get("engine", {})
        ex  = r.get("extrapolation", {})
        flags = []
        ksp = val.get("ks_uniformity_pvalue")
        if ksp is not None:
            flags.append(("✅" if ksp >= 0.05 else "⚠️") +
                         f" calibration KS p={ksp:.3f}")
        auc = val.get("null_model_auc")
        if auc is not None:
            flags.append(("✅" if auc >= 0.7 else "⚠️") + f" null AUC={auc:.2f}")
        if ex:
            flags.append(("⚠️ EXTRAPOLATION" if ex.get("is_extrapolation")
                          else "✅ interpolation") + f" MESS={_n(ex.get('mess'), '{:.1f}')}")
        flags.append(f"folds: {eng.get('fold_type','?')} × {eng.get('n_folds','?')}")
        flags.append(f"n={eng.get('n_records_used','?')}, "
                     f"p={eng.get('n_features_used','?')}")
        self.lbl_validity.setText("   |   ".join(flags))
        self.lbl_validity.setToolTip("\n".join(val.get("verdict", [])))

        top5 = comp.get("top_risk_variables", [])
        self.lbl_risk.setText(
            f"🎯 Most Influential Variables  "
            f"({len(top5)} shown — full analysis in the ⚠️ tab)")
        self._populate_risk_cards(top5)

        # Component table
        cw = comp.get("component_weights", {})
        cs = comp["component_scores"]
        rows = [
            ("Ensemble zone (read-out)", cs["threshold_zone"], cw.get("threshold_zone", "—")),
            ("GMM (Gaussian Mixture)",   cs["gmm"],              cw.get("gmm", "—")),
            ("Isolation Forest",         cs["isolation_forest"], cw.get("isolation_forest", "—")),
            ("One-Class SVM",            cs["ocsvm"],            cw.get("ocsvm", "—")),
            ("Mahalanobis (shrinkage)",  cs["mahalanobis"],      cw.get("mahalanobis", "—")),
            ("Topography (sub-score)",   cs["topo"],             cw.get("topo", "—")),
        ]
        for i, (lbl, sv, wt) in enumerate(rows):
            self.comp_table.setItem(i, 0, QTableWidgetItem(lbl))
            si = QTableWidgetItem(_n(sv, "{:.1f}"))
            si.setTextAlignment(Qt_AlignCenter)
            try:
                fv = float(sv)
                si.setBackground(QColor("#d5f5e3") if fv >= 65 else
                                 QColor("#fef9e7") if fv >= 50 else
                                 QColor("#fdecea"))
            except (TypeError, ValueError):
                pass
            self.comp_table.setItem(i, 1, si)
            wtxt = (f"{wt:.0%}" if isinstance(wt, float) else str(wt))
            self.comp_table.setItem(i, 2, QTableWidgetItem(wtxt))

        self._populate_ml_tabs()
        self._populate_bio_tab()
        self._populate_topo_tab()
        self._populate_opt_tab()
        self._populate_importance_tab()
        self._populate_risk_detail_tab()
        self._populate_validation_tab()
        if HAS_MPL: self._draw_charts()
        self._build_report()

    def _populate_ml_tabs(self):
        r = self.results
        g = r["gmm"]; iso = r["isolation_forest"]; svm = r["ocsvm"]
        km = r["kmeans"]; tz = r["threshold_zone"]; m = r["mahalanobis_stat"]

        def _f(x, fmt="{:.4f}", dash="—"):
            try:
                return fmt.format(float(x))
            except (TypeError, ValueError):
                return dash

        bic_lines = "\n".join(
            f"  fold {i+1}: {v}" for i, v in enumerate(g.get("bic_per_fold", []))) \
            or "  (not available)"

        self.gmm_text.setPlainText(
            f"Gaussian Mixture Model\n{'─'*52}\n"
            f"  Components (full fit) : {g['best_n_components']}\n"
            f"  PCA dimensions used   : {g.get('pca_dims_used','—')}\n"
            f"  Target log-likelihood : {_f(g.get('log_probability'))}\n"
            f"  Niche breadth (entropy of weights): {g.get('niche_breadth','—')}\n"
            f"  Cross-conformal p     : {_f(g['conformal_p'])}\n"
            f"  Zone                  : {g['zone']}\n"
            f"  Calibrated score      : {_f(g['score'], '{:.2f}')}\n\n"
            f"BIC per cross-validation fold:\n{bic_lines}\n\n"
            f"{g.get('note','')}"
        )

        self.iso_text.setPlainText(
            f"Isolation Forest\n{'─'*52}\n"
            f"  Anomaly score (higher = more normal) : "
            f"{_f(iso.get('anomaly_score'))}\n"
            f"  Not rejected at 5%?   : "
            f"{'✅ Yes' if iso['is_normal'] else '❌ No'}\n"
            f"  Cross-conformal p     : {_f(iso['conformal_p'])}\n"
            f"  Species median stat   : {_f(iso.get('statistic_species_median'))}\n"
            f"  Species P90 stat      : {_f(iso.get('statistic_species_p90'))}\n"
            f"  Zone                  : {iso['zone']}\n"
            f"  Calibrated score      : {_f(iso['score'], '{:.2f}')}\n\n"
            f"  contamination = {iso.get('contamination','auto')}\n"
            f"  {iso.get('note','')}"
        )

        self.svm_text.setPlainText(
            f"One-Class SVM  (RBF kernel)\n{'─'*52}\n"
            f"  Decision value (>0 = inside) : "
            f"{_f(svm.get('decision_value'))}\n"
            f"  Not rejected at 5%?   : "
            f"{'✅ Yes' if svm['is_inside'] else '❌ No'}\n"
            f"  Cross-conformal p     : {_f(svm['conformal_p'])}\n"
            f"  nu                    : {svm.get('nu','—')}\n"
            f"  Zone                  : {svm['zone']}\n"
            f"  Calibrated score      : {_f(svm['score'], '{:.2f}')}\n\n"
            f"  {svm.get('note','')}"
        )

        sil = "\n".join(f"  k={k}: {v}" for k, v in km.get("silhouette", {}).items())
        self.km_text.setPlainText(
            f"K-Means Clustering   [DIAGNOSTIC ONLY]\n{'─'*52}\n"
            f"  Optimal k             : {km['best_k']}\n"
            f"  Best silhouette       : {km['best_silhouette']}\n"
            f"  Target cluster        : {km['target_cluster']}\n"
            f"  Cluster size fraction : {km['cluster_size_frac']:.2%}\n"
            f"  Distance to centre    : {_f(km['dist_to_center'])}\n"
            f"  Distance percentile within cluster : "
            f"{km.get('dist_percentile_in_cluster','—')}\n\n"
            f"Silhouette values:\n{sil}\n\n"
            f"  {km.get('note','')}"
        )

        vote_str = "\n".join(f"    {model:<20}: {zone}"
                             for model, zone in tz["votes"].items())
        self.tz_text.setPlainText(
            f"Ensemble Zone  (single conformal statistic)\n{'─'*52}\n\n"
            f"Per-model zones (each from its own conformal p):\n{vote_str}\n\n"
            f"  Ensemble p  : {_f(tz.get('ensemble_p'))}\n"
            f"  Decision    : {tz['final_zone']}\n"
            f"  Zone label  : {tz['zone_label']}\n"
            f"  Score       : {_f(tz['score'], '{:.2f}')}\n\n"
            f"  {tz.get('note','')}\n\n"
            f"Mahalanobis (Ledoit-Wolf shrinkage, circular aspect)\n"
            f"  Covariance estimator : {m.get('covariance_estimator','—')}"
            f"   (shrinkage = {m.get('shrinkage','—')})\n"
            f"  Thresholds are EMPIRICAL out-of-fold quantiles, not chi2:\n"
            f"    D50 = {_f(m['d50_threshold'])}\n"
            f"    D90 = {_f(m['d90_threshold'])}\n"
            f"    D95 = {_f(m['d95_threshold'])}\n"
            f"    D99 = {_f(m['d99_threshold'])}\n"
            f"  Target D  = {_f(m['distance'])}\n"
            f"  Conformal p        = {_f(m['p_value_conformal'])}   (used)\n"
            f"  Parametric F ref.  = {_f(m.get('p_value_parametric_F'))}\n"
            f"  {m.get('parametric_note','')}\n"
            f"  {m['interpretation']}"
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
                  f"{d['rank_pvalue']:.4f}"]
            for col, val in enumerate(vs):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt_AlignCenter)
                if col==8:
                    s = d["score"]
                    item.setBackground(QColor("#d5f5e3") if s>=65 else
                                       QColor("#fef9e7") if s>=40 else QColor("#fdecea"))
                if col==0 and bio in CRITICAL_BIOS:
                    item.setBackground(QColor("#d6eaf8"))
                if col==9 and d["rank_pvalue"] < 0.05:
                    item.setForeground(QColor("#e74c3c"))
                self.bio_table.setItem(row, col, item)

    def _populate_topo_tab(self):
        t = self.results["topo"]
        if not t.get("topo_vars_used"):
            self.topo_text.setPlainText(
                "No topographic variables were included in this analysis.")
            return

        def _n(x, fmt="{:.2f}", dash="—"):
            try:
                v = float(x)
                return dash if v != v else fmt.format(v)
            except (TypeError, ValueError):
                return dash

        out = ["TOPOGRAPHIC COMPATIBILITY REPORT", "═" * 56, "",
               f"Calibrated topography-only score : {_n(t.get('score'))} / 100",
               f"  (conformal p = {_n(t.get('conformal_p'), '{:.4f}')})",
               f"Descriptive weighted score       : "
               f"{_n(t.get('descriptive_score'))} / 100",
               f"  weights: {t.get('descriptive_weights', {})}",
               ""]

        e = t.get("elevation") or {}
        if e:
            out += [f"── ELEVATION "
                    f"({'Within' if e['in_tolerance_range'] else 'Outside'} P5–P95)",
                    f"   Target        : {e['target_value']} m",
                    f"   Species range : [{e['species_p5']} – {e['species_p95']}] m  "
                    f"(median {e['species_median']} m)",
                    f"   Percentile    : P{e['percentile']:.0f}",
                    f"   Marginal score: {_n(e['score'], '{:.1f}')}", ""]
        sl = t.get("slope") or {}
        if sl:
            out += [f"── SLOPE "
                    f"({'Within' if sl['in_tolerance_range'] else 'Outside'} P5–P95)",
                    f"   Target        : {sl['target_value']} %",
                    f"   Species range : [{sl['species_p5']} – {sl['species_p95']}] %  "
                    f"(median {sl['species_median']} %)",
                    f"   Percentile    : P{sl['percentile']:.0f}",
                    f"   Marginal score: {_n(sl['score'], '{:.1f}')}", ""]
        a = t.get("aspect") or {}
        if a:
            out += ["── ASPECT  (circular; encoded as northness/eastness "
                    "in every multivariate stage)",
                    f"   Target aspect        : {a['target_deg']}°  "
                    f"({a['target_exposure']})",
                    f"   Species circular mean: {a.get('circular_mean_deg','—')}°  "
                    f"({a['sp_exposure']})",
                    f"   Target N/E           : {a['target_northness']:.4f} / "
                    f"{a['target_eastness']:.4f}",
                    f"   Species mean N/E     : {a['sp_mean_north']:.4f} / "
                    f"{a['sp_mean_east']:.4f}",
                    f"   Vector length        : {a['vec_length']:.4f}  "
                    f"(0 = uniform, 1 = one direction)",
                    f"   Cosine similarity    : {a['cos_similarity']:.4f}",
                    f"   Marginal score       : {_n(a['score'], '{:.1f}')}", ""]

        out += ["", t.get("note", "")]
        self.topo_text.setPlainText("\n".join(out))

    def _populate_importance_tab(self):
        vi   = self.results["variable_importance"]
        ci   = vi["combined_importance"]
        rf   = vi["rf_importance"]
        ks   = vi["ks_deviation"]
        top5 = vi["top5_variables"]

        all_cols = list(ci.keys())
        self.imp_table.setRowCount(len(all_cols))

        pi = vi.get("permutation_importance", {})
        for row, col in enumerate(all_cols):
            comb = ci[col]; rf_v = rf.get(col,0); ks_v = ks.get(col,0)
            pi_v = pi.get(col, 0)
            items = [col.upper(), f"{pi_v:.5f}", f"{rf_v:.5f}",
                     f"{ks_v:.4f}", f"{comb:.5f}"]
            for c,val in enumerate(items):
                item = QTableWidgetItem(val); item.setTextAlignment(Qt_AlignCenter)
                if c==4:
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
                       f"KS Dev  : {info.get('ks_deviation',0):.4f}\n"
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
        ax1.set_ylabel(f"PC2 ({(ev[1] if len(ev) > 1 else 0.0)*100:.1f}%)",
                       fontsize=9)
        ax1.set_title("PCA Climate Space",fontsize=10,fontweight="bold")
        ax1.legend(fontsize=8)

        # ML model scores
        _raw_ml = [
            ("Zone",      r["threshold_zone"]["score"]),
            ("GMM",       r["gmm"]["score"]),
            ("IsoForest", r["isolation_forest"]["score"]),
            ("OCSVM",     r["ocsvm"]["score"]),
            ("Mahal.",    r["mahalanobis_stat"]["score"]),
            ("Topo",      r["topo"].get("score")),
        ]
        _raw_ml = [(lb, sv) for lb, sv in _raw_ml
                   if isinstance(sv, (int, float)) and sv == sv]
        ml_labels = [lb for lb, _ in _raw_ml]
        ml_scores = [float(sv) for _, sv in _raw_ml]
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"cresta_charts_{timestamp}.png"

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
        pr   = r["percentile"]
        t    = r["topo"]
        vi   = r["variable_importance"]
        val  = r.get("validation", {})
        unc  = r.get("uncertainty", {})
        ext  = r.get("extrapolation", {})
        cc   = r.get("climate_change", {})
        ta   = r.get("target_area", {})
        eng  = r.get("engine", {})
        col  = r.get("collinearity", {})

        def _n(x, fmt="{:.2f}", dash="—"):
            try:
                v = float(x)
                return dash if v != v else fmt.format(v)
            except (TypeError, ValueError):
                return dash

        mode = ("CLIMATE-CHANGE RESPONSE" if cc.get("available")
                else "NICHE SIMILARITY / SITE MATCHING (current climate)")

        lines = [
            "=" * 74,
            "  CRESTA v2.0  —  conformal bioclimatic niche analysis",
            f"  Analysis mode : {mode}",
            "=" * 74,
            "",
            f"  COMPOSITE SCORE        : {_n(comp['composite_score'])} / 100",
            f"  Cross-conformal p      : {_n(comp.get('conformal_p'), '{:.4f}')}",
            f"  95 % CI (bootstrap)    : {comp.get('score_ci95', '—')}",
            f"  Class                  : {comp['resilience_class']}",
            f"  Niche zone             : {comp['zone_label']}",
            f"  Climate sub-score      : {_n(comp.get('climate_score'))}   "
            f"(p = {_n(comp.get('climate_conformal_p'), '{:.4f}')})",
            f"  Topography sub-score   : {_n(comp.get('topo_score'))}   "
            f"(p = {_n(comp.get('topo_conformal_p'), '{:.4f}')})",
            "",
            f"  {comp.get('interpretation','')}",
            "",
            f"  {comp.get('weighting_note','')}",
            "",
            "-" * 74,
            "  DATA & MODEL SETUP",
            "-" * 74,
            f"  Occurrence records used  : {eng.get('n_records_used','—')}"
            f"  (dropped for NaN: {eng.get('n_records_dropped_nonfinite',0)})",
            f"  Model features           : {eng.get('n_features_used','—')} "
            f"of {eng.get('n_features_before_pruning','—')} after collinearity pruning",
            f"  Effective dimensions     : {col.get('effective_dimension','—')}"
            f"   condition number {col.get('condition_number','—')}",
            f"  Pruned (|r| >= {col.get('corr_threshold','—')})   : "
            f"{', '.join(col.get('dropped', [])) or 'none'}",
            f"  Cross-validation         : {eng.get('n_folds','—')} "
            f"{eng.get('fold_type','—')} folds",
            f"  Aspect encoding          : "
            f"{m.get('aspect_encoding','n/a')}",
            f"  Covariance estimator     : {m.get('covariance_estimator','—')}"
            f"  (shrinkage = {m.get('shrinkage','—')})",
            "",
            "-" * 74,
            "  VALIDATION  (can this number be trusted?)",
            "-" * 74,
            f"  KS test of p-value uniformity : D = "
            f"{_n(val.get('ks_uniformity_statistic'), '{:.4f}')}, "
            f"p = {_n(val.get('ks_uniformity_pvalue'), '{:.4f}')}",
            f"  Empirical type-I error        : "
            f"{val.get('empirical_rejection_rate', {})}",
            f"  Own-record class distribution : "
            f"{val.get('self_class_percent', {})}   (targets A=80 %, A+B=90 %)",
            f"  Null-model discrimination AUC : "
            f"{_n(val.get('null_model_auc'), '{:.3f}')}",
            f"  MESS (extrapolation)          : {_n(ext.get('mess'))}"
            f"   limiting: {ext.get('limiting_variable','—')}",
        ]
        for v_ in val.get("verdict", []):
            lines.append(f"    * {v_}")

        lines += [
            "",
            "-" * 74,
            "  COMPONENT MODELS  (each ranked against out-of-fold values)",
            "-" * 74,
            f"  Ensemble zone     : {tz['final_zone']:<14} "
            f"p={_n(tz.get('ensemble_p'), '{:.4f}')}  score={_n(tz['score'])}",
            f"    GMM             : {g['zone']:<14} "
            f"p={_n(g['conformal_p'], '{:.4f}')}  score={_n(g['score'])}"
            f"  (k={g['best_n_components']})",
            f"    Isolation Forest: {iso['zone']:<14} "
            f"p={_n(iso['conformal_p'], '{:.4f}')}  score={_n(iso['score'])}",
            f"    One-Class SVM   : {svm['zone']:<14} "
            f"p={_n(svm['conformal_p'], '{:.4f}')}  score={_n(svm['score'])}",
            f"    Mahalanobis     : {m['zone']:<14} "
            f"p={_n(m['p_value_conformal'], '{:.4f}')}  score={_n(m['score'])}"
            f"  (D={_n(m['distance'], '{:.4f}')})",
            f"    K-Means         : diagnostic only — k={km['best_k']}, "
            f"silhouette={km['best_silhouette']}",
            f"    PCA / Kernel PCA: diagnostic only (visualisation)",
            "",
            f"  Mahalanobis thresholds are EMPIRICAL out-of-fold quantiles, not chi2:",
            f"    D50={_n(m['d50_threshold'], '{:.4f}')}  "
            f"D90={_n(m['d90_threshold'], '{:.4f}')}  "
            f"D95={_n(m['d95_threshold'], '{:.4f}')}  "
            f"D99={_n(m['d99_threshold'], '{:.4f}')}",
            f"    Parametric F reference p = "
            f"{_n(m.get('p_value_parametric_F'), '{:.4f}')} "
            f"({m.get('parametric_note','')})",
            "",
            "-" * 74,
            "  SENSITIVITY TO THE ENSEMBLE WEIGHTS",
            "-" * 74,
        ]
        ws = r.get("weight_sensitivity", {})
        for k_, v_ in ws.get("schemes", {}).items():
            lines.append(f"    {k_:<24} p={v_['p']:.4f}  score={v_['score']:6.2f}"
                         f"  {v_['zone']}")
        lines.append(f"    score spread across weightings: "
                     f"{ws.get('score_spread','—')}")

        lines += [
            "",
            "-" * 74,
            "  MOST INFLUENTIAL VARIABLES",
            "-" * 74,
            f"  Primary measure: {vi.get('primary_measure','—')}",
            f"  Background     : {vi.get('background','—')}",
            f"  {vi.get('caveat','')}",
            "",
        ]
        rd  = vi.get("risk_details", {})
        top = vi.get("top7_variables", vi.get("top5_variables", []))
        for idx, v_ in enumerate(top, 1):
            info = rd.get(v_, {})
            lines.append(
                f"  {info.get('risk_emoji','—')} #{idx:<2} {v_:<14} "
                f"perm={info.get('permutation_importance',0):.4f}  "
                f"rf={info.get('rf_importance',0):.4f}  "
                f"marg.dev={info.get('ks_deviation',0):.3f}  "
                f"P{info.get('percentile',0):.0f}  "
                f"{info.get('n_std',0):+.2f}sd  [{info.get('risk_level','—')}]")

        lines += [
            "",
            "-" * 74,
            "  PER-VARIABLE SUMMARY  (marginal view; not part of the composite)",
            "-" * 74,
            f"  Within P25–P75      : {pr['bios_in_core_range']} / {pr['n_bio_used']}",
            f"  Within P5–P95       : {pr['bios_in_tolerance_range']} / {pr['n_bio_used']}",
            f"  Outside tolerance   : {pr['bios_outside_range']} / {pr['n_bio_used']}",
            f"  {pr.get('note','')}",
        ]

        if t.get("topo_vars_used"):
            e_, s_, a_ = t.get("elevation") or {}, t.get("slope") or {}, t.get("aspect") or {}
            lines += ["", "-" * 74, "  TOPOGRAPHY", "-" * 74]
            if e_:
                lines.append(f"  Elevation : {e_['target_value']} m   "
                             f"P{e_['percentile']:.0f}   marg={_n(e_['score'],'{:.1f}')}")
            if s_:
                lines.append(f"  Slope     : {s_['target_value']} %   "
                             f"P{s_['percentile']:.0f}   marg={_n(s_['score'],'{:.1f}')}")
            if a_:
                lines.append(f"  Aspect    : {a_['target_deg']}° "
                             f"({a_['target_exposure']})  species circular mean "
                             f"{a_.get('circular_mean_deg','—')}°  "
                             f"cos={a_['cos_similarity']:.4f}   "
                             f"marg={_n(a_['score'],'{:.1f}')}")

        if ta.get("available"):
            lines += ["", "-" * 74, "  TARGET AREA — CELL-WISE DISTRIBUTION", "-" * 74,
                      f"  Cells evaluated : {ta['n_cells_evaluated']}",
                      f"  Aggregate score : {ta['aggregate_score']}   "
                      f"median cell : {ta['cell_score_median']}   "
                      f"P5–P95 : [{ta['cell_score_p5']}, {ta['cell_score_p95']}]",
                      f"  Class shares    : " +
                      "  ".join(f"{k}:{v}%" for k, v in ta["class_percent"].items()),
                      f"  {ta['interpretation']}"]

        if cc.get("available"):
            lines += ["", "-" * 74, "  CLIMATE-CHANGE RESPONSE", "-" * 74,
                      f"  Current : {cc['current']['score']}  "
                      f"(p={cc['current']['p']}, {cc['current']['zone']})",
                      f"  Future  : {cc['future']['score']}  "
                      f"(p={cc['future']['p']}, {cc['future']['zone']})  "
                      f"MESS={cc['future']['mess']}",
                      f"  Delta   : {cc['delta_score']:+.2f}  →  {cc['direction']}"
                      f"   [class {cc['class_change']}]",
                      f"  Largest niche shifts: "
                      f"{', '.join(cc.get('largest_niche_shifts', []))}",
                      f"  {cc['caveat']}"]
        else:
            lines += ["", "-" * 74, "  CLIMATE-CHANGE RESPONSE", "-" * 74,
                      f"  {cc.get('note','—')}"]

        cb_ = unc.get("calibration_bootstrap", {})
        rb_ = unc.get("refit_bootstrap", {})
        lines += [
            "", "-" * 74, "  UNCERTAINTY", "-" * 74,
            f"  Calibration bootstrap 95 % CI : {cb_.get('score_ci95','—')} "
            f"(n={cb_.get('n',0)})",
            f"  Model-refit bootstrap 95 % CI : {rb_.get('score_ci95','—')} "
            f"(n={rb_.get('n',0)})",
            f"  {unc.get('note','')}",
            "", "-" * 74, "  RECOMMENDATION", "-" * 74,
            f"  {comp['recommendation']}",
        ]

        warns = eng.get("warnings", [])
        if warns:
            lines += ["", "-" * 74, "  WARNINGS", "-" * 74]
            lines += [f"  ! {w}" for w in warns]

        lines += [
            "", "-" * 74, "  METHOD & KEY REFERENCES", "-" * 74,
            "  The composite score is a strictly monotone map of a "
            "cross-conformal",
            "  p-value (Vovk 2015). Under the hypothesis that the target is "
            "exchangeable",
            "  with the occurrence records, P(score >= 80) = 0.80 and "
            "P(score < 35) = 0.01,",
            "  so the class thresholds have an exact frequentist meaning.",
            "",
            "  Elith J, Kearney M, Phillips S (2010) Methods Ecol Evol 1:330-342.",
            "  Ledoit O, Wolf M (2004) J Multivar Anal 88:365-411.",
            "  Raes N, ter Steege H (2007) Ecography 30:727-736.",
            "  Roberts DR et al. (2017) Ecography 40:913-929.",
            "  Soberon J, Nakamura M (2009) PNAS 106:19644-19650.",
            "  Vovk V (2015) Ann Math Artif Intell 74:9-28.",
            "",
            "  Caveat: occurrence records describe the REALISED niche, which is "
            "truncated",
            "  by dispersal, biotic interactions and sampling bias. This tool "
            "does not",
            "  estimate the fundamental niche.",
            "=" * 74,
        ]

        self.report_text.setPlainText("\n".join(lines))

    def _copy_report(self):
        from qgis.PyQt.QtWidgets import QApplication
        QApplication.clipboard().setText(self.report_text.toPlainText())

    # ── Export ────────────────────────────────────────────────────────────────
    def _export_json(self):
        if not self.results: return
        path,_ = QFileDialog.getSaveFileName(self,"Save JSON","","JSON (*.json)")
        if not path: return
        def conv(o):
            if isinstance(o,(np.integer,)): return int(o)
            if isinstance(o,(np.floating,)): return float(o)
            if isinstance(o,np.ndarray): return o.tolist()
            raise TypeError(type(o))
        with open(path,"w",encoding="utf-8") as f:
            json.dump(self.results,f,indent=2,default=conv,ensure_ascii=False)
        QMessageBox.information(self,"Saved",f"JSON saved:\n{path}")

    def _export_csv(self):
        if not self.results: return
        path,_ = QFileDialog.getSaveFileName(self,"Save CSV","","CSV (*.csv)")
        if not path: return
        r = self.results; comp=r["composite"]; p=r["percentile"]["per_bio"]
        vi=r["variable_importance"]
        lines = [
            "# CRESTA v2.0 — conformal niche-similarity analysis",
            f"# Composite Score,{comp['composite_score']:.2f}",
            f"# Class,{comp['resilience_class']}",
            f"# Niche Zone,{comp['final_zone']}",
            "",
            "bio,target,p5,p25,median,p75,p95,percentile,marginal_score,rank_p,rf_importance,ks_deviation",
        ]
        for bio in list(p.keys()):
            d  = p[bio]
            rf = vi["rf_importance"].get(bio,0)
            ks = vi["ks_deviation"].get(bio,0)
            lines.append(",".join([
                bio, str(d["target_value"]),
                str(d["species_p5"]),str(d["species_p25"]),str(d["species_median"]),
                str(d["species_p75"]),str(d["species_p95"]),
                f"{d['percentile']:.2f}",f"{d['score']:.2f}",
                f"{d['rank_pvalue']:.4f}",f"{rf:.5f}",f"{ks:.4f}",
            ]))
        with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
        QMessageBox.information(self,"Saved",f"CSV saved:\n{path}")
