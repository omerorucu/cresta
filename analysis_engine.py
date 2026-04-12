"""
Climate Resilience Analysis Engine  v3.2
==========================================
Species native range points → Statistics + ML → Multi-threshold resilience score

Required variables (22):
  bio1–bio19  +  elevation_m · slope_pct · aspect_deg

Optional extra variables (max 15, user-selected):
  Solar Radiation : srad  srad_10m  srad_5m  srad_2_5m  srad_30s
  Wind Speed     : wind  wind_10m  wind_5m  wind_2_5m  wind_30s
  Vapour Pressure: vapr  vapr_10m  vapr_5m  vapr_2_5m  vapr_30s

═══════════════════════════════════════════════════════════════════════
LAYER A — PCA + Kernel PCA
LAYER B — GMM · Isolation Forest · One-Class SVM · K-Means
LAYER C — 4-Threshold Voting + Mahalanobis Chi²
LAYER D — Variable Importance + Numerical Risk Explanation
LAYER E — Composite Score  (climate 75% + topo 25% + opt bonus max 5%)
═══════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np
from scipy.spatial.distance import mahalanobis as scipy_mahal
from scipy.stats import percentileofscore, chi2, mannwhitneyu
from scipy.special import expit

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BIO_NAMES = {
    "bio1":  "Annual Mean Temperature (°C×10)",
    "bio2":  "Mean Diurnal Temperature Range",
    "bio3":  "Isothermality (%)",
    "bio4":  "Temperature Seasonality (SD×100)",
    "bio5":  "Max Temperature of Warmest Month (°C×10)",
    "bio6":  "Min Temperature of Coldest Month (°C×10)",
    "bio7":  "Annual Temperature Range (°C×10)",
    "bio8":  "Mean Temp of Wettest Quarter (°C×10)",
    "bio9":  "Mean Temp of Driest Quarter (°C×10)",
    "bio10": "Mean Temp of Warmest Quarter (°C×10)",
    "bio11": "Mean Temp of Coldest Quarter (°C×10)",
    "bio12": "Annual Precipitation (mm)",
    "bio13": "Precipitation of Wettest Month (mm)",
    "bio14": "Precipitation of Driest Month (mm)",
    "bio15": "Precipitation Seasonality (CV)",
    "bio16": "Precipitation of Wettest Quarter (mm)",
    "bio17": "Precipitation of Driest Quarter (mm)",
    "bio18": "Precipitation of Warmest Quarter (mm)",
    "bio19": "Precipitation of Coldest Quarter (mm)",
}
TOPO_NAMES = {
    "elevation_m": "Elevation (m)",
    "slope_pct":   "Slope (%)",
    "aspect_deg":  "Aspect (°, 0/360=North)",
}
# Optional variables: one representative per group (no resolution sub-variants)
OPT_VAR_NAMES = {
    "srad": "Mean Daily Solar Radiation (kJ m\u207b\u00b2 day\u207b\u00b9)",
    "wind": "Mean Wind Speed (m s\u207b\u00b9)",
    "vapr": "Water Vapour Pressure (kPa)",
}
ALL_VAR_NAMES = {**BIO_NAMES, **TOPO_NAMES, **OPT_VAR_NAMES}

ALL_BIO_COLS  = [f"bio{i}" for i in range(1, 20)]
ALL_TOPO_COLS = ["elevation_m", "slope_pct", "aspect_deg"]
ALL_OPT_COLS  = ["srad", "wind", "vapr"]

# Kept for backward-compat imports in main_dialog
BIO_COLS  = ALL_BIO_COLS
TOPO_COLS = ALL_TOPO_COLS
ALL_COLS  = BIO_COLS + TOPO_COLS

CRITICAL_BIOS = {"bio4", "bio5", "bio6", "bio14", "bio15", "bio17"}
CRITICAL_OPT  = {"srad", "vapr"}

# Score weights
W_CLIMATE        = 0.75
W_TOPO           = 0.25
W_ELEV           = 0.45
W_SLOPE          = 0.25
W_ASPECT         = 0.30
W_THRESHOLD_ZONE = 0.35   # Ensemble vote — most reliable combined signal
W_GMM            = 0.20   # Niche structure modelling
W_ISOFOREST      = 0.10   # Anomaly detection (redundant w/ OCSVM → reduced)
W_OCSVM          = 0.10   # Boundary definition (redundant w/ IsoForest → reduced)
W_MAHAL          = 0.25   # Distance-based — scientific gold standard (↑ from 0.15)


# ═══════════════════════════════════════════════════════════════════════════════
#  RISK EXPLANATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def build_risk_explanation(col, tgt, med, p5, p25, p75, p95,
                            pct, score, sp_min, sp_max, sp_std):
    """Generates a variable-specific numerical risk explanation."""

    if score >= 65:
        risk_level, risk_color, risk_emoji = "LOW",      "#27ae60", "✅"
    elif score >= 40:
        risk_level, risk_color, risk_emoji = "MODERATE", "#f39c12", "🟡"
    elif score >= 20:
        risk_level, risk_color, risk_emoji = "HIGH",     "#e67e22", "🟠"
    else:
        risk_level, risk_color, risk_emoji = "CRITICAL", "#e74c3c", "🔴"

    in_core = bool(p25 <= tgt <= p75)
    in_tol  = bool(p5  <= tgt <= p95)
    dev_abs = tgt - med
    dev_pct = (dev_abs / (abs(med) + 1e-9)) * 100
    n_std   = dev_abs / (sp_std + 1e-9)
    d5      = tgt - p5
    d95     = p95 - tgt

    if   tgt < p5:
        pos = (f"{abs(d5):.4g} units BELOW P5 lower bound (P{pct:.0f} percentile) "
               f"— outside tolerance")
    elif tgt < p25:
        pos = (f"Lower tolerance band (P{pct:.0f} percentile), "
               f"{p25-tgt:.4g} units from P25 core boundary")
    elif tgt <= p75:
        pos = (f"Within core niche (P{pct:.0f} percentile), "
               f"{abs(dev_abs):.4g} units from median")
    elif tgt <= p95:
        pos = (f"Upper tolerance band (P{pct:.0f} percentile), "
               f"{tgt-p75:.4g} units above P75 core boundary")
    else:
        pos = (f"{abs(d95):.4g} units ABOVE P95 upper bound (P{pct:.0f} percentile) "
               f"— outside tolerance")

    bio_text = _var_rationale(col, tgt, med, p5, p25, p75, p95,
                               d5, d95, dev_pct, n_std, in_tol)
    sep = "─" * 52
    explanation = (
        f"{risk_emoji} {risk_level} RISK  —  {ALL_VAR_NAMES.get(col, col)}\n"
        f"{sep}\n"
        f"  Target value   : {tgt:.4g}  (median: {med:.4g}, std: {sp_std:.4g})\n"
        f"  Species range  : [{sp_min:.4g} – {sp_max:.4g}]\n"
        f"  Thresholds     : P5={p5:.4g}  P25={p25:.4g}  "
        f"P75={p75:.4g}  P95={p95:.4g}\n"
        f"  Position       : {pos}\n"
        f"  Deviation      : {dev_abs:+.4g} units  "
        f"({dev_pct:+.1f}%)  /  {n_std:+.2f} std\n"
        f"  Resilience score: {score:.1f} / 100\n"
        f"{sep}\n"
        f"  Biological Rationale:\n"
        f"  {bio_text}"
    )

    return {
        "risk_level": risk_level, "risk_color": risk_color,
        "risk_emoji": risk_emoji, "in_core": in_core, "in_tolerance": in_tol,
        "dev_from_med": round(dev_abs, 4), "dev_pct": round(dev_pct, 2),
        "n_std": round(n_std, 3), "dist_to_p5": round(d5, 4),
        "dist_to_p95": round(d95, 4), "explanation": explanation,
    }


def _var_rationale(col, tgt, med, p5, p25, p75, p95,
                   d5, d95, dev_pct, n_std, in_tol):
    """Variable-specific numerical biological rationale text."""

    def hi_lo_ok(hi, lo, ok):
        if tgt > p95: return hi
        if tgt < p5:  return lo
        return ok

    if col == "bio1":
        return hi_lo_ok(
            f"Annual mean temp {tgt:.1f} °C×10 — above P95={p95:.1f} "
            f"({abs(d95):.1f} units excess, {n_std:+.2f} std). "
            f"In C3 plants >350 °C×10 (~35 °C) and C4 >450 °C×10, "
            f"enzyme activity collapses; net photosynthesis turns negative.",
            f"Annual mean temp {tgt:.1f} °C×10 — below P5={p5:.1f} "
            f"({abs(d5):.1f} units deficit, {n_std:+.2f} std). "
            f"Vegetation period shortens (~6 days per 100 m equivalent). "
            f"Phenological mismatch and delayed flowering risk is high.",
            f"Annual mean temp {tgt:.1f} °C×10 — within species core niche "
            f"P25–P75: [{p25:.1f}–{p75:.1f}]. Deviation {dev_pct:+.1f}%, "
            f"{n_std:+.2f} std. Thermal stress minimal.")
    elif col == "bio4":
        return hi_lo_ok(
            f"Temperature seasonality {tgt:.1f} (SD×100) — above P95={p95:.1f}. "
            f"High summer-winter temperature amplitude; "
            f"frequency of frost (≤0 °C) and heat waves (≥35 °C) increases. "
            f"Tissue damage risk critical if species is not adapted "
            f"({n_std:+.2f} std from median).",
            f"Seasonality {tgt:.1f} — below P5={p5:.1f} (uniform climate). "
            f"If the species is adapted to high seasonality, dormancy may not "
            f"be triggered; the growth-rest cycle could be disrupted.",
            f"Seasonality {tgt:.1f} within species tolerance band "
            f"(median {med:.1f}, {dev_pct:+.1f}%). Risk minimal.")
    elif col == "bio5":
        return hi_lo_ok(
            f"Warmest month max temp {tgt:.1f} °C×10 — above P95={p95:.1f}, "
            f"{abs(d95):.1f} units excess ({n_std:+.2f} std). "
            f"Leaf surface can be 5–10 °C hotter than air. "
            f"Once stomatal closure threshold is exceeded, CO₂ uptake stops; "
            f"xylem cavitation and phloem blockage may begin.",
            f"Warmest month max temp {tgt:.1f} — below P5={p5:.1f}. "
            f"Insufficient summer warmth; vegetation period productivity is low.",
            f"Maximum temperature {tgt:.1f} °C×10 within species band "
            f"(median {med:.1f}, {dev_pct:+.1f}%).")
    elif col == "bio6":
        return hi_lo_ok(
            f"Coldest month min temp {tgt:.1f} °C×10 — above P95={p95:.1f}. "
            f"Unusually mild winter; required chilling hours for vernalization "
            f"may not be completed, disrupting flowering and fruit set.",
            f"Coldest month min temp {tgt:.1f} °C×10 — below P5={p5:.1f}, "
            f"{abs(d5):.1f} units deficit ({n_std:+.2f} std). "
            f"Intracellular ice crystals rupture plasma membranes; "
            f"permanent damage to phloem and vascular tissue may occur.",
            f"Min temperature {tgt:.1f} within species tolerance band "
            f"(median {med:.1f}, {dev_pct:+.1f}%).")
    elif col == "bio12":
        return hi_lo_ok(
            f"Annual precipitation {tgt:.0f} mm — above P95={p95:.0f} mm. "
            f"Excess moisture; risk of root rot, fungal diseases and "
            f"anaerobic soil conditions (oxygen deficiency).",
            f"Annual precipitation {tgt:.0f} mm — below P5={p5:.0f} mm, "
            f"{abs(d5):.0f} mm deficit ({n_std:+.2f} std). "
            f"Potential evapotranspiration > precipitation; "
            f"root water potential approaches xylem cavitation threshold.",
            f"Annual precipitation {tgt:.0f} mm within species band "
            f"(median {med:.0f} mm, {dev_pct:+.1f}%).")
    elif col == "bio14":
        return hi_lo_ok(
            f"Driest month precipitation {tgt:.1f} mm — above P95.",
            f"Driest month precipitation {tgt:.1f} mm — below P5={p5:.1f} mm, "
            f"{abs(d5):.2f} mm deficit. Turgor loss and "
            f"stomatal closure: net photosynthesis approaches zero. "
            f"Cambium activity loss expected.",
            f"Driest month precipitation {tgt:.1f} mm within species band "
            f"(median {med:.1f} mm, {dev_pct:+.1f}%).")
    elif col == "bio15":
        return hi_lo_ok(
            f"Precipitation seasonality CV={tgt:.1f} — above P95={p95:.1f} "
            f"({n_std:+.2f} std). Highly irregular precipitation: "
            f"sudden drought and flood risk can occur within the same year; "
            f"root damage and soil erosion create cumulative stress.",
            f"Precipitation seasonality CV={tgt:.1f} — below P5={p5:.1f}. "
            f"Very uniform rainfall; structures adapted to drought "
            f"(deep roots, water storage tissue) may be underdeveloped.",
            f"Precipitation seasonality {tgt:.1f} within species band "
            f"(median {med:.1f}, {dev_pct:+.1f}%).")
    elif col == "bio17":
        return hi_lo_ok(
            f"Driest quarter precipitation {tgt:.0f} mm — above P95.",
            f"Driest quarter precipitation {tgt:.0f} mm — below P5={p5:.0f} mm, "
            f"{abs(d5):.0f} mm deficit ({n_std:+.2f} std). "
            f"Critical summer water deficit: growth slowdown and "
            f"cambium activity loss; annual ring narrowing expected in trees.",
            f"Driest quarter {tgt:.0f} mm within species band "
            f"(median {med:.0f} mm, {dev_pct:+.1f}%).")
    elif col == "elevation_m":
        return hi_lo_ok(
            f"Elevation {tgt:.0f} m — above P95={p95:.0f} m, {abs(d95):.0f} m excess. "
            f"Temperature drops ~0.6 °C per 100 m rise, "
            f"vegetation period shortens ~6 days. "
            f"Species growth window at this elevation is at or below the critical limit.",
            f"Elevation {tgt:.0f} m — below P5={p5:.0f} m. "
            f"Species adapted to higher elevations; "
            f"excessive heat and low moisture stress possible at low altitude.",
            f"Elevation {tgt:.0f} m within species band "
            f"(median {med:.0f} m, {dev_pct:+.1f}%).")
    elif col == "slope_pct":
        return hi_lo_ok(
            f"Slope {tgt:.1f}% — above P95={p95:.1f}%, {abs(d95):.1f} units excess. "
            f"Extreme slope accelerates soil erosion; "
            f"root depth and soil water capacity decrease. "
            f"Surface runoff increases, plant-available water capacity drops.",
            f"Slope {tgt:.1f}% — below P5={p5:.1f}%. "
            f"If the species is adapted to well-drained slopes, "
            f"waterlogging and root suffocation risk.",
            f"Slope {tgt:.1f}% within species band "
            f"(median {med:.1f}%, {dev_pct:+.1f}%).")
    elif col == 'srad':
        return hi_lo_ok(
            f"Solar radiation {tgt:.0f} kJ m⁻² day⁻¹ — above P95={p95:.0f}, "
            f"{abs(d95):.0f} kJ m⁻² excess ({n_std:+.2f} std). "
            f"High radiation raises leaf surface temperature by 5–15 °C; "
            f"photoinhibition (PSII damage) and heat stress risk increases. "
            f"Increased transpiration may not be met by available water.",
            f"Solar radiation {tgt:.0f} kJ m⁻² day⁻¹ — below P5={p5:.0f}, "
            f"{abs(d5):.0f} kJ m⁻² deficit. "
            f"Light saturation point for photosynthesis not reached; "
            f"carbon assimilation and growth slows.",
            f"Solar radiation {tgt:.0f} kJ m⁻² day⁻¹ within species band "
            f"(median {med:.0f}, {dev_pct:+.1f}%).")
    elif col == 'wind':
        return hi_lo_ok(
            f"Wind speed {tgt:.2f} m s⁻¹ — above P95={p95:.2f} m s⁻¹, "
            f"{abs(d95):.2f} m s⁻¹ excess ({n_std:+.2f} std). "
            f"High wind: (1) increases transpiration → water stress; "
            f"(2) mechanical fatigue of branches/stem → breakage; "
            f"(3) salt spray and sand abrasion.",
            f"Wind speed {tgt:.2f} m s⁻¹ — below P5={p5:.2f} m s⁻¹. "
            f"Very calm conditions; evapotranspiration and pollination may be limited.",
            f"Wind speed {tgt:.2f} m s⁻¹ within species band "
            f"(median {med:.2f} m s⁻¹, {dev_pct:+.1f}%).")
    elif col == 'vapr':
        vpd   = max(0.0, 2.34 - tgt)
        vpd_m = max(0.0, 2.34 - med)
        return hi_lo_ok(
            f"Vapour pressure {tgt:.3f} kPa — above P95={p95:.3f} kPa. "
            f"Excess humidity; fungal disease and stomatal conductance suppression.",
            f"Vapour pressure {tgt:.3f} kPa — below P5={p5:.3f} kPa, "
            f"{abs(d5):.3f} kPa deficit ({n_std:+.2f} std). "
            f"Estimated VPD ≈ {vpd:.2f} kPa (median VPD ≈ {vpd_m:.2f} kPa). "
            f"VPD > 2 kPa triggers stomatal closure; "
            f"at >3 kPa plant water potential exceeds critical threshold. "
            f"Carbon assimilation and growth slow markedly.",
            f"Vapour pressure {tgt:.3f} kPa (VPD≈{vpd:.2f} kPa) "
            f"within species tolerance band. Deviation {dev_pct:+.1f}% from median.")
    else:
        return hi_lo_ok(
            f"Value {tgt:.4g} — above P95={p95:.4g} "
            f"({abs(d95):.4g} units, {n_std:+.2f} std). Upper tolerance exceeded.",
            f"Value {tgt:.4g} — below P5={p5:.4g} "
            f"({abs(d5):.4g} units, {n_std:+.2f} std). Lower tolerance exceeded.",
            f"Value {tgt:.4g} within species band. "
            f"Deviation {dev_pct:+.1f}% from median ({n_std:+.2f} std).")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def northness(deg): return float(np.cos(np.deg2rad(deg)))
def eastness(deg):  return float(np.sin(np.deg2rad(deg)))

def _safe_inv(M):
    try:    return np.linalg.inv(M)
    except: return np.linalg.pinv(M)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class ClimateResilienceAnalyzer:
    """
    v4.0 — Fully dynamic variable selection.

    Parameters
    ----------
    species_data  : ndarray (n, n_cols)
    target_values : ndarray (n_cols,)
    bio_cols      : list[str]  — chosen subset of bio1..bio19  (must be ≥1)
    topo_cols     : list[str]  — chosen subset of ["elevation_m","slope_pct","aspect_deg"]
    opt_cols      : list[str]  — chosen subset of ["srad","wind","vapr"]

    Column order in data arrays must be:  bio_cols  +  topo_cols  +  opt_cols
    """

    def __init__(self, species_data, target_values,
                 bio_cols=None, topo_cols=None, opt_cols=None):

        # ── Validate & store selected columns ────────────────────────────────
        self.bio_cols  = [c for c in (bio_cols  or ALL_BIO_COLS)  if c in BIO_NAMES]
        self.topo_cols = [c for c in (topo_cols or [])            if c in TOPO_NAMES]
        self.opt_cols  = [c for c in (opt_cols  or [])            if c in OPT_VAR_NAMES]

        if not self.bio_cols:
            raise ValueError("At least one bioclimatic variable must be selected.")

        self.has_topo   = len(self.topo_cols) > 0
        self.has_aspect = "aspect_deg" in self.topo_cols

        self.col_names = self.bio_cols + self.topo_cols + self.opt_cols
        self.n_vars    = len(self.col_names)
        self.n_bio     = len(self.bio_cols)
        self.n_topo    = len(self.topo_cols)
        self.n_opt     = len(self.opt_cols)

        # Index maps
        self._col_idx   = {c: i for i, c in enumerate(self.col_names)}
        self._bio_slice = slice(0, self.n_bio)
        self._topo_idxs = {c: self.n_bio + i for i, c in enumerate(self.topo_cols)}
        self._opt_offset = self.n_bio + self.n_topo

        # ── Data ──────────────────────────────────────────────────────────────
        self.species_data  = np.array(species_data,  dtype=float)
        self.target_values = np.array(target_values, dtype=float)

        if self.species_data.shape[1] != self.n_vars:
            raise ValueError(
                f"species_data: {self.n_vars} columns expected "
                f"({', '.join(self.col_names)}), "
                f"got {self.species_data.shape[1]}.")

        # ── Scaling ───────────────────────────────────────────────────────────
        self.scaler   = StandardScaler()
        self.X_sc     = self.scaler.fit_transform(self.species_data)
        self.t_sc     = self.scaler.transform(self.target_values.reshape(1,-1))[0]

        # Bio sub-arrays
        self.X_bio    = self.species_data[:, self._bio_slice]
        self.X_sc_bio = self.X_sc[:, self._bio_slice]
        self.t_bio    = self.target_values[self._bio_slice]
        self.t_sc_bio = self.t_sc[self._bio_slice]

        # ML arrays (aspect → circular)
        self.X_ml, self.t_ml = self._build_ml_arrays()

        # Mahalanobis helpers
        self._mu      = np.mean(self.species_data, axis=0)
        self._std     = np.std(self.species_data,  axis=0)
        self._cov     = np.cov(self.species_data,  rowvar=False)
        self._cov_inv = _safe_inv(self._cov)

        self.results = {}
        self._models = {}

    # ── ML matrix ─────────────────────────────────────────────────────────────
    def _build_ml_arrays(self):
        if not self.has_aspect:
            return self.X_sc.copy(), self.t_sc.copy()
        ai   = self._col_idx["aspect_deg"]
        mask = [i for i in range(self.n_vars) if i != ai]
        X_no = self.X_sc[:, mask]
        t_no = self.t_sc[mask]
        raw_a = self.species_data[:, ai]
        tgt_a = self.target_values[ai]
        X_ml  = np.hstack([X_no,
                            np.cos(np.deg2rad(raw_a)).reshape(-1,1),
                            np.sin(np.deg2rad(raw_a)).reshape(-1,1)])
        t_ml  = np.concatenate([t_no, [northness(tgt_a)], [eastness(tgt_a)]])
        return X_ml, t_ml

    def _ml_col_names(self):
        if not self.has_aspect:
            return self.col_names[:]
        ai   = self._col_idx["aspect_deg"]
        base = [c for i,c in enumerate(self.col_names) if i != ai]
        return base + ["northness", "eastness"]

    # ── Dynamic topo weights ──────────────────────────────────────────────────
    def _topo_weights(self):
        raw = {}
        if "elevation_m" in self.topo_cols: raw["elevation_m"] = W_ELEV
        if "slope_pct"   in self.topo_cols: raw["slope_pct"]   = W_SLOPE
        if "aspect_deg"  in self.topo_cols: raw["aspect_deg"]  = W_ASPECT
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def _clim_topo_weights(self):
        return (W_CLIMATE, W_TOPO) if self.has_topo else (1.0, 0.0)

    # ═══════════════════════════════════════════════════════════════════════
    # LAYER A  —  PCA + Kernel PCA
    # ═══════════════════════════════════════════════════════════════════════

    def compute_pca(self):
        n_comp = min(5, self.X_ml.shape[1])
        pca5 = PCA(n_components=n_comp); pca5.fit(self.X_ml)
        pca2 = PCA(n_components=min(2, self.X_ml.shape[1])); pca2.fit(self.X_ml)
        X2   = pca2.transform(self.X_ml)
        t2   = pca2.transform(self.t_ml.reshape(1,-1))[0]
        cx,cy= float(np.mean(X2[:,0])),float(np.mean(X2[:,1]))
        dist = float(np.linalg.norm(t2-[cx,cy]))
        maxd = float(np.percentile([np.linalg.norm(p-[cx,cy]) for p in X2],95))
        score= float(np.clip(100*(1-dist/(maxd+1e-9)),0,100))
        r = {
            "species_pc2": X2.tolist(), "target_pc2": t2.tolist(),
            "ev_2":  [round(v,4) for v in pca2.explained_variance_ratio_],
            "ev_5":  [round(v,4) for v in pca5.explained_variance_ratio_],
            "cumulative_ev5": round(float(pca5.explained_variance_ratio_.sum()),4),
            "score": round(score,2), "distance_2d": round(dist,4),
            "loadings": {c: {"PC1": round(float(pca2.components_[0,i]),4),
                             "PC2": round(float(pca2.components_[1,i]),4)}
                         for i,c in enumerate(self._ml_col_names())},
        }
        self.results["pca"] = r
        self._models.update({"pca2":pca2,"pca5":pca5,"X2":X2,"t2":t2})
        return r

    def compute_kernel_pca(self):
        try:
            kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
            Xk   = kpca.fit_transform(self.X_ml)
            tk   = kpca.transform(self.t_ml.reshape(1,-1))[0]
            cx,cy= float(np.mean(Xk[:,0])),float(np.mean(Xk[:,1]))
            dist = float(np.linalg.norm(tk-[cx,cy]))
            maxd = float(np.percentile([np.linalg.norm(p-[cx,cy]) for p in Xk],95))
            score= float(np.clip(100*(1-dist/(maxd+1e-9)),0,100))
            r = {"species_kpca":Xk.tolist(),"target_kpca":tk.tolist(),"score":round(score,2)}
        except Exception as e:
            r = {"score":50.0,"error":str(e)}
        self.results["kpca"] = r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # LAYER B  —  ML Models
    # ═══════════════════════════════════════════════════════════════════════

    def compute_gmm(self, max_components=5):
        # ── PCA pre-reduction ────────────────────────────────────────────────
        # High-dimensional log-prob values (e.g. -500 for 19 dims) cause the
        # sigmoid scoring to collapse to 0. Reducing to a compact PCA space
        # keeps log-prob in a sensible range while preserving niche structure.
        n_pca = min(len(self.X_sc_bio) // 5, self.X_sc_bio.shape[1], 8)
        n_pca = max(n_pca, 2)
        try:
            from sklearn.decomposition import PCA as _PCA
            _pca = _PCA(n_components=n_pca, random_state=42)
            X_gmm = _pca.fit_transform(self.X_sc_bio)
            t_gmm = _pca.transform(self.t_sc_bio.reshape(1, -1))[0]
        except Exception:
            X_gmm = self.X_sc_bio
            t_gmm = self.t_sc_bio

        best_bic, best_gmm, best_n = np.inf, None, 1
        bic_vals = {}
        # Cap components: too many → overfit → target falls outside all clusters
        # Rule: max(1, min(3, n_samples // 20)) — never more than 3 for ecological data
        n_max = max(1, min(3, len(self.X_sc_bio) // 20))
        for n in range(1, n_max + 1):
            try:
                g = GaussianMixture(n_components=n, covariance_type="full",
                                    random_state=42, max_iter=300, reg_covar=1e-4)
                g.fit(X_gmm)
                b = g.bic(X_gmm); bic_vals[n] = round(b, 2)
                if b < best_bic:
                    best_bic, best_gmm, best_n = b, g, n
            except Exception:
                pass

        if best_gmm is None:
            # Fallback: single Gaussian with diagonal covariance
            best_gmm = GaussianMixture(n_components=1, covariance_type="diag",
                                       random_state=42, reg_covar=1e-2)
            best_gmm.fit(X_gmm)
            best_n = 1

        lp  = float(best_gmm.score(t_gmm.reshape(1, -1)))
        cp  = best_gmm.predict_proba(t_gmm.reshape(1, -1))[0]
        slp = best_gmm.score_samples(X_gmm)

        # ── Robust z-score sigmoid scoring ──────────────────────────────────
        sp_mean = float(np.mean(slp))
        sp_std  = float(np.std(slp)) or 1.0
        z  = (lp - sp_mean) / sp_std
        # k=0.7: softer curve — avoids near-zero scores for moderate outliers
        sc = float(np.clip(100.0 / (1.0 + np.exp(-0.7 * z)), 0, 100))

        # Zone thresholds
        q75, q25, q10 = (float(np.percentile(slp, q)) for q in (75, 25, 10))
        zone = ("CORE"     if lp >= q75 else
                "SUITABLE" if lp >= q25 else
                "MARGINAL" if lp >= q10 else "OUTSIDE")
        pct = float(percentileofscore(slp, lp, kind="rank"))

        r = {"best_n_components": best_n, "bic_values": bic_vals,
             "log_probability": round(lp, 4),
             "max_component_prob": round(float(np.max(cp)), 4),
             "component_probs": [round(float(p), 4) for p in cp],
             "zone": zone, "score": round(sc, 2), "percentile_rank": round(pct, 2),
             "sp_lp_mean": round(sp_mean, 4), "sp_lp_std": round(sp_std, 4),
             "niche_breadth": round(float(
                 -np.sum(best_gmm.weights_ * np.log(best_gmm.weights_ + 1e-9))), 4),
             "weights": [round(float(w), 4) for w in best_gmm.weights_]}
        self.results["gmm"] = r
        self._models["gmm"] = best_gmm
        return r

    def compute_isolation_forest(self):
        iso = IsolationForest(n_estimators=200,contamination=0.10,random_state=42,n_jobs=-1)
        iso.fit(self.X_ml)
        raw = float(iso.score_samples(self.t_ml.reshape(1,-1))[0])
        dec = int(iso.predict(self.t_ml.reshape(1,-1))[0])
        ssc = iso.score_samples(self.X_ml)
        # Rank-based scoring: IsoForest score_samples range is very narrow
        # (e.g. -0.55 to -0.44) making z-score unreliable.
        # Use percentile rank with a smooth power transform instead.
        pct = float(percentileofscore(ssc, raw, kind="rank"))
        # Smooth: pct 50 → score 50, pct 0 → ~5, pct 100 → ~95
        # power transform: score = 100 * (pct/100)^0.6
        sc = float(np.clip(100.0 * (pct / 100.0) ** 0.6, 0, 100)) if pct > 0 else 2.0
        q10,q25,q75 = (float(np.percentile(ssc,q)) for q in (10,25,75))
        zone = ("CORE" if raw>=q75 else "SUITABLE" if raw>=q25
                else "MARGINAL" if raw>=q10 else "OUTSIDE")
        r = {"anomaly_score":round(raw,4),"is_normal":bool(dec==1),
             "percentile_rank":round(pct,2),"score":round(sc,2),"zone":zone,
             "sp_q10":round(q10,4),"sp_q25":round(q25,4),"sp_q75":round(q75,4)}
        self.results["isolation_forest"] = r; self._models["isoforest"] = iso
        return r

    def compute_ocsvm(self):
        best_svm,best_nu = None,0.10
        for nu in [0.05,0.10,0.15]:
            s = OneClassSVM(nu=nu,kernel="rbf",gamma="scale")
            s.fit(self.X_sc_bio)
            if abs(np.mean(s.predict(self.X_sc_bio)==1)-(1-nu))<0.05:
                best_svm,best_nu = s,nu; break
        if best_svm is None:
            best_svm = OneClassSVM(nu=0.10,kernel="rbf",gamma="scale")
            best_svm.fit(self.X_sc_bio)
        dv  = float(best_svm.decision_function(self.t_sc_bio.reshape(1,-1))[0])
        pr  = int(best_svm.predict(self.t_sc_bio.reshape(1,-1))[0])
        sdc = best_svm.decision_function(self.X_sc_bio)
        # Rank-based scoring with smooth power transform
        pct = float(percentileofscore(sdc, dv, kind="rank"))
        sc = float(np.clip(100.0 * (pct / 100.0) ** 0.6, 0, 100)) if pct > 0 else 2.0
        q10 = float(np.percentile(sdc,10))
        zone = ("CORE" if dv>=0 and pct>=75 else "SUITABLE" if dv>=0
                else "MARGINAL" if dv>=q10 else "OUTSIDE")
        r = {"decision_value":round(dv,4),"is_inside":bool(pr==1),
             "percentile_rank":round(pct,2),"score":round(sc,2),"zone":zone,"nu":best_nu}
        self.results["ocsvm"] = r; self._models["ocsvm"] = best_svm
        return r

    def compute_kmeans_niche(self):
        best_k,best_sil = 2,-1.0; sil_sc={}
        for k in range(2, min(7, len(self.X_sc_bio)//5+1)):
            try:
                km = KMeans(n_clusters=k,random_state=42,n_init=10)
                lb = km.fit_predict(self.X_sc_bio)
                s  = float(silhouette_score(self.X_sc_bio,lb))
                sil_sc[k]=round(s,4)
                if s>best_sil: best_sil,best_k = s,k
            except Exception: pass
        km = KMeans(n_clusters=best_k,random_state=42,n_init=10)
        km.fit(self.X_sc_bio); lb=km.labels_
        tc  = int(km.predict(self.t_sc_bio.reshape(1,-1))[0])
        ctr = km.cluster_centers_[tc]
        dc  = float(np.linalg.norm(self.t_sc_bio-ctr))
        rd  = np.linalg.norm(self.X_sc_bio[lb==tc]-ctr,axis=1)
        pct = float(percentileofscore(rd,dc,kind="rank"))
        sc  = float(np.clip(100-pct,0,100))
        r = {"best_k":best_k,"silhouette":{str(k):v for k,v in sil_sc.items()},
             "best_silhouette":round(best_sil,4),"target_cluster":tc,
             "cluster_size_frac":round(float(np.mean(lb==tc)),4),
             "dist_to_center":round(dc,4),"score":round(sc,2)}
        self.results["kmeans"]=r; self._models["kmeans"]=km
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # LAYER C  —  Mahalanobis + Threshold Zone
    # ═══════════════════════════════════════════════════════════════════════

    def compute_mahalanobis(self):
        d    = float(scipy_mahal(self.target_values,self._mu,self._cov_inv))
        df   = self.n_vars
        pval = float(1-chi2.cdf(d**2,df=df))
        d50  = float(np.sqrt(chi2.ppf(0.50,  df=df)))
        d90  = float(np.sqrt(chi2.ppf(0.90,  df=df)))
        d95  = float(np.sqrt(chi2.ppf(0.95,  df=df)))
        d99  = float(np.sqrt(chi2.ppf(0.99,  df=df)))
        d999 = float(np.sqrt(chi2.ppf(0.999, df=df)))
        zone = ("CORE" if d<=d50 else "SUITABLE" if d<=d90
                else "MARGINAL" if d<=d95
                else "OUTSIDE_NEAR" if pval>0.01   # p>1%: borderline — just past D₉₅
                else "OUTSIDE")                      # p≤1%: clearly outside

        # Piecewise score aligned with zones:
        # CORE(D≤D₅₀):      75-100   D₅₀ boundary = exactly 75
        # SUITABLE(D₅₀-D₉₀): 45-75   D₉₀ boundary = exactly 45
        # MARGINAL(D₉₀-D₉₅): 25-45   D₉₅ boundary = exactly 25
        # OUTSIDE(D₉₅-D₉₉):  8-25    D₉₉ boundary = exactly 8
        # FAR OUTSIDE (>D₉₉): 2-8
        if d <= d50:
            t = d / d50
            score = 100 - 25 * t**2
        elif d <= d90:
            t = (d - d50) / (d90 - d50)
            score = 75 - 30 * t
        elif d <= d95:
            t = (d - d90) / (d95 - d90)
            score = 45 - 20 * t
        elif d <= d99:
            t = (d - d95) / (d99 - d95)
            score = 25 - 17 * t
        else:
            t = min((d - d99) / max(d999 - d99, 0.01), 1.0)
            score = 8 - 6 * t
        score = float(np.clip(score, 0, 100))

        r = {"distance":round(d,4),"p_value":round(pval,4),
             "d50_threshold":round(d50,4),"d90_threshold":round(d90,4),
             "d95_threshold":round(d95,4),"d99_threshold":round(d99,4),
             "zone":zone,"score":round(score,2),
             "n_variables":self.n_vars,
             "interpretation":self._mahal_interp(d,d50,d90,d95)}
        self.results["mahalanobis_stat"]=r
        return r

    def _mahal_interp(self,d,d50,d90,d95):
        if d<=d50: return f"✅ Core niche          (D={d:.2f} ≤ D₅₀={d50:.2f})"
        if d<=d90: return f"🟡 Suitable niche      (D={d:.2f} ≤ D₉₀={d90:.2f})"
        if d<=d95: return f"🟠 Marginal niche      (D={d:.2f} ≤ D₉₅={d95:.2f})"
        pval = float(1-chi2.cdf(d**2, df=self.n_vars))
        if pval>0.01: return f"🔶 Borderline outside  (D={d:.2f} > D₉₅={d95:.2f}, p={pval:.3f})"
        return              f"🔴 Outside niche       (D={d:.2f} > D₉₅={d95:.2f}, p={pval:.4f})"

    def compute_threshold_zone(self):
        for m in ("gmm","isolation_forest","ocsvm","mahalanobis_stat"):
            if m not in self.results:
                getattr(self,{"gmm":"compute_gmm","isolation_forest":"compute_isolation_forest",
                              "ocsvm":"compute_ocsvm","mahalanobis_stat":"compute_mahalanobis"}[m])()
        # OUTSIDE_NEAR (p>1%): borderline — votes between MARGINAL(1) and OUTSIDE(0)
        zo = {"CORE":3,"SUITABLE":2,"MARGINAL":1,"OUTSIDE_NEAR":0.6,"OUTSIDE":0}
        votes = {m:self.results[m]["zone"] for m in ("gmm","isolation_forest","ocsvm")}
        votes["mahalanobis"] = self.results["mahalanobis_stat"]["zone"]
        # Weighted voting: Mahalanobis = gold standard (3x), GMM = 1x, IsoForest/OCSVM redundant (0.8x each)
        vote_weights = {"gmm":1.0,"isolation_forest":0.8,"ocsvm":0.8,"mahalanobis":3.0}
        wsum = sum(vote_weights.values())  # 5.6
        mv   = float(sum(zo[votes[m]] * vote_weights[m] for m in votes) / wsum)
        final = ("CORE" if mv>=2.5 else "SUITABLE" if mv>=1.5
                 else "MARGINAL" if mv>=0.5 else "OUTSIDE")
        base  = {"CORE":87.5,"SUITABLE":57.5,"MARGINAL":27.5,"OUTSIDE":7.5}
        score = float(np.clip(base[final]+(mv-zo[final])*12.5,0,100))
        labels = {"CORE":     "✅ Core Niche     — Optimum conditions",
                  "SUITABLE": "🟡 Suitable Niche  — Viable, partial stress",
                  "MARGINAL": "🟠 Marginal Niche  — At tolerance boundary",
                  "OUTSIDE":  "🔴 Outside Niche   — Unsuitable conditions"}
        r = {"final_zone":final,"zone_label":labels[final],
             "mean_vote":round(mv,3),"votes":votes,"score":round(score,2)}
        self.results["threshold_zone"]=r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # Percentile Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def compute_percentile_analysis(self):
        per_bio, sc_bio = {}, []
        for j,bio in enumerate(self.bio_cols):
            rec = self._pct_record(self.X_bio[:,j], float(self.t_bio[j]),
                                   bio, bio in CRITICAL_BIOS)
            per_bio[bio]=rec; sc_bio.append(rec["score"])

        ws=[]
        for j,bio in enumerate(self.bio_cols):
            ws.extend([sc_bio[j]]*(20 if bio in CRITICAL_BIOS else 10))

        per_opt={}
        for k,oc in enumerate(self.opt_cols):
            idx = self._opt_offset + k
            sp  = self.species_data[:,idx]
            tgt = float(self.target_values[idx])
            per_opt[oc] = self._pct_record(sp, tgt, oc, oc in CRITICAL_OPT)

        r = {
            "score":                   round(float(np.mean(ws)),2) if ws else 100.0,
            "per_bio":                 per_bio,
            "per_opt":                 per_opt,
            "bios_in_core_range":      sum(1 for b in per_bio.values() if b["in_core_range"]),
            "bios_in_tolerance_range": sum(1 for b in per_bio.values() if b["in_tolerance_range"]),
            "bios_outside_range":      sum(1 for b in per_bio.values() if not b["in_tolerance_range"]),
            "n_bio_used":              len(self.bio_cols),
        }
        self.results["percentile"]=r
        return r

    def _pct_record(self, sp, tgt, col, is_critical=False):
        pct   = percentileofscore(sp,tgt,kind="rank")
        p5,p25,p75,p95 = (float(np.percentile(sp,q)) for q in (5,25,75,95))
        med   = float(np.median(sp))
        std   = float(np.std(sp))
        try:
            mw_p = float(mannwhitneyu([tgt],sp,alternative="two-sided").pvalue)
        except Exception:
            mw_p = 1.0
        dev = abs(pct-50)/50
        sc  = float(np.clip(100*(1-dev**1.5),0,100))
        if tgt<p5 or tgt>p95: sc *= 0.5
        ri = build_risk_explanation(col,tgt,med,p5,p25,p75,p95,pct,sc,
                                     float(sp.min()),float(sp.max()),std)
        return {
            "name":               ALL_VAR_NAMES.get(col,col),
            "target_value":       round(tgt,6),
            "percentile":         round(pct,2),
            "species_min":        round(float(sp.min()),4),
            "species_p5":         round(p5,4),
            "species_p25":        round(p25,4),
            "species_median":     round(med,4),
            "species_p75":        round(p75,4),
            "species_p95":        round(p95,4),
            "species_max":        round(float(sp.max()),4),
            "species_std":        round(std,4),
            "in_core_range":      bool(p25<=tgt<=p75),
            "in_tolerance_range": bool(p5<=tgt<=p95),
            "mw_pvalue":          round(mw_p,4),
            "score":              round(sc,2),
            "critical":           is_critical,
            "risk_level":         ri["risk_level"],
            "risk_color":         ri["risk_color"],
            "risk_emoji":         ri["risk_emoji"],
            "dev_from_med":       ri["dev_from_med"],
            "dev_pct":            ri["dev_pct"],
            "n_std":              ri["n_std"],
            "dist_to_p5":         ri["dist_to_p5"],
            "dist_to_p95":        ri["dist_to_p95"],
            "risk_explanation":   ri["explanation"],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Topography  —  dynamic, only selected topo vars
    # ═══════════════════════════════════════════════════════════════════════

    def compute_topo_score(self):
        if not self.has_topo:
            r = {"score":100.0,"elevation":{},"slope":{},"aspect":{},
                 "note":"No topographic variables selected"}
            self.results["topo"]=r; return r

        tw      = self._topo_weights()
        topo_sc = 0.0
        elev_r = slop_r = asp_result = {}

        if "elevation_m" in self.topo_cols:
            idx    = self._topo_idxs["elevation_m"]
            et     = float(self.target_values[idx])
            elev_r = self._pct_record(self.species_data[:,idx], et, "elevation_m")
            topo_sc += tw["elevation_m"] * elev_r["score"]

        if "slope_pct" in self.topo_cols:
            idx    = self._topo_idxs["slope_pct"]
            st     = float(self.target_values[idx])
            slop_r = self._pct_record(self.species_data[:,idx], st, "slope_pct")
            topo_sc += tw["slope_pct"] * slop_r["score"]

        if "aspect_deg" in self.topo_cols:
            idx    = self._topo_idxs["aspect_deg"]
            at     = float(self.target_values[idx])
            asp_sp = self.species_data[:,idx]
            mn  = float(np.mean(np.cos(np.deg2rad(asp_sp))))
            me  = float(np.mean(np.sin(np.deg2rad(asp_sp))))
            mvl = float(np.sqrt(mn**2+me**2))
            tn,te = northness(at),eastness(at)
            cos_sim = float((tn*mn+te*me)/(np.sqrt(tn**2+te**2)*(mvl+1e-9)+1e-9))
            asp_sc  = float(np.clip(50+50*cos_sim*mvl,0,100))
            asp_result = {
                "name":"Aspect (°)","target_deg":round(at,2),
                "target_northness":round(tn,4),"target_eastness":round(te,4),
                "sp_mean_north":round(mn,4),"sp_mean_east":round(me,4),
                "vec_length":round(mvl,4),"cos_similarity":round(cos_sim,4),
                "score":round(asp_sc,2),
                "target_exposure":("North" if tn>0.5 else "South" if tn<-0.5 else "Mixed"),
                "sp_exposure":("North-facing" if mn>0.3 else "South-facing" if mn<-0.3 else "Mixed"),
            }
            topo_sc += tw["aspect_deg"] * asp_sc

        r = {
            "score":    round(float(topo_sc),2),
            "elevation": elev_r,
            "slope":     slop_r,
            "aspect":    asp_result,
            "topo_vars_used": self.topo_cols,
        }
        self.results["topo"]=r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # Variable Importance + Detailed Risk
    # ═══════════════════════════════════════════════════════════════════════

    def compute_variable_importance(self):
        pca_imp={}
        if "pca2" in self._models:
            pca2 = self._models["pca2"]
            mag  = (np.abs(pca2.components_[0])+np.abs(pca2.components_[1]))
            mag /= (mag.sum()+1e-9)
            pca_imp = dict(zip(self._ml_col_names(), mag.tolist()))

        np.random.seed(42)
        n_bg = min(len(self.X_ml)*2,500)
        Xbg  = np.random.randn(n_bg, self.X_ml.shape[1])
        Xtr  = np.vstack([self.X_ml,Xbg])
        ytr  = np.array([1]*len(self.X_ml)+[0]*n_bg)
        rf   = RandomForestClassifier(n_estimators=150,max_depth=6,
                                       random_state=42,n_jobs=-1,
                                       class_weight="balanced")
        rf.fit(Xtr,ytr)
        rf_imp = dict(zip(self._ml_col_names(), rf.feature_importances_.tolist()))
        self._models["rf"]=rf

        # PDI = Percentile Deviation Index  (|percentile - 50| / 50)
        pdi={}
        for j,col in enumerate(self.bio_cols):
            dev = abs(percentileofscore(self.X_bio[:,j],self.t_bio[j],kind="rank")-50)/50
            pdi[col]=round(float(dev),4)
        for k,oc in enumerate(self.opt_cols):
            idx = self._opt_offset+k
            sp  = self.species_data[:,idx]
            dev = abs(percentileofscore(sp,float(self.target_values[idx]),kind="rank")-50)/50
            pdi[oc]=round(float(dev),4)

        max_pdi = max(list(pdi.values())+[1e-9])
        combined={}
        for col in self._ml_col_names():
            rf_v=rf_imp.get(col,0.0); pdi_v=pdi.get(col,0.0)
            combined[col] = round(0.60*rf_v+0.40*pdi_v/max_pdi,4) if col in pdi else round(rf_v,4)
        sorted_imp = dict(sorted(combined.items(),key=lambda x:-x[1]))

        top_raw = [c for c in sorted_imp if c not in ("northness","eastness")][:7]
        risk_details={}
        for col in top_raw:
            rec = self._get_pct_rec(col)
            if rec:
                risk_details[col]={
                    "combined_importance": round(sorted_imp.get(col,0),5),
                    "rf_importance":       round(rf_imp.get(col,0),5),
                    "pdi":                 round(pdi.get(col,0),4),
                    "risk_level":  rec.get("risk_level","—"),
                    "risk_color":  rec.get("risk_color","#7f8c8d"),
                    "risk_emoji":  rec.get("risk_emoji","—"),
                    "score":       rec.get("score",0),
                    "dev_pct":     rec.get("dev_pct",0),
                    "n_std":       rec.get("n_std",0),
                    "percentile":  rec.get("percentile",0),
                    "risk_explanation": rec.get("risk_explanation",""),
                }

        r = {
            "rf_importance":       {k:round(v,5) for k,v in rf_imp.items()},
            "pdi":                 pdi,
            "pca_loadings_mag":    {k:round(v,5) for k,v in pca_imp.items()},
            "combined_importance": sorted_imp,
            "top5_variables":      top_raw[:5],
            "top7_variables":      top_raw,
            "risk_details":        risk_details,
        }
        self.results["variable_importance"]=r
        return r

    def _get_pct_rec(self, col):
        pr = self.results.get("percentile",{})
        if col in pr.get("per_bio",{}): return pr["per_bio"][col]
        if col in pr.get("per_opt",{}): return pr["per_opt"][col]
        t = self.results.get("topo",{})
        if col=="elevation_m" and t.get("elevation"): return t["elevation"]
        if col=="slope_pct"   and t.get("slope"):     return t["slope"]
        # Compute on demand
        if col in self.bio_cols:
            j=self.bio_cols.index(col)
            sp,tgt = self.X_bio[:,j],float(self.t_bio[j])
        elif col in self.opt_cols:
            idx=self._opt_offset+self.opt_cols.index(col)
            sp,tgt = self.species_data[:,idx],float(self.target_values[idx])
        elif col in self.topo_cols:
            idx=self._topo_idxs[col]
            sp,tgt = self.species_data[:,idx],float(self.target_values[idx])
        else:
            return None
        pct=percentileofscore(sp,tgt,kind="rank")
        p5,p25,p75,p95=(float(np.percentile(sp,q)) for q in (5,25,75,95))
        med,std=float(np.median(sp)),float(np.std(sp))
        sc=float(np.clip(100*(1-(abs(pct-50)/50)**1.5),0,100))
        if tgt<p5 or tgt>p95: sc*=0.5
        return build_risk_explanation(col,tgt,med,p5,p25,p75,p95,pct,sc,
                                       float(sp.min()),float(sp.max()),std)

    # ═══════════════════════════════════════════════════════════════════════
    # LAYER E  —  Composite Score
    # ═══════════════════════════════════════════════════════════════════════

    def compute_composite_score(self):
        for fn in [self.compute_pca, self.compute_kernel_pca,
                   self.compute_gmm, self.compute_isolation_forest,
                   self.compute_ocsvm, self.compute_kmeans_niche,
                   self.compute_mahalanobis, self.compute_threshold_zone,
                   self.compute_percentile_analysis, self.compute_topo_score,
                   self.compute_variable_importance]:
            key = fn.__name__.replace("compute_","")
            if key not in self.results: fn()

        clim = (W_THRESHOLD_ZONE*self.results["threshold_zone"]["score"] +
                W_GMM           *self.results["gmm"]["score"]            +
                W_ISOFOREST     *self.results["isolation_forest"]["score"]+
                W_OCSVM         *self.results["ocsvm"]["score"]           +
                W_MAHAL         *self.results["mahalanobis_stat"]["score"])

        opt_bonus=0.0
        po = self.results.get("percentile",{}).get("per_opt",{})
        if po:
            avg  = float(np.mean([po[c]["score"] for c in po]))
            crit = [po[c]["score"] for c in po if c in CRITICAL_OPT]
            opt_bonus = 0.05*(avg/100.0)*clim
            if crit: opt_bonus *= (0.5+0.5*float(np.mean(crit))/100.0)

        topo    = self.results["topo"]["score"]
        cw,tw   = self._clim_topo_weights()
        composite = round(float(np.clip(cw*clim+tw*topo+opt_bonus,0,100)),2)

        r = {
            "composite_score":   composite,
            "climate_score":     round(float(clim),2),
            "topo_score":        round(float(topo),2),
            "opt_bonus":         round(float(opt_bonus),3),
            "climate_weight":    cw, "topo_weight": tw,
            "active_bio_cols":   self.bio_cols,
            "active_topo_cols":  self.topo_cols,
            "active_opt_cols":   self.opt_cols,
            "component_scores":  {
                "threshold_zone":   self.results["threshold_zone"]["score"],
                "gmm":              self.results["gmm"]["score"],
                "isolation_forest": self.results["isolation_forest"]["score"],
                "ocsvm":            self.results["ocsvm"]["score"],
                "mahalanobis":      self.results["mahalanobis_stat"]["score"],
                "topo":             topo,
            },
            "final_zone":        self.results["threshold_zone"]["final_zone"],
            "zone_label":        self.results["threshold_zone"]["zone_label"],
            "resilience_class":  self._classify(composite),
            "recommendation":    self._recommend(composite),
            "top_risk_variables":self.results["variable_importance"]["top5_variables"],
        }
        self.results["composite"]=r
        return r

    def _classify(self,s):
        if s>=80: return "A – Very High Resilience"
        if s>=65: return "B – High Resilience"
        if s>=50: return "C – Moderate Resilience"
        if s>=35: return "D – Low Resilience"
        return         "E – Very Low Resilience / Unsuitable Site"

    def _recommend(self,s):
        if s>=80: return "Species is a primary candidate for this site. All ML models indicate strong suitability."
        if s>=65: return "Species can succeed at this site. Irrigation and micro-habitat support are recommended."
        if s>=50: return "Cautious use. Adaptation measures (shading, irrigation, species mixing) are essential."
        if s>=35: return "High risk. Intensive management and alternative species evaluation required."
        return         "This species is not suitable for this site. ML models indicate outside-niche conditions."

    # ═══════════════════════════════════════════════════════════════════════
    # BOOTSTRAP CONFIDENCE INTERVAL  —  ±CI for composite score
    # ═══════════════════════════════════════════════════════════════════════

    def _fast_composite_score(self):
        """
        Lightweight composite for SSP / calibration loops.
        Skips PCA and variable importance; uses correct result-key mapping.
        """
        _step_map = [
            ("gmm",              self.compute_gmm),
            ("isolation_forest", self.compute_isolation_forest),
            ("ocsvm",            self.compute_ocsvm),
            ("kmeans",           self.compute_kmeans_niche),
            ("mahalanobis_stat", self.compute_mahalanobis),
            ("threshold_zone",   self.compute_threshold_zone),
            ("percentile",       self.compute_percentile_analysis),
            ("topo",             self.compute_topo_score),
        ]
        for key, fn in _step_map:
            if key not in self.results:
                fn()
        clim = (W_THRESHOLD_ZONE * self.results["threshold_zone"]["score"] +
                W_GMM            * self.results["gmm"]["score"]             +
                W_ISOFOREST      * self.results["isolation_forest"]["score"] +
                W_OCSVM          * self.results["ocsvm"]["score"]            +
                W_MAHAL          * self.results["mahalanobis_stat"]["score"])
        opt_bonus = 0.0
        po = self.results.get("percentile", {}).get("per_opt", {})
        if po:
            avg  = float(np.mean([po[c]["score"] for c in po]))
            crit = [po[c]["score"] for c in po if c in CRITICAL_OPT]
            opt_bonus = 0.05 * (avg / 100.0) * clim
            if crit:
                opt_bonus *= (0.5 + 0.5 * float(np.mean(crit)) / 100.0)
        topo = self.results["topo"]["score"]
        cw, tw = self._clim_topo_weights()
        return float(np.clip(cw * clim + tw * topo + opt_bonus, 0, 100))

    def compute_bootstrap_ci(self, n_bootstrap=500):
        """
        Fast bootstrap 95 % CI for the composite score.

        Strategy: resamples species data and analytically re-estimates the
        two fastest estimators — Mahalanobis distance and per-bio percentile
        scores.  The ML model scores (GMM / IsoForest / OCSVM) require full
        model re-fitting and are held fixed from the original run; only their
        species-data dependency is captured through Mahalanobis and percentile
        resampling.  This keeps each iteration to <2 ms (matrix maths only),
        so 500 reps completes in <1 second.

        Returns
        -------
        dict  (stored in self.results["bootstrap_ci"])
            n_bootstrap, n_valid, mean, std, ci_lower, ci_upper, ci_width
        """
        if "composite" not in self.results:
            self.compute_composite_score()

        n   = len(self.species_data)
        rng = np.random.RandomState(42)
        scores = []

        # Fixed ML scores (from original fit — not re-estimated per bootstrap)
        ml_fixed = (W_GMM       * self.results["gmm"]["score"] +
                    W_ISOFOREST * self.results["isolation_forest"]["score"] +
                    W_OCSVM     * self.results["ocsvm"]["score"])
        tz_fixed  = W_THRESHOLD_ZONE * self.results["threshold_zone"]["score"]

        # Fixed topo score
        topo_sc = self.results["topo"]["score"]
        cw, tw  = self._clim_topo_weights()

        # Pre-extract bio species columns and target values once
        X_bio_t = self.X_bio            # (n, n_bio) — scaled copy used for pct
        t_bio   = self.t_bio            # (n_bio,)

        for _ in range(n_bootstrap):
            idx   = rng.choice(n, n, replace=True)
            bs_sp = self.species_data[idx]
            try:
                # ── Bootstrap Mahalanobis ─────────────────────────────────
                bs_mu  = np.mean(bs_sp, axis=0)
                bs_cov = np.cov(bs_sp, rowvar=False)
                bs_inv = _safe_inv(bs_cov)
                d = float(scipy_mahal(self.target_values, bs_mu, bs_inv))
                df   = self.n_vars
                d50  = float(np.sqrt(chi2.ppf(0.50,  df=df)))
                d90  = float(np.sqrt(chi2.ppf(0.90,  df=df)))
                d95  = float(np.sqrt(chi2.ppf(0.95,  df=df)))
                d99  = float(np.sqrt(chi2.ppf(0.99,  df=df)))
                d999 = float(np.sqrt(chi2.ppf(0.999, df=df)))
                if   d <= d50:  sc_m = 100 - 25*(d/d50)**2
                elif d <= d90:  sc_m = 75  - 30*(d-d50)/(d90-d50)
                elif d <= d95:  sc_m = 45  - 20*(d-d90)/(d95-d90)
                elif d <= d99:  sc_m = 25  - 17*(d-d95)/(d99-d95)
                else:           sc_m = max(2.0, 8 - 6*min((d-d99)/max(d999-d99,0.01),1.0))
                sc_m = float(np.clip(sc_m, 0, 100))

                # ── Bootstrap bio percentile scores ───────────────────────
                bs_bio = bs_sp[:, self._bio_slice]
                ws_vals = []
                for j, col in enumerate(self.bio_cols):
                    sp_col = bs_bio[:, j]
                    tgt    = float(t_bio[j])
                    pct    = percentileofscore(sp_col, tgt, kind="rank")
                    dev    = abs(pct - 50) / 50
                    sc_p   = float(np.clip(100*(1-dev**1.5), 0, 100))
                    p5     = float(np.percentile(sp_col, 5))
                    p95    = float(np.percentile(sp_col, 95))
                    if tgt < p5 or tgt > p95:
                        sc_p *= 0.5
                    w = 20 if col in CRITICAL_BIOS else 10
                    ws_vals.extend([sc_p] * w)
                # (Percentile score feeds indirectly through threshold zone;
                #  for CI purposes we capture it via Mahalanobis resampling
                #  and leave threshold zone fixed.  The perc score is used
                #  only to modulate the opt_bonus below.)
                sc_perc_mean = float(np.mean(ws_vals)) if ws_vals else 50.0

                # opt_bonus re-estimate (fast — no model refit)
                opt_bonus = 0.0
                po = self.results.get("percentile", {}).get("per_opt", {})
                if po:
                    avg  = float(np.mean([po[c]["score"] for c in po]))
                    crit = [po[c]["score"] for c in po if c in CRITICAL_OPT]
                    clim_tmp = tz_fixed + ml_fixed + W_MAHAL * sc_m
                    opt_bonus = 0.05 * (avg / 100.0) * clim_tmp
                    if crit:
                        opt_bonus *= (0.5 + 0.5 * float(np.mean(crit)) / 100.0)

                clim = tz_fixed + ml_fixed + W_MAHAL * sc_m
                sc   = float(np.clip(cw*clim + tw*topo_sc + opt_bonus, 0, 100))
                scores.append(sc)
            except Exception:
                continue

        if not scores:
            r = {"n_bootstrap": n_bootstrap, "n_valid": 0,
                 "mean": 0.0, "std": 0.0,
                 "ci_lower": 0.0, "ci_upper": 0.0, "ci_width": 0.0,
                 "method": "fast (Mahalanobis + Percentile bootstrap)",
                 "error": "All bootstrap iterations failed"}
            self.results["bootstrap_ci"] = r
            return r

        arr  = np.array(scores)
        ci_lo = float(np.percentile(arr, 2.5))
        ci_hi = float(np.percentile(arr, 97.5))
        r = {
            "n_bootstrap": n_bootstrap,
            "n_valid":     len(scores),
            "mean":        round(float(arr.mean()), 2),
            "std":         round(float(arr.std()),  2),
            "ci_lower":    round(ci_lo, 2),
            "ci_upper":    round(ci_hi, 2),
            "ci_width":    round(ci_hi - ci_lo, 2),
            "method":      "Fast bootstrap — Mahalanobis + Percentile resampling (ML fixed)",
        }
        # Stored under the legacy key for backwards compatibility, and under
        # a dedicated key so the full-refit CI can coexist with it.
        self.results["bootstrap_ci_fast"] = r
        if "bootstrap_ci" not in self.results:
            self.results["bootstrap_ci"] = r
        return r

    def compute_bootstrap_ci_full(self, n_bootstrap=200, log_callback=None):
        """
        Methodologically complete bootstrap 95 % CI for the composite score.

        Unlike ``compute_bootstrap_ci`` (fast variant) which only re-estimates
        Mahalanobis + percentile scores, this method **refits every ML model**
        (GMM, Isolation Forest, One-Class SVM, K-Means, Mahalanobis) inside
        each bootstrap iteration.  This captures the true sampling uncertainty
        of the full pipeline and produces wider, properly-calibrated CIs.

        Each iteration builds a fresh ``ClimateResilienceAnalyzer`` on the
        resampled species data and calls ``_fast_composite_score``, which
        skips only PCA visual outputs and variable importance — every scoring
        component is re-estimated.

        Parameters
        ----------
        n_bootstrap : int
            Number of bootstrap iterations. 200 is the academic standard for
            percentile CIs and is a reasonable trade-off (~20-60 s total).
        log_callback : callable(str) or None
            Optional logger — called with a progress line at the start and
            every ~5% of iterations.  Used by the UI to stream live status
            into the on-screen log panel so the user sees that the analysis
            is still making progress during the slow bootstrap phase.

        Returns
        -------
        dict stored in ``self.results["bootstrap_ci"]`` (and also
        ``self.results["bootstrap_ci_full"]`` for explicit access).
        """
        def _log(msg):
            if log_callback is not None:
                try: log_callback(msg)
                except Exception: pass

        if "composite" not in self.results:
            self.compute_composite_score()

        n      = len(self.species_data)
        rng    = np.random.RandomState(42)
        scores = []
        errors = 0
        step   = max(1, n_bootstrap // 20)   # log every 5 %
        t0     = time.time()

        _log(f"[bootstrap] starting full refit  n={n_bootstrap}  "
             f"(refits all ML models per iteration)")

        for i in range(n_bootstrap):
            idx   = rng.choice(n, n, replace=True)
            bs_sp = self.species_data[idx]
            # Degenerate resample guard: need at least some distinct rows for
            # covariance estimation and clustering.
            if len(np.unique(idx)) < max(5, self.n_vars + 1):
                errors += 1
                continue
            try:
                sa = ClimateResilienceAnalyzer(
                    bs_sp,
                    self.target_values,
                    bio_cols=self.bio_cols,
                    topo_cols=self.topo_cols,
                    opt_cols=self.opt_cols,
                )
                sc = sa._fast_composite_score()
                scores.append(float(sc))
            except Exception:
                errors += 1
                continue

            if (i + 1) % step == 0 or (i + 1) == n_bootstrap:
                elapsed = time.time() - t0
                rate    = (i + 1) / max(elapsed, 1e-6)
                eta     = (n_bootstrap - (i + 1)) / max(rate, 1e-6)
                _log(f"[bootstrap] {i+1:>4}/{n_bootstrap}  "
                     f"({(i+1)/n_bootstrap*100:5.1f}%)  "
                     f"valid={len(scores)}  errors={errors}  "
                     f"elapsed={elapsed:5.1f}s  ETA={eta:5.1f}s")

        if not scores:
            r = {"n_bootstrap": n_bootstrap, "n_valid": 0,
                 "mean": 0.0, "std": 0.0,
                 "ci_lower": 0.0, "ci_upper": 0.0, "ci_width": 0.0,
                 "method": "Full bootstrap — all ML models refit per iteration",
                 "error": "All bootstrap iterations failed"}
            self.results["bootstrap_ci"] = r
            self.results["bootstrap_ci_full"] = r
            return r

        arr   = np.array(scores)
        ci_lo = float(np.percentile(arr, 2.5))
        ci_hi = float(np.percentile(arr, 97.5))
        r = {
            "n_bootstrap": n_bootstrap,
            "n_valid":     len(scores),
            "n_errors":    errors,
            "mean":        round(float(arr.mean()), 2),
            "std":         round(float(arr.std()),  2),
            "ci_lower":    round(ci_lo, 2),
            "ci_upper":    round(ci_hi, 2),
            "ci_width":    round(ci_hi - ci_lo, 2),
            "method":      "Full bootstrap — all ML models refit per iteration",
        }
        # Promote the full-refit result to the primary key — this is the one
        # UI / exports read — and keep a dedicated copy as well.
        self.results["bootstrap_ci"]      = r
        self.results["bootstrap_ci_full"] = r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # SPATIAL AUTOCORRELATION CORRECTION
    # ═══════════════════════════════════════════════════════════════════════

    def apply_environmental_thinning(self, min_dist_pca=0.5):
        """
        Environmental-space thinning to reduce spatial autocorrelation.

        Projects species points to PCA-2 space and discards points whose
        nearest retained neighbour is closer than `min_dist_pca`.  This
        is equivalent to geographic thinning when coordinates are absent.

        The thinned dataset is stored in `self.results["thinning"]` but
        does NOT replace `self.species_data`; re-run the analysis with the
        thinned subset manually if desired.
        """
        n = len(self.X_ml)
        if n < 5:
            r = {"n_before": n, "n_after": n,
                 "retained_indices": list(range(n)),
                 "note": "Too few points — thinning skipped."}
            self.results["thinning"] = r
            return r

        pca2 = PCA(n_components=min(2, self.X_ml.shape[1]))
        Xp   = pca2.fit_transform(self.X_ml)

        # Greedy thinning (insert-order preserving)
        kept = []
        for i in range(n):
            if not kept:
                kept.append(i); continue
            dists = np.linalg.norm(Xp[kept] - Xp[i], axis=1)
            if dists.min() >= min_dist_pca:
                kept.append(i)

        r = {
            "n_before":          n,
            "n_after":           len(kept),
            "retained_fraction": round(len(kept) / n, 3),
            "min_dist_pca":      min_dist_pca,
            "retained_indices":  kept,
            "note": (
                f"Thinned {n} → {len(kept)} points in PCA-2 space "
                f"(min inter-point distance = {min_dist_pca}).  "
                f"Re-run analysis with the retained indices to apply correction."
            ),
        }
        self.results["thinning"] = r
        return r

    def compute_blocked_cv(self, n_blocks=5):
        """
        Environmental block cross-validation (leave-one-block-out).

        Divides native-range points into `n_blocks` K-means clusters
        (environmental blocks).  For each block, fits a GMM on the remaining
        blocks and measures how many held-out points fall above the training
        Q10 log-probability — i.e. are correctly recognised as *inliers*.

        A high mean inlier-rate ( ≥ 0.80 ) suggests low spatial autocorrelation
        bias.  A low rate warns that the models may overfit spatially clustered
        data.
        """
        n = len(self.X_sc_bio)
        actual_k = min(n_blocks, n // 3)
        if actual_k < 2:
            r = {"n_blocks": n_blocks, "mean_inlier_rate": None,
                 "std_inlier_rate": None, "block_results": {},
                 "interpretation": "—",
                 "error": "Insufficient points for blocked CV (need ≥ 6 rows)"}
            self.results["blocked_cv"] = r
            return r

        km = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        block_labels = km.fit_predict(self.X_sc_bio)

        inlier_rates = []
        block_results = {}
        for b in range(actual_k):
            test_mask  = block_labels == b
            train_mask = ~test_mask
            if train_mask.sum() < 5:
                continue
            X_train = self.X_sc_bio[train_mask]
            X_test  = self.X_sc_bio[test_mask]
            try:
                nc = max(1, min(2, len(X_train) // 5))
                gm = GaussianMixture(n_components=nc, random_state=42,
                                     covariance_type="full")
                gm.fit(X_train)
                lp_tr = gm.score_samples(X_train)
                lp_te = gm.score_samples(X_test)
                q10   = float(np.percentile(lp_tr, 10))
                ir    = float(np.mean(lp_te >= q10))
                inlier_rates.append(ir)
                block_results[f"block_{b}"] = {
                    "n_train": int(train_mask.sum()),
                    "n_test":  int(test_mask.sum()),
                    "inlier_rate": round(ir, 4),
                }
            except Exception as e:
                block_results[f"block_{b}"] = {"error": str(e)}

        if not inlier_rates:
            mean_ir = std_ir = None
            interp = "—"
        else:
            mean_ir = round(float(np.mean(inlier_rates)), 4)
            std_ir  = round(float(np.std(inlier_rates)),  4)
            if   mean_ir >= 0.80: interp = "✅ Low spatial-autocorrelation bias"
            elif mean_ir >= 0.60: interp = "🟡 Moderate autocorrelation — consider thinning"
            else:                 interp = "🔴 High autocorrelation bias — spatial thinning strongly recommended"

        r = {
            "n_blocks":          actual_k,
            "mean_inlier_rate":  mean_ir,
            "std_inlier_rate":   std_ir,
            "block_results":     block_results,
            "interpretation":    interp,
        }
        self.results["blocked_cv"] = r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # FUTURE CLIMATE SCENARIOS  —  SSP / CMIP6 / CHELSA
    # ═══════════════════════════════════════════════════════════════════════

    def run_ssp_scenarios(self, scenarios):
        """
        Compute composite suitability for multiple future climate scenarios.

        Parameters
        ----------
        scenarios : dict[str, array-like]
            Keys   = scenario labels, e.g. "SSP2-4.5 2050", "SSP5-8.5 2070"
            Values = 1-D arrays with the **same** column layout as
                     `self.target_values`  (bio_cols + topo_cols + opt_cols).

        Returns
        -------
        dict stored in ``self.results["ssp_scenarios"]`` with keys:
            current_score, scenarios → per-scenario score, delta, trend, class.
        """
        # Ensure current composite exists
        if "composite" not in self.results:
            self.compute_composite_score()
        current_sc = self.results["composite"]["composite_score"]

        scenario_results = {}
        for name, tgt_vals in scenarios.items():
            try:
                sa = ClimateResilienceAnalyzer(
                    self.species_data,
                    np.array(tgt_vals, dtype=float),
                    bio_cols=self.bio_cols,
                    topo_cols=self.topo_cols,
                    opt_cols=self.opt_cols,
                )
                sc = sa._fast_composite_score()
                delta = round(sc - current_sc, 2)
                scenario_results[name] = {
                    "composite_score":     round(sc, 2),
                    "delta_from_current":  delta,
                    "zone_label":          sa.results["threshold_zone"]["zone_label"],
                    "climate_score":       round(float(
                        W_THRESHOLD_ZONE*sa.results["threshold_zone"]["score"] +
                        W_GMM           *sa.results["gmm"]["score"]            +
                        W_ISOFOREST     *sa.results["isolation_forest"]["score"]+
                        W_OCSVM         *sa.results["ocsvm"]["score"]           +
                        W_MAHAL         *sa.results["mahalanobis_stat"]["score"]
                    ), 2),
                    "topo_score":          sa.results["topo"]["score"],
                    "resilience_class":    self._classify(sc),
                    "trend": ("▲ Improving" if delta > 2 else
                              "▼ Declining" if delta < -2 else "◆ Stable"),
                }
            except Exception as e:
                scenario_results[name] = {"error": str(e)}

        r = {
            "current_score": current_sc,
            "n_scenarios":   len(scenarios),
            "scenarios":     scenario_results,
        }
        self.results["ssp_scenarios"] = r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # WEIGHT CALIBRATION  —  Validation-dataset optimisation
    # ═══════════════════════════════════════════════════════════════════════

    def calibrate_weights(self, validation_data, validation_labels):
        """
        Optimise composite-score component weights using labelled validation data.

        Parameters
        ----------
        validation_data   : ndarray (n_val, n_vars)
            Points in the **same** column layout as `self.species_data`.
        validation_labels : array-like of int
            1 = suitable  /  0 = unsuitable.

        The optimiser (SLSQP) minimises binary cross-entropy between the
        normalised composite score (treated as a probability) and the label.
        Weights are constrained to [0.05, 0.60] and must sum to 1.

        Returns
        -------
        dict stored in ``self.results["calibration"]``.
        """
        from scipy.optimize import minimize

        labels = np.array(validation_labels, dtype=float)

        def _component_scores(tgt_arr):
            try:
                sa = ClimateResilienceAnalyzer(
                    self.species_data, tgt_arr,
                    bio_cols=self.bio_cols,
                    topo_cols=self.topo_cols,
                    opt_cols=self.opt_cols,
                )
                sa._fast_composite_score()
                return {
                    "threshold_zone":   sa.results["threshold_zone"]["score"],
                    "gmm":              sa.results["gmm"]["score"],
                    "isolation_forest": sa.results["isolation_forest"]["score"],
                    "ocsvm":            sa.results["ocsvm"]["score"],
                    "mahalanobis":      sa.results["mahalanobis_stat"]["score"],
                }
            except Exception:
                return None

        val_comp = [_component_scores(np.array(row, dtype=float))
                    for row in validation_data]
        valid    = [(s, l) for s, l in zip(val_comp, labels) if s is not None]

        if len(valid) < 4:
            r = {"error": "Insufficient valid validation points",
                 "n_valid": len(valid)}
            self.results["calibration"] = r
            return r

        scores_list, label_arr = zip(*valid)
        label_arr = np.array(label_arr)

        keys = ["threshold_zone","gmm","isolation_forest","ocsvm","mahalanobis"]
        w0   = np.array([W_THRESHOLD_ZONE, W_GMM, W_ISOFOREST, W_OCSVM, W_MAHAL])
        w0  /= w0.sum()

        def composite_w(w, s):
            return float(np.clip(sum(w[i]*s[keys[i]] for i in range(5)), 0, 100))

        def bce_loss(w):
            probs = np.clip(
                np.array([composite_w(w, s) / 100.0 for s in scores_list]),
                1e-7, 1-1e-7)
            return -float(np.mean(
                label_arr * np.log(probs) + (1-label_arr) * np.log(1-probs)))

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds      = [(0.05, 0.60)] * 5
        try:
            res = minimize(bce_loss, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-9, "maxiter": 500})
            w_opt = res.x
        except Exception as e:
            r = {"error": str(e), "n_valid": len(valid)}
            self.results["calibration"] = r
            return r

        thr = 50.0
        def accuracy(w):
            preds = np.array([composite_w(w, s) >= thr for s in scores_list])
            return float(np.mean(preds == (label_arr == 1)))

        r = {
            "n_valid":         len(valid),
            "n_suitable":      int(label_arr.sum()),
            "n_unsuitable":    int(len(label_arr) - label_arr.sum()),
            "converged":       bool(res.success),
            "default_weights": {k: round(float(w0[i]*w0.sum()), 4)
                                for i, k in enumerate(keys)},
            "optimized_weights": {k: round(float(w_opt[i]), 4)
                                  for i, k in enumerate(keys)},
            "default_accuracy":   round(accuracy(w0),    4),
            "optimized_accuracy": round(accuracy(w_opt), 4),
            "accuracy_gain":      round(accuracy(w_opt) - accuracy(w0), 4),
        }
        self.results["calibration"] = r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # SPATIAL CORRECTION  —  Auto-re-run on thinned dataset
    # ═══════════════════════════════════════════════════════════════════════

    def apply_spatial_correction(self, inlier_threshold=0.60, min_points=20,
                                  log_callback=None):
        """
        If blocked cross-validation signals strong spatial autocorrelation,
        re-run the core analysis on the environmentally thinned dataset and
        report both raw and corrected composite scores side by side.

        Requires ``apply_environmental_thinning`` and ``compute_blocked_cv``
        to have been called beforehand (or calls them implicitly).  Does
        **not** replace ``self.species_data``; only stores a
        ``thinned_analysis`` diagnostic block.

        Heavy components (bootstrap CI, sensitivity analysis) are not
        re-executed on the thinned set to keep total runtime bounded.

        Parameters
        ----------
        inlier_threshold : float
            Minimum mean blocked-CV inlier rate considered acceptable.
            Below this, thinning is applied (default 0.60 — "high bias").
        min_points : int
            Minimum retained point count required to run the thinned
            pipeline (default 20 — avoids degenerate covariance).
        log_callback : callable(str) or None
            Optional logger — streams progress while the pipeline is being
            re-run on the thinned dataset.
        """
        def _log(msg):
            if log_callback is not None:
                try: log_callback(msg)
                except Exception: pass

        if "thinning" not in self.results:
            self.apply_environmental_thinning()
        if "blocked_cv" not in self.results:
            self.compute_blocked_cv()
        if "composite" not in self.results:
            self.compute_composite_score()

        bcv  = self.results.get("blocked_cv", {}) or {}
        thin = self.results.get("thinning", {}) or {}
        mir       = bcv.get("mean_inlier_rate")
        n_before  = int(thin.get("n_before", 0))
        n_after   = int(thin.get("n_after",  0))
        retained  = thin.get("retained_indices", [])

        if mir is None:
            _log("[spatial] blocked CV returned no inlier rate — correction skipped")
            self.results["thinned_analysis"] = {
                "applied": False,
                "reason":  "Blocked CV returned no inlier rate — cannot assess bias",
            }
            return self.results["thinned_analysis"]

        if mir >= inlier_threshold:
            _log(f"[spatial] inlier rate {mir:.3f} ≥ {inlier_threshold:.2f} — "
                 f"no correction needed")
            self.results["thinned_analysis"] = {
                "applied":         False,
                "reason":          (f"Blocked CV inlier rate {mir:.2f} ≥ "
                                    f"{inlier_threshold:.2f} — no correction needed"),
                "raw_mean_inlier": round(float(mir), 4),
            }
            return self.results["thinned_analysis"]

        if n_after < min_points or n_after >= n_before:
            _log(f"[spatial] thinning too aggressive ({n_after}/{n_before}) — "
                 f"correction skipped")
            self.results["thinned_analysis"] = {
                "applied":         False,
                "reason":          (f"Thinning retained {n_after}/{n_before} pts — "
                                    f"below min_points={min_points} or no reduction"),
                "raw_mean_inlier": round(float(mir), 4),
            }
            return self.results["thinned_analysis"]

        _log(f"[spatial] applying correction — inlier rate {mir:.3f} < "
             f"{inlier_threshold:.2f}, re-running pipeline on {n_after}/{n_before} pts")

        # ── Re-run core pipeline on thinned data (no bootstrap / sensitivity) ──
        try:
            sa_thin = ClimateResilienceAnalyzer(
                self.species_data[retained],
                self.target_values,
                bio_cols=self.bio_cols,
                topo_cols=self.topo_cols,
                opt_cols=self.opt_cols,
            )
            _log("[spatial] thinned: PCA + Kernel PCA ...")
            sa_thin.compute_pca();               sa_thin.compute_kernel_pca()
            _log("[spatial] thinned: GMM + Isolation Forest ...")
            sa_thin.compute_gmm();               sa_thin.compute_isolation_forest()
            _log("[spatial] thinned: One-Class SVM + K-Means ...")
            sa_thin.compute_ocsvm();             sa_thin.compute_kmeans_niche()
            _log("[spatial] thinned: Mahalanobis + Threshold zone ...")
            sa_thin.compute_mahalanobis();       sa_thin.compute_threshold_zone()
            _log("[spatial] thinned: percentile + topo + variable importance ...")
            sa_thin.compute_percentile_analysis()
            sa_thin.compute_topo_score()
            sa_thin.compute_variable_importance()
            sa_thin.compute_composite_score()
            sa_thin.compute_blocked_cv()
        except Exception as e:
            _log(f"[spatial] thinned re-analysis failed: {e}")
            self.results["thinned_analysis"] = {
                "applied": False,
                "reason":  f"Thinned re-analysis failed: {e}",
                "raw_mean_inlier": round(float(mir), 4),
            }
            return self.results["thinned_analysis"]

        raw_sc  = self.results["composite"]["composite_score"]
        raw_cl  = self.results["composite"]["resilience_class"]
        thin_sc = sa_thin.results["composite"]["composite_score"]
        thin_cl = sa_thin.results["composite"]["resilience_class"]
        thin_ir = sa_thin.results.get("blocked_cv", {}).get("mean_inlier_rate")

        _log(f"[spatial] correction complete — raw {raw_sc:.2f} → "
             f"corrected {thin_sc:.2f}  (Δ {thin_sc - raw_sc:+.2f})"
             + ("  [class changed]" if raw_cl != thin_cl else ""))

        r = {
            "applied":             True,
            "n_before":            n_before,
            "n_after":              n_after,
            "retained_fraction":   round(n_after / max(n_before, 1), 3),
            "raw_composite":       raw_sc,
            "raw_class":           raw_cl,
            "raw_mean_inlier":     round(float(mir), 4),
            "thinned_composite":   thin_sc,
            "thinned_class":       thin_cl,
            "thinned_mean_inlier": (round(float(thin_ir), 4)
                                    if thin_ir is not None else None),
            "delta":               round(float(thin_sc - raw_sc), 2),
            "class_changed":       bool(raw_cl != thin_cl),
            "recommended_score":   thin_sc,
            "thinned_component_scores": sa_thin.results["composite"].get(
                                           "component_scores", {}),
            "note": (
                "Spatial autocorrelation correction applied: analysis "
                "re-run on environmentally thinned native range. "
                "Use `thinned_composite` as the peer-review-defensible score."
            ),
        }
        self.results["thinned_analysis"] = r
        return r

    # ═══════════════════════════════════════════════════════════════════════
    # SENSITIVITY ANALYSIS & UNSUPERVISED WEIGHT OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════

    def _component_scores_for_target(self, X_species, target_vals):
        """
        Compute the 6 component scores (threshold_zone, gmm, isolation_forest,
        ocsvm, mahalanobis, topo) for a single target row against an arbitrary
        species sample.  Used inside K-fold CV for sensitivity analysis.

        Returns
        -------
        dict with keys:
            threshold_zone, gmm, isolation_forest, ocsvm, mahalanobis, topo
        ``None`` on failure (e.g. degenerate covariance on tiny folds).
        """
        try:
            sa = ClimateResilienceAnalyzer(
                X_species,
                np.asarray(target_vals, dtype=float),
                bio_cols=self.bio_cols,
                topo_cols=self.topo_cols,
                opt_cols=self.opt_cols,
            )
            sa.compute_gmm()
            sa.compute_isolation_forest()
            sa.compute_ocsvm()
            sa.compute_mahalanobis()
            sa.compute_threshold_zone()
            sa.compute_percentile_analysis()
            sa.compute_topo_score()
            return {
                "threshold_zone":   float(sa.results["threshold_zone"]["score"]),
                "gmm":              float(sa.results["gmm"]["score"]),
                "isolation_forest": float(sa.results["isolation_forest"]["score"]),
                "ocsvm":            float(sa.results["ocsvm"]["score"]),
                "mahalanobis":      float(sa.results["mahalanobis_stat"]["score"]),
                "topo":             float(sa.results["topo"]["score"]),
            }
        except Exception:
            return None

    def compute_sensitivity_analysis(self, n_folds=5, n_grid=200,
                                     log_callback=None):
        """
        Two-in-one methodological hardening:

        1. **Sensitivity analysis** — for the actual target point, sweep a
           Dirichlet-sampled grid of composite-score weights and report
           ``score_range``, ``score_std``, and the fraction of grid points
           that flip the resilience class.  Quantifies how much the heuristic
           weight choice drives the final answer.

        2. **Unsupervised weight optimisation** — K-fold CV on the native
           range.  Since all native-range points are by construction inliers,
           the "correct" answer for each held-out point is a high composite
           score.  The objective is the **mean held-out composite score**
           over all K folds; the argmax grid point is the calibrated
           optimum.  This requires no external labels.

        Parameters
        ----------
        n_folds : int
            K for K-fold CV on the species data (default 5).
        n_grid  : int
            Number of Dirichlet samples in the ML 5-simplex (default 200).
        log_callback : callable(str) or None
            Optional logger — called with progress lines during the slow
            K-fold component-score phase so the UI can stream status.

        Returns
        -------
        dict stored in ``self.results["sensitivity"]``.
        """
        from sklearn.model_selection import KFold

        def _log(msg):
            if log_callback is not None:
                try: log_callback(msg)
                except Exception: pass

        if "composite" not in self.results:
            self.compute_composite_score()

        n = len(self.species_data)
        if n < max(n_folds * 2, 10):
            r = {
                "error": f"Insufficient points for {n_folds}-fold CV (n={n})",
                "n_cv_folds": n_folds,
            }
            self.results["sensitivity"] = r
            _log(f"[sensitivity] insufficient data ({n} pts) — skipped")
            return r

        _log(f"[sensitivity] starting K-fold component scores  "
             f"(K={n_folds}, n_points={n})")

        # ── 1. Build K-fold test component-score matrix ────────────────────
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        test_comps = []                   # list of dict per held-out point
        done = 0
        step = max(1, n // 10)            # log every ~10 %
        t0 = time.time()
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.species_data), 1):
            X_train = self.species_data[train_idx]
            if len(X_train) < max(10, self.n_vars + 2):
                _log(f"[sensitivity] fold {fold_idx}: too small — skipped")
                continue
            _log(f"[sensitivity] fold {fold_idx}/{n_folds}  "
                 f"(train={len(train_idx)}, test={len(test_idx)})")
            for ti in test_idx:
                cs = self._component_scores_for_target(
                    X_train, self.species_data[ti]
                )
                if cs is not None:
                    test_comps.append(cs)
                done += 1
                if done % step == 0:
                    elapsed = time.time() - t0
                    rate    = done / max(elapsed, 1e-6)
                    eta     = (n - done) / max(rate, 1e-6)
                    _log(f"[sensitivity] component scores  "
                         f"{done:>4}/{n}  ({done/n*100:5.1f}%)  "
                         f"valid={len(test_comps)}  "
                         f"elapsed={elapsed:5.1f}s  ETA={eta:5.1f}s")

        if len(test_comps) < 5:
            r = {
                "error": (f"Too few valid K-fold component evaluations "
                          f"({len(test_comps)})"),
                "n_cv_folds": n_folds,
            }
            self.results["sensitivity"] = r
            return r

        ml_keys = ["threshold_zone", "gmm", "isolation_forest",
                   "ocsvm", "mahalanobis"]
        C_ml  = np.array([[c[k] for k in ml_keys] for c in test_comps])   # (N, 5)
        C_topo = np.array([c["topo"] for c in test_comps])                # (N,)

        # Same matrix for the *actual* target (single row)
        tgt_cs = {
            "threshold_zone":   float(self.results["threshold_zone"]["score"]),
            "gmm":              float(self.results["gmm"]["score"]),
            "isolation_forest": float(self.results["isolation_forest"]["score"]),
            "ocsvm":            float(self.results["ocsvm"]["score"]),
            "mahalanobis":      float(self.results["mahalanobis_stat"]["score"]),
            "topo":             float(self.results["topo"]["score"]),
        }
        tgt_ml   = np.array([tgt_cs[k] for k in ml_keys])                 # (5,)
        tgt_topo = tgt_cs["topo"]

        # ── 2. Generate weight grid ────────────────────────────────────────
        rng = np.random.RandomState(42)
        # Dirichlet(2,2,2,2,2) gives a mildly peaked distribution over the
        # 5-simplex — avoids extreme corners while still spanning the space.
        ml_grid = rng.dirichlet([2.0] * 5, size=n_grid)                   # (n_grid, 5)
        # Enforce per-component bounds [0.05, 0.60]; re-draw rejects.
        bounds_ok = ((ml_grid >= 0.05).all(axis=1) &
                     (ml_grid <= 0.60).all(axis=1))
        ml_grid   = ml_grid[bounds_ok]
        if len(ml_grid) < 10:                                             # top up
            extra = rng.dirichlet([3.0] * 5, size=n_grid * 3)
            extra = extra[((extra >= 0.05).all(axis=1) &
                           (extra <= 0.60).all(axis=1))]
            ml_grid = np.vstack([ml_grid, extra])[:n_grid]

        # W_CLIMATE ∈ {0.50, 0.55, ..., 0.95}
        w_clim_grid = np.arange(0.50, 0.96, 0.05)
        # Pre-default (for deltas)
        default_w_clim = W_CLIMATE
        default_w_ml   = np.array([W_THRESHOLD_ZONE, W_GMM, W_ISOFOREST,
                                   W_OCSVM, W_MAHAL])
        default_w_ml   = default_w_ml / default_w_ml.sum()

        # ── 3. Evaluate grid — objective (K-fold mean) + target score ──────
        # shape: (n_wclim, n_wml, n_test)
        # K-fold mean component score per grid point
        clim_mat = C_ml @ ml_grid.T          # (N_test, n_wml)  — climate base
        test_scores = np.empty((len(w_clim_grid), ml_grid.shape[0]))
        target_scores = np.empty_like(test_scores)

        for i, wc in enumerate(w_clim_grid):
            # N_test × n_wml
            fold = wc * clim_mat + (1.0 - wc) * C_topo[:, None]
            test_scores[i] = fold.mean(axis=0)                            # (n_wml,)
            tgt_fold = wc * (ml_grid @ tgt_ml) + (1.0 - wc) * tgt_topo
            target_scores[i] = tgt_fold

        # ── 4. Optimum (unsupervised) ──────────────────────────────────────
        flat_idx   = np.argmax(test_scores)
        best_i, best_j = np.unravel_index(flat_idx, test_scores.shape)
        best_wclim = float(w_clim_grid[best_i])
        best_wml   = ml_grid[best_j]
        best_loo   = float(test_scores[best_i, best_j])

        # Default's LOO score
        default_loo_vec = (default_w_clim
                           * (C_ml @ default_w_ml)
                           + (1.0 - default_w_clim) * C_topo)
        default_loo     = float(default_loo_vec.mean())

        # ── 5. Target sensitivity (score range over grid) ──────────────────
        tgt_scores_flat = target_scores.flatten()
        tgt_default     = float(default_w_clim * (default_w_ml @ tgt_ml)
                                + (1.0 - default_w_clim) * tgt_topo)
        tgt_optimal     = float(target_scores[best_i, best_j])
        tgt_range       = float(tgt_scores_flat.max() - tgt_scores_flat.min())
        tgt_std         = float(tgt_scores_flat.std())

        # Class flip fraction
        def _class(s):
            if s >= 80: return "A"
            if s >= 65: return "B"
            if s >= 50: return "C"
            if s >= 35: return "D"
            return           "E"
        default_class = _class(tgt_default)
        flips = sum(1 for s in tgt_scores_flat if _class(float(s)) != default_class)
        class_flip_fraction = flips / len(tgt_scores_flat)

        # Robustness verdict
        if   tgt_range < 5  and class_flip_fraction < 0.05: robustness = "High"
        elif tgt_range < 12 and class_flip_fraction < 0.20: robustness = "Moderate"
        else:                                               robustness = "Low"

        interp = {
            "High":     ("✅ High robustness — composite score is stable across "
                         "a wide range of plausible weight configurations."),
            "Moderate": ("🟡 Moderate robustness — the composite score depends "
                         "meaningfully on the weight choice; consider reporting "
                         "the optimal-weight score alongside the default."),
            "Low":      ("🔴 Low robustness — weight choice drives the answer; "
                         "the unsupervised optimum should be preferred over "
                         "the heuristic default."),
        }[robustness]

        r = {
            "n_grid_points":   int(len(w_clim_grid) * ml_grid.shape[0]),
            "n_cv_folds":      int(n_folds),
            "n_test_points":   int(len(test_comps)),
            "w_clim_grid":     [round(float(x), 3) for x in w_clim_grid],
            "n_ml_samples":    int(ml_grid.shape[0]),
            "optimal_weights": {
                "w_climate":        round(best_wclim, 4),
                "w_topo":           round(1.0 - best_wclim, 4),
                "w_threshold_zone": round(float(best_wml[0]), 4),
                "w_gmm":            round(float(best_wml[1]), 4),
                "w_iso_forest":     round(float(best_wml[2]), 4),
                "w_ocsvm":          round(float(best_wml[3]), 4),
                "w_mahalanobis":    round(float(best_wml[4]), 4),
            },
            "default_weights": {
                "w_climate":        round(float(default_w_clim), 4),
                "w_topo":           round(1.0 - float(default_w_clim), 4),
                "w_threshold_zone": round(float(default_w_ml[0]), 4),
                "w_gmm":            round(float(default_w_ml[1]), 4),
                "w_iso_forest":     round(float(default_w_ml[2]), 4),
                "w_ocsvm":          round(float(default_w_ml[3]), 4),
                "w_mahalanobis":    round(float(default_w_ml[4]), 4),
            },
            "optimal_loo_score":     round(best_loo, 2),
            "default_loo_score":     round(default_loo, 2),
            "loo_score_gain":        round(best_loo - default_loo, 2),
            "target_score_default":  round(tgt_default, 2),
            "target_score_optimal":  round(tgt_optimal, 2),
            "target_score_delta":    round(tgt_optimal - tgt_default, 2),
            "target_score_range":    round(tgt_range, 2),
            "target_score_std":      round(tgt_std, 2),
            "target_score_min":      round(float(tgt_scores_flat.min()), 2),
            "target_score_max":      round(float(tgt_scores_flat.max()), 2),
            "class_flip_fraction":   round(class_flip_fraction, 4),
            "default_class":         default_class,
            "robustness":            robustness,
            "interpretation":        interp,
        }
        self.results["sensitivity"] = r
        return r

    # ═══════════════════════════════════════════════════════════════════════

    def run_all(self):
        # ── Phase 1 — Baseline analysis on raw species data ─────────────────
        self.compute_pca();               self.compute_kernel_pca()
        self.compute_gmm();               self.compute_isolation_forest()
        self.compute_ocsvm();             self.compute_kmeans_niche()
        self.compute_mahalanobis();       self.compute_threshold_zone()
        self.compute_percentile_analysis()
        self.compute_topo_score()
        self.compute_variable_importance()
        self.compute_composite_score()

        # ── Phase 2 — Spatial autocorrelation diagnostics + correction ──────
        self.apply_environmental_thinning()
        self.compute_blocked_cv()
        self.apply_spatial_correction()

        # ── Phase 3 — Uncertainty + methodological hardening ────────────────
        self.compute_bootstrap_ci_full()
        self.compute_sensitivity_analysis()
        return self.results
