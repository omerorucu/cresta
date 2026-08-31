"""
CRESTA Analysis Engine  v2.0  (scientifically revised)
=======================================================
Bioclimatic niche-similarity and (optionally) climate-change response
assessment for plant species, based on presence-only occurrence records.

WHAT THIS ENGINE DOES
---------------------
Given (a) the environmental values at a species' occurrence records and
(b) the environmental values of a target site, it tests the hypothesis

    H0 : the target site is exchangeable with the species' occurrence records
         (i.e. the site lies inside the species' realised environmental niche)

and returns a *calibrated* p-value for that hypothesis, plus a 0-100 score
that is a strictly monotone re-expression of it.  If future-climate values
for the target site are also supplied, the same statistic is computed under
the future climate and the difference (DeltaScore) is reported.

WHY IT IS BUILT THIS WAY  (v1.0 -> v2.0 changes)
------------------------------------------------
v1.0 combined several model outputs through hand-tuned monotone transforms
(sigmoids, power laws, piecewise ramps) whose parameters were chosen so the
output "looked right".  The resulting 0-100 number had no probabilistic
meaning and was badly calibrated: in testing, 50 % of a species' OWN
occurrence records scored below class B and 10 % were reported as
"not suitable for this site".

v2.0 replaces all of that with cross-conformal prediction
(Vovk, Gammerman & Shafer 2005; Vovk 2015).  A single ensemble
non-conformity statistic is computed, and its value for the target is
ranked against out-of-fold values for the occurrence records.  Under
exchangeability the resulting p-value is Uniform(0,1) by construction, so
the score is calibrated by design and the class thresholds have an exact
frequentist meaning (see SCORE_BANDS below).  The engine additionally
reports the empirical calibration achieved on the user's own data, so the
assumption can be checked rather than assumed.

METHOD SUMMARY
--------------
1.  Feature construction   - aspect encoded circularly as (northness,
    eastness) EVERYWHERE, including the Mahalanobis stage (v1.0 fed raw
    degrees into the covariance, so 359 deg and 1 deg were 358 units apart).
2.  Collinearity control   - Bio1-19 are strongly collinear.  A correlation
    filter (|r| >= corr_threshold) and a Ledoit-Wolf shrinkage covariance
    (Ledoit & Wolf 2004) replace the raw sample covariance + pseudo-inverse,
    which in v1.0 produced NaN or ~1e7 distances whenever n <= p.
3.  Non-conformity ensemble - Mahalanobis distance, negative GMM
    log-likelihood, negative Isolation-Forest score, negative One-Class-SVM
    decision value, each standardised on the fit fold and averaged with
    equal weights (no hand-tuned weights; a sensitivity analysis over
    alternative weightings is reported).
4.  Cross-conformal calibration - K-fold (spatially blocked when
    coordinates are supplied, per Roberts et al. 2017).  Every model is
    fitted without the record it scores, so occurrence records and the
    target are on equal footing.
5.  Validation - Kolmogorov-Smirnov test of p-value uniformity, empirical
    type-I error at alpha = 0.05 / 0.10, class distribution of the species'
    own records, and a permutation null-model discrimination AUC
    (Raes & ter Steege 2007).
6.  Uncertainty - percentile bootstrap CI on the score (calibration-set
    bootstrap plus optional model-refit bootstrap).
7.  Extrapolation - MESS / multivariate environmental similarity surface
    (Elith, Kearney & Phillips 2010) flags target sites outside the
    training range, where any model output is an extrapolation.

WHAT THIS ENGINE DOES NOT DO
----------------------------
* It does not estimate the FUNDAMENTAL niche.  Occurrence records describe
  the realised niche, which is truncated by dispersal limitation, biotic
  interactions and sampling bias (Soberon & Nakamura 2009).
* It does not correct sampling bias beyond optional spatial blocking of
  the CV folds; spatial thinning of the input records is the user's job
  (Aiello-Lammens et al. 2015).
* Without future-climate input it performs SITE MATCHING under the current
  climate, not a climate-change assessment.  The class label and the wording
  of the recommendation change accordingly.

REFERENCES
----------
Aiello-Lammens ME et al. (2015) spThin. Ecography 38:541-545.
Barbet-Massin M et al. (2012) Selecting pseudo-absences. Methods Ecol Evol 3:327-338.
Elith J, Kearney M, Phillips S (2010) The art of modelling ranging species. Methods Ecol Evol 1:330-342.
Ledoit O, Wolf M (2004) A well-conditioned estimator for large-dimensional covariance matrices. J Multivar Anal 88:365-411.
Raes N, ter Steege H (2007) A null-model for significance testing of presence-only species distribution models. Ecography 30:727-736.
Roberts DR et al. (2017) Cross-validation strategies for data with spatial, temporal, phylogenetic or hierarchical structure. Ecography 40:913-929.
Soberon J, Nakamura M (2009) Niches and distributional areas. PNAS 106:19644-19650.
Vovk V, Gammerman A, Shafer G (2005) Algorithmic Learning in a Random World. Springer.
Vovk V (2015) Cross-conformal predictors. Ann Math Artif Intell 74:9-28.
Monteith JL, Unsworth MH (2013) Principles of Environmental Physics, 4th ed. (Tetens equation).

Author : Omer K. Orucu   License: GPL-3.0
"""

import warnings
import numpy as np

from scipy.stats import percentileofscore, kstest
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import KFold


# ==============================================================================
#  CONSTANTS
# ==============================================================================

BIO_NAMES = {
    "bio1":  "Annual Mean Temperature",
    "bio2":  "Mean Diurnal Temperature Range",
    "bio3":  "Isothermality (%)",
    "bio4":  "Temperature Seasonality (SD x100)",
    "bio5":  "Max Temperature of Warmest Month",
    "bio6":  "Min Temperature of Coldest Month",
    "bio7":  "Annual Temperature Range",
    "bio8":  "Mean Temp of Wettest Quarter",
    "bio9":  "Mean Temp of Driest Quarter",
    "bio10": "Mean Temp of Warmest Quarter",
    "bio11": "Mean Temp of Coldest Quarter",
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
    "aspect_deg":  "Aspect (deg, 0/360 = North)",
}
OPT_VAR_NAMES = {
    "srad": "Mean Daily Solar Radiation (kJ m⁻² day⁻¹)",
    "wind": "Mean Wind Speed (m s⁻¹)",
    "vapr": "Water Vapour Pressure (kPa)",
}
ALL_VAR_NAMES = {**BIO_NAMES, **TOPO_NAMES, **OPT_VAR_NAMES}

ALL_BIO_COLS  = [f"bio{i}" for i in range(1, 20)]
ALL_TOPO_COLS = ["elevation_m", "slope_pct", "aspect_deg"]
ALL_OPT_COLS  = ["srad", "wind", "vapr"]

# Backward-compatible aliases (imported by main_dialog)
BIO_COLS  = ALL_BIO_COLS
TOPO_COLS = ALL_TOPO_COLS
ALL_COLS  = BIO_COLS + TOPO_COLS

# Temperature-scaled WorldClim variables (v1 stores these as degC x 10)
TEMP_BIOS = {"bio1", "bio2", "bio5", "bio6", "bio7", "bio8",
             "bio9", "bio10", "bio11"}

CRITICAL_BIOS = {"bio4", "bio5", "bio6", "bio14", "bio15", "bio17"}
CRITICAL_OPT  = {"srad", "vapr"}

# --- Score bands -------------------------------------------------------------
# The composite score is a piecewise-linear, strictly monotone map of the
# cross-conformal p-value.  Consequence, provable and testable:
#   under H0 (target really is an in-niche site)
#     P(class A)      = 0.80      P(class A or B) = 0.90
#     P(class >= C)   = 0.95      P(class E)      = 0.01
# i.e. the type-I error of declaring an in-niche site "unsuitable" is
# controlled at exactly 1 %.  v1.0 had no such guarantee and measured ~10 %.
SCORE_BANDS = [   # (p_low, p_high, score_low, score_high)
    (0.20, 1.00, 80.0, 100.0),
    (0.10, 0.20, 65.0,  80.0),
    (0.05, 0.10, 50.0,  65.0),
    (0.01, 0.05, 35.0,  50.0),
    (0.00, 0.01,  0.0,  35.0),
]

# Equal weights: no hand tuning.  Reported together with a sensitivity
# analysis over alternative weightings (compute_weight_sensitivity).
ENSEMBLE_WEIGHTS = {"mahalanobis": 0.25, "gmm": 0.25,
                    "isolation_forest": 0.25, "ocsvm": 0.25}

# Topographic sub-weights, used ONLY for the descriptive topo-compatibility
# figure.  The composite itself no longer applies arbitrary climate/topo
# weights - topography enters the unified non-conformity statistic directly.
W_ELEV, W_SLOPE, W_ASPECT = 0.45, 0.25, 0.30

# Retained as module attributes for backward compatibility with v1 importers.
W_CLIMATE, W_TOPO = 0.75, 0.25
W_THRESHOLD_ZONE = 0.0
W_GMM = W_ISOFOREST = W_OCSVM = W_MAHAL = 0.25


# ==============================================================================
#  SMALL HELPERS
# ==============================================================================

def northness(deg): return float(np.cos(np.deg2rad(float(deg))))
def eastness(deg):  return float(np.sin(np.deg2rad(float(deg))))


def circular_mean_deg(degrees):
    """Circular (vector) mean of angles, in degrees on [0, 360).

    v1.0 aggregated multi-cell target areas with a plain arithmetic mean, so
    two north-facing cells at 350 deg and 10 deg averaged to 180 deg (due
    south).  This is the correct estimator.
    """
    a = np.deg2rad(np.asarray(degrees, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    m = np.arctan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
    d = float(np.rad2deg(m) % 360.0)
    return 0.0 if d > 359.9995 else d


def detect_temp_scale(values, col):
    """Detect whether a temperature variable is stored as degC or degC x 10.

    WorldClim v1 distributes temperatures as integers scaled by 10; WorldClim
    v2 and CHELSA distribute degrees Celsius directly.  v1.0 of this engine
    assumed the x10 convention unconditionally, which silently corrupted every
    threshold statement it made about v2 / CHELSA input.

    Returns (divisor, label).
    """
    if col not in TEMP_BIOS:
        return 1.0, ""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0, "degC"
    # Any terrestrial bioclim temperature expressed in degC sits inside -60..60.
    if float(np.median(np.abs(v))) > 60.0:
        return 10.0, "°C×10"
    return 1.0, "°C"


def saturation_vapour_pressure_kpa(t_celsius):
    """Tetens saturation vapour pressure, kPa.

    v1.0 estimated VPD as (2.34 - vapr): it hard-coded the saturation vapour
    pressure of air at 20 degC and applied it to every site on Earth.
    """
    t = float(t_celsius)
    return 0.6108 * float(np.exp(17.27 * t / (t + 237.3)))


def _finite(x, default=0.0):
    """Replace non-finite values so they can never reach a user-facing class."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def score_from_pvalue(p):
    """Piecewise-linear, continuous, strictly monotone map p -> [0, 100]."""
    if p is None or not np.isfinite(p):
        return float("nan")
    p = float(np.clip(p, 0.0, 1.0))
    for p_lo, p_hi, s_lo, s_hi in SCORE_BANDS:
        if p_lo <= p <= p_hi:
            span = (p_hi - p_lo) or 1.0
            return float(np.clip(s_lo + (s_hi - s_lo) * (p - p_lo) / span, 0, 100))
    return 0.0


def conformal_pvalue(calibration_alphas, target_alpha):
    """Standard conformal p-value: (1 + #{a_i >= a_t}) / (n + 1).

    Valid (Uniform-dominating in finite samples) whenever the calibration
    scores and the target score are exchangeable, which is what the
    cross-fitting in _crossfit_alphas() buys us.
    """
    a = np.asarray(calibration_alphas, dtype=float)
    a = a[np.isfinite(a)]
    t = float(target_alpha) if target_alpha is not None else np.nan
    if a.size == 0 or not np.isfinite(t):
        return float("nan")
    return float((1.0 + np.sum(a >= t)) / (a.size + 1.0))


# ==============================================================================
#  RISK EXPLANATION SYSTEM
# ==============================================================================
#
#  v1.0 emitted species-agnostic physiological assertions ("in C3 plants
#  >350 degC x10 enzyme activity collapses", "VPD > 2 kPa triggers stomatal
#  closure") with no sources, identical for a succulent and a boreal conifer,
#  and presented them as if they were findings about the analysed species.
#
#  v2.0 keeps the quantitative, data-derived part (position relative to the
#  occurrence distribution, deviation, distance to the P5/P95 bounds) as the
#  primary content, and separates the mechanistic part into short, explicitly
#  SOURCED and explicitly GENERIC statements carrying a standing disclaimer.
#  Nothing here is an inference about the analysed species.
# ==============================================================================

MECHANISM_DISCLAIMER = (
    "NOTE: the mechanism below is a general, literature-sourced statement "
    "about plants in the cited functional context. It is NOT an inference "
    "about the analysed species, whose actual tolerance may differ widely. "
    "Only the numerical position relative to this species' own occurrence "
    "records is data-derived."
)

# col -> (high_extreme_text, low_extreme_text, source)
GENERIC_MECHANISMS = {
    "bio1": (
        "C3 photosynthesis peaks at roughly 15-30 degC and declines steeply "
        "above about 35 degC as Rubisco activase is inactivated and "
        "photorespiration rises; C4 optima sit higher.",
        "A cooler annual mean shortens the thermally suitable growing period "
        "and can decouple phenology from the local season.",
        "Sage & Kubien 2007, Plant Cell Environ 30:1086-1106"),
    "bio4": (
        "Greater seasonality means larger summer-winter amplitude and a higher "
        "joint frequency of frost and heat events; damage depends on whether "
        "the species has a matching dormancy strategy.",
        "Very low seasonality may fail to supply the temperature cues that "
        "trigger dormancy and bud break in temperate species.",
        "Luedeling 2012, Sci Hortic 144:218-229"),
    "bio5": (
        "Sustained high maximum temperature pushes leaves past their thermal "
        "optimum; combined with stomatal closure this can drive carbon balance "
        "negative and increase the risk of hydraulic failure.",
        "Insufficient summer warmth limits cumulative carbon gain over the "
        "growing season.",
        "Sage & Kubien 2007, Plant Cell Environ 30:1086-1106; Choat et al. 2012, Nature 491:752-755"),
    "bio6": (
        "Unusually mild winters may not accumulate the chilling required for "
        "dormancy release, which can disrupt bud break, flowering and fruit set "
        "in temperate species.",
        "Below a species' freezing tolerance, ice formation in tissues causes "
        "membrane and vascular injury; the threshold is highly species-specific.",
        "Pearce 2001, Ann Bot 87:417-424; Luedeling 2012, Sci Hortic 144:218-229"),
    "bio12": (
        "Persistent excess water reduces soil oxygen; root anoxia impairs "
        "nutrient uptake and predisposes plants to root pathogens.",
        "When potential evapotranspiration exceeds precipitation, declining "
        "water potential moves xylem towards its cavitation threshold.",
        "Kozlowski 1997, Tree Physiol Monograph 1; Choat et al. 2012, Nature 491:752-755"),
    "bio14": (
        "A wetter dry month generally relaxes the seasonal water constraint.",
        "A drier driest month lengthens the period of stomatal closure and "
        "near-zero net assimilation, and can suppress cambial activity.",
        "Choat et al. 2012, Nature 491:752-755"),
    "bio15": (
        "Highly irregular rainfall increases the chance of both drought and "
        "flood pulses within a single year, which is a distinct stressor from "
        "the annual total.",
        "Very evenly distributed rainfall relaxes the selection pressure for "
        "drought-avoidance traits.",
        "Knapp et al. 2008, BioScience 58:811-821"),
    "bio17": (
        "A wetter driest quarter relaxes the seasonal water constraint.",
        "A pronounced seasonal water deficit slows growth and cambial activity; "
        "in trees this typically registers as narrower annual rings.",
        "Choat et al. 2012, Nature 491:752-755"),
    "elevation_m": (
        "Air temperature falls with elevation at roughly 0.5-0.7 degC per 100 m "
        "and the thermally defined growing season shortens correspondingly.",
        "At lower elevation the same species faces higher heat load and "
        "evaporative demand than at its typical altitude.",
        "Korner 2007, Trends Ecol Evol 22:569-574; Barry & Chorley, Atmosphere, Weather and Climate"),
    "slope_pct": (
        "Steeper slopes generally hold thinner soils, less plant-available "
        "water and higher runoff.",
        "Flat ground drains more slowly, which raises waterlogging risk for "
        "species adapted to well-drained slopes.",
        "general soil-geomorphology relationship; no single source"),
    "aspect_deg": (
        "Aspect modulates received radiation and evaporative demand; the "
        "relevant comparison is circular, not linear.",
        "Aspect modulates received radiation and evaporative demand; the "
        "relevant comparison is circular, not linear.",
        "general microclimate relationship"),
    "srad": (
        "Radiation in excess of the photosynthetic light saturation point does "
        "not increase assimilation but does raise leaf temperature and the risk "
        "of photosystem II photoinhibition.",
        "Below light saturation, carbon assimilation is light-limited.",
        "Murata et al. 2007, Biochim Biophys Acta 1767:414-421"),
    "wind": (
        "Higher wind raises boundary-layer conductance and transpiration, and "
        "adds mechanical loading that can cause fatigue damage.",
        "Very still air reduces evaporative demand and can limit wind "
        "pollination.",
        "Gardiner et al. 2016, Plant Science 245:94-118"),
    "vapr": (
        "Higher vapour pressure at a given temperature means lower VPD and "
        "lower evaporative demand.",
        "Lower vapour pressure at a given temperature means higher VPD; "
        "stomatal conductance declines roughly hyperbolically with rising VPD, "
        "reducing assimilation before soil water is depleted.",
        "Grossiord et al. 2020, New Phytologist 226:1550-1566"),
}


def build_risk_explanation(col, tgt, med, p5, p25, p75, p95,
                           pct, score, sp_min, sp_max, sp_std,
                           unit_divisor=1.0, unit_label="", context=None):
    """Per-variable numerical position report plus a sourced generic mechanism.

    Parameters
    ----------
    unit_divisor, unit_label : from detect_temp_scale(); used so that a
        WorldClim v2 / CHELSA input in degC is not mislabelled as degC x 10.
    context : dict, optional
        Extra derived quantities, currently {'mean_temp_c': float} used to
        compute VPD correctly from the Tetens equation instead of assuming
        20 degC air as v1.0 did.
    """
    context = context or {}

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
    dev_pct = (dev_abs / (abs(med) + 1e-9)) * 100.0
    n_std   = dev_abs / (sp_std + 1e-9)
    d5      = tgt - p5
    d95     = p95 - tgt

    if tgt < p5:
        pos = (f"{abs(d5):.4g} units BELOW the P5 lower bound "
               f"(P{pct:.0f} percentile) - outside the observed tolerance band")
    elif tgt < p25:
        pos = (f"Lower tolerance band (P{pct:.0f} percentile), "
               f"{p25 - tgt:.4g} units from the P25 core boundary")
    elif tgt <= p75:
        pos = (f"Within the core band (P{pct:.0f} percentile), "
               f"{abs(dev_abs):.4g} units from the median")
    elif tgt <= p95:
        pos = (f"Upper tolerance band (P{pct:.0f} percentile), "
               f"{tgt - p75:.4g} units above the P75 core boundary")
    else:
        pos = (f"{abs(d95):.4g} units ABOVE the P95 upper bound "
               f"(P{pct:.0f} percentile) - outside the observed tolerance band")

    # Two-sided exact rank test of "target is a draw from the occurrence
    # distribution of this variable".  v1.0 reported a Mann-Whitney U p-value
    # computed from a one-element sample, which carries no information beyond
    # the rank itself and was labelled as if it were an inferential test.
    rank_p = float(np.clip(2.0 * min(pct, 100.0 - pct) / 100.0, 0.0, 1.0))

    unit_note = ""
    if col in TEMP_BIOS and unit_label:
        unit_note = (f"  Detected unit  : {unit_label}"
                     + (f"   (= {tgt / unit_divisor:.2f} °C)"
                        if unit_divisor != 1.0 else "") + "\n")

    mech = _mechanism_text(col, tgt, p5, p95, unit_divisor, context)
    sep = "─" * 58
    explanation = (
        f"{risk_emoji} {risk_level} RISK  —  {ALL_VAR_NAMES.get(col, col)}\n"
        f"{sep}\n"
        f"  Target value   : {tgt:.4g}  (median: {med:.4g}, sd: {sp_std:.4g})\n"
        f"{unit_note}"
        f"  Species range  : [{sp_min:.4g} – {sp_max:.4g}]\n"
        f"  Thresholds     : P5={p5:.4g}  P25={p25:.4g}  "
        f"P75={p75:.4g}  P95={p95:.4g}\n"
        f"  Position       : {pos}\n"
        f"  Deviation      : {dev_abs:+.4g} units  ({dev_pct:+.1f}%)  "
        f"/  {n_std:+.2f} sd\n"
        f"  Rank test      : two-sided exact rank p = {rank_p:.4f}\n"
        f"  Marginal score : {score:.1f} / 100   "
        f"(single-variable typicality; NOT the composite)\n"
        f"{sep}\n"
        f"  Generic mechanism (see note):\n  {mech}\n"
        f"{sep}\n"
        f"  {MECHANISM_DISCLAIMER}"
    )

    return {
        "risk_level": risk_level, "risk_color": risk_color,
        "risk_emoji": risk_emoji, "in_core": in_core, "in_tolerance": in_tol,
        "dev_from_med": round(dev_abs, 4), "dev_pct": round(dev_pct, 2),
        "n_std": round(n_std, 3), "dist_to_p5": round(d5, 4),
        "dist_to_p95": round(d95, 4), "rank_pvalue": round(rank_p, 4),
        "unit_label": unit_label, "explanation": explanation,
    }


def _mechanism_text(col, tgt, p5, p95, unit_divisor, context):
    """Return the sourced generic mechanism for the relevant tail."""
    entry = GENERIC_MECHANISMS.get(col)
    if entry is None:
        if tgt > p95:
            return "Above the upper tolerance bound observed for this species."
        if tgt < p5:
            return "Below the lower tolerance bound observed for this species."
        return "Within the tolerance band observed for this species."

    high_txt, low_txt, source = entry

    if col == "vapr":
        # Correct VPD from the Tetens equation at the site's own mean annual
        # temperature, when bio1 is available.  v1.0 used a constant 2.34 kPa,
        # i.e. it assumed 20 degC air everywhere.
        t_c = context.get("mean_temp_c")
        if t_c is not None and np.isfinite(t_c):
            es = saturation_vapour_pressure_kpa(t_c)
            vpd = max(0.0, es - float(tgt))
            extra = (f" Estimated VPD at the site's mean annual temperature "
                     f"({t_c:.1f} °C, e_s = {es:.2f} kPa) is {vpd:.2f} kPa.")
        else:
            extra = (" VPD cannot be estimated because bio1 was not selected; "
                     "vapour pressure alone does not determine evaporative demand.")
    else:
        extra = ""

    if tgt > p95:
        body = high_txt
    elif tgt < p5:
        body = low_txt
    else:
        body = ("Within the tolerance band observed for this species; "
                "the mechanism below becomes relevant only outside it. " + high_txt)

    return f"{body}{extra}  [{source}]"


# ==============================================================================
#  MAIN CLASS
# ==============================================================================

class InsufficientDataError(ValueError):
    """Raised when the sample cannot support the requested variable set.

    v1.0 accepted n = 15 occurrence records with 22 variables (its own default
    minimum), produced a singular covariance, silently fell back to a
    pseudo-inverse, and returned NaN - which its classifier then reported to
    the user as 'E - Very Low Resilience / This species is not suitable for
    this site'.  Refusing to run is the correct behaviour.
    """


class ClimateResilienceAnalyzer:
    """Conformal niche-similarity analyser.

    Parameters
    ----------
    species_data : ndarray (n, n_cols)
        Environmental values at the species' occurrence records.
        Column order must be  bio_cols + topo_cols + opt_cols.
    target_values : ndarray (n_cols,)
        Environmental values of the target site under the CURRENT climate.
    bio_cols, topo_cols, opt_cols : list[str]
        Selected variables; at least one bioclimatic variable is required.
    target_future : ndarray (n_cols,), optional
        Same variables for the target site under a FUTURE climate scenario
        (e.g. a CMIP6 SSP).  Supplying it turns the analysis from static site
        matching into an actual climate-change response assessment and
        populates results['climate_change'].
    coords : ndarray (n, 2), optional
        x/y (or lon/lat) of the occurrence records.  When given, the
        cross-validation folds are spatial blocks rather than random, which
        removes the optimism caused by spatial autocorrelation between
        neighbouring records (Roberts et al. 2017).
    n_folds : int
        Number of cross-conformal folds.
    corr_threshold : float
        |r| above which one member of a correlated pair is dropped from the
        MULTIVARIATE stage only.  Per-variable reporting always uses every
        selected variable.
    n_bootstrap : int
        Calibration-set bootstrap replicates for the score CI.
    n_refit_bootstrap : int
        Full model-refit bootstrap replicates (slower; captures fit
        uncertainty as well as calibration uncertainty).  0 disables it.
    """

    VERSION = "2.0"

    def __init__(self, species_data, target_values,
                 bio_cols=None, topo_cols=None, opt_cols=None,
                 target_future=None, target_cells=None, coords=None,
                 n_folds=5, random_state=42,
                 corr_threshold=0.95, prune_collinear=True,
                 n_bootstrap=400, n_refit_bootstrap=10,
                 compute_subspaces=True):

        # ---- selected columns -------------------------------------------
        self.bio_cols  = [c for c in (bio_cols  or ALL_BIO_COLS) if c in BIO_NAMES]
        self.topo_cols = [c for c in (topo_cols or [])           if c in TOPO_NAMES]
        self.opt_cols  = [c for c in (opt_cols  or [])           if c in OPT_VAR_NAMES]
        if not self.bio_cols:
            raise ValueError("At least one bioclimatic variable must be selected.")

        self.has_topo   = len(self.topo_cols) > 0
        self.has_aspect = "aspect_deg" in self.topo_cols
        self.col_names  = self.bio_cols + self.topo_cols + self.opt_cols
        self.n_vars     = len(self.col_names)
        self.n_bio      = len(self.bio_cols)
        self.n_topo     = len(self.topo_cols)
        self.n_opt      = len(self.opt_cols)

        self._col_idx    = {c: i for i, c in enumerate(self.col_names)}
        self._bio_slice  = slice(0, self.n_bio)
        self._topo_idxs  = {c: self.n_bio + i for i, c in enumerate(self.topo_cols)}
        self._opt_offset = self.n_bio + self.n_topo

        self.rng            = np.random.default_rng(random_state)
        self.random_state   = random_state
        self.n_folds        = int(max(2, n_folds))
        self.corr_threshold = float(corr_threshold)
        self.prune_collinear = bool(prune_collinear)
        self.n_bootstrap     = int(max(0, n_bootstrap))
        self.n_refit_bootstrap = int(max(0, n_refit_bootstrap))
        self.compute_subspaces = bool(compute_subspaces)

        # ---- data --------------------------------------------------------
        X = np.asarray(species_data, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_vars:
            raise ValueError(
                f"species_data: {self.n_vars} columns expected "
                f"({', '.join(self.col_names)}), got shape {X.shape}.")

        self.target_values = np.asarray(target_values, dtype=float).ravel()
        if self.target_values.size != self.n_vars:
            raise ValueError(
                f"target_values: {self.n_vars} values expected, "
                f"got {self.target_values.size}.")
        if not np.all(np.isfinite(self.target_values)):
            raise ValueError("target_values contains non-finite (NaN/inf) entries.")

        self.max_target_cells = 2000
        self.target_cells = None
        if target_cells is not None:
            TC = np.asarray(target_cells, dtype=float)
            if TC.ndim == 2 and TC.shape[1] == self.n_vars:
                TC = TC[np.all(np.isfinite(TC), axis=1)]
                if TC.shape[0] > self.max_target_cells:
                    step = int(np.ceil(TC.shape[0] / self.max_target_cells))
                    TC = TC[::step]
                if TC.shape[0] >= 2:
                    self.target_cells = TC

        self.target_future = None
        if target_future is not None:
            tf = np.asarray(target_future, dtype=float).ravel()
            if tf.size != self.n_vars:
                raise ValueError(
                    f"target_future: {self.n_vars} values expected, got {tf.size}.")
            if np.all(np.isfinite(tf)):
                self.target_future = tf

        # Drop non-finite rows explicitly rather than letting NaN propagate.
        finite_rows = np.all(np.isfinite(X), axis=1)
        self.n_dropped_rows = int((~finite_rows).sum())
        X = X[finite_rows]
        self.species_data = X
        self.n_records = int(X.shape[0])

        self.coords = None
        if coords is not None:
            C = np.asarray(coords, dtype=float)
            if C.ndim == 2 and C.shape[0] == finite_rows.size and C.shape[1] >= 2:
                C = C[finite_rows][:, :2]
                if np.all(np.isfinite(C)):
                    self.coords = C

        # ---- feature construction (circular aspect everywhere) -----------
        self.F, self.f_names, self.f_group = self._build_features(self.species_data)
        self.f_target = self._build_features_row(self.target_values)
        self.f_target_future = (self._build_features_row(self.target_future)
                                if self.target_future is not None else None)
        self.f_target_cells = (self._build_features(self.target_cells)[0]
                               if self.target_cells is not None else None)
        self.n_features_raw = self.F.shape[1]

        # ---- sample-size gate --------------------------------------------
        self._check_sample_size()

        # ---- collinearity control ----------------------------------------
        self.collinearity = self._collinearity_report()
        keep = self.collinearity["kept_indices"]
        self.Fu        = self.F[:, keep]
        self.fu_names  = [self.f_names[i] for i in keep]
        self.fu_group  = [self.f_group[i] for i in keep]
        self.tu        = self.f_target[keep]
        self.tu_future = (self.f_target_future[keep]
                          if self.f_target_future is not None else None)
        self.tu_cells = (self.f_target_cells[:, keep]
                         if self.f_target_cells is not None else None)

        self.scaler = StandardScaler().fit(self.Fu)
        self.Z   = self.scaler.transform(self.Fu)
        self.z_t = self.scaler.transform(self.tu.reshape(1, -1))[0]
        self.z_t_future = (self.scaler.transform(self.tu_future.reshape(1, -1))[0]
                           if self.tu_future is not None else None)
        self.z_t_cells = (self.scaler.transform(self.tu_cells)
                          if self.tu_cells is not None else None)

        # Legacy attributes kept so v1-era helper code keeps working.
        self.X_bio    = self.species_data[:, self._bio_slice]
        self.t_bio    = self.target_values[self._bio_slice]
        self.X_sc     = StandardScaler().fit_transform(self.species_data)
        self.X_sc_bio = self.X_sc[:, self._bio_slice]
        self.X_ml, self.t_ml = self.Z, self.z_t

        self.results = {}
        self._models = {}
        self._cache  = {}

        self.results["collinearity"] = self.collinearity
        self.results["engine"] = {
            "version": self.VERSION,
            "mode": ("climate_change_response" if self.target_future is not None
                     else "current_climate_site_matching"),
            "n_records_used": self.n_records,
            "n_records_dropped_nonfinite": self.n_dropped_rows,
            "n_features_before_pruning": self.n_features_raw,
            "n_features_used": int(self.Fu.shape[1]),
            "n_folds": self.n_folds,
            "fold_type": "spatial_block" if self.coords is not None else "random",
            "warnings": list(self._warnings),
        }

    # -- feature construction ---------------------------------------------

    def _build_features(self, data):
        """Matrix used for every multivariate stage, with circular aspect.

        v1.0 built such a matrix for the ML models but fed RAW degrees into
        the Mahalanobis covariance, so 359 deg and 1 deg were 358 units apart.
        Here there is exactly one feature space and every stage uses it.
        """
        cols, names, groups = [], [], []
        for j, c in enumerate(self.col_names):
            if c == "aspect_deg":
                a = np.deg2rad(data[:, j])
                cols.append(np.cos(a)); names.append("northness"); groups.append("topo")
                cols.append(np.sin(a)); names.append("eastness");  groups.append("topo")
            else:
                cols.append(data[:, j])
                names.append(c)
                groups.append("bio" if c in BIO_NAMES else
                              "topo" if c in TOPO_NAMES else "opt")
        return np.column_stack(cols), names, groups

    def _build_features_row(self, row):
        return self._build_features(np.asarray(row, dtype=float).reshape(1, -1))[0][0]

    # -- guards ------------------------------------------------------------

    _warnings = ()

    def _check_sample_size(self):
        p = self.n_features_raw
        n = self.n_records
        warns = []
        if n < p + 2:
            raise InsufficientDataError(
                f"Only {n} usable occurrence records for {p} model features.\n"
                f"The covariance matrix is singular at n <= p and every "
                f"multivariate statistic becomes meaningless.\n"
                f"Options: (a) supply at least {p + 2} records "
                f"(>= {max(20, 3 * p)} recommended), or "
                f"(b) select fewer variables.")
        if n < 3 * p:
            warns.append(
                f"n = {n} records for p = {p} features (n < 3p). Shrinkage "
                f"covariance keeps the computation stable, but the conformal "
                f"p-value is coarse (resolution 1/{n + 1} = "
                f"{1.0 / (n + 1):.3f}) and classes D/E cannot be reached "
                f"reliably. At least {3 * p} records are recommended.")
        if n < 20:
            warns.append(
                f"n = {n}: the finest attainable p-value is {1.0 / (n + 1):.3f}, "
                f"so class E (p < 0.01) is unreachable at this sample size.")
        if self.n_dropped_rows:
            warns.append(f"{self.n_dropped_rows} record(s) dropped for "
                         f"non-finite values.")
        if self.coords is None:
            warns.append(
                "No coordinates supplied: cross-validation folds are random, "
                "so spatial autocorrelation between nearby occurrence records "
                "makes the reported calibration mildly optimistic "
                "(Roberts et al. 2017).")
        self._warnings = tuple(warns)

    @staticmethod
    def recommended_min_records(n_bio, n_topo, n_opt, has_aspect):
        """Minimum n the GUI should enforce for a given variable selection."""
        p = n_bio + n_topo + n_opt + (1 if has_aspect else 0)
        return int(max(20, 3 * p)), int(p + 2)

    # -- collinearity ------------------------------------------------------

    def _collinearity_report(self):
        """Correlation screen + conditioning diagnostics.

        Bio1-19 are strongly collinear by construction (bio1/5/6/10/11 are all
        temperature summaries).  v1.0 applied no screen at all and inverted the
        resulting near-singular covariance with a pseudo-inverse while still
        reporting df = p.
        """
        F = self.F
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            C = np.corrcoef(F, rowvar=False)
        C = np.nan_to_num(C, nan=0.0)
        p = C.shape[0]

        dropped, reasons = [], {}
        if self.prune_collinear:
            # Greedy: walk pairs in descending |r|, drop the later column.
            iu = np.triu_indices(p, k=1)
            order = np.argsort(-np.abs(C[iu]))
            for k in order:
                i, j = int(iu[0][k]), int(iu[1][k])
                r = float(C[i, j])
                if abs(r) < self.corr_threshold:
                    break
                if i in dropped or j in dropped:
                    continue
                dropped.append(j)
                reasons[self.f_names[j]] = (
                    f"|r| = {abs(r):.3f} with {self.f_names[i]}")

        keep = [i for i in range(p) if i not in dropped]
        if not keep:
            keep = list(range(p)); dropped, reasons = [], {}

        Ck = C[np.ix_(keep, keep)]
        eig = np.linalg.eigvalsh(Ck)
        eig = np.clip(eig, 1e-12, None)
        # Participation ratio: effective number of independent dimensions.
        eff_dim = float((eig.sum() ** 2) / np.sum(eig ** 2))
        cond = float(eig.max() / eig.min())

        iu2 = np.triu_indices(p, k=1)
        high_pairs = [
            {"a": self.f_names[int(iu2[0][k])],
             "b": self.f_names[int(iu2[1][k])],
             "r": round(float(C[iu2][k]), 4)}
            for k in np.argsort(-np.abs(C[iu2]))[:15]
            if abs(float(C[iu2][k])) >= 0.80
        ]

        return {
            "kept_indices": keep,
            "kept": [self.f_names[i] for i in keep],
            "dropped": [self.f_names[i] for i in dropped],
            "dropped_reason": reasons,
            "corr_threshold": self.corr_threshold,
            "condition_number": round(cond, 2),
            "effective_dimension": round(eff_dim, 2),
            "nominal_dimension": len(keep),
            "high_correlation_pairs": high_pairs,
            "note": ("Pruning applies to the multivariate stage only; every "
                     "selected variable is still reported individually."),
        }

    # ==================================================================
    #  NON-CONFORMITY ENSEMBLE  +  CROSS-CONFORMAL CALIBRATION
    # ==================================================================
    #
    #  One statistic, one calibration.  v1.0 summed five component scores in
    #  which the "threshold zone" term was itself a weighted vote over four of
    #  the other terms, so Mahalanobis entered the composite twice with an
    #  effective weight of ~0.44 while nominally weighted 0.25.
    #
    #  Because the p-value is obtained by ranking, the hyperparameters below
    #  (number of trees, nu, number of mixture components) affect only the
    #  SHARPNESS of the test, never its validity.  That is the point of doing
    #  it this way: no hyperparameter choice can make the output miscalibrated.
    # ==================================================================

    MODEL_KEYS = ("mahalanobis", "gmm", "isolation_forest", "ocsvm")

    def _fit_bundle(self, Z_fit):
        n_fit, p_fit = Z_fit.shape
        b = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Ledoit-Wolf shrinkage: well conditioned even when n approaches
            # p, which is exactly where v1.0 returned NaN or ~1e7 distances.
            b["lw"] = LedoitWolf(store_precision=True).fit(Z_fit)

            n_pc = int(max(2, min(p_fit, max(2, n_fit // 10))))
            b["pca"] = PCA(n_components=n_pc,
                           random_state=self.random_state).fit(Z_fit)
            G = b["pca"].transform(Z_fit)

            k_max = int(max(1, min(3, n_fit // 30)))
            best, best_bic = None, np.inf
            for k in range(1, k_max + 1):
                try:
                    g = GaussianMixture(n_components=k, covariance_type="full",
                                        random_state=self.random_state,
                                        max_iter=300, reg_covar=1e-4).fit(G)
                    bic = g.bic(G)
                    if np.isfinite(bic) and bic < best_bic:
                        best, best_bic = g, bic
                except Exception:
                    continue
            if best is None:
                best = GaussianMixture(n_components=1, covariance_type="diag",
                                       random_state=self.random_state,
                                       reg_covar=1e-2).fit(G)
                best_bic = float("nan")
            b["gmm"], b["gmm_bic"] = best, float(best_bic)

            # contamination='auto' replaces v1.0's hard-coded 0.10, which
            # declared 10 % of the species' own records anomalous by fiat.
            b["iso"] = IsolationForest(n_estimators=200, contamination="auto",
                                       random_state=self.random_state,
                                       n_jobs=-1).fit(Z_fit)
            b["ocsvm"] = OneClassSVM(nu=0.10, kernel="rbf",
                                     gamma="scale").fit(Z_fit)

        raw = self._raw_components(b, Z_fit)
        b["mu"] = {k: float(np.mean(v)) for k, v in raw.items()}
        b["sd"] = {k: (float(np.std(v)) or 1.0) for k, v in raw.items()}
        b["n_fit"] = int(n_fit)
        return b

    def _raw_components(self, b, Z):
        """Raw model statistics; higher = more atypical for every model."""
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d2 = np.asarray(b["lw"].mahalanobis(Z), dtype=float)
            return {
                "mahalanobis":      np.sqrt(np.clip(d2, 0.0, None)),
                "gmm":              -np.asarray(
                    b["gmm"].score_samples(b["pca"].transform(Z)), dtype=float),
                "isolation_forest": -np.asarray(
                    b["iso"].score_samples(Z), dtype=float),
                "ocsvm":            -np.asarray(
                    b["ocsvm"].decision_function(Z), dtype=float).ravel(),
            }

    def _standardise(self, b, raw):
        return {k: (raw[k] - b["mu"][k]) / b["sd"][k] for k in self.MODEL_KEYS}

    def _components(self, b, Z):
        return self._standardise(b, self._raw_components(b, Z))

    @classmethod
    def _combine(cls, comp, weights):
        w = np.array([weights[k] for k in cls.MODEL_KEYS], dtype=float)
        w = w / (w.sum() or 1.0)
        M = np.column_stack([np.atleast_1d(comp[k]) for k in cls.MODEL_KEYS])
        return M @ w

    # -- folds -------------------------------------------------------------

    def _make_folds(self, n):
        """Spatial blocks when coordinates are available, else random K-fold.

        Random folds let a record be predicted from its immediate spatial
        neighbours, which inflates apparent performance whenever the
        occurrence records are spatially clustered (Roberts et al. 2017).
        """
        k = int(min(self.n_folds, max(2, n // 5)))
        if self.coords is not None and self.coords.shape[0] == n and n >= 2 * k:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    n_blocks = int(min(k * 3, max(k, n // 5)))
                    lab = KMeans(n_clusters=n_blocks, n_init=10,
                                 random_state=self.random_state
                                 ).fit_predict(self.coords)
                # Assign whole spatial blocks to folds, largest block first,
                # always to the currently lightest fold.
                blocks = sorted(range(n_blocks),
                                key=lambda bl: -int(np.sum(lab == bl)))
                fold_of_block, load = {}, [0] * k
                for bl in blocks:
                    f = int(np.argmin(load))
                    fold_of_block[bl] = f
                    load[f] += int(np.sum(lab == bl))
                fold_id = np.array([fold_of_block[int(l)] for l in lab])
                folds = []
                for f in range(k):
                    te = np.where(fold_id == f)[0]
                    tr = np.where(fold_id != f)[0]
                    if te.size and tr.size >= 5:
                        folds.append((tr, te))
                if len(folds) >= 2:
                    return folds
            except Exception:
                pass
        kf = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
        return list(kf.split(np.arange(n)))

    # -- the core routine --------------------------------------------------

    def _crossconformal(self, Z, z_targets, tag="full"):
        """Cross-conformal predictor (Vovk 2015).

        For every fold k: fit on the other folds, then score both the held-out
        records and every target with that same fold-k model.  Occurrence
        records and targets are therefore always out-of-sample for the model
        that scores them, which is what makes the pooled p-value valid.
        """
        n = int(Z.shape[0])
        folds = self._make_folds(n)
        z_targets = np.atleast_2d(np.asarray(z_targets, dtype=float))
        n_t = int(z_targets.shape[0])
        n_m = len(self.MODEL_KEYS)

        alphas    = np.full(n, np.nan)
        fold_of   = np.full(n, -1, dtype=int)
        comp_rec  = {k: np.full(n, np.nan) for k in self.MODEL_KEYS}
        raw_rec   = {k: np.full(n, np.nan) for k in self.MODEL_KEYS}
        comp_tgt  = np.full((len(folds), n_t, n_m), np.nan)
        raw_tgt   = np.full((len(folds), n_t, n_m), np.nan)
        alpha_tgt = np.full((len(folds), n_t), np.nan)
        bics, bundles = [], []

        for fi, (tr, te) in enumerate(folds):
            b = self._fit_bundle(Z[tr])
            bundles.append(b)
            bics.append(b["gmm_bic"])

            raw_te = self._raw_components(b, Z[te])
            c_te   = self._standardise(b, raw_te)
            alphas[te]  = self._combine(c_te, ENSEMBLE_WEIGHTS)
            fold_of[te] = fi
            for k in self.MODEL_KEYS:
                comp_rec[k][te] = c_te[k]
                raw_rec[k][te]  = raw_te[k]

            raw_tg = self._raw_components(b, z_targets)
            c_tg   = self._standardise(b, raw_tg)
            for mi, k in enumerate(self.MODEL_KEYS):
                comp_tgt[fi, :, mi] = c_tg[k]
                raw_tgt[fi, :, mi]  = raw_tg[k]
            alpha_tgt[fi] = self._combine(c_tg, ENSEMBLE_WEIGHTS)

        # Pooled cross-conformal p-value, one per target.
        pvals = [self._pool_pvalue(alphas, fold_of, alpha_tgt[:, t], n)
                 for t in range(n_t)]

        # Per-record self p-values: under exchangeability these are ~U(0,1).
        # This is the calibration evidence v1.0 never produced.
        self_p = np.full(n, np.nan)
        finite = np.isfinite(alphas)
        a_f = alphas[finite]
        if a_f.size > 1:
            srt = np.sort(a_f)                       # ascending
            ge  = a_f.size - np.searchsorted(srt, a_f, side="left")
            self_p[finite] = np.clip(ge / float(a_f.size), 1e-9, 1.0)

        self._cache[tag] = {
            "folds": folds, "bundles": bundles, "alphas": alphas,
            "fold_of": fold_of, "comp_rec": comp_rec, "raw_rec": raw_rec,
            "comp_tgt": comp_tgt, "raw_tgt": raw_tgt,
            "alpha_tgt": alpha_tgt, "n": n, "Z": Z, "z_targets": z_targets,
        }
        return {
            "p_values": pvals, "alphas": alphas, "self_p": self_p,
            "alpha_targets": alpha_tgt, "folds": folds,
            "gmm_bic_per_fold": [round(float(x), 2) if np.isfinite(x) else None
                                 for x in bics],
            "resolution": round(1.0 / (n + 1.0), 5),
        }

    @staticmethod
    def _pool_pvalue(alphas, fold_of, alpha_tgt_per_fold, n):
        """(1 + sum_k #{i in fold k : alpha_i >= alpha_target^(k)}) / (n + 1)."""
        cnt = 0
        for fi, a_t in enumerate(alpha_tgt_per_fold):
            if not np.isfinite(a_t):
                continue
            sel = (fold_of == fi) & np.isfinite(alphas)
            cnt += int(np.sum(alphas[sel] >= a_t))
        return float((1.0 + cnt) / (n + 1.0))

    def _model_pvalue(self, model_key, target_index=0, tag="full"):
        """Cross-conformal p-value for one component model on its own."""
        c = self._cache.get(tag)
        if c is None:
            return float("nan")
        mi = self.MODEL_KEYS.index(model_key)
        a  = c["comp_rec"][model_key]
        at = c["comp_tgt"][:, target_index, mi]
        return self._pool_pvalue(a, c["fold_of"], at, c["n"])

    @staticmethod
    def zone_from_pvalue(p):
        """Zone labels tied to the same thresholds as the score bands."""
        if p is None or not np.isfinite(p):
            return "INDETERMINATE"
        if p >= 0.20: return "CORE"
        if p >= 0.10: return "SUITABLE"
        if p >= 0.05: return "MARGINAL"
        if p >= 0.01: return "OUTSIDE_NEAR"
        return "OUTSIDE"

    def _run_core(self):
        """Fit the full-space cross-conformal predictor (idempotent)."""
        if "core" in self._models:
            return self._models["core"]
        z_list = [self.z_t]
        self._idx_future = None
        self._idx_cells = None
        if self.z_t_future is not None:
            self._idx_future = len(z_list)
            z_list.append(self.z_t_future)
        if self.z_t_cells is not None:
            self._idx_cells = (len(z_list), len(z_list) + len(self.z_t_cells))
            z_list.extend(list(self.z_t_cells))
        core = self._crossconformal(self.Z, np.array(z_list), tag="full")
        self._models["core"] = core
        self._models["bundle_full"] = self._fit_bundle(self.Z)
        return core

    def _subspace(self, group_names, tag):
        """Cross-conformal restricted to a subset of feature groups."""
        idx = [i for i, g in enumerate(self.fu_group) if g in group_names]
        if len(idx) < 1:
            return None
        z_list = [self.z_t[idx]]
        if self.z_t_future is not None:
            z_list.append(self.z_t_future[idx])
        return self._crossconformal(self.Z[:, idx], np.array(z_list), tag=tag)

    # ==================================================================
    #  LAYER A  -  ORDINATION  (DIAGNOSTIC ONLY, not part of the score)
    # ==================================================================
    #  v1.0's documentation listed PCA, Kernel PCA and K-Means as scoring
    #  layers.  None of them ever entered the composite.  They are kept
    #  because the plots are useful, and are now labelled for what they are.

    def compute_pca(self):
        Z = self.Z
        n_comp = int(min(5, Z.shape[1]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca5 = PCA(n_components=n_comp,
                       random_state=self.random_state).fit(Z)
            pca2 = PCA(n_components=int(min(2, Z.shape[1])),
                       random_state=self.random_state).fit(Z)
        X2 = pca2.transform(Z)
        t2 = pca2.transform(self.z_t.reshape(1, -1))[0]
        if X2.shape[1] < 2:
            X2 = np.column_stack([X2, np.zeros(len(X2))])
            t2 = np.array([t2[0], 0.0])
        c = X2.mean(axis=0)
        dist = float(np.linalg.norm(t2 - c))
        r = {
            "role": "diagnostic_only",
            "note": "Ordination for visualisation. Does NOT contribute to the score.",
            "species_pc2": X2.tolist(), "target_pc2": t2.tolist(),
            "ev_2": [round(float(v), 4) for v in pca2.explained_variance_ratio_],
            "ev_5": [round(float(v), 4) for v in pca5.explained_variance_ratio_],
            "cumulative_ev5": round(float(pca5.explained_variance_ratio_.sum()), 4),
            "distance_2d": round(dist, 4),
            "score": None,
            "loadings": {c_: {"PC1": round(float(pca2.components_[0, i]), 4),
                              "PC2": round(float(pca2.components_[1, i]), 4)
                              if pca2.components_.shape[0] > 1 else 0.0}
                         for i, c_ in enumerate(self.fu_names)},
        }
        self.results["pca"] = r
        self._models.update({"pca2": pca2, "X2": X2, "t2": t2})
        return r

    def compute_kernel_pca(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kp = KernelPCA(n_components=2, kernel="rbf",
                               gamma=1.0 / max(1, self.Z.shape[1]),
                               random_state=self.random_state)
                Xk = kp.fit_transform(self.Z)
                tk = kp.transform(self.z_t.reshape(1, -1))[0]
            r = {"species_kpca": Xk.tolist(), "target_kpca": tk.tolist()}
        except Exception as e:
            r = {"error": str(e)}
        r.update({"role": "diagnostic_only", "score": None,
                  "note": "Visualisation only. Does NOT contribute to the score."})
        self.results["kpca"] = r
        return r

    def compute_kmeans_niche(self):
        Z = self.Z
        best_k, best_sil, sil = 2, -1.0, {}
        hi = int(min(7, max(3, len(Z) // 5 + 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(2, hi):
                try:
                    lb = KMeans(n_clusters=k, random_state=self.random_state,
                                n_init=10).fit_predict(Z)
                    s = float(silhouette_score(Z, lb))
                    sil[str(k)] = round(s, 4)
                    if s > best_sil:
                        best_sil, best_k = s, k
                except Exception:
                    continue
            km = KMeans(n_clusters=best_k, random_state=self.random_state,
                        n_init=10).fit(Z)
        lb = km.labels_
        tc = int(km.predict(self.z_t.reshape(1, -1))[0])
        ctr = km.cluster_centers_[tc]
        dc = float(np.linalg.norm(self.z_t - ctr))
        rd = np.linalg.norm(Z[lb == tc] - ctr, axis=1)
        pct = float(percentileofscore(rd, dc, kind="rank")) if rd.size else 50.0
        r = {"role": "diagnostic_only", "score": None,
             "note": "Cluster structure description. Does NOT contribute to the score.",
             "best_k": best_k, "silhouette": sil,
             "best_silhouette": round(best_sil, 4), "target_cluster": tc,
             "cluster_size_frac": round(float(np.mean(lb == tc)), 4),
             "dist_to_center": round(dc, 4),
             "dist_percentile_in_cluster": round(pct, 2)}
        self.results["kmeans"] = r
        self._models["kmeans"] = km
        return r

    # ==================================================================
    #  LAYER B  -  CALIBRATED COMPONENT MODELS
    # ==================================================================

    def _component_result(self, key):
        core = self._run_core()
        c = self._cache["full"]
        mi = self.MODEL_KEYS.index(key)
        p = self._model_pvalue(key, 0)
        raw_t = float(np.nanmean(c["raw_tgt"][:, 0, mi]))
        raw_r = c["raw_rec"][key]
        zone = self.zone_from_pvalue(p)
        return {
            "conformal_p": round(float(p), 5),
            "score": round(score_from_pvalue(p), 2),
            "zone": zone,
            "statistic_target": round(raw_t, 5),
            "statistic_species_median": round(float(np.nanmedian(raw_r)), 5),
            "statistic_species_p90": round(float(np.nanpercentile(raw_r, 90)), 5),
            "percentile_rank": round(float(100.0 * (1.0 - p)), 2),
            "calibration": "cross-conformal, out-of-fold",
        }, c, mi, core

    def compute_gmm(self):
        r, c, mi, core = self._component_result("gmm")
        b = self._models["bundle_full"]
        r.update({
            "best_n_components": int(b["gmm"].n_components),
            "bic_full_fit": round(float(b["gmm_bic"]), 2)
            if np.isfinite(b["gmm_bic"]) else None,
            "bic_per_fold": core["gmm_bic_per_fold"],
            "weights": [round(float(w), 4) for w in b["gmm"].weights_],
            "log_probability": round(-r["statistic_target"], 4),
            "niche_breadth": round(float(
                -np.sum(b["gmm"].weights_ * np.log(b["gmm"].weights_ + 1e-9))), 4),
            "pca_dims_used": int(b["pca"].n_components_),
            "note": ("Log-likelihood is ranked against out-of-fold values; "
                     "no sigmoid or hand-tuned slope is involved."),
        })
        self.results["gmm"] = r
        return r

    def compute_isolation_forest(self):
        r, c, mi, core = self._component_result("isolation_forest")
        r.update({
            "anomaly_score": round(-r["statistic_target"], 4),
            "contamination": "auto",
            "is_normal": bool(r["conformal_p"] >= 0.05),
            "note": ("contamination='auto'; v1.0 fixed it at 0.10, declaring "
                     "10 % of the species' own records anomalous a priori."),
        })
        self.results["isolation_forest"] = r
        return r

    def compute_ocsvm(self):
        r, c, mi, core = self._component_result("ocsvm")
        r.update({
            "decision_value": round(-r["statistic_target"], 4),
            "nu": 0.10,
            "is_inside": bool(r["conformal_p"] >= 0.05),
            "note": ("nu is fixed; under conformal calibration hyperparameters "
                     "affect sharpness only, never validity."),
        })
        self.results["ocsvm"] = r
        return r

    def compute_mahalanobis(self):
        r, c, mi, core = self._component_result("mahalanobis")
        d = r["statistic_target"]
        raw = c["raw_rec"]["mahalanobis"]
        raw = raw[np.isfinite(raw)]

        # Distribution-FREE thresholds from the out-of-fold distances of the
        # species' own records.  v1.0 derived them from chi2 with df = p,
        # which additionally requires mu and Sigma to be KNOWN rather than
        # estimated from the same sample.
        d50, d90, d95, d99 = (float(np.percentile(raw, q))
                              for q in (50, 90, 95, 99)) if raw.size else (
                                  np.nan, np.nan, np.nan, np.nan)

        # Parametric reference, reported for comparison only.  For a new
        # observation scored against an ESTIMATED mean and covariance the
        # correct reference is an F distribution, not chi2:
        #     D^2 ~ p(n^2-1)/(n(n-p)) * F(p, n-p)
        p_param, param_note = None, ""
        n, pdim = self.n_records, int(self.Fu.shape[1])
        if n > pdim + 1:
            try:
                from scipy.stats import f as f_dist
                S = np.cov(self.Z, rowvar=False)
                mu = self.Z.mean(axis=0)
                diff = self.z_t - mu
                d2_plain = float(diff @ np.linalg.solve(S, diff))
                scale = pdim * (n ** 2 - 1.0) / (n * (n - pdim))
                p_param = float(1.0 - f_dist.cdf(d2_plain / scale, pdim, n - pdim))
                param_note = ("F reference on the unshrunk sample covariance; "
                              "valid only under multivariate normality. "
                              "Reported for comparison - the conformal p-value "
                              "is the one used for scoring.")
            except Exception:
                p_param = None
        else:
            param_note = ("n <= p + 1: no parametric reference is defined. "
                          "The conformal p-value remains valid.")

        r.update({
            "distance": round(float(d), 4),
            "covariance_estimator": "Ledoit-Wolf shrinkage",
            "shrinkage": round(float(self._models["bundle_full"]["lw"].shrinkage_), 4),
            "p_value": round(float(r["conformal_p"]), 5),
            "p_value_conformal": round(float(r["conformal_p"]), 5),
            "p_value_parametric_F": (round(p_param, 5)
                                     if p_param is not None else None),
            "parametric_note": param_note,
            "d50_threshold": round(d50, 4), "d90_threshold": round(d90, 4),
            "d95_threshold": round(d95, 4), "d99_threshold": round(d99, 4),
            "threshold_basis": "empirical out-of-fold quantiles (distribution-free)",
            "n_variables": pdim,
            "aspect_encoding": ("circular (northness/eastness)"
                                if self.has_aspect else "n/a"),
            "interpretation": self._mahal_interp(r["conformal_p"], d, d95),
        })
        self.results["mahalanobis_stat"] = r
        return r

    @staticmethod
    def _mahal_interp(p, d, d95):
        z = ClimateResilienceAnalyzer.zone_from_pvalue(p)
        txt = {"CORE":         "✅ Core niche",
               "SUITABLE":     "🟡 Suitable niche",
               "MARGINAL":     "🟠 Marginal niche",
               "OUTSIDE_NEAR": "🔶 Borderline outside",
               "OUTSIDE":      "🔴 Outside niche",
               "INDETERMINATE": "⚪ Indeterminate"}[z]
        return f"{txt}   (D = {d:.2f}, D95 = {d95:.2f}, conformal p = {p:.4f})"

    # ==================================================================
    #  LAYER C  -  ENSEMBLE ZONE  (now a read-out, not an extra addend)
    # ==================================================================

    def compute_threshold_zone(self):
        for k, fn in (("gmm", self.compute_gmm),
                      ("isolation_forest", self.compute_isolation_forest),
                      ("ocsvm", self.compute_ocsvm),
                      ("mahalanobis_stat", self.compute_mahalanobis)):
            if k not in self.results:
                fn()
        core = self._run_core()
        p = float(core["p_values"][0])
        final = self.zone_from_pvalue(p)
        votes = {m: self.results[
            "mahalanobis_stat" if m == "mahalanobis" else m]["zone"]
            for m in self.MODEL_KEYS}
        labels = {
            "CORE":          "✅ Core Niche      — indistinguishable from occurrence records",
            "SUITABLE":      "🟡 Suitable Niche  — inside the niche, towards its edge",
            "MARGINAL":      "🟠 Marginal Niche  — at the tolerance boundary (p < 0.10)",
            "OUTSIDE_NEAR":  "🔶 Borderline Out  — rejected at 5 % but not at 1 %",
            "OUTSIDE":       "🔴 Outside Niche   — rejected at the 1 % level",
            "INDETERMINATE": "⚪ Indeterminate   — insufficient data",
        }
        r = {"role": "diagnostic_readout",
             "note": ("Zone is read off the single ensemble p-value. It is NOT "
                      "an additional weighted term: v1.0 added it on top of the "
                      "same four models, double-counting them."),
             "final_zone": final, "zone_label": labels[final],
             "ensemble_p": round(p, 5),
             "score": round(score_from_pvalue(p), 2),
             "votes": votes,
             "mean_vote": round(float(np.mean(
                 [{"CORE": 3, "SUITABLE": 2, "MARGINAL": 1,
                   "OUTSIDE_NEAR": 0.6, "OUTSIDE": 0,
                   "INDETERMINATE": 0}[v] for v in votes.values()])), 3)}
        self.results["threshold_zone"] = r
        return r

    # ==================================================================
    #  PER-VARIABLE DESCRIPTION  (marginal; never enters the composite)
    # ==================================================================

    def _context(self):
        """Derived quantities used by the mechanism texts (e.g. VPD)."""
        ctx = {}
        if "bio1" in self.bio_cols:
            j = self.bio_cols.index("bio1")
            div, _ = detect_temp_scale(self.X_bio[:, j], "bio1")
            ctx["mean_temp_c"] = float(self.t_bio[j]) / div
        return ctx

    def _pct_record(self, sp, tgt, col, is_critical=False):
        sp = np.asarray(sp, dtype=float)
        sp = sp[np.isfinite(sp)]
        pct = float(percentileofscore(sp, tgt, kind="rank"))
        p5, p25, p75, p95 = (float(np.percentile(sp, q)) for q in (5, 25, 75, 95))
        med, std = float(np.median(sp)), float(np.std(sp))
        div, label = detect_temp_scale(sp, col)

        dev = abs(pct - 50.0) / 50.0
        sc  = float(np.clip(100.0 * (1.0 - dev ** 1.5), 0, 100))
        if tgt < p5 or tgt > p95:
            sc *= 0.5
        ri = build_risk_explanation(col, float(tgt), med, p5, p25, p75, p95,
                                    pct, sc, float(sp.min()), float(sp.max()),
                                    std, unit_divisor=div, unit_label=label,
                                    context=self._context())
        return {
            "name": ALL_VAR_NAMES.get(col, col),
            "target_value": round(float(tgt), 6),
            "percentile": round(pct, 2),
            "species_min": round(float(sp.min()), 4),
            "species_p5": round(p5, 4), "species_p25": round(p25, 4),
            "species_median": round(med, 4),
            "species_p75": round(p75, 4), "species_p95": round(p95, 4),
            "species_max": round(float(sp.max()), 4),
            "species_std": round(std, 4),
            "in_core_range": bool(p25 <= tgt <= p75),
            "in_tolerance_range": bool(p5 <= tgt <= p95),
            "rank_pvalue": ri["rank_pvalue"],
            "unit_label": label,
            "score": round(sc, 2),
            "score_meaning": "marginal typicality only - not part of the composite",
            "critical": bool(is_critical),
            "risk_level": ri["risk_level"], "risk_color": ri["risk_color"],
            "risk_emoji": ri["risk_emoji"], "dev_from_med": ri["dev_from_med"],
            "dev_pct": ri["dev_pct"], "n_std": ri["n_std"],
            "dist_to_p5": ri["dist_to_p5"], "dist_to_p95": ri["dist_to_p95"],
            "risk_explanation": ri["explanation"],
        }

    def compute_percentile_analysis(self):
        per_bio = {}
        for j, bio in enumerate(self.bio_cols):
            per_bio[bio] = self._pct_record(self.X_bio[:, j],
                                            float(self.t_bio[j]),
                                            bio, bio in CRITICAL_BIOS)
        per_opt = {}
        for k, oc in enumerate(self.opt_cols):
            idx = self._opt_offset + k
            per_opt[oc] = self._pct_record(self.species_data[:, idx],
                                           float(self.target_values[idx]),
                                           oc, oc in CRITICAL_OPT)
        r = {
            "role": "descriptive",
            "note": ("Marginal, one-variable-at-a-time view. It deliberately "
                     "does NOT feed the composite: averaging per-variable "
                     "distance-from-median over many collinear variables is "
                     "dominated by dimensionality, not by ecology."),
            "per_bio": per_bio, "per_opt": per_opt,
            "bios_in_core_range": sum(1 for b in per_bio.values() if b["in_core_range"]),
            "bios_in_tolerance_range": sum(1 for b in per_bio.values() if b["in_tolerance_range"]),
            "bios_outside_range": sum(1 for b in per_bio.values() if not b["in_tolerance_range"]),
            "n_bio_used": len(self.bio_cols),
            "score": round(float(np.mean([b["score"] for b in per_bio.values()])), 2)
            if per_bio else 100.0,
        }
        self.results["percentile"] = r
        return r

    # ==================================================================
    #  TOPOGRAPHY
    # ==================================================================

    def _topo_weights(self):
        raw = {}
        if "elevation_m" in self.topo_cols: raw["elevation_m"] = W_ELEV
        if "slope_pct"   in self.topo_cols: raw["slope_pct"]   = W_SLOPE
        if "aspect_deg"  in self.topo_cols: raw["aspect_deg"]  = W_ASPECT
        tot = sum(raw.values()) or 1.0
        return {k: v / tot for k, v in raw.items()}

    def _clim_topo_weights(self):
        """Share of model features that are climate vs topography.

        These are NOT scoring weights any more - topography enters the single
        non-conformity statistic directly, so no arbitrary 75/25 (documented
        as 80/20 in v1.0's README, a further inconsistency) is applied.
        """
        n_topo_f = sum(1 for g in self.fu_group if g == "topo")
        n_tot = len(self.fu_group) or 1
        return (round((n_tot - n_topo_f) / n_tot, 4), round(n_topo_f / n_tot, 4))

    def compute_topo_score(self):
        if not self.has_topo:
            r = {"score": None, "elevation": {}, "slope": {}, "aspect": {},
                 "note": "No topographic variables selected.",
                 "topo_vars_used": []}
            self.results["topo"] = r
            return r

        tw = self._topo_weights()
        elev_r = slop_r = {}
        asp_result = {}
        desc = 0.0

        if "elevation_m" in self.topo_cols:
            i = self._topo_idxs["elevation_m"]
            elev_r = self._pct_record(self.species_data[:, i],
                                      float(self.target_values[i]), "elevation_m")
            desc += tw["elevation_m"] * elev_r["score"]
        if "slope_pct" in self.topo_cols:
            i = self._topo_idxs["slope_pct"]
            slop_r = self._pct_record(self.species_data[:, i],
                                      float(self.target_values[i]), "slope_pct")
            desc += tw["slope_pct"] * slop_r["score"]
        if "aspect_deg" in self.topo_cols:
            i = self._topo_idxs["aspect_deg"]
            at = float(self.target_values[i])
            sp = self.species_data[:, i]
            mn = float(np.mean(np.cos(np.deg2rad(sp))))
            me = float(np.mean(np.sin(np.deg2rad(sp))))
            mvl = float(np.sqrt(mn ** 2 + me ** 2))
            tn, te = northness(at), eastness(at)
            cos_sim = float((tn * mn + te * me) /
                            (np.sqrt(tn ** 2 + te ** 2) * (mvl + 1e-9) + 1e-9))
            asp_sc = float(np.clip(50 + 50 * cos_sim * mvl, 0, 100))
            asp_result = {
                "name": "Aspect (deg)", "target_deg": round(at, 2),
                "target_northness": round(tn, 4), "target_eastness": round(te, 4),
                "sp_mean_north": round(mn, 4), "sp_mean_east": round(me, 4),
                "vec_length": round(mvl, 4), "cos_similarity": round(cos_sim, 4),
                "score": round(asp_sc, 2),
                "circular_mean_deg": round(circular_mean_deg(sp), 2),
                "target_exposure": ("North" if tn > 0.5 else
                                    "South" if tn < -0.5 else "Mixed"),
                "sp_exposure": ("North-facing" if mn > 0.3 else
                                "South-facing" if mn < -0.3 else "Mixed"),
            }
            desc += tw["aspect_deg"] * asp_sc

        # Calibrated topography-only sub-score.
        p_topo = None
        if self.compute_subspaces:
            sub = self._subspace({"topo"}, "topo")
            if sub is not None:
                p_topo = float(sub["p_values"][0])

        r = {
            "score": round(score_from_pvalue(p_topo), 2) if p_topo is not None
            else round(desc, 2),
            "conformal_p": round(p_topo, 5) if p_topo is not None else None,
            "descriptive_score": round(desc, 2),
            "descriptive_weights": {k: round(v, 3) for k, v in tw.items()},
            "elevation": elev_r, "slope": slop_r, "aspect": asp_result,
            "topo_vars_used": self.topo_cols,
            "note": ("'score' is the calibrated topography-only conformal "
                     "score; 'descriptive_score' is the old weighted "
                     "percentile figure, kept for continuity."),
        }
        self.results["topo"] = r
        return r

    # ==================================================================
    #  VALIDATION   (the layer v1.0 had none of)
    # ==================================================================

    def _alpha_oof_like(self, Zpoints, max_bundles=3):
        """Alpha for out-of-sample points, averaged over fold bundles.

        Gives points that were never in any fit set a treatment comparable to
        the out-of-fold alphas of the occurrence records.
        """
        c = self._cache["full"]
        bundles = c["bundles"][:max_bundles] or [self._models["bundle_full"]]
        acc = None
        for b in bundles:
            a = self._combine(self._components(b, Zpoints), ENSEMBLE_WEIGHTS)
            acc = a if acc is None else acc + a
        return acc / float(len(bundles))

    def _permuted_null(self, m=None):
        """Raes & ter Steege (2007) style null: permute each variable
        independently, destroying the covariance structure while preserving
        every marginal distribution."""
        n = self.Z.shape[0]
        m = int(m or min(max(n, 100), 500))
        rng = np.random.default_rng(self.random_state + 101)
        idx = rng.integers(0, n, size=(m, self.Z.shape[1]))
        return np.take_along_axis(self.Z, idx, axis=0)

    def compute_validation(self):
        core = self._run_core()
        self_p = np.asarray(core["self_p"], dtype=float)
        sp = self_p[np.isfinite(self_p)]

        ks_stat = ks_p = None
        if sp.size >= 8:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ks = kstest(sp, "uniform")
                ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)

        rej = {f"alpha_{a:.2f}": round(float(np.mean(sp < a)), 4)
               for a in (0.01, 0.05, 0.10, 0.20)} if sp.size else {}

        scores = np.array([score_from_pvalue(x) for x in sp]) if sp.size else np.array([])
        def _cls(s):
            return ("A" if s >= 80 else "B" if s >= 65 else
                    "C" if s >= 50 else "D" if s >= 35 else "E")
        dist = {k: 0 for k in "ABCDE"}
        for s in scores:
            dist[_cls(s)] += 1
        n_sc = max(1, scores.size)
        dist_pct = {k: round(100.0 * v / n_sc, 1) for k, v in dist.items()}

        # Null-model discrimination: can the ensemble tell real records from
        # marginal-preserving permutations?  AUC ~ 0.5 means the multivariate
        # structure adds nothing over the one-variable-at-a-time view.
        auc = None
        try:
            null_Z = self._permuted_null()
            a_real = core["alphas"][np.isfinite(core["alphas"])]
            # Score the null points with a SINGLE fold bundle so that both
            # sides are one-model, out-of-sample evaluations; averaging over
            # several bundles would shrink the null spread and bias the AUC.
            a_null = self._alpha_oof_like(null_Z, max_bundles=1)
            y = np.r_[np.zeros(a_real.size), np.ones(a_null.size)]
            auc = float(roc_auc_score(y, np.r_[a_real, a_null]))
        except Exception:
            auc = None

        verdict = []
        if ks_p is not None:
            verdict.append(
                "Calibration OK (KS p = %.3f): out-of-fold p-values are "
                "consistent with Uniform(0,1)." % ks_p if ks_p >= 0.05 else
                "Calibration WARNING (KS p = %.4f): out-of-fold p-values "
                "depart from Uniform(0,1). Likely causes are strong spatial "
                "clustering of the records (supply coordinates for spatial "
                "block folds) or a small sample." % ks_p)
        if rej:
            verdict.append(
                "Empirical type-I error at alpha = 0.05 is %.1f%% "
                "(nominal 5.0%%); at alpha = 0.01 it is %.1f%% (nominal 1.0%%)."
                % (100 * rej["alpha_0.05"], 100 * rej["alpha_0.01"]))
        verdict.append(
            "%.0f%% of the species' own occurrence records fall in class A "
            "and %.0f%% in class A or B (design targets: 80%% and 90%%)."
            % (dist_pct["A"], dist_pct["A"] + dist_pct["B"]))
        if auc is not None:
            verdict.append(
                "Null-model discrimination AUC = %.3f %s"
                % (auc, "(the multivariate structure carries real signal)."
                   if auc >= 0.7 else
                   "(weak: the ensemble barely improves on the marginal "
                   "distributions; interpret the multivariate score cautiously)."))

        r = {
            "method": ("Cross-conformal self-calibration: every occurrence "
                       "record is scored by a model fitted without its fold, "
                       "then ranked against the others."),
            "n_calibration_records": int(sp.size),
            "pvalue_resolution": core["resolution"],
            "ks_uniformity_statistic": round(ks_stat, 4) if ks_stat is not None else None,
            "ks_uniformity_pvalue": round(ks_p, 5) if ks_p is not None else None,
            "empirical_rejection_rate": rej,
            "self_class_counts": dist,
            "self_class_percent": dist_pct,
            "self_score_mean": round(float(scores.mean()), 2) if scores.size else None,
            "self_score_median": round(float(np.median(scores)), 2) if scores.size else None,
            "null_model_auc": round(auc, 4) if auc is not None else None,
            "fold_type": "spatial_block" if self.coords is not None else "random",
            "n_folds_used": len(core["folds"]),
            "verdict": verdict,
        }
        self.results["validation"] = r
        return r

    # ==================================================================
    #  EXTRAPOLATION  (MESS - Elith, Kearney & Phillips 2010)
    # ==================================================================

    def compute_extrapolation(self, target_row=None, key="extrapolation"):
        F = self.F
        t = self.f_target if target_row is None else np.asarray(target_row, float)
        per = {}
        sims = []
        for j, name in enumerate(self.f_names):
            col = F[:, j]
            lo, hi = float(col.min()), float(col.max())
            rng_ = (hi - lo) or 1e-9
            f = float(np.mean(col < t[j])) * 100.0
            if f == 0.0:
                s = (t[j] - lo) / rng_ * 100.0
            elif f < 50.0:
                s = 2.0 * f
            elif f < 100.0:
                s = 2.0 * (100.0 - f)
            else:
                s = (hi - t[j]) / rng_ * 100.0
            per[name] = round(float(s), 2)
            sims.append(float(s))
        mess = float(min(sims)) if sims else float("nan")
        limiting = min(per, key=per.get) if per else None
        r = {
            "mess": round(mess, 2),
            "limiting_variable": limiting,
            "per_variable_similarity": per,
            "is_extrapolation": bool(mess < 0),
            "n_variables_outside_range": int(sum(1 for v in sims if v < 0)),
            "interpretation": (
                "MESS < 0: the target lies OUTSIDE the training range for at "
                "least one variable. Every model output for it is an "
                "extrapolation and the conformal p-value, while still valid, "
                "will simply be at its floor."
                if mess < 0 else
                "MESS >= 0: the target is inside the training range of every "
                "variable; the models interpolate rather than extrapolate."),
            "reference": "Elith, Kearney & Phillips 2010, Methods Ecol Evol 1:330-342",
        }
        self.results[key] = r
        return r

    # ==================================================================
    #  UNCERTAINTY
    # ==================================================================

    def compute_uncertainty(self):
        core = self._run_core()
        c = self._cache["full"]
        n = c["n"]
        alphas, fold_of = c["alphas"], c["fold_of"]
        a_t = c["alpha_tgt"][:, 0]

        rng = np.random.default_rng(self.random_state + 7)
        boot_p = []
        valid = np.where(np.isfinite(alphas) & (fold_of >= 0))[0]
        if valid.size > 5 and self.n_bootstrap:
            tgt_per_rec = a_t[fold_of[valid]]
            base = alphas[valid] >= tgt_per_rec
            for _ in range(self.n_bootstrap):
                idx = rng.integers(0, valid.size, valid.size)
                boot_p.append((1.0 + float(np.sum(base[idx]))) / (n + 1.0))
        boot_p = np.asarray(boot_p, dtype=float)

        refit_p = []
        if self.n_refit_bootstrap and n >= 30:
            for b in range(self.n_refit_bootstrap):
                saved_coords = self.coords
                try:
                    # Coordinates no longer line up with resampled rows, so
                    # fall back to random folds for the bootstrap replicate.
                    self.coords = None
                    r2 = np.random.default_rng(self.random_state + 1000 + b)
                    idx = r2.integers(0, n, n)
                    sub = self._crossconformal(self.Z[idx],
                                               self.z_t.reshape(1, -1),
                                               tag=f"_boot{b}")
                    refit_p.append(float(sub["p_values"][0]))
                except Exception:
                    continue
                finally:
                    self.coords = saved_coords
                    self._cache.pop(f"_boot{b}", None)
        refit_p = np.asarray(refit_p, dtype=float)

        def ci(arr):
            if arr.size < 10:
                return None
            lo, hi = np.percentile(arr, [2.5, 97.5])
            return [round(score_from_pvalue(lo), 2), round(score_from_pvalue(hi), 2)]

        r = {
            "point_score": round(score_from_pvalue(core["p_values"][0]), 2),
            "calibration_bootstrap": {
                "n": int(boot_p.size),
                "p_ci95": [round(float(np.percentile(boot_p, 2.5)), 5),
                           round(float(np.percentile(boot_p, 97.5)), 5)]
                if boot_p.size >= 10 else None,
                "score_ci95": ci(boot_p),
                "captures": "uncertainty from the finite calibration sample",
            },
            "refit_bootstrap": {
                "n": int(refit_p.size),
                "score_ci95": ci(refit_p),
                "captures": "uncertainty from model fitting as well",
            },
            "pvalue_resolution": core["resolution"],
            "note": ("A single occurrence set supports only a limited "
                     "resolution: the finest attainable p-value is "
                     f"{core['resolution']:.4f}."),
        }
        self.results["uncertainty"] = r
        return r

    # ==================================================================
    #  WEIGHT SENSITIVITY
    # ==================================================================

    def compute_weight_sensitivity(self):
        """How much does the answer depend on the ensemble weights?

        v1.0 hand-tuned five weights (one comment reads '↑ from 0.15') and
        never reported the sensitivity.  Here the default is equal weights and
        the alternatives are recomputed exactly, from the cached components.
        """
        self._run_core()
        c = self._cache["full"]
        n, fold_of = c["n"], c["fold_of"]

        schemes = {
            "equal (default)":      {"mahalanobis": .25, "gmm": .25,
                                     "isolation_forest": .25, "ocsvm": .25},
            "Mahalanobis-dominant": {"mahalanobis": .70, "gmm": .10,
                                     "isolation_forest": .10, "ocsvm": .10},
            "GMM-dominant":         {"mahalanobis": .10, "gmm": .70,
                                     "isolation_forest": .10, "ocsvm": .10},
            "anomaly-dominant":     {"mahalanobis": .10, "gmm": .10,
                                     "isolation_forest": .40, "ocsvm": .40},
            "v1.0 weights":         {"mahalanobis": .4464, "gmm": .2589,
                                     "isolation_forest": .1473, "ocsvm": .1473},
        }
        out = {}
        for name, w in schemes.items():
            a_rec = self._combine({k: c["comp_rec"][k] for k in self.MODEL_KEYS}, w)
            a_tgt = np.array([
                self._combine({k: c["comp_tgt"][fi, 0, mi]
                               for mi, k in enumerate(self.MODEL_KEYS)}, w)[0]
                for fi in range(c["comp_tgt"].shape[0])])
            p = self._pool_pvalue(a_rec, fold_of, a_tgt, n)
            out[name] = {"p": round(float(p), 5),
                         "score": round(score_from_pvalue(p), 2),
                         "zone": self.zone_from_pvalue(p)}
        vals = [v["score"] for v in out.values()]
        r = {"schemes": out,
             "score_range": [round(min(vals), 2), round(max(vals), 2)],
             "score_spread": round(max(vals) - min(vals), 2),
             "note": ("v1.0's effective weights are shown for comparison: its "
                      "nominal 0.25 for Mahalanobis became ~0.45 because the "
                      "threshold-zone term re-added the same four models.")}
        self.results["weight_sensitivity"] = r
        return r

    # ==================================================================
    #  VARIABLE IMPORTANCE
    # ==================================================================

    def compute_variable_importance(self):
        """Permutation importance on null-model discrimination, plus a
        transparent background-based Random Forest importance.

        v1.0 drew its 'background' as np.random.randn in the standardised
        space.  Because the occurrence data are themselves standardised, that
        background overlapped the presence cloud in every marginal, so the
        resulting importances were largely an artefact of the background
        design (Barbet-Massin et al. 2012).
        """
        self._run_core()
        c = self._cache["full"]

        # --- environmental background over the observed hyper-rectangle ---
        rng = np.random.default_rng(self.random_state + 31)
        lo, hi = self.Z.min(axis=0), self.Z.max(axis=0)
        pad = 0.10 * (hi - lo)
        n_bg = int(min(max(2 * self.n_records, 200), 800))
        BG = rng.uniform(lo - pad, hi + pad, size=(n_bg, self.Z.shape[1]))

        n_eval = int(min(self.n_records, 300))
        sel = rng.choice(self.n_records, n_eval, replace=False)
        Zr = self.Z[sel]
        y  = np.r_[np.zeros(n_eval), np.ones(len(BG))]

        def auc_of(Za, Zb):
            a = np.r_[self._alpha_oof_like(Za), self._alpha_oof_like(Zb)]
            try:
                return float(roc_auc_score(y, a))
            except Exception:
                return 0.5

        base_auc = auc_of(Zr, BG)
        perm_imp = {}
        for j, name in enumerate(self.fu_names):
            Za, Zb = Zr.copy(), BG.copy()
            Za[:, j] = Za[rng.permutation(n_eval), j]
            Zb[:, j] = Zb[rng.permutation(len(BG)), j]
            perm_imp[name] = max(0.0, base_auc - auc_of(Za, Zb))
        tot = sum(perm_imp.values()) or 1.0
        perm_imp = {k: v / tot for k, v in perm_imp.items()}

        # --- Random Forest on the same transparent background -------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                        random_state=self.random_state,
                                        n_jobs=-1, class_weight="balanced")
            rf.fit(np.vstack([self.Z, BG]),
                   np.r_[np.ones(self.n_records), np.zeros(len(BG))])
        rf_imp = dict(zip(self.fu_names, rf.feature_importances_.tolist()))
        self._models["rf"] = rf

        # --- marginal deviation (unchanged in spirit) ---------------------
        ks = {}
        for j, col in enumerate(self.bio_cols):
            ks[col] = round(float(abs(percentileofscore(
                self.X_bio[:, j], self.t_bio[j], kind="rank") - 50) / 50), 4)
        for k, oc in enumerate(self.opt_cols):
            i = self._opt_offset + k
            ks[oc] = round(float(abs(percentileofscore(
                self.species_data[:, i], float(self.target_values[i]),
                kind="rank") - 50) / 50), 4)
        for col in self.topo_cols:
            if col == "aspect_deg":
                continue
            i = self._topo_idxs[col]
            ks[col] = round(float(abs(percentileofscore(
                self.species_data[:, i], float(self.target_values[i]),
                kind="rank") - 50) / 50), 4)
        max_ks = max(list(ks.values()) + [1e-9])

        combined = {}
        for name in self.fu_names:
            p_ = perm_imp.get(name, 0.0)
            k_ = ks.get(name, 0.0) / max_ks
            combined[name] = round(0.5 * p_ + 0.5 * k_ if name in ks else p_, 5)
        sorted_imp = dict(sorted(combined.items(), key=lambda x: -x[1]))

        top_raw = [c_ for c_ in sorted_imp
                   if c_ not in ("northness", "eastness")][:7]
        risk_details = {}
        for col in top_raw:
            rec = self._get_pct_rec(col)
            if rec:
                risk_details[col] = {
                    "combined_importance": round(sorted_imp.get(col, 0), 5),
                    "permutation_importance": round(perm_imp.get(col, 0), 5),
                    "rf_importance": round(rf_imp.get(col, 0), 5),
                    "ks_deviation": round(ks.get(col, 0), 4),
                    "risk_level": rec.get("risk_level", "—"),
                    "risk_color": rec.get("risk_color", "#7f8c8d"),
                    "risk_emoji": rec.get("risk_emoji", "—"),
                    "score": rec.get("score", 0),
                    "dev_pct": rec.get("dev_pct", 0),
                    "n_std": rec.get("n_std", 0),
                    "percentile": rec.get("percentile", 0),
                    "risk_explanation": rec.get("risk_explanation", ""),
                }

        r = {
            "primary_measure": "permutation importance on null-model discrimination AUC",
            "baseline_auc": round(base_auc, 4),
            "permutation_importance": {k: round(v, 5) for k, v in perm_imp.items()},
            "rf_importance": {k: round(v, 5) for k, v in rf_imp.items()},
            "ks_deviation": ks,
            "combined_importance": sorted_imp,
            "top5_variables": top_raw[:5],
            "top7_variables": top_raw,
            "risk_details": risk_details,
            "background": ("uniform over the observed environmental "
                           "hyper-rectangle, padded by 10 %"),
            "caveat": ("Bio1-19 are strongly collinear, so importance is "
                       "shared arbitrarily among correlated variables; "
                       "%d variable(s) were pruned before modelling. Read the "
                       "ranking as a group-level, not a variable-level, "
                       "statement." % len(self.collinearity["dropped"])),
        }
        self.results["variable_importance"] = r
        return r

    def _get_pct_rec(self, col):
        pr = self.results.get("percentile", {})
        if col in pr.get("per_bio", {}): return pr["per_bio"][col]
        if col in pr.get("per_opt", {}): return pr["per_opt"][col]
        t = self.results.get("topo", {})
        if col == "elevation_m" and t.get("elevation"): return t["elevation"]
        if col == "slope_pct" and t.get("slope"):       return t["slope"]
        if col in self.bio_cols:
            j = self.bio_cols.index(col)
            return self._pct_record(self.X_bio[:, j], float(self.t_bio[j]), col)
        if col in self.opt_cols:
            i = self._opt_offset + self.opt_cols.index(col)
            return self._pct_record(self.species_data[:, i],
                                    float(self.target_values[i]), col)
        if col in self.topo_cols:
            i = self._topo_idxs[col]
            return self._pct_record(self.species_data[:, i],
                                    float(self.target_values[i]), col)
        return None

    # ==================================================================
    #  CLIMATE-CHANGE RESPONSE  (only when a future target is supplied)
    # ==================================================================

    def compute_climate_change(self):
        core = self._run_core()
        if self.target_future is None:
            r = {"available": False,
                 "note": ("No future-climate values supplied. The analysis is "
                          "SITE MATCHING under the current climate, not a "
                          "climate-change assessment: supply target_future "
                          "(e.g. CMIP6 SSP bioclim values for the same site) "
                          "to obtain a response.")}
            self.results["climate_change"] = r
            return r

        p_now = float(core["p_values"][0])
        p_fut = float(core["p_values"][self._idx_future])
        s_now, s_fut = score_from_pvalue(p_now), score_from_pvalue(p_fut)
        mess_fut = self.compute_extrapolation(self.f_target_future,
                                              key="extrapolation_future")

        shifts = {}
        for j, col in enumerate(self.col_names):
            cur, fut = float(self.target_values[j]), float(self.target_future[j])
            sp = self.species_data[:, j]
            shifts[col] = {
                "current": round(cur, 4), "future": round(fut, 4),
                "delta": round(fut - cur, 4),
                "current_percentile": round(float(percentileofscore(sp, cur, kind="rank")), 1),
                "future_percentile": round(float(percentileofscore(sp, fut, kind="rank")), 1),
            }
        moved = sorted(shifts.items(),
                       key=lambda kv: -abs(kv[1]["future_percentile"]
                                           - kv[1]["current_percentile"]))[:5]

        delta = s_fut - s_now
        if delta <= -15:   direction = "STRONG DECLINE"
        elif delta <= -5:  direction = "DECLINE"
        elif delta < 5:    direction = "STABLE"
        elif delta < 15:   direction = "IMPROVEMENT"
        else:              direction = "STRONG IMPROVEMENT"

        r = {
            "available": True,
            "current": {"p": round(p_now, 5), "score": round(s_now, 2),
                        "zone": self.zone_from_pvalue(p_now),
                        "class": self._classify(s_now)},
            "future": {"p": round(p_fut, 5), "score": round(s_fut, 2),
                       "zone": self.zone_from_pvalue(p_fut),
                       "class": self._classify(s_fut),
                       "mess": mess_fut["mess"],
                       "is_extrapolation": mess_fut["is_extrapolation"]},
            "delta_score": round(delta, 2),
            "direction": direction,
            "class_change": (self._classify(s_now)[0] + " -> "
                             + self._classify(s_fut)[0]),
            "variable_shifts": shifts,
            "largest_niche_shifts": [k for k, _ in moved],
            "caveat": ("Both scores are conditioned on the same present-day "
                       "occurrence records, i.e. on the species' realised "
                       "niche as currently observed. Adaptation, plasticity, "
                       "dispersal and biotic interactions are not modelled, "
                       "and a single GCM/SSP realisation carries no scenario "
                       "uncertainty - run several and compare."),
        }
        self.results["climate_change"] = r
        return r

    # ==================================================================
    #  TARGET AREA - CELL-WISE EVALUATION
    # ==================================================================

    def compute_target_area(self):
        """Score every target cell, not just the aggregated centroid.

        v1.0 averaged the environmental values over the whole target polygon
        and evaluated that single mean vector.  Suitability is a non-linear
        function of the environment, so by Jensen inequality the mean of a
        heterogeneous area can sit inside the niche while few or none of its
        cells do (the ecological fallacy).  Here the aggregate is kept as a
        representative summary and the cell-wise distribution is reported
        alongside it.
        """
        core = self._run_core()
        if self._idx_cells is None:
            r = {"available": False,
                 "note": ("Only one target row (or an aggregate) was supplied. "
                          "Pass the individual cells/points of the target area "
                          "to obtain the within-area distribution instead of a "
                          "single averaged value.")}
            self.results["target_area"] = r
            return r

        i0, i1 = self._idx_cells
        ps = np.array(core["p_values"][i0:i1], dtype=float)
        sc = np.array([score_from_pvalue(x) for x in ps])

        def _cls(v):
            return ("A" if v >= 80 else "B" if v >= 65 else
                    "C" if v >= 50 else "D" if v >= 35 else "E")
        counts = {k: 0 for k in "ABCDE"}
        for v in sc:
            counts[_cls(v)] += 1
        n = max(1, sc.size)

        agg_p = float(core["p_values"][0])
        agg_s = score_from_pvalue(agg_p)
        r = {
            "available": True,
            "n_cells_evaluated": int(sc.size),
            "n_cells_supplied": int(len(self.target_cells)),
            "aggregate_score": round(agg_s, 2),
            "cell_score_mean": round(float(sc.mean()), 2),
            "cell_score_median": round(float(np.median(sc)), 2),
            "cell_score_p5": round(float(np.percentile(sc, 5)), 2),
            "cell_score_p95": round(float(np.percentile(sc, 95)), 2),
            "cell_score_min": round(float(sc.min()), 2),
            "cell_score_max": round(float(sc.max()), 2),
            "class_counts": counts,
            "class_percent": {k: round(100.0 * v / n, 1) for k, v in counts.items()},
            "pct_cells_inside_niche_p05": round(float(np.mean(ps >= 0.05) * 100), 1),
            "pct_cells_rejected_p01": round(float(np.mean(ps < 0.01) * 100), 1),
            "aggregate_bias": round(agg_s - float(np.median(sc)), 2),
            "interpretation": (
                "The aggregated target scores %.1f while the median cell scores "
                "%.1f; %.0f%% of cells are not rejected at the 5%% level. A "
                "large gap between the two means the target area is "
                "environmentally heterogeneous, so a single averaged value "
                "misrepresents it."
                % (agg_s, float(np.median(sc)),
                   float(np.mean(ps >= 0.05) * 100))),
        }
        self.results["target_area"] = r
        return r

    # ==================================================================
    #  COMPOSITE
    # ==================================================================

    def compute_composite_score(self):
        core = self._run_core()
        for key, fn in (("percentile", self.compute_percentile_analysis),
                        ("topo", self.compute_topo_score),
                        ("threshold_zone", self.compute_threshold_zone),
                        ("variable_importance", self.compute_variable_importance),
                        ("validation", self.compute_validation),
                        ("extrapolation", self.compute_extrapolation),
                        ("uncertainty", self.compute_uncertainty),
                        ("weight_sensitivity", self.compute_weight_sensitivity),
                        ("climate_change", self.compute_climate_change),
                        ("target_area", self.compute_target_area)):
            if key not in self.results:
                fn()

        p = float(core["p_values"][0])
        composite = score_from_pvalue(p)

        # Sub-scores on the climate-only and topography-only subspaces.
        p_clim = None
        if self.compute_subspaces:
            sub = self._subspace({"bio", "opt"}, "climate")
            if sub is not None:
                p_clim = float(sub["p_values"][0])
        clim_score = (score_from_pvalue(p_clim) if p_clim is not None
                      else composite)
        topo_score = self.results["topo"].get("score")

        cw, tw = self._clim_topo_weights()
        mess = self.results["extrapolation"]["mess"]
        ci = (self.results["uncertainty"]["calibration_bootstrap"]
              .get("score_ci95"))

        r = {
            "composite_score": round(composite, 2) if np.isfinite(composite) else None,
            "conformal_p": round(p, 5),
            "score_ci95": ci,
            "climate_score": round(clim_score, 2) if clim_score is not None else None,
            "climate_conformal_p": round(p_clim, 5) if p_clim is not None else None,
            "topo_score": round(topo_score, 2) if topo_score is not None else None,
            "topo_conformal_p": self.results["topo"].get("conformal_p"),
            "opt_bonus": 0.0,
            "climate_weight": cw, "topo_weight": tw,
            "weighting_note": ("climate_weight / topo_weight are the SHARE OF "
                               "MODEL VARIABLES in each block, not scoring "
                               "weights: the composite comes from one joint "
                               "statistic, so no arbitrary climate/topography "
                               "split is applied. v1.0 applied 75/25 in code "
                               "while documenting 80/20."),
            "active_bio_cols": self.bio_cols,
            "active_topo_cols": self.topo_cols,
            "active_opt_cols": self.opt_cols,
            "component_scores": {
                "threshold_zone":   self.results["threshold_zone"]["score"],
                "gmm":              self.results["gmm"]["score"],
                "isolation_forest": self.results["isolation_forest"]["score"],
                "ocsvm":            self.results["ocsvm"]["score"],
                "mahalanobis":      self.results["mahalanobis_stat"]["score"],
                "topo":             topo_score,
            },
            "component_weights": {
                "threshold_zone":   "read-out",
                "gmm":              ENSEMBLE_WEIGHTS["gmm"],
                "isolation_forest": ENSEMBLE_WEIGHTS["isolation_forest"],
                "ocsvm":            ENSEMBLE_WEIGHTS["ocsvm"],
                "mahalanobis":      ENSEMBLE_WEIGHTS["mahalanobis"],
                "topo":             "in joint statistic",
            },
            "final_zone": self.results["threshold_zone"]["final_zone"],
            "zone_label": self.results["threshold_zone"]["zone_label"],
            "resilience_class": self._classify(composite),
            "recommendation": self._recommend(composite, mess),
            "top_risk_variables": self.results["variable_importance"]["top5_variables"],
            "mess": mess,
            "is_extrapolation": self.results["extrapolation"]["is_extrapolation"],
            "mode": self.results["engine"]["mode"],
            "interpretation": (
                "Score %s corresponds to a cross-conformal p-value of %.4f for "
                "the hypothesis that the target site is exchangeable with this "
                "species' occurrence records. Under that hypothesis the score "
                "exceeds 80 with probability 0.80 and falls below 35 with "
                "probability 0.01."
                % (f"{composite:.1f}" if np.isfinite(composite) else "n/a", p)),
        }
        self.results["composite"] = r
        return r

    # -- classification and wording ---------------------------------------

    @staticmethod
    def _classify(s):
        """NaN-safe.  v1.0's version let NaN fall through every comparison and
        reported it as 'E - Very Low Resilience / Unsuitable Site'."""
        if s is None or not np.isfinite(s):
            return "— Indeterminate (insufficient or degenerate data)"
        if s >= 80: return "A – Very High Niche Similarity"
        if s >= 65: return "B – High Niche Similarity"
        if s >= 50: return "C – Moderate Niche Similarity"
        if s >= 35: return "D – Low Niche Similarity"
        return "E – Very Low / Outside the Observed Niche"

    def _recommend(self, s, mess=None):
        if s is None or not np.isfinite(s):
            return ("No score could be computed. Check the sample size, the "
                    "variable selection and the input values before drawing "
                    "any conclusion. Do NOT read this as an unsuitable site.")

        future = self.target_future is not None
        head = ("Under the supplied future scenario, " if future else
                "Under the current climate, ")
        if s >= 80:
            body = ("the target is statistically indistinguishable from the "
                    "species' occurrence records. It is a primary candidate "
                    "site on environmental grounds.")
        elif s >= 65:
            body = ("the target lies inside the observed niche but towards its "
                    "edge. Establishment is plausible; monitor the variables "
                    "listed under Variable Importance.")
        elif s >= 50:
            body = ("the target sits at the tolerance boundary (p < 0.10). Use "
                    "with caution and plan mitigation for the limiting "
                    "variables.")
        elif s >= 35:
            body = ("the hypothesis that the target belongs to the observed "
                    "niche is rejected at the 5 % level. High risk; evaluate "
                    "alternative species or provenances.")
        else:
            body = ("the hypothesis is rejected at the 1 % level: the target "
                    "is outside the species' observed environmental niche.")

        tail = ""
        if mess is not None and np.isfinite(mess) and mess < 0:
            tail = (" CAUTION: MESS < 0 - the target is outside the training "
                    "range for at least one variable, so this is an "
                    "extrapolation and the score is a lower bound only.")
        if not future:
            tail += (" Note: without future-climate input this is site "
                     "matching under the present climate, not a climate-change "
                     "projection.")
        val = self.results.get("validation", {})
        ksp = val.get("ks_uniformity_pvalue")
        if ksp is not None and ksp < 0.05:
            tail += (" Note: the calibration check failed (KS p = %.4f); treat "
                     "the exact p-value as approximate." % ksp)
        return head + body + tail

    # -- driver ------------------------------------------------------------

    def run_all(self, progress_cb=None):
        steps = [
            ("Cross-conformal calibration…",      self._run_core),
            ("Ordination (diagnostic)…",          self.compute_pca),
            ("Kernel PCA (diagnostic)…",          self.compute_kernel_pca),
            ("Gaussian mixture…",                 self.compute_gmm),
            ("Isolation Forest…",                 self.compute_isolation_forest),
            ("One-Class SVM…",                    self.compute_ocsvm),
            ("K-Means (diagnostic)…",             self.compute_kmeans_niche),
            ("Mahalanobis (shrinkage)…",          self.compute_mahalanobis),
            ("Ensemble zone…",                    self.compute_threshold_zone),
            ("Per-variable analysis…",            self.compute_percentile_analysis),
            ("Topographic compatibility…",        self.compute_topo_score),
            ("Extrapolation (MESS)…",             self.compute_extrapolation),
            ("Cross-validation & calibration…",   self.compute_validation),
            ("Variable importance…",              self.compute_variable_importance),
            ("Bootstrap uncertainty…",            self.compute_uncertainty),
            ("Weight sensitivity…",               self.compute_weight_sensitivity),
            ("Climate-change response…",          self.compute_climate_change),
            ("Target-area cell distribution…",     self.compute_target_area),
            ("Composite score…",                  self.compute_composite_score),
        ]
        for i, (msg, fn) in enumerate(steps):
            if progress_cb:
                progress_cb(int(100 * i / len(steps)), msg)
            fn()
        if progress_cb:
            progress_cb(100, "Done.")
        self.results["engine"]["warnings"] = list(self._warnings)
        return self.results
