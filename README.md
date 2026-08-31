<p align="center">
  <img src="resources/icon.png" width="80" alt="CRESTA icon"/>
</p>

<h1 align="center">CRESTA</h1>
<p align="center">
  <strong>Climate Resilience Ensemble Score &amp; Topographic Analysis</strong><br>
  Conformal bioclimatic niche analysis for QGIS · v2.0.0
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License"/></a>
  <img src="https://img.shields.io/badge/QGIS-3.16%2B%20%7C%204.x-green" alt="QGIS"/>
  <img src="https://img.shields.io/badge/Qt-5%20%7C%206-informational" alt="Qt"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-yellow" alt="Python"/>
</p>

---

## What CRESTA does

Given the environmental values recorded at a species' **occurrence points** and
the values of a **target site**, CRESTA tests one hypothesis:

> **H₀** — the target site is *exchangeable* with the species' occurrence
> records, i.e. it lies inside the species' realised environmental niche.

It returns a calibrated **p-value** for that hypothesis and a 0–100 score that
is a strictly monotone re-expression of it. If you also supply **future-climate
values** for the same site, it computes the score under that scenario and
reports the change (ΔScore).

### Why the score means something

The score is a piecewise-linear map of the cross-conformal p-value
(Vovk 2015), which is Uniform(0,1) under H₀ by construction. That gives the
class thresholds an exact frequentist meaning:

| Under H₀ | Probability |
|---|---|
| score ≥ 80 (class A) | **0.80** |
| score ≥ 65 (class A or B) | **0.90** |
| score ≥ 50 (class ≥ C) | **0.95** |
| score < 35 (class E) | **0.01** |

In other words, the chance of declaring a genuinely in-niche site
"unsuitable" is controlled at 1 %. Every run re-measures this on *your* data
and reports it in the **🧪 Validation** tab — you do not have to take it on
trust.

### Classes

| Class | Score | Conformal p | Meaning |
|---|---|---|---|
| **A** | ≥ 80 | ≥ 0.20 | Statistically indistinguishable from the occurrence records |
| **B** | ≥ 65 | ≥ 0.10 | Inside the niche, towards its edge |
| **C** | ≥ 50 | ≥ 0.05 | At the tolerance boundary |
| **D** | ≥ 35 | ≥ 0.01 | Rejected at the 5 % level |
| **E** | < 35 | < 0.01 | Rejected at the 1 % level — outside the observed niche |

---

## Method

| Stage | What happens |
|---|---|
| **Features** | Aspect is encoded circularly as (northness, eastness) in **every** stage, Mahalanobis included |
| **Collinearity** | Bio1–19 are strongly correlated; pairs with \|r\| ≥ 0.95 are pruned before modelling, and the effective dimension + condition number are reported |
| **Ensemble** | Mahalanobis (Ledoit-Wolf shrinkage), GMM, Isolation Forest, One-Class SVM — standardised on the fit fold, **equal weights** |
| **Calibration** | 5-fold cross-conformal; **spatial block folds** when coordinates are supplied (Roberts et al. 2017) |
| **Validation** | KS test of p-value uniformity, empirical type-I error, class distribution of the species' own records, permutation null-model AUC (Raes & ter Steege 2007) |
| **Uncertainty** | Percentile bootstrap CI (calibration set + model refit) |
| **Extrapolation** | MESS (Elith et al. 2010) — flags targets outside the training range |
| **Sensitivity** | The score recomputed under four alternative weightings, so weight choices are auditable |

PCA, Kernel PCA and K-Means are computed for **visualisation only** and are
explicitly labelled as such — they do not enter the score.

Because the p-value comes from a rank, the model hyperparameters (tree count,
`nu`, mixture components) affect the test's **sharpness only, never its
validity**.

---

## Features

- **Dual input** — QGIS vector layers *or* plain CSV
- **19 bioclimatic variables** (WorldClim v1/v2, CHELSA) with **automatic unit
  detection** (°C vs °C×10)
- **Topography** — elevation, slope, aspect (circular)
- **Optional** — solar radiation, wind speed, vapour pressure (VPD computed
  from the Tetens equation at the site's own temperature)
- **Cell-wise target areas** — a multi-row target is scored cell by cell and
  the within-area distribution is reported; the representative aggregate uses
  the median and the **circular** mean for aspect
- **Climate-change mode** — supply future bioclim values to get ΔScore
- **Spatial-block cross-validation** from optional `x`/`y` (or `lon`/`lat`) columns
- **Refuses to guess** — with n ≤ p + 2 the covariance is singular, so the
  analysis stops with an explanation instead of returning a number
- Qt5/Qt6 dual compatibility (QGIS 3.16+ and 4.x)

---

## Requirements

| Package | Version | |
|---|---|---|
| numpy | ≥ 1.21 | shipped with QGIS |
| scipy | ≥ 1.7 | shipped with QGIS |
| **scikit-learn** | **≥ 1.0** | **NOT shipped with QGIS — install it** |
| matplotlib | ≥ 3.5 | optional (Charts tab); usually shipped |

> ⚠️ scikit-learn is **not** part of a standard QGIS installation (verified on
> QGIS 3.40 LTR and 4.2). CRESTA detects this and shows an install hint rather
> than a traceback.

**Windows** — OSGeo4W Shell (as administrator):

```
python -m pip install scikit-learn>=1.0
```

**Linux / macOS:**

```bash
python3 -m pip install scikit-learn>=1.0
```

Then restart QGIS.

---

## Installation

1. Download `CRESTA.zip` from Releases.
2. QGIS → **Plugins → Manage and Install Plugins → Install from ZIP**.
3. Install the requirement above, then restart QGIS.

---

## Input data format

Columns named **bio1 … bio19** (case-insensitive). Optional columns:

| Column | Description | Unit |
|---|---|---|
| `bio1` … `bio19` | WorldClim / CHELSA bioclimatic variables | see below |
| `elevation_m` | Elevation | m |
| `slope_pct` | Slope | % |
| `aspect_deg` | Aspect (north = 0°) | degrees |
| `srad` / `wind` / `vapr` | Solar radiation / wind / vapour pressure | kJ m⁻² day⁻¹ / m s⁻¹ / kPa |
| `x`, `y` *(or `lon`,`lat`)* | Coordinates — enable spatial-block CV | any projected/geographic CRS |

Temperature variables may be in **°C** (WorldClim v2, CHELSA) or **°C×10**
(WorldClim v1); the convention is detected automatically and reported.

**Sample size.** At least **3 × the number of variables** is recommended.
Below *n = p + 2* the analysis is refused. With 19 bio + 3 topo variables that
means ≥ 69 occurrence records recommended, 25 the hard floor.

**Target area.** Supply the individual cells/points, not a pre-averaged row —
suitability is non-linear, so the mean of a heterogeneous area can fall inside
the niche when almost none of its cells do.

---

## Limitations (please read)

- CRESTA describes the **realised** niche, which is truncated by dispersal
  limitation, biotic interactions and sampling bias (Soberón & Nakamura 2009).
  It is not the fundamental niche and it is not a physiological model.
- It does **not** correct sampling bias beyond optional spatial blocking of the
  CV folds. Thin your occurrence records first (Aiello-Lammens et al. 2015).
- **Without future-climate input the analysis is site matching under the
  present climate, not a climate-change projection.** The plugin says so in
  the report and in the recommendation text.
- A single GCM/SSP realisation carries no scenario uncertainty. Run several and
  compare.
- Variable importance is shared arbitrarily among collinear predictors; read it
  as a group-level statement.
- The "biological rationale" texts are **generic, sourced statements about
  plants**, not inferences about the analysed species. They carry a standing
  disclaimer.

---

## Testing

```bash
python test_engine.py
```

19 tests covering calibration validity, circular aspect, singular-sample
refusal, NaN safety, collinearity detection, cell-wise target areas,
climate-change mode, and the result contract the GUI depends on.

---

## References

- Aiello-Lammens ME et al. (2015) *spThin*. Ecography 38:541–545.
- Barbet-Massin M et al. (2012) Selecting pseudo-absences. Methods Ecol Evol 3:327–338.
- Elith J, Kearney M, Phillips S (2010) The art of modelling range-shifting species. Methods Ecol Evol 1:330–342.
- Ledoit O, Wolf M (2004) A well-conditioned estimator for large-dimensional covariance matrices. J Multivar Anal 88:365–411.
- Raes N, ter Steege H (2007) A null-model for significance testing of presence-only SDMs. Ecography 30:727–736.
- Roberts DR et al. (2017) Cross-validation strategies for data with spatial structure. Ecography 40:913–929.
- Soberón J, Nakamura M (2009) Niches and distributional areas. PNAS 106:19644–19650.
- Vovk V (2015) Cross-conformal predictors. Ann Math Artif Intell 74:9–28.

---

## Citation

```
Örücü, Ö. K. (2026). CRESTA: conformal bioclimatic niche analysis for QGIS
(v2.0.0). https://github.com/omerorucu/CRESTA
```

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Author

**Ömer K. Örücü** · Süleyman Demirel University · [omerorucu@sdu.edu.tr](mailto:omerorucu@sdu.edu.tr)
