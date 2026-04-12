<p align="center">
  <img src="resources/icon.svg" width="80" alt="CRESTA icon"/>
</p>

<h1 align="center">CRESTA</h1>
<p align="center">
  <strong>Climate Resilience Ensemble Score &amp; Topographic Analysis</strong><br>
  QGIS Plugin · v1.0.0
</p>

<p align="center">
  <a href="https://github.com/omerorucu/CRESTA/releases"><img src="https://img.shields.io/github/v/release/omerorucu/CRESTA?color=2ecc71" alt="Release"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-blue" alt="License"/></a>
  <img src="https://img.shields.io/badge/QGIS-3.16%2B%20%7C%204.x-green" alt="QGIS"/>
  <img src="https://img.shields.io/badge/Qt-5%20%7C%206-informational" alt="Qt"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-yellow" alt="Python"/>
</p>

---

## Overview

**CRESTA** quantifies the **climate change resilience** of plant species by comparing the bioclimatic niche of a species' native distribution against a target area, producing a single composite score (0–100) and a letter-grade resilience class.

The analysis engine combines five ML/statistical layers:

| Layer | Method | Weight |
|-------|--------|--------|
| A | PCA + Kernel PCA (dimensionality reduction) | — |
| B | GMM · Isolation Forest · One-Class SVM · K-Means | — |
| C | 4-threshold zone voting + Mahalanobis χ² | 35 % |
| D | Variable importance (Random Forest + KS deviation) | — |
| E | Composite score  *(climate 80 % · topography 20 %)* | — |

### Resilience Classes

| Class | Score | Interpretation |
|-------|-------|----------------|
| **A** | ≥ 80 | Very High — target area within the species' core niche |
| **B** | ≥ 65 | High — suitable niche conditions |
| **C** | ≥ 50 | Moderate — marginal conditions |
| **D** | ≥ 35 | Low — outside typical niche |
| **E** | < 35 | Very Low — highly unfavourable |

---

## Features

- **Dual input mode** — QGIS vector layers *or* plain CSV files
- **19 bioclimatic variables** (Bio1–Bio19, WorldClim / CHELSA compatible)
- **Topographic variables** — elevation, slope, aspect (optional, individually toggleable)
- **Optional extra variables** — solar radiation, wind speed, vapour pressure
- **Interactive results** — Summary, ML Models, Bio Detail, Topography, Variable Importance, Risk Details, Charts (matplotlib), Report
- **Export** — JSON (full results) and CSV (per-Bio table)
- **Qt5 / Qt6 dual compatibility** — runs on QGIS 3.16+ and QGIS 4.x unchanged

---

## Installation

### From ZIP (recommended)

1. Download the latest `CRESTA.zip` from [Releases](https://github.com/omerorucu/CRESTA/releases).
2. In QGIS: **Plugins → Manage and Install Plugins → Install from ZIP**.
3. Browse to `CRESTA.zip` → **Install Plugin**.

### Manual

```bash
# Windows
xcopy /E CRESTA "%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\CRESTA\"

# Linux / macOS
cp -r CRESTA ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
```

Then enable it in **Plugins → Manage and Install Plugins**.

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.8 |
| numpy | ≥ 1.21 |
| scipy | ≥ 1.7 |
| scikit-learn | ≥ 1.0 |
| matplotlib | ≥ 3.5 *(optional — charts tab)* |

These packages are included in the QGIS Python environment. If `matplotlib` is missing the Charts tab is simply hidden.

---

## Input Data Format

### Species native range layer / CSV

Must contain columns named **bio1 … bio19** (case-insensitive). Topographic columns are optional:

| Column | Description | Unit |
|--------|-------------|------|
| `bio1` … `bio19` | WorldClim bioclimatic variables | see WorldClim docs |
| `elevation_m` | Elevation above sea level | m |
| `slope_pct` | Terrain slope | % |
| `aspect_deg` | Aspect (north = 0°) | degrees |
| `srad` | Solar radiation *(optional)* | kJ m⁻² day⁻¹ |
| `wind` | Wind speed *(optional)* | m s⁻¹ |
| `vapr` | Vapour pressure *(optional)* | kPa |

### Target area layer / CSV

A single row (or multiple rows — CRESTA takes the row-wise mean) with the same columns as above representing the target location.

---

## Usage

1. Load your layers in QGIS (or prepare CSV files).
2. Open **Plugins → CRESTA → CRESTA — Climate Resilience Analyzer**.
3. Select the **native range** source and **target area** source.
4. Choose which bioclimatic, topographic, and optional variables to include.
5. Click **▶ Run Analysis**.
6. Browse results across the tabbed interface and export as needed.

---

## Citation

If you use CRESTA in academic work please cite:

```
Örücü, Ö. K. (2025). CRESTA: Climate Resilience Ensemble Score & Topographic
Analysis — A QGIS plugin for species climate change vulnerability assessment.
GitHub. https://github.com/omerorucu/CRESTA
```

---

## License

CRESTA is released under the **GNU General Public License v3.0**.  
See [LICENSE](LICENSE) for the full text.

---

## Author

**Ömer K. Örücü**  
Süleyman Demirel University  
[omerorucu@sdu.edu.tr](mailto:omerorucu@sdu.edu.tr)

*Developed with Claude AI (Anthropic)*
