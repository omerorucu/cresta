# Changelog

All notable changes to CRESTA will be documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] — 2025-03-16

### Added
- 5-layer ML ensemble: PCA, Kernel PCA, GMM, Isolation Forest, One-Class SVM, K-Means, Mahalanobis χ²
- 4-threshold zone voting (CORE / SUITABLE / MARGINAL / OUTSIDE)
- Random Forest + KS deviation variable importance with biological risk explanations
- Composite scoring: climate 80 % · topography 20 %
- Topographic compatibility (elevation, slope, aspect — circular cosine method)
- Optional extra variables: solar radiation, wind speed, vapour pressure
- Dual input mode: QGIS vector layers and plain CSV files
- Interactive tabbed GUI: Summary, ML Models, Bio Detail, Topography, Variable Importance, Risk Details, Charts, Report
- JSON and CSV export
- matplotlib Charts tab (optional dependency)
- Qt5 / Qt6 dual compatibility — QGIS 3.16+ and QGIS 4.x

### Fixed
- Qt6: replaced all unscoped enum usages (`QFrame.Box`, `QFont.Bold`, `QPainter.Antialiasing`, etc.) with scoped shims
- Qt6: replaced deprecated `exec_()` with `exec()`
