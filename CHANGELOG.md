# Changelog

All notable changes to CRESTA are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.2.0] — 2026-11-15

### Fixed — bounded ensemble members ignored extrapolation

- Isolation-forest path lengths and the RBF one-class-SVM decision value are
  bounded: beyond the occurrence data they settle on a constant. In a
  synthetic warming test (*Pinus brutia*, bio1 + bio5 shifted by k SD) the
  isolation forest alone returned p = 0.41 at 3, 10, 20 and 50 SD, so the
  two members diluted the unbounded Mahalanobis and GMM members.
- Both members now add the Euclidean distance (standardised units) by which
  a point lies outside the per-variable range of the records each fold model
  was fitted on (`EXTRAPOLATION_PENALTY`, `BOUNDED_MEMBERS`). The term is 0
  inside that envelope, so in-range scores are unchanged, and it is part of
  the score function fixed before calibration, so conformal validity holds.
  Same test: isolation forest alone p = 0.070 at 3 SD and 1/(n+1) from 10 SD.
- `EXTRAPOLATION_PENALTY = False` reproduces v2.1.

## [2.1.0] — not released separately; included in 2.2.0

### Fixed — validation was tautological

- v2.0's 🧪 Validation tab tested `self_p`, the rank of each record's
  out-of-fold score among the pooled out-of-fold scores. Ranks of any
  continuous statistic are Uniform by construction, so the KS test, the
  empirical type-I error and the "own records in class A" share could not
  fail whatever the model did. The figures quoted for v2.0 below
  (A = 81 %, type-I 0.83 %) are therefore not evidence of calibration.
- Validation now uses **held-out p-values**: each outer fold (spatial blocks
  when coordinates are supplied) is withheld, a complete cross-conformal
  predictor is fitted on the remaining records (inner folds spatial as
  well), and the withheld records are scored as new targets.
- New diagnostic `rejection_rate_if_random_calibration`: the same spatial
  hold-out scored by a predictor calibrated with random folds. It shows how
  optimistic a randomly cross-validated calibration would be at new
  locations (Roberts et al. 2017). A new test (`test_validation_can_fail`)
  confirms the check is falsifiable.

### Added

- `ClimateResilienceAnalyzer.score_matrix(rows)`: scores any number of
  target rows (cells × scenarios, or other species' ranges) with the cached
  fold models. The fold models depend on the occurrence records only, so the
  result is bit-identical to passing the rows as targets
  (`test_batch_scoring_equals_target_scoring`), at a fraction of the cost.
  Returns p, score, class and vectorised MESS.

## [2.0.0] — 2026-08-30

Scientific overhaul in response to a peer review of v1.0. The score's
definition changed, so **v2.0 numbers are not comparable with v1.0 numbers**.

### Fixed — correctness

- **Singular samples no longer produce a fabricated verdict.** With v1.0's own
  default minimum (n = 15) and its default variable set (p = 22) the covariance
  was singular, `_safe_inv` fell back to a pseudo-inverse, the composite came
  out `NaN`, and `_classify(NaN)` fell through every comparison to report
  *"E – Very Low Resilience / This species is not suitable for this site"*.
  The analysis is now refused below *n = p + 2* with an explanation, warns
  below *3p*, and every classifier and recommendation path is NaN-guarded.
- **Aspect is circular everywhere.** v1.0 encoded aspect as
  northness/eastness for the ML models but fed raw degrees into the
  Mahalanobis covariance, so 359° and 1° were 358 units apart (measured: D =
  1.63 vs 1.05 for the same slope). There is now one feature space.
- **Target areas use the circular mean.** v1.0 aggregated multi-row targets
  with an arithmetic mean, turning two north-facing cells at 350° and 10° into
  180° — due south (measured: aspect score 95 → 5).
- **Target areas are scored cell by cell.** Averaging a heterogeneous polygon
  before evaluating it is an ecological fallacy; the within-area distribution
  is now reported alongside the representative aggregate.
- `resources/icon.png` was referenced by `metadata.txt`, `cresta.py` and
  `README.md` but the directory did not exist. Added.
- `LICENSE` contained only the GPL header and a URL; GPL-3.0 requires the full
  text to be distributed. Added.
- Both shipped test files raised `TypeError` on their first call (they passed a
  `has_topo=` argument removed from the constructor) and read a result key that
  no longer existed. Replaced with a 19-test suite that runs.

### Changed — statistics

- **Calibration by cross-conformal prediction** (Vovk 2015) replaces v1.0's
  hand-tuned score transforms (sigmoid slope `k = 0.7`, power `0.6`, power
  `1.5`, a five-branch piecewise Mahalanobis ramp, `score *= 0.5` outside
  P5/P95). The composite is now a monotone map of a p-value, so under H₀
  P(class A) = 0.80, P(A or B) = 0.90 and P(class E) = 0.01 by construction.
  Measured on v1.0: 50 % of a species' *own* occurrence records scored below
  class B and 10 % were reported unsuitable. Measured on v2.0: A = 81 %,
  A+B = 91 %, E = 0.8 %, empirical type-I error 0.83 % at α = 0.01.
- **Double counting removed.** v1.0's composite added a "threshold zone" term
  that was itself a weighted vote over the same four models, giving
  Mahalanobis an effective weight of ~0.44 while nominally weighting it 0.25.
  There is now one statistic; the zone is a read-out.
- **Mahalanobis reference distribution corrected.** v1.0 used χ² with df = p,
  which requires μ and Σ to be *known*; both were estimated from the same
  sample. Thresholds are now distribution-free empirical out-of-fold quantiles,
  with the correct F-distribution reference reported for comparison.
- **Ledoit-Wolf shrinkage covariance** replaces the sample covariance +
  pseudo-inverse, which returned distances up to 3 × 10⁷ near n ≈ p.
- **Collinearity control.** Pairs with |r| ≥ 0.95 are pruned from the
  multivariate stage; effective dimension and condition number are reported.
- **Equal ensemble weights** replace five hand-tuned ones (one v1.0 comment
  read `↑ from 0.15`), and a sensitivity analysis over four alternative
  weightings is reported every run.
- **No arbitrary climate/topography split.** v1.0 applied 75/25 in code while
  documenting 80/20; topography now enters the single joint statistic and the
  reported "weights" are the share of variables in each block.
- `contamination=0.10` on the Isolation Forest (which declared 10 % of the
  species' own records anomalous by fiat) → `contamination='auto'`.
- The tautological One-Class SVM `nu` selection loop was removed; under
  conformal calibration hyperparameters affect sharpness, not validity.
- The Mann-Whitney U test on a one-element sample — reported as `mw_pvalue` as
  though it were inferential — was replaced by an exact two-sided rank p-value.
- The per-variable percentile aggregate, PCA, Kernel PCA and K-Means never
  entered v1.0's composite despite being documented as scoring layers. They are
  now labelled `role: diagnostic_only` and the README says so.

### Added

- **Validation layer** (new 🧪 tab): KS test of p-value uniformity, empirical
  type-I error at α = 0.01/0.05/0.10/0.20, class distribution of the species'
  own records against the design targets, and a permutation null-model
  discrimination AUC (Raes & ter Steege 2007).
- **Cross-validation**, with **spatial block folds** when `x`/`y` (or
  `lon`/`lat`) columns are present (Roberts et al. 2017).
- **Uncertainty**: percentile bootstrap 95 % CI on the score, from both the
  calibration set and full model refits.
- **MESS extrapolation index** (Elith et al. 2010) with a limiting-variable
  read-out; extrapolating targets are flagged in the recommendation.
- **Climate-change mode**: supply future bioclim values for the target site and
  CRESTA reports current score, future score, ΔScore, direction, class change
  and the variables whose niche position moved most. Without it, the analysis
  is explicitly labelled *site matching under the present climate*.
- **Automatic temperature-unit detection** (°C vs °C×10). v1.0 assumed the
  WorldClim v1 ×10 convention unconditionally, silently corrupting every
  threshold statement made about WorldClim v2 / CHELSA input.
- **VPD from the Tetens equation** at the site's own mean annual temperature.
  v1.0 computed `VPD = 2.34 − vapr`, hard-coding the saturation vapour pressure
  of 20 °C air for every site on Earth.
- **Dependency check**: scikit-learn is not shipped with QGIS (verified on 3.40
  LTR and 4.2). CRESTA now shows an install hint instead of a traceback.
- `requirements.txt`.

### Changed — reporting

- Mechanistic "biological rationale" texts are now **sourced** (Choat et al.
  2012, Grossiord et al. 2020, Körner 2007, Murata et al. 2007, Pearce 2001,
  Sage & Kubien 2007, Knapp et al. 2008, Luedeling 2012, Gardiner et al. 2016,
  Kozlowski 1997) and carry a standing disclaimer that they are generic
  statements about plants, not inferences about the analysed species. v1.0
  asserted unsourced species-agnostic thresholds and gave a succulent and a
  boreal conifer the same physiological warning.
- Variable importance now uses permutation importance on null-model
  discrimination as the primary measure, with a transparent environmental
  background over the observed hyper-rectangle. v1.0 drew its "background"
  as `np.random.randn` in the standardised space, where it overlapped the
  presence cloud in every marginal (cf. Barbet-Massin et al. 2012).
- The report carries the method, the validation results, the caveats and the
  literature.
- Global `warnings.filterwarnings("ignore")` replaced with scoped suppression.

## [1.0.0] — 2025-03-16

- Initial release. 5-layer ML ensemble, Qt5/Qt6 dual compatibility, CSV + QGIS
  layer input, JSON/CSV export, matplotlib charts.
