"""
CRESTA engine test suite  (no QGIS required)
=============================================
    python test_engine.py            # all tests
    python -m pytest test_engine.py  # same, under pytest

v1.0 shipped two test files (test_engine.py, test_engine_v3.py) that both
raised TypeError on the first call because they passed a `has_topo=` argument
that had been removed from the constructor, and read a `niche_overlap` result
key that no longer existed.  Neither had run for a long time.  These tests do
run, and they assert the properties the engine actually claims.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_engine import (            # noqa: E402
    ClimateResilienceAnalyzer, InsufficientDataError,
    ALL_BIO_COLS, ALL_TOPO_COLS,
    circular_mean_deg, detect_temp_scale, saturation_vapour_pressure_kpa,
    score_from_pvalue, conformal_pvalue,
)

BIO_MEAN = [180, 95, 55, 600, 340, 30, 310, 220, 140,
            250, 55, 650, 90, 8, 73, 248, 30, 50, 200]
BIO_SD   = [22, 10, 5, 75, 18, 14, 23, 19, 18,
            20, 18, 95, 18, 4, 14, 48, 12, 18, 48]
# Loadings on a shared temperature factor: makes bio1/5/6/10/11 collinear the
# way real WorldClim data are.
LOAD = [1, .3, .2, .6, .95, .9, .4, .7, .6, .98, .92, 0, 0, 0, 0, 0, 0, 0, 0]


def make_data(n=150, seed=11):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    cols = []
    for j, (m, s) in enumerate(zip(BIO_MEAN, BIO_SD)):
        drv = f1 if j < 11 else f2
        cols.append(m + s * (LOAD[j] * drv +
                    np.sqrt(max(1e-6, 1 - LOAD[j] ** 2)) * rng.normal(0, 1, n)))
    bio = np.column_stack(cols)
    sp = np.hstack([bio,
                    rng.normal(750, 200, n).reshape(-1, 1),
                    np.abs(rng.normal(12, 6, n)).reshape(-1, 1),
                    (rng.normal(10, 25, n) % 360).reshape(-1, 1)])
    return sp, rng


def analyse(sp, target, **kw):
    kw.setdefault("n_refit_bootstrap", 0)
    kw.setdefault("compute_subspaces", False)
    return ClimateResilienceAnalyzer(
        sp, target, bio_cols=ALL_BIO_COLS, topo_cols=ALL_TOPO_COLS,
        opt_cols=[], **kw).run_all()


# ---------------------------------------------------------------- helpers

def test_circular_mean():
    assert abs(circular_mean_deg([350.0, 10.0])) < 1e-6, \
        "circular mean of 350 and 10 must be 0, not the arithmetic 180"
    assert abs(circular_mean_deg([80.0, 100.0]) - 90.0) < 1e-6


def test_temperature_unit_detection():
    v1 = np.array(BIO_MEAN[0] + np.zeros(20))          # degC x 10 convention
    assert detect_temp_scale(v1, "bio1")[0] == 10.0
    assert detect_temp_scale(v1 / 10.0, "bio1")[0] == 1.0   # WorldClim v2 / CHELSA
    assert detect_temp_scale(v1, "bio12")[0] == 1.0         # precipitation: n/a


def test_vpd_is_temperature_dependent():
    """v1.0 hard-coded e_s = 2.34 kPa, i.e. it assumed 20 degC air at every
    site on Earth.  The Tetens value must track temperature."""
    assert abs(saturation_vapour_pressure_kpa(20.0) - 2.34) < 0.02
    assert saturation_vapour_pressure_kpa(5.0) < 1.0
    assert saturation_vapour_pressure_kpa(35.0) > 5.0
    assert (saturation_vapour_pressure_kpa(5.0)
            < saturation_vapour_pressure_kpa(20.0)
            < saturation_vapour_pressure_kpa(35.0))


def test_score_map_is_monotone_and_anchored():
    assert score_from_pvalue(1.00) == 100.0
    assert score_from_pvalue(0.20) == 80.0
    assert score_from_pvalue(0.10) == 65.0
    assert score_from_pvalue(0.05) == 50.0
    assert score_from_pvalue(0.01) == 35.0
    assert score_from_pvalue(0.00) == 0.0
    xs = np.linspace(0, 1, 401)
    ys = [score_from_pvalue(x) for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:])), "must be monotone"
    assert not np.isfinite(score_from_pvalue(float("nan")))


def test_conformal_pvalue_definition():
    cal = np.arange(10.0)
    assert conformal_pvalue(cal, 100.0) == 1.0 / 11.0     # most extreme
    assert conformal_pvalue(cal, -1.0) == 11.0 / 11.0     # least extreme


# ------------------------------------------------------------ correctness

def test_refuses_singular_sample():
    """v1.0 accepted n=15 with p=22, produced NaN, and reported it as
    'E - Very Low Resilience / This species is not suitable for this site'."""
    sp, _ = make_data(n=15)
    tgt = np.median(sp, axis=0)
    for n in (15, 20, 22):
        try:
            ClimateResilienceAnalyzer(sp[:n], tgt, bio_cols=ALL_BIO_COLS,
                                      topo_cols=ALL_TOPO_COLS)
        except InsufficientDataError:
            continue
        raise AssertionError(f"n={n} with p=23 must be refused, not scored")


def test_nan_never_becomes_class_e():
    cls = ClimateResilienceAnalyzer._classify(float("nan"))
    assert "Indeterminate" in cls and not cls.startswith("E")
    assert ClimateResilienceAnalyzer._classify(None).startswith("—")


def test_aspect_is_circular_everywhere():
    """0 deg and 359 deg are the same slope; v1.0's Mahalanobis stage put them
    358 units apart because it used raw degrees."""
    sp, _ = make_data()
    med = np.median(sp, axis=0)
    out = {}
    for a in (0.0, 359.0, 180.0):
        t = med.copy(); t[21] = a
        r = analyse(sp, t)
        out[a] = (r["mahalanobis_stat"]["distance"],
                  r["composite"]["composite_score"])
    d0, s0 = out[0.0]; d359, s359 = out[359.0]; d180, s180 = out[180.0]
    assert abs(d0 - d359) / max(d0, 1e-9) < 0.15, \
        f"0 deg ({d0:.3f}) and 359 deg ({d359:.3f}) must be nearly identical"
    assert abs(s0 - s359) < 6
    assert d180 > 2 * d0, "a due-south target must still be distinguishable"


def test_calibration_is_valid():
    """The central claim: under H0 the score is calibrated by construction.

    v1.0 measured 50 % of a species' own records below class B and 10 %
    declared unsuitable.  Design targets here are A = 80 %, A+B = 90 %,
    E = 1 %.
    """
    sp, _ = make_data(n=200)
    r = analyse(sp, np.median(sp, axis=0))
    v = r["validation"]
    pc = v["self_class_percent"]
    assert v["ks_uniformity_pvalue"] >= 0.05, \
        f"out-of-fold p-values not uniform (KS p={v['ks_uniformity_pvalue']})"
    assert 70.0 <= pc["A"] <= 90.0, f"class A share {pc['A']}% (target 80%)"
    assert pc["A"] + pc["B"] >= 82.0, \
        f"A+B share {pc['A'] + pc['B']}% (target 90%)"
    assert pc["E"] <= 3.0, f"class E share {pc['E']}% (target 1%)"
    rej = v["empirical_rejection_rate"]
    assert rej["alpha_0.05"] <= 0.09, f"type-I error {rej['alpha_0.05']} at 5%"
    assert rej["alpha_0.01"] <= 0.04, f"type-I error {rej['alpha_0.01']} at 1%"


def test_unsuitable_target_is_still_rejected():
    """Calibration must not come at the cost of never rejecting anything."""
    sp, _ = make_data()
    med = np.median(sp, axis=0)
    off = np.array([100, 30, 10, 200, 60, 30, 50, 40, 30, 50, 30,
                    -300, -50, -6, 40, -150, -20, -30, -100])
    bad = np.concatenate([med[:19] + off, [2500.0, 55.0, 200.0]])
    r = analyse(sp, bad)
    assert r["composite"]["composite_score"] < 40, "clearly unsuitable site"
    assert r["composite"]["conformal_p"] < 0.05
    assert r["extrapolation"]["is_extrapolation"], "MESS must flag this"


def test_collinearity_is_detected():
    sp, _ = make_data()
    az = ClimateResilienceAnalyzer(sp, np.median(sp, axis=0),
                                   bio_cols=ALL_BIO_COLS,
                                   topo_cols=ALL_TOPO_COLS)
    c = az.collinearity
    assert c["effective_dimension"] < c["nominal_dimension"], \
        "bio1-19 are collinear; effective dimension must be lower"
    assert c["condition_number"] > 10


def test_no_double_counting_in_composite():
    """The composite must be one statistic, not four models plus a vote over
    the same four models (v1.0 gave Mahalanobis an effective weight of ~0.44
    while nominally weighting it 0.25)."""
    sp, _ = make_data()
    r = analyse(sp, np.median(sp, axis=0))
    assert r["threshold_zone"]["role"] == "diagnostic_readout"
    assert r["composite"]["component_weights"]["threshold_zone"] == "read-out"
    spread = r["weight_sensitivity"]["score_spread"]
    assert spread < 25, f"answer too sensitive to arbitrary weights ({spread})"


def test_diagnostic_layers_do_not_score():
    """v1.0's README listed PCA, Kernel PCA and K-Means as scoring layers
    although none of them ever entered the composite."""
    sp, _ = make_data()
    r = analyse(sp, np.median(sp, axis=0))
    for k in ("pca", "kpca", "kmeans"):
        assert r[k]["score"] is None
        assert r[k]["role"] == "diagnostic_only"


def test_target_area_is_evaluated_cell_wise():
    """Averaging a heterogeneous target area before scoring it is an
    ecological fallacy; the distribution must be reported."""
    sp, rng = make_data()
    cells = sp[rng.choice(len(sp), 30, replace=False)] + rng.normal(0, 4, (30, 22))
    r = analyse(sp, np.median(cells, axis=0), target_cells=cells)
    ta = r["target_area"]
    assert ta["available"] and ta["n_cells_evaluated"] == 30
    assert sum(ta["class_counts"].values()) == 30
    assert ta["cell_score_min"] <= ta["cell_score_median"] <= ta["cell_score_max"]


def test_climate_change_mode():
    sp, _ = make_data()
    med = np.median(sp, axis=0)
    fut = med.copy()
    fut[0] += 35; fut[4] += 45; fut[9] += 40; fut[11] -= 130
    r = analyse(sp, med, target_future=fut)
    cc = r["climate_change"]
    assert cc["available"]
    assert cc["future"]["score"] <= cc["current"]["score"]
    assert cc["delta_score"] <= 0
    assert r["engine"]["mode"] == "climate_change_response"
    # Without a future target the mode and wording must change.
    r2 = analyse(sp, med)
    assert r2["engine"]["mode"] == "current_climate_site_matching"
    assert not r2["climate_change"]["available"]
    assert "site matching" in r2["composite"]["recommendation"]


def test_spatial_block_folds():
    sp, rng = make_data()
    coords = np.column_stack([rng.normal(30, 2, len(sp)),
                              rng.normal(38, 2, len(sp))])
    r = analyse(sp, np.median(sp, axis=0), coords=coords)
    assert r["engine"]["fold_type"] == "spatial_block"


def test_uncertainty_is_reported():
    sp, _ = make_data()
    r = analyse(sp, np.median(sp, axis=0))
    ci = r["composite"]["score_ci95"]
    assert ci is not None and len(ci) == 2 and ci[0] <= ci[1]


def test_mechanism_text_is_sourced_and_disclaimed():
    sp, _ = make_data()
    r = analyse(sp, np.median(sp, axis=0))
    txt = r["percentile"]["per_bio"]["bio1"]["risk_explanation"]
    assert "[" in txt and "]" in txt, "mechanism must carry a citation"
    assert "NOT an inference about the analysed species" in txt
    assert "rank p" in txt          # replaced the n=1 Mann-Whitney misuse
    assert "mw_pvalue" not in r["percentile"]["per_bio"]["bio1"]


def test_results_contract_for_the_gui():
    sp, _ = make_data()
    r = analyse(sp, np.median(sp, axis=0))
    comp = r["composite"]
    for k in ("composite_score", "climate_score", "topo_score",
              "climate_weight", "topo_weight", "component_scores",
              "final_zone", "zone_label", "resilience_class",
              "recommendation", "top_risk_variables", "active_bio_cols"):
        assert k in comp, f"missing composite key: {k}"
    for k in ("threshold_zone", "gmm", "isolation_forest", "ocsvm",
              "mahalanobis", "topo"):
        assert k in comp["component_scores"]
    for k in ("gmm", "isolation_forest", "ocsvm", "mahalanobis_stat",
              "threshold_zone", "percentile", "topo", "variable_importance",
              "validation", "uncertainty", "extrapolation",
              "weight_sensitivity", "collinearity", "engine", "pca"):
        assert k in r, f"missing result block: {k}"


# ---------------------------------------------------------------- runner

def main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    passed, failed = 0, []
    print("=" * 70)
    print("  CRESTA v2.0 engine test suite  (%d tests)" % len(tests))
    print("=" * 70)
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print("  PASS  %s" % name)
        except Exception as e:
            failed.append((name, e))
            print("  FAIL  %s\n          %s: %s" % (name, type(e).__name__, e))
    print("-" * 70)
    print("  %d passed, %d failed" % (passed, len(failed)))
    print("=" * 70)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
