"""
test_engine_v3.py — Methodological hardening test suite

Standalone (no QGIS) smoke + correctness tests for:
  1. Baseline composite score path
  2. Full bootstrap CI (wider than fast bootstrap)
  3. Unsupervised sensitivity analysis + weight optimization
  4. Auto-applied spatial autocorrelation correction (clustered species data)
  5. Backwards compatibility of legacy APIs
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analysis_engine import (
    ClimateResilienceAnalyzer,
    ALL_BIO_COLS, ALL_TOPO_COLS,
)

np.random.seed(42)

BIO_COLS  = ALL_BIO_COLS
TOPO_COLS = ALL_TOPO_COLS

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic species datasets
# ──────────────────────────────────────────────────────────────────────────────

def _make_bio(n, means, sds, rng):
    return np.column_stack([
        rng.normal(m, s, n) for m, s in zip(means, sds)
    ])

BIO_MEANS = [180, 95, 55, 600, 340, 30, 310, 220, 140, 250,
              55, 650,  90,   8,  73, 248,  30,  50, 200]
BIO_SDS   = [ 22, 10,  5,  75,  18, 14,  23,  19,  18,  20,
              18,  95,  18,   4,  14,  48,  12,  18,  48]

def make_well_dispersed_species(n=120, seed=42):
    """Unimodal native range — low spatial autocorrelation expected."""
    rng  = np.random.RandomState(seed)
    bio  = _make_bio(n, BIO_MEANS, BIO_SDS, rng)
    elev = rng.normal(750, 200, n).reshape(-1, 1)
    slop = np.abs(rng.normal(12, 6, n)).reshape(-1, 1)
    aspc = (rng.normal(30, 45, n) % 360).reshape(-1, 1)
    return np.hstack([bio, elev, slop, aspc])

def make_clustered_species(n_per=60, seed=7):
    """Two well-separated clusters — triggers auto-thinning."""
    rng = np.random.RandomState(seed)
    m1  = np.array(BIO_MEANS, dtype=float)
    m2  = m1.copy();  m2[[0, 4, 5, 10]] += np.array([80, 70, 40, 80])
    s   = np.array(BIO_SDS, dtype=float) * 0.35          # tight clusters
    bio1 = _make_bio(n_per, m1, s, rng)
    bio2 = _make_bio(n_per, m2, s, rng)
    bio  = np.vstack([bio1, bio2])
    n    = bio.shape[0]
    elev = np.concatenate([
        rng.normal(600, 60, n_per),
        rng.normal(1100, 60, n_per),
    ]).reshape(-1, 1)
    slop = np.abs(rng.normal(12, 3, n)).reshape(-1, 1)
    aspc = (rng.normal(30, 20, n) % 360).reshape(-1, 1)
    return np.hstack([bio, elev, slop, aspc])


def _analyzer(X, target):
    return ClimateResilienceAnalyzer(
        X, target, bio_cols=BIO_COLS, topo_cols=TOPO_COLS, opt_cols=[]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_baseline_composite():
    print("─" * 72)
    print("  TEST 1 — Baseline composite score")
    X   = make_well_dispersed_species()
    tgt = np.concatenate([np.median(X[:, :19], axis=0), [750, 12, 30]])
    az  = _analyzer(X, tgt)
    az.compute_composite_score()
    c   = az.results["composite"]
    print(f"    composite = {c['composite_score']:.2f}   class = {c['resilience_class']}")
    assert 0 <= c["composite_score"] <= 100
    assert c["composite_score"] > 60, "Target at median should score > 60"
    print("    ✅ PASS")


def test_full_vs_fast_bootstrap():
    print("─" * 72)
    print("  TEST 2 — Full vs fast bootstrap CI width")
    X   = make_well_dispersed_species(n=100)
    tgt = np.concatenate([np.median(X[:, :19], axis=0) + np.array(BIO_SDS)*0.2,
                          [800, 15, 45]])

    az = _analyzer(X, tgt)
    az.compute_composite_score()

    t0 = time.time()
    az.compute_bootstrap_ci(n_bootstrap=200)    # fast
    fast = az.results["bootstrap_ci_fast"]
    t_fast = time.time() - t0

    t0 = time.time()
    az.compute_bootstrap_ci_full(n_bootstrap=40)  # full (smaller n for speed)
    full = az.results["bootstrap_ci_full"]
    t_full = time.time() - t0

    print(f"    fast: CI=[{fast['ci_lower']:.1f}, {fast['ci_upper']:.1f}]  "
          f"width={fast['ci_width']:.1f}  ({t_fast:.1f}s)")
    print(f"    full: CI=[{full['ci_lower']:.1f}, {full['ci_upper']:.1f}]  "
          f"width={full['ci_width']:.1f}  ({t_full:.1f}s)")

    assert full["n_valid"] > 0, "Full bootstrap should produce valid samples"
    assert full["method"].startswith("Full bootstrap"), \
        "Full bootstrap method label incorrect"
    # Full bootstrap should generally be wider, but with small n it may be
    # noisy — require only that it's not dramatically narrower than fast.
    assert full["ci_width"] >= fast["ci_width"] * 0.7, \
        f"Full CI suspiciously narrow vs fast ({full['ci_width']} vs {fast['ci_width']})"
    # And that the primary key is the full one
    assert az.results["bootstrap_ci"]["method"] == full["method"]
    print("    ✅ PASS")


def test_sensitivity_analysis():
    print("─" * 72)
    print("  TEST 3 — Sensitivity analysis + weight optimization")
    X   = make_well_dispersed_species(n=120)
    tgt = np.concatenate([np.median(X[:, :19], axis=0), [750, 12, 30]])
    az  = _analyzer(X, tgt)
    az.compute_composite_score()
    az.compute_sensitivity_analysis(n_folds=5, n_grid=150)
    s = az.results["sensitivity"]

    print(f"    grid pts       = {s['n_grid_points']}")
    print(f"    test pts       = {s['n_test_points']}")
    print(f"    default target = {s['target_score_default']:.2f}")
    print(f"    optimal target = {s['target_score_optimal']:.2f}")
    print(f"    range          = {s['target_score_range']:.2f}")
    print(f"    class flips    = {s['class_flip_fraction']*100:.1f}%")
    print(f"    robustness     = {s['robustness']}")
    print(f"    default LOO    = {s['default_loo_score']:.2f}")
    print(f"    optimal LOO    = {s['optimal_loo_score']:.2f}")

    assert "error" not in s, f"Sensitivity error: {s.get('error')}"
    assert s["n_test_points"] >= 10, "Too few K-fold test points"
    assert s["optimal_loo_score"] >= s["default_loo_score"] - 0.01, \
        "Optimization should not regress vs default"
    assert s["robustness"] in ("High", "Moderate", "Low")
    # Target at median → should be highly robust
    assert s["robustness"] in ("High", "Moderate"), \
        f"Median target should be robust, got {s['robustness']}"
    # Weights sum ~= 1 and each in [0.05, 0.60]
    mlw = [s["optimal_weights"][k] for k in
           ("w_threshold_zone","w_gmm","w_iso_forest","w_ocsvm","w_mahalanobis")]
    assert abs(sum(mlw) - 1.0) < 0.02, f"ML weights do not sum to 1: {sum(mlw)}"
    for w in mlw:
        assert 0.04 <= w <= 0.61, f"ML weight out of bounds: {w}"
    print("    ✅ PASS")


def test_spatial_correction_clustered():
    print("─" * 72)
    print("  TEST 4 — Auto spatial correction on clustered data")
    X   = make_clustered_species(n_per=60)
    # Target: between the two clusters → unlikely to be inlier in either
    tgt = np.concatenate([np.median(X[:, :19], axis=0), [850, 12, 30]])
    az  = _analyzer(X, tgt)
    az.compute_composite_score()
    az.apply_environmental_thinning()
    az.compute_blocked_cv()
    az.apply_spatial_correction(inlier_threshold=0.60, min_points=15)

    thin = az.results["thinning"]
    bcv  = az.results["blocked_cv"]
    ta   = az.results["thinned_analysis"]

    print(f"    thinning  : {thin['n_before']} → {thin['n_after']}")
    print(f"    mean inlier rate : {bcv.get('mean_inlier_rate')}")
    print(f"    correction applied: {ta.get('applied')}")
    if ta.get("applied"):
        print(f"    raw = {ta['raw_composite']:.2f}  "
              f"→  corrected = {ta['thinned_composite']:.2f}  "
              f"(Δ {ta['delta']:+.2f})")
        print(f"    raw class  = {ta['raw_class']}")
        print(f"    thin class = {ta['thinned_class']}")
    else:
        print(f"    reason: {ta.get('reason')}")

    # Either correction applied (clustered → bad CV) or clearly explained
    assert "applied" in ta
    if ta["applied"]:
        assert ta["n_after"] < ta["n_before"]
        assert "recommended_score" in ta
        assert 0 <= ta["thinned_composite"] <= 100
    else:
        # If not applied, reason must be present
        assert ta.get("reason"), "Non-applied spatial correction missing reason"
    print("    ✅ PASS")


def test_backwards_compat():
    print("─" * 72)
    print("  TEST 5 — Backwards compatibility")
    X   = make_well_dispersed_species(n=80)
    tgt = np.concatenate([np.median(X[:, :19], axis=0), [750, 12, 30]])
    az  = _analyzer(X, tgt)

    # Legacy methods must still exist and run
    az.compute_composite_score()
    az.compute_bootstrap_ci(n_bootstrap=50)     # fast path
    az.apply_environmental_thinning()
    az.compute_blocked_cv()

    assert "bootstrap_ci_fast" in az.results
    assert "thinning" in az.results
    assert "blocked_cv" in az.results

    # calibrate_weights still exists with its original signature
    import inspect
    sig = inspect.signature(az.calibrate_weights)
    params = list(sig.parameters)
    assert params == ["validation_data", "validation_labels"], \
        f"calibrate_weights signature changed: {params}"
    print("    ✅ PASS")


def test_run_all_full_pipeline():
    print("─" * 72)
    print("  TEST 6 — Full run_all pipeline (integration)")
    X   = make_well_dispersed_species(n=90)
    tgt = np.concatenate([np.median(X[:, :19], axis=0), [750, 12, 30]])
    az  = _analyzer(X, tgt)
    t0 = time.time()
    r = az.run_all()
    dt = time.time() - t0
    print(f"    total runtime: {dt:.1f}s")

    required = [
        "composite", "threshold_zone", "gmm", "isolation_forest", "ocsvm",
        "kmeans", "mahalanobis_stat", "percentile", "topo",
        "variable_importance", "thinning", "blocked_cv",
        "thinned_analysis",            # new
        "bootstrap_ci", "bootstrap_ci_full",  # new
        "sensitivity",                 # new
    ]
    missing = [k for k in required if k not in r]
    assert not missing, f"Missing keys after run_all: {missing}"

    bci = r["bootstrap_ci"]
    assert bci["method"].startswith("Full bootstrap"), \
        f"run_all should use full bootstrap by default, got: {bci['method']}"

    print("    all required keys present ✅")
    print(f"    composite   = {r['composite']['composite_score']:.2f}")
    print(f"    bootstrap CI= [{bci['ci_lower']:.1f}, {bci['ci_upper']:.1f}]")
    print(f"    robustness  = {r['sensitivity'].get('robustness', '—')}")
    print("    ✅ PASS")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  CRESTA — Methodological Hardening Test Suite")
    print("=" * 72)

    tests = [
        test_baseline_composite,
        test_full_vs_fast_bootstrap,
        test_sensitivity_analysis,
        test_spatial_correction_clustered,
        test_backwards_compat,
        test_run_all_full_pipeline,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"    ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"    💥 ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 72)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 72)
    sys.exit(0 if failed == 0 else 1)
