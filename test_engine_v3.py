"""test_engine_v3.py — QGIS bağımsız tam test"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analysis_engine import ClimateResilienceAnalyzer

np.random.seed(42)
N = 80

bio_means = [180,95,55,600,340,30,310,220,140,250,55,650,90,8,73,248,30,50,200]
bio_sds   = [22, 10,  5, 75, 18,14, 23, 19, 18, 20,18, 95,18,4,14, 48,12,18, 48]
bio_data  = np.column_stack([
    np.array([m + s*np.random.randn() for _ in range(N)])
    for m,s in zip(bio_means, bio_sds)
])
elev = (750 + 200*np.random.randn(N)).reshape(-1,1)
slop = np.abs(12 + 6*np.random.randn(N)).reshape(-1,1)
aspc = ((30 + 45*np.random.randn(N)) % 360).reshape(-1,1)
species_data = np.hstack([bio_data, elev, slop, aspc])

bio_off_b = np.array([25,5,0,80,20,8,15,8,10,18,8,-120,-15,-2,15,-60,-8,-10,-50])
bio_off_e = np.array([100,30,10,200,60,30,50,40,30,50,30,-300,-50,-6,40,-150,-20,-30,-100])

scenarios = {
    "A — Optimal":            np.concatenate([np.median(bio_data,0),        [750,12,30]]),
    "B — Sınır iklim":        np.concatenate([np.median(bio_data,0)+bio_off_b,[750,12,30]]),
    "C — Uyumsuz topo":       np.concatenate([np.median(bio_data,0),        [2200,50,180]]),
    "D — Tam uyumsuz":        np.concatenate([np.median(bio_data,0)+bio_off_e,[2500,55,200]]),
}

print("="*72)
print("  CLIMResAnalyzer v3.0 — ML + Çok Eşikli Skor TESTİ")
print("="*72)

for name, target in scenarios.items():
    print(f"\n{'─'*72}")
    print(f"  📍 {name}")
    az = ClimateResilienceAnalyzer(species_data, target, has_topo=True)
    r  = az.run_all()
    c  = r["composite"]
    tz = r["threshold_zone"]
    g  = r["gmm"]
    iso= r["isolation_forest"]
    svm= r["ocsvm"]
    m  = r["mahalanobis_stat"]
    km = r["kmeans"]
    vi = r["variable_importance"]

    print(f"\n  KATMAN B — Sınıflandırma Modelleri:")
    print(f"    GMM  log-prob={g['log_probability']:>7.3f}  zone={g['zone']:<10} skor={g['score']:.1f}")
    print(f"    IsoF anom    ={iso['anomaly_score']:>7.3f}  zone={iso['zone']:<10} skor={iso['score']:.1f}")
    print(f"    OCSVM dec   ={svm['decision_value']:>7.3f}  zone={svm['zone']:<10} skor={svm['score']:.1f}")
    print(f"    Mahal D     ={m['distance']:>7.3f}  zone={m['zone']:<10} skor={m['score']:.1f}")
    print(f"    K-Means cluster={km['target_cluster']}  k={km['best_k']}  skor={km['score']:.1f}")

    print(f"\n  KATMAN C — Eşik Bölgesi Oylama:")
    print(f"    Oylar: {tz['votes']}")
    print(f"    → {tz['zone_label']}  (ort.oy={tz['mean_vote']:.2f}, skor={tz['score']:.1f})")

    print(f"\n  KATMAN D — Top 5 Risk Değişkeni:")
    for v in vi["top5_variables"]:
        rf_v = vi["rf_importance"].get(v, 0)
        ks_v = vi["ks_deviation"].get(v, 0)
        print(f"    {v:<14} RF önem={rf_v:.4f}  KS sap={ks_v:.4f}")

    print(f"\n  KATMAN E — Bileşik Skor:")
    print(f"    İklim skoru  : {c['climate_score']:.2f}  (ağırlık %{int(c['climate_weight']*100)})")
    print(f"    Topo  skoru  : {c['topo_score']:.2f}  (ağırlık %{int(c['topo_weight']*100)})")
    print(f"    ┌─────────────────────────────────────────┐")
    print(f"    │  BİLEŞİK DİRENÇ SKORU : {c['composite_score']:>5.2f} / 100   │")
    print(f"    │  Sınıf  : {c['resilience_class']:<32}│")
    print(f"    │  Bölge  : {c['final_zone']:<32}│")
    print(f"    └─────────────────────────────────────────┘")

print("\n" + "="*72)
print("  ✅ v3.0 testleri tamamlandı.")
print("="*72)
