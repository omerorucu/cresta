"""
test_engine.py  —  QGIS olmadan bağımsız test
================================================
Kullanım:  python test_engine.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analysis_engine import ClimateResilienceAnalyzer

np.random.seed(42)

# ── Test verisi üret ──────────────────────────────────────────────────────────
N = 80

# Bio1-19: Akdeniz iklimi türü
bio_means = [180,95,55,600,340,30,310,220,140,250,55,650,90,8,73,248,30,50,200]
bio_sds   = [22, 10, 5, 75, 18, 14, 23, 19, 18, 20, 18, 95, 18, 4, 14, 48, 12, 18, 48]

# Topografik değerler: tür tipik olarak 400-1200 m, %5-25 eğim, K-KD bakı
elev_mean, elev_sd = 750,  200    # m
slop_mean, slop_sd = 12,   6      # %
aspc_mean, aspc_sd = 30,   45     # derece (kuzey-kuzeydoğu ağırlıklı)

def rnorm_col(mu, sd, n):
    return np.array([mu + sd*np.random.randn() for _ in range(n)])

bio_data  = np.column_stack([rnorm_col(m,s,N) for m,s in zip(bio_means, bio_sds)])
elev_data = rnorm_col(elev_mean, elev_sd, N).reshape(-1,1)
slop_data = np.abs(rnorm_col(slop_mean, slop_sd, N)).reshape(-1,1)  # negatif olmasın
aspc_data = (rnorm_col(aspc_mean, aspc_sd, N) % 360).reshape(-1,1)

species_data = np.hstack([bio_data, elev_data, slop_data, aspc_data])  # (80, 22)

print(f"Tür noktaları: {species_data.shape}  (80 nokta × 22 değişken)")
print(f"  Bio:         sütun  0-18")
print(f"  elevation_m: sütun 19")
print(f"  slope_pct:   sütun 20")
print(f"  aspect_deg:  sütun 21")
print()

# ── Senaryo tanımları ─────────────────────────────────────────────────────────
bio_offsets_b = [25,5,0,80,20,8,15,8,10,18,8,-120,-15,-2,15,-60,-8,-10,-50]
bio_offsets_e = [100,30,10,200,60,30,50,40,30,50,30,-300,-50,-6,40,-150,-20,-30,-100]

scenarios = {
    "A — Optimal (medyan + ideal topo)": np.concatenate([
        np.median(bio_data, axis=0),
        [elev_mean, slop_mean, aspc_mean]
    ]),
    "B — İyi iklim + uygun yükseklik, yüksek eğim": np.concatenate([
        np.median(bio_data, axis=0),
        [900, 35, 45]   # eğim yüksek ama iklim iyi
    ]),
    "C — Sınır iklim + uygun topo": np.concatenate([
        np.median(bio_data, axis=0) + bio_offsets_b,
        [elev_mean, slop_mean, aspc_mean]
    ]),
    "D — Sınır iklim + uyumsuz topo (güney bakı, çok yüksek)": np.concatenate([
        np.median(bio_data, axis=0) + bio_offsets_b,
        [1800, 40, 180]  # güney bakı, yüksek rakım, dik eğim
    ]),
    "E — Uyumsuz iklim + uyumsuz topo": np.concatenate([
        np.median(bio_data, axis=0) + bio_offsets_e,
        [2500, 55, 200]
    ]),
}

print("=" * 70)
print("  SENARYO BAZLI TEST SONUÇLARI")
print("=" * 70)

for name, target in scenarios.items():
    az = ClimateResilienceAnalyzer(species_data, target, has_topo=True)
    r  = az.run_all()
    c  = r["composite"]
    m  = r["mahalanobis"]
    t  = r["topo"]
    n  = r["niche_overlap"]
    p  = r["percentile"]

    print(f"\n🔹 {name}")
    print(f"   Topografi girdi : elevation={target[19]:.0f}m  slope={target[20]:.1f}%  aspect={target[21]:.0f}°")
    print(f"   ─────────────────────────────────────────────────────────")
    print(f"   Mahalanobis D    : {m['distance']:.4f}   skor: {m['score']:.1f}")
    print(f"   Niş örtüşmesi D  : {n['mean_D']:.4f}   skor: {n['score']:.1f}")
    print(f"   Yüzdelik dilim   : {p['score']:.1f}   Bio aralık içi: {p['bios_in_tolerance_range']}/19")
    print(f"   PCA konumu       : {r['pca']['score']:.1f}")
    print(f"   Topografi skoru  : {t['score']:.1f}   "
          f"(elev={t['elevation']['score']:.1f}  "
          f"slope={t['slope']['score']:.1f}  "
          f"aspect={t['aspect']['score']:.1f})")
    print(f"   ─────────────────────────────────────────────────────────")
    print(f"   BİLEŞİK SKOR     : {c['composite_score']:.2f} / 100   →  {c['resilience_class']}")
    print(f"   İklim skoru      : {c['climate_score']:.2f}  (ağırlık %80)")
    print(f"   Topo skoru       : {c['topo_score']:.2f}  (ağırlık %20)")
    print(f"   Bakı yorum       : {t['aspect']['target_exposure']}  →  kosinüs={t['aspect']['cos_similarity']:.3f}")

# ── Topografisiz test ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  TOPOGRAFISIZ TEST (has_topo=False — sadece Bio1-19)")
print("=" * 70)

target_no_topo = np.median(bio_data, axis=0)
az2 = ClimateResilienceAnalyzer(bio_data, target_no_topo, has_topo=False)
r2  = az2.run_all()
c2  = r2["composite"]
print(f"  Bileşik Skor: {c2['composite_score']:.2f}  →  {c2['resilience_class']}")
print(f"  Topo ağırlık: {c2['topo_weight']} (veri yok → %100 iklim)")

print("\n" + "=" * 70)
print("  ✅ Tüm testler başarıyla tamamlandı.")
print("=" * 70)
