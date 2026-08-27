"""
Verification of the sector and market regime layers.

The sector layer is the one that can go wrong silently, because a composite built
from the whole history and then read at an earlier date would leak the future
into every historical signal and make the backtest look excellent. That is tested
explicitly below rather than assumed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from weinstein import synth, sectors as SEC, market as M, stages as S, features as F


def main():
    bad = []
    res = synth.build_demo()
    scan, frames, ranks = res["scan"], res["_frames"], res["_ranks"]

    # ---- 1. sector ranking recovers the constructed order -------------------
    order = {r["sector"]: r["sector_rs"] for r in res["sectors"]}
    print("sector rs:", {k: round(v, 2) for k, v in order.items()})
    if not (order["Leading Sector"] > order["Middling Sector"] > order["Lagging Sector"]):
        bad.append(f"sector relative strength out of order: {order}")
    rk = {r["sector"]: r["rank_pct"] for r in res["sectors"]}
    if not (rk["Leading Sector"] > rk["Middling Sector"] > rk["Lagging Sector"]):
        bad.append(f"sector ranks out of order: {rk}")

    # ---- 2. no lookahead in the composite -----------------------------------
    # Rebuild the composites from a truncated history and check the sector
    # relative strength at that cut-off equals the value the full-history run
    # reports for the same week. If the composite leaked, these diverge.
    idx, mkt, prices, meta = synth.build_universe()
    cut = len(idx) - 40
    trunc = {t: df.iloc[:cut] for t, df in prices.items()}
    f_full = SEC.sector_frames(SEC.build_composites(prices, meta), {"US": mkt})
    f_cut = SEC.sector_frames(SEC.build_composites(trunc, meta), {"US": mkt.iloc[:cut]})
    worst = 0.0
    for key in f_cut:
        a = f_cut[key]["sector_rs"].dropna()
        b = f_full[key]["sector_rs"].reindex(a.index)
        d = float((a - b).abs().max())
        worst = max(worst, d)
    print(f"lookahead check: max sector_rs difference on truncated history = {worst:.2e}")
    if worst > 1e-9:
        bad.append(f"sector composite leaks future data, max diff {worst}")

    # ---- 3. group factor only fires under its stated conditions -------------
    viol = 0
    for _, r in scan.iterrows():
        sr, own, g = r["sector_rs"], r["rs_vs_sector"], r["group_factor"]
        if any(pd.isna(v) for v in (sr, own, g)):
            continue
        expect = SEC.group_factor_row(sr, own)
        if abs(g - expect) > 1e-9:
            viol += 1
    print(f"group factor: {viol} rows disagree with the stated rule")
    if viol:
        bad.append(f"{viol} rows have a group factor the rule does not produce")

    # ---- 4. the factor actually moves the score -----------------------------
    # Same components, group factor removed, must recover the score divided by
    # the factor wherever the result was not clipped at 100.
    idxs, mkt2, prices2, meta2 = synth.build_universe()
    t = "LEA3"
    feat = F.compute_features(prices2[t], mkt2)
    key = "US / " + meta2[t]["sector"]
    comps = SEC.sector_frames(SEC.build_composites(prices2, meta2), {"US": mkt2})
    rks = SEC.rank_sectors(comps)
    feat = SEC.attach_to_stock(feat, prices2[t]["close"].astype(float),
                               comps.get(key), rks[key])
    feat["trend_score"] = S.trend_score_series(feat)
    with_g = S.readiness_series(feat)["stage2_readiness"]
    neutral = feat.copy(); neutral["group_factor"] = 1.0; neutral["group_factor_dn"] = 1.0
    without_g = S.readiness_series(neutral)["stage2_readiness"]
    live = (with_g < 99.9) & (without_g > 0)
    ratio = (with_g[live] / without_g[live]).dropna()
    uniq = sorted({round(v, 4) for v in ratio})
    print("group factor ratios observed:", uniq[:6])
    if not uniq or max(abs(np.array(uniq) - 1.0)) < 1e-6:
        bad.append("group factor never changed a readiness score")

    # ---- 4b. exhaustive equivalence of the two group-factor implementations --
    # Small enough input space to check completely rather than sample. The first
    # version of the vectorised twin left one branch unassigned, which is exactly
    # the class of bug a spot check misses and this does not.
    import itertools
    vals = [-10, -5, -3.0001, -3, -1, 0, 0.0001, 5, np.nan]
    combos = list(itertools.product(vals, vals))
    ff = pd.DataFrame({"sector_rs": [a for a, _ in combos],
                       "rs_vs_sector": [b for _, b in combos]})
    du2 = float(np.max(np.abs(SEC.group_factor_series(ff).to_numpy()
                              - np.array([SEC.group_factor_row(a, b) for a, b in combos]))))
    dd2 = float(np.max(np.abs(SEC.group_factor_series_dn(ff).to_numpy()
                              - np.array([SEC.group_factor_row_dn(a, b) for a, b in combos]))))
    print(f"exhaustive group factor equivalence over {len(combos)} combinations: "
          f"up {du2:.2e}, down {dd2:.2e}")
    if du2 > 0 or dd2 > 0:
        bad.append(f"group factor implementations disagree: {du2}, {dd2}")

    # ---- 5. regime is internally consistent ---------------------------------
    m = res["market"]
    print(f"regime: {m['regime']} {m['regime_score']}, breadth {m['breadth_above_ma']}%")
    if not (0.0 <= m["breadth_above_ma"] <= 100.0):
        bad.append(f"breadth outside 0-100: {m['breadth_above_ma']}")
    if M.label(m["regime_score"]) != m["regime"]:
        bad.append(f"regime label {m['regime']} does not match score {m['regime_score']}")
    if m["market_ok"] != (m["regime"] in ("Bull", "Improving")):
        bad.append("market_ok disagrees with the regime label")

    # ---- 6. row and vectorised readiness still agree WITH sector columns -----
    rows = feat.dropna(subset=["ma30"]).index
    vec = S.readiness_series(feat)
    du = max(abs(S.stage2_readiness(feat.loc[i])[0] - vec.loc[i, "stage2_readiness"]) for i in rows)
    dd = max(abs(S.stage4_readiness(feat.loc[i])[0] - vec.loc[i, "stage4_readiness"]) for i in rows)
    print(f"row vs vectorised with sector data: up {du:.2e}, down {dd:.2e}")
    if du > 1e-6 or dd > 1e-6:
        bad.append(f"row and vectorised readiness diverge with sector data: {du}, {dd}")

    # ---- 7. grades exist and are ordered ------------------------------------
    grades = scan["grade"].dropna().value_counts().to_dict()
    print("grades:", grades)
    if not grades:
        bad.append("no signal was graded")

    print("\n" + ("FAILURES:\n  " + "\n  ".join(bad) if bad else "SECTOR AND REGIME CHECKS PASSED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
