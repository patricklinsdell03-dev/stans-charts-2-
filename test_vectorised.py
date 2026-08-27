"""
Consistency check between the readable row-wise model and the fast vectorised
one. Any divergence here means the backtest is calibrating a different model to
the one the weekly scan reports, which is the kind of bug that stays invisible
until the numbers have already been trusted.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from weinstein import features as F, stages as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_synthetic import path_stage1_to_2, path_stage3_to_4, path_chop, flat_benchmark

def main():
    bench = flat_benchmark()
    bad = []
    for label, df in [("s1to2", path_stage1_to_2()), ("s3to4", path_stage3_to_4()),
                      ("chop", path_chop())]:
        feat = F.compute_features(df, bench)
        feat = feat.assign(trend_score=S.trend_score_series(feat))

        ts_row = feat.apply(S.trend_score, axis=1)
        dmax = float((ts_row - feat["trend_score"]).abs().max())
        if dmax > 1e-6:
            bad.append(f"{label}: trend_score max diff {dmax}")

        vec = S.readiness_series(feat)
        rows = feat.dropna(subset=["ma30"]).index
        r_up, r_dn = [], []
        for i in rows:
            u, _ = S.stage2_readiness(feat.loc[i]); r_up.append(u)
            d, _ = S.stage4_readiness(feat.loc[i]); r_dn.append(d)
        du = float((pd.Series(r_up, index=rows) - vec.loc[rows, "stage2_readiness"]).abs().max())
        dd = float((pd.Series(r_dn, index=rows) - vec.loc[rows, "stage4_readiness"]).abs().max())
        if du > 1e-6 or dd > 1e-6:
            bad.append(f"{label}: readiness max diff up={du} dn={dd}")

        br = S.breaks_series(feat)
        mism_up = mism_dn = 0
        for i in rows:
            sig = S.detect_break(feat.loc[i])
            up = bool(sig and sig.kind == "stage2_breakout")
            dn = bool(sig and sig.kind == "stage4_breakdown")
            if up != bool(br.loc[i, "flagged_up"]):
                mism_up += 1
            # the row version returns only one signal per bar, so a bar flagged
            # both ways can legitimately differ on the second one
            if dn != bool(br.loc[i, "flagged_dn"]) and not bool(br.loc[i, "flagged_up"]):
                mism_dn += 1
        print(f"{label}: trend diff {dmax:.2e}  readiness diff {max(du,dd):.2e}  "
              f"break mismatches up={mism_up} dn={mism_dn}")
        if mism_up or mism_dn:
            bad.append(f"{label}: break flag mismatches up={mism_up} dn={mism_dn}")

    print("\n" + ("FAILURES:\n  " + "\n  ".join(bad) if bad else "VECTORISED MODEL MATCHES ROW MODEL"))
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
