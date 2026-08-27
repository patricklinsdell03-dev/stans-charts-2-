"""
Verification against synthetic price paths whose correct stage label is known by
construction. If the engine cannot label a textbook Weinstein sequence it will
certainly not label a real one.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from weinstein import features as F
from weinstein import stages as S

rng = np.random.default_rng(11)
N = 220
IDX = pd.date_range("2022-01-07", periods=N, freq="W-FRI")


def make_bars(closes, vols=None, noise=0.004):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if vols is None:
        vols = np.full(n, 1_000_000.0)
    hi = closes * (1 + np.abs(rng.normal(0, noise, n)) + 0.006)
    lo = closes * (1 - np.abs(rng.normal(0, noise, n)) - 0.006)
    op = closes * (1 + rng.normal(0, noise, n))
    return pd.DataFrame({"open": op, "high": hi, "low": lo,
                         "close": closes, "volume": vols}, index=IDX[:n])


def flat_benchmark(n=N, drift=0.0008):
    return pd.Series(100 * np.cumprod(1 + drift + rng.normal(0, 0.008, n)), index=IDX[:n])


def path_stage1_to_2():
    """80 weeks of decline, 100 weeks of flat base, 40 weeks of advance."""
    px, v = [], []
    p = 100.0
    for _ in range(80):                       # stage 4 decline
        p *= (1 - 0.010 + rng.normal(0, 0.012))
        px.append(p); v.append(1.2e6)
    base = p
    for i in range(100):                      # stage 1 base, range contracts
        amp = 0.055 * (1 - i / 130)
        p = base * (1 + amp * np.sin(i / 4.5) + rng.normal(0, 0.006))
        px.append(p); v.append(0.55e6)        # volume dries up
    for i in range(40):                       # stage 2 advance
        p = px[-1] * (1 + 0.016 + rng.normal(0, 0.010))
        px.append(p)
        v.append(3.0e6 if i == 0 else 1.6e6)  # breakout week on 3x volume
    return make_bars(px, v)


def path_stage3_to_4():
    """80 weeks of advance, 80 weeks of topping, 60 weeks of decline."""
    px, v = [], []
    p = 100.0
    for _ in range(80):
        p *= (1 + 0.012 + rng.normal(0, 0.012))
        px.append(p); v.append(1.2e6)
    top = p
    for i in range(80):
        p = top * (1 + 0.05 * np.sin(i / 5.0) + rng.normal(0, 0.010))
        px.append(p)
        v.append(1.7e6 if np.sin(i / 5.0) < 0 else 0.9e6)   # distribution
    for i in range(60):
        p = px[-1] * (1 - 0.014 + rng.normal(0, 0.012))
        px.append(p); v.append(1.5e6)
    return make_bars(px, v)


def path_chop():
    p = 100 * np.cumprod(1 + rng.normal(0, 0.018, N))
    return make_bars(p)


def summarize(df, bench, label):
    feat = F.compute_features(df, bench)
    rec = S.evaluate(feat, label)
    return feat, rec


def main():
    bench = flat_benchmark()
    failures = []

    # ---- Case 1: the base and the breakout --------------------------------
    df = path_stage1_to_2()
    feat = F.compute_features(df, bench)

    # (a) at the end of the base, before the advance, expect stage 1 family and
    #     a high stage 2 readiness score.
    at_base = feat.iloc[:180]
    rec_base = S.evaluate(at_base, "BASE")
    print(f"[base]  stage={rec_base['stage']:<14} trend={rec_base['trend_score']:>6} "
          f"ready2={rec_base['stage2_readiness']:>5} ready4={rec_base['stage4_readiness']:>5} "
          f"dist_res={rec_base['dist_to_resistance_pct']}")
    if "Stage 1" not in rec_base["stage"] and rec_base["stage"] != "Stage 4 to 1":
        failures.append(f"base misclassified as {rec_base['stage']}")
    if rec_base["stage2_readiness"] < 40:
        failures.append(f"base readiness too low: {rec_base['stage2_readiness']}")

    # (b) the breakout week itself must fire a stage2_breakout signal.
    fired = None
    for k in range(178, 190):
        r = feat.iloc[k].copy()
        r["trend_score"] = S.trend_score(r)
        sig = S.detect_break(r)
        if sig and sig.kind == "stage2_breakout":
            fired = (k, sig); break
    print(f"[break] fired at bar {fired[0] if fired else None} "
          f"confirmed={fired[1].confirmed if fired else None} "
          f"vol_ok={fired[1].volume_ok if fired else None} "
          f"rs_ok={fired[1].rs_ok if fired else None}" if fired else "[break] NONE")
    if fired is None:
        failures.append("no stage 2 breakout detected on the constructed breakout")

    # (c) well into the advance, expect Stage 2.
    rec_adv = S.evaluate(feat, "ADV")
    print(f"[adv]   stage={rec_adv['stage']:<14} trend={rec_adv['trend_score']:>6} "
          f"rs={rec_adv['mansfield_rs']} slope={rec_adv['ma_slope_pct']}")
    if rec_adv["stage"] != "Stage 2":
        failures.append(f"advance misclassified as {rec_adv['stage']}")

    # ---- Case 2: the top and the breakdown --------------------------------
    df2 = path_stage3_to_4()
    feat2 = F.compute_features(df2, bench)

    at_top = feat2.iloc[:158]
    rec_top = S.evaluate(at_top, "TOP")
    print(f"[top]   stage={rec_top['stage']:<14} trend={rec_top['trend_score']:>6} "
          f"ready4={rec_top['stage4_readiness']:>5} dist_sup={rec_top['dist_to_support_pct']} "
          f"dvol={rec_top['down_vol_share']}")
    if rec_top["stage"] not in ("Stage 3", "Stage 2 to 3", "Stage 3 to 4"):
        failures.append(f"top misclassified as {rec_top['stage']}")
    if rec_top["stage4_readiness"] < 40:
        failures.append(f"top stage4 readiness too low: {rec_top['stage4_readiness']}")

    fired2 = None
    for k in range(156, 172):
        r = feat2.iloc[k].copy()
        r["trend_score"] = S.trend_score(r)
        sig = S.detect_break(r)
        if sig and sig.kind == "stage4_breakdown":
            fired2 = (k, sig); break
    print(f"[bdown] fired at bar {fired2[0] if fired2 else None} "
          f"confirmed={fired2[1].confirmed if fired2 else None}" if fired2 else "[bdown] NONE")
    if fired2 is None:
        failures.append("no stage 4 breakdown detected on the constructed breakdown")

    rec_dec = S.evaluate(feat2, "DEC")
    print(f"[dec]   stage={rec_dec['stage']:<14} trend={rec_dec['trend_score']:>6}")
    if rec_dec["stage"] != "Stage 4":
        failures.append(f"decline misclassified as {rec_dec['stage']}")

    # ---- Case 3: noise should not score highly either way -----------------
    rec_chop = S.evaluate(F.compute_features(path_chop(), bench), "CHOP")
    print(f"[chop]  stage={rec_chop['stage']:<14} trend={rec_chop['trend_score']:>6} "
          f"ready2={rec_chop['stage2_readiness']} ready4={rec_chop['stage4_readiness']}")

    print("\n" + ("FAILURES:\n  " + "\n  ".join(failures) if failures else "ALL CHECKS PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
