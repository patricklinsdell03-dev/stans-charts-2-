"""
Mechanical invariants.

The other test files check that the model produces the right answer on cases
whose answer is known. This one checks properties that must hold for EVERY
input, which is a different and in some ways stronger guarantee: it catches the
class of defect that only appears on inputs nobody thought to construct.

Five properties are enforced.

Causality: no value attributed to week t may depend on data after week t. This
is checked by truncation, recomputing the whole feature engine on history cut at
week k and demanding bit-identical values at every week up to k, and by
perturbation, changing week k and demanding nothing before it moves.

Scale invariance: multiplying every price by a constant is a change of units. A
percentage, a ratio and a score must be unchanged by it; a level must scale by
exactly that constant. Anything that does neither is mixing units somewhere.

Determinism: the same input twice gives the same output.

Bounds: every score stays inside its stated range.

Missing data is never rewarded: for every scored component, supplying NaN must
never score higher than supplying the best real value. This is the property the
base-quality bug violated for twelve points out of a hundred.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from weinstein import features as F, stages as S, plan as PL, sectors as SEC

RNG = np.random.default_rng(20)

# Features whose value is a price level and must therefore scale with price.
LEVEL_COLS = {"close", "high", "low", "ma30", "atr", "resistance", "support",
              "recent_low_10w", "recent_high_10w", "stop_suggestion"}
# Features that are pure counts or already dimensionless.
UNITLESS_COLS = {"volume", "base_age_weeks", "base_age_recent", "tightness",
                 "vol_ratio", "vol_ratio_3w", "vol_dryup", "down_vol_share",
                 "weeks_since_rs_cross_up", "weeks_since_rs_cross_dn"}


def make_series(n=240, seed=1):
    r = np.random.default_rng(seed)
    close = 50 * np.cumprod(1 + r.normal(0.001, 0.025, n))
    idx = pd.date_range("2021-01-01", periods=n, freq="W-FRI")
    hi = close * (1 + np.abs(r.normal(0, .01, n)) + .004)
    lo = close * (1 - np.abs(r.normal(0, .01, n)) - .004)
    vol = np.abs(r.normal(1e6, 3e5, n)) + 1e5
    df = pd.DataFrame({"open": close, "high": hi, "low": lo, "close": close,
                       "volume": vol}, index=idx)
    bench = pd.Series(80 * np.cumprod(1 + r.normal(0.0012, 0.012, n)), index=idx)
    return df, bench


def scored(df, bench):
    f = F.compute_features(df, bench)
    f["trend_score"] = S.trend_score_series(f)
    return f.join(S.readiness_series(f))


def main():
    bad = []
    df, bench = make_series()

    # ---- 1. causality by truncation ----------------------------------------
    full = scored(df, bench)
    worst_col, worst_diff = None, 0.0
    for k in (120, 170, 200, 230):
        cut = scored(df.iloc[:k], bench.iloc[:k])
        common = cut.index
        for col in cut.columns:
            a = pd.to_numeric(full.loc[common, col], errors="coerce")
            b = pd.to_numeric(cut[col], errors="coerce")
            both = a.notna() & b.notna()
            if not both.any():
                if a.notna().sum() != b.notna().sum():
                    bad.append(f"causality: {col} NaN pattern differs at cut {k}")
                continue
            d = float((a[both] - b[both]).abs().max())
            if d > worst_diff:
                worst_diff, worst_col = d, f"{col}@{k}"
            if d > 1e-9:
                bad.append(f"LOOKAHEAD: {col} differs by {d:.3e} at cut {k}")
    print(f"causality by truncation: worst difference {worst_diff:.3e} ({worst_col})")

    # ---- 2. causality by perturbation --------------------------------------
    for k in (150, 200):
        pert = df.copy()
        pert.iloc[k, pert.columns.get_loc("close")] *= 1.4
        pert.iloc[k, pert.columns.get_loc("high")] *= 1.5
        pert.iloc[k, pert.columns.get_loc("low")] *= 0.7
        pert.iloc[k, pert.columns.get_loc("volume")] *= 8
        p = scored(pert, bench)
        pre = full.index[:k]
        moved = []
        for col in full.columns:
            a = pd.to_numeric(full.loc[pre, col], errors="coerce")
            b = pd.to_numeric(p.loc[pre, col], errors="coerce")
            both = a.notna() & b.notna()
            if both.any() and float((a[both] - b[both]).abs().max()) > 1e-9:
                moved.append(col)
        print(f"perturbation at bar {k}: {len(moved)} earlier columns moved")
        if moved:
            bad.append(f"LOOKAHEAD: perturbing bar {k} changed earlier {moved[:5]}")

    # ---- 3. scale invariance -----------------------------------------------
    K = 7.3
    scaled = df.copy()
    for c in ("open", "high", "low", "close"):
        scaled[c] *= K
    fs = scored(scaled, bench)
    for col in full.columns:
        a = pd.to_numeric(full[col], errors="coerce")
        b = pd.to_numeric(fs[col], errors="coerce")
        both = a.notna() & b.notna()
        if not both.any():
            continue
        if col in LEVEL_COLS:
            d = float(((b[both] / K) - a[both]).abs().max())
            tol = 1e-8 * max(1.0, float(a[both].abs().max()))
            label = "scales"
        else:
            d = float((b[both] - a[both]).abs().max())
            tol = 1e-7 * max(1.0, float(a[both].abs().max()))
            label = "invariant"
        if d > tol:
            bad.append(f"SCALE: {col} should be {label} under price scaling, off by {d:.3e}")
    print(f"price scale invariance: checked {len(full.columns)} columns at k={K}")

    # ---- 4. volume scale invariance ----------------------------------------
    vscaled = df.copy(); vscaled["volume"] *= 1234.5
    fv = scored(vscaled, bench)
    for col in ("vol_ratio", "vol_ratio_3w", "vol_dryup", "down_vol_share",
                "trend_score", "stage2_readiness", "stage4_readiness"):
        a, b = full[col], fv[col]
        both = a.notna() & b.notna()
        if both.any() and float((a[both] - b[both]).abs().max()) > 1e-7:
            bad.append(f"SCALE: {col} changed when volume units changed")
    print("volume scale invariance: ratios and scores unchanged")

    # ---- 5. determinism ----------------------------------------------------
    again = scored(df, bench)
    for col in full.columns:
        a = pd.to_numeric(full[col], errors="coerce")
        b = pd.to_numeric(again[col], errors="coerce")
        both = a.notna() & b.notna()
        if both.any() and float((a[both] - b[both]).abs().max()) != 0.0:
            bad.append(f"DETERMINISM: {col} differs between identical runs")
    print("determinism: identical inputs give identical outputs")

    # ---- 6. bounds ---------------------------------------------------------
    checks = {
        "trend_score": (-100.0, 100.0),
        "stage2_readiness": (0.0, 100.0),
        "stage4_readiness": (0.0, 100.0),
        "group_factor": (0.7, 1.15),
    }
    for col, (lo, hi) in checks.items():
        if col not in full:
            continue
        v = full[col].dropna()
        if len(v) and (v.min() < lo - 1e-9 or v.max() > hi + 1e-9):
            bad.append(f"BOUNDS: {col} ranges [{v.min():.3f}, {v.max():.3f}], expected [{lo}, {hi}]")
    print("bounds: scores inside their stated ranges")

    # ---- 7. missing data must never score better than the best real value --
    base = dict(close=100.0, ma30=95.0, atr=3.0, resistance=104.0, support=88.0,
                recent_low_10w=93.0, recent_high_10w=104.0, base_age_recent=52.0,
                base_age_weeks=52.0, dist_to_resistance_pct=2.0,
                dist_to_support_pct=8.0, ma_slope_pct=0.4, ma_slope_delta=0.4,
                mansfield_rs=4.0, rs_slope=3.0, base_width_pct=18.0,
                tightness=0.5, vol_dryup=0.8, down_vol_share=0.5,
                ret_52w_pct=20.0, ret_13w_pct=-2.0, px_vs_ma_pct=5.0,
                prior_trend_pct=-20.0, group_factor=1.0, group_factor_dn=1.0)
    inputs = ["dist_to_resistance_pct", "ma_slope_pct", "ma_slope_delta", "mansfield_rs",
              "rs_slope", "base_age_weeks", "base_width_pct", "tightness", "vol_dryup",
              "px_vs_ma_pct", "dist_to_support_pct", "ret_52w_pct", "ret_13w_pct",
              "down_vol_share"]
    rewarded = []
    for fn, name in ((S.stage2_readiness, "stage2"), (S.stage4_readiness, "stage4")):
        for col in inputs:
            best = -1e9
            for v in (-1e3, -50, -10, -1, 0, 0.5, 1, 10, 50, 1e3):
                r = dict(base); r[col] = v
                best = max(best, fn(pd.Series(r))[0])
            r = dict(base); r[col] = np.nan
            nan_score = fn(pd.Series(r))[0]
            if nan_score > best + 1e-9:
                rewarded.append(f"{name}/{col}: NaN scores {nan_score:.2f} vs best real {best:.2f}")
    print(f"missing-data reward check: {len(rewarded)} components reward absence")
    bad.extend(f"NaN REWARDED: {r}" for r in rewarded)

    # ---- 8. plan invariants over a random sweep ----------------------------
    viol = {"order": 0, "cap": 0, "budget": 0, "alerts": 0, "r_sign": 0}
    made = 0
    cfg = PL.PlanConfig(account_size=50_000.0, risk_pct=1.0)
    for _ in range(4000):
        px = float(RNG.uniform(1, 500))
        row = pd.Series(dict(
            close=px,
            ma30=px * float(RNG.uniform(0.7, 1.3)),
            atr=px * float(RNG.uniform(0.005, 0.15)),
            resistance=px * float(RNG.uniform(0.9, 1.4)),
            support=px * float(RNG.uniform(0.4, 0.95)),
            recent_low_10w=px * float(RNG.uniform(0.5, 1.05)),
            recent_high_10w=px * float(RNG.uniform(0.95, 1.5)),
            base_age_recent=float(RNG.integers(0, 220))))
        for side in ("long", "short"):
            p = PL.build_plan(row, cfg, side=side)
            if p is None:
                continue
            made += 1
            up = side == "long"
            ok_order = (p["stop"] < p["trigger"] < p["t1"] < p["t2"]) if up else \
                       (p["stop"] > p["trigger"] > p["t1"] > p["t2"])
            if not ok_order:
                viol["order"] += 1
            if p["risk_pct"] > cfg.max_stop_pct + 1e-6:
                viol["cap"] += 1
            if p["t1_r"] < 0 or p["t2_r"] < 0:
                viol["r_sign"] += 1
            sz = p["size"]
            if "shares" in sz and sz["shares"] * p["risk_per_share"] > sz["cash_at_risk"] + 1e-6:
                viol["budget"] += 1
            a = {r["type"]: r["price"] for r in PL.alert_rows({"ticker": "X", "plan": p})}
            seq = ([a["STOP LOSS"], a["BUY LIMIT"], a["BUY STOP"], a["TARGET 1"], a["TARGET 2"]]
                   if up else
                   [a["TARGET 2"], a["TARGET 1"], a["SELL STOP"], a["SELL LIMIT"], a["STOP LOSS"]])
            if any(seq[i] > seq[i + 1] + 1e-9 for i in range(len(seq) - 1)):
                viol["alerts"] += 1
    print(f"plan sweep: {made} plans built, violations {viol}")
    for k, v in viol.items():
        if v:
            bad.append(f"PLAN {k}: {v} of {made} plans violate the invariant")

    # ---- 9. group factor stays on its stated support -----------------------
    vals = [-20, -3.0001, -3, -1, 0, 0.0001, 5, np.nan]
    allowed = {SEC.GROUP_LEADER, SEC.GROUP_MIXED, SEC.GROUP_LAGGARD, 1.0}
    off = [(a, b, SEC.group_factor_row(a, b)) for a in vals for b in vals
           if SEC.group_factor_row(a, b) not in allowed]
    print(f"group factor support: {len(off)} values outside {sorted(allowed)}")
    if off:
        bad.append(f"group factor produced unexpected values: {off[:3]}")

    print("\n" + ("FAILURES:\n  " + "\n  ".join(bad) if bad else "ALL INVARIANTS HOLD"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
