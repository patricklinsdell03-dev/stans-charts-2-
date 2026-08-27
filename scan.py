"""
Orchestration.

Order matters here and is not the obvious one. Sector composites have to exist
before any stock is scored, because a stock's group factor depends on them.
Breadth can only be computed after every stock has a 30 week average, and the
market regime depends on breadth, and the grade on each signal depends on the
regime. So the run is: composites, then one feature pass, then breadth and
regime, then the verdicts.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import backtest as B
from . import data as D
from . import features as F
from . import market as M
from . import plan as PL
from . import sectors as SEC
from . import explain as EX
from . import precheck as PC
from . import stages as S
from . import universe as U


def run(indices: list[str], years: int = 8, use_cache: bool = False,
        keep_partial: bool = False, calibrate: bool = True,
        min_price: float = 1.0, plan_cfg: "PL.PlanConfig | None" = None,
        positions: list[dict] | None = None,
        prices: dict | None = None, universe_file: str | None = None) -> dict:
    plan_cfg = plan_cfg or PL.DEFAULT
    print("building universe")
    uni = U.build_universe(indices, refresh=not use_cache, universe_file=universe_file)
    print(f"  {len(uni)} tickers across {uni['index'].nunique()} indices")

    bench_tickers = sorted({U.BENCHMARKS[m] for m in uni["market"].unique()})
    wanted = list(uni["ticker"]) + bench_tickers

    if prices is not None:
        # Supplied by the caller, which is how the horizon comparison runs several
        # history lengths off one download instead of hammering the source.
        px = prices
    elif use_cache:
        print("loading cached prices")
        px = D.load_cache()
    else:
        print("downloading weekly bars")
        px = D.download_weekly(wanted, years=years)
        D.save_cache(px)

    if not keep_partial:
        px = {t: D.drop_partial_week(df) for t, df in px.items()}

    benches = {m: px[bt]["close"] for m, bt in U.BENCHMARKS.items() if bt in px}
    if not benches:
        raise RuntimeError("no benchmark series available, relative strength cannot be computed")

    meta = uni.set_index("ticker")[["name", "sector", "market", "index"]].to_dict("index")
    stocks = {t: df for t, df in px.items() if t not in bench_tickers and t in meta}

    # --- 1. sector composites ------------------------------------------------
    print("building sector composites")
    comps = SEC.build_composites(stocks, meta)
    sec_frames = SEC.sector_frames(comps, benches)
    sec_ranks = SEC.rank_sectors(sec_frames)
    print(f"  {len(sec_frames)} sectors with enough history")

    # --- 2. one feature pass -------------------------------------------------
    print("scoring")
    feats: dict[str, pd.DataFrame] = {}
    above_flags: list[pd.Series] = []

    for t, df in stocks.items():
        if len(df) < F.MIN_HISTORY:
            continue
        # A ticker below the price floor is excluded from the RANKED OUTPUT but
        # still contributes to breadth. Dropping it from history entirely, which
        # is what the combined filter used to do, conditions the whole breadth
        # series on today's price: a name that traded at 200 in 2019 and 50p now
        # was retroactively deleted from every week it ever traded, which shifted
        # historical breadth by up to seven points and relabelled the regime in
        # roughly one week in twelve.
        too_cheap = float(df["close"].iloc[-1]) < min_price
        m = meta[t]
        bench = benches.get(m["market"]) or next(iter(benches.values()))
        try:
            feat = F.compute_features(df, bench)
        except Exception as e:                       # noqa: BLE001
            print(f"  {t}: feature error {e}")
            continue

        key = f"{m['market']} / {str(m.get('sector') or '').strip()}"
        feat = SEC.attach_to_stock(
            feat, df["close"].astype(float), sec_frames.get(key),
            sec_ranks[key] if key in sec_ranks else None)

        feat["trend_score"] = S.trend_score_series(feat)
        feat = feat.join(S.readiness_series(feat))
        flag = (feat["close"] > feat["ma30"]).astype(float)
        above_flags.append(flag.where(feat["ma30"].notna()).rename(t))
        if too_cheap:
            continue
        feats[t] = feat

    # --- 3. breadth and regime ----------------------------------------------
    breadth = M.breadth_from_features(above_flags)
    primary = "US" if "US" in benches else next(iter(benches))
    regime = M.regime_series(benches[primary], breadth)
    regime_now = str(regime["regime"].iloc[-1]) if not regime.empty else "Unknown"
    ok_now = M.market_ok(regime_now)
    print(f"  market regime: {regime_now} (breadth {breadth.iloc[-1]:.0f}% above the 30 week line)"
          if len(breadth) else f"  market regime: {regime_now}")

    # --- 4. verdicts ---------------------------------------------------------
    records, outcomes, leads = [], [], []
    for t, feat in feats.items():
        m = meta[t]
        rec = S.evaluate(feat, t, m["name"], m["market"], m["sector"], market_ok=ok_now)
        if rec is None:
            continue
        rec["index"] = m["index"]

        # A plan for the side the setup is actually on. Attaching both would
        # double the payload and invite reading the wrong one.
        kind = (rec.get("signal") or {}).get("kind")
        side = ("short" if kind == "stage4_breakdown"
                else "long" if kind == "stage2_breakout"
                else "short" if (rec["stage4_readiness"] or 0) > (rec["stage2_readiness"] or 0)
                else "long")
        try:
            rec["plan"] = PL.build_plan(feat.iloc[-1], plan_cfg, side=side)
        except Exception:                            # noqa: BLE001
            rec["plan"] = None
        records.append(rec)

        if calibrate:
            try:
                bench = benches.get(m["market"]) or next(iter(benches.values()))
                o = B.signal_outcomes(feat, bench)
                if not o.empty:
                    o["ticker"] = t
                    outcomes.append(o)
                leads.append(B.readiness_lead(feat))
            except Exception:                        # noqa: BLE001
                pass

    # Plain-English translation and the pre-trade prompt fields, added once the
    # regime is known so both can refer to it.
    asof_str = str(pd.Timestamp(records[0]["date"]).date()) if records else ""
    for rec in records:
        rec["explain"] = EX.explain(rec, {"regime": regime_now})
        rec["prompt_vals"] = PC.fields(rec, asof_str, regime_now)

    scan = pd.DataFrame(records)
    print(f"  scored {len(scan)} tickers")

    calib = {}
    if calibrate and outcomes:
        oc = pd.concat(outcomes, ignore_index=True)
        ld = pd.concat(leads, ignore_index=True)
        oc = B.join_regime(oc, regime)
        calib = {
            "signals": B.summarise_outcomes(oc).to_dict("records"),
            "by_regime": B.split_table(oc, "regime").to_dict("records"),
            "by_group": B.split_table(oc, "group_bucket").to_dict("records"),
            "stage2_deciles": B.readiness_decile_table(
                ld, "stage2_readiness", "break_up_next").to_dict("records"),
            "stage4_deciles": B.readiness_decile_table(
                ld, "stage4_readiness", "break_dn_next").to_dict("records"),
            "discrimination_up": B.readiness_discrimination(
                ld, "stage2_readiness", "dist_to_resistance_pct", "break_up_next"),
            "discrimination_dn": B.readiness_discrimination(
                ld, "stage4_readiness", "dist_to_support_pct", "break_dn_next"),
            "uncertainty": B.uncertainty_table(oc).to_dict("records"),
            "sample_signals": int(len(oc)),
            "sample_weeks": int(len(ld)),
        }
        calib["comparisons"] = B.comparison_budget(calib)

    # --- 5. open positions ---------------------------------------------------
    managed = []
    for pos in (positions or []):
        t = str(pos.get("ticker", "")).strip()
        if t not in feats:
            managed.append({"ticker": t, "action": "UNKNOWN",
                            "reason": "not in the scanned universe this week"})
            continue
        try:
            managed.append(PL.manage_position(feats[t], pos, plan_cfg))
        except Exception as e:                       # noqa: BLE001
            managed.append({"ticker": t, "action": "UNKNOWN", "reason": f"plan error: {e}"})

    asof = scan["date"].max() if not scan.empty else pd.NaT
    return {"scan": scan, "calibration": calib, "asof": asof, "positions": managed,
            "generated": dt.datetime.now(dt.timezone.utc),
            "market": M.summary(regime, scan),
            "sectors": SEC.summary_table(sec_frames, sec_ranks, scan),
            "regime_history": regime[["regime_score", "breadth_above_ma"]].dropna().tail(156)}


def slice_lists(scan: pd.DataFrame, top: int = 40) -> dict[str, pd.DataFrame]:
    """The four lists the dashboard is built from."""
    if scan.empty:
        return {}
    sig_kind = scan["signal"].apply(lambda s: s.get("kind") if isinstance(s, dict) else None)
    sig_conf = scan["signal"].apply(lambda s: bool(s.get("confirmed")) if isinstance(s, dict) else False)
    grade_rank = scan.get("grade", pd.Series(index=scan.index, dtype=object)).map(
        {"A": 0, "B": 1, "C": 2, "D": 3}).fillna(9)
    s = scan.assign(_kind=sig_kind, _conf=sig_conf, _g=grade_rank)

    breakouts = s[s["_kind"] == "stage2_breakout"].sort_values(
        ["_g", "stage2_readiness"], ascending=[True, False])
    breakdowns = s[s["_kind"] == "stage4_breakdown"].sort_values(
        ["_g", "stage4_readiness"], ascending=[True, False])

    pre2 = s[(~s["stage"].isin(["Stage 2", "Stage 2 to 3"]))
             & (s["_kind"] != "stage2_breakout")
             & (s["dist_to_resistance_pct"].fillna(99) >= 0)]
    pre2 = pre2.sort_values("stage2_readiness", ascending=False).head(top)

    pre4 = s[(s["stage"].isin(["Stage 2 to 3", "Stage 3", "Stage 3 to 4"]))
             & (s["_kind"] != "stage4_breakdown")]
    pre4 = pre4.sort_values("stage4_readiness", ascending=False).head(top)

    return {"breakouts": breakouts, "breakdowns": breakdowns,
            "watch_stage2": pre2, "watch_stage4": pre4}
