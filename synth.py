"""
Synthetic multi-sector universe.

Used by the offline demo and by the sector test. The whole scanner runs on it
without a network connection, which is what makes it possible to verify the
sector and regime layers rather than assert that they work.

Construction is deliberate rather than random. Each sector has a common factor
that drives every member, plus idiosyncratic noise per name. The same base and
breakout pattern is planted in one member of the strongest sector and one member
of the weakest, so the two charts are near identical and only their group
differs. If the group layer is doing anything, those two come out graded
differently.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import backtest as B
from . import features as F
from . import market as M
from . import plan as PL
from . import sectors as SEC
from . import explain as EX
from . import precheck as PC
from . import stages as S

N_WEEKS = 300
PER_SECTOR = 8

SECTOR_DRIFT = {          # weekly drift of each sector's common factor
    "Leading Sector": 0.0055,
    "Middling Sector": 0.0015,
    "Lagging Sector": -0.0035,
}


def build_universe(seed: int = 5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp("2026-08-21"), periods=N_WEEKS, freq="W-FRI")
    n = len(idx)

    market = pd.Series(100 * np.cumprod(1 + 0.0018 + rng.normal(0, 0.012, n)), index=idx)

    def to_bars(closes, vols):
        c = np.asarray(closes, float)
        hi = c * (1 + np.abs(rng.normal(0, .005, len(c))) + .006)
        lo = c * (1 - np.abs(rng.normal(0, .005, len(c))) - .006)
        return pd.DataFrame({"open": c, "high": hi, "low": lo, "close": c,
                             "volume": vols}, index=idx[:len(c)])

    def planted_base_and_break():
        """70 weeks of decline, 227 of base, then a fresh breakout on triple volume.

        The advance is only three weeks long on purpose, so the break is still
        inside the four week window the scanner reports on and the planted
        signal actually appears in the current output rather than in history.
        """
        px, v, p = [], [], 100.0
        for _ in range(70):
            p *= (1 - .010 + rng.normal(0, .012)); px.append(p); v.append(1.2e6)
        b = p
        for i in range(227):
            px.append(b * (1 + .05 * (1 - i / 272) * np.sin(i / 4.5) + rng.normal(0, .006)))
            v.append(.55e6)
        for i in range(n - len(px)):
            px.append(px[-1] * (1 + .017 + rng.normal(0, .010)))
            v.append(3.2e6 if i == 0 else 1.6e6)
        return to_bars(px[:n], v[:n])

    prices, meta = {}, {}
    for sector, drift in SECTOR_DRIFT.items():
        common = np.cumprod(1 + drift + rng.normal(0, 0.010, n))
        tag = sector.split()[0][:3].upper()
        for j in range(PER_SECTOR):
            tkr = f"{tag}{j + 1}"
            if j == 0:
                df = planted_base_and_break()
            else:
                idio = np.cumprod(1 + rng.normal(0, 0.014, n))
                closes = 100 * common * idio
                vols = np.full(n, 1.0e6) * (1 + np.abs(rng.normal(0, .3, n)))
                df = to_bars(closes, vols)
            prices[tkr] = df
            meta[tkr] = {"name": f"{sector} name {j + 1}", "sector": sector,
                         "market": "US", "index": "SYNTHETIC"}
    return idx, market, prices, meta


def build_demo(seed: int = 5) -> dict:
    """Runs the full pipeline offline and returns the same dict shape as scan.run."""
    idx, market_close, prices, meta = build_universe(seed)
    benches = {"US": market_close}

    comps = SEC.build_composites(prices, meta)
    sec_frames = SEC.sector_frames(comps, benches)
    sec_ranks = SEC.rank_sectors(sec_frames)

    feats, flags = {}, []
    for t, df in prices.items():
        feat = F.compute_features(df, market_close)
        key = f"US / {meta[t]['sector']}"
        feat = SEC.attach_to_stock(feat, df["close"].astype(float),
                                   sec_frames.get(key),
                                   sec_ranks[key] if key in sec_ranks else None)
        feat["trend_score"] = S.trend_score_series(feat)
        feat = feat.join(S.readiness_series(feat))
        feats[t] = feat
        flags.append((feat["close"] > feat["ma30"]).astype(float)
                     .where(feat["ma30"].notna()).rename(t))

    breadth = M.breadth_from_features(flags)
    regime = M.regime_series(market_close, breadth)
    ok_now = M.market_ok(str(regime["regime"].iloc[-1]))

    records, outs, leads = [], [], []
    for t, feat in feats.items():
        m = meta[t]
        rec = S.evaluate(feat, t, m["name"], m["market"], m["sector"], market_ok=ok_now)
        if rec is None:
            continue
        rec["index"] = m["index"]
        kind = (rec.get("signal") or {}).get("kind")
        side = ("short" if kind == "stage4_breakdown"
                else "long" if kind == "stage2_breakout"
                else "short" if (rec["stage4_readiness"] or 0) > (rec["stage2_readiness"] or 0)
                else "long")
        rec["plan"] = PL.build_plan(feat.iloc[-1], PL.DEFAULT, side=side)
        records.append(rec)
        o = B.signal_outcomes(feat, market_close)
        if not o.empty:
            o["ticker"] = t
            outs.append(o)
        leads.append(B.readiness_lead(feat))

    regime_now = str(regime["regime"].iloc[-1])
    asof_str = str(pd.Timestamp(records[0]["date"]).date()) if records else ""
    for rec in records:
        rec["explain"] = EX.explain(rec, {"regime": regime_now})
        rec["prompt_vals"] = PC.fields(rec, asof_str, regime_now)

    scan = pd.DataFrame(records)
    oc = pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()
    ld = pd.concat(leads, ignore_index=True)
    if not oc.empty:
        oc = B.join_regime(oc, regime)
    calib = {
        "signals": B.summarise_outcomes(oc).to_dict("records") if not oc.empty else [],
        "by_regime": B.split_table(oc, "regime").to_dict("records") if not oc.empty else [],
        "by_group": B.split_table(oc, "group_bucket").to_dict("records") if not oc.empty else [],
        "stage2_deciles": B.readiness_decile_table(ld, "stage2_readiness", "break_up_next").to_dict("records"),
        "stage4_deciles": B.readiness_decile_table(ld, "stage4_readiness", "break_dn_next").to_dict("records"),
        "discrimination_up": B.readiness_discrimination(
            ld, "stage2_readiness", "dist_to_resistance_pct", "break_up_next"),
        "discrimination_dn": B.readiness_discrimination(
            ld, "stage4_readiness", "dist_to_support_pct", "break_dn_next"),
        "uncertainty": B.uncertainty_table(oc).to_dict("records") if not oc.empty else [],
        "sample_signals": int(len(oc)), "sample_weeks": int(len(ld)),
    }
    calib["comparisons"] = B.comparison_budget(calib)
    # Two synthetic open positions so the management panel has something to say.
    held = []
    for t, weeks_ago in (("LEA3", 30), ("MID4", 12)):
        if t not in feats:
            continue
        f = feats[t]
        d = f.index[-weeks_ago]
        held.append({"ticker": t, "entry_date": str(d.date()), "side": "long",
                     "entry_price": float(f["close"].loc[d]), "shares": 100,
                     "initial_stop": float(f["stop_suggestion"].loc[d])})
    managed = [PL.manage_position(feats[h["ticker"]], h) for h in held]

    return {"scan": scan, "calibration": calib, "asof": scan["date"].max(),
            "positions": managed,
            "generated": dt.datetime.now(dt.timezone.utc),
            "market": M.summary(regime, scan),
            "sectors": SEC.summary_table(sec_frames, sec_ranks, scan),
            "regime_history": regime[["regime_score", "breadth_above_ma"]].dropna().tail(156),
            "_frames": sec_frames, "_ranks": sec_ranks, "_regime": regime}
