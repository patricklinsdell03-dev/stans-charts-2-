"""
Sector relative strength.

Weinstein's instruction is to buy leaders inside leading groups, and he treats
group strength as a precondition rather than a tiebreak: a good chart in a
deteriorating group is a setup he passes on. Two measurements are needed to
encode that, and they are different questions.

The first is whether the group is beating the market. That is Mansfield relative
strength computed on a sector composite against the same index every stock is
measured against, so a sector and a stock are on directly comparable scales.

The second is whether the stock is beating its own group. A stock can outperform
the market purely by belonging to a hot sector while being one of the weakest
names in it, and that stock is the one that gives back the move first. Measuring
the stock against its own sector composite separates the two.

Composites are built equal weighted from the constituents already downloaded, by
chaining the cross-sectional mean weekly return. Equal weighting is deliberate:
a cap weighted sector index tracks its two largest members, and the question here
is what the group is doing rather than what its megacaps are doing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F

MIN_CONSTITUENTS = 5     # below this a composite is noise, the sector is skipped
LEAD_RS = 0.0            # sector Mansfield above this counts as leading
LAG_RS = -3.0            # below this the sector is a drag

# Multiplicative group factors, same shape as the stage context factor.
GROUP_LEADER = 1.15      # leading sector and the stock leads inside it
GROUP_MIXED = 0.88       # one of the two conditions fails
GROUP_LAGGARD = 0.70     # lagging sector and the stock is not leading it


def build_composites(prices: dict[str, pd.DataFrame],
                     meta: dict[str, dict]) -> dict[str, pd.Series]:
    """
    Returns {sector_key: composite close series}. Sector keys are namespaced by
    market, because the American and British listings use different sector
    taxonomies and ranking them in one pool would be comparing GICS against ICB.
    """
    buckets: dict[str, list[pd.Series]] = {}
    for t, df in prices.items():
        m = meta.get(t)
        if not m:
            continue
        sec = str(m.get("sector") or "").strip()
        if not sec or sec.lower() in ("nan", "none", "-"):
            continue
        key = f"{m['market']} / {sec}"
        buckets.setdefault(key, []).append(df["close"].astype(float))

    out = {}
    for key, series in buckets.items():
        if len(series) < MIN_CONSTITUENTS:
            continue
        wide = pd.concat(series, axis=1).sort_index()
        rets = wide.pct_change()
        # Equal weighted cross-sectional mean, ignoring names without a bar that
        # week rather than treating a missing bar as a zero return.
        mean_ret = rets.mean(axis=1, skipna=True)
        n_live = rets.notna().sum(axis=1)
        mean_ret = mean_ret.where(n_live >= MIN_CONSTITUENTS)
        comp = 100.0 * (1.0 + mean_ret.fillna(0.0)).cumprod()
        comp = comp.where(n_live.cummax() >= MIN_CONSTITUENTS)
        out[key] = comp.dropna()
    return out


def sector_frames(composites: dict[str, pd.Series],
                  benchmarks: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """Mansfield relative strength and trend for each composite, against its market."""
    out = {}
    for key, comp in composites.items():
        market = key.split(" / ")[0]
        bench = benchmarks.get(market)
        if bench is None or len(comp) < F.MIN_HISTORY:
            continue
        ma = comp.rolling(F.MA_LENGTH, min_periods=F.MA_LENGTH).mean()
        df = pd.DataFrame({
            "close": comp,
            "ma30": ma,
            "ma_slope_pct": (ma / ma.shift(F.MA_SLOPE_LOOKBACK) - 1.0) * 100.0,
            "px_vs_ma_pct": (comp / ma - 1.0) * 100.0,
            "sector_rs": F.mansfield_rs(comp, bench),
        })
        df["sector_rs_slope"] = df["sector_rs"] - df["sector_rs"].shift(4)
        out[key] = df
    return out


def rank_sectors(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Cross-sectional percentile rank of sector relative strength, computed week by
    week within each market. Ranking across markets would be meaningless, since
    the two composites are measured against different indices.
    """
    if not frames:
        return pd.DataFrame()
    rs = pd.DataFrame({k: v["sector_rs"] for k, v in frames.items()})
    ranks = pd.DataFrame(index=rs.index, columns=rs.columns, dtype=float)
    for market in {k.split(" / ")[0] for k in rs.columns}:
        cols = [c for c in rs.columns if c.startswith(market + " / ")]
        sub = rs[cols]
        ranks[cols] = sub.rank(axis=1, pct=True) * 100.0
    return ranks


def attach_to_stock(feat: pd.DataFrame, close: pd.Series,
                    sector_frame: pd.DataFrame | None,
                    sector_rank: pd.Series | None) -> pd.DataFrame:
    """
    Adds sector_rs, sector_rank_pct, rs_vs_sector and group_factor to a stock's
    feature frame. Missing sector data leaves a neutral factor of 1.0 rather than
    a penalty, so a stock with no sector label is ranked on its own chart instead
    of being quietly demoted.
    """
    feat = feat.copy()
    if sector_frame is None or sector_frame.empty:
        feat["sector_rs"] = np.nan
        feat["sector_rs_slope"] = np.nan
        feat["sector_rank_pct"] = np.nan
        feat["rs_vs_sector"] = np.nan
        feat["group_factor"] = 1.0
        feat["group_factor_dn"] = 1.0
        return feat

    sf = sector_frame.reindex(feat.index).ffill()
    feat["sector_rs"] = sf["sector_rs"]
    feat["sector_rs_slope"] = sf["sector_rs_slope"]
    feat["sector_rank_pct"] = (sector_rank.reindex(feat.index).ffill()
                               if sector_rank is not None else np.nan)
    # The stock measured against its own group rather than against the market.
    feat["rs_vs_sector"] = F.mansfield_rs(close, sf["close"])
    feat["group_factor"] = group_factor_series(feat)
    feat["group_factor_dn"] = group_factor_series_dn(feat)
    return feat


def group_factor_series(feat: pd.DataFrame) -> pd.Series:
    """
    Vectorised twin of group_factor_row, written as an ordered np.select so the
    branch order is literally the same as the row version. Expressing the same
    branches as independent boolean masks is what let a case fall through
    unassigned the first time this was written.
    """
    sec, own = feat["sector_rs"], feat["rs_vs_sector"]
    leading = (sec > LEAD_RS).fillna(False).to_numpy()
    lagging = (sec < LAG_RS).fillna(False).to_numpy()
    leads = (own > 0).fillna(False).to_numpy()
    f = np.select(
        [leading & leads, lagging & ~leads, lagging | ~leads],
        [GROUP_LEADER, GROUP_LAGGARD, GROUP_MIXED],
        default=1.0)
    f = pd.Series(f, index=feat.index, dtype=float)
    f[sec.isna() | own.isna()] = 1.0
    return f


def group_factor_series_dn(feat: pd.DataFrame) -> pd.Series:
    """
    Mirror of the long factor. A weak group with the stock lagging inside it is
    the best short backdrop, and a leading group is the worst, because a strong
    group repeatedly rescues its weakest members before they break.
    """
    sec, own = feat["sector_rs"], feat["rs_vs_sector"]
    leading = (sec > LEAD_RS).fillna(False).to_numpy()
    lagging = (sec < LAG_RS).fillna(False).to_numpy()
    leads = (own > 0).fillna(False).to_numpy()
    f = np.select(
        [lagging & ~leads, leading & leads, leading | leads],
        [GROUP_LEADER, GROUP_LAGGARD, GROUP_MIXED],
        default=1.0)
    f = pd.Series(f, index=feat.index, dtype=float)
    f[sec.isna() | own.isna()] = 1.0
    return f


def group_factor_row_dn(sector_rs, rs_vs_sector) -> float:
    if sector_rs is None or rs_vs_sector is None:
        return 1.0
    if (isinstance(sector_rs, float) and np.isnan(sector_rs)) or \
       (isinstance(rs_vs_sector, float) and np.isnan(rs_vs_sector)):
        return 1.0
    leading = sector_rs > LEAD_RS
    lagging = sector_rs < LAG_RS
    leads_group = rs_vs_sector > 0
    if lagging and not leads_group:
        return GROUP_LEADER
    if leading and leads_group:
        return GROUP_LAGGARD
    if leading or leads_group:
        return GROUP_MIXED
    return 1.0


def group_factor_row(sector_rs, rs_vs_sector) -> float:
    if sector_rs is None or rs_vs_sector is None:
        return 1.0
    if (isinstance(sector_rs, float) and np.isnan(sector_rs)) or \
       (isinstance(rs_vs_sector, float) and np.isnan(rs_vs_sector)):
        return 1.0
    leading = sector_rs > LEAD_RS
    lagging = sector_rs < LAG_RS
    leads_group = rs_vs_sector > 0
    if leading and leads_group:
        return GROUP_LEADER
    if lagging and not leads_group:
        return GROUP_LAGGARD
    if lagging or not leads_group:
        return GROUP_MIXED
    return 1.0


def summary_table(frames: dict[str, pd.DataFrame], ranks: pd.DataFrame,
                  scan: pd.DataFrame) -> list[dict]:
    """One row per sector for the dashboard, as of the latest common week."""
    rows = []
    for key, df in frames.items():
        d = df.dropna(subset=["sector_rs"])
        if d.empty:
            continue
        last = d.iloc[-1]
        rank = float(ranks[key].dropna().iloc[-1]) if key in ranks and ranks[key].notna().any() else np.nan
        market, sector = key.split(" / ", 1)
        sub = scan[(scan["market"] == market) & (scan["sector"] == sector)] if not scan.empty else scan
        rows.append({
            "sector": sector, "market": market,
            "sector_rs": round(float(last["sector_rs"]), 2),
            "sector_rs_slope": None if pd.isna(last["sector_rs_slope"]) else round(float(last["sector_rs_slope"]), 2),
            "ma_slope_pct": None if pd.isna(last["ma_slope_pct"]) else round(float(last["ma_slope_pct"]), 2),
            "rank_pct": None if np.isnan(rank) else round(rank, 1),
            "n": int(len(sub)),
            "in_stage2": int((sub["stage"] == "Stage 2").sum()) if len(sub) else 0,
            "in_stage4": int((sub["stage"] == "Stage 4").sum()) if len(sub) else 0,
        })
    rows.sort(key=lambda r: (r["market"], -r["sector_rs"]))
    return rows
