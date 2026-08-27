"""
Calibration. The part that turns the scanner from an assertion into a measurement.

Two questions are worth answering before trusting any of this with money.

First, when a confirmed stage 2 breakout fires, what actually happened next? Not
on average across a cherry picked decade, but as a distribution: median forward
excess return over the benchmark at four, thirteen and twenty six weeks, the share
of signals that were still above the entry, and the share that hit the suggested
stop first. Weinstein's own claim is that the method wins by cutting losers at the
stop and letting a minority of large winners carry the book, which means the mean
and the hit rate tell you almost nothing on their own.

Second, does the readiness score have any forward information? The scanner claims
to find breakouts before they happen, so the test is direct: bucket every historic
week by its readiness decile and measure how often a confirmed breakout arrived
within the next eight weeks. If the top decile does not clearly beat the bottom
decile, the readiness model is decoration and should be rebuilt or ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from . import stages as S


def signal_outcomes(feat: pd.DataFrame, bench_close: pd.Series,
                    horizons=(4, 13, 26)) -> pd.DataFrame:
    """Forward outcomes for every flagged and confirmed break in one ticker."""
    br = S.breaks_series(feat)
    close = feat["close"]
    bench = bench_close.reindex(close.index).ffill()
    rows = []

    for kind, flag_col, conf_col, sign in (
        ("stage2_breakout", "flagged_up", "confirmed_up", 1),
        ("stage4_breakdown", "flagged_dn", "confirmed_dn", -1),
    ):
        idx = feat.index[br[flag_col].to_numpy()]
        for ts in idx:
            i = feat.index.get_loc(ts)
            rec = {
                "date": ts, "kind": kind,
                "confirmed": bool(br[conf_col].iloc[i]),
                "entry": float(close.iloc[i]),
                "stage2_readiness_prev": _prev(feat, i, "stage2_readiness"),
                "stage4_readiness_prev": _prev(feat, i, "stage4_readiness"),
                "mansfield_rs": float(feat["mansfield_rs"].iloc[i]),
                "vol_ratio_3w": float(feat["vol_ratio_3w"].iloc[i])
                if "vol_ratio_3w" in feat else np.nan,
                "group_factor": _at(feat, i, "group_factor"),
                "group_factor_dn": _at(feat, i, "group_factor_dn"),
                "sector_rs": _at(feat, i, "sector_rs"),
            }
            stop = feat["stop_suggestion"].iloc[i]
            for h in horizons:
                j = i + h
                if j >= len(close):
                    rec[f"fwd_{h}w"] = np.nan
                    rec[f"exc_{h}w"] = np.nan
                    continue
                # LOG excess return. Differencing simple returns carries a
                # Jensen bias proportional to variance: at zero alpha a 55 per
                # cent volatility stock shows a median simple return 6.7 points
                # BELOW a 16 per cent volatility index over 26 weeks purely from
                # the arithmetic. Worse, the sign flip for shorts reverses that
                # bias rather than cancelling it, so the long table was pushed
                # down and the short table up by the same three to eleven points,
                # which is larger than any plausible real edge. Logs difference
                # exactly and the bias disappears.
                r = np.log(close.iloc[j] / close.iloc[i]) * 100.0 * sign
                b = np.log(bench.iloc[j] / bench.iloc[i]) * 100.0 * sign
                rec[f"fwd_{h}w"] = r
                rec[f"exc_{h}w"] = r - b
            # Did the initial stop get hit inside 13 weeks, before any gain?
            # Against the weekly LOW, not the close. A stop is a resting order;
            # it fills on the low. Testing closes understated the stop-out rate
            # by four to six percentage points, which is eight to fourteen per
            # cent of the number itself, and the stop-out rate is the single
            # most load-bearing statistic in a method whose stated edge is
            # cutting losers.
            low_col = "low" if "low" in feat else "close"
            if sign == 1 and not np.isnan(stop):
                seg = feat[low_col].iloc[i + 1:i + 14]
                # Right-censored rather than recorded as a survival.
                rec["stopped_13w"] = (bool((seg < stop).any())
                                      if len(seg) >= 13 else np.nan)
            else:
                rec["stopped_13w"] = np.nan
            rows.append(rec)

    return pd.DataFrame(rows)


def _at(feat, i, col):
    if col not in feat:
        return np.nan
    v = feat[col].iloc[i]
    return float(v) if pd.notna(v) else np.nan


def _prev(feat, i, col):
    if col not in feat or i == 0:
        return np.nan
    v = feat[col].iloc[i - 1]
    return float(v) if pd.notna(v) else np.nan


def readiness_lead(feat: pd.DataFrame, horizon: int = 8) -> pd.DataFrame:
    """
    For every week, the readiness score and whether a confirmed break followed
    inside `horizon` weeks. This is the sample the decile table is built from.
    """
    br = S.breaks_series(feat)

    def forward_any(flags: pd.Series) -> pd.Series:
        # Reverse, take a rolling max, reverse back, then shift by one so the
        # window covers weeks i+1 through i+horizon and never includes the
        # current week. Including the current week would leak the answer into
        # the feature and make the readiness score look prescient.
        f = flags.astype(float)
        return f.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)

    fut_up = forward_any(br["confirmed_up"])
    fut_dn = forward_any(br["confirmed_dn"])
    out = pd.DataFrame({
        "stage2_readiness": feat.get("stage2_readiness"),
        "stage4_readiness": feat.get("stage4_readiness"),
        "dist_to_resistance_pct": feat.get("dist_to_resistance_pct"),
        "dist_to_support_pct": feat.get("dist_to_support_pct"),
        "break_up_next": fut_up.fillna(0).astype(bool),
        "break_dn_next": fut_dn.fillna(0).astype(bool),
    }, index=feat.index)
    # The final `horizon` weeks cannot have a full look-ahead window, and
    # labelling them False says "no break followed" when the window was simply
    # too short. That is a one-directional error landing on the most recent and
    # highest-scoring rows, so they are dropped rather than mislabelled.
    return out.iloc[:-horizon] if horizon < len(out) else out.iloc[0:0]


def summarise_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = []
    for (kind, conf), g in df.groupby(["kind", "confirmed"]):
        row = {"kind": kind, "confirmed": conf, "n": len(g)}
        for h in (4, 13, 26):
            c = f"exc_{h}w"
            if c in g:
                row[f"median_excess_{h}w"] = round(float(g[c].median()), 2)
                row[f"win_rate_{h}w"] = round(float((g[c] > 0).mean() * 100), 1)
                row[f"p90_excess_{h}w"] = round(float(g[c].quantile(0.90)), 2)
                row[f"p10_excess_{h}w"] = round(float(g[c].quantile(0.10)), 2)
        if "stopped_13w" in g:
            row["stopped_out_13w_pct"] = round(float(g["stopped_13w"].mean() * 100), 1) \
                if g["stopped_13w"].notna().any() else None
        out.append(row)
    return pd.DataFrame(out)


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC. 0.5 is coin-flip, 1.0 is perfect separation."""
    ok = ~(np.isnan(score) | np.isnan(label.astype(float)))
    s, y = score[ok], label[ok].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def readiness_discrimination(df: pd.DataFrame, score_col: str, dist_col: str,
                             hit_col: str) -> dict:
    """
    The honest version of "does the readiness score lead the break".

    The decile table this replaces could not fail. Its largest single component
    is proximity to the level, worth 25 of 100 points, and its label is crossing
    that level. Those are the same monotone function of the same variable, so on
    a pure random walk with no forecastable structure at all the top decile beat
    the bottom by 6.5 points at an apparent 28.7 sigma. A test that noise passes
    at 28.7 sigma measures nothing.

    What actually needs answering is whether the score adds anything BEYOND
    knowing how far price sits from the pivot. So this reports three numbers:
    the score's own AUC, the AUC of proximity alone, and the increment. If the
    increment is near zero, the eighty points of moving-average turn, relative
    strength, base quality and volume behaviour are decoration, and the tool is
    a distance-to-pivot screen wearing a Weinstein costume.
    """
    d = df[[score_col, dist_col, hit_col]].dropna()
    if len(d) < 200 or d[hit_col].nunique() < 2:
        return {}
    score = d[score_col].to_numpy(dtype=float)
    dist = -d[dist_col].to_numpy(dtype=float)      # nearer is better
    y = d[hit_col].to_numpy()
    auc_score = _auc(score, y)
    auc_dist = _auc(dist, y)

    # The score with its proximity contribution removed, by residualising the
    # score on distance and scoring the residual. Anything the residual predicts
    # is information the other components carry on their own.
    ok = np.isfinite(score) & np.isfinite(dist)
    if ok.sum() > 50 and np.std(dist[ok]) > 0:
        beta = np.polyfit(dist[ok], score[ok], 1)
        resid = score - np.polyval(beta, dist)
        auc_resid = _auc(resid, y)
    else:
        auc_resid = float("nan")
    return {
        "n": int(len(d)), "base_rate": round(float(y.mean()) * 100, 2),
        "auc_score": round(auc_score, 4),
        "auc_proximity_only": round(auc_dist, 4),
        "auc_residual": round(auc_resid, 4),
        "increment": round(auc_score - auc_dist, 4),
    }


def readiness_decile_table(df: pd.DataFrame, score_col: str, hit_col: str) -> pd.DataFrame:
    d = df[[score_col, hit_col]].dropna()
    if len(d) < 200:
        return pd.DataFrame()
    d = d.copy()
    # Retained as a description of the sample, NOT as evidence that the score
    # works. The cut points are pooled across the whole history, so decile
    # membership is partly "which calendar week is this", and the target is
    # nearly a restatement of the score's own proximity term. Read
    # readiness_discrimination instead.
    d["decile"] = pd.qcut(d[score_col].rank(method="first"), 10, labels=False) + 1
    g = d.groupby("decile").agg(n=(hit_col, "size"),
                                hit_rate=(hit_col, "mean"),
                                score_lo=(score_col, "min"),
                                score_hi=(score_col, "max")).reset_index()
    g["hit_rate"] = (g["hit_rate"] * 100).round(1)
    g["score_lo"] = g["score_lo"].round(1)
    g["score_hi"] = g["score_hi"].round(1)
    return g


# ----------------------------------------------------------------------------
# Environment splits
#
# The point of the market and sector layers is a testable claim: the same chart
# pattern should do measurably better in a supportive regime and a leading group
# than in a hostile one. These tables are how that claim gets checked rather than
# assumed. If the splits come back flat, the two filters are costing signals
# without buying anything and should be loosened or dropped.
# ----------------------------------------------------------------------------

def join_regime(outcomes: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if outcomes.empty:
        return outcomes
    out = outcomes.copy()
    if not regime.empty and "regime" in regime:
        lab = regime["regime"].reindex(pd.DatetimeIndex(out["date"]), method="ffill")
        out["regime"] = lab.to_numpy()
    else:
        out["regime"] = "Unknown"

    def bucket(r):
        """
        The label describes the SETUP quality, not the sector direction.

        The short-side factor is a mirror: 1.15 there means a lagging group with
        the stock lagging inside it, which is the best short backdrop. Labelling
        that "leading group" made the long and short blocks of the same table run
        on inverted scales, under a caption inviting the reader to compare them.
        Both sides now read as "favourable" through "hostile", which is what the
        factor actually measures on each side.
        """
        short = r["kind"] == "stage4_breakdown"
        g = r["group_factor_dn"] if short else r["group_factor"]
        if g is None or (isinstance(g, float) and np.isnan(g)):
            return "no sector data"
        if g >= 1.10:
            return "favourable group (leader in a leading group)" if not short \
                else "favourable group (laggard in a lagging group)"
        if g >= 1.0:
            return "neutral group"
        if g >= 0.85:
            return "mixed group"
        return "hostile group"

    out["group_bucket"] = out.apply(bucket, axis=1)
    # A group factor is never NaN out of the pipeline, because sectors.py sets a
    # neutral 1.0 when it has no sector data. That silently pooled the
    # unclassified names into the neutral bucket, contaminating the one row the
    # table uses as its baseline. Mark them explicitly instead.
    if "sector_rs" in out:
        out.loc[out["sector_rs"].isna(), "group_bucket"] = "no sector data"
    return out


def split_table(outcomes: pd.DataFrame, by: str) -> pd.DataFrame:
    """Median excess return and win rate for confirmed signals, split by `by`."""
    if outcomes.empty or by not in outcomes:
        return pd.DataFrame()
    d = outcomes[outcomes["confirmed"] == True]  # noqa: E712
    if d.empty:
        return pd.DataFrame()
    rows = []
    for (kind, key), g in d.groupby(["kind", by]):
        if len(g) < 10:
            continue
        row = {"kind": kind, by: key, "n": int(len(g))}
        for h in (4, 13, 26):
            c = f"exc_{h}w"
            if c in g and g[c].notna().any():
                row[f"median_excess_{h}w"] = round(float(g[c].median()), 2)
                row[f"win_rate_{h}w"] = round(float((g[c] > 0).mean() * 100), 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Uncertainty
#
# Nothing in this file used to compute a standard error, a confidence interval
# or an effective sample size, so every table was read as a set of point
# estimates carrying no stated uncertainty at all. That is the most misleading
# thing a backtest can do, because the row counts are large and the independent
# information in them is not.
#
# Two dependence structures destroy the naive count. Signals from the same
# ticker in consecutive weeks share most of their forward window, and signals
# across tickers in the same week share the market. The second is by far the
# larger: on a breakout week hundreds of names fire together and their forward
# returns are one draw, not hundreds. The hard ceiling is the number of
# non-overlapping calendar episodes in the sample, which over eight years at a
# 26 week horizon is sixteen. Not sixteen thousand.
# ----------------------------------------------------------------------------

def _block_bootstrap_median(values: np.ndarray, blocks: np.ndarray,
                            n_boot: int = 400, seed: int = 0) -> tuple:
    """
    Median with a confidence interval from resampling whole calendar blocks.

    Resampling weeks rather than rows keeps the cross-sectional correlation
    intact, which is exactly the dependence a naive interval ignores.
    """
    ok = np.isfinite(values)
    v, b = values[ok], blocks[ok]
    if len(v) < 20:
        return float("nan"), float("nan"), float("nan"), 0
    keys = np.unique(b)
    rng = np.random.default_rng(seed)
    idx = {k: np.flatnonzero(b == k) for k in keys}
    stats = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sample = np.concatenate([idx[k] for k in pick])
        stats[i] = np.median(v[sample])
    return (float(np.median(v)), float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)), int(len(keys)))


def uncertainty_table(outcomes: pd.DataFrame, horizon: int = 13) -> pd.DataFrame:
    """Median excess with a week-clustered 95% interval, per signal type."""
    col = f"exc_{horizon}w"
    if outcomes.empty or col not in outcomes:
        return pd.DataFrame()
    d = outcomes.dropna(subset=[col]).copy()
    if d.empty:
        return pd.DataFrame()
    d["_week"] = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="W").astype(str)
    rows = []
    for (kind, conf), g in d.groupby(["kind", "confirmed"]):
        med, lo, hi, nweeks = _block_bootstrap_median(
            g[col].to_numpy(dtype=float), g["_week"].to_numpy())
        if np.isnan(med):
            continue
        # Effective sample size, capped by the number of non-overlapping
        # episodes the horizon allows. Overlapping windows from the same weeks
        # are not additional evidence.
        episodes = max(1, int(nweeks / horizon))
        rows.append({
            "kind": kind, "confirmed": bool(conf),
            "rows": int(len(g)), "distinct_weeks": nweeks,
            "independent_episodes": episodes,
            f"median_excess_{horizon}w": round(med, 2),
            "ci_low": round(lo, 2), "ci_high": round(hi, 2),
            "ci_width": round(hi - lo, 2),
        })
    return pd.DataFrame(rows)


def comparison_budget(calib: dict) -> dict:
    """
    How many numbers the calibration reports, and what threshold a single one
    would need to clear to mean anything after accounting for the family.

    With roughly a hundred statistics on the page and no correction, the chance
    that at least one looks convincing when every filter is inert is above 99
    per cent. Stating the budget is the minimum honest treatment.
    """
    n = 0
    n += len(calib.get("signals") or []) * 9
    n += len(calib.get("by_regime") or []) * 5
    n += len(calib.get("by_group") or []) * 5
    n += len(calib.get("stage2_deciles") or [])
    n += len(calib.get("stage4_deciles") or [])
    if n == 0:
        return {}
    from math import sqrt
    alpha = 0.05 / n
    # Two-sided normal quantile without scipy.
    z = 0.0
    lo, hi = 0.0, 10.0
    for _ in range(80):
        z = (lo + hi) / 2
        # survival of the standard normal via erfc
        import math
        p = math.erfc(z / sqrt(2.0))
        if p > alpha:
            lo = z
        else:
            hi = z
    return {"statistics_reported": n,
            "expected_false_positives_at_5pct": round(n * 0.05, 1),
            "bonferroni_alpha": round(alpha, 6),
            "bonferroni_z": round(z, 2)}
