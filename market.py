"""
Market regime.

Weinstein's most repeated instruction is also the one a stock screener finds
easiest to ignore: take breakouts freely when the market itself is in Stage 2,
take almost none when it is in Stage 4. The same chart pattern has a different
expected outcome depending on what the tape around it is doing, and a scanner
that reports fifty perfect breakouts in the third week of a bear market is
reporting an artefact of the screen rather than an opportunity.

Regime here is built from two independent sources rather than one, because the
index alone is misleading. A cap weighted index can hold up on a handful of
megacaps while the median stock is already in Stage 4, which is precisely the
condition in which breakout signals fail most often. Breadth, the share of the
scanned universe trading above its own 30 week average, sees that directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F

# Regime score bands
BULL = 40.0
IMPROVING = 15.0
DETERIORATING = -15.0
BEAR = -40.0


def index_frame(close: pd.Series) -> pd.DataFrame:
    """
    The index put on exactly the same footing as an individual stock.

    The volatility denominator here used mean absolute weekly return, while
    features.py uses mean true range over the close. Those differ by roughly a
    factor of two, because a true range spans the whole high-to-low excursion
    and an absolute return only spans close to close. Both were then squashed
    with the same literal tanh scale of 0.60, so the regime term saturated more
    than twice as fast as the stock term for identical price behaviour, pushing
    every reading toward Bull or Bear and away from Neutral. An index series has
    no high or low here, so the absolute return is scaled to the true-range
    equivalent rather than the two being silently compared.
    """
    ma = close.rolling(F.MA_LENGTH, min_periods=F.MA_LENGTH).mean()
    df = pd.DataFrame({
        "close": close,
        "ma30": ma,
        "px_vs_ma_pct": (close / ma.where(ma > 0) - 1.0) * 100.0,
        "ma_slope_pct": (ma / ma.shift(F.MA_SLOPE_LOOKBACK) - 1.0) * 100.0,
    })
    ret = close.pct_change()
    # E|N(0,s)| = s*sqrt(2/pi); the expected weekly true range of a driftless
    # random walk is about 2.2 times that, which is the ratio measured directly
    # against features.atr_pct on the same series.
    ABS_RET_TO_TRUE_RANGE = 2.2
    atr_pct = (ret.abs().rolling(F.ATR_LENGTH, min_periods=F.ATR_LENGTH).mean()
               ) * 100.0 * ABS_RET_TO_TRUE_RANGE
    df["atr_pct"] = atr_pct
    # Same sqrt(5) time-base correction as features.ma_slope_norm, and the same
    # slow denominator, so the two are directly comparable.
    atr_vs_ma = atr_pct * (close / ma.where(ma > 0))
    df["ma_slope_norm"] = df["ma_slope_pct"] / (
        atr_vs_ma * np.sqrt(F.MA_SLOPE_LOOKBACK)).replace(0, np.nan)
    return df


def regime_series(index_close: pd.Series, breadth_above_ma: pd.Series) -> pd.DataFrame:
    """
    Regime score from minus one hundred to plus one hundred, weekly.

    Weights put breadth close to parity with the index trend on purpose. The
    index says what the average dollar is doing and breadth says what the average
    stock is doing, and when they disagree it is breadth that has historically
    described the environment a breakout is entering.
    """
    idx = index_frame(index_close)
    b = breadth_above_ma.reindex(idx.index).ffill()

    def sq(s, scale):
        return np.tanh(s.astype(float) / scale).fillna(0.0)

    score = (0.45 * sq(idx["ma_slope_norm"], 0.60)
             + 0.20 * sq(idx["px_vs_ma_pct"], 6.0)
             + 0.35 * sq((b - 50.0), 20.0)) * 100.0
    score = score.clip(-100.0, 100.0)

    out = idx.copy()
    out["breadth_above_ma"] = b
    out["regime_score"] = score
    out["regime"] = [label(v) for v in score]
    return out


def label(score: float) -> str:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "Unknown"
    if score >= BULL:
        return "Bull"
    if score >= IMPROVING:
        return "Improving"
    if score <= BEAR:
        return "Bear"
    if score <= DETERIORATING:
        return "Deteriorating"
    return "Neutral"


def market_ok(regime_label: str) -> bool:
    """Whether the environment supports taking long breakouts at all."""
    return regime_label in ("Bull", "Improving")


GUIDANCE = {
    "Bull": "Take valid breakouts freely. This is the environment the method was "
            "designed for and the one in which its base rates were formed.",
    "Improving": "Take grade A and grade B breakouts. The average stock has "
                 "crossed back above its own 30 week line but the trend is young.",
    "Neutral": "Grade A only, and expect more failed breakouts than usual. Roughly "
               "half the universe is below its 30 week average.",
    "Deteriorating": "Stop initiating longs. Work the Stage 4 lists instead and "
                     "tighten stops on what is already open.",
    "Bear": "No long breakouts. Weinstein's own instruction is that the great "
            "majority fail here regardless of how good the individual chart looks.",
    "Unknown": "Not enough history to judge the environment.",
}


def breadth_from_features(flags: list[pd.Series]) -> pd.Series:
    """
    Share of the universe above its own 30 week average, week by week.

    Built by summing per-ticker boolean series and dividing by the count that had
    data that week, so a stock that only listed three years ago does not drag the
    early history toward zero.
    """
    if not flags:
        return pd.Series(dtype=float)
    wide = pd.concat(flags, axis=1)
    above = wide.sum(axis=1, skipna=True)
    live = wide.notna().sum(axis=1)
    return (above / live.replace(0, np.nan)) * 100.0


def summary(regime: pd.DataFrame, scan: pd.DataFrame) -> dict:
    if regime.empty:
        return {}
    last = regime.dropna(subset=["regime_score"]).iloc[-1]
    counts = scan["stage"].value_counts().to_dict() if not scan.empty else {}
    n = max(len(scan), 1)
    return {
        "regime": str(last["regime"]),
        "regime_score": round(float(last["regime_score"]), 1),
        "guidance": GUIDANCE.get(str(last["regime"]), ""),
        "index_ma_slope_pct": round(float(last["ma_slope_pct"]), 2) if pd.notna(last["ma_slope_pct"]) else None,
        "index_px_vs_ma_pct": round(float(last["px_vs_ma_pct"]), 2) if pd.notna(last["px_vs_ma_pct"]) else None,
        "breadth_above_ma": round(float(last["breadth_above_ma"]), 1) if pd.notna(last["breadth_above_ma"]) else None,
        "stage2_share": round(100.0 * counts.get("Stage 2", 0) / n, 1),
        "stage4_share": round(100.0 * counts.get("Stage 4", 0) / n, 1),
        "market_ok": market_ok(str(last["regime"])),
    }
