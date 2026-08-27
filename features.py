"""
Weekly-bar feature engine for Stan Weinstein stage analysis.

Every feature here is computed on WEEKLY bars. Weinstein's method is explicitly a
weekly-chart method: the 30-week moving average, the Mansfield relative strength
line and the volume confirmation rule are all defined on weekly data. Running the
same logic on daily bars produces a different and much noisier system.

No network access and no third party dependencies beyond pandas and numpy, so this
module can be unit tested against synthetic series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Tunable constants. These are the knobs worth revisiting after calibration.
# ----------------------------------------------------------------------------

MA_LENGTH = 30           # Weinstein's 30-week simple moving average
MA_SLOPE_LOOKBACK = 5    # weeks used to measure the slope of the 30-week MA
RS_SMA_LENGTH = 52       # Mansfield normalisation window, 52 weeks
BASE_LOOKBACK = 30       # weeks of prior range used to define base support/resistance
TIGHT_WINDOW = 10        # recent window used to measure coiling inside the base
VOL_FAST = 4             # weeks, fast volume average
VOL_SLOW = 26            # weeks, slow volume average
VOL_BREAKOUT_WINDOW = 10 # weeks, baseline for breakout volume ratio
ATR_LENGTH = 14          # weeks, for volatility normalisation and stop sizing

MIN_HISTORY = MA_LENGTH + RS_SMA_LENGTH + 4  # weeks needed before output is trustworthy


def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    a = df["high"] - df["low"]
    b = (df["high"] - prev_close).abs()
    c = (df["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = ATR_LENGTH) -> pd.Series:
    """
    Wilder's ATR, an exponential average with alpha = 1/n.

    A simple rolling mean was used here originally and is more responsive, which
    sounds like an improvement until you notice what it does to a trailing stop.
    A rectangular window drops its oldest bar entirely, so a single crash week
    leaving the window moves the ATR down by two thirds in one step, and the
    trailing stop at ma30 minus half an ATR jumps two points closer to price with
    no price event at all. Wilder's decays smoothly and has no such discontinuity.
    """
    tr = true_range(df)
    seed = tr.rolling(n, min_periods=n).mean()
    out = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return out.where(seed.notna())


def mansfield_rs(close: pd.Series, benchmark_close: pd.Series,
                 n: int = RS_SMA_LENGTH) -> pd.Series:
    """
    Mansfield relative strength, the version Weinstein uses.

    Step one is the raw relative strength ratio, the stock divided by the market
    index. Step two divides that ratio by its own 52 week average and subtracts
    one, which re-centres the line on zero. A reading above zero means the stock
    has outperformed its own one year relative trend, which is the condition
    Weinstein requires before he will buy a breakout. A reading below zero on a
    breakout is his single most common reason to reject an otherwise valid setup.
    """
    bench = benchmark_close.reindex(close.index).ffill()
    # A non-positive benchmark is not a small number, it is a different sign.
    # Left unguarded, one zero bar makes the ratio infinite, that infinity sits
    # in the 52 week average for a full year, and the whole year returns NaN
    # while the caller reports "relative strength below zero, Weinstein would
    # pass" about data that does not exist. A negative bar flips the sign of the
    # entire measure and saturates every downstream score instead.
    bench = bench.where(bench > 0)
    covered = bench.notna().mean() if len(bench) else 0.0
    if covered < 0.5:
        # Almost always an index-alignment failure (weekly bars stamped on
        # different weekdays), which otherwise fails silently as an all-NaN
        # column that every consumer treats as "no opinion".
        raise ValueError(
            f"benchmark covers only {covered:.0%} of the stock's weeks; "
            "the two series are probably not on the same weekly grid")
    ratio = close / bench
    base = _sma(ratio, n)
    return (ratio / base.where(base > 0) - 1.0) * 100.0


def compute_features(df: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    """
    df must have a DatetimeIndex of weekly bars and columns
    open, high, low, close, volume. Returns the same index with feature columns.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)

    out["close"] = close
    out["high"] = high
    out["low"] = low
    out["volume"] = vol

    # --- Trend: the 30 week moving average and its slope -------------------
    ma = _sma(close, MA_LENGTH)
    out["ma30"] = ma
    out["px_vs_ma_pct"] = (close / ma - 1.0) * 100.0

    # Slope expressed as percent change of the average itself over 5 weeks.
    # Percent rather than absolute so a 400 dollar stock and a 4 dollar stock
    # are directly comparable.
    out["ma_slope_pct"] = (ma / ma.shift(MA_SLOPE_LOOKBACK) - 1.0) * 100.0

    # Second derivative. A negative slope that is rising is the earliest
    # mechanical sign of a stage 1 base forming under a stage 4 decline.
    #
    # Differenced over 2 weeks rather than 5. At a 5 week lag the statistic
    # expands to four disjoint 5 week blocks of price, two recent and two from
    # 30 to 39 weeks ago, at equal weight: exactly half its sensitivity came from
    # what happened seven months back, so "the average is turning up" could fire
    # with no recent price action at all, purely from a weak old block leaving
    # the window. A 2 week difference keeps the recent blocks and shrinks the
    # ancient ones to a 2 week sliver.
    out["ma_slope_delta"] = out["ma_slope_pct"] - out["ma_slope_pct"].shift(2)

    # Volatility normalised slope, so that a quiet utility and a volatile
    # biotech are held to comparable standards of "the average is rising".
    a = atr(df)
    out["atr"] = a
    atr_pct = (a / close.where(close > 0)) * 100.0
    out["atr_pct"] = atr_pct
    # Two corrections against the naive version.
    #
    # The numerator is a 5 week change and the denominator a 1 week range, so
    # dividing one by the other does not give a signal-to-noise ratio, it gives
    # a number inflated by roughly the square root of 5. Noise accumulates as
    # sqrt(time) for a diffusive series, so that is the right divisor.
    #
    # The denominator also used the current close, which moves against the
    # numerator: a sharp drop raises ATR and lowers the close at the same moment,
    # collapsing the measure exactly when a stage 2 trend is being tested, and
    # inflating it when price is most extended. Dividing ATR by the 30 week
    # average instead keeps the denominator on the same slow footing as the
    # numerator and stops this term double-counting extension, which
    # px_vs_ma_pct already carries.
    atr_vs_ma = (a / ma.where(ma > 0)) * 100.0
    out["ma_slope_norm"] = out["ma_slope_pct"] / (
        atr_vs_ma * np.sqrt(MA_SLOPE_LOOKBACK)).replace(0, np.nan)

    # --- Relative strength --------------------------------------------------
    rs = mansfield_rs(close, benchmark_close)
    out["mansfield_rs"] = rs
    out["rs_slope"] = rs - rs.shift(4)
    # Weeks since the RS line last crossed up through zero, capped for sanity.
    crossed_up = (rs > 0) & (rs.shift(1) <= 0)
    out["weeks_since_rs_cross_up"] = _weeks_since(crossed_up)
    crossed_dn = (rs < 0) & (rs.shift(1) >= 0)
    out["weeks_since_rs_cross_dn"] = _weeks_since(crossed_dn)

    # --- Base geometry ------------------------------------------------------
    # Resistance and support are taken from the PRIOR n weeks, excluding the
    # current bar, otherwise a breakout can never be detected because the
    # breakout bar itself defines the high.
    # min_periods was 12 against a 30 week window, so a range built from 12 bars
    # was published as, and compared against thresholds calibrated for, the 30
    # week level. Raised so the label matches the measurement. Rows with a gap in
    # the series now correctly return NaN instead of a short-window impostor.
    resistance = high.shift(1).rolling(BASE_LOOKBACK, min_periods=BASE_LOOKBACK).max()
    support = low.shift(1).rolling(BASE_LOOKBACK, min_periods=BASE_LOOKBACK).min()
    out["resistance"] = resistance
    out["support"] = support
    # Support must be positive for a ratio to mean anything. A negative support,
    # reachable through back-adjustment artefacts, produces a large NEGATIVE
    # width, which then passes a "narrower than 45 percent" test and scores the
    # maximum tightness reward. The most degenerate possible base would rank as
    # the best one. NaN is the correct answer instead.
    pos_support = support.where(support > 0)
    out["base_width_pct"] = (resistance / pos_support - 1.0) * 100.0
    out["dist_to_resistance_pct"] = (resistance / close.where(close > 0) - 1.0) * 100.0
    out["dist_to_support_pct"] = (close / pos_support - 1.0) * 100.0

    # Coiling: recent range as a fraction of the whole base range. Values well
    # under 1 mean the stock is contracting inside its base, which is what a
    # healthy stage 1 looks like just before it resolves upward.
    recent_hi = high.rolling(TIGHT_WINDOW, min_periods=TIGHT_WINDOW).max()
    recent_lo = low.rolling(TIGHT_WINDOW, min_periods=TIGHT_WINDOW).min()
    # The 10 week window is not a subset of the 30 week base window, so this
    # ratio is NOT bounded by 1: a breakout week lifts recent_hi above the base
    # high and drives it well past 1. Capped at 3 so a legitimate reading stays
    # legible while a runaway stops polluting the dashboard. The guard is a
    # relative epsilon rather than an exact-zero test, because a base range of
    # 1e-9 divides just as badly as one of exactly zero.
    base_range = (resistance - support)
    base_range = base_range.where(base_range > 1e-6 * close.abs())
    out["tightness"] = ((recent_hi - recent_lo) / base_range).clip(upper=3.0)
    # The nearest significant swing levels. A stop belongs under the most recent
    # meaningful low, not under the floor of a base that may be two years old and
    # thirty percent below the pivot.
    out["recent_low_10w"] = recent_lo
    out["recent_high_10w"] = recent_hi

    # How long price has been inside the base band, a proxy for base maturity.
    out["base_age_weeks"] = _base_age(close, resistance, support)
    # Base maturity as it stood just BEFORE the current bar, taken as the best
    # of the last four weeks. A breakout week resets the counter to zero by
    # definition, so testing the current value would reject every breakout, and
    # testing only last week would reject any breakout that took two or three
    # weeks to clear the pivot. This is what separates a genuine break out of a
    # base from a stock that has simply been making new highs for a year.
    out["base_age_recent"] = out["base_age_weeks"].shift(1).rolling(4, min_periods=1).max()

    # Trend leading INTO the current flat stretch. Measuring the last 52 weeks
    # is useless once a stock has been basing for a year, because the window sits
    # entirely inside the base. This looks back past the base to ask what the
    # stock was doing before it went quiet, which is the only thing that
    # separates a stage 1 floor from a stage 3 ceiling.
    # Anchored on base_age_recent rather than base_age_weeks. The plain counter
    # resets to zero on the breakout week by definition, which would collapse the
    # look-back to a bare 26 week return on exactly the week the answer matters
    # most, and report a long base after a deep decline as flat.
    out["prior_trend_pct"] = _prior_trend(close, out["base_age_recent"], window=26)

    # --- Volume -------------------------------------------------------------
    vol_base = _sma(vol, VOL_BREAKOUT_WINDOW).shift(1)
    out["vol_ratio"] = vol / vol_base.replace(0, np.nan)
    out["vol_dryup"] = _sma(vol, VOL_FAST) / _sma(vol, VOL_SLOW).replace(0, np.nan)
    # Best volume ratio over the breakout week and the two before it. Price
    # frequently clears the pivot a week or two after the accumulation surge,
    # and testing only the closing week throws away otherwise valid breakouts.
    out["vol_ratio_3w"] = out["vol_ratio"].rolling(3, min_periods=1).max()

    # Distribution pressure: share of the last 10 weeks' volume that occurred on
    # down weeks. Above roughly 0.6 the base is being sold into, which is the
    # volume signature under a stage 3 top.
    ret = close.pct_change()
    down_vol = vol.where(ret < 0, 0.0)
    out["down_vol_share"] = (down_vol.rolling(10, min_periods=10).sum()
                             / vol.rolling(10, min_periods=10).sum().replace(0, np.nan))

    # --- Context ------------------------------------------------------------
    out["ret_13w_pct"] = (close / close.shift(13) - 1.0) * 100.0
    out["ret_26w_pct"] = (close / close.shift(26) - 1.0) * 100.0
    out["ret_52w_pct"] = (close / close.shift(52) - 1.0) * 100.0
    out["pct_off_52w_high"] = (close / high.rolling(52, min_periods=52).max() - 1.0) * 100.0

    # Weinstein's initial stop sits under BOTH the base support and the 30 week
    # average, so this takes the lower of the two, which is the further stop
    # rather than the nearer one. np.fmin rather than np.minimum because
    # np.minimum propagates NaN: a missing support would otherwise blank the
    # stop even where the 30 week average is perfectly well defined, and the
    # trade then silently drops out of the stop statistics.
    out["stop_suggestion"] = np.fmin(
        support.to_numpy(dtype=float),
        (ma * 0.97).to_numpy(dtype=float),
    )
    out["stop_risk_pct"] = (1.0 - out["stop_suggestion"] / close) * 100.0

    return out


def _weeks_since(flags: pd.Series, cap: int = 999) -> pd.Series:
    """Number of bars since the last True in flags, NaN if it has never been True."""
    idx = np.arange(len(flags))
    last = np.where(flags.fillna(False).to_numpy(), idx, np.nan)
    last = pd.Series(last, index=flags.index).ffill()
    res = pd.Series(idx, index=flags.index) - last
    return res.clip(upper=cap)


def _base_age(close: pd.Series, resistance: pd.Series, support: pd.Series) -> pd.Series:
    """
    Consecutive weeks the close has stayed inside the prevailing support to
    resistance band. A long count means a mature base, which Weinstein treats as
    a stronger platform than a two or three week pause.
    """
    inside = (close <= resistance) & (close >= support)
    inside = inside.fillna(False)
    ages, run = [], 0
    for flag in inside.to_numpy():
        run = run + 1 if flag else 0
        ages.append(run)
    return pd.Series(ages, index=close.index, dtype=float)


def _prior_trend(close: pd.Series, base_age: pd.Series, window: int = 26) -> pd.Series:
    arr = close.to_numpy(dtype=float)
    age = base_age.fillna(0).to_numpy(dtype=int)
    n = len(arr)
    res = np.full(n, np.nan)
    for i in range(n):
        anchor = i - int(age[i])          # bar at which the base began
        start = anchor - window
        if start < 0 or anchor < 0 or arr[start] <= 0:
            # fall back to a plain 26 week return when there is not enough history
            if i - window >= 0 and arr[i - window] > 0:
                res[i] = (arr[i] / arr[i - window] - 1.0) * 100.0
            continue
        res[i] = (arr[anchor] / arr[start] - 1.0) * 100.0
    return pd.Series(res, index=close.index)
