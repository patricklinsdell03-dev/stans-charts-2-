"""
Stage classification and readiness scoring.

Two layers sit on top of the feature engine.

The first is a continuum. Every ticker gets a trend score from minus one hundred
to plus one hundred built from four pieces of evidence that Weinstein weighs by
eye: where price sits relative to the 30 week average, whether that average is
rising or falling, whether the Mansfield relative strength line is above or below
zero, and whether that line is improving or deteriorating. The continuum matters
because the four stages are not four boxes, they are a cycle, and the interesting
names are always the ones part way between two of them.

The second layer is readiness. A confirmed breakout is a fact you can screen for
with a single comparison. What is worth more is the list of stocks that have done
everything except break out, because those are the ones you can still buy near
the pivot rather than ten percent above it. Readiness scores that condition.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import features as F

# ----------------------------------------------------------------------------
# Thresholds. Textbook values first, relaxed values used only for flagging.
# ----------------------------------------------------------------------------

MA_FLAT_BAND = 0.75      # percent 5-week MA change inside which the MA counts as flat
VOL_TEXTBOOK = 2.0       # Weinstein's breakout volume rule, at least double
VOL_RELAXED = 1.4        # still notable, flagged separately as unconfirmed volume
MAX_EXTENSION = 12.0     # percent above the 30 week MA beyond which entry is chasing
MIN_BASE_AGE = 8         # weeks inside the band before a base is taken seriously
MAX_BASE_WIDTH = 45.0    # percent, wider than this is not a base, it is a downtrend


def _inv(x, lo, hi):
    """Inverted saturation that scores zero, not full marks, on a missing input."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    return 1.0 - _sat(x, lo, hi)


def _grp(v):
    """A missing sector label means no opinion about the group, never a penalty."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if np.isnan(v) else v


def _sat(x, lo, hi):
    """Linear saturation of x onto 0..1 between lo and hi. Handles NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if hi == lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))



# ----------------------------------------------------------------------------
# Stage context factor
#
# A range near its highs means opposite things depending on what preceded it.
# After a decline it is a stage 1 base and the break is the entry Weinstein
# waits for. After an advance, with the 30 week average gone flat, the identical
# picture is a stage 3 top and the break is a bull trap. The raw readiness
# components cannot see the difference because they only measure the range, so
# the context factor scales the score by what came before.
# ----------------------------------------------------------------------------

TOP_DISCOUNT = 0.45      # range after an advance with a flat average
FREEFALL_DISCOUNT = 0.65 # still declining hard, no base has formed yet
CONTINUATION = 0.85      # already in stage 2, a valid but later entry


def _ctx_up(prior, slope, rs):
    after_advance = (prior is not None) and (not np.isnan(prior)) and prior > 10.0
    rising = (not np.isnan(slope)) and slope > MA_FLAT_BAND
    if after_advance and not rising:
        return TOP_DISCOUNT
    if rising and after_advance:
        return CONTINUATION
    if (not np.isnan(slope)) and slope < -2.0 and (not np.isnan(rs)) and rs < -5.0:
        return FREEFALL_DISCOUNT
    return 1.0


def _ctx_dn(prior, slope, rs):
    after_decline = (prior is not None) and (not np.isnan(prior)) and prior < -10.0
    falling = (not np.isnan(slope)) and slope < -MA_FLAT_BAND
    if after_decline and not falling:
        return TOP_DISCOUNT
    if falling and after_decline:
        return CONTINUATION
    if (not np.isnan(slope)) and slope > 2.0 and (not np.isnan(rs)) and rs > 5.0:
        return FREEFALL_DISCOUNT
    return 1.0


# ----------------------------------------------------------------------------
# Continuum trend score
# ----------------------------------------------------------------------------

def trend_score(row: pd.Series) -> float:
    """
    Minus one hundred to plus one hundred. Positive is stage 2 territory,
    negative is stage 4 territory, near zero is stage 1 or stage 3 and needs the
    prior trend to tell those two apart.

    Weights reflect the order Weinstein applies his own filters. The direction of
    the 30 week average is the primary trend statement, relative strength is the
    veto, and position relative to the average is confirmation rather than cause.
    """
    slope = row.get("ma_slope_norm", np.nan)      # ATR normalised
    pos = row.get("px_vs_ma_pct", np.nan)
    rs = row.get("mansfield_rs", np.nan)
    rs_slope = row.get("rs_slope", np.nan)

    def squash(v, scale):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return float(np.tanh(v / scale))

    # Weights are renormalised over the terms that actually have data. Treating a
    # missing input as 0.0 does not make it neutral: it silently shrinks the
    # score toward the middle, and with the relative strength pair absent, which
    # is the ordinary case for a recent listing, it demoted 28 percent of true
    # Stage 2 rows and 60 percent of true Stage 4 rows below their thresholds.
    terms = [(0.35, slope, 0.60), (0.20, pos, 8.0),
             (0.30, rs, 6.0), (0.15, rs_slope, 5.0)]
    total = 0.0
    weight = 0.0
    for w, v, scale in terms:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        total += w * squash(v, scale)
        weight += w
    if weight < 0.5:
        # Under half the evidence present. No opinion is honest; zero is not,
        # because zero sits at the 44th percentile of a real universe and would
        # rank a data-free ticker above the entire negative half of the board.
        return float("nan")
    return float(np.clip(total / weight * 100.0, -100.0, 100.0))


def classify_stage(row: pd.Series) -> str:
    """
    Returns one of: Stage 1, Stage 1 to 2, Stage 2, Stage 2 to 3, Stage 3,
    Stage 3 to 4, Stage 4, Stage 4 to 1.

    The logic is deliberately ordered the way Weinstein reads a chart rather than
    the way a scoring model would. He looks first at whether the 30 week average
    is flat, because a flat average is the definition of a transition, and only
    then asks what came before it. A flat average after a decline is a stage 1
    floor. The identical flat average after an advance is a stage 3 ceiling. The
    two look the same in isolation and behave in opposite ways, so prior trend is
    load bearing, and it is measured from before the flat stretch began rather
    than over a trailing year that may sit entirely inside it.

    Only when the average is clearly sloping does the continuum score take over.
    """
    ts = row.get("trend_score", np.nan)
    slope = row.get("ma_slope_pct", np.nan)
    pxma = row.get("px_vs_ma_pct", np.nan)
    prior = row.get("prior_trend_pct", np.nan)
    if np.isnan(prior):
        prior = row.get("ret_52w_pct", np.nan)
    rs_slope = row.get("rs_slope", np.nan)

    if np.isnan(ts) or np.isnan(slope):
        return "Insufficient history"

    flat = abs(slope) <= MA_FLAT_BAND
    near = (not np.isnan(pxma)) and abs(pxma) <= 10.0

    if flat and near:
        after_advance = (not np.isnan(prior)) and prior > 10.0
        after_decline = (not np.isnan(prior)) and prior < -10.0
        if after_advance:
            rolling_over = ((not np.isnan(rs_slope)) and rs_slope < -1.5) or slope < -0.2
            return "Stage 3 to 4" if (rolling_over and ts < 0) else "Stage 3"
        if after_decline:
            turning_up = ((not np.isnan(rs_slope)) and rs_slope > 1.5) and slope >= -0.3
            return "Stage 1 to 2" if turning_up else "Stage 1"
        # No decisive prior trend. Lean on the continuum, but stay in the
        # transitional stages rather than claiming a trend that is not there.
        # A flat average with a strongly positive continuum is a base in the act
        # of resolving, which is worth saying out loud rather than filing under
        # a plain Stage 1 that reads as "nothing happening here".
        if ts >= 30:
            return "Stage 1 to 2"
        if ts <= -30:
            return "Stage 3 to 4"
        # A bare sign test on ts sat between the most bullish and most bearish
        # labels in the set, so a two thousandth of a point of relative strength
        # flipped Stage 1 to Stage 3. With no decisive prior trend and a score
        # near zero the honest answer is that the stage is undetermined.
        if ts > 5:
            return "Stage 1"
        if ts < -5:
            return "Stage 3"
        return "Stage 1 or 3, undetermined"

    if ts >= 45:
        # The stronger score had no structural test at all while the weaker one
        # below did, so a falling 30 week average could be published as Stage 2,
        # which contradicts Weinstein's definition outright.
        return "Stage 2" if slope > -MA_FLAT_BAND else "Stage 2 to 3"
    if ts >= 15:
        return "Stage 1 to 2" if slope < 1.0 else "Stage 2"
    if ts <= -45:
        return "Stage 4"
    if ts <= -15:
        # Was `prior > 0` while every other prior test in this function uses
        # +/-10. Five different thresholds on one input is five different
        # definitions of "what came before".
        return "Stage 3 to 4" if (not np.isnan(prior) and prior > 10) else "Stage 4"

    # Mildly scored but the average is still sloping. A bare `slope > 0` test
    # here flipped a buy label to a sell label on 1e-12 of slope, so this uses a
    # dead band a tenth the width of MA_FLAT_BAND rather than a sign test.
    eps = MA_FLAT_BAND * 0.1
    if slope > eps:
        return "Stage 2 to 3" if (not np.isnan(prior) and prior > 10) else "Stage 1 to 2"
    if slope < -eps:
        return "Stage 4 to 1" if (not np.isnan(prior) and prior < -10) else "Stage 3 to 4"
    return "Stage 1 or 3, undetermined"


# ----------------------------------------------------------------------------
# Confirmed breaks
# ----------------------------------------------------------------------------

@dataclass
class BreakSignal:
    kind: str            # "stage2_breakout" or "stage4_breakdown"
    confirmed: bool      # all textbook conditions met
    volume_ok: bool
    rs_ok: bool
    ma_ok: bool
    level: float         # the base boundary that was crossed
    notes: str


def detect_break(row: pd.Series) -> BreakSignal | None:
    close = row.get("close", np.nan)
    res = row.get("resistance", np.nan)
    sup = row.get("support", np.nan)
    ma = row.get("ma30", np.nan)
    slope = row.get("ma_slope_pct", np.nan)
    rs = row.get("mansfield_rs", np.nan)
    volr = row.get("vol_ratio", np.nan)
    volr3 = row.get("vol_ratio_3w", volr)
    width = row.get("base_width_pct", np.nan)
    age = row.get("base_age_recent", row.get("base_age_weeks", np.nan))

    if any(np.isnan(v) for v in (close, ma, slope)):
        return None

    # --- Stage 2 breakout -------------------------------------------------
    if not np.isnan(res) and close > res:
        ma_ok = close > ma and slope >= -0.25       # average flat or turning up
        rs_ok = (not np.isnan(rs)) and rs > 0
        vtest = volr3 if not np.isnan(volr3) else volr
        vol_ok = (not np.isnan(vtest)) and vtest >= VOL_TEXTBOOK
        vol_soft = (not np.isnan(vtest)) and vtest >= VOL_RELAXED
        base_ok = (np.isnan(width) or width <= MAX_BASE_WIDTH)
        age_ok = (not np.isnan(age)) and age >= MIN_BASE_AGE
        notes = []
        if not rs_ok:
            notes.append("relative strength still below zero, Weinstein would pass")
        if not vol_ok:
            notes.append(
                f"peak volume {vtest:.1f}x the ten week average, textbook wants 2.0x"
                if not np.isnan(vtest) else "volume unavailable")
        if not base_ok:
            notes.append("prior range too wide to count as a base")
        if not age_ok:
            # Already trending, not breaking out of anything. Weinstein's entry
            # is at the pivot of a base; a stock making routine new highs inside
            # an established stage 2 is a different, later, worse entry.
            return None
        confirmed = ma_ok and rs_ok and vol_ok and base_ok
        if ma_ok and (vol_soft or rs_ok):
            return BreakSignal("stage2_breakout", confirmed, vol_ok, rs_ok, ma_ok,
                               float(res), "; ".join(notes) or "all textbook conditions met")

    # --- Stage 4 breakdown ------------------------------------------------
    # Weinstein does not require heavy volume on breakdowns. Selling can happen
    # on an absence of bids, so a quiet break of support is still a break.
    if not np.isnan(sup) and close < sup:
        ma_ok = close < ma and slope <= 0.25
        rs_ok = (not np.isnan(rs)) and rs < 0
        notes = []
        if not rs_ok:
            notes.append("relative strength still positive, may be a shakeout")
        if not ma_ok:
            notes.append("30 week average not yet rolling over")
        if not ((not np.isnan(age)) and age >= MIN_BASE_AGE):
            return None
        confirmed = ma_ok and rs_ok
        if ma_ok or rs_ok:
            return BreakSignal("stage4_breakdown", confirmed, True, rs_ok, ma_ok,
                               float(sup), "; ".join(notes) or "all textbook conditions met")

    return None


# ----------------------------------------------------------------------------
# Readiness: the forward looking part
# ----------------------------------------------------------------------------

def stage2_readiness(row: pd.Series) -> tuple[float, dict]:
    """
    Zero to one hundred. How close is this to a valid stage 2 breakout that has
    not happened yet.

    The components are deliberately separable so the dashboard can show why a
    name scored what it did, and so any single component can be reweighted after
    the backtest tells you which ones actually carry information.
    """
    c = {}

    # Proximity to the pivot. Best zone is just under resistance. Already above
    # it means the break has happened and belongs in the confirmed list instead.
    d = row.get("dist_to_resistance_pct", np.nan)
    if np.isnan(d):
        c["proximity"] = 0.0
    elif d < 0:
        c["proximity"] = 4.0            # already through, keep a residual
    else:
        c["proximity"] = 25.0 * (1.0 - _sat(d, 0.0, 12.0))

    # The 30 week average flattening then turning up under the base.
    slope = row.get("ma_slope_pct", np.nan)
    delta = row.get("ma_slope_delta", np.nan)
    turn = 0.0
    if not np.isnan(slope):
        turn += 12.0 * _sat(slope, -2.0, 1.0)     # rewards flat to rising
    if not np.isnan(delta):
        turn += 8.0 * _sat(delta, -0.5, 2.0)      # rewards the slope improving
    c["ma_turn"] = min(turn, 20.0)

    # Relative strength. The single most predictive input in Weinstein's own
    # description: the RS line commonly turns up weeks before the price pivot.
    rs = row.get("mansfield_rs", np.nan)
    rs_slope = row.get("rs_slope", np.nan)
    r = 0.0
    if not np.isnan(rs):
        r += 15.0 * _sat(rs, -8.0, 3.0)
    if not np.isnan(rs_slope):
        r += 10.0 * _sat(rs_slope, -1.0, 6.0)
    c["rel_strength"] = min(r, 25.0)

    # Base quality. Long, narrow and coiling beats short, wide and erratic.
    age = row.get("base_age_weeks", np.nan)
    width = row.get("base_width_pct", np.nan)
    tight = row.get("tightness", np.nan)
    b = 0.0
    b += 8.0 * _sat(age, 4.0, 26.0)
    # _inv rather than 1 - _sat. Phrased as "one minus", a missing input scores
    # as a PERFECT reading, so a ticker with too little history to have a base
    # width at all was collecting the full narrow-base reward. Twelve free points
    # on a hundred point scale, awarded for a base the model cannot see, and
    # awarded on exactly the early rows that feed the calibration sample.
    b += 6.0 * _inv(width, 12.0, MAX_BASE_WIDTH)
    b += 6.0 * _inv(tight, 0.35, 0.95)
    c["base_quality"] = min(b, 20.0)

    # Volume drying up through the base is the classic accumulation signature.
    dry = row.get("vol_dryup", np.nan)
    c["volume_dryup"] = 10.0 * (1.0 - _sat(dry, 0.75, 1.25)) if not np.isnan(dry) else 0.0

    total = sum(c.values())

    # Penalty for chasing. If price is already far above the average, the entry
    # Weinstein wanted is gone whatever the other components say.
    ext = row.get("px_vs_ma_pct", np.nan)
    if not np.isnan(ext) and ext > MAX_EXTENSION:
        pen = 30.0 * _sat(ext, MAX_EXTENSION, MAX_EXTENSION + 20.0)
        c["extension_penalty"] = -pen
        total -= pen

    ctx = _ctx_up(row.get("prior_trend_pct", np.nan), row.get("ma_slope_pct", np.nan),
                  row.get("mansfield_rs", np.nan))
    grp = _grp(row.get("group_factor", 1.0))
    c["context_factor"] = ctx
    c["group_factor"] = grp
    return float(np.clip(total * ctx * grp, 0.0, 100.0)), c


def stage4_readiness(row: pd.Series) -> tuple[float, dict]:
    """
    Zero to one hundred. How close is this to breaking down out of a stage 3 top.
    Mirror of the above with the volume logic replaced: what matters on the way
    down is not a volume surge but the share of recent volume transacting on
    down weeks, which is the mechanical footprint of distribution.
    """
    c = {}

    d = row.get("dist_to_support_pct", np.nan)
    if np.isnan(d):
        c["proximity"] = 0.0
    elif d < 0:
        c["proximity"] = 4.0
    else:
        c["proximity"] = 25.0 * (1.0 - _sat(d, 0.0, 12.0))

    slope = row.get("ma_slope_pct", np.nan)
    delta = row.get("ma_slope_delta", np.nan)
    turn = 0.0
    if not np.isnan(slope):
        turn += 12.0 * (1.0 - _sat(slope, -1.0, 2.0))
    if not np.isnan(delta):
        turn += 8.0 * (1.0 - _sat(delta, -2.0, 0.5))
    c["ma_rollover"] = min(turn, 20.0)

    rs = row.get("mansfield_rs", np.nan)
    rs_slope = row.get("rs_slope", np.nan)
    r = 0.0
    if not np.isnan(rs):
        r += 15.0 * (1.0 - _sat(rs, -3.0, 8.0))
    if not np.isnan(rs_slope):
        r += 10.0 * (1.0 - _sat(rs_slope, -6.0, 1.0))
    c["rel_weakness"] = min(r, 25.0)

    # Top structure: up over the year but stalled recently is the definition of
    # a stage 3 ceiling.
    y = row.get("ret_52w_pct", np.nan)
    q = row.get("ret_13w_pct", np.nan)
    t = 0.0
    if not np.isnan(y):
        t += 10.0 * _sat(y, 0.0, 40.0)
    if not np.isnan(q):
        t += 10.0 * (1.0 - _sat(q, -8.0, 8.0))
    c["top_structure"] = min(t, 20.0)

    dv = row.get("down_vol_share", np.nan)
    c["distribution"] = 10.0 * _sat(dv, 0.45, 0.70) if not np.isnan(dv) else 0.0

    total = sum(c.values())
    ctx = _ctx_dn(row.get("prior_trend_pct", np.nan), row.get("ma_slope_pct", np.nan),
                  row.get("mansfield_rs", np.nan))
    grp = _grp(row.get("group_factor_dn", 1.0))
    c["context_factor"] = ctx
    c["group_factor"] = grp
    return float(np.clip(total * ctx * grp, 0.0, 100.0)), c


# ----------------------------------------------------------------------------
# Per ticker summary
# ----------------------------------------------------------------------------

def grade_signal(kind: str, confirmed: bool, group_factor: float,
                 market_ok: bool) -> str:
    """
    A to D, combining the three things Weinstein requires and most screens
    report separately: the chart, the group, and the tape.

    A is a textbook break by a leader in a leading group while the market
    supports it. C is a textbook break with the group or the market against it,
    which is the case he explicitly tells you to pass on. D failed a price,
    volume or relative strength rule outright.
    """
    from .sectors import GROUP_LEADER
    if not confirmed:
        return "D"
    env_ok = market_ok if kind == "stage2_breakout" else (not market_ok)
    g = 1.0 if group_factor is None or np.isnan(group_factor) else float(group_factor)
    if g >= GROUP_LEADER and env_ok:
        return "A"
    if g >= 1.0 and env_ok:
        return "B"
    return "C"


def evaluate(feat: pd.DataFrame, ticker: str, name: str = "",
             market: str = "US", sector: str = "", market_ok: bool = True) -> dict | None:
    """Reduce a full feature frame to a single current-week verdict."""
    feat = feat.dropna(subset=["ma30"])
    if len(feat) < 4:
        return None

    row = feat.iloc[-1].copy()
    row["trend_score"] = trend_score(row)
    stage = classify_stage(row)

    sig = detect_break(row)
    # Also look back four weeks: a break two weeks ago is still actionable and
    # would otherwise vanish from a strictly current-week screen.
    recent_break, recent_age = None, None
    for k in range(1, 5):
        if len(feat) <= k:
            break
        r = feat.iloc[-1 - k].copy()
        r["trend_score"] = trend_score(r)
        s = detect_break(r)
        if s is not None and (recent_break is None):
            recent_break, recent_age = s, k

    up_score, up_parts = stage2_readiness(row)
    dn_score, dn_parts = stage4_readiness(row)

    rec = {
        "ticker": ticker,
        "name": name,
        "market": market,
        "sector": sector,
        "date": feat.index[-1],
        "close": float(row["close"]),
        "stage": stage,
        "trend_score": round(float(row["trend_score"]), 1),
        "stage2_readiness": round(up_score, 1),
        "stage4_readiness": round(dn_score, 1),
        "ma30": _f(row.get("ma30")),
        "px_vs_ma_pct": _f(row.get("px_vs_ma_pct")),
        "ma_slope_pct": _f(row.get("ma_slope_pct")),
        "mansfield_rs": _f(row.get("mansfield_rs")),
        "rs_slope": _f(row.get("rs_slope")),
        "resistance": _f(row.get("resistance")),
        "support": _f(row.get("support")),
        "dist_to_resistance_pct": _f(row.get("dist_to_resistance_pct")),
        "dist_to_support_pct": _f(row.get("dist_to_support_pct")),
        "base_age_weeks": _f(row.get("base_age_weeks")),
        "base_age_recent": _f(row.get("base_age_recent")),
        "base_width_pct": _f(row.get("base_width_pct")),
        "tightness": _f(row.get("tightness")),
        "vol_ratio": _f(row.get("vol_ratio")),
        "vol_ratio_3w": _f(row.get("vol_ratio_3w")),
        "vol_dryup": _f(row.get("vol_dryup")),
        "down_vol_share": _f(row.get("down_vol_share")),
        "atr_pct": _f(row.get("atr_pct")),
        "stop_suggestion": _f(row.get("stop_suggestion")),
        "stop_risk_pct": _f(row.get("stop_risk_pct")),
        "ret_13w_pct": _f(row.get("ret_13w_pct")),
        "ret_52w_pct": _f(row.get("ret_52w_pct")),
        "prior_trend_pct": _f(row.get("prior_trend_pct")),
        "sector_rs": _f(row.get("sector_rs")),
        "sector_rs_slope": _f(row.get("sector_rs_slope")),
        "sector_rank_pct": _f(row.get("sector_rank_pct")),
        "rs_vs_sector": _f(row.get("rs_vs_sector")),
        "group_factor": _f(row.get("group_factor")),
        "group_factor_dn": _f(row.get("group_factor_dn")),
        "up_parts": {k: round(v, 1) for k, v in up_parts.items()},
        "dn_parts": {k: round(v, 1) for k, v in dn_parts.items()},
        "signal": None,
        "signal_age_weeks": None,
    }

    chosen, age = (sig, 0) if sig is not None else (recent_break, recent_age)
    if chosen is not None:
        d = asdict(chosen)
        gf = row.get("group_factor" if chosen.kind == "stage2_breakout"
                     else "group_factor_dn", 1.0)
        d["grade"] = grade_signal(chosen.kind, bool(chosen.confirmed),
                                  _grp(gf), market_ok)
        d["market_ok"] = bool(market_ok)
        d["group_factor"] = _grp(gf)
        rec["signal"] = d
        rec["signal_age_weeks"] = age
        rec["grade"] = d["grade"]
    else:
        rec["grade"] = None
    return rec


def _f(v):
    try:
        v = float(v)
        return None if np.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------
# Vectorised twins
#
# The row-wise functions above are the readable definition. These produce the
# same numbers across a whole history at once, which is what makes a backtest
# over a thousand tickers finish in seconds rather than minutes. The test suite
# asserts the two agree, so the readable version stays the specification and this
# one stays an optimisation rather than a second, silently diverging model.
# ----------------------------------------------------------------------------

def _satv(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((x - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def _invv(x: pd.Series, lo: float, hi: float) -> pd.Series:
    """Inverted saturation that contributes nothing when the input is missing.

    Without the mask a NaN would score as a perfect reading on any term phrased
    as "one minus", which silently awards full marks to tickers that simply do
    not have enough history yet.
    """
    return (1.0 - _satv(x, lo, hi)).where(x.notna(), 0.0)


def trend_score_series(feat: pd.DataFrame) -> pd.Series:
    spec = [(0.35, "ma_slope_norm", 0.60), (0.20, "px_vs_ma_pct", 8.0),
            (0.30, "mansfield_rs", 6.0), (0.15, "rs_slope", 5.0)]
    total = pd.Series(0.0, index=feat.index)
    weight = pd.Series(0.0, index=feat.index)
    for w, col, scale in spec:
        v = feat[col].astype(float)
        present = v.notna()
        total = total + (w * np.tanh(v / scale)).where(present, 0.0)
        weight = weight + w * present.astype(float)
    out = (total / weight.where(weight > 0) * 100.0).clip(-100.0, 100.0)
    return out.where(weight >= 0.5)


def readiness_series(feat: pd.DataFrame) -> pd.DataFrame:
    d = feat["dist_to_resistance_pct"]
    prox = np.where(d < 0, 4.0, 25.0 * (1.0 - _satv(d, 0.0, 12.0)))
    prox = pd.Series(prox, index=feat.index).where(d.notna(), 0.0)

    turn = (12.0 * _satv(feat["ma_slope_pct"], -2.0, 1.0)
            + 8.0 * _satv(feat["ma_slope_delta"], -0.5, 2.0)).clip(upper=20.0)
    rel = (15.0 * _satv(feat["mansfield_rs"], -8.0, 3.0)
           + 10.0 * _satv(feat["rs_slope"], -1.0, 6.0)).clip(upper=25.0)
    base = (8.0 * _satv(feat["base_age_weeks"], 4.0, 26.0)
            + 6.0 * _invv(feat["base_width_pct"], 12.0, MAX_BASE_WIDTH)
            + 6.0 * _invv(feat["tightness"], 0.35, 0.95)).clip(upper=20.0)
    dry = (10.0 * (1.0 - _satv(feat["vol_dryup"], 0.75, 1.25))).where(
        feat["vol_dryup"].notna(), 0.0)
    up = prox + turn + rel + base + dry
    ext = feat["px_vs_ma_pct"]
    pen = (30.0 * _satv(ext, MAX_EXTENSION, MAX_EXTENSION + 20.0)).where(
        ext > MAX_EXTENSION, 0.0)
    up = up - pen
    prior, slope, rs = feat["prior_trend_pct"], feat["ma_slope_pct"], feat["mansfield_rs"]
    after_adv = (prior > 10.0).fillna(False)
    rising = (slope > MA_FLAT_BAND).fillna(False)
    freefall = ((slope < -2.0) & (rs < -5.0)).fillna(False)
    ctx_up = pd.Series(1.0, index=feat.index)
    ctx_up[freefall] = FREEFALL_DISCOUNT
    ctx_up[after_adv & rising] = CONTINUATION
    ctx_up[after_adv & ~rising] = TOP_DISCOUNT
    grp_up = (feat["group_factor"].astype(float).fillna(1.0)
              if "group_factor" in feat else pd.Series(1.0, index=feat.index))
    up = (up * ctx_up * grp_up).clip(0.0, 100.0)

    ds = feat["dist_to_support_pct"]
    prox4 = np.where(ds < 0, 4.0, 25.0 * (1.0 - _satv(ds, 0.0, 12.0)))
    prox4 = pd.Series(prox4, index=feat.index).where(ds.notna(), 0.0)
    roll = (12.0 * _invv(feat["ma_slope_pct"], -1.0, 2.0)
            + 8.0 * _invv(feat["ma_slope_delta"], -2.0, 0.5)).clip(upper=20.0)
    weak = (15.0 * _invv(feat["mansfield_rs"], -3.0, 8.0)
            + 10.0 * _invv(feat["rs_slope"], -6.0, 1.0)).clip(upper=25.0)
    top = (10.0 * _satv(feat["ret_52w_pct"], 0.0, 40.0)
           + 10.0 * _invv(feat["ret_13w_pct"], -8.0, 8.0)).clip(upper=20.0)
    dist = (10.0 * _satv(feat["down_vol_share"], 0.45, 0.70)).where(
        feat["down_vol_share"].notna(), 0.0)
    dn = (prox4 + roll + weak + top + dist)

    after_dec = (prior < -10.0).fillna(False)
    falling = (slope < -MA_FLAT_BAND).fillna(False)
    rip = ((slope > 2.0) & (rs > 5.0)).fillna(False)
    ctx_dn = pd.Series(1.0, index=feat.index)
    ctx_dn[rip] = FREEFALL_DISCOUNT
    ctx_dn[after_dec & falling] = CONTINUATION
    ctx_dn[after_dec & ~falling] = TOP_DISCOUNT
    grp_dn = (feat["group_factor_dn"].astype(float).fillna(1.0)
              if "group_factor_dn" in feat else pd.Series(1.0, index=feat.index))
    dn = (dn * ctx_dn * grp_dn).clip(0.0, 100.0)

    return pd.DataFrame({"stage2_readiness": up, "stage4_readiness": dn})


def breaks_series(feat: pd.DataFrame) -> pd.DataFrame:
    close, ma = feat["close"], feat["ma30"]
    slope, rs = feat["ma_slope_pct"], feat["mansfield_rs"]
    volr = feat["vol_ratio_3w"] if "vol_ratio_3w" in feat else feat["vol_ratio"]

    up_cross = close > feat["resistance"]
    ma_ok = (close > ma) & (slope >= -0.25)
    rs_ok = rs > 0
    vol_ok = volr >= VOL_TEXTBOOK
    vol_soft = volr >= VOL_RELAXED
    base_ok = feat["base_width_pct"].isna() | (feat["base_width_pct"] <= MAX_BASE_WIDTH)
    age_col = feat["base_age_recent"] if "base_age_recent" in feat else feat["base_age_weeks"]
    age_ok = (age_col >= MIN_BASE_AGE).fillna(False)

    flagged_up = up_cross & ma_ok & (vol_soft | rs_ok) & age_ok
    confirmed_up = up_cross & ma_ok & rs_ok & vol_ok & base_ok & age_ok

    dn_cross = close < feat["support"]
    ma_ok_d = (close < ma) & (slope <= 0.25)
    rs_ok_d = rs < 0
    flagged_dn = dn_cross & (ma_ok_d | rs_ok_d) & age_ok
    confirmed_dn = dn_cross & ma_ok_d & rs_ok_d & age_ok

    return pd.DataFrame({
        "flagged_up": flagged_up.fillna(False),
        "confirmed_up": confirmed_up.fillna(False),
        "flagged_dn": flagged_dn.fillna(False),
        "confirmed_dn": confirmed_dn.fillna(False),
    })
