"""
Trade plans: entry trigger, stop, targets, size and a written exit sequence.

The scanner up to this point answered "which stocks" and stopped there, which
leaves the three decisions that actually determine the outcome unmade. This
module turns each candidate into the numbers an order ticket needs.

One design decision is worth stating up front because it changes what the levels
mean. Weinstein's entry rule is a WEEKLY CLOSE above the pivot on at least double
volume. A resting buy-stop order does not implement that rule: it fills the
moment price touches the level on any day, including on spikes that are back
below the pivot by Friday and that the weekly system would have rejected. Both
entry styles are produced below, with the trade-off made explicit, because the
choice between them is a real one and the levels are identical while the fill
behaviour is not.

Nothing here is a recommendation. These are the levels the stated rules produce
from the current bar, and the rules themselves have not yet been calibrated
against real prices.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class PlanConfig:
    account_size: float | None = None   # in the stock's own currency
    risk_pct: float = 1.0               # percent of the account risked per trade
    entry_buffer_atr: float = 0.25      # trigger sits this many weekly ATRs above the pivot
    min_buffer_pct: float = 0.4         # floor on that buffer, percent
    max_stop_pct: float = 12.0          # beyond this the structural stop is too far away
    # Targets are the base height multiplied by a factor that grows with how long
    # the base took to form. Weinstein's own objectives come from point and figure
    # horizontal counts, whose whole mechanism is that a wider base counts to a
    # bigger target, and his repeated claim is that the longer the base the larger
    # the move. A flat one-times-height target ignores that and, worse, is
    # arithmetically almost always under 1R, because the stop is set by the same
    # base the target is measured from.
    target_weeks_per_mult: float = 26.0
    t1_mult_min: float = 1.0
    t1_mult_max: float = 3.0
    t2_over_t1: float = 1.8
    first_scale_pct: float = 50.0       # portion sold at the first target
    breakeven_at_r: float = 1.0         # move the stop to entry after this many R of gain
    trail_atr: float = 0.5              # trailing stop sits this far under the 30 week average
    max_exposure_pct: float = 25.0      # cap on one position as a share of the account
    min_risk_pct: float = 0.75          # a stop nearer than this is noise, not a stop


DEFAULT = PlanConfig()


def _num(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def build_plan(row, cfg: PlanConfig = DEFAULT, side: str = "long") -> dict | None:
    """
    Build a plan from one scored row. `side` is "long" for a Stage 2 setup and
    "short" for a Stage 4 setup, which mirrors every level.
    """
    close = _num(row.get("close"))
    ma30 = _num(row.get("ma30"))
    atr = _num(row.get("atr"))
    res = _num(row.get("resistance"))
    sup = _num(row.get("support"))
    if close is None or ma30 is None or res is None or sup is None or atr is None:
        return None
    if res <= sup:
        return None

    base_height = res - sup
    sign = 1.0 if side == "long" else -1.0
    pivot = res if side == "long" else sup
    base_age = _num(row.get("base_age_recent")) or _num(row.get("base_age_weeks")) or 0.0
    t1_mult = float(np.clip(base_age / cfg.target_weeks_per_mult,
                            cfg.t1_mult_min, cfg.t1_mult_max))

    # --- entry ------------------------------------------------------------
    buffer_abs = max(cfg.entry_buffer_atr * atr, pivot * cfg.min_buffer_pct / 100.0)
    trigger = pivot + sign * buffer_abs

    # Weinstein explicitly allows the pullback entry, where price breaks out and
    # then returns to the old resistance which now acts as support. It is the
    # lower risk of the two entries and the one that most often does not happen.
    pull_far = pivot - sign * (0.5 * atr)
    pullback_zone = sorted([pivot, pull_far])

    # --- stop -------------------------------------------------------------
    # Weinstein puts the stop below the most recent significant low AND below the
    # 30 week average. On a long base the base floor can be thirty percent down,
    # which is not a stop, it is a decision to hold through a full Stage 4. The
    # nearest ten week swing low is the significant level in that phrasing.
    swing_lo = _num(row.get("recent_low_10w"))
    swing_hi = _num(row.get("recent_high_10w"))
    # The stop must sit beyond BOTH entries. When the 30 week average or the
    # recent swing low happens to sit above the pivot, taking either as the stop
    # reference puts the stop above the pullback buy price, so the ticket says
    # buy at 198.54 and stop out at 199.61. Both are valid Weinstein references
    # in the normal case and neither is a stop when it lands on the wrong side.
    if side == "long":
        base_ref = min(swing_lo if swing_lo is not None else sup, ma30)
        structural = min(base_ref - 0.25 * atr, pullback_zone[0] - 0.05 * atr)
        widest = trigger * (1 - cfg.max_stop_pct / 100.0)
        if widest > pullback_zone[0] - 0.05 * atr:
            # Honouring the percentage cap would require a stop above the entry.
            # Only reachable when a weekly ATR is a quarter of the price, where
            # there is no sane ticket to write.
            return None
        stop = max(structural, widest)
        tightened = structural < widest
    else:
        base_ref = max(swing_hi if swing_hi is not None else res, ma30)
        structural = max(base_ref + 0.25 * atr, pullback_zone[1] + 0.05 * atr)
        widest = trigger * (1 + cfg.max_stop_pct / 100.0)
        if widest < pullback_zone[1] + 0.05 * atr:
            return None
        stop = min(structural, widest)
        tightened = structural > widest

    # Signed, not absolute. With ma30 or the recent swing low above the trigger,
    # the structural stop lands on the FAR side of the entry and abs() turned a
    # long whose stop sat 39 per cent above the buy price into a plausible
    # looking ticket, with the percentage cap silently failing open.
    risk_per_share = (trigger - stop) * sign
    if not np.isfinite(risk_per_share) or risk_per_share <= 0:
        return None
    risk_pct = risk_per_share / trigger * 100.0
    if not np.isfinite(risk_pct) or risk_pct < cfg.min_risk_pct:
        # A stop inside a fraction of a percent is not a stop, it is a coin
        # flip that also produces an absurd position size.
        return None

    # --- targets ----------------------------------------------------------
    # Measured move: the height of the base projected from the pivot. Weinstein
    # takes his objectives from point and figure horizontal counts, which cannot
    # be reconstructed cleanly from weekly OHLCV, so this is the defensible
    # proxy rather than the identical method.
    t1 = pivot + sign * t1_mult * base_height
    t2 = pivot + sign * t1_mult * cfg.t2_over_t1 * base_height
    # A short's move is bounded below by zero, and a measured move taken from a
    # wide base on a low priced stock runs straight through it: the projection
    # produced targets at minus 0.15 and minus 1.42, which are not prices. The
    # floor is a nominal 5 per cent of the pivot, flagged, rather than a negative
    # number presented as a level to buy back at.
    floored = False
    if side == "short":
        floor = pivot * 0.05
        if t1 < floor or t2 < floor:
            floored = True
            t1, t2 = max(t1, floor), max(t2, floor * 0.5)
    # Signed as well. A target behind the entry, reachable when the base is
    # narrower than a quarter of a weekly ATR, was reported as a positive R.
    r1 = (t1 - trigger) * sign / risk_per_share
    r2 = (t2 - trigger) * sign / risk_per_share
    if r1 <= 0 or r2 <= r1:
        # The entry buffer exceeds the whole measured move, which happens when a
        # weekly ATR dwarfs the base. A ticket whose first target is behind its
        # own entry is not a trade with a poor reward shape, it is not a trade.
        return None

    # --- size -------------------------------------------------------------
    # Size from the PUBLISHED risk per share, not the full-precision one, so a
    # reader who multiplies the two numbers on the page gets the third number on
    # the page. Rounding the output of an unrounded calculation leaves the three
    # mutually inconsistent at the fourth decimal, which is small in money and
    # corrosive in a table that claims to show its working.
    risk_per_share = round(risk_per_share, 4)
    if risk_per_share <= 0:
        return None
    shares_per_1000 = 1000.0 / risk_per_share
    size = {
        "risk_per_share": risk_per_share,
        "shares_per_1000_risked": round(shares_per_1000, 2),
        "exposure_per_1000_risked": round(shares_per_1000 * trigger, 2),
    }
    if cfg.account_size and cfg.account_size > 0:
        cash_at_risk = cfg.account_size * cfg.risk_pct / 100.0
        shares = cash_at_risk / risk_per_share
        # A fixed fractional risk says nothing about exposure. Since
        # exposure = risk_budget / risk_pct, a stop 1 per cent away at 1 per cent
        # account risk buys the entire account, and three such positions is 300
        # per cent invested with nothing anywhere objecting.
        max_shares = cfg.account_size * cfg.max_exposure_pct / 100.0 / trigger
        capped = shares > max_shares
        shares = min(shares, max_shares)
        size.update({
            "account_size": cfg.account_size,
            "risk_pct_of_account": cfg.risk_pct,
            "cash_at_risk": round(cash_at_risk, 2),
            "shares": int(shares),
            "exposure": round(int(shares) * trigger, 2),
            "exposure_pct_of_account": round(int(shares) * trigger / cfg.account_size * 100.0, 1),
            "capped_by_exposure": bool(capped),
        })

    notes = []
    if floored:
        notes.append("the measured move projects through zero, so the targets are floored. "
                     "Treat them as a bound rather than an objective.")
    if cfg.account_size and size.get("capped_by_exposure"):
        notes.append(
            f"position size cut to the {cfg.max_exposure_pct:.0f}% exposure cap. The stop is "
            "close enough that risking the full budget would put an outsized share of the "
            "account into one name.")
    if cfg.account_size and size.get("shares") == 0:
        notes.append("the account is too small to take one share at this risk budget.")
    if tightened:
        notes.append(
            f"structural stop sat more than {cfg.max_stop_pct:.0f}% away, so it has been "
            "pulled in to that cap. A stop inside the base is more likely to be hit by "
            "noise than by the trade being wrong.")
    if r1 < 1.5:
        notes.append(
            f"first target is only {r1:.1f}R from the trigger. The base is shallow "
            "relative to the stop distance, which is a poor reward shape.")
    if side == "long" and close > trigger:
        notes.append("price is already above the trigger, so the breakout entry has gone. "
                     "The pullback zone is the remaining entry.")

    return {
        "side": side,
        "pivot": round(pivot, 4),
        "trigger": round(trigger, 4),
        "trigger_pct_from_close": round((trigger / close - 1.0) * 100.0, 2),
        "pullback_low": round(pullback_zone[0], 4),
        "pullback_high": round(pullback_zone[1], 4),
        "stop": round(stop, 4),
        "stop_tightened": bool(tightened),
        "risk_per_share": round(risk_per_share, 4),
        "risk_pct": round(risk_pct, 2),
        "base_height": round(base_height, 4),
        "base_age_weeks": round(base_age, 0),
        "t1_mult": round(t1_mult, 2),
        "t1": round(t1, 4), "t2": round(t2, 4),
        "t1_r": round(r1, 2), "t2_r": round(r2, 2),
        "breakeven_trigger": round(trigger + sign * cfg.breakeven_at_r * risk_per_share, 4),
        "trail_level_now": round(ma30 - sign * cfg.trail_atr * atr, 4),
        "size": size,
        "notes": notes,
        "sequence": exit_sequence(cfg, side),
    }


def exit_sequence(cfg: PlanConfig, side: str = "long") -> list[str]:
    """
    The written plan. Weinstein's selling doctrine is a sequence rather than a
    single exit: take part of the position off at the measured objective, protect
    the rest, then let the 30 week average decide when the trend is over.
    """
    d = "rises" if side == "long" else "falls"
    below = "below" if side == "long" else "above"
    return [
        f"Initial stop stays where it is until price {d} one R from the trigger.",
        f"At one R of open gain, move the stop to the entry price. The trade can no "
        f"longer lose money and the decision from here is only about how much it makes.",
        f"At the first target, sell {cfg.first_scale_pct:.0f}% of the position. This is the "
        f"measured move of the base and the point Weinstein takes a partial profit.",
        f"Trail the remainder at {cfg.trail_atr:g} weekly ATR {below} the 30 week average, "
        f"updated once a week on the Friday close. Do not tighten it intra-week.",
        f"Exit the remainder in full on a weekly close {below} the 30 week average while "
        f"that average has stopped rising. That is the Stage 3 warning and it is the "
        f"condition the trade was predicated on being absent.",
        "Exit in full regardless of price if Mansfield relative strength turns negative "
        "for three consecutive weeks, or if the stage classification drops out of Stage 2.",
    ]


# ----------------------------------------------------------------------------
# Managing an open position
# ----------------------------------------------------------------------------

def manage_position(feat: pd.DataFrame, pos: dict, cfg: PlanConfig = DEFAULT) -> dict:
    """
    Given a stock's feature history and a held position, work out what the plan
    says to do this week.

    The original plan is rebuilt from the bar at the entry date rather than from
    today's bar, because the targets and the initial stop were set by the base
    that existed then. Recomputing them from the current bar would silently move
    the goalposts every week and make the plan unfalsifiable.
    """
    entry_price = float(pos["entry_price"])
    shares = float(pos.get("shares") or 0)
    side = str(pos.get("side") or "long")
    sign = 1.0 if side == "long" else -1.0

    entry_date = pd.Timestamp(pos["entry_date"])
    hist = feat.loc[:entry_date]
    if hist.empty:
        return {"ticker": pos["ticker"], "action": "UNKNOWN",
                "reason": "no price history at or before the entry date"}
    if entry_date > feat.index[-1]:
        # Otherwise the slice silently returns the whole frame and the "plan as
        # it stood at entry" is rebuilt from this week's bar, which is the exact
        # failure the entry-date anchoring exists to prevent. A typed year in
        # positions.csv should not produce confident nonsense.
        return {"ticker": pos["ticker"], "action": "CHECK INPUT",
                "reason": f"entry date {entry_date.date()} is after the last bar "
                          f"{feat.index[-1].date()}"}
    plan = build_plan(hist.iloc[-1], cfg, side=side)
    row = feat.iloc[-1]
    close = float(row["close"])
    ma30 = _num(row.get("ma30"))
    atr = _num(row.get("atr"))
    stage = str(pos.get("stage") or "")

    init_stop = float(pos.get("initial_stop") or (plan or {}).get("stop") or np.nan)
    risk = abs(entry_price - init_stop) if not np.isnan(init_stop) else np.nan
    open_r = (close - entry_price) * sign / risk if risk and not np.isnan(risk) else np.nan

    # Highest close since entry decides whether the first target was reached and
    # whether the breakeven move was earned, so a target hit and given back is
    # still recorded rather than quietly forgotten.
    seg = feat.loc[entry_date:]["close"]
    peak = float(seg.max() if side == "long" else seg.min()) if len(seg) else close
    t1 = (plan or {}).get("t1")
    t1_hit = bool(t1 is not None and ((peak >= t1) if side == "long" else (peak <= t1)))
    be_hit = bool(not np.isnan(open_r) and
                  ((peak - entry_price) * sign / risk >= cfg.breakeven_at_r)) if risk else False

    # The trailing stop must ratchet. Recomputing it from this week's average
    # alone meant a falling ma30 LOWERED the stop week after week, handing back
    # protection the position had already earned, while the written plan told
    # the user never to loosen it. There is nowhere to persist last week's stop,
    # so it is reconstructed as the running maximum of the trail level over every
    # week the position has been open, which is what a ratchet would have done.
    seg_feat = feat.loc[entry_date:]
    trail_series = None
    if "ma30" in seg_feat and "atr" in seg_feat:
        trail_series = seg_feat["ma30"] - sign * cfg.trail_atr * seg_feat["atr"]
        trail_series = (trail_series.cummax() if side == "long"
                        else trail_series.cummin())
    trail = float(trail_series.iloc[-1]) if trail_series is not None and \
        len(trail_series) and pd.notna(trail_series.iloc[-1]) else None

    stop_now = init_stop
    if be_hit and not np.isnan(init_stop):
        # Only ever a tightening. Assigning the entry price unconditionally could
        # LOWER a stop that already sat above entry, under a rule whose stated
        # purpose is that the trade can no longer lose money.
        stop_now = max(init_stop, entry_price) if side == "long" \
            else min(init_stop, entry_price)
    if t1_hit and trail is not None and not np.isnan(stop_now):
        stop_now = max(stop_now, trail) if side == "long" else min(stop_now, trail)

    # --- what the plan says this week ---------------------------------------
    action, reason = "HOLD", "no plan condition met this week"
    ma_ok = ma30 is not None and ((close > ma30) if side == "long" else (close < ma30))
    slope = _num(row.get("ma_slope_pct"))
    slope_ok = slope is not None and ((slope > 0) if side == "long" else (slope < 0))
    rs = _num(row.get("mansfield_rs"))

    if np.isnan(stop_now):
        # Reachable when the CSV omits initial_stop and the entry bar was too
        # early for a plan. Previously the stop test was skipped in silence and
        # the position ran unprotected forever while the panel read HOLD.
        action = "SET A STOP"
        reason = ("no initial stop recorded and none could be rebuilt from the entry "
                  "bar, so this position has no exit level at all")
    elif (close <= stop_now) if side == "long" else (close >= stop_now):
        action, reason = "EXIT ALL", f"weekly close through the stop at {stop_now:.2f}"
    elif ma30 is None or slope is None:
        action, reason = "CHECK INPUT", ("the 30 week average is unavailable this week, so no "
                                         "trend condition can be evaluated")
    elif not ma_ok and not slope_ok:
        action, reason = "EXIT ALL", ("weekly close on the wrong side of the 30 week average "
                                      "with that average no longer trending in your favour")
    elif rs is not None and ((rs < 0) if side == "long" else (rs > 0)) and \
            _rs_streak(feat.loc[entry_date:], side) >= 3:
        # Counted from the entry date. Measured over the whole series, a
        # position opened last week inherited a twelve week streak that predated
        # it and was exited immediately, under a reason that asserted something
        # false about the position.
        action, reason = "EXIT ALL", "relative strength has been against the position " \
                                     "for three weeks since entry"
    elif t1_hit and not _flag(pos, "trimmed"):
        action = f"TRIM {cfg.first_scale_pct:.0f}%"
        reason = (f"first target at {t1:.2f} was reached "
                  f"{_weeks_since_level(seg, t1, side)} weeks ago"
                  if _weeks_since_level(seg, t1, side) else
                  f"first target at {t1:.2f} has been reached")
    elif be_hit and not _flag(pos, "stop_moved"):
        action, reason = "MOVE STOP TO BREAKEVEN", f"open gain passed {cfg.breakeven_at_r:g}R"

    return {
        "ticker": pos["ticker"], "side": side,
        "entry_date": str(entry_date.date()), "entry_price": round(entry_price, 4),
        "shares": shares, "close": round(close, 4),
        "open_pct": round((close / entry_price - 1.0) * 100.0 * sign, 2),
        "open_r": None if np.isnan(open_r) else round(float(open_r), 2),
        "initial_stop": None if np.isnan(init_stop) else round(init_stop, 4),
        "stop_now": None if np.isnan(stop_now) else round(float(stop_now), 4),
        "trail_level": None if trail is None else round(trail, 4),
        "t1": t1, "t1_hit": t1_hit, "t2": (plan or {}).get("t2"),
        "stage": str(row.get("stage", stage)) if "stage" in row else stage,
        "action": action, "reason": reason,
    }


def _flag(pos: dict, name: str) -> bool:
    """
    Read a done-flag from the positions row.

    These are set by the user in positions.csv, and nothing in the code writes
    them. Until one is set the same instruction repeats every week, which is the
    correct default for an instruction that has not been carried out, but it
    means TRIM will also mask MOVE STOP TO BREAKEVEN indefinitely. The columns
    are documented so the loop can actually be closed.
    """
    v = pos.get(name)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "y", "done")


def _weeks_since_level(seg: pd.Series, level, side: str):
    if level is None or seg is None or not len(seg):
        return None
    hit = (seg >= level) if side == "long" else (seg <= level)
    if not hit.any():
        return None
    return int(len(seg) - 1 - int(np.flatnonzero(hit.to_numpy())[0]))


def _rs_streak(feat: pd.DataFrame, side: str) -> int:
    rs = feat["mansfield_rs"].dropna()
    n = 0
    for v in reversed(rs.to_numpy()):
        against = v < 0 if side == "long" else v > 0
        if against:
            n += 1
        else:
            break
    return n


def alert_rows(rec: dict) -> list[dict]:
    """
    Flat price alerts for one candidate, in the shape a broker or a phone app
    wants: one row per level, with the reason attached so a fired alert says
    what it means rather than only that a number was touched.
    """
    p = rec.get("plan")
    if not p:
        return []
    up = p["side"] == "long"
    rows = [
        ("BUY STOP" if up else "SELL STOP", p["trigger"],
         "breakout trigger, " + ("above" if up else "below") + " the base pivot"),
        ("BUY LIMIT" if up else "SELL LIMIT", p["pullback_high"] if up else p["pullback_low"],
         "pullback entry, old resistance acting as support" if up
         else "pullback entry, old support acting as resistance"),
        ("STOP LOSS", p["stop"], "initial stop, below base support and the 30 week average"
         if up else "initial stop, above base resistance and the 30 week average"),
        ("BREAKEVEN", p["breakeven_trigger"], "move the stop to entry here"),
        ("TARGET 1", p["t1"], f"measured move, sell part of the position ({p['t1_r']}R)"),
        ("TARGET 2", p["t2"], f"second measured move ({p['t2_r']}R)"),
    ]
    return [{"ticker": rec["ticker"], "type": t, "price": round(v, 4), "why": w}
            for t, v, w in rows]
