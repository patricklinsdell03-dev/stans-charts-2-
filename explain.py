"""
Plain-English translation of a scored row.

Everything the dashboard shows is a number with a threshold behind it, and a
number with a threshold behind it is only legible if you already hold the whole
model in your head. This module turns one row into sentences: what the model
thinks is happening, what it says to do, and what would make it wrong.

Two rules govern what may be written here. Every sentence is derived from a value
in the row, so nothing is asserted that the scan did not measure. And where the
model is uncertain or its own audit says a component is weak, the text says so
rather than rounding it into confidence.
"""

from __future__ import annotations

import numpy as np

STAGE_PLAIN = {
    "Stage 1": ("Basing.",
                "It fell, then stopped falling. Price is drifting sideways and the 30 week "
                "average has gone flat. Nothing is happening yet, which is the point: this is "
                "where a position is prepared, not taken."),
    "Stage 1 to 2": ("Basing, and starting to resolve upward.",
                     "Still sideways, but the average has begun to lift and relative strength "
                     "is improving. This is the last quiet stretch before a breakout, if one "
                     "comes. Most of them do not."),
    "Stage 2": ("Advancing.",
                "Price is above a rising 30 week average and beating the market. This is the "
                "only stage Weinstein buys in. Whether it is a good buy depends on how far "
                "into the advance it already is."),
    "Stage 2 to 3": ("Advancing, but losing momentum.",
                     "Still above the average, and the average has stopped accelerating. The "
                     "trend is intact and it is later than it was. A place to tighten stops "
                     "rather than to add."),
    "Stage 3": ("Topping.",
                "It rose, then stopped rising. Sideways after an advance, with the average "
                "flat. Looks identical to a base on the chart and resolves the opposite way "
                "more often than not. Weinstein does not buy here."),
    "Stage 3 to 4": ("Topping, and starting to break down.",
                     "The average has rolled over and relative strength is deteriorating. If "
                     "you own it, this is the exit. If you do not, it is a short candidate "
                     "rather than a bargain."),
    "Stage 4": ("Declining.",
                "Price below a falling 30 week average, underperforming the market. Every "
                "bounce here is a bounce in a downtrend until the average flattens. This is "
                "the stage that ruins people who buy dips."),
    "Stage 4 to 1": ("Declining, but the fall is slowing.",
                     "The average is still falling and falling less steeply. Too early to buy "
                     "and worth watching, because this is what the beginning of a base looks "
                     "like from the inside."),
    "Stage 1 or 3, undetermined": ("Undetermined.",
                                   "Flat average, and no decisive trend before it, so the model "
                                   "cannot tell a floor from a ceiling. It is saying it does not "
                                   "know rather than guessing."),
    "Insufficient history": ("Not enough history.",
                             "Fewer weeks of data than the 30 week average and the 52 week "
                             "relative strength line need. Anything scored here would be noise."),
}

GRADE_PLAIN = {
    "A": "Everything lines up: a textbook break, by a leader inside a leading sector, with the "
         "market behind it. This is the case the whole system exists to find, and it is rare.",
    "B": "A textbook break with a neutral sector behind it. Valid, one notch below the best case.",
    "C": "A textbook break with the sector or the market against it. This is the setup Weinstein "
         "tells you in plain words to pass on, and the grade exists to stop it looking like the "
         "ones above.",
    "D": "It crossed the level but failed at least one of the price, volume or relative strength "
         "rules. Not a Weinstein breakout. The tooltip on the signal badge says which rule.",
}


def _ago(age):
    try:
        a = int(age)
    except (TypeError, ValueError):
        return "recently"
    if a <= 0:
        return "this week"
    return "1 week ago" if a == 1 else f"{a} weeks ago"


def _n(row, key):
    v = row.get(key)
    try:
        v = float(v)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def explain(row: dict, market: dict | None = None) -> dict:
    """Returns {headline, what, group, action, invalidate, caveats:[...]}"""
    market = market or {}
    stage = str(row.get("stage") or "")
    head, what = STAGE_PLAIN.get(stage, (stage or "Unclassified.", ""))

    sig = row.get("signal") or {}
    kind = sig.get("kind")
    grade = row.get("grade")
    conf = bool(sig.get("confirmed"))
    age = row.get("signal_age_weeks")

    # --- what the group and the market are doing -------------------------
    sec_rs, own = _n(row, "sector_rs"), _n(row, "rs_vs_sector")
    rank = _n(row, "sector_rank_pct")
    if sec_rs is None or own is None:
        group = ("No sector information for this name, so it is being ranked on its own chart "
                 "alone and the group half of the method is simply not applied to it.")
    else:
        lead = "ahead of" if sec_rs > 0 else "behind"
        inside = "leads" if own > 0 else "lags"
        group = (f"Its sector is {lead} the market ({sec_rs:+.1f})"
                 + (f", ranking {rank:.0f}th percentile" if rank is not None else "")
                 + f", and within that sector this name {inside} its peers ({own:+.1f}). ")
        if sec_rs > 0 and own > 0:
            group += "Leader in a leading group, which is the combination Weinstein wants."
        elif sec_rs > 0 and own <= 0:
            group += ("So it is riding a strong sector rather than driving it, and the weaker "
                      "members of a hot group are the ones that give the move back first.")
        elif sec_rs <= 0 and own > 0:
            group += ("So it is the best house on a poor street. Worth something, and the "
                      "sector is still a headwind.")
        else:
            group += "Laggard in a lagging group, which is the case to leave alone."

    regime = str(market.get("regime") or "")
    allows = regime in ("Bull", "Improving")

    # --- what to do -------------------------------------------------------
    plan = row.get("plan") or {}
    trig, stop, t1 = plan.get("trigger"), plan.get("stop"), plan.get("t1")
    r1, riskpct = plan.get("t1_r"), plan.get("risk_pct")
    long_side = plan.get("side", "long") == "long"

    if kind == "stage2_breakout":
        when = _ago(age)
        action = (f"It broke above its base high {when}"
                  + (", meeting every textbook condition. " if conf else
                     ", but failed at least one textbook condition. "))
        if trig:
            action += (f"The entry on the plan is {trig:.2f}, the stop is {stop:.2f} "
                       f"({riskpct:.1f}% away) and the first target is {t1:.2f}, which is "
                       f"{r1:.1f} times what you are risking. ")
        if not allows and regime:
            action += (f"The market regime reads {regime}, which under the rules means do not "
                       "take new long breakouts regardless of how this chart looks.")
        elif grade == "C":
            action += "Graded C, so the rules say pass."
        elif grade in ("A", "B"):
            action += "The regime and the grade both permit this one."
    elif kind == "stage4_breakdown":
        when = _ago(age)
        action = (f"It broke below its support {when}. If you own it, this is the exit the plan "
                  "has been waiting for. As a short it needs the market and the sector against "
                  "the stock rather than merely the chart looking weak.")
    else:
        d = _n(row, "dist_to_resistance_pct")
        rdy = _n(row, "stage2_readiness")
        if long_side and trig and d is not None and d >= 0:
            action = (f"Nothing to do today. It has not broken out. The level to watch is "
                      f"{trig:.2f}, which is {d:.1f}% above the current price. Set an alert "
                      "there rather than buying now")
            action += (f", and note the readiness score of {rdy:.0f} is a ranking against other "
                       "candidates, not a probability." if rdy is not None else ".")
        else:
            action = ("Nothing to do. There is no valid entry level on this row, which usually "
                      "means the base geometry is missing or the setup is on the other side.")

    # --- what would make this wrong --------------------------------------
    ma = _n(row, "ma30")
    inval = []
    if plan.get("stop"):
        inval.append(f"a weekly close through {plan['stop']:.2f}, which is the stop")
    if ma:
        inval.append(f"a weekly close back below the 30 week average at {ma:.2f} while that "
                     "average has stopped rising")
    inval.append("relative strength turning negative for three weeks running")
    invalidate = "Walk away on " + "; or ".join(inval) + "."

    # --- honest caveats ---------------------------------------------------
    caveats = []
    for n in (plan.get("notes") or []):
        caveats.append(n)
    if not long_side:
        caveats.append("On the short side the audit found the readiness score adds almost "
                       "nothing over simple distance to support, so treat its ranking as "
                       "little more than a proximity sort.")
    vol = _n(row, "vol_ratio_3w")
    if kind == "stage2_breakout" and vol is not None and vol < 2.0:
        caveats.append(f"Peak volume was {vol:.1f} times the ten week average against a textbook "
                       "requirement of 2.0, so the buying that is supposed to confirm the break "
                       "did not show up.")
    ext = _n(row, "px_vs_ma_pct")
    if ext is not None and ext > 12:
        caveats.append(f"Price is {ext:.0f}% above its 30 week average, so this is late. "
                       "Weinstein's entry was at the pivot, not here.")
    caveats.append("None of the thresholds behind these numbers have been calibrated against "
                   "real prices yet, so treat the ranking as a way to order your own chart "
                   "reading rather than as a verdict.")

    return {"headline": head, "what": what, "group": group,
            "action": action, "invalidate": invalidate, "caveats": caveats,
            "grade_note": GRADE_PLAIN.get(grade or "", ""),
            "regime": regime, "regime_allows": allows}
