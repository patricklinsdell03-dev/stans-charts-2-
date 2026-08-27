"""
The pre-trade check prompt.

A weekly scan is a filter over twelve hundred charts. It has never looked at the
one chart you are about to trade, it cannot see anything that is not in weekly
OHLCV, and by the time you place the order it is between one and six days stale.
This builds a prompt that hands an assistant the scan's own numbers and asks it
to try to talk you out of the trade.

It is written to be adversarial on purpose. A prompt that asks "is this a good
setup" gets agreement, because agreement is the path of least resistance for a
model handed a confident-looking table. This one asks for the reasons not to,
and requires an explicit verdict with the single strongest objection named.
"""

from __future__ import annotations

TEMPLATE = """You are checking a single stock before I place a real order. Assume the
setup is flawed and try to find out how. If you cannot find a flaw, say so plainly, but
look first.

## What my scanner says

Ticker: {ticker} ({name}), {market} listing, {sector} sector
Scan week ending: {asof}
Stage: {stage}          Trend score: {trend_score} (-100 to +100)
Signal: {signal}        Grade: {grade}
Price at that close: {close}

Weinstein conditions as measured:
- 30 week average: {ma30}, slope {ma_slope_pct}% over 5 weeks
- Price vs that average: {px_vs_ma_pct}%
- Mansfield relative strength vs the index: {mansfield_rs} (4 week change {rs_slope})
- Sector relative strength: {sector_rs}, this stock vs its own sector: {rs_vs_sector}
- Base: {base_age_weeks} weeks old, {base_width_pct}% wide, resistance {resistance}, support {support}
- Breakout volume: {vol_ratio_3w}x the ten week average (textbook wants 2.0x)
- Market regime: {regime}

The plan it produced:
- Entry trigger {trigger}, pullback zone {pullback_low} to {pullback_high}
- Stop {stop} ({risk_pct}% risk)
- Target 1 {t1} ({t1_r}R), target 2 {t2} ({t2_r}R)

## What I need you to do

1. Get current data. Search for {ticker}'s price now and its recent weekly closes.
   My numbers are from the week ending {asof} and may be several days stale. If price
   has already run past the trigger or broken the stop, say so first, because that
   changes the trade rather than merely dating it.

2. Re-derive the four Weinstein conditions yourself from a current chart rather than
   trusting my table: price above a 30 week average that is flat or rising, relative
   strength positive against the index, a real base of at least eight weeks rather than
   a pause in a downtrend, and volume expansion on the breakout week. Tell me which of
   the four you can confirm, which you cannot, and which my numbers get wrong.

3. Look for what a weekly OHLCV scan structurally cannot see:
   - an earnings date inside the next four weeks, which turns a chart trade into an
     event bet
   - a takeover approach, a placing, a profit warning or a regulatory decision that
     explains the move and will not repeat
   - a share consolidation, split, spin-off or ticker change that corrupts the price
     history the base was measured from
   - dilution, going-concern language, or a heavily shorted float
   - liquidity: is the typical daily value traded enough that my position size is a
     small fraction of it, and what is the spread
   - for a UK line, whether it is priced in pence and whether my levels match

4. Check the trade shape, not just the setup. Is the stop somewhere the chart would
   actually invalidate the idea, or is it an arbitrary percentage? Is the first target
   at a real prior level or floating in space? Would this position be correlated with
   what I already hold?

5. Give me a verdict in this form and nothing more elaborate:
   VERDICT: GO / NO / WAIT
   The single strongest reason against it, in one sentence.
   What specifically would have to change for the answer to become GO.

Do not be agreeable. If the honest answer is that the setup is thin, say it is thin.
If my scanner's numbers disagree with what you find, my scanner is more likely to be
wrong than the market is.
"""

FIELDS = ["ticker", "name", "market", "sector", "asof", "stage", "trend_score",
          "signal", "grade", "close", "ma30", "ma_slope_pct", "px_vs_ma_pct",
          "mansfield_rs", "rs_slope", "sector_rs", "rs_vs_sector",
          "base_age_weeks", "base_width_pct", "resistance", "support",
          "vol_ratio_3w", "regime", "trigger", "pullback_low", "pullback_high",
          "stop", "risk_pct", "t1", "t1_r", "t2", "t2_r"]


def _fmt(v, dp=2):
    if v is None:
        return "not available"
    try:
        return f"{float(v):,.{dp}f}"
    except (TypeError, ValueError):
        return str(v)


def fields(rec: dict, asof: str = "", regime: str = "") -> dict:
    """
    The substitution values for one record.

    Separated from `build` so the dashboard can ship the template once and fill
    it in the browser, rather than embedding a two-kilobyte prompt per row.
    One template, one set of field names, no drift between the file you can read
    and the button you press.
    """
    p = rec.get("plan") or {}
    sig = rec.get("signal") or {}
    kind = sig.get("kind")
    sig_txt = "none, this has not broken out" if not kind else (
        f"{kind}, {'confirmed' if sig.get('confirmed') else 'unconfirmed'}"
        f", {int(rec.get('signal_age_weeks') or 0)} "
        f"{'week' if int(rec.get('signal_age_weeks') or 0) == 1 else 'weeks'} ago")
    vals = {
        "ticker": rec.get("ticker", ""), "name": rec.get("name") or "name not recorded",
        "market": rec.get("market", ""), "sector": rec.get("sector") or "unclassified",
        "asof": asof or "unknown", "stage": rec.get("stage", ""),
        "trend_score": _fmt(rec.get("trend_score"), 0), "signal": sig_txt,
        "grade": rec.get("grade") or "not graded",
        "close": _fmt(rec.get("close")), "ma30": _fmt(rec.get("ma30")),
        "ma_slope_pct": _fmt(rec.get("ma_slope_pct")),
        "px_vs_ma_pct": _fmt(rec.get("px_vs_ma_pct"), 1),
        "mansfield_rs": _fmt(rec.get("mansfield_rs"), 1),
        "rs_slope": _fmt(rec.get("rs_slope"), 1),
        "sector_rs": _fmt(rec.get("sector_rs"), 1),
        "rs_vs_sector": _fmt(rec.get("rs_vs_sector"), 1),
        "base_age_weeks": _fmt(rec.get("base_age_recent") or rec.get("base_age_weeks"), 0),
        "base_width_pct": _fmt(rec.get("base_width_pct"), 1),
        "resistance": _fmt(rec.get("resistance")), "support": _fmt(rec.get("support")),
        "vol_ratio_3w": _fmt(rec.get("vol_ratio_3w"), 1),
        "regime": regime or "not recorded",
        "trigger": _fmt(p.get("trigger")), "pullback_low": _fmt(p.get("pullback_low")),
        "pullback_high": _fmt(p.get("pullback_high")), "stop": _fmt(p.get("stop")),
        "risk_pct": _fmt(p.get("risk_pct"), 1), "t1": _fmt(p.get("t1")),
        "t1_r": _fmt(p.get("t1_r"), 1), "t2": _fmt(p.get("t2")),
        "t2_r": _fmt(p.get("t2_r"), 1),
    }
    return vals


def build(rec: dict, asof: str = "", regime: str = "") -> str:
    """Fill the template from one scored record."""
    return TEMPLATE.format(**fields(rec, asof, regime))
