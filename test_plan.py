"""
Verification of the trade plan arithmetic and the position state machine.

Every level here becomes an order somewhere, so the invariants are checked
directly rather than eyeballed: a long stop below its trigger, targets above it,
R multiples that reconcile with the risk per share, sizing that risks what it
claims to risk, and a management sequence that fires in the right order.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from weinstein import plan as PL, synth, features as F


def row(**kw):
    base = dict(close=100.0, ma30=95.0, atr=3.0, resistance=104.0, support=88.0,
                recent_low_10w=93.0, recent_high_10w=104.0, base_age_recent=52.0)
    base.update(kw)
    return pd.Series(base)


def main():
    bad = []

    # ---- 1. long invariants ------------------------------------------------
    p = PL.build_plan(row(), side="long")
    print(f"long: trigger {p['trigger']}  stop {p['stop']}  risk {p['risk_pct']}%  "
          f"T1 {p['t1']} ({p['t1_r']}R)  T2 {p['t2']} ({p['t2_r']}R)  mult x{p['t1_mult']}")
    checks = [
        ("stop below trigger", p["stop"] < p["trigger"]),
        ("t1 above trigger", p["t1"] > p["trigger"]),
        ("t2 above t1", p["t2"] > p["t1"]),
        ("trigger above pivot", p["trigger"] > p["pivot"]),
        ("risk reconciles", abs(p["risk_per_share"] - (p["trigger"] - p["stop"])) < 1e-6),
        ("t1_r reconciles", abs(p["t1_r"] - (p["t1"] - p["trigger"]) / p["risk_per_share"]) < 0.01),
        ("breakeven is one R up", abs(p["breakeven_trigger"] - (p["trigger"] + p["risk_per_share"])) < 1e-6),
    ]
    for name, ok in checks:
        if not ok:
            bad.append(f"long plan: {name} failed")

    # ---- 2. short plan mirrors --------------------------------------------
    q = PL.build_plan(row(), side="short")
    print(f"short: trigger {q['trigger']}  stop {q['stop']}  T1 {q['t1']} ({q['t1_r']}R)")
    for name, ok in [("stop above trigger", q["stop"] > q["trigger"]),
                     ("t1 below trigger", q["t1"] < q["trigger"]),
                     ("t2 below t1", q["t2"] < q["t1"]),
                     ("trigger below pivot", q["trigger"] < q["pivot"])]:
        if not ok:
            bad.append(f"short plan: {name} failed")

    # ---- 3. the stop cap actually binds ------------------------------------
    far = PL.build_plan(row(recent_low_10w=60.0, ma30=62.0), side="long")
    print(f"distant support: risk {far['risk_pct']}%  tightened={far['stop_tightened']}  "
          f"notes={len(far['notes'])}")
    if not far["stop_tightened"]:
        bad.append("stop cap did not engage on a distant support")
    if far["risk_pct"] > PL.DEFAULT.max_stop_pct + 0.01:
        bad.append(f"risk {far['risk_pct']}% exceeds the {PL.DEFAULT.max_stop_pct}% cap")
    if not far["notes"]:
        bad.append("tightened stop produced no warning note")

    # ---- 4. target multiple scales with base maturity ----------------------
    short_base = PL.build_plan(row(base_age_recent=10.0))["t1_mult"]
    long_base = PL.build_plan(row(base_age_recent=200.0))["t1_mult"]
    print(f"target multiple: 10 week base x{short_base}, 200 week base x{long_base}")
    if not (short_base < long_base):
        bad.append("a longer base did not produce a larger target multiple")
    if long_base > PL.DEFAULT.t1_mult_max + 1e-9:
        bad.append("target multiple exceeded its cap")

    # ---- 5. sizing risks what it says it risks -----------------------------
    cfg = PL.PlanConfig(account_size=10_000.0, risk_pct=1.0)
    sized = PL.build_plan(row(), cfg)["size"]
    actual = sized["shares"] * (PL.build_plan(row(), cfg)["risk_per_share"])
    print(f"sizing: {sized['shares']} shares, {actual:.2f} at risk against a "
          f"{sized['cash_at_risk']:.2f} budget")
    if actual > sized["cash_at_risk"] + 1e-6:
        bad.append(f"position risks {actual:.2f} against a stated budget of {sized['cash_at_risk']}")
    if actual < sized["cash_at_risk"] - PL.build_plan(row(), cfg)["risk_per_share"]:
        bad.append("position is more than one share smaller than the budget allows")

    # ---- 6. alert rows -----------------------------------------------------
    rows = PL.alert_rows({"ticker": "TEST", "plan": p})
    kinds = [r["type"] for r in rows]
    print("alerts:", kinds)
    if kinds != ["BUY STOP", "BUY LIMIT", "STOP LOSS", "BREAKEVEN", "TARGET 1", "TARGET 2"]:
        bad.append(f"unexpected alert rows: {kinds}")
    prices = {r["type"]: r["price"] for r in rows}
    if not (prices["STOP LOSS"] < prices["BUY LIMIT"] <= prices["BUY STOP"]
            < prices["TARGET 1"] < prices["TARGET 2"]):
        bad.append(f"alert prices out of order: {prices}")

    # ---- 7. the management state machine -----------------------------------
    idx, mkt, prices_d, meta = synth.build_universe()
    feat = F.compute_features(prices_d["LEA3"], mkt)
    entry_i = len(feat) - 40
    entry_date = feat.index[entry_i]
    entry_px = float(feat["close"].iloc[entry_i])

    # A stop far below can never be hit, so the sequence should progress.
    live = PL.manage_position(feat, {"ticker": "LEA3", "entry_date": str(entry_date.date()),
                                     "entry_price": entry_px, "shares": 100,
                                     "initial_stop": entry_px * 0.85, "side": "long"})
    print(f"open position: {live['open_pct']}% ({live['open_r']}R) -> {live['action']} "
          f"[{live['reason']}]")
    if live["action"] not in ("HOLD", "TRIM 50%", "MOVE STOP TO BREAKEVEN", "EXIT ALL"):
        bad.append(f"unexpected action {live['action']}")
    if live["t1_hit"] and live["action"] not in ("TRIM 50%", "EXIT ALL"):
        bad.append("first target was reached but the plan did not call a trim or an exit")

    # A stop set above the current price must produce an immediate exit.
    stopped = PL.manage_position(feat, {"ticker": "LEA3", "entry_date": str(entry_date.date()),
                                        "entry_price": entry_px, "shares": 100,
                                        "initial_stop": float(feat["close"].iloc[-1]) * 1.05,
                                        "side": "long"})
    print(f"stopped position: {stopped['action']} [{stopped['reason']}]")
    if stopped["action"] != "EXIT ALL":
        bad.append(f"price through the stop did not produce EXIT ALL, got {stopped['action']}")

    # The plan must be rebuilt from the entry bar, not this week's.
    late = PL.manage_position(feat, {"ticker": "LEA3", "entry_date": str(feat.index[-3].date()),
                                     "entry_price": float(feat["close"].iloc[-3]), "shares": 10,
                                     "initial_stop": float(feat["close"].iloc[-3]) * 0.9,
                                     "side": "long"})
    if late["t1"] is not None and live["t1"] is not None and late["t1"] == live["t1"]:
        bad.append("two entries on different dates produced an identical target, "
                   "which means the plan is being rebuilt from the current bar")
    print(f"entry-date anchoring: T1 at week -40 = {live['t1']}, at week -3 = {late['t1']}")

    print("\n" + ("FAILURES:\n  " + "\n  ".join(bad) if bad else "PLAN CHECKS PASSED"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
