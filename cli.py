"""
Command line entry point.

  python -m weinstein.cli scan                     full live scan, writes docs/index.html
  python -m weinstein.cli scan --indices SP500     one index only
  python -m weinstein.cli scan --cache             rescore from cached prices, no network
  python -m weinstein.cli demo                     synthetic data, proves the pipeline offline
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from . import features as F
from . import plan as PL
from . import report as R
from . import scan as SC
from . import stages as S

ALL_INDICES = ["SP500", "SP400", "FTSE100", "FTSE250"]


def _load_positions(path):
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"ticker", "entry_date", "entry_price"}
    if not need.issubset(df.columns):
        raise SystemExit(f"{path} needs at least the columns {sorted(need)}")
    return df.to_dict("records")


def cmd_scan(a):
    cfg = PL.PlanConfig(account_size=a.account, risk_pct=a.risk_pct)
    res = SC.run(a.indices, years=a.years, use_cache=a.cache,
                 keep_partial=a.keep_partial, calibrate=not a.no_calibrate,
                 min_price=a.min_price, plan_cfg=cfg,
                 positions=_load_positions(a.positions),
                 universe_file=getattr(a, "universe_file", None))
    lists = SC.slice_lists(res["scan"], top=a.top)
    _write(res, lists, a.out)


def _write(res, lists, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    html = R.build_html(lists, res["scan"], res["calibration"], res["asof"], res["generated"],
                        market=res.get("market"), sectors=res.get("sectors"),
                        regime_history=res.get("regime_history"),
                        positions=res.get("positions"))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    # Flat alert levels, one row per price, ready to paste into a broker or a
    # phone alert app. Written for the two long lists and the two short lists.
    alerts = []
    for key in ("breakouts", "watch_stage2", "breakdowns", "watch_stage4"):
        for r in lists.get(key, pd.DataFrame()).to_dict("records") if key in lists else []:
            alerts.extend(PL.alert_rows(r))
    if alerts:
        ap = os.path.splitext(out)[0] + "_alerts.csv"
        pd.DataFrame(alerts).to_csv(ap, index=False)
        print(f"wrote {ap} ({len(alerts)} alert levels)")

    csv_path = os.path.splitext(out)[0] + "_full.csv"
    if not res["scan"].empty:
        res["scan"].drop(columns=["up_parts", "dn_parts", "signal"], errors="ignore") \
            .to_csv(csv_path, index=False)
    print(f"wrote {out}")
    for k, v in lists.items():
        print(f"  {k}: {len(v)}")


# ---------------------------------------------------------------------------
# Offline demo. Generates synthetic tickers spanning the full stage cycle so the
# whole pipeline can be exercised without a network connection. Every ticker is
# fabricated and is labelled as such in the output.
# ---------------------------------------------------------------------------

def cmd_demo(a):
    """
    Offline end to end run on a synthetic multi-sector universe.

    Three sectors are constructed with known relative behaviour, and the same
    base-and-breakout pattern is planted in the strongest and the weakest of
    them. If the sector layer works, those two identical charts come out of the
    run with different grades, which is the whole point of adding it.
    """
    from . import synth
    res = synth.build_demo()
    lists = SC.slice_lists(res["scan"], top=a.top)
    _write(res, lists, a.out)
    m = res["market"]
    print(f"  regime: {m.get('regime')} {m.get('regime_score')}, "
          f"breadth {m.get('breadth_above_ma')}%")
    for r in res["sectors"]:
        print(f"    {r['sector']:<22} rs {r['sector_rs']:>7.2f}  rank {r['rank_pct']}")
    cols = ["ticker", "sector", "stage", "trend_score", "stage2_readiness",
            "sector_rs", "rs_vs_sector", "group_factor", "grade"]
    sc = res["scan"]
    show = sc[sc["signal"].notna() | (sc["stage2_readiness"] > 45)][cols]
    print(show.to_string(index=False))


def cmd_horizons(a):
    """
    Run the same rules over several history lengths from one download.

    This exists to make survivorship bias observable rather than assumed. The
    rules do not change between runs, and index membership is today's in every
    run, so a longer window is by construction a longer history OF THE SURVIVORS.
    If the measured edge improves as the window lengthens, that improvement is
    mostly the bias becoming visible, because nothing else moved.

    A flat comparison is the reassuring result. A steeply improving one means the
    eight year numbers are the honest ones and the long-window numbers are the
    flattering ones.
    """
    from . import data as D, universe as U
    years = sorted(a.years_list, reverse=True)
    uni = U.build_universe(a.indices, refresh=not a.cache)
    wanted = list(uni["ticker"]) + sorted({U.BENCHMARKS[m] for m in uni["market"].unique()})
    if a.cache:
        px = D.load_cache()
    else:
        px = D.download_weekly(wanted, years=years[0])
        D.save_cache(px)

    out = []
    for y in years:
        cut = {}
        for t, df in px.items():
            n = int(y * 52)
            cut[t] = df.iloc[-n:] if len(df) > n else df
        print(f"\n=== {y} year window ===")
        res = SC.run(a.indices, years=y, use_cache=True, calibrate=True,
                     min_price=a.min_price, prices=cut)
        cal = res.get("calibration") or {}
        conf = [s for s in (cal.get("signals") or [])
                if s.get("kind") == "stage2_breakout" and s.get("confirmed")]
        unc = [u for u in (cal.get("uncertainty") or [])
               if u.get("kind") == "stage2_breakout" and u.get("confirmed")]
        d = cal.get("discrimination_up") or {}
        out.append({
            "years": y,
            "signals": conf[0]["n"] if conf else 0,
            "median_excess_13w": conf[0].get("median_excess_13w") if conf else None,
            "win_rate_13w": conf[0].get("win_rate_13w") if conf else None,
            "stopped_out_13w_pct": conf[0].get("stopped_out_13w_pct") if conf else None,
            "ci_low": unc[0].get("ci_low") if unc else None,
            "ci_high": unc[0].get("ci_high") if unc else None,
            "episodes": unc[0].get("independent_episodes") if unc else None,
            "auc_increment": d.get("increment"),
        })

    df = pd.DataFrame(out).set_index("years")
    print("\n" + "=" * 72)
    print("CONFIRMED STAGE 2 BREAKOUTS, SAME RULES, DIFFERENT HISTORY LENGTHS")
    print("=" * 72)
    print(df.to_string())
    if len(out) > 1:
        a0, a1 = out[-1], out[0]          # shortest, longest
        m0, m1 = a0["median_excess_13w"], a1["median_excess_13w"]
        if m0 is not None and m1 is not None:
            print(f"\nMedian 13 week excess moves from {m0:.2f}% over {a0['years']} years "
                  f"to {m1:.2f}% over {a1['years']} years.")
            print("Index membership is today's in both, so most of any improvement is "
                  "survivorship\nbecoming visible rather than the method working better "
                  "in the past.")
    df.to_csv("docs/horizons.csv")
    print("\nwrote docs/horizons.csv")


def cmd_doctor(a):
    from . import doctor
    raise SystemExit(doctor.run())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="weinstein")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan")
    s.add_argument("--indices", nargs="+", default=ALL_INDICES, choices=ALL_INDICES)
    s.add_argument("--years", type=int, default=8)
    s.add_argument("--cache", action="store_true", help="rescore from cached prices")
    s.add_argument("--keep-partial", action="store_true",
                   help="keep the in-progress weekly bar, off by default")
    s.add_argument("--no-calibrate", action="store_true")
    s.add_argument("--min-price", type=float, default=1.0)
    s.add_argument("--top", type=int, default=40)
    s.add_argument("--account", type=float, default=None,
                   help="account size, so position sizes are in shares rather than per 1000 risked")
    s.add_argument("--risk-pct", type=float, default=1.0,
                   help="percent of the account risked per trade, default 1")
    s.add_argument("--universe-file", default=None, dest="universe_file",
                   help="CSV of tickers to scan instead of scraping index membership")
    s.add_argument("--positions", default="positions.csv",
                   help="CSV of open positions to manage: ticker,entry_date,entry_price,shares,initial_stop")
    s.add_argument("--out", default="docs/index.html")
    s.set_defaults(func=cmd_scan)

    doc = sub.add_parser("doctor", help="check every external dependency and say which one broke")
    doc.set_defaults(func=cmd_doctor)

    h = sub.add_parser("horizons", help="same rules over several history lengths")
    h.add_argument("--indices", nargs="+", default=ALL_INDICES, choices=ALL_INDICES)
    h.add_argument("--years-list", nargs="+", type=int, default=[8, 15, 25],
                   dest="years_list")
    h.add_argument("--cache", action="store_true")
    h.add_argument("--min-price", type=float, default=1.0)
    h.set_defaults(func=cmd_horizons)

    d = sub.add_parser("demo")
    d.add_argument("--top", type=int, default=40)
    d.add_argument("--out", default="docs/demo.html")
    d.add_argument("--account", type=float, default=None)
    d.add_argument("--risk-pct", type=float, default=1.0)
    d.set_defaults(func=cmd_demo)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
