"""
Preflight diagnosis.

The scan touches two external services and both can fail in ways that produce a
single unhelpful exit code: Wikipedia for index membership, Yahoo for prices.
This runs each dependency separately and says which one broke and how, so a
failed run names its cause instead of leaving you to infer it.

Every check is independent and none of them stop the others, so one run produces
a complete picture rather than the first problem only.
"""

from __future__ import annotations

import io
import sys
import traceback

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


def _line(status, name, detail=""):
    print(f"  [{status:^4}] {name}" + (f"\n         {detail}" if detail else ""))
    return status


def run() -> int:
    print("=" * 72)
    print("WEINSTEIN TRACKER PREFLIGHT")
    print("=" * 72)
    results = []

    # --- 1. environment ---------------------------------------------------
    print("\nEnvironment")
    print(f"  python {sys.version.split()[0]}")
    for mod in ("pandas", "numpy", "requests", "lxml", "yfinance", "pyarrow"):
        try:
            m = __import__(mod)
            _line(OK, f"{mod} {getattr(m, '__version__', 'unknown')}")
        except Exception as e:                        # noqa: BLE001
            results.append(_line(FAIL, mod, f"not importable: {e}"))

    # --- 2. Wikipedia, per index -------------------------------------------
    print("\nIndex membership from Wikipedia")
    from . import universe as U
    got_any = False
    for key in U.SOURCES:
        try:
            df = U.fetch_index(key)
            got_any = True
            sample = ", ".join(df["ticker"].head(4))
            _line(OK, f"{key}: {len(df)} constituents", f"first four: {sample}")
        except Exception as e:                        # noqa: BLE001
            results.append(_line(FAIL, key, f"{type(e).__name__}: {e}"))
            url = U.SOURCES[key][0]
            try:
                import requests, pandas as pd
                r = requests.get(url, headers=U.UA, timeout=30)
                tables = pd.read_html(io.StringIO(r.text))
                print(f"         page fetched OK, HTTP {r.status_code}, "
                      f"{len(tables)} tables found")
                for i, t in enumerate(tables[:4]):
                    if len(t) > 40:
                        cols = [str(c)[:28] for c in t.columns][:8]
                        print(f"         table {i} has {len(t)} rows, columns: {cols}")
            except Exception as e2:                   # noqa: BLE001
                print(f"         page itself unreachable: {type(e2).__name__}: {e2}")
    if not got_any:
        print("\n  Every index scrape failed. That is almost always Wikipedia refusing")
        print("  the request or having changed its table headings, not your setup.")
        print("  Workaround: commit a universe.csv with columns")
        print("  ticker,name,sector,market,index and pass --universe-file universe.csv")

    # --- 3. Yahoo prices ----------------------------------------------------
    print("\nPrices from Yahoo")
    try:
        from . import data as D
        px = D.download_weekly(["AAPL", "MSFT"], years=2, batch=2, pause=0.2)
        if not px:
            results.append(_line(FAIL, "two-ticker download",
                                 "returned nothing; yfinance reached the service but "
                                 "produced no usable series"))
        else:
            t = sorted(px)[0]
            df = px[t]
            _line(OK, f"downloaded {len(px)} of 2 tickers",
                  f"{t}: {len(df)} weekly bars, last {df.index[-1].date()}, "
                  f"columns {list(df.columns)}")
    except Exception as e:                            # noqa: BLE001
        results.append(_line(FAIL, "price download", f"{type(e).__name__}: {e}"))
        traceback.print_exc()

    # --- 4. benchmarks ------------------------------------------------------
    print("\nBenchmarks")
    try:
        from . import data as D, universe as U2
        bt = sorted(set(U2.BENCHMARKS.values()))
        bx = D.download_weekly(bt, years=2, batch=len(bt), pause=0.2)
        for b in bt:
            if b in bx:
                _line(OK, f"{b}: {len(bx[b])} bars")
            else:
                results.append(_line(FAIL, b, "no data; relative strength cannot be computed "
                                              "for that market"))
    except Exception as e:                            # noqa: BLE001
        results.append(_line(FAIL, "benchmark download", f"{type(e).__name__}: {e}"))

    # --- 5. disk ------------------------------------------------------------
    print("\nLocal")
    try:
        import pandas as pd, os
        os.makedirs("cache", exist_ok=True)
        pd.DataFrame({"a": [1]}).to_parquet("cache/_probe.parquet")
        os.remove("cache/_probe.parquet")
        _line(OK, "parquet write")
    except Exception as e:                            # noqa: BLE001
        results.append(_line(WARN, "parquet write", f"{type(e).__name__}: {e}; "
                                                    "price caching will not work"))

    fails = [r for r in results if r == FAIL]
    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED. The first FAIL above is the cause of the "
              f"scan's exit code 1.")
    else:
        print("All checks passed. If the scan still fails, the fault is inside the "
              "scoring rather than the inputs.")
    print("=" * 72)
    return 1 if fails else 0
