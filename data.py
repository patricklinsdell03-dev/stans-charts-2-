"""
Price data layer. Weekly OHLCV from Yahoo through yfinance, cached to disk.

Two details matter more than they look.

The first is the partial week. Yahoo returns an in-progress weekly bar whenever
you ask before Friday's close. That bar has a truncated volume figure and a close
that is not a weekly close, so every volume rule and every breakout comparison
silently misfires. The scanner drops it unless told otherwise.

The second is adjustment. Weekly bars are pulled adjusted for splits and
dividends so that the 30 week average and the base boundaries are not broken by
corporate actions.
"""

from __future__ import annotations

import os
import time
import datetime as dt

import pandas as pd

CACHE_DIR = os.environ.get("WEINSTEIN_CACHE", "cache")
PRICE_CACHE = os.path.join(CACHE_DIR, "weekly_prices.parquet")


def _import_yf():
    try:
        import yfinance as yf
        return yf
    except ImportError as e:                 # noqa: BLE001
        raise SystemExit(
            "yfinance is not installed. Run: pip install -r requirements.txt"
        ) from e


def download_weekly(tickers: list[str], years: int = 8, batch: int = 120,
                    pause: float = 1.0) -> dict[str, pd.DataFrame]:
    """Returns {ticker: DataFrame[open,high,low,close,volume]} on a weekly index."""
    yf = _import_yf()
    start = (dt.date.today() - dt.timedelta(days=int(365.25 * years))).isoformat()
    out: dict[str, pd.DataFrame] = {}

    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        print(f"  downloading {i + 1}-{i + len(chunk)} of {len(tickers)}")
        try:
            raw = yf.download(chunk, start=start, interval="1wk",
                              auto_adjust=True, group_by="ticker",
                              threads=True, progress=False, timeout=45)
        except Exception as e:               # noqa: BLE001
            print(f"    batch failed: {e}")
            time.sleep(pause * 3)
            continue

        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                df = df.dropna(subset=["close"])
                if len(df) >= 60:
                    out[t] = df
            except Exception:                # noqa: BLE001
                continue
        time.sleep(pause)

    print(f"  usable series: {len(out)} of {len(tickers)}")
    return out


def drop_partial_week(df: pd.DataFrame, asof: dt.date | None = None) -> pd.DataFrame:
    """
    Remove the final bar if its week has not finished. Yahoo stamps weekly bars
    with the Monday of the week, so a bar is complete once today is past the
    following Saturday.
    """
    if df.empty:
        return df
    asof = asof or dt.date.today()
    last = df.index[-1]
    last_date = last.date() if hasattr(last, "date") else last
    week_end = last_date + dt.timedelta(days=(6 - last_date.weekday()) % 7)
    if asof <= week_end:
        return df.iloc[:-1]
    return df


def save_cache(data: dict[str, pd.DataFrame]) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = []
    for t, df in data.items():
        d = df.copy()
        d["ticker"] = t
        frames.append(d.reset_index().rename(columns={d.index.name or "index": "date",
                                                      "Date": "date"}))
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(PRICE_CACHE, index=False)


def load_cache() -> dict[str, pd.DataFrame]:
    if not os.path.exists(PRICE_CACHE):
        return {}
    df = pd.read_parquet(PRICE_CACHE)
    out = {}
    for t, g in df.groupby("ticker"):
        g = g.set_index("date").sort_index()
        out[t] = g[["open", "high", "low", "close", "volume"]]
    return out
