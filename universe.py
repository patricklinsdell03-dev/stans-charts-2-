"""
Universe construction: S&P 500, S&P 400 and FTSE 350, mapped to Yahoo symbols.

Index membership is scraped from Wikipedia and then cached to CSV. The cache is
the source of truth on any run where the scrape fails, so a Wikipedia edit or an
outage degrades the run into "last known membership" rather than killing it.
"""

from __future__ import annotations

import io
import os
import time

import pandas as pd
import requests

CACHE_DIR = os.environ.get("WEINSTEIN_CACHE", "cache")
UA = {"User-Agent": "Mozilla/5.0 (weinstein-tracker; personal research)"}

# Column headings on these pages get edited, so every field is a list of
# candidates matched case-insensitively on a prefix, and if none match the table
# is located by looking for a column whose values actually look like tickers.
# Hard-coding one exact heading per page was the original approach and it makes
# the whole scan fail on a Wikipedia copy-edit.
SOURCES = {
    "SP500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
              ["symbol", "ticker"], ["security", "company"], ["gics sector", "sector"]),
    "SP400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
              ["symbol", "ticker"], ["security", "company"], ["gics sector", "sector"]),
    "FTSE100": ("https://en.wikipedia.org/wiki/FTSE_100_Index",
                ["ticker", "epic", "symbol"], ["company", "name"],
                ["ftse industry classification benchmark sector", "icb sector", "sector", "industry"]),
    "FTSE250": ("https://en.wikipedia.org/wiki/FTSE_250_Index",
                ["ticker", "epic", "symbol"], ["company", "name"],
                ["ftse industry classification benchmark sector", "icb sector", "sector", "industry"]),
}
MIN_ROWS = 40          # a constituent table is never smaller than this
TICKERY = 0.75         # share of values that must look like a ticker

BENCHMARKS = {"US": "^GSPC", "UK": "^FTAS"}


def to_yahoo(symbol: str, market: str) -> str:
    s = str(symbol).strip().upper()
    if market == "UK":
        s = s.replace(".", "-")          # BT.A becomes BT-A
        if not s.endswith(".L"):
            s = s + ".L"
        return s
    return s.replace(".", "-")           # BRK.B becomes BRK-B


def _read_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def _flat(cols) -> list[str]:
    """Wikipedia sometimes returns a MultiIndex header; flatten to plain strings."""
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if "Unnamed" not in str(x)]
            out.append(" ".join(dict.fromkeys(parts)).strip())
        else:
            out.append(str(c))
    return out


def _match(cols: list[str], candidates: list[str]) -> str | None:
    low = [c.strip().lower() for c in cols]
    for cand in candidates:
        for i, c in enumerate(low):
            if c == cand:
                return cols[i]
    for cand in candidates:
        for i, c in enumerate(low):
            if c.startswith(cand) or cand in c:
                return cols[i]
    return None


def _looks_like_tickers(series: pd.Series) -> bool:
    """
    Shape test for a ticker column, used only when no heading matches.

    Uniqueness is the load-bearing condition. Without it a column of fifty
    repeated single characters passes the character-class test perfectly, and the
    fallback happily returns a notes column from an unrelated table.
    """
    v = series.dropna().astype(str).str.strip()
    if len(v) < MIN_ROWS:
        return False
    if v.nunique() < len(v) * 0.9:
        return False
    good = v.str.fullmatch(r"[A-Z][A-Z0-9.\-]{0,7}", case=False)
    if good.mean() < TICKERY:
        return False
    # Tickers are short and mostly letters. Free text that happens to be short
    # and unique, such as a column of dates, fails on the letter share.
    return bool(v.str.contains(r"[A-Za-z]", regex=True).mean() >= 0.9)


def _pick(tables: list[pd.DataFrame], sym_cands: list[str]):
    """Returns (table, symbol_column) or (None, None)."""
    best = None
    for t in tables:
        if len(t) < MIN_ROWS:
            continue
        t = t.copy()
        t.columns = _flat(t.columns)
        col = _match(list(t.columns), sym_cands)
        if col is not None and _looks_like_tickers(t[col]):
            return t, col
        if best is None:
            # Fall back to any column that simply looks like a list of tickers.
            for c in t.columns:
                if _looks_like_tickers(t[c]):
                    best = (t, c)
                    break
    return best if best else (None, None)


def fetch_index(key: str) -> pd.DataFrame:
    url, sym_cands, name_cands, sec_cands = SOURCES[key]
    market = "UK" if key.startswith("FTSE") else "US"
    tables = _read_tables(url)
    t, sym_col = _pick(tables, sym_cands)
    if t is None:
        shapes = [(len(x), _flat(x.columns)[:6]) for x in tables[:6]]
        raise RuntimeError(
            f"could not locate a constituent table for {key}. "
            f"{len(tables)} tables on the page; first few are {shapes}")
    name_col = _match(list(t.columns), name_cands)
    sec_col = _match(list(t.columns), sec_cands)
    df = pd.DataFrame({
        "raw_symbol": t[sym_col].astype(str),
        "name": t[name_col].astype(str) if name_col else "",
        "sector": t[sec_col].astype(str) if sec_col else "",
    })
    df["index"] = key
    df["market"] = market
    df["ticker"] = [to_yahoo(s, market) for s in df["raw_symbol"]]
    return df.drop_duplicates("ticker").reset_index(drop=True)


def load_universe_file(path: str) -> pd.DataFrame:
    """
    A hand-supplied constituent list, for when the scrape is unavailable.

    Needs at least `ticker` and `market`; `name`, `sector` and `index` are used
    if present. Tickers are taken as given, so they must already be in Yahoo
    form, meaning BRK-B rather than BRK.B and BP.L rather than BP.
    """
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise RuntimeError(f"{path} needs a 'ticker' column; found {list(df.columns)}")
    if "market" not in df.columns:
        df["market"] = ["UK" if str(t).upper().endswith(".L") else "US" for t in df["ticker"]]
    for c, default in (("name", ""), ("sector", ""), ("index", "MANUAL")):
        if c not in df.columns:
            df[c] = default
    df["raw_symbol"] = df["ticker"]
    return df.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)


def build_universe(indices: list[str], refresh: bool = True,
                   universe_file: str | None = None) -> pd.DataFrame:
    if universe_file:
        df = load_universe_file(universe_file)
        print(f"  using {universe_file}: {len(df)} tickers")
        return df

    os.makedirs(CACHE_DIR, exist_ok=True)
    frames = []
    for key in indices:
        path = os.path.join(CACHE_DIR, f"universe_{key}.csv")
        df = None
        if refresh:
            try:
                df = fetch_index(key)
                df.to_csv(path, index=False)
            except Exception as e:            # noqa: BLE001
                print(f"  universe fetch failed for {key}: {e}")
        if df is None and os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  using cached membership for {key} ({len(df)} names)")
        if df is not None:
            frames.append(df)
        time.sleep(0.5)
    if not frames:
        raise RuntimeError("no universe available, neither live nor cached")
    out = pd.concat(frames, ignore_index=True).drop_duplicates("ticker")
    return out.reset_index(drop=True)
