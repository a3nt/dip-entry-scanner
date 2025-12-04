# scanner.py
# Dip-Entry Scanner v3 – hourly GitHub Actions runner with options ideas.
# Writes timestamped CSV to data/ only during NYSE hours.

import os, sys
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal

# -------- Config --------
TICKERS = ["GLD", "SLV", "URNM", "URA", "UROY"]
LOOKBACK_PERIOD = "2y"
PROX_PCT = 0.02
RSI_CALL = 46.0
RSI_STRONG = 40.0
RSI_EXTREME = 35.0
SAVE_DIR = os.environ.get("SCAN_SAVE_DIR", "data")

# -------- Time/Calendar --------
_EASTERN = pytz.timezone("US/Eastern")
_CAL = mcal.get_calendar('XNYS')

def market_open_now(dt_utc: datetime | None = None) -> bool:
    """True when within NYSE regular session (holiday-aware)."""
    now_utc = dt_utc or datetime.utcnow().replace(tzinfo=pytz.utc)
    now_et = now_utc.astimezone(_EASTERN)
    sched = _CAL.schedule(start_date=now_et.date(), end_date=now_et.date())
    if sched.empty:
        return False
    open_et = sched.iloc[0]["market_open"].to_pydatetime().astimezone(_EASTERN)
    close_et = sched.iloc[0]["market_close"].to_pydatetime().astimezone(_EASTERN)
    return open_et <= now_et <= close_et

# -------- Helpers --------
def normalize_ohlcv(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        if symbol in df.columns.get_level_values(-1):
            df = df.xs(symbol, level=-1, axis=1)
        else:
            df = df.droplevel(0, axis=1)
    df.columns = [str(c) for c in df.columns]
    return df

def _as_series1d(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1-column DataFrame, got {x.shape}")
        x = x.iloc[:, 0]
    return pd.Series(x, dtype="float64")

def compute_rsi(close, period: int = 14) -> pd.Series:
    """Wilder RSI; masks avoid ndarray fillna issues."""
    close = _as_series1d(close)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rma_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rma_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = rma_gain / rma_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    both_zero = (rma_gain == 0) & (rma_loss == 0)
    loss_zero = (rma_loss == 0) & ~both_zero
    gain_zero = (rma_gain == 0) & ~both_zero
    rsi = rsi.where(~loss_zero, 100.0).where(~gain_zero, 0.0).where(~both_zero, 50.0)
    return rsi.clip(0, 100)

def near(x: float, y: float, pct: float = PROX_PCT) -> bool:
    if x is None or y is None or pd.isna(x) or pd.isna(y) or y == 0:
        return False
    return abs(x - y) / abs(y) <= pct

def proximity_pct(x: float, y: float) -> float:
    if x is None or y is None or pd.isna(x) or pd.isna(y) or y == 0:
        return np.nan
    return abs(x - y) / abs(y)

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

def option_mid(bid, ask):
    b = safe_float(bid); a = safe_float(ask)
    if pd.isna(b) and pd.isna(a): return np.nan
    if pd.isna(b): return a
    if pd.isna(a): return b
    return (b + a) / 2.0

# -------- Options helper --------
def pick_expiration(symbol: str, min_dte: int, max_dte: int, now_utc: datetime) -> str | None:
    tk = yf.Ticker(symbol)
    exps = list(tk.options or [])
    if not exps:
        return None
    today = now_utc.date()
    window = []
    beyond = None
    best_beyond = 10**9
    for exp in exps:
        try:
            dte = (pd.Timestamp(exp).date() - today).days
        except Exception:
            continue
        if min_dte <= dte <= max_dte:
            window.append((dte, exp))
        elif dte >= min_dte and dte < best_beyond:
            best_beyond, beyond = dte, exp
    if window:
        window.sort(key=lambda x: x[0])
        return window[0][1]
    return beyond

def suggest_options(symbol: str, price: float, signal: str, now_utc: datetime):
    """Heuristic suggestions via moneyness + DTE; uses delayed yfinance mids."""
    if signal not in ("CALL BIAS", "STRONG CALL BIAS", "EXTREME DIP - BUY ZONE"):
        return None

    if signal == "CALL BIAS":
        min_dte, max_dte, mny_buy = 30, 45, 0.05
    elif signal == "STRONG CALL BIAS":
        min_dte, max_dte, mny_buy = 45, 60, 0.10
    else:
        min_dte, max_dte, mny_buy = 60, 90, 0.12  # EXTREME

    exp = pick_expiration(symbol, min_dte, max_dte, now_utc)
    if not exp:
        return {"error": "no expirations"}

    tk = yf.Ticker(symbol)
    try:
        chain = tk.option_chain(exp)
        calls = chain.calls.copy()
    except Exception:
        return {"error": "option chain unavailable"}

    if calls.empty:
        return {"error": "empty calls"}

    calls["mid"] = [option_mid(b, a) for b, a in zip(calls.get("bid"), calls.get("ask"))]
    calls["moneyness"] = (calls["strike"] / price) - 1.0
    calls = calls.replace([np.inf, -np.inf], np.nan).dropna(subset=["mid","moneyness"])
    if calls.empty:
        return {"error": "no priced calls"}

    def nearest(target):
        idx = (calls["moneyness"] - target).abs().idxmin()
        return calls.loc[idx]

    asof = now_utc.astimezone(_EASTERN).strftime("%Y-%m-%d %H:%M %Z")

    if signal != "EXTREME DIP - BUY ZONE":
        lc = nearest(mny_buy)
        k = float(lc["strike"]); debit = float(lc["mid"])
        if not np.isfinite(debit): return {"error": "no mid price"}
        return {
            "type": "LONG CALL", "as_of": asof, "expiration": exp,
            "strike": round(k,2), "mid_debit": round(debit,2),
            "breakeven_at_expiry": round(k+debit,2),
            "notes": "Heuristic: OTM by moneyness; delayed quotes."
        }

    buy = nearest(mny_buy)
    sell = nearest(0.20)
    k_buy = float(buy["strike"]); d_buy = float(buy["mid"])
    k_sell = float(sell["strike"]); d_sell = float(sell["mid"])
    if not (np.isfinite(d_buy) and np.isfinite(d_sell)):
        return {"error": "invalid mids for spread"}
    if k_sell <= k_buy:
        higher = calls[calls["strike"] > k_buy].sort_values("strike").head(1)
        if higher.empty: return {"error": "no higher strike for short leg"}
        k_sell = float(higher["strike"]); d_sell = float(higher["mid"])
    width = k_sell - k_buy
    debit = d_buy - d_sell
    return {
        "type": "BULL CALL SPREAD", "as_of": asof, "expiration": exp,
        "long_strike": round(k_buy,2), "short_strike": round(k_sell,2),
        "net_debit": round(debit,2), "max_gain": round(width-debit,2),
        "max_loss": round(debit,2), "breakeven_at_expiry": round(k_buy+debit,2),
        "notes": "Heuristic: +12%/+20% OTM spread; delayed quotes."
    }

# -------- Scan once --------
def scan_once(now_utc: datetime | None = None) -> pd.DataFrame:
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=pytz.utc)
    rows = []
    for symbol in TICKERS:
        print(f"Downloading {symbol}...")
        data = yf.download(symbol, period=LOOKBACK_PERIOD, auto_adjust=True, progress=False, threads=True)
        if data is None or data.empty:
            rows.append({"Symbol": symbol, "Signal": "N/A", "Reason": "no data"})
            continue

        df = normalize_ohlcv(data, symbol).copy()
        if "Close" not in df.columns or "Volume" not in df.columns:
            rows.append({"Symbol": symbol, "Signal": "N/A", "Reason": "missing cols"})
            continue

        df["Close"] = _as_series1d(df["Close"])
        df["Volume"] = _as_series1d(df["Volume"])
        df["SMA50"]  = df["Close"].rolling(50,  min_periods=50).mean()
        df["SMA100"] = df["Close"].rolling(100, min_periods=100).mean()
        df["SMA200"] = df["Close"].rolling(200, min_periods=200).mean()
        df["RSI14"]  = compute_rsi(df["Close"])
        df["VolMA5"]  = df["Volume"].rolling(5,  min_periods=5).mean()
        df["VolMA20"] = df["Volume"].rolling(20, min_periods=20).mean()

        needed = ["Close","SMA50","SMA100","SMA200","RSI14","Volume","VolMA5","VolMA20"]
        last_valid = df.dropna(subset=needed).tail(1)
        if last_valid.empty:
            last_date = df.index[-1].strftime("%Y-%m-%d")
            rows.append({"Symbol": symbol, "Date": last_date, "Signal": "INSUFFICIENT DATA",
                         "Reason": "need full SMA/RSI/Vol history"})
            continue

        latest = last_valid.iloc[0]
        last_date = last_valid.index[-1].strftime("%Y-%m-%d")

        price  = safe_float(latest["Close"])
        sma50  = safe_float(latest["SMA50"])
        sma100 = safe_float(latest["SMA100"])
        sma200 = safe_float(latest["SMA200"])
        rsi    = safe_float(latest["RSI14"])
        vol    = safe_float(latest["Volume"])
        v5     = safe_float(latest["VolMA5"])
        v20    = safe_float(latest["VolMA20"])

        trend = "UP" if price > sma200 else "DOWN"
        prox50  = proximity_pct(price, sma50)
        prox100 = proximity_pct(price, sma100)
        near50  = near(price, sma50, PROX_PCT)
        near100 = near(price, sma100, PROX_PCT)
        near_ma, near_px = (None, np.nan)
        if pd.notna(prox50) and pd.notna(prox100):
            near_ma, near_px = ("SMA50", prox50) if prox50 <= prox100 else ("SMA100", prox100)
        elif pd.notna(prox50):  near_ma, near_px = ("SMA50", prox50)
        elif pd.notna(prox100): near_ma, near_px = ("SMA100", prox100)
        volume_rising = (v5 > v20) and (vol >= v5) if all(pd.notna(x) for x in (v5, v20, vol)) else False

        signal, reason = "NEUTRAL", "Downtrend: no dip calls" if trend == "DOWN" else ""
        if trend == "UP":
            if near100 and rsi < RSI_EXTREME and volume_rising:
                signal, reason = "EXTREME DIP - BUY ZONE", "Near SMA100 + RSI < 35 + rising volume"
            elif (near50 or near100) and rsi < RSI_STRONG:
                signal, reason = "STRONG CALL BIAS", "Dip near MA + RSI < 40"
            elif (near50 or near100) and rsi < RSI_CALL:
                signal, reason = "CALL BIAS", "Dip near MA + RSI < 46"

        option_idea = suggest_options(symbol, price, signal, now_utc)

        rows.append({
            "Symbol": symbol, "Date": last_date,
            "Price": round(price, 2), "SMA50": round(sma50, 2), "SMA100": round(sma100, 2), "SMA200": round(sma200, 2),
            "RSI14": round(rsi, 1), "VolMA5": round(v5, 0), "VolMA20": round(v20, 0),
            "VolumeRising": bool(volume_rising),
            "Trend": trend, "NearMA": near_ma, "ProximityPct": round(float(near_px), 4) if pd.notna(near_px) else None,
            "Signal": signal, "Reason": reason, "OptionIdea": option_idea
        })

    df_results = pd.DataFrame(rows, columns=[
        "Symbol","Date","Price","SMA50","SMA100","SMA200","RSI14",
        "VolMA5","VolMA20","VolumeRising","Trend","NearMA","ProximityPct",
        "Signal","Reason","OptionIdea"
    ])
    return df_results

def main():
    now_utc = datetime.utcnow().replace(tzinfo=pytz.utc)
    if not market_open_now(now_utc):
        print("NYSE closed. Skipping save to avoid noise.")
        return 0

    os.makedirs(SAVE_DIR, exist_ok=True)
    df = scan_once(now_utc)
    ts = datetime.now(tz=_EASTERN).strftime("%Y%m%d_%H%M%Z")
    out_csv = os.path.join(SAVE_DIR, f"dip_scanner_results_{ts}.csv")
    df.to_csv(out_csv, index=False)
    df.to_csv(os.path.join(SAVE_DIR, "latest.csv"), index=False)
    print(f"Saved → {out_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
