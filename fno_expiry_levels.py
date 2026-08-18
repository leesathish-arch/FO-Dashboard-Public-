"""
SVLK ANALYSIS — F&O MONTHLY EXPIRY LEVEL COMMAND CENTRE  (v2 — multi-source)
============================================================================
Live NSE F&O dashboard on the monthly expiry-day reference level strategy:

    Previous month's expiry-day DAILY CANDLE HIGH / LOW  ->  reference levels
    Live LTP vs those levels                             ->  breakout categories

v2 changes: NSE is no longer required. If NSE blocks the connection (common on
non-Indian IPs), the app falls back to Yahoo for live prices and to a local CSV
for the F&O universe. Source in use is always shown on screen.

Run:  streamlit run fno_expiry_levels_v2.py
"""

import io
import os
import time
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except Exception:
    PLOTLY = False

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
IST = ZoneInfo("Asia/Kolkata")
NSE = "https://www.nseindia.com"

UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/market-data/live-equity-market",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Connection": "keep-alive",
}

YAHOO_FIX = {"M&M": "M&M.NS", "M&MFIN": "M&MFIN.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS"}

STATUS_ORDER = ["🟢 ABOVE HIGH", "🟠 NEAR HIGH", "⚪ INSIDE", "🟡 NEAR LOW", "🔴 BELOW LOW"]
STATUS_BG = {"🟢 ABOVE HIGH": "#0f7b3d", "🟠 NEAR HIGH": "#c96a12", "⚪ INSIDE": "#4b5563",
             "🟡 NEAR LOW": "#b3960b", "🔴 BELOW LOW": "#a51d1d"}

st.set_page_config(page_title="SVLK Analysis — F&O Expiry Levels",
                   page_icon="📊", layout="wide")


def find_default_csv() -> str:
    """Look for the universe CSV beside the script first, then common fallbacks."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "fno_symbols.csv"),
                 os.path.join(here, "symbols.csv"),
                 r"D:\Python\Dashboard\fno_symbols.csv",
                 r"D:\Python\symbols.csv"):
        if os.path.exists(cand):
            return cand
    return os.path.join(here, "fno_symbols.csv")


DEFAULT_CSV = find_default_csv()


def yahoo_ticker(sym: str) -> str:
    return YAHOO_FIX.get(sym, str(sym).strip().replace(" ", "") + ".NS")


# ----------------------------------------------------------------------------
# SOURCE A — NSE
# ----------------------------------------------------------------------------
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update(UA)
    for url in (NSE, NSE + "/market-data/live-equity-market"):
        try:
            s.get(url, timeout=8)
        except Exception:
            pass
    return s


def nse_get(path, params=None, tries=3):
    s = get_session()
    for _ in range(tries):
        try:
            r = s.get(NSE + path, params=params, timeout=12)
            if r.status_code == 200 and r.text.strip().startswith("{"):
                return r.json()
        except Exception:
            pass
        try:
            s.cookies.clear()
            s.get(NSE, timeout=8)
            s.get(NSE + "/api/marketStatus", timeout=8)
        except Exception:
            pass
        time.sleep(1.2)
    return None


@st.cache_data(ttl=120, show_spinner=False)
def nse_index_live(index_name: str) -> pd.DataFrame:
    js = nse_get("/api/equity-stockIndices", {"index": index_name})
    if not js or "data" not in js:
        return pd.DataFrame()
    rows = []
    for d in js["data"]:
        sym = str(d.get("symbol", "")).strip()
        if not sym or sym.upper() == index_name.upper():
            continue
        meta = d.get("meta") or {}
        rows.append({
            "SYMBOL": sym, "COMPANY": meta.get("companyName", ""),
            "SECTOR": meta.get("industry", "") or d.get("industry", ""),
            "LTP": pd.to_numeric(d.get("lastPrice"), errors="coerce"),
            "CHG": pd.to_numeric(d.get("change"), errors="coerce"),
            "%CHG": pd.to_numeric(d.get("pChange"), errors="coerce"),
            "52W_H": pd.to_numeric(d.get("yearHigh"), errors="coerce"),
            "52W_L": pd.to_numeric(d.get("yearLow"), errors="coerce"),
            "VOLUME": pd.to_numeric(d.get("totalTradedVolume"), errors="coerce"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=120, show_spinner=False)
def nse_index_quote(name: str):
    js = nse_get("/api/allIndices")
    if not js or "data" not in js:
        return None
    for d in js["data"]:
        if str(d.get("index", "")).upper() == name.upper():
            return {"last": d.get("last"), "pchange": d.get("percentChange")}
    return None


# ----------------------------------------------------------------------------
# SOURCE B — CSV universe (your symbols.csv from the VWAP project works as-is)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_universe_csv(raw: bytes) -> pd.DataFrame:
    """Accepts SYMBOL[,COMPANY][,SECTOR][,GROUPS] or separate NIFTY50/BANKNIFTY/FNO cols."""
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [str(c).strip().upper() for c in df.columns]
    symcol = next((c for c in df.columns if "SYMBOL" in c or c in ("TICKER", "STOCK")), None)
    if symcol is None:
        raise ValueError(f"No SYMBOL column found. Columns present: {list(df.columns)}")

    out = pd.DataFrame()
    out["SYMBOL"] = df[symcol].astype(str).str.strip().str.upper()
    out["COMPANY"] = df["COMPANY"].astype(str) if "COMPANY" in df.columns else ""
      out["SECTOR"] = df["SECTOR"].fillna("").astype(str).replace({"nan": "", "None": ""}) if "SECTOR" in df.columns else ""

    def tagged(tag, altcol):
        if "GROUPS" in df.columns:
            return df["GROUPS"].astype(str).str.upper().str.contains(tag, na=False)
        if altcol in df.columns:
            v = df[altcol].astype(str).str.strip().str.upper()
            return v.isin(["1", "Y", "YES", "TRUE", "X"])
        return pd.Series(False, index=df.index)

    out["N50"] = tagged("N50", "NIFTY50").values
    out["BNF"] = tagged("BNF", "BANKNIFTY").values
    out = out[out["SYMBOL"].str.len() > 0].drop_duplicates("SYMBOL").reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
# SOURCE C — Yahoo live prices (works from anywhere)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=180, show_spinner=False)
def yahoo_live(symbols: tuple) -> pd.DataFrame:
    """Latest 5-minute close as LTP; previous session close for CHG / %CHG."""
    syms, rows, chunk = list(symbols), [], 50
    prog = st.progress(0.0, text="Fetching live prices from Yahoo…")

    # previous completed daily close
    prev = {}
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        tk = [yahoo_ticker(s) for s in part]
        try:
            raw = yf.download(tk, period="7d", interval="1d", group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception:
            raw = None
        for s, t in zip(part, tk):
            try:
                d = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(how="all")
                if len(d) >= 2:
                    prev[s] = float(d["Close"].iloc[-2])
                    rows.append({"SYMBOL": s, "LTP": float(d["Close"].iloc[-1])})
            except Exception:
                continue
        prog.progress(min(0.5, (i + chunk) / max(1, len(syms)) * 0.5),
                      text=f"Fetching prices… {min(i+chunk, len(syms))}/{len(syms)}")

    live = {r["SYMBOL"]: r["LTP"] for r in rows}

    # refine with the most recent 5-minute bar (intraday accuracy)
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        tk = [yahoo_ticker(s) for s in part]
        try:
            raw = yf.download(tk, period="1d", interval="5m", group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception:
            raw = None
        for s, t in zip(part, tk):
            try:
                d = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(how="all")
                if len(d):
                    live[s] = float(d["Close"].iloc[-1])
            except Exception:
                continue
        prog.progress(min(1.0, 0.5 + (i + chunk) / max(1, len(syms)) * 0.5),
                      text=f"Refining intraday… {min(i+chunk, len(syms))}/{len(syms)}")
    prog.empty()

    out = []
    for s, ltp in live.items():
        p = prev.get(s)
        out.append({"SYMBOL": s, "LTP": ltp,
                    "CHG": (ltp - p) if p else np.nan,
                    "%CHG": ((ltp / p - 1) * 100) if p else np.nan})
    return pd.DataFrame(out)


@st.cache_data(ttl=180, show_spinner=False)
def yahoo_index(ticker: str):
    try:
        d = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d.dropna()
        if len(d) >= 2:
            last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
            return {"last": last, "pchange": (last / prev - 1) * 100}
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# HISTORY
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_daily_history(symbols: tuple, period: str = "1y") -> dict:
    out, chunk, syms = {}, 40, list(symbols)
    prog = st.progress(0.0, text="Downloading daily candles…")
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        tk = [yahoo_ticker(s) for s in part]
        try:
            raw = yf.download(tk, period=period, interval="1d", group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception:
            raw = None
        for s, t in zip(part, tk):
            try:
                df = (raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw).dropna(how="all")
                if df is not None and len(df):
                    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                    out[s] = df
            except Exception:
                continue
        prog.progress(min(1.0, (i + chunk) / max(1, len(syms))),
                      text=f"Downloading daily candles… {min(i+chunk, len(syms))}/{len(syms)}")
    prog.empty()
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def load_intraday(symbol: str, days: int = 5) -> pd.DataFrame:
    try:
        df = yf.download(yahoo_ticker(symbol), period=f"{days}d", interval="5m",
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(symbol: str) -> dict:
    try:
        info = yf.Ticker(yahoo_ticker(symbol)).info or {}
    except Exception:
        info = {}
    return {"PE": info.get("trailingPE"), "FWD_PE": info.get("forwardPE"),
            "EPS": info.get("trailingEps"), "PB": info.get("priceToBook"),
            "MCAP": info.get("marketCap"), "BETA": info.get("beta"),
            "INDUSTRY": info.get("industry")}


# ----------------------------------------------------------------------------
# EXPIRY + INDICATORS
# ----------------------------------------------------------------------------
def last_weekday_of_month(year, month, weekday):
    nxt = dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
    d = nxt - dt.timedelta(days=1)
    while d.weekday() != weekday:
        d -= dt.timedelta(days=1)
    return d


def prev_month_expiry(today, weekday):
    prev = today.replace(day=1) - dt.timedelta(days=1)
    return last_weekday_of_month(prev.year, prev.month, weekday)


def snap_to_trading_day(target, ref_index):
    if ref_index is None or len(ref_index) == 0:
        return target
    valid = ref_index[ref_index <= pd.Timestamp(target)]
    return valid[-1].date() if len(valid) else None


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def true_range(h, l, c):
    pc = c.shift(1)
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def atr(h, l, c, n=14):
    return true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr_n = true_range(h, l, c).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / tr_n
    mdi = 100 * pd.Series(minus, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / tr_n
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def macd(close, f=12, s=26, sig=9):
    line = ema(close, f) - ema(close, s)
    signal = ema(line, sig)
    return line, signal, line - signal


def rolling_vwap(df, n=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).rolling(n).sum() / df["Volume"].rolling(n).sum()


def session_vwap(intra):
    if intra.empty:
        return None
    idx = intra.index
    if idx.tz is not None:
        idx = idx.tz_convert(IST)
    d = intra[pd.Series(idx.date == idx[-1].date(), index=intra.index).values]
    if d.empty:
        return None
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    return float((tp * d["Volume"]).sum() / max(d["Volume"].sum(), 1))


def classify(ltp, hi, lo, near_pct):
    if any(pd.isna(x) for x in (ltp, hi, lo)) or hi <= 0 or lo <= 0:
        return "⚪ INSIDE"
    if ltp > hi:
        return "🟢 ABOVE HIGH"
    if ltp < lo:
        return "🔴 BELOW LOW"
    if ltp >= hi * (1 - near_pct / 100):
        return "🟠 NEAR HIGH"
    if ltp <= lo * (1 + near_pct / 100):
        return "🟡 NEAR LOW"
    return "⚪ INSIDE"


def build_board(live, hist, expiry, near_pct):
    ts, rows = pd.Timestamp(expiry), []
    for r in live.itertuples(index=False):
        df = hist.get(r.SYMBOL)
        if df is None or ts not in df.index:
            continue
        hi, lo = float(df.loc[ts, "High"]), float(df.loc[ts, "Low"])
        ltp = float(r.LTP) if pd.notna(r.LTP) else np.nan
        if not np.isfinite(ltp) or hi <= 0 or lo <= 0:
            continue
        rows.append({"SYMBOL": r.SYMBOL, "SECTOR": r.SECTOR, "LTP": ltp,
                     "EXP_HIGH": hi, "EXP_LOW": lo,
                     "%FROM_HIGH": (ltp - hi) / hi * 100,
                     "%FROM_LOW": (ltp - lo) / lo * 100,
                     "RANGE_POS": (ltp - lo) / max(hi - lo, 1e-9) * 100,
                     "STATUS": classify(ltp, hi, lo, near_pct),
                     "N50": bool(getattr(r, "N50", False)),
                     "BNF": bool(getattr(r, "BNF", False))})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("CHG", "%CHG"):
        out[col] = out["SYMBOL"].map(dict(zip(live["SYMBOL"], live[col])))
    out["STATUS"] = pd.Categorical(out["STATUS"], categories=STATUS_ORDER, ordered=True)
    return out


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
now_ist = dt.datetime.now(IST)
market_open = now_ist.weekday() < 5 and dt.time(9, 15) <= now_ist.time() <= dt.time(15, 30)

st.sidebar.title("⚙️ Controls")
st.sidebar.subheader("Data source")
price_src = st.sidebar.radio("Live prices", ["Yahoo only", "Auto (NSE → Yahoo)", "NSE only"],
                             index=0, help="NSE blocks non-Indian IPs; Yahoo works globally.")
uni_src = st.sidebar.radio("F&O universe", ["CSV file only", "Auto (NSE → CSV)"], index=0)
csv_file = st.sidebar.file_uploader("Universe CSV (SYMBOL / SECTOR / GROUPS)", type=["csv"])
csv_path = st.sidebar.text_input("…or path on disk", value=DEFAULT_CSV)

st.sidebar.subheader("Levels")
wd = 1 if st.sidebar.selectbox("Monthly expiry weekday", ["Tuesday", "Thursday"], 0) == "Tuesday" else 3
auto_exp = prev_month_expiry(now_ist.date(), wd)
manual = st.sidebar.checkbox("Override expiry date", value=False)
chosen_expiry = st.sidebar.date_input("Reference expiry date", value=auto_exp) if manual else auto_exp
near_pct = st.sidebar.slider("‘About to cross’ band (%)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.subheader("Refresh")
refresh_min = st.sidebar.slider("Auto-refresh (minutes)", 1, 15, 5)
only_when_open = st.sidebar.checkbox("Only during market hours", value=True)
if st.sidebar.button("🔄 Force refresh all data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"IST now: {now_ist:%d-%b-%Y %H:%M:%S}")
st.sidebar.caption("Market: " + ("🟢 OPEN" if market_open else "🔴 CLOSED"))

if (not only_when_open) or market_open:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_min * 60_000, key="tick")
    except Exception:
        import streamlit.components.v1 as components
        components.html(f"<script>setTimeout(()=>window.parent.location.reload(),"
                        f"{refresh_min*60_000});</script>", height=0)

# ----------------------------------------------------------------------------
# LOAD UNIVERSE
# ----------------------------------------------------------------------------
st.markdown("## 📊 SVLK ANALYSIS — F&O MONTHLY EXPIRY LEVEL COMMAND CENTRE")

universe, uni_note = pd.DataFrame(), ""
if uni_src.startswith("Auto"):
    with st.spinner("Trying NSE F&O list…"):
        nse_uni = nse_index_live("SECURITIES IN F&O")
    if not nse_uni.empty:
        n50 = set(nse_index_live("NIFTY 50")["SYMBOL"]) if not nse_index_live("NIFTY 50").empty else set()
        bnf = set(nse_index_live("NIFTY BANK")["SYMBOL"]) if not nse_index_live("NIFTY BANK").empty else set()
        nse_uni["N50"] = nse_uni["SYMBOL"].isin(n50)
        nse_uni["BNF"] = nse_uni["SYMBOL"].isin(bnf)
        universe, uni_note = nse_uni, "NSE live F&O list"

if universe.empty:
    raw = None
    if csv_file is not None:
        raw = csv_file.getvalue()
    elif csv_path.strip():
        try:
            with open(csv_path.strip(), "rb") as fh:
                raw = fh.read()
        except Exception as e:
            st.warning(f"Could not read {csv_path} — {type(e).__name__}")
    if raw:
        try:
            universe, uni_note = load_universe_csv(raw), f"CSV ({csv_path or csv_file.name})"
        except Exception as e:
            st.error(f"CSV problem: {e}")

if universe.empty:
    st.error("No universe loaded. NSE is unreachable from this network and no CSV was found.\n\n"
             "Upload a CSV in the sidebar with a **SYMBOL** column (a **SECTOR** column and a "
             "**GROUPS** column tagged N50 / BNF are optional but enable the group tabs). "
             "Your existing symbols.csv from the VWAP project works as-is.")
    st.stop()

symbols = tuple(sorted(universe["SYMBOL"].unique()))

# ----------------------------------------------------------------------------
# LOAD PRICES
# ----------------------------------------------------------------------------
live, price_note = pd.DataFrame(), ""
if price_src != "Yahoo only":
    nse_live = nse_index_live("SECURITIES IN F&O")
    if not nse_live.empty:
        live, price_note = nse_live, "NSE (real-time)"
if live.empty and price_src != "NSE only":
    live, price_note = yahoo_live(symbols), "Yahoo (may lag NSE by up to 15 min)"

if live.empty:
    st.error("No live prices available from the selected source. Switch 'Live prices' to "
             "**Yahoo only** in the sidebar.")
    st.stop()

live = live.drop(columns=[c for c in ("SECTOR", "N50", "BNF") if c in live.columns])
live = universe.merge(live, on="SYMBOL", how="inner")

hist = load_daily_history(symbols, "1y")
ref_index = next((hist[s].index for s in ("RELIANCE", "HDFCBANK", "INFY") if s in hist), None)
expiry = snap_to_trading_day(chosen_expiry, ref_index)
board = build_board(live, hist, expiry, near_pct)

if board.empty:
    st.error(f"No daily candle found for {expiry:%d-%b-%Y}. Tick 'Override expiry date' and pick "
             "a trading day.")
    st.stop()

# tape
c = st.columns(4)
for col, (nm, ytk) in zip(c[:3], [("NIFTY 50", "^NSEI"), ("NIFTY BANK", "^NSEBANK"),
                                  ("INDIA VIX", "^INDIAVIX")]):
    q = nse_index_quote(nm) if price_src != "Yahoo only" else None
    q = q or yahoo_index(ytk)
    if q and q.get("last") is not None:
        col.metric(nm, f"{float(q['last']):,.2f}",
                   f"{float(q['pchange']):+.2f}%" if q.get("pchange") is not None else None)
    else:
        col.metric(nm, "—")
c[3].metric("Reference expiry", f"{expiry:%d-%b-%Y}",
            f"{len(board)} stocks • {'LIVE' if market_open else 'CLOSED'}")

st.info(f"**Universe:** {uni_note} ({len(universe)} symbols) • **Prices:** {price_note} • "
        f"**Levels:** daily candle HIGH/LOW of {expiry:%d %b %Y} • "
        f"updated {now_ist:%H:%M:%S} IST • near-cross band {near_pct:.1f}%")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_all, tab_n50, tab_bnf, tab_uni = st.tabs(
    [f"🌐 ALL F&O ({len(board)})", f"🏆 NIFTY 50 ({int(board['N50'].sum())})",
     f"🏦 BANK NIFTY ({int(board['BNF'].sum())})", "📜 UNIVERSE"])


def counters(df):
    for col, s in zip(st.columns(5), STATUS_ORDER):
        n = int((df["STATUS"] == s).sum())
        col.markdown(f"<div style='background:{STATUS_BG[s]};padding:10px;border-radius:10px;"
                     f"text-align:center;color:#fff'><div style='font-size:26px;font-weight:700'>"
                     f"{n}</div><div style='font-size:12px'>{s}</div></div>",
                     unsafe_allow_html=True)


def style_table(df):
    view = df[["SYMBOL", "SECTOR", "LTP", "CHG", "%CHG", "EXP_LOW", "EXP_HIGH",
               "%FROM_LOW", "%FROM_HIGH", "RANGE_POS", "STATUS"]].copy()
    sty = view.style
    cell = getattr(sty, "map", None) or sty.applymap
    sty = cell(lambda v: f"background-color:{STATUS_BG.get(v,'')};color:#fff;font-weight:600",
               subset=["STATUS"])
    cell = getattr(sty, "map", None) or sty.applymap
    return (cell(lambda v: "color:#16a34a" if pd.notna(v) and v > 0 else
                           ("color:#dc2626" if pd.notna(v) and v < 0 else ""),
                 subset=["CHG", "%CHG"])
            .format({"LTP": "{:,.2f}", "CHG": "{:+,.2f}", "%CHG": "{:+.2f}%",
                     "EXP_LOW": "{:,.2f}", "EXP_HIGH": "{:,.2f}", "%FROM_LOW": "{:+.2f}%",
                     "%FROM_HIGH": "{:+.2f}%", "RANGE_POS": "{:.0f}%"}))


def pick_row(df, key):
    try:
        ev = st.dataframe(style_table(df), use_container_width=True, hide_index=True,
                          height=520, on_select="rerun", selection_mode="single-row", key=key)
        rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
        if rows:
            return df.iloc[rows[0]]["SYMBOL"]
    except Exception:
        st.dataframe(style_table(df), use_container_width=True, hide_index=True, height=520)
    return None


def render_group(df, key):
    if df.empty:
        st.caption("No stocks in this group — add a GROUPS column (N50 / BNF tags) to your CSV.")
        return None
    counters(df)
    st.write("")
    mode = st.radio("View", ["🔎 Scanner", "📋 Category lists"], horizontal=True, key=f"vm_{key}")
    f1, f2, f3 = st.columns([2, 2, 3])
    sec = f1.selectbox("Sector", ["All"] + sorted([s for s in df["SECTOR"].dropna().unique() if s]),
                       key=f"sec_{key}")
    stat = f2.multiselect("Status", STATUS_ORDER, default=STATUS_ORDER, key=f"st_{key}")
    q = f3.text_input("Search symbol", key=f"q_{key}").strip().upper()

    d = df.copy()
    if sec != "All":
        d = d[d["SECTOR"] == sec]
    if stat:
        d = d[d["STATUS"].isin(stat)]
    if q:
        d = d[d["SYMBOL"].str.contains(q)]

    if mode == "🔎 Scanner":
        d = d.sort_values(["STATUS", "RANGE_POS"], ascending=[True, False])
        st.caption(f"{len(d)} stocks • click a row to open the deep-dive below")
        picked = pick_row(d, f"tbl_{key}")
        st.download_button("⬇️ Export CSV", d.to_csv(index=False).encode(),
                           f"fno_levels_{expiry:%Y%m%d}_{key}.csv", key=f"dl_{key}")
        return picked
    for s in STATUS_ORDER:
        sub = d[d["STATUS"] == s].sort_values("RANGE_POS", ascending=(s in STATUS_ORDER[3:]))
        with st.expander(f"{s} — {len(sub)} stocks", expanded=s != "⚪ INSIDE"):
            if sub.empty:
                st.caption("none")
            else:
                st.dataframe(style_table(sub), use_container_width=True, hide_index=True,
                             height=min(420, 40 + 35 * len(sub)))
    return None


selected = None
with tab_all:
    selected = render_group(board, "all") or selected
with tab_n50:
    selected = render_group(board[board["N50"]], "n50") or selected
with tab_bnf:
    selected = render_group(board[board["BNF"]], "bnf") or selected
with tab_uni:
    st.caption(f"{uni_note} — {len(universe)} symbols")
    st.dataframe(universe, use_container_width=True, hide_index=True, height=600)
    st.download_button("⬇️ Download universe CSV", universe.to_csv(index=False).encode(),
                       "fno_universe.csv")

# ----------------------------------------------------------------------------
# DEEP DIVE
# ----------------------------------------------------------------------------
st.divider()
st.markdown("### 🔬 Stock deep-dive")
syms = sorted(board["SYMBOL"].tolist())
sym = st.selectbox("Stock", syms, index=syms.index(selected) if selected in syms else 0)

df = hist.get(sym)
if df is None or df.empty:
    st.warning("No history for this symbol.")
    st.stop()

row = board[board["SYMBOL"] == sym].iloc[0]
close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
ltp = float(row["LTP"])
r14 = rsi(close).iloc[-1]
a14 = atr(high, low, close).iloc[-1]
adx14, pdi, mdi = adx(high, low, close)
adx_v, pdi_v, mdi_v = adx14.iloc[-1], pdi.iloc[-1], mdi.iloc[-1]
e20, e50, e200 = ema(close, 20).iloc[-1], ema(close, 50).iloc[-1], ema(close, 200).iloc[-1]
ml, msig, _ = macd(close)
vwap20 = rolling_vwap(df).iloc[-1]
vwap_day = session_vwap(load_intraday(sym))
roc = lambda n: (ltp / close.iloc[-(n + 1)] - 1) * 100 if len(close) > n + 1 else np.nan
w52h, w52l = float(max(high.max(), ltp)), float(min(low.min(), ltp))
avgvol20 = vol.rolling(20).mean().iloc[-1]
fun = load_fundamentals(sym)

trend = ("🟢 Strong Uptrend" if ltp > e20 > e50 > e200 else
         "🔴 Strong Downtrend" if ltp < e20 < e50 < e200 else
         "🟢 Uptrend" if ltp > e50 > e200 else
         "🔴 Downtrend" if ltp < e50 < e200 else "🟡 Sideways / Mixed")

score = sum([1 if ltp > e20 else -1, 1 if ltp > e50 else -1, 1 if ltp > e200 else -1,
             1 if ml.iloc[-1] > msig.iloc[-1] else -1,
             1 if r14 > 55 else (-1 if r14 < 45 else 0),
             1 if pdi_v > mdi_v else -1,
             2 if row["STATUS"] == "🟢 ABOVE HIGH" else (-2 if row["STATUS"] == "🔴 BELOW LOW" else 0),
             (1 if ltp > vwap_day else -1) if vwap_day else 0])
signal = ("🟢 STRONG BUY BIAS" if score >= 5 else "🟢 BULLISH" if score >= 2 else
          "🔴 STRONG SELL BIAS" if score <= -5 else "🔴 BEARISH" if score <= -2 else "⚪ NEUTRAL")
momentum = ("🚀 Very Strong" if adx_v > 40 and pdi_v > mdi_v else
            "📉 Very Weak" if adx_v > 40 and mdi_v > pdi_v else
            "💪 Trending" if adx_v > 25 else "😴 Range / No trend")

h = st.columns(5)
h[0].metric(sym, f"₹{ltp:,.2f}", f"{row['%CHG']:+.2f}%" if pd.notna(row["%CHG"]) else None)
h[1].metric("Expiry HIGH", f"{row['EXP_HIGH']:,.2f}", f"{row['%FROM_HIGH']:+.2f}%")
h[2].metric("Expiry LOW", f"{row['EXP_LOW']:,.2f}", f"{row['%FROM_LOW']:+.2f}%")
h[3].metric("Status", str(row["STATUS"]))
h[4].metric("Signal", signal, f"score {score:+d}")


def fmt(v, s="", d=2):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}{s}"


st.dataframe(pd.DataFrame([
    ("RSI (14)", fmt(r14), "🟢 Overbought" if r14 > 70 else "🔴 Oversold" if r14 < 30 else "⚪ Neutral"),
    ("VWAP today (5m)", fmt(vwap_day), "🟢 Above" if vwap_day and ltp > vwap_day else "🔴 Below" if vwap_day else "—"),
    ("VWAP 20D", fmt(vwap20), "🟢 Above" if ltp > vwap20 else "🔴 Below"),
    ("EMA 20/50/200", f"{e20:,.1f} / {e50:,.1f} / {e200:,.1f}", trend),
    ("ATR (14)", fmt(a14), f"{a14/ltp*100:.2f}% of price"),
    ("ADX (14)", fmt(adx_v), momentum),
    ("+DI / −DI", f"{pdi_v:.1f} / {mdi_v:.1f}", "🟢 Bulls" if pdi_v > mdi_v else "🔴 Bears"),
    ("MACD", f"{ml.iloc[-1]:.2f} vs {msig.iloc[-1]:.2f}",
     "🟢 Bullish" if ml.iloc[-1] > msig.iloc[-1] else "🔴 Bearish"),
    ("Momentum 1W/1M/3M", f"{fmt(roc(5),'%')} / {fmt(roc(20),'%')} / {fmt(roc(60),'%')}", ""),
    ("52W High / Low", f"{w52h:,.1f} / {w52l:,.1f}", f"{(ltp-w52h)/w52h*100:+.1f}% from high"),
    ("Range position", f"{row['RANGE_POS']:.0f}%", "within the expiry-day candle"),
    ("Volume vs 20D avg", fmt(vol.iloc[-1]/avgvol20, "x") if avgvol20 else "—",
     "🔥 High" if avgvol20 and vol.iloc[-1] > 1.5 * avgvol20 else "normal"),
    ("PE / Fwd PE", f"{fmt(fun['PE'])} / {fmt(fun['FWD_PE'])}", ""),
    ("EPS (TTM)", fmt(fun["EPS"]), ""),
    ("P/B • Beta", f"{fmt(fun['PB'])} • {fmt(fun['BETA'])}", ""),
    ("Market cap", fmt(fun["MCAP"] / 1e7, " Cr", 0) if fun["MCAP"] else "—", fun.get("INDUSTRY") or ""),
], columns=["Metric", "Value", "Read"]), use_container_width=True, hide_index=True, height=580)

if PLOTLY:
    d = df.tail(252).copy()
    d["E20"], d["E50"], d["E200"] = ema(d["Close"], 20), ema(d["Close"], 50), ema(d["Close"], 200)
    d["VW20"], d["RSI"] = rolling_vwap(d), rsi(d["Close"])
    ad, pd_, md_ = adx(d["High"], d["Low"], d["Close"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.55, 0.13, 0.16, 0.16],
                        subplot_titles=(f"{sym} — 1 year daily", "Volume", "RSI (14)", "ADX / DI"))
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                 close=d["Close"], name="Price"), row=1, col=1)
    for col, clr in (("E20", "#3b82f6"), ("E50", "#f59e0b"), ("E200", "#ef4444"), ("VW20", "#a855f7")):
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(width=1.2, color=clr)), row=1, col=1)
    fig.add_hline(y=row["EXP_HIGH"], line=dict(color="#16a34a", dash="dash"),
                  annotation_text=f"Expiry High {row['EXP_HIGH']:,.1f}", row=1, col=1)
    fig.add_hline(y=row["EXP_LOW"], line=dict(color="#dc2626", dash="dash"),
                  annotation_text=f"Expiry Low {row['EXP_LOW']:,.1f}", row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Vol", marker_color="#64748b"), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(color="#0ea5e9")), row=3, col=1)
    fig.add_hline(y=70, line=dict(color="#16a34a", dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="#dc2626", dash="dot"), row=3, col=1)
    for series, nm, clr in ((ad, "ADX", "#eab308"), (pd_, "+DI", "#16a34a"), (md_, "−DI", "#dc2626")):
        fig.add_trace(go.Scatter(x=d.index, y=series, name=nm, line=dict(color=clr, width=1.2)), row=4, col=1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(orientation="h", y=1.02), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("pip install plotly for the interactive chart")
    st.line_chart(df["Close"].tail(252))

st.caption("Sources: NSE India when reachable, otherwise Yahoo Finance via yfinance. "
           "Informational only — not investment advice.")
