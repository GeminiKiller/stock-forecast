#!/usr/bin/env python3
"""
Stock Forecast Pipeline — Modular Orchestrator
Runs each model in its own subprocess to avoid dependency conflicts.

Usage:
  python forecast.py TARGET [HELPER] [OPTIONS]
  python forecast.py TARGET [HELPER] --data-only   # Skip models + chart, just download data

Examples:
  python forecast.py APLD
  python forecast.py APLD NVDA
  python forecast.py APLD NVDA --data-only --horizon 12
  python forecast.py TSLA SPY --start 2025-01-01 --horizon 20
  python forecast.py BTC-USD --no-timesfm --no-lstm
"""
import argparse
import sys
import os
import json
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ==========================================
# PARSE ARGS
# ==========================================
parser = argparse.ArgumentParser(description="Stock Forecast Pipeline")
parser.add_argument("target",        type=str, help="Target ticker (e.g. APLD, TSLA, BTC-USD)")
parser.add_argument("helper",        type=str, nargs="?", default="NVDA",
                    help="Helper/macro ticker (default: NVDA)")
parser.add_argument("--start",       type=str, default=None,
                    help="Start date (default: auto from data-range)")
parser.add_argument("--horizon",     type=int, default=12,
                    help="Forecast horizon in days (default: 12)")
parser.add_argument("--data-range",  type=str, default="3m",
                    choices=["3m","6m","1y","2y","5y","max"],
                    help="Historical data range (default: 3m)")
parser.add_argument("--chart-type",  type=str, default="candle",
                    choices=["candle","line"],
                    help="Chart type: candle or line (default: candle)")
parser.add_argument("--no-timesfm",  action="store_true", help="Skip TimesFM model")
parser.add_argument("--no-chronos",  action="store_true", help="Skip Chronos-2 model")
parser.add_argument("--no-lstm",     action="store_true", help="Skip LSTM model")
parser.add_argument("--data-only",   action="store_true",
                    help="Only download data + indicators (skip models and chart). Outputs JSON metadata.")
parser.add_argument("--output",      type=str, default=None,
                    help="Output chart path (default: TARGET_forecast.png)")

args = parser.parse_args()
TARGET    = args.target.upper()
HELPER    = args.helper.upper()
HORIZON   = args.horizon
SKIP_TFM  = args.no_timesfm
SKIP_CHR  = args.no_chronos
SKIP_LSTM = args.no_lstm
DATA_ONLY = args.data_only
CHART_TYPE = args.chart_type

# Compute start date from data-range if not explicitly set
range_map = {"3m": "3mo", "6m": "6mo", "1y": "1y", "2y": "2y", "5y": "5y", "max": "max"}
if args.start:
    START = args.start
    PERIOD = None
else:
    START = range_map.get(args.data_range, "1y")
    PERIOD = START

PYTHON     = "/opt/homebrew/bin/python3.11"
WORKSPACE  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(WORKSPACE, "models")
DATA_CSV   = os.path.join(WORKSPACE, "_forecast_data.csv")
if args.output:
    OUT_CHART = args.output
else:
    OUT_CHART = os.path.join(WORKSPACE, f"{TARGET}_{HELPER}_{args.horizon}_{args.data_range}_{args.chart_type}_forecast.png")

# ==========================================
# TECHNICAL INDICATORS (pure numpy/pandas)
# ==========================================

def compute_rsi(series, length=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_bbands(series, length=20, std=2.0):
    sma = series.rolling(window=length).mean()
    rstd = series.rolling(window=length).std()
    return sma + rstd * std, sma, sma - rstd * std

def compute_ema(series, length=50):
    return series.ewm(span=length, adjust=False).mean()

def get_td_setup(arr):
    setup = np.zeros(len(arr))
    count = 0
    for i in range(4, len(arr)):
        count = count + 1 if arr[i] > arr[i-4] else 0
        setup[i] = count
    return setup

# ==========================================
# 1. DATA ACQUISITION
# ==========================================
print(f"\n{'━'*40}")
print(f"  {TARGET} Forecast (Helper: {HELPER})")
print(f"{'━'*40}")
print(f"\n--- Step 1: Downloading {TARGET} + {HELPER} from {START} ---")
print(f"[STATUS:data:running]", flush=True)

dl_kwargs = {"auto_adjust": True, "progress": False}
if PERIOD:
    dl_kwargs["period"] = PERIOD
else:
    dl_kwargs["start"] = START
raw = yf.download([TARGET, HELPER], **dl_kwargs)

df = pd.DataFrame(index=raw.index)
close_col  = ('Close', TARGET) if ('Close', TARGET) in raw.columns else f'Close_{TARGET}'
helper_col = ('Close', HELPER) if ('Close', HELPER) in raw.columns else f'Close_{HELPER}'
df['close']  = raw[close_col].ffill()
df['helper'] = raw[helper_col].ffill()
df.dropna(inplace=True)

print(f"[STATUS:data:done]", flush=True)
print(f"[STATUS:indicators:running]", flush=True)
df['RSI_14'] = compute_rsi(df['close'], 14)
df['BBU_20'], df['BBM_20'], df['BBL_20'] = compute_bbands(df['close'], 20, 2.0)
df['EMA_50'] = compute_ema(df['close'], 50)
df['td_count'] = get_td_setup(df['close'].values)
print(f"[STATUS:indicators:done]", flush=True)

target_vals = df['close'].values.astype(np.float32)
helper_vals = df['helper'].values.astype(np.float32)
current_price = float(target_vals[-1])

print(f"  Data: {len(target_vals)} rows | Current {TARGET}: ${current_price:.2f}")

# Save CSV for model inference
df[['close', 'helper']].to_csv(DATA_CSV)

# ── DATA-ONLY MODE: output JSON and exit ──
if DATA_ONLY:
    print(f"[STATUS:data_ready:done]", flush=True)
    # Output metadata as JSON to stdout for the Flask process to consume
    metadata = {
        "target": TARGET,
        "helper": HELPER,
        "horizon": HORIZON,
        "data_range": args.data_range,
        "chart_type": CHART_TYPE,
        "current_price": current_price,
        "data_rows": len(target_vals),
        "csv_path": DATA_CSV,
        "chart_path": OUT_CHART,
        "data_range_raw": START if PERIOD else args.start,
        # Technical indicators for parsing
        "rsi": float(df['RSI_14'].iloc[-1]),
        "ema_50": float(df['EMA_50'].iloc[-1]),
        "bb_upper": float(df['BBU_20'].iloc[-1]),
        "td_count": int(df['td_count'].iloc[-1]),
        "rsi_label": "[OVERBOUGHT]" if float(df['RSI_14'].iloc[-1]) > 70 else "[OVERSOLD]" if float(df['RSI_14'].iloc[-1]) < 30 else "[STABLE]",
        "ema_label": "BULLISH" if current_price > float(df['EMA_50'].iloc[-1]) else "BEARISH",
        "td_label": "!!! SELL FLIP !!!" if int(df['td_count'].iloc[-1]) >= 8 else "Trend Continuing",
        "vol_label": "OVEREXTENDED" if current_price > float(df['BBU_20'].iloc[-1]) else "NORMAL",
    }
    print(f"[METADATA_JSON]{json.dumps(metadata)}[/METADATA_JSON]", flush=True)
    # Print verification log
    print(f"\n{'='*60}")
    print(f"  DATA-ONLY: {TARGET}  (macro: {HELPER},  current: ${current_price:.2f})")
    print(f"  CSV: {DATA_CSV}")
    print(f"  Rows: {len(target_vals)}")
    print(f"{'='*60}\n")
    sys.exit(0)

# ==========================================
# 2. RUN MODELS (subprocess mode — original behavior)
# ==========================================

def run_subprocess(name, script, extra_args=None):
    """Run a model subprocess and return parsed JSON result."""
    path = os.path.join(MODELS_DIR, script)
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: script not found")
        return None
    cmd = [PYTHON, path, DATA_CSV, str(HORIZON)] + (extra_args or [])
    print(f"\n--- Running {name} ---")
    print(f"[STATUS:{name}:running]", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  ✗ {name} exited {r.returncode}")
            print(f"[STATUS:{name}:failed]", flush=True)
            if r.stderr:
                for line in r.stderr.strip().split('\n'):
                    if line.strip():
                        print(f"    {line.strip()}")
            return None
        stdout = r.stdout.strip()
        json_start = stdout.find('{')
        if json_start == -1:
            print(f"  ✗ {name}: no JSON output")
            print(f"[STATUS:{name}:failed]", flush=True)
            return None
        data = json.loads(stdout[json_start:])
        if "error" in data:
            print(f"  ✗ {name} error: {data['error']}")
            print(f"[STATUS:{name}:failed]", flush=True)
            return None
        print(f"  ✓ {name} done")
        print(f"[STATUS:{name}:done]", flush=True)
        return data
    except subprocess.TimeoutExpired:
        print(f"  ✗ {name} timed out")
        return None
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return None

tfm_final = tfm_lo = tfm_hi = None
chronos_median = chronos_lo = chronos_hi = None
lstm_final = lstm_lo = lstm_hi = None

if not SKIP_TFM:
    res = run_subprocess("TimesFM 2.5", "timesfm_model.py")
    if res and "predictions" in res:
        tfm_final = np.array(res["predictions"])

if not SKIP_CHR or not SKIP_LSTM:
    extra = []
    if SKIP_CHR: extra.append("--no-chronos")
    if SKIP_LSTM: extra.append("--no-lstm")
    res = run_subprocess("Chronos-2 + LSTM", "chronos_lstm_model.py", extra)
    if res:
        if "chronos" in res:
            chronos_median = np.array(res["chronos"])
            chronos_lo = np.array(res.get("chronos_lower", res["chronos"]))
            chronos_hi = np.array(res.get("chronos_upper", res["chronos"]))
        if "lstm" in res:
            lstm_final = np.array(res["lstm"])
            lstm_lo = np.array(res.get("lstm_lower", res["lstm"]))
            lstm_hi = np.array(res.get("lstm_upper", res["lstm"]))

# ==========================================
# 3. CONFORMAL CALIBRATION
# ==========================================
print("\n--- Conformal Calibration ---")
print(f"[STATUS:conformal:running]", flush=True)
from mapie.regression import SplitConformalRegressor
from sklearn.linear_model import LinearRegression

# Collect all available model predictions for ensemble median
all_preds = []
if tfm_final is not None:      all_preds.append(tfm_final)
if chronos_median is not None: all_preds.append(chronos_median)
if lstm_final is not None:     all_preds.append(lstm_final)

# Default: no band
band_lo = band_hi = None

if len(all_preds) >= 2:
    # Use ensemble median as center, conformal interval for width
    ensemble_median = np.median(all_preds, axis=0)
    # Calibrate residual width from last 30 days using MAD
    recent = target_vals[-30:]
    resid_scale = np.median(np.abs(recent - np.median(recent))) * 1.5
    band_lo = ensemble_median - resid_scale
    band_hi = ensemble_median + resid_scale
else:
    # Fallback to original linear regression band
    X_cal = np.arange(len(target_vals)-30, len(target_vals)).reshape(-1, 1)
    y_cal = target_vals[-30:].ravel()
    mapie = SplitConformalRegressor(
        estimator=LinearRegression().fit(X_cal, y_cal),
        confidence_level=0.9, prefit=True
    )
    mapie.conformalize(X_cal, y_cal)
    _, y_pis = mapie.predict_interval(
        np.arange(len(target_vals), len(target_vals)+HORIZON).reshape(-1, 1)
    )
    band_lo = y_pis[:, 0, 0]
    band_hi = y_pis[:, 1, 0]
print("  ✓ Calibration done")
print(f"[STATUS:conformal:done]", flush=True)

# ==========================================
# 4. CHART (candlestick + confidence bands)
# ==========================================
print("[STATUS:chart:running]", flush=True)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Fetch OHLC data for chart
    ohlc_dl_kwargs = {"auto_adjust": True, "progress": False}
    if PERIOD:
        ohlc_dl_kwargs["period"] = PERIOD
    else:
        ohlc_dl_kwargs["start"] = START
    ohlc_raw = yf.download([TARGET], **ohlc_dl_kwargs)
    if len(ohlc_raw) == 0:
        raise Exception("No OHLC data")

    # Build OHLC DataFrame — handle multi-index columns from yfinance
    ohlc = pd.DataFrame(index=ohlc_raw.index)
    for col in ['Open', 'High', 'Low', 'Close']:
        raw_vals = ohlc_raw[col]
        if isinstance(raw_vals, pd.DataFrame):
            ohlc[col] = raw_vals.iloc[:, 0].values
        else:
            ohlc[col] = raw_vals.values
    ohlc.dropna(inplace=True)

    SHOW_BARS = min(60, len(ohlc))
    ohlc_dates = ohlc.index[-SHOW_BARS:]
    ohlc = ohlc.tail(SHOW_BARS).copy()

    # Align TD data to same bars
    td_df = df.tail(SHOW_BARS).copy()

    # Extract volume from ohlc_raw
    vol_raw = ohlc_raw['Volume']
    if isinstance(vol_raw, pd.DataFrame):
        vol_vals = vol_raw.iloc[:, 0].values.astype(float)
    else:
        vol_vals = vol_raw.values.astype(float)
    vol_vals = vol_vals[-SHOW_BARS:]

    # ── Figure: 3 panels (price, volume, RSI) ──
    fig = plt.figure(figsize=(16, 10), facecolor='#0c0c0c')
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.02,
                          left=0.07, right=0.97, top=0.94, bottom=0.10)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#0c0c0c')
        ax.tick_params(colors='#555', labelsize=8)
        ax.grid(True, alpha=0.08, color='#333')
        for sp in ax.spines.values():
            sp.set_color('#333')

    n_hist = len(ohlc)
    x_hist = np.arange(n_hist)
    x_pred = np.arange(n_hist, n_hist + HORIZON)

    # ── Price history: candlestick or line ──
    ohlc_arr = ohlc[['Open','High','Low','Close']].values
    if CHART_TYPE == 'candle':
        for j in range(n_hist):
            o, h, l, c = ohlc_arr[j]
            color = '#00ff88' if c >= o else '#ff4444'
            body_bottom = min(o, c)
            body_height = max(abs(c - o), 0.02)
            ax1.bar(j, body_height, bottom=body_bottom, color=color, width=0.7, alpha=0.95, zorder=2)
            ax1.plot([j, j], [l, h], color=color, linewidth=0.7, alpha=0.6, zorder=1)
    else:
        # Line chart with close price + optional high/low fill
        close_line = ohlc_arr[:, 3]
        ax1.plot(x_hist, close_line, color="#00ff88", linewidth=1.2, zorder=2, label="Close")
        ax1.fill_between(x_hist, ohlc_arr[:, 2], ohlc_arr[:, 1], color="#00ff88", alpha=0.08, zorder=1, label="High/Low range")
        # TD annotations on line
        td_vals_line = td_df['td_count'].values.astype(int)
        for j in range(n_hist):
            count = td_vals_line[j] if j < len(td_vals_line) else 0
            if count == 0: continue
            y_val = close_line[j]
            if count == 9:
                lbl, color, sz = '⚠9', '#ff4444', 8
                y_off = (ohlc_arr[j, 1] - ohlc_arr[j, 2]) * 0.4
            elif count > 9:
                lbl, color, sz = str(count), '#ff4444', 7
                y_off = (ohlc_arr[j, 1] - ohlc_arr[j, 2]) * 0.4
            else:
                lbl, color, sz = str(count), '#00ff88', 7
                y_off = -(ohlc_arr[j, 1] - ohlc_arr[j, 2]) * 0.6
            ax1.text(j, y_val + y_off, lbl, ha='center', va='center', fontsize=sz,
                     color=color, fontweight='bold' if count >= 9 else 'normal', alpha=0.85, zorder=3)

    # ── Volume bars (ax2) ──
    max_vol = max(vol_vals) if len(vol_vals) > 0 else 1
    for j in range(n_hist):
        v = vol_vals[j] if j < len(vol_vals) else 0
        c = ohlc_arr[j, 3]
        o = ohlc_arr[j, 0]
        color = '#00ff88' if c >= o else '#ff4444'
        bar_h = (v / max_vol) * 0.8 if max_vol > 0 else 0
        ax2.bar(j, bar_h, color=color, width=0.7, alpha=0.5, zorder=2, bottom=0.05)
    # Volume moving average line
    if len(vol_vals) >= 20:
        vol_ma = pd.Series(vol_vals).rolling(20).mean().values
        vol_ma_norm = [(v / max_vol) * 0.8 if max_vol > 0 else 0 for v in vol_ma]
        ax2.plot(x_hist, vol_ma_norm, color="#ff6b35", linewidth=1, alpha=0.8, label="Vol MA20")
    # Volume profile (horizontal histogram on the right)
    n_bins = 10
    price_bins = np.linspace(float(ohlc_arr[:, 2].min()), float(ohlc_arr[:, 1].max()), n_bins + 1)
    vol_profile = np.zeros(n_bins)
    for j in range(n_hist):
        c = ohlc_arr[j, 3]
        v = vol_vals[j] if j < len(vol_vals) else 0
        for b in range(n_bins):
            if price_bins[b] <= c < price_bins[b + 1]:
                vol_profile[b] += v
                break
    max_profile = max(vol_profile) if max(vol_profile) > 0 else 1
    for b in range(n_bins):
        bar_w = (vol_profile[b] / max_profile) * 0.15
        y_center = (price_bins[b] + price_bins[b + 1]) / 2
        ax1.barh(y_center, bar_w, height=(price_bins[1] - price_bins[0]) * 0.9,
                 color='#333', alpha=0.4, zorder=0, left=n_hist + HORIZON * 0.3)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Vol", color='#777', fontsize=9)
    ax2.set_yticks([0.2, 0.5, 0.8])
    ax2.set_yticklabels(["20%", "50%", "80%"], color='#555', fontsize=7)
    ax2.legend(loc="upper right", fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#aaa')

    # ── Prediction zone ──
    ax1.axvspan(n_hist - 0.5, n_hist + HORIZON + 0.5, color='#151520', alpha=0.6, zorder=0)
    ax1.axvline(n_hist - 0.5, color='#333', linewidth=1)
    ax2.axvspan(n_hist - 0.5, n_hist + HORIZON + 0.5, color='#151520', alpha=0.4)
    ax3.axvspan(n_hist - 0.5, n_hist + HORIZON + 0.5, color='#151520', alpha=0.4)

    # ── Confidence bands ──
    if chronos_median is not None:
        ax1.fill_between(x_pred, chronos_lo, chronos_hi, color="#00d4ff", alpha=0.12, label="Chronos 90% CI")
    if lstm_final is not None:
        ax1.fill_between(x_pred, lstm_lo, lstm_hi, color="#2ca02c", alpha=0.10, label="LSTM 90% CI")
    if band_lo is not None:
        ax1.fill_between(x_pred, band_lo, band_hi, color="#ff6b35", alpha=0.08, label="Ensemble 90% Band")

    # ── Model lines ──
    if tfm_final is not None:
        ax1.plot(x_pred, tfm_final, color="#ff7f0e", label="TimesFM", linestyle="--", marker='x', ms=3)
    if chronos_median is not None:
        ax1.plot(x_pred, chronos_median, color="#00d4ff", label="Chronos-2", linestyle="--", marker='o', ms=3)
    if lstm_final is not None:
        ax1.plot(x_pred, lstm_final, color="#00ff88", label="LSTM", linestyle="-.", lw=1.5)
    if len(all_preds) >= 2:
        ax1.plot(x_pred, ensemble_median, color="#ff6b35", label="Ensemble", linestyle="-", lw=2)

    # ── TD annotations (candlestick mode only — line mode handles inline) ──
    if CHART_TYPE == 'candle':
        td_vals = td_df['td_count'].values.astype(int)
        for j in range(n_hist):
            count = td_vals[j] if j < len(td_vals) else 0
            if count == 0: continue
            _, h_j, l_j, c_j = ohlc_arr[j]
            if count == 9:
                lbl, color, sz = '⚠9', '#ff4444', 8
                y_pos = h_j + (h_j - l_j) * 0.4
            elif count > 9:
                lbl, color, sz = str(count), '#ff4444', 7
                y_pos = h_j + (h_j - l_j) * 0.4
            else:
                lbl, color, sz = str(count), '#00ff88', 7
                y_pos = l_j - (h_j - l_j) * 0.6
            ax1.text(j, y_pos, lbl, ha='center', va='center', fontsize=sz,
                 color=color, fontweight='bold' if count >= 9 else 'normal', alpha=0.85, zorder=3)

    # ── Axes limits ──
    price_min = float(ohlc_arr[:, 2].min())
    price_max = float(ohlc_arr[:, 1].max())
    yr = price_max - price_min
    ax1.set_ylim(price_min - yr * 0.2, price_max + yr * 0.3)
    ax1.set_xlim(-1.5, n_hist + HORIZON + 1)
    ax2.set_xlim(-1.5, n_hist + HORIZON + 1)
    ax3.set_xlim(-1.5, n_hist + HORIZON + 1)

    ax1.set_title(f"  {TARGET}  |  {HELPER}  |  H={HORIZON}",
                  fontsize=13, color='white', pad=8, loc='left')
    ax1.legend(loc="upper left", fontsize=8, facecolor='#111', edgecolor='#333', labelcolor='#aaa')
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])

    # ── RSI (ax3) ──
    rsi_vals = df['RSI_14'].iloc[-SHOW_BARS:].values
    ax3.plot(x_hist, rsi_vals, color="#aa66ff", linewidth=1.2)
    ax3.axhline(70, color='#ff4444', linestyle=':', lw=0.7, alpha=0.5)
    ax3.axhline(30, color='#00ff88', linestyle=':', lw=0.7, alpha=0.5)
    ax3.fill_between(x_hist, 30, 70, color='#222', alpha=0.2)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI", color='#777', fontsize=9)

    # ── Date labels on RSI x-axis ──
    n_ticks = min(7, n_hist)
    step = max(1, n_hist // n_ticks)
    tick_positions = list(range(0, n_hist, step))
    if n_hist - 1 not in tick_positions:
        tick_positions.append(n_hist - 1)
    tick_labels = []
    for i in tick_positions:
        src_idx = len(ohlc_dates) - n_hist + i
        if 0 <= src_idx < len(ohlc_dates):
            tick_labels.append(ohlc_dates[src_idx].strftime('%b %d'))
        else:
            tick_labels.append('')
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels, rotation=40, ha='right', fontsize=8, color='#888')

    plt.savefig(OUT_CHART, dpi=110, facecolor='#0c0c0c', edgecolor='none')
    print(f"  ✓ Chart saved: {OUT_CHART}")
    print("[STATUS:chart:done]", flush=True)
except Exception as e:
    print(f"  Plot skipped: {e}")

# ==========================================
# 5. VERIFICATION LOG
# ==========================================
print(f"\n{'='*60}")
print(f"  FORECAST: {TARGET}  (macro: {HELPER},  current: ${current_price:.2f})")
print(f"{'='*60}")

rsi_val  = float(df['RSI_14'].iloc[-1])
ema_50   = float(df['EMA_50'].iloc[-1])
td_count = int(df['td_count'].iloc[-1])
bb_u     = float(df['BBU_20'].iloc[-1])

rsi_label = '[OVERBOUGHT]' if rsi_val > 70 else '[OVERSOLD]' if rsi_val < 30 else '[STABLE]'
print(f"| RSI (14)    : {rsi_val:6.2f} | {rsi_label}")
print(f"| EMA 50      : {'BULLISH' if current_price > ema_50 else 'BEARISH':>12} | (EMA: ${ema_50:.2f})")
print(f"| TD SETUP    : {td_count:>12} | {'!!! SELL FLIP !!!' if td_count >= 8 else 'Trend Continuing'}")
bb_label  = 'OVEREXTENDED' if current_price > bb_u else 'NORMAL'
print(f"| VOLATILITY  : {bb_label:>12} | (BB Upper: ${bb_u:.2f})")

print(f"\n{'─'*60}")
print(f"{'MODEL':<15} | {'TARGET':<12} | {'DELTA %':<10} | {'BIAS'}")
print(f"{'─'*60}")

predictions = []
if tfm_final is not None:
    p = float(tfm_final[-1]); d = ((p-current_price)/current_price)*100
    print(f"{'TimesFM 2.5':<15} | ${p:<11.2f} | {d:+9.2f}% | {'UP' if d>0 else 'DOWN'}")
    predictions.append(d)
if chronos_median is not None:
    p = float(chronos_median[-1]); d = ((p-current_price)/current_price)*100
    print(f"{'Chronos-2':<15} | ${p:<11.2f} | {d:+9.2f}% | {'UP' if d>0 else 'DOWN'}")
    predictions.append(d)
if lstm_final is not None:
    p = float(lstm_final[-1]); d = ((p-current_price)/current_price)*100
    print(f"{'Neural LSTM':<15} | ${p:<11.2f} | {d:+9.2f}% | {'UP' if d>0 else 'DOWN'}")
    predictions.append(d)

if predictions:
    avg_delta = np.mean(predictions)
    consensus = ("STRONG UP" if all(d>0 for d in predictions) else
                 "STRONG DOWN" if all(d<0 for d in predictions) else "CONTRADICTORY")
    print(f"{'─'*60}")
    print(f"  CONSENSUS : {consensus}")
    print(f"  AVG DELTA : {avg_delta:+.2f}%")
else:
    print("  [WARN] No predictions available.")
print(f"{'='*60}\n")

# Cleanup
if os.path.exists(DATA_CSV):
    os.remove(DATA_CSV)