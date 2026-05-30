#!/usr/bin/env python3
"""
Chart generation module — called from Flask after model inference.
Generates the candlestick/line chart with model prediction overlays.
"""
import os
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_forecast_chart(
    target, helper, horizon, data_range, chart_type,
    # Technical data
    rsi_val, ema_50, td_count, bb_upper, current_price,
    ema_label, rsi_label, td_label, vol_label,
    # Model predictions (all optional, may be None)
    tfm_predictions=None,
    chronos_median=None, chronos_lo=None, chronos_hi=None,
    lstm_final=None, lstm_lo=None, lstm_hi=None,
    band_lo=None, band_hi=None,
    ensemble_median=None,
    # Output
    output_path=None,
    csv_path=None,
):
    """Generate forecast chart with model overlays."""
    # Load data from CSV for OHLC/volume
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        return False

    target_vals = df['close'].values.astype(np.float32)

    # Determine period for OHLC download
    range_map = {"3m": "3mo", "6m": "6mo", "1y": "1y", "2y": "2y", "5y": "5y", "max": "max"}
    period = range_map.get(data_range, "1y")

    # Compute indicators
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

    df['RSI_14'] = compute_rsi(df['close'], 14)
    df['BBU_20'], df['BBM_20'], df['BBL_20'] = compute_bbands(df['close'], 20, 2.0)
    df['EMA_50'] = compute_ema(df['close'], 50)
    df['td_count'] = get_td_setup(df['close'].values)

    # Fetch OHLC data
    ohlc_raw = yf.download([target], period=period, auto_adjust=True, progress=False)
    if len(ohlc_raw) == 0:
        return False

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

    # Align TD data
    td_df = df.tail(SHOW_BARS).copy()

    # Volume
    vol_raw = ohlc_raw['Volume']
    if isinstance(vol_raw, pd.DataFrame):
        vol_vals = vol_raw.iloc[:, 0].values.astype(float)
    else:
        vol_vals = vol_raw.values.astype(float)
    vol_vals = vol_vals[-SHOW_BARS:]

    # ── Figure ──
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
    x_pred = np.arange(n_hist, n_hist + horizon)

    # ── Price history ──
    ohlc_arr = ohlc[['Open', 'High', 'Low', 'Close']].values
    if chart_type == 'candle':
        for j in range(n_hist):
            o, h, l, c = ohlc_arr[j]
            color = '#00ff88' if c >= o else '#ff4444'
            body_bottom = min(o, c)
            body_height = max(abs(c - o), 0.02)
            ax1.bar(j, body_height, bottom=body_bottom, color=color, width=0.7, alpha=0.95, zorder=2)
            ax1.plot([j, j], [l, h], color=color, linewidth=0.7, alpha=0.6, zorder=1)
    else:
        close_line = ohlc_arr[:, 3]
        ax1.plot(x_hist, close_line, color="#00ff88", linewidth=1.2, zorder=2, label="Close")
        ax1.fill_between(x_hist, ohlc_arr[:, 2], ohlc_arr[:, 1], color="#00ff88", alpha=0.08, zorder=1, label="High/Low range")
        td_vals_line = td_df['td_count'].values.astype(int)
        for j in range(n_hist):
            count = td_vals_line[j] if j < len(td_vals_line) else 0
            if count == 0:
                continue
            y_val = close_line[j]
            if count == 9:
                lbl, clr, sz = '⚠9', '#ff4444', 8
                y_off = (ohlc_arr[j, 1] - ohlc_arr[j, 2]) * 0.4
            elif count > 9:
                lbl, clr, sz = str(count), '#ff4444', 7
                y_off = (ohlc_arr[j, 1] - ohlc_arr[j, 2]) * 0.4
            else:
                lbl, clr, sz = str(count), '#00ff88', 7
                y_off = -(ohlc_arr[j, 1] - ohlc_arr[j, 2]) * 0.6
            ax1.text(j, y_val + y_off, lbl, ha='center', va='center', fontsize=sz,
                     color=clr, fontweight='bold' if count >= 9 else 'normal', alpha=0.85, zorder=3)

    # ── Volume bars ──
    max_vol = max(vol_vals) if len(vol_vals) > 0 else 1
    for j in range(n_hist):
        v = vol_vals[j] if j < len(vol_vals) else 0
        c = ohlc_arr[j, 3]
        o = ohlc_arr[j, 0]
        color = '#00ff88' if c >= o else '#ff4444'
        bar_h = (v / max_vol) * 0.8 if max_vol > 0 else 0
        ax2.bar(j, bar_h, color=color, width=0.7, alpha=0.5, zorder=2, bottom=0.05)
    if len(vol_vals) >= 20:
        vol_ma = pd.Series(vol_vals).rolling(20).mean().values
        vol_ma_norm = [(v / max_vol) * 0.8 if max_vol > 0 else 0 for v in vol_ma]
        ax2.plot(x_hist, vol_ma_norm, color="#ff6b35", linewidth=1, alpha=0.8, label="Vol MA20")
    # Volume profile
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
                 color='#333', alpha=0.4, zorder=0, left=n_hist + horizon * 0.3)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Vol", color='#777', fontsize=9)
    ax2.set_yticks([0.2, 0.5, 0.8])
    ax2.set_yticklabels(["20%", "50%", "80%"], color='#555', fontsize=7)
    ax2.legend(loc="upper right", fontsize=7, facecolor='#111', edgecolor='#333', labelcolor='#aaa')

    # ── Prediction zone ──
    ax1.axvspan(n_hist - 0.5, n_hist + horizon + 0.5, color='#151520', alpha=0.6, zorder=0)
    ax1.axvline(n_hist - 0.5, color='#333', linewidth=1)
    ax2.axvspan(n_hist - 0.5, n_hist + horizon + 0.5, color='#151520', alpha=0.4)
    ax3.axvspan(n_hist - 0.5, n_hist + horizon + 0.5, color='#151520', alpha=0.4)

    # ── Confidence bands ──
    if chronos_median is not None and chronos_lo is not None:
        ax1.fill_between(x_pred, chronos_lo, chronos_hi, color="#00d4ff", alpha=0.12, label="Chronos 90% CI")
    if lstm_final is not None and lstm_lo is not None:
        ax1.fill_between(x_pred, lstm_lo, lstm_hi, color="#2ca02c", alpha=0.10, label="LSTM 90% CI")
    if band_lo is not None:
        ax1.fill_between(x_pred, band_lo, band_hi, color="#ff6b35", alpha=0.08, label="Ensemble 90% Band")

    # ── Model lines ──
    if tfm_predictions is not None:
        ax1.plot(x_pred, tfm_predictions, color="#ff7f0e", label="TimesFM", linestyle="--", marker='x', ms=3)
    if chronos_median is not None:
        ax1.plot(x_pred, chronos_median, color="#00d4ff", label="Chronos-2", linestyle="--", marker='o', ms=3)
    if lstm_final is not None:
        ax1.plot(x_pred, lstm_final, color="#00ff88", label="LSTM", linestyle="-.", lw=1.5)
    if ensemble_median is not None:
        ax1.plot(x_pred, ensemble_median, color="#ff6b35", label="Ensemble", linestyle="-", lw=2)

    # ── TD annotations (candlestick only) ──
    if chart_type == 'candle':
        td_vals = td_df['td_count'].values.astype(int)
        for j in range(n_hist):
            count = td_vals[j] if j < len(td_vals) else 0
            if count == 0:
                continue
            _, h_j, l_j, c_j = ohlc_arr[j]
            if count == 9:
                lbl, clr, sz = '⚠9', '#ff4444', 8
                y_pos = h_j + (h_j - l_j) * 0.4
            elif count > 9:
                lbl, clr, sz = str(count), '#ff4444', 7
                y_pos = h_j + (h_j - l_j) * 0.4
            else:
                lbl, clr, sz = str(count), '#00ff88', 7
                y_pos = l_j - (h_j - l_j) * 0.6
            ax1.text(j, y_pos, lbl, ha='center', va='center', fontsize=sz,
                     color=clr, fontweight='bold' if count >= 9 else 'normal', alpha=0.85, zorder=3)

    # ── Axes limits ──
    price_min = float(ohlc_arr[:, 2].min())
    price_max = float(ohlc_arr[:, 1].max())
    yr = price_max - price_min
    ax1.set_ylim(price_min - yr * 0.2, price_max + yr * 0.3)
    ax1.set_xlim(-1.5, n_hist + horizon + 1)
    ax2.set_xlim(-1.5, n_hist + horizon + 1)
    ax3.set_xlim(-1.5, n_hist + horizon + 1)

    ax1.set_title(f"  {target}  |  {helper}  |  H={horizon}",
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

    # ── Date labels ──
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

    plt.savefig(output_path, dpi=110, facecolor='#0c0c0c', edgecolor='none')
    plt.close(fig)
    return True