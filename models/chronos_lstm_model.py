#!/usr/bin/env python3
"""
Chronos-2 + LSTM ensemble — run as standalone subprocess.
Usage: python chronos_lstm_model.py <target_csv> <horizon> [--no-chronos] [--no-lstm]
Output: JSON with predictions and confidence bands to stdout
"""
import sys, json, numpy as np, pandas as pd

TARGET_CSV = sys.argv[1] if len(sys.argv) > 1 else None
HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 12
USE_CHRONOS = "--no-chronos" not in sys.argv
USE_LSTM    = "--no-lstm"    not in sys.argv

df = pd.read_csv(TARGET_CSV, index_col=0, parse_dates=True)
target_vals  = df['close'].values.astype(np.float32)
helper_vals  = df['helper'].values.astype(np.float32)
current_price = float(target_vals[-1])

result = {}

# ── Chronos-2 (returns multiple quantiles for confidence band) ──
if USE_CHRONOS:
    try:
        import torch
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="auto")
        ctx = torch.tensor(np.expand_dims(np.stack([target_vals, helper_vals]), axis=0))
        out = pipeline.predict(ctx, prediction_length=HORIZON)
        # out shape: (batch, num_quantiles, horizon)
        q10 = out[0][0, 1, :].cpu().numpy()   # 10th percentile
        q50 = out[0][0, 4, :].cpu().numpy()   # median
        q90 = out[0][0, 7, :].cpu().numpy()   # 90th percentile
        # Anchor to current price
        offset = current_price - q50[0]
        result["chronos"]       = (q50 + offset).tolist()
        result["chronos_lower"] = (q10 + offset).tolist()
        result["chronos_upper"] = (q90 + offset).tolist()
    except Exception as e:
        result["chronos_error"] = str(e)

# ── LSTM with MC Dropout confidence band ──
if USE_LSTM:
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
        import tensorflow as tf

        fdf = df[['close', 'helper']].dropna().copy()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(fdf)

        X, y = [], []
        for i in range(60, len(scaled)):
            X.append(scaled[i-60:i, :])
            y.append(scaled[i, 0])

        # Build model with dropout for MC inference
        model = Sequential([
            Input(shape=(60, 2)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(np.array(X), np.array(y), epochs=20, batch_size=32, verbose=0)

        # Point estimate (single forward pass)
        last_win = scaled[-60:].copy()
        point_preds = []
        for _ in range(HORIZON):
            p = model.predict(last_win.reshape(1, 60, 2), verbose=0)[0, 0]
            point_preds.append(p)
            last_win = np.append(last_win[1:], [[p, scaled[-1, 1]]], axis=0)

        # MC Dropout: run N forward passes with training=True to get distribution
        N_SAMPLES = 30
        all_samples = []
        for _ in range(N_SAMPLES):
            win = scaled[-60:].copy()
            sample_preds = []
            for _ in range(HORIZON):
                # training=True keeps dropout active
                p = model(win.reshape(1, 60, 2), training=True).numpy()[0, 0]
                sample_preds.append(p)
                win = np.append(win[1:], [[p, scaled[-1, 1]]], axis=0)
            inv = scaler.inverse_transform(
                np.column_stack([sample_preds, np.zeros(HORIZON)])
            )[:, 0]
            all_samples.append(inv)

        samples = np.array(all_samples)  # (N_SAMPLES, HORIZON)
        lstm_mean   = samples.mean(axis=0)
        lstm_lower  = np.percentile(samples, 10, axis=0)
        lstm_upper  = np.percentile(samples, 90, axis=0)

        inv_point = scaler.inverse_transform(
            np.column_stack([point_preds, np.zeros(HORIZON)])
        )[:, 0]

        result["lstm"]       = inv_point.tolist()
        result["lstm_lower"] = lstm_lower.tolist()
        result["lstm_upper"] = lstm_upper.tolist()
    except Exception as e:
        result["lstm_error"] = str(e)

print(json.dumps(result))
