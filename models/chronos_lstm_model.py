#!/usr/bin/env python3
"""
Chronos-2 + LSTM ensemble.

Two modes:
  1. Standalone subprocess (legacy):
     python chronos_lstm_model.py <target_csv> <horizon> [--no-chronos] [--no-lstm]
     Output: JSON to stdout

  2. In-process (called from Flask for model caching):
     from models.chronos_lstm_model import run_inference
     result = run_inference(csv_path, horizon, use_chronos=True, use_lstm=True)
"""
import sys, json, numpy as np, pandas as pd

# ── Module-level cache for Chronos pipeline ──
_chronos_pipeline = None

def get_chronos_pipeline():
    """Load and cache the Chronos-2 pipeline. Returns the pipeline object."""
    global _chronos_pipeline
    if _chronos_pipeline is not None:
        return _chronos_pipeline
    import torch
    from chronos import BaseChronosPipeline
    print("[MODEL_CACHE] Loading Chronos-2 pipeline...", flush=True)
    _chronos_pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="auto")
    print("[MODEL_CACHE] Chronos-2 pipeline loaded and cached.", flush=True)
    return _chronos_pipeline

def is_chronos_cached():
    """Check if the Chronos pipeline is already in memory."""
    return _chronos_pipeline is not None


def run_inference(csv_path, horizon, use_chronos=True, use_lstm=True):
    """
    Run Chronos-2 + LSTM inference in-process (uses cached model).
    Returns a dict with predictions and confidence bands.
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    target_vals  = df['close'].values.astype(np.float32)
    helper_vals  = df['helper'].values.astype(np.float32)
    current_price = float(target_vals[-1])

    result = {}

    # ── Chronos-2 (uses cached pipeline) ──
    if use_chronos:
        try:
            import torch
            pipeline = get_chronos_pipeline()
            ctx = torch.tensor(np.expand_dims(np.stack([target_vals, helper_vals]), axis=0))
            out = pipeline.predict(ctx, prediction_length=horizon)
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
    if use_lstm:
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
            for _ in range(horizon):
                p = model.predict(last_win.reshape(1, 60, 2), verbose=0)[0, 0]
                point_preds.append(p)
                last_win = np.append(last_win[1:], [[p, scaled[-1, 1]]], axis=0)

            # MC Dropout: run N forward passes with training=True to get distribution
            N_SAMPLES = 30
            all_samples = []
            for _ in range(N_SAMPLES):
                win = scaled[-60:].copy()
                sample_preds = []
                for _ in range(horizon):
                    p = model(win.reshape(1, 60, 2), training=True).numpy()[0, 0]
                    sample_preds.append(p)
                    win = np.append(win[1:], [[p, scaled[-1, 1]]], axis=0)
                inv = scaler.inverse_transform(
                    np.column_stack([sample_preds, np.zeros(horizon)])
                )[:, 0]
                all_samples.append(inv)

            samples = np.array(all_samples)  # (N_SAMPLES, horizon)
            lstm_mean   = samples.mean(axis=0)
            lstm_lower  = np.percentile(samples, 10, axis=0)
            lstm_upper  = np.percentile(samples, 90, axis=0)

            inv_point = scaler.inverse_transform(
                np.column_stack([point_preds, np.zeros(horizon)])
            )[:, 0]

            result["lstm"]       = inv_point.tolist()
            result["lstm_lower"] = lstm_lower.tolist()
            result["lstm_upper"] = lstm_upper.tolist()
        except Exception as e:
            result["lstm_error"] = str(e)

    return result


# ── Legacy subprocess mode ──
if __name__ == "__main__":
    TARGET_CSV = sys.argv[1] if len(sys.argv) > 1 else None
    HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    USE_CHRONOS = "--no-chronos" not in sys.argv
    USE_LSTM    = "--no-lstm"    not in sys.argv

    result = run_inference(TARGET_CSV, HORIZON, use_chronos=USE_CHRONOS, use_lstm=USE_LSTM)
    print(json.dumps(result))