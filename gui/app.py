#!/usr/bin/env python3
"""
Stock Forecast GUI v2 — WebSocket + Model Caching + Error Display

Changes from v1:
  - Flask-SocketIO for real-time progress (replaces 3s polling)
  - Chronos-2 model cached in-process (subprocess for data download only)
  - Error events and 2-min timeout in frontend

Run:  python gui/app.py [port]
"""
import os, sys, json, re, time, subprocess, threading, traceback, urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_socketio import SocketIO, emit as socketio_emit
import numpy as np
import pandas as pd

WORKSPACE  = Path(__file__).resolve().parent.parent
FORECAST   = WORKSPACE / "forecast.py"
PYTHON     = "/opt/homebrew/bin/python3.11"
PORT       = int(sys.argv[1]) if len(sys.argv) > 1 else 9876

app = Flask(__name__)
app.config['SECRET_KEY'] = 'forecast-v2-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

_jobs = {}
_jobs_lock = threading.Lock()


# ── WebSocket namespace ──────────────────────────────────────
@socketio.on('connect')
def handle_connect():
    print(f"[WS] Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WS] Client disconnected: {request.sid}")


@socketio.on('start_forecast')
def handle_start_forecast(data):
    """Client initiates a forecast via websocket."""
    target     = (data.get('target') or 'APLD').upper()
    helper     = (data.get('helper') or 'NVDA').upper()
    horizon    = int(data.get('horizon', 12))
    data_range = (data.get('dataRange') or '1y').lower()
    chart_type = data.get('chartType', 'candle')
    use_tfm    = data.get('use_timesfm', False)
    use_chr    = data.get('use_chronos', True)
    use_lstm   = data.get('use_lstm', False)

    jid, cached = _submit_job(target, helper, horizon, use_tfm, use_chr, use_lstm, data_range, chart_type)

    if cached:
        with _jobs_lock:
            j = _jobs.get(jid, {})
        socketio_emit('done', {'job_id': jid, 'result': j.get('result', {})})
    else:
        socketio_emit('started', {'job_id': jid})


# ── Job submission ────────────────────────────────────────────
def _submit_job(target, helper, horizon, use_tfm, use_chr, use_lstm, data_range="1y", chart_type="candle"):
    data_range = data_range.lower()
    flags = f"{'T' if use_tfm else ''}{'C' if use_chr else ''}{'L' if use_lstm else ''}"
    jid = f"{target}_{helper}_{horizon}_{data_range}_{chart_type}_{flags}"

    with _jobs_lock:
        j = _jobs.get(jid)
        if j and j["status"] == "done" and (time.time() - j["ts"]) < 300:
            return jid, True
        if j and j["status"] == "running":
            return jid, False

    # Start new job
    chart_path = WORKSPACE / f"{target}_{helper}_{horizon}_{data_range}_{chart_type}_forecast.png"
    csv_path = WORKSPACE / "_forecast_data.csv"

    # Launch subprocess for data download + indicators only (no models, no chart)
    cmd = [
        PYTHON, str(FORECAST), target, helper,
        "--horizon", str(horizon), "--data-range", data_range,
        "--chart-type", chart_type,
        "--output", str(chart_path),
        "--data-only",  # Only download data + compute indicators, skip models + chart
    ]
    if not use_tfm:
        cmd.append("--no-timesfm")
    # Models are handled in-process, not in subprocess

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Background thread to read status lines from subprocess
    status_lines = []

    def _read_status():
        try:
            for line in proc.stdout:
                line = line.strip()
                if line:
                    status_lines.append(line)
        except: pass

    reader_thread = threading.Thread(target=_read_status, daemon=True)
    reader_thread.start()

    with _jobs_lock:
        _jobs[jid] = {
            "status": "running",
            "target": target,
            "helper": helper,
            "horizon": horizon,
            "data_range": data_range,
            "chart_type": chart_type,
            "use_tfm": use_tfm,
            "use_chr": use_chr,
            "use_lstm": use_lstm,
            "ts": time.time(),
            "proc": proc,
            "chart_path": str(chart_path),
            "csv_path": str(csv_path),
            "status_lines": status_lines,
            "reader_thread": reader_thread,
            "last_emitted_line": -1,
        }

    # Start the websocket emitter + job completion thread
    t = threading.Thread(target=_job_lifecycle, args=(jid,), daemon=True)
    t.start()

    return jid, False


def _emit_progress(jid, stage, state, line=""):
    """Emit a progress event via websocket."""
    socketio.emit('progress', {
        'job_id': jid,
        'stage': stage,
        'state': state,
        'line': line
    })


def _emit_error(jid, message, exit_code=None):
    """Emit an error event via websocket."""
    socketio.emit('error', {
        'job_id': jid,
        'message': message,
        'exit_code': exit_code
    })


def _job_lifecycle(jid):
    """Background thread: monitors subprocess, runs models, generates chart, emits events."""
    with _jobs_lock:
        job = _jobs.get(jid)
        if not job:
            return

    proc = job["proc"]
    status_lines = job["status_lines"]
    last_emitted = -1

    # Phase 1: Monitor subprocess for data download + indicators
    while True:
        time.sleep(0.3)

        # Emit new status lines
        new_lines = status_lines[last_emitted + 1:]
        for line in new_lines:
            if "[STATUS:" in line:
                parts = line.strip("[]").split(":")
                if len(parts) >= 3:
                    stage = parts[1].strip()
                    state = parts[2].strip()
                    _emit_progress(jid, stage, state, line)
            else:
                _emit_progress(jid, 'log', 'running', line[:120])
        last_emitted = len(status_lines) - 1

        with _jobs_lock:
            job = _jobs.get(jid, {})

        # Check if subprocess exited
        if proc.poll() is not None:
            break

    # Subprocess finished
    exit_code = proc.poll()

    # Wait for reader thread
    reader_thread = job.get("reader_thread")
    if reader_thread and reader_thread.is_alive():
        reader_thread.join(timeout=2)

    # Check for errors
    if exit_code is not None and exit_code != 0:
        err_lines = [l for l in status_lines
                    if "error" in l.lower() or "traceback" in l.lower()
                    or "exception" in l.lower()][-5:]
        # Also check for Traceback blocks
        in_traceback = False
        tb_lines = []
        for l in status_lines:
            if "Traceback" in l:
                in_traceback = True
            if in_traceback:
                tb_lines.append(l)
                if l.strip().startswith("Error:") or l.strip().startswith("Exception:"):
                    in_traceback = False
        if tb_lines:
            err_lines = tb_lines[-3:]
        err_msg = "; ".join(err_lines[-3:]) if err_lines else f"Subprocess exited with code {exit_code}"
        _emit_error(jid, err_msg, exit_code)
        with _jobs_lock:
            _jobs[jid] = {"status": "error", "result": {"error": err_msg, "exit_code": exit_code}, "ts": time.time()}
        return

    # Parse metadata from subprocess output
    metadata = None
    for line in status_lines:
        if "[METADATA_JSON]" in line:
            start = line.find("[METADATA_JSON]") + len("[METADATA_JSON]")
            end = line.find("[/METADATA_JSON]")
            if end > start:
                try:
                    metadata = json.loads(line[start:end])
                except: pass
            break

    if not metadata:
        # Try to extract current price from output lines
        metadata = {}
        for line in status_lines:
            m = re.search(r'Current \w+: \$([\d.]+)', line)
            if m:
                metadata["current_price"] = float(m.group(1))
                break

    csv_path = Path(job.get("csv_path", ""))
    chart_path = Path(job.get("chart_path", ""))

    # Check if CSV exists
    if not csv_path.exists():
        _emit_error(jid, "Data CSV not found after download. Check if data download succeeded.")
        with _jobs_lock:
            _jobs[jid] = {"status": "error", "result": {"error": "Data CSV not found"}, "ts": time.time()}
        return

    # Phase 2: Run models in-process (with caching)
    current_price = metadata.get("current_price", 0)
    model_result = {}

    use_chr = job.get("use_chr", True)
    use_lstm = job.get("use_lstm", False)
    use_tfm = job.get("use_tfm", False)
    horizon = job.get("horizon", 12)

    chronos_median = chronos_lo = chronos_hi = None
    lstm_final = lstm_lo = lstm_hi = None
    tfm_final = None

    if use_chr or use_lstm:
        _emit_progress(jid, 'models', 'running', '[STATUS:models:running] Running Chronos-2/LSTM in-process')
        try:
            from models.chronos_lstm_model import run_inference
            result = run_inference(str(csv_path), horizon, use_chronos=use_chr, use_lstm=use_lstm)

            if "chronos" in result:
                chronos_median = np.array(result["chronos"])
                chronos_lo = np.array(result.get("chronos_lower", result["chronos"]))
                chronos_hi = np.array(result.get("chronos_upper", result["chronos"]))
                _emit_progress(jid, 'chronos', 'done', '[STATUS:chronos:done] Chronos-2 complete')

            if "chronos_error" in result:
                _emit_progress(jid, 'chronos', 'failed', f'[STATUS:chronos:failed] {result["chronos_error"]}')

            if "lstm" in result:
                lstm_final = np.array(result["lstm"])
                lstm_lo = np.array(result.get("lstm_lower", result["lstm"]))
                lstm_hi = np.array(result.get("lstm_upper", result["lstm"]))
                _emit_progress(jid, 'lstm', 'done', '[STATUS:lstm:done] LSTM complete')

            if "lstm_error" in result:
                _emit_progress(jid, 'lstm', 'failed', f'[STATUS:lstm:failed] {result["lstm_error"]}')

            model_result.update(result)
        except Exception as e:
            _emit_progress(jid, 'models', 'failed', f'[STATUS:models:failed] {e}')

    if use_tfm:
        _emit_progress(jid, 'TimesFM 2.5', 'running', '[STATUS:TimesFM 2.5:running]')
        try:
            cmd = [PYTHON, str(WORKSPACE / "models" / "timesfm_model.py"), str(csv_path), str(horizon)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and r.stdout.strip():
                json_start = r.stdout.find('{')
                if json_start >= 0:
                    tfm_data = json.loads(r.stdout[json_start:])
                    if "predictions" in tfm_data:
                        tfm_final = np.array(tfm_data["predictions"])
                        _emit_progress(jid, 'TimesFM 2.5', 'done', '[STATUS:TimesFM 2.5:done]')
                    elif "error" in tfm_data:
                        model_result["tfm_error"] = tfm_data["error"]
                        _emit_progress(jid, 'TimesFM 2.5', 'failed', f'[STATUS:TimesFM 2.5:failed] {tfm_data["error"]}')
            else:
                stderr_snippet = (r.stderr or "")[-200:]
                model_result["tfm_error"] = f"TimesFM exited with code {r.returncode}: {stderr_snippet}"
                _emit_progress(jid, 'TimesFM 2.5', 'failed', f'[STATUS:TimesFM 2.5:failed] exit code {r.returncode}')
        except Exception as e:
            model_result["tfm_error"] = str(e)
            _emit_progress(jid, 'TimesFM 2.5', 'failed', f'[STATUS:TimesFM 2.5:failed] {e}')

    # Phase 3: Conformal calibration
    _emit_progress(jid, 'conformal', 'running', '[STATUS:conformal:running]')
    try:
        from mapie.regression import SplitConformalRegressor
        from sklearn.linear_model import LinearRegression

        all_preds = []
        if tfm_final is not None:      all_preds.append(tfm_final)
        if chronos_median is not None: all_preds.append(chronos_median)
        if lstm_final is not None:     all_preds.append(lstm_final)

        band_lo = band_hi = None
        ensemble_median = None

        if len(all_preds) >= 2:
            ensemble_median = np.median(all_preds, axis=0)
            df_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            target_vals = df_data['close'].values.astype(np.float32)
            recent = target_vals[-30:]
            resid_scale = np.median(np.abs(recent - np.median(recent))) * 1.5
            band_lo = ensemble_median - resid_scale
            band_hi = ensemble_median + resid_scale
        elif len(all_preds) == 1:
            # Only 1 model — skip MAPIE band, model's own CI is sufficient
            pass

        _emit_progress(jid, 'conformal', 'done', '[STATUS:conformal:done]')
    except Exception as e:
        _emit_progress(jid, 'conformal', 'failed', f'[STATUS:conformal:failed] {e}')
        band_lo = band_hi = None
        ensemble_median = None

    # Phase 4: Generate chart in-process
    _emit_progress(jid, 'chart', 'running', '[STATUS:chart:running]')
    try:
        from charting import generate_forecast_chart

        chart_ok = generate_forecast_chart(
            target=job["target"],
            helper=job["helper"],
            horizon=horizon,
            data_range=job["data_range"],
            chart_type=job["chart_type"],
            rsi_val=metadata.get("rsi", 0),
            ema_50=metadata.get("ema_50", 0),
            td_count=metadata.get("td_count", 0),
            bb_upper=metadata.get("bb_upper", 0),
            current_price=current_price,
            ema_label=metadata.get("ema_label", ""),
            rsi_label=metadata.get("rsi_label", ""),
            td_label=metadata.get("td_label", ""),
            vol_label=metadata.get("vol_label", ""),
            tfm_predictions=tfm_final,
            chronos_median=chronos_median,
            chronos_lo=chronos_lo,
            chronos_hi=chronos_hi,
            lstm_final=lstm_final,
            lstm_lo=lstm_lo,
            lstm_hi=lstm_hi,
            band_lo=band_lo,
            band_hi=band_hi,
            ensemble_median=ensemble_median,
            output_path=str(chart_path),
            csv_path=str(csv_path),
        )
        if chart_ok:
            _emit_progress(jid, 'chart', 'done', '[STATUS:chart:done]')
        else:
            _emit_progress(jid, 'chart', 'failed', '[STATUS:chart:failed] Chart generation returned False')
    except Exception as e:
        _emit_progress(jid, 'chart', 'failed', f'[STATUS:chart:failed] {e}')

    # Phase 5: Build result
    result = {
        "target": job["target"],
        "helper": job["helper"],
        "horizon": job.get("horizon", 12),
        "predictions": [],
        "consensus": "N/A",
        "avg_delta": 0,
        "technicals": {},
    }

    # Technical indicators from metadata
    tech = {}
    if "rsi" in metadata:
        tech["rsi"] = metadata["rsi"]
        tech["rsi_label"] = metadata.get("rsi_label", "")
    if "ema_50" in metadata:
        tech["ema"] = metadata["ema_50"]
        tech["ema_label"] = metadata.get("ema_label", "")
    if "td_count" in metadata:
        tc = metadata["td_count"]
        tech["td_count"] = tc if tc else None
        tech["td_label"] = metadata.get("td_label", "")
    if "bb_upper" in metadata:
        tech["bb_upper"] = metadata["bb_upper"]
        tech["vol_label"] = metadata.get("vol_label", "")
    if tech:
        result["technicals"] = tech

    # Build predictions
    predictions = []
    if tfm_final is not None:
        p = float(tfm_final[-1])
        d = ((p - current_price) / current_price) * 100 if current_price else 0
        predictions.append({"model": "TimesFM 2.5", "target": p, "delta": d})
    if chronos_median is not None:
        p = float(chronos_median[-1])
        d = ((p - current_price) / current_price) * 100 if current_price else 0
        predictions.append({"model": "Chronos-2", "target": p, "delta": d})
    if lstm_final is not None:
        p = float(lstm_final[-1])
        d = ((p - current_price) / current_price) * 100 if current_price else 0
        predictions.append({"model": "Neural LSTM", "target": p, "delta": d})

    result["predictions"] = predictions

    # Consensus
    if predictions:
        deltas = [p["delta"] for p in predictions]
        result["avg_delta"] = sum(deltas) / len(deltas)
        if all(d > 0 for d in deltas):
            result["consensus"] = "STRONG UP"
        elif all(d < 0 for d in deltas):
            result["consensus"] = "STRONG DOWN"
        else:
            result["consensus"] = "CONTRADICTORY"

    # Chart URL
    if chart_path.exists():
        result["chart_url"] = f"/chart/{chart_path.name}"
    else:
        import glob as _glob
        pattern = f"{job['target']}_{job['helper']}_*_forecast.png"
        matches = _glob.glob(str(WORKSPACE / pattern))
        if matches:
            result["chart_url"] = f"/chart/{os.path.basename(matches[-1])}"

    # Phase 6: Fetch news
    _emit_progress(jid, 'news', 'running', '[STATUS:news:running]')
    try:
        from news import get_news_and_ratings
        nd = get_news_and_ratings(job["target"])
        _emit_progress(jid, 'news', 'done', '[STATUS:news:done]')
        result["news"] = nd.get("news", [])[:10]
        result["analyst"] = nd.get("analyst", {"available": False})
        result["earnings"] = nd.get("earnings", {"available": False})
        result["sentiment_summary"] = nd.get("sentiment_summary", {"label": "Neutral", "emoji": "⚪", "avg_compound": 0.0})
    except Exception as e:
        _emit_progress(jid, 'news', 'failed', f'[STATUS:news:failed] {e}')
        result["news"] = []
        result["analyst"] = {"available": False}
        result["earnings"] = {"available": False}
        result["sentiment_summary"] = {"label": "Neutral", "emoji": "⚪", "avg_compound": 0.0}

    # Compute AI Score for the result
    try:
        from news import compute_ai_score
        tech_for_ai = {
            "rsi": metadata.get("rsi"),
            "ema_label": metadata.get("ema_label", ""),
            "td_count": metadata.get("td_count", 0),
            "52wk_high": metadata.get("52wk_high"),
            "52wk_low": metadata.get("52wk_low"),
            "current_price": current_price,
            "vol_label": metadata.get("vol_label", ""),
        }
        result["ai_score"] = compute_ai_score({}, tech_for_ai, result.get("sentiment_summary", {}), result.get("analyst", {"available": False}))
    except Exception:
        result["ai_score"] = None

    # Clean up
    with _jobs_lock:
        _jobs[jid] = {"status": "done", "result": result, "ts": time.time()}

    # Clean up CSV
    try:
        if csv_path.exists():
            csv_path.unlink()
    except: pass

    # Emit done
    socketio.emit('done', {'job_id': jid, 'result': result})
    print(f"[JOB] Completed: {jid}")


# ── REST Endpoints ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/start", methods=["POST"])
def api_start():
    """REST fallback — starts job and returns job_id. Progress comes via websocket."""
    jid, cached = _submit_job(
        request.form.get("target", "APLD").upper(),
        request.form.get("helper", "NVDA").upper(),
        int(request.form.get("horizon", 12)),
        request.form.get("use_timesfm") == "true",
        request.form.get("use_chronos") == "true",
        request.form.get("use_lstm") == "true",
        request.form.get("dataRange", "1y"),
        request.form.get("chartType", "candle"),
    )
    if cached:
        with _jobs_lock:
            j = _jobs.get(jid, {})
        return jsonify({"status": "done", "job_id": jid, "result": j.get("result", {})})
    return jsonify({"status": "started", "job_id": jid})


@app.route("/api/cache-status")
def api_cache_status():
    """Check which models are cached in memory."""
    try:
        from models.chronos_lstm_model import is_chronos_cached
        chronos = is_chronos_cached()
    except:
        chronos = False
    return jsonify({
        "chronos_cached": chronos,
    })


@app.route("/chart/<path:chart_name>")
def chart(chart_name):
    p = WORKSPACE / chart_name
    return send_file(p, mimetype="image/png") if p.exists() else ("", 404)


@app.route("/api/news/<ticker>")
def api_news(ticker):
    """Fetch news/analyst/earnings immediately — no forecast needed."""
    try:
        from news import get_news_and_ratings
        nd = get_news_and_ratings(ticker.upper())
        return jsonify({
            "news": nd.get("news", [])[:10],
            "analyst": nd.get("analyst", {"available": False}),
            "earnings": nd.get("earnings", {"available": False}),
            "sentiment_summary": nd.get("sentiment_summary", {"label": "Neutral", "emoji": "⚪", "avg_compound": 0.0}),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/preview/<ticker>")
def api_preview(ticker):
    """Fast technical indicator preview — no model inference, no chart generation."""
    ticker = ticker.upper()
    try:
        import yfinance as yf

        data_range = request.args.get("data_range", "3m").lower()
        range_map = {"3m": "3mo", "6m": "6mo", "1y": "1y", "2y": "2y", "5y": "5y", "max": "max"}
        period = range_map.get(data_range, "3mo")

        # Download data
        raw = yf.download([ticker], period=period, auto_adjust=True, progress=False)
        if len(raw) == 0:
            return jsonify({"error": f"No data found for {ticker}"}), 404

        df = pd.DataFrame(index=raw.index)
        close_col = ('Close', ticker) if ('Close', ticker) in raw.columns else 'Close'
        if isinstance(raw[close_col], pd.DataFrame):
            df['close'] = raw[close_col].iloc[:, 0].ffill()
        else:
            df['close'] = raw[close_col].ffill()
        df.dropna(inplace=True)

        if len(df) == 0:
            return jsonify({"error": f"No data found for {ticker}"}), 404

        target_vals = df['close'].values.astype(float)
        current_price = float(target_vals[-1])

        # Compute indicators (same as forecast.py)
        def compute_rsi(series, length=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
            rs = gain / loss
            return 100.0 - (100.0 / (1.0 + rs))

        def compute_ema(series, length=50):
            return series.ewm(span=length, adjust=False).mean()

        def compute_bbands(series, length=20, std=2.0):
            sma = series.rolling(window=length).mean()
            rstd = series.rolling(window=length).std()
            return sma + rstd * std, sma, sma - rstd * std

        def get_td_setup(arr):
            setup = np.zeros(len(arr))
            count = 0
            for i in range(4, len(arr)):
                count = count + 1 if arr[i] > arr[i-4] else 0
                setup[i] = count
            return setup

        df['RSI_14'] = compute_rsi(df['close'], 14)
        df['EMA_50'] = compute_ema(df['close'], 50)
        df['BBU_20'], df['BBM_20'], df['BBL_20'] = compute_bbands(df['close'], 20, 2.0)
        df['td_count'] = get_td_setup(df['close'].values)

        rsi_val = float(df['RSI_14'].iloc[-1]) if not pd.isna(df['RSI_14'].iloc[-1]) else None
        ema_50 = float(df['EMA_50'].iloc[-1]) if not pd.isna(df['EMA_50'].iloc[-1]) else None
        bb_upper = float(df['BBU_20'].iloc[-1]) if not pd.isna(df['BBU_20'].iloc[-1]) else None
        td_count = int(df['td_count'].iloc[-1])

        # Labels
        rsi_label = "[OVERBOUGHT]" if rsi_val and rsi_val > 70 else "[OVERSOLD]" if rsi_val and rsi_val < 30 else "[STABLE]"
        ema_label = "BULLISH" if ema_50 and current_price > ema_50 else "BEARISH"
        td_label = "⚠️ TD 9" if td_count == 9 else f"Countdown ({td_count})" if td_count >= 10 else "Setup Active" if td_count >= 5 else "Setup Building" if td_count > 0 else "—"
        vol_label = "OVEREXTENDED" if bb_upper and current_price > bb_upper else "NORMAL"

        # 52-week high/low
        info = {}
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
        except:
            pass
        wk52_high = info.get("fiftyTwoWeekHigh")
        wk52_low = info.get("fiftyTwoWeekLow")
        company_name = info.get("shortName", ticker)

        # Also fetch news + sentiment for this ticker
        try:
            from news import get_news_and_ratings, get_fundamentals, compute_ai_score
            nd = get_news_and_ratings(ticker)
            news = nd.get("news", [])[:10]
            sentiment_summary = nd.get("sentiment_summary", {"label": "Neutral", "emoji": "⚪", "avg_compound": 0.0, "compound": 0.0})
            analyst = nd.get("analyst", {"available": False})
            earnings = nd.get("earnings", {"available": False})
            fundamentals = get_fundamentals(ticker)
        except:
            news = []
            sentiment_summary = {"label": "Neutral", "emoji": "⚪", "avg_compound": 0.0, "compound": 0.0}
            analyst = {"available": False}
            earnings = {"available": False}
            fundamentals = {"available": False}

        # Compute AI Score
        tech_data = {
            "rsi": rsi_val,
            "ema_label": ema_label,
            "td_count": td_count,
            "52wk_high": wk52_high,
            "52wk_low": wk52_low,
            "current_price": current_price,
            "vol_label": vol_label,
        }
        ai_score = compute_ai_score({}, tech_data, sentiment_summary, analyst)

        return jsonify({
            "ticker": ticker,
            "name": company_name,
            "current_price": round(current_price, 2),
            "rsi": round(rsi_val, 2) if rsi_val else None,
            "rsi_label": rsi_label,
            "ema_50": round(ema_50, 2) if ema_50 else None,
            "ema_label": ema_label,
            "td_count": td_count,
            "td_label": td_label,
            "bb_upper": round(bb_upper, 2) if bb_upper else None,
            "vol_label": vol_label,
            "52wk_high": wk52_high,
            "52wk_low": wk52_low,
            "news": news,
            "sentiment_summary": sentiment_summary,
            "analyst": analyst,
            "earnings": earnings,
            "fundamentals": fundamentals,
            "ai_score": ai_score,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def api_search():
    """Ticker autocomplete via Yahoo Finance search API."""
    query = request.args.get("q", "").strip()
    if len(query) < 1:
        return jsonify({"results": []})

    try:
        import urllib.request
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}&quotesCount=8&newsCount=0&quotesQueryId=tss_query_phrase"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for item in data.get("quotes", [])[:8]:
            if item.get("quoteType") in ("EQUITY", "ETF", "CRYPTOCURRENCY", "CURRENCY", "INDEX", "MUTUALFUND", None):
                results.append({
                    "symbol": item.get("symbol", ""),
                    "name": item.get("shortname", item.get("longname", "")),
                    "type": item.get("quoteType", "EQUITY"),
                    "exchange": item.get("exchange", ""),
                })
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)})


@app.route("/api/chart-ready/<path:chart_name>")
def api_chart_ready(chart_name):
    p = WORKSPACE / chart_name
    return jsonify({"ready": p.exists() and p.stat().st_size > 1000})


@app.route("/test")
def test():
    return send_file(WORKSPACE / "gui" / "test_standalone.html")


# ── Chart Data (for Chart.js browser rendering) ──
@app.route("/api/chart-data/<ticker>")
def api_chart_data(ticker):
    """Return OHLC + indicator data as JSON for Chart.js browser rendering."""
    ticker = ticker.upper()
    try:
        import yfinance as yf
        import numpy as np

        data_range = request.args.get("data_range", "3m").lower()
        range_map = {"3m": "3mo", "6m": "6mo", "1y": "1y", "2y": "2y", "5y": "5y", "max": "max"}
        period = range_map.get(data_range, "3mo")

        raw = yf.download([ticker], period=period, auto_adjust=True, progress=False)
        if len(raw) == 0:
            return jsonify({"error": "No data found"}), 404

        # Build OHLC array
        close_col = ('Close', ticker) if ('Close', ticker) in raw.columns else 'Close'
        open_col = ('Open', ticker) if ('Open', ticker) in raw.columns else 'Open'
        high_col = ('High', ticker) if ('High', ticker) in raw.columns else 'High'
        low_col = ('Low', ticker) if ('Low', ticker) in raw.columns else 'Low'

        ohlc = []
        for idx in raw.index:
            ohlc.append({
                "x": idx.strftime("%Y-%m-%d"),
                "o": float(raw[open_col].loc[idx]),
                "h": float(raw[high_col].loc[idx]),
                "l": float(raw[low_col].loc[idx]),
                "c": float(raw[close_col].loc[idx]),
            })

        # Compute indicators
        close_series = raw[close_col].squeeze()
        def compute_rsi(series, length=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=length).mean()
            rs = gain / loss
            return 100.0 - (100.0 / (1.0 + rs))

        def compute_ema(series, length=50):
            return series.ewm(span=length, adjust=False).mean()

        def compute_bbands(series, length=20, std=2.0):
            sma = series.rolling(window=length).mean()
            rstd = series.rolling(window=length).std()
            return sma + rstd * std, sma, sma - rstd * std

        rsi = compute_rsi(close_series, 14)
        ema50 = compute_ema(close_series, 50)
        bb_upper, bb_mid, bb_lower = compute_bbands(close_series, 20, 2.0)

        # Build indicator arrays (aligned with OHLC dates)
        indicators = {"rsi": [], "ema_50": [], "bb_upper": [], "bb_lower": [], "bb_mid": []}
        for idx in raw.index:
            indicators["rsi"].append({"x": idx.strftime("%Y-%m-%d"), "y": round(float(rsi.loc[idx]), 2) if not pd.isna(rsi.loc[idx]) else None})
            indicators["ema_50"].append({"x": idx.strftime("%Y-%m-%d"), "y": round(float(ema50.loc[idx]), 2) if not pd.isna(ema50.loc[idx]) else None})
            indicators["bb_upper"].append({"x": idx.strftime("%Y-%m-%d"), "y": round(float(bb_upper.loc[idx]), 2) if not pd.isna(bb_upper.loc[idx]) else None})
            indicators["bb_lower"].append({"x": idx.strftime("%Y-%m-%d"), "y": round(float(bb_lower.loc[idx]), 2) if not pd.isna(bb_lower.loc[idx]) else None})
            indicators["bb_mid"].append({"x": idx.strftime("%Y-%m-%d"), "y": round(float(bb_mid.loc[idx]), 2) if not pd.isna(bb_mid.loc[idx]) else None})

        return jsonify({
            "ticker": ticker,
            "ohlc": ohlc,
            "indicators": indicators,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── HTML ──────────────────────────────────────────────────────
PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Forecast v2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#09090f;--surface:#10101a;--surface2:#15151f;--surface3:#1a1a26;
  --border:#1e1e2e;--border2:#252538;
  --text:#c9c9d8;--text-dim:#6b6b80;--text-muted:#3a3a4e;
  --accent:#22c55e;--warn:#f97316;--danger:#ef4444;--info:#60a5fa;
}
body{font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;font-size:13px;line-height:1.5}
.app{max-width:1100px;margin:0 auto;padding:14px 20px 40px}

/* Header */
.hdr{display:flex;align-items:center;justify-content:space-between;padding:0 0 14px;border-bottom:1px solid var(--border);margin-bottom:16px}
.hdr-title{font-size:14px;font-weight:600;color:var(--text)}
.hdr-title em{color:var(--accent);font-style:normal}
.hdr-right{display:flex;align-items:center;gap:14px}
.ws-badge{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--text-muted)}
.ws-badge .dot{width:6px;height:6px;border-radius:50%;background:var(--text-muted);flex-shrink:0}
.ws-badge.connected{color:var(--text-dim)}.ws-badge.connected .dot{background:var(--accent);box-shadow:0 0 5px var(--accent)}
.ws-badge.disconnected .dot{background:var(--danger)}
.cache-badge{font-size:10px;color:var(--text-muted);display:flex;align-items:center;gap:4px}
.cache-badge .cdot{width:6px;height:6px;border-radius:50%;background:var(--text-muted)}
.cache-badge .cdot.on{background:var(--accent)}

/* Controls */
.ctrl{display:flex;align-items:flex-end;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.7px;font-weight:500}
.field input,.field select{background:var(--surface2);border:1px solid var(--border2);color:var(--text);padding:7px 10px;border-radius:6px;font-size:13px;font-family:inherit;outline:none;width:112px;transition:border-color .15s}
.field input:focus,.field select:focus{border-color:var(--accent)}
.field input::placeholder{color:var(--text-muted)}
.btn-primary{background:var(--accent);color:#050a05;border:none;padding:7px 22px;border-radius:6px;font-size:13px;font-family:inherit;font-weight:700;cursor:pointer;letter-spacing:.3px;transition:opacity .15s;white-space:nowrap}
.btn-primary:hover{opacity:.88}.btn-primary:disabled{opacity:.35;cursor:not-allowed}
.btn-ghost{background:transparent;color:var(--text-dim);border:1px solid var(--border2);padding:7px 14px;border-radius:6px;font-size:12px;font-family:inherit;cursor:pointer;transition:all .15s}
.btn-ghost:hover{color:var(--text)}

/* Model toggles */
.model-bar{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px;align-items:center}
.mtog{display:flex;align-items:center;gap:6px;padding:4px 12px;border:1px solid var(--border2);border-radius:20px;cursor:pointer;font-size:11px;color:var(--text-muted);background:var(--surface2);transition:all .15s;user-select:none}
.mtog input{display:none}
.mtog .mdot{width:6px;height:6px;border-radius:50%;background:var(--text-muted);transition:background .15s}
.mtog.on{color:var(--text);border-color:var(--border2)}
.mtog.m-tfm.on .mdot{background:#f59e0b}
.mtog.m-chr.on .mdot{background:#38bdf8}
.mtog.m-lstm.on .mdot{background:var(--accent)}

/* Autocomplete */
.acwrap{position:relative}
.aclist{position:absolute;top:calc(100% + 3px);left:0;right:0;background:var(--surface2);border:1px solid var(--border2);border-radius:6px;max-height:200px;overflow-y:auto;z-index:200;display:none;box-shadow:0 8px 24px rgba(0,0,0,.5)}
.aclist.open{display:block}
.acitem{padding:7px 12px;cursor:pointer;font-size:12px;color:var(--text);border-bottom:1px solid var(--border)}
.acitem:last-child{border-bottom:none}
.acitem:hover,.acitem.sel{background:var(--surface3);color:var(--accent)}
.acitem .sym{color:var(--accent);font-weight:700;margin-right:6px}
.acitem .exch{color:var(--text-muted);font-size:10px;margin-left:4px}

/* Watchlist */
.wl-row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:12px;min-height:24px}
.wl-chip{display:inline-flex;align-items:center;gap:4px;background:var(--surface2);border:1px solid var(--border2);border-radius:5px;padding:3px 9px;font-size:11px;cursor:pointer;color:var(--text-dim);transition:all .15s}
.wl-chip:hover,.wl-chip.cur{border-color:var(--accent);color:var(--accent)}
.wl-chip .x{color:var(--text-muted);font-size:9px;margin-left:2px;padding:1px 3px}
.wl-chip .x:hover{color:var(--danger)}
.wl-add{display:inline-flex;align-items:center;gap:4px;background:transparent;border:1px dashed var(--border2);border-radius:5px;padding:3px 9px;font-size:11px;cursor:pointer;color:var(--text-muted);transition:all .15s}
.wl-add:hover{border-color:var(--accent);color:var(--accent)}
.wl-add.watching{color:var(--accent);border-color:var(--accent);border-style:solid}

/* Status / errors */
.status-bar{min-height:34px;display:flex;align-items:center;justify-content:center;gap:8px;font-size:12px;color:var(--accent);padding:4px 0}
.warn-bar{font-size:11px;color:var(--warn);text-align:center;padding:4px 0;display:none}
.err-box{background:#130808;border:1px solid #3a1010;border-radius:6px;padding:10px 14px;font-size:12px;color:var(--danger);margin:6px 0;display:none;line-height:1.5}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid var(--border2);border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}

/* Help panel */
.help-panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:14px;display:none}
.help-panel.open{display:block}
.help-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px;font-size:12px;color:var(--text-dim);line-height:1.7}
.help-grid b{color:var(--text)}
code{background:var(--surface3);padding:1px 5px;border-radius:3px;font-family:'SF Mono','Fira Code',monospace;font-size:11px}

/* Preview Panel */
.preview-panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:14px;display:none}
.preview-panel.open{display:block}
.prev-header{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px}
.prev-label{font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.7px;font-weight:500;display:block}
.prev-name{font-size:13px;font-weight:600;color:var(--text);display:block}
.prev-price{font-size:24px;font-weight:700;color:var(--text);font-variant-numeric:tabular-nums;letter-spacing:-.5px}

/* AI Band - single source, lives in preview, updates after forecast */
.ai-band{display:flex;align-items:stretch;background:var(--surface2);border:1px solid var(--border);border-radius:7px;overflow:hidden;margin-bottom:12px}
.ai-score-col{padding:10px 16px;text-align:center;min-width:76px;border-right:1px solid var(--border)}
.ai-score-lbl{font-size:9px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.7px;margin-bottom:3px}
.ai-score-num{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;line-height:1}
.ai-score-num.bull{color:var(--accent)}.ai-score-num.bear{color:var(--danger)}.ai-score-num.neut{color:var(--text-dim)}
.ai-verdict-col{flex:1;padding:10px 14px;display:flex;flex-direction:column;justify-content:center;border-right:1px solid var(--border)}
.ai-verdict{font-size:13px;font-weight:600;color:var(--text);margin-bottom:2px}
.ai-verdict-sub{font-size:10px;color:var(--text-muted)}
.ai-comps-col{display:grid;grid-template-columns:repeat(4,1fr)}
.ai-comp{padding:8px 6px;text-align:center;border-left:1px solid var(--border)}
.ai-comp .ck{font-size:9px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.4px;display:block;margin-bottom:2px}
.ai-comp .cv{font-size:12px;font-weight:600;color:var(--text);display:block}

/* Technicals grid */
.tech-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-bottom:12px}
.tech-item{background:var(--surface2);border:1px solid var(--border);border-radius:5px;padding:7px 10px}
.tech-item .tk{font-size:9px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px;display:block;margin-bottom:2px}
.tech-item .tv{font-size:12px;font-weight:600;color:var(--text)}
.tech-item .tv.up{color:var(--accent)}.tech-item .tv.down{color:var(--danger)}.tech-item .tv.warn{color:var(--warn)}

/* 52-week bar */
.wk52{display:flex;align-items:center;gap:8px;margin-bottom:12px;font-size:11px}
.wk52-bar{flex:1;height:4px;background:var(--border2);border-radius:2px;position:relative}
.wk52-pos{position:absolute;top:-4px;width:2px;height:12px;background:#fff;border-radius:1px;transform:translateX(-50%)}
.wk52 .wlo,.wk52 .whi{color:var(--text-muted)}
.wk52 .wpct{color:var(--text-dim);font-size:10px}

/* Sentiment */
.sent-row{display:flex;align-items:center;gap:8px;font-size:12px;padding:8px 0;border-top:1px solid var(--border);margin-top:4px;flex-wrap:wrap}
.sent-lbl{font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.5px}
.ss-label{font-weight:600}
.ss-label.bull{color:var(--accent)}.ss-label.bear{color:var(--danger)}.ss-label.neut{color:var(--text-dim)}
.trend-pill{font-size:10px;padding:2px 8px;border-radius:10px;font-weight:500}
.trend-pill.trend-improving{color:var(--accent);background:rgba(34,197,94,.1)}
.trend-pill.trend-worsening{color:var(--danger);background:rgba(239,68,68,.1)}
.trend-pill.trend-stable{color:var(--text-dim);background:var(--border)}

/* Fundamentals */
.fund-hdr{font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.7px;font-weight:600;padding:10px 0 6px;border-top:1px solid var(--border);margin-top:8px}
.fund-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.fund-item{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:6px 8px;display:flex;justify-content:space-between;align-items:center}
.fk{font-size:9px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.4px}
.fv{font-size:11px;font-weight:600;color:var(--text)}
.fv.up{color:var(--accent)}.fv.down{color:var(--danger)}.fv.warn{color:var(--warn)}

/* Results */
.results-section{display:none}
.results-section.open{display:block}

/* Chart area */
.chart-area{background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:12px}
.chart-bar{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;border-bottom:1px solid var(--border)}
.chart-tabs{display:flex;gap:2px;align-items:center}
.ctab{background:transparent;border:none;padding:4px 10px;border-radius:5px;font-size:11px;font-family:inherit;color:var(--text-muted);cursor:pointer;transition:all .15s}
.ctab.active{background:var(--surface3);color:var(--text)}
.ctab:hover:not(.active){color:var(--text-dim)}
.csep{width:1px;height:14px;background:var(--border2);margin:0 4px}
.chart-body img{width:100%;display:block;min-height:60px}
.chart-body canvas{display:none;width:100%!important;height:400px!important;background:var(--bg)}
.chart-overlays{display:flex;gap:4px;padding:8px 14px;border-bottom:1px solid var(--border);flex-wrap:wrap;align-items:center}
.chart-overlays-label{font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.6px;margin-right:4px}
.otog{display:inline-flex;align-items:center;gap:4px;padding:3px 9px;border-radius:4px;border:1px solid var(--border2);background:transparent;font-size:10px;font-family:inherit;color:var(--text-muted);cursor:pointer;transition:all .15s;user-select:none}
.otog .odot{width:5px;height:5px;border-radius:50%;background:currentColor;opacity:.4}
.otog.on{color:var(--text);border-color:var(--border2);background:var(--surface3)}
.otog.on .odot{opacity:1}
.otog.bb.on{color:#7dd3fc}.otog.ema.on{color:#f97316}
.otog.trend.on{color:#a78bfa}.otog.fvg.on{color:#4ade80}
.otog.ifvg.on{color:#f87171}.otog.rsi.on{color:#facc15}
/* RSI sub-chart */
.rsi-body{display:none;border-top:1px solid var(--border);position:relative}
.rsi-body.show{display:block}
.rsi-body canvas{width:100%!important;height:100px!important;background:var(--bg)}

/* Data cards */
.data-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px}
@media(max-width:680px){.data-grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px}
.card-title{font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.8px;font-weight:600;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--border)}
.row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;font-size:12px;border-bottom:1px solid var(--border)}
.row:last-child{border-bottom:none}
.rk{color:var(--text-dim)}.rv{font-weight:600;color:var(--text)}
.rv.up{color:var(--accent)}.rv.down{color:var(--danger)}.rv.warn{color:var(--warn)}
.analyst-bar{display:flex;height:14px;border-radius:3px;overflow:hidden;margin:8px 0 5px}
.analyst-bar div{display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:rgba(0,0,0,.8)}
.bar-buy{background:var(--accent)}.bar-hold{background:#525252}.bar-sell{background:var(--danger)}
.a-legend{display:flex;gap:12px;font-size:10px;color:var(--text-muted)}
.a-legend span::before{content:'';display:inline-block;width:7px;height:7px;border-radius:1px;margin-right:3px;vertical-align:middle}
.al-buy::before{background:var(--accent)}.al-hold::before{background:#525252}.al-sell::before{background:var(--danger)}
.news-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:10px}
.news-list{list-style:none}
.news-item{padding:7px 0;border-bottom:1px solid var(--border)}
.news-item:last-child{border-bottom:none}
.news-item a{color:var(--info);text-decoration:none;font-size:12px;line-height:1.5}
.news-item a:hover{color:var(--accent)}
.news-item .meta{color:var(--text-muted);font-size:10px;margin-top:2px}
.news-sent-row{display:flex;align-items:center;gap:8px;font-size:11px;padding-bottom:8px;margin-bottom:6px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.news-counts{display:flex;gap:8px;font-size:11px;color:var(--text-dim)}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.min.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="app">

<div class="hdr">
  <div class="hdr-title">Forecast <em>v2</em></div>
  <div class="hdr-right">
    <div id="cacheInfo" class="cache-badge"></div>
    <div id="wsStatus" class="ws-badge disconnected"><span class="dot"></span><span id="wsText">offline</span></div>
  </div>
</div>

<div class="ctrl">
  <div class="field acwrap">
    <label>Target</label>
    <input id="target" placeholder="AAPL" value="APLD" autocomplete="off">
    <div id="targetAC" class="aclist"></div>
  </div>
  <div class="field acwrap">
    <label>Helper</label>
    <input id="helper" placeholder="SPY" value="NVDA" autocomplete="off">
    <div id="helperAC" class="aclist"></div>
  </div>
  <div class="field">
    <label>Horizon</label>
    <select id="horizon">
      <option value="5">5 days</option><option value="12" selected>12 days</option>
      <option value="20">20 days</option><option value="30">30 days</option>
    </select>
  </div>
  <div class="field">
    <label>Data Range</label>
    <select id="dataRange" onchange="checkResources()">
      <option value="3m" selected>3 months</option><option value="6m">6 months</option>
      <option value="1y">1 year</option><option value="2y">2 years</option>
      <option value="5y">5 years</option><option value="max">Max</option>
    </select>
  </div>
  <div class="field"><label>&nbsp;</label><button id="btn" class="btn-primary" onclick="run()">&#9654; Run</button></div>
  <div class="field"><label>&nbsp;</label><button class="btn-ghost" onclick="toggleHelp()">Help</button></div>
</div>

<div class="model-bar">
  <span style="font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:.7px;margin-right:4px">Models</span>
  <label class="mtog m-tfm" id="togT" onclick="syncToggle('oT','togT');checkResources()">
    <input type="checkbox" id="oT"><span class="mdot"></span>TimesFM 2.5
  </label>
  <label class="mtog m-chr on" id="togC" onclick="syncToggle('oC','togC');checkResources()">
    <input type="checkbox" id="oC" checked><span class="mdot"></span>Chronos-2
  </label>
  <label class="mtog m-lstm" id="togL" onclick="syncToggle('oL','togL');checkResources()">
    <input type="checkbox" id="oL"><span class="mdot"></span>LSTM
  </label>
</div>

<div id="watchlistBar" class="wl-row"></div>

<div id="helpPanel" class="help-panel">
  <div style="display:flex;align-items:center;justify-content:space-between">
    <span style="font-size:12px;font-weight:600;color:var(--text)">How to Use</span>
    <button class="btn-ghost" style="padding:3px 10px;font-size:11px" onclick="toggleHelp()">Close</button>
  </div>
  <div class="help-grid">
    <div>
      <p><b>Target</b> &#8212; stock to forecast. e.g. <code>APLD</code>, <code>TSLA</code>, <code>BTC-USD</code></p>
      <p><b>Helper</b> &#8212; macro correlation. e.g. <code>NVDA</code>, <code>SPY</code></p>
      <p><b>Horizon</b> &#8212; forecast days ahead. Shorter = more accurate.</p>
    </div>
    <div>
      <p><b>TimesFM 2.5</b> &#8212; Google's time-series foundation model (~2 min)</p>
      <p><b>Chronos-2</b> &#8212; Amazon's probabilistic forecaster (cached in-process)</p>
      <p><b>LSTM</b> &#8212; Neural network with Monte Carlo dropout (~2 min)</p>
    </div>
  </div>
</div>

<div class="status-bar" id="st"></div>
<div class="warn-bar" id="warnBox"></div>
<div class="err-box" id="errBox"></div>

<div id="previewPanel" class="preview-panel">
  <div class="prev-header">
    <div>
      <span class="prev-label" id="prevTicker">Quick Preview</span>
      <span class="prev-name" id="prevName"></span>
    </div>
    <span class="prev-price" id="prevPrice"></span>
  </div>
  <div id="previewContent"><div style="color:var(--text-muted);font-size:12px;padding:8px 0">Type a ticker symbol to see live technicals and AI score.</div></div>
</div>

<div class="results-section" id="res">
  <div class="chart-area">
    <div class="chart-bar">
      <div class="chart-tabs">
        <button id="btnCandle" class="ctab active" onclick="setChartType('candle')">Candlestick</button>
        <button id="btnLine" class="ctab" onclick="setChartType('line')">Line</button>
        <div class="csep"></div>
        <button id="btnServer" class="ctab active" onclick="setRenderMode('server')">Static</button>
        <button id="btnBrowser" class="ctab" onclick="setRenderMode('browser')">Interactive</button>
      </div>
    </div>
    <div class="chart-overlays" id="chartOverlays" style="display:none">
      <span class="chart-overlays-label">Overlays</span>
      <button class="otog bb on" id="togBB" onclick="toggleOverlay('bb')"><span class="odot"></span>Bollinger</button>
      <button class="otog ema on" id="togEMA" onclick="toggleOverlay('ema')"><span class="odot"></span>EMA 50</button>
      <button class="otog trend on" id="togTrend" onclick="toggleOverlay('trend')"><span class="odot"></span>Trend Lines</button>
      <button class="otog fvg on" id="togFVG" onclick="toggleOverlay('fvg')"><span class="odot"></span>FVG</button>
      <button class="otog ifvg" id="togIFVG" onclick="toggleOverlay('ifvg')"><span class="odot"></span>IFVG</button>
      <button class="otog rsi" id="togRSI" onclick="toggleOverlay('rsi')"><span class="odot"></span>RSI</button>
    </div>
    <div class="chart-body">
      <img id="chart" src="" style="width:100%;display:block">
      <canvas id="chartCanvas"></canvas>
    </div>
    <div class="rsi-body" id="rsiBody"><canvas id="rsiCanvas"></canvas></div>
  </div>
  <div class="data-grid">
    <div class="card"><div class="card-title">Predictions</div><div id="pBody"></div></div>
    <div class="card"><div class="card-title">Technicals</div><div id="tBody"></div></div>
    <div class="card"><div class="card-title">Analyst</div><div id="aBody"></div></div>
  </div>
  <div class="news-card"><div class="card-title">News</div><ul class="news-list" id="nList"></ul></div>
</div>

</div>
<script>
var socket=io();
var currentJobId=null,lastProgressTs=Date.now(),timeoutInterval=null;
var completedStages=[],currentStage='',chartType='candle';

socket.on('connect',function(){
  var b=$('wsStatus');b.className='ws-badge connected';$('wsText').textContent='connected';loadCacheStatus();
});
socket.on('disconnect',function(){
  var b=$('wsStatus');b.className='ws-badge disconnected';$('wsText').textContent='offline';
});
socket.on('started',function(data){currentJobId=data.job_id;lastProgressTs=Date.now();startTimeoutWatch();});
socket.on('progress',function(data){
  lastProgressTs=Date.now();clearError();
  if(data.stage==='log'){$('st').innerHTML='<span class="spinner"></span> '+escHtml(data.line);return;}
  if(data.state==='done'&&!completedStages.includes(data.stage))completedStages.push(data.stage);
  if(data.state==='running')currentStage=data.stage;
  if(data.state==='failed'&&!completedStages.includes(data.stage))completedStages.push('x '+data.stage);
  var total=3+($('oT').checked?1:0)+($('oC').checked?1:0)+($('oL').checked?1:0);
  var parts=completedStages.map(function(m){return(m.startsWith('x ')?'':'+ ')+m;});
  if(currentStage&&!completedStages.includes(currentStage))parts.push('> '+currentStage);
  $('st').innerHTML='<span class="spinner"></span> '+parts.join('  ')+(parts.length?' ('+completedStages.length+'/'+total+')':'initializing...');
});
socket.on('done',function(data){
  stopTimeoutWatch();showPredictions(data.result);$('st').innerHTML='';$('btn').disabled=false;currentJobId=null;loadCacheStatus();
});
socket.on('error',function(data){
  stopTimeoutWatch();showError(data.message||'Server error');$('btn').disabled=false;currentJobId=null;
});

function startTimeoutWatch(){
  stopTimeoutWatch();lastProgressTs=Date.now();
  timeoutInterval=setInterval(function(){
    if(Date.now()-lastProgressTs>120000)$('st').innerHTML='<span style="color:var(--warn)">No response for 2 min</span>';
  },10000);
}
function stopTimeoutWatch(){if(timeoutInterval){clearInterval(timeoutInterval);timeoutInterval=null;}}

function $(id){return document.getElementById(id)}
function val(id){return $(id).value.trim().toUpperCase()}
function escHtml(s){if(!s)return'';var d=document.createElement('div');d.appendChild(document.createTextNode(String(s)));return d.innerHTML}

function syncToggle(cbId,togId){$(togId).classList.toggle('on',$(cbId).checked);}
function toggleHelp(){$('helpPanel').classList.toggle('open');}
function checkResources(){
  var dr=val('dataRange'),tfm=$('oT').checked,lstm=$('oL').checked,w=$('warnBox');
  var m=[];
  if(dr==='2y'||dr==='5y'||dr==='max')m.push('Large data range');
  if(tfm)m.push('TimesFM ~2 min');if(lstm)m.push('LSTM ~2 min');
  w.innerHTML=m.length?m.map(function(s){return'! '+s;}).join(' / '):'';
  w.style.display=m.length?'block':'none';
}
function setChartType(t){
  chartType=t;
  $('btnCandle').classList.toggle('active',t==='candle');
  $('btnLine').classList.toggle('active',t==='line');
  if($('res').classList.contains('open'))run();
}

// ── Chart Overlays ──
var _renderMode='server',_chartJS=null,_rsiJS=null,_chartData=null,_lastPreds=null,_lastHz=12;
var _overlays={bb:true,ema:true,trend:true,fvg:true,ifvg:false,rsi:false};

var _togIds={bb:'togBB',ema:'togEMA',trend:'togTrend',fvg:'togFVG',ifvg:'togIFVG',rsi:'togRSI'};
function toggleOverlay(key){
  _overlays[key]=!_overlays[key];
  $(_togIds[key]).classList.toggle('on',_overlays[key]);
  if(key==='rsi'){
    $('rsiBody').classList.toggle('show',_overlays.rsi);
    if(_overlays.rsi&&_chartData)renderRSI(_chartData);
    else if(!_overlays.rsi&&_rsiJS){_rsiJS.destroy();_rsiJS=null;}
  }
  if(_chartData&&_renderMode==='browser')renderChart(_chartData,_lastPreds,_lastHz);
}

function setRenderMode(m){
  _renderMode=m;
  $('btnBrowser').classList.toggle('active',m==='browser');
  $('btnServer').classList.toggle('active',m==='server');
  $('chartOverlays').style.display=m==='browser'?'flex':'none';
  var img=$('chart'),cvs=$('chartCanvas');
  if(m==='browser'){
    img.style.display='none';cvs.style.display='block';
    var t=val('target');if(t)fetchChartData(t);
  }else{img.style.display='block';cvs.style.display='none';destroyChart();}
}
function destroyChart(){
  if(_chartJS){_chartJS.destroy();_chartJS=null;}
  if(_rsiJS){_rsiJS.destroy();_rsiJS=null;}
}
function fetchChartData(ticker){
  fetch('/api/chart-data/'+encodeURIComponent(ticker)+'?data_range='+encodeURIComponent(val('dataRange')))
    .then(function(r){return r.json();}).then(function(d){
      if(d.error){console.error('chart-data:',d.error);return;}
      _chartData=d;renderChart(d,_lastPreds,_lastHz);
      if(_overlays.rsi)renderRSI(d);
    }).catch(function(e){console.error('fetchChartData failed:',e);});
}

function futureDates(lastDateStr,n){
  var dates=[],d=new Date(lastDateStr+'T12:00:00Z');
  while(dates.length<n){
    d=new Date(d.getTime()+86400000);
    var dow=d.getUTCDay();
    if(dow!==0&&dow!==6)dates.push(d.toISOString().slice(0,10));
  }
  return dates;
}

/* ── Technical analysis helpers ── */
function detectSwingHighs(ohlc,lookback){
  lookback=lookback||3;var pts=[];
  for(var i=lookback;i<ohlc.length-lookback;i++){
    var h=ohlc[i].h,isHigh=true;
    for(var j=i-lookback;j<=i+lookback;j++){if(j!==i&&ohlc[j].h>=h){isHigh=false;break;}}
    if(isHigh)pts.push({i:i,x:ohlc[i].x,y:h});
  }
  return pts;
}
function detectSwingLows(ohlc,lookback){
  lookback=lookback||3;var pts=[];
  for(var i=lookback;i<ohlc.length-lookback;i++){
    var l=ohlc[i].l,isLow=true;
    for(var j=i-lookback;j<=i+lookback;j++){if(j!==i&&ohlc[j].l<=l){isLow=false;break;}}
    if(isLow)pts.push({i:i,x:ohlc[i].x,y:l});
  }
  return pts;
}

/* Build trend line datasets from swing points */
function buildTrendLines(ohlc){
  var datasets=[];
  var highs=detectSwingHighs(ohlc,3);
  var lows=detectSwingLows(ohlc,3);
  var last=ohlc[ohlc.length-1];

  /* Downtrend resistance: most recent descending pair of swing highs */
  if(highs.length>=2){
    var h1=highs[highs.length-2],h2=highs[highs.length-1];
    if(h2.y<h1.y){
      var slope=(h2.y-h1.y)/(h2.i-h1.i);
      var extY=h2.y+slope*(ohlc.length-1-h2.i);
      datasets.push({
        label:'Resistance',
        data:[{x:h1.x,y:h1.y},{x:h2.x,y:h2.y},{x:last.x,y:+(h2.y+slope*(ohlc.length-1-h2.i)).toFixed(4)}],
        borderColor:'rgba(248,113,113,.7)',backgroundColor:'transparent',
        borderWidth:1.5,pointRadius:[4,4,0],fill:false,tension:0,borderDash:[4,3],order:3
      });
    }
  }
  /* Uptrend support: most recent ascending pair of swing lows */
  if(lows.length>=2){
    var l1=lows[lows.length-2],l2=lows[lows.length-1];
    if(l2.y>l1.y){
      var slope2=(l2.y-l1.y)/(l2.i-l1.i);
      datasets.push({
        label:'Support',
        data:[{x:l1.x,y:l1.y},{x:l2.x,y:l2.y},{x:last.x,y:+(l2.y+slope2*(ohlc.length-1-l2.i)).toFixed(4)}],
        borderColor:'rgba(74,222,128,.7)',backgroundColor:'transparent',
        borderWidth:1.5,pointRadius:[4,4,0],fill:false,tension:0,borderDash:[4,3],order:3
      });
    }
  }
  return datasets;
}

/* Detect Fair Value Gaps (3-candle imbalance) */
function detectFVGs(ohlc){
  var bullish=[],bearish=[];
  for(var i=2;i<ohlc.length;i++){
    var c0=ohlc[i-2],c2=ohlc[i];
    if(c0.h<c2.l){bullish.push({x0:c0.x,x2:c2.x,lo:c0.h,hi:c2.l,i0:i-2,i2:i,filled:false});}
    if(c0.l>c2.h){bearish.push({x0:c0.x,x2:c2.x,lo:c2.h,hi:c0.l,i0:i-2,i2:i,filled:false});}
  }
  /* Mark filled FVGs (price returned into the gap) */
  bullish.forEach(function(g){
    for(var k=g.i2+1;k<ohlc.length;k++){if(ohlc[k].l<=g.lo)g.filled=true;}
  });
  bearish.forEach(function(g){
    for(var k=g.i2+1;k<ohlc.length;k++){if(ohlc[k].h>=g.hi)g.filled=true;}
  });
  return{bullish:bullish,bearish:bearish};
}

/* Custom Chart.js plugin to draw FVG / IFVG boxes */
var fvgPlugin={
  id:'fvgBoxes',
  afterDraw:function(chart){
    var meta=chart.getDatasetMeta(0);
    if(!meta||!meta.data||!meta.data.length)return;
    var xScale=chart.scales.x,yScale=chart.scales.y;
    if(!xScale||!yScale||!chart._fvgData)return;
    var ctx=chart.ctx,area=chart.chartArea;
    var fvgs=chart._fvgData,labels=chart.data.labels;

    function xPos(dateStr){
      var idx=labels.indexOf(dateStr);
      if(idx<0)return null;
      return xScale.getPixelForValue(idx);
    }
    function drawBox(x0px,x2px,lo,hi,fillColor){
      if(x0px===null||x2px===null)return;
      var y0=yScale.getPixelForValue(hi);
      var y1=yScale.getPixelForValue(lo);
      ctx.save();
      ctx.fillStyle=fillColor;
      ctx.fillRect(x0px,y0,x2px-x0px,y1-y0);
      ctx.restore();
    }

    ctx.save();ctx.beginPath();
    ctx.rect(area.left,area.top,area.right-area.left,area.bottom-area.top);
    ctx.clip();

    var showFVG=chart._showFVG,showIFVG=chart._showIFVG;
    if(showFVG){
      fvgs.bullish.forEach(function(g){
        if(!g.filled)drawBox(xPos(g.x0),xPos(g.x2),g.lo,g.hi,'rgba(74,222,128,.12)');
      });
      fvgs.bearish.forEach(function(g){
        if(!g.filled)drawBox(xPos(g.x0),xPos(g.x2),g.lo,g.hi,'rgba(248,113,113,.12)');
      });
    }
    if(showIFVG){
      fvgs.bullish.forEach(function(g){
        if(g.filled)drawBox(xPos(g.x0),xPos(g.x2),g.lo,g.hi,'rgba(74,222,128,.06)');
      });
      fvgs.bearish.forEach(function(g){
        if(g.filled)drawBox(xPos(g.x0),xPos(g.x2),g.lo,g.hi,'rgba(248,113,113,.06)');
      });
    }
    ctx.restore();
  }
};

function renderChart(data,predictions,horizon){
  destroyChart();
  var cvs=$('chartCanvas');if(!cvs)return;
  var ctx=cvs.getContext('2d');
  var ohlc=data&&data.ohlc||[];if(!ohlc.length)return;

  var closeData=ohlc.map(function(p){return{x:p.x,y:p.c};});
  var ema50=(data.indicators&&data.indicators.ema_50||[]).filter(function(p){return p.y!==null;});
  var bbU=(data.indicators&&data.indicators.bb_upper||[]).filter(function(p){return p.y!==null;});
  var bbL=(data.indicators&&data.indicators.bb_lower||[]).filter(function(p){return p.y!==null;});
  var bbM=(data.indicators&&data.indicators.bb_mid||[]).filter(function(p){return p.y!==null;});

  var datasets=[{
    label:'Price',data:closeData,borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,.06)',
    borderWidth:1.5,pointRadius:0,fill:true,tension:0.1,order:10
  }];

  if(_overlays.ema&&ema50.length)
    datasets.push({label:'EMA 50',data:ema50,borderColor:'#f97316',borderWidth:1.5,pointRadius:0,fill:false,tension:0.1,order:9});

  if(_overlays.bb&&bbU.length){
    datasets.push({label:'BB Upper',data:bbU,borderColor:'rgba(125,211,252,.5)',borderWidth:1,pointRadius:0,fill:false,tension:0.1,borderDash:[3,3],order:8});
    if(bbM.length)datasets.push({label:'BB Mid',data:bbM,borderColor:'rgba(125,211,252,.25)',borderWidth:1,pointRadius:0,fill:false,tension:0.1,borderDash:[2,4],order:8});
    if(bbL.length)datasets.push({label:'BB Lower',data:bbL,borderColor:'rgba(125,211,252,.5)',borderWidth:1,pointRadius:0,fill:'-2',backgroundColor:'rgba(125,211,252,.04)',tension:0.1,order:8});
  }

  if(_overlays.trend){
    var trendDS=buildTrendLines(ohlc);
    trendDS.forEach(function(ds){datasets.push(ds);});
  }

  var allLabels=ohlc.map(function(p){return p.x;});

  if(predictions&&predictions.length&&ohlc.length){
    var lastClose=ohlc[ohlc.length-1].c;
    var lastDate=ohlc[ohlc.length-1].x;
    var hz=Math.max(horizon||12,2);
    var fDates=futureDates(lastDate,hz);
    fDates.forEach(function(d){allLabels.push(d);});
    var MODEL_COLORS={'Chronos-2':'#38bdf8','Neural LSTM':'#a78bfa','TimesFM 2.5':'#fbbf24'};
    var avgTarget=0,cnt=0;
    predictions.forEach(function(pred){
      var color=MODEL_COLORS[pred.model]||'#94a3b8';
      var predData=[{x:lastDate,y:lastClose}];
      for(var i=0;i<fDates.length;i++){
        var frac=(i+1)/fDates.length;
        predData.push({x:fDates[i],y:+(lastClose+(pred.target-lastClose)*frac).toFixed(4)});
      }
      avgTarget+=pred.target;cnt++;
      datasets.push({label:pred.model,data:predData,borderColor:color,backgroundColor:'transparent',borderWidth:2,pointRadius:0,pointHoverRadius:4,fill:false,tension:0.3,borderDash:[5,4],order:5});
    });
    if(cnt>1){
      var ensTarget=avgTarget/cnt;
      var ensData=[{x:lastDate,y:lastClose}];
      for(var j=0;j<fDates.length;j++){
        var fr=(j+1)/fDates.length;
        ensData.push({x:fDates[j],y:+(lastClose+(ensTarget-lastClose)*fr).toFixed(4)});
      }
      datasets.push({label:'Ensemble',data:ensData,borderColor:'#fff',backgroundColor:'transparent',borderWidth:2.5,pointRadius:0,pointHoverRadius:5,fill:false,tension:0.3,borderDash:[8,3],order:4});
    }
  }

  var fvgData=(_overlays.fvg||_overlays.ifvg)?detectFVGs(ohlc):null;

  _chartJS=new Chart(ctx,{
    type:'line',
    data:{labels:allLabels,datasets:datasets},
    options:{
      responsive:true,maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{
        legend:{
          display:true,position:'top',
          labels:{color:'#6b6b80',font:{size:10},boxWidth:16,padding:12,
            filter:function(i){
              var hide=['BB Upper','BB Mid','BB Lower'];
              return hide.indexOf(i.text)<0;
            }}
        },
        tooltip:{
          backgroundColor:'#15151f',titleColor:'#22c55e',bodyColor:'#c9c9d8',
          borderColor:'#1e1e2e',borderWidth:1,padding:10,
          callbacks:{label:function(c){return' '+c.dataset.label+': $'+Number(c.parsed.y).toFixed(2);}}
        },
        fvgBoxes:{}
      },
      scales:{
        x:{type:'category',ticks:{color:'#3a3a4e',maxRotation:0,autoSkip:true,maxTicksLimit:8,font:{size:10}},grid:{color:'#1e1e2e'}},
        y:{position:'right',ticks:{color:'#3a3a4e',font:{size:10},callback:function(v){return'$'+v.toFixed(0);}},grid:{color:'#1e1e2e'}}
      }
    },
    plugins:[fvgPlugin]
  });

  /* Attach FVG data to chart instance */
  _chartJS._fvgData=fvgData;
  _chartJS._showFVG=_overlays.fvg;
  _chartJS._showIFVG=_overlays.ifvg;
  _chartJS.update();
}

/* RSI sub-chart */
function renderRSI(data){
  if(_rsiJS){_rsiJS.destroy();_rsiJS=null;}
  var cvs=$('rsiCanvas');if(!cvs)return;
  var rsiData=(data.indicators&&data.indicators.rsi||[]).filter(function(p){return p.y!==null;});
  if(!rsiData.length)return;
  var labels=rsiData.map(function(p){return p.x;});
  _rsiJS=new Chart(cvs.getContext('2d'),{
    type:'line',
    data:{labels:labels,datasets:[
      {label:'RSI 14',data:rsiData.map(function(p){return{x:p.x,y:p.y};}),borderColor:'#facc15',borderWidth:1.5,pointRadius:0,fill:false,tension:0.1},
      {label:'OB',data:labels.map(function(x){return{x:x,y:70};}),borderColor:'rgba(248,113,113,.4)',borderWidth:1,pointRadius:0,fill:false,borderDash:[3,3]},
      {label:'OS',data:labels.map(function(x){return{x:x,y:30};}),borderColor:'rgba(74,222,128,.4)',borderWidth:1,pointRadius:0,fill:false,borderDash:[3,3]}
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{enabled:false}},
      scales:{
        x:{type:'category',display:false,ticks:{maxTicksLimit:0},grid:{display:false}},
        y:{position:'right',min:0,max:100,ticks:{color:'#3a3a4e',font:{size:9},stepSize:30,callback:function(v){return v;}},grid:{color:'#1e1e2e'}}
      }
    }
  });
}

function loadCacheStatus(){
  fetch('/api/cache-status').then(function(r){return r.json();}).then(function(d){
    var on=d.chronos_cached;
    $('cacheInfo').innerHTML='<span class="cdot'+(on?' on':'')+'"></span>Chronos '+(on?'cached':'not cached');
  }).catch(function(){});
}

function run(){
  var t=val('target'),h=val('helper'),hz=val('horizon');
  if(!t){alert('Enter a ticker');return;}
  $('btn').disabled=true;
  $('st').innerHTML='<span class="spinner"></span> connecting...';
  $('errBox').style.display='none';
  $('res').classList.remove('open');
  $('chart').src='';_lastPreds=null;
  $('pBody').innerHTML='<div class="row"><span class="rk">waiting...</span></div>';
  $('tBody').innerHTML='<div class="row"><span class="rk">waiting...</span></div>';
  $('aBody').innerHTML='<div class="row"><span class="rk">loading...</span></div>';
  $('nList').innerHTML='<li class="news-item" style="color:var(--text-muted)">loading...</li>';
  $('res').classList.add('open');
  completedStages=[];currentStage='';lastProgressTs=Date.now();
  fetch('/api/news/'+encodeURIComponent(t)).then(function(r){return r.json();}).then(function(d){showAnalyst(d);showNews(d);}).catch(function(){});
  socket.emit('start_forecast',{target:t,helper:h,horizon:hz,dataRange:val('dataRange'),chartType:chartType,use_timesfm:$('oT').checked,use_chronos:$('oC').checked,use_lstm:$('oL').checked});
}

function showError(msg){$('errBox').innerHTML=escHtml(msg);$('errBox').style.display='block';$('st').innerHTML='';}
function clearError(){$('errBox').style.display='none';}

function renderAIBand(ai){
  if(!ai)return'';
  var cls=ai.label==='Strong Buy'||ai.label==='Buy'?'bull':ai.label==='Strong Sell'||ai.label==='Sell'?'bear':'neut';
  var sub=ai.label==='Strong Buy'?'Strong signal':ai.label==='Buy'?'Positive signal':ai.label==='Sell'?'Negative signal':ai.label==='Strong Sell'?'Strong negative signal':'Mixed signals';
  var comps='';
  if(ai.components){
    var c=ai.components;
    comps='<div class="ai-comps-col">'
      +'<div class="ai-comp"><span class="ck">Tech</span><span class="cv">'+escHtml(String(c.technical))+'</span></div>'
      +'<div class="ai-comp"><span class="ck">Sent</span><span class="cv">'+escHtml(String(c.sentiment))+'</span></div>'
      +'<div class="ai-comp"><span class="ck">Analyst</span><span class="cv">'+escHtml(String(c.analyst))+'</span></div>'
      +'<div class="ai-comp"><span class="ck">Mom</span><span class="cv">'+escHtml(String(c.momentum))+'</span></div>'
      +'</div>';
  }
  return'<div class="ai-band">'
    +'<div class="ai-score-col"><div class="ai-score-lbl">AI Score</div><div class="ai-score-num '+cls+'">'+escHtml(String(ai.score))+'</div></div>'
    +'<div class="ai-verdict-col"><div class="ai-verdict">'+escHtml(ai.emoji)+' '+escHtml(ai.label)+'</div><div class="ai-verdict-sub">'+sub+'</div></div>'
    +comps+'</div>';
}

function showPredictions(d){
  clearError();
  /* Update AI band in preview (single, authoritative) */
  if(d.ai_score){
    var cur=$('previewContent');
    var existing=cur.querySelector('.ai-band');
    var newBand=renderAIBand(d.ai_score);
    if(existing)existing.outerHTML=newBand;
    else cur.insertAdjacentHTML('afterbegin',newBand);
    $('previewPanel').classList.add('open');
  }
  if(d.chart_url)$('chart').src=d.chart_url+'?t='+Date.now();
  _lastPreds=d.predictions||null;_lastHz=d.horizon||12;
  if(_renderMode==='browser'){
    if(_chartData)renderChart(_chartData,_lastPreds,_lastHz);
    else{var t=val('target');if(t)fetchChartData(t);}
  }
  /* Predictions card */
  var ph='';
  (d.predictions||[]).forEach(function(p){
    var c=p.delta>=0?'up':'down';
    ph+='<div class="row"><span class="rk">'+escHtml(p.model)+'</span><span class="rv '+c+'">$'+p.target.toFixed(2)+' ('+(p.delta>=0?'+':'')+p.delta.toFixed(2)+'%)</span></div>';
  });
  if(ph)ph+='<div class="row" style="border-top:1px solid var(--border2);margin-top:4px;padding-top:4px">'
    +'<span class="rk">Consensus</span><span class="rv">'+escHtml(d.consensus||'N/A')+'</span></div>'
    +'<div class="row"><span class="rk">Avg delta</span><span class="rv '+(d.avg_delta>=0?'up':'down')+'">'+(d.avg_delta||0).toFixed(2)+'%</span></div>';
  if(!ph)ph='<div class="row"><span class="rk" style="color:var(--text-muted)">No prediction data</span></div>';
  $('pBody').innerHTML=ph;
  /* Technicals card */
  var t2=d.technicals||{},th='';
  th+='<div class="row"><span class="rk">RSI 14</span><span class="rv">'+(t2.rsi!=null?t2.rsi.toFixed(1):'--')+' <span style="font-size:10px;color:var(--text-muted)">'+(t2.rsi_label||'')+'</span></span></div>';
  th+='<div class="row"><span class="rk">EMA 50</span><span class="rv '+(t2.ema_label==='BEARISH'?'down':t2.ema_label==='BULLISH'?'up':'')+'">'+escHtml(t2.ema_label||'--')+'</span></div>';
  th+='<div class="row"><span class="rk">TD Setup</span><span class="rv '+(t2.td_count===9?'warn':'')+'">'+escHtml(t2.td_label||'--')+'</span></div>';
  th+='<div class="row"><span class="rk">Volatility</span><span class="rv '+(t2.vol_label==='OVEREXTENDED'?'warn':'')+'">'+escHtml(t2.vol_label||'--')+'</span></div>';
  var ss2=d.sentiment_summary;
  if(ss2){
    var sc2=ss2.label==='Bullish'?'up':ss2.label==='Bearish'?'down':'';
    var sc2s=ss2.avg_compound!=null?(parseFloat(ss2.avg_compound)*100).toFixed(0):'';
    th+='<div class="row" style="border-top:1px solid var(--border2);margin-top:4px;padding-top:4px">'
      +'<span class="rk">Sentiment</span>'
      +'<span class="rv '+sc2+'">'+escHtml(ss2.emoji)+' '+escHtml(ss2.label)+(sc2s?' ('+sc2s+')':'')+'</span></div>';
  }
  $('tBody').innerHTML=th;
}

function showAnalyst(d){
  var ah='';
  var e=d.earnings||{};
  if(e.available){
    var dc=e.days_until<=7?'var(--danger)':e.days_until<=14?'var(--warn)':e.days_until<=30?'#eab308':'var(--accent)';
    ah+='<div class="row"><span class="rk">Earnings</span><span class="rv" style="color:'+dc+'">'+escHtml(e.date)+' ('+e.days_until+'d)</span></div>';
  }
  var a=d.analyst||{};
  if(a.available){
    var tot=a.total||1;
    var b2=((a.strong_buy+a.buy)/tot*100).toFixed(0),h2=(a.hold/tot*100).toFixed(0),s=((a.sell+a.strong_sell)/tot*100).toFixed(0);
    ah+='<div class="analyst-bar">'
      +'<div class="bar-buy" style="width:'+b2+'%">'+(b2>12?b2+'%':'')+'</div>'
      +'<div class="bar-hold" style="width:'+h2+'%">'+(h2>12?h2+'%':'')+'</div>'
      +'<div class="bar-sell" style="width:'+s+'%">'+(s>12?s+'%':'')+'</div></div>';
    ah+='<div class="a-legend"><span class="al-buy">buy '+(a.strong_buy+a.buy)+'</span><span class="al-hold">hold '+a.hold+'</span><span class="al-sell">sell '+(a.sell+a.strong_sell)+'</span></div>';
  }else if(!e.available){
    ah='<div class="row"><span class="rk" style="color:var(--text-muted)">No analyst data</span></div>';
  }
  $('aBody').innerHTML=ah;
}

function showNews(d){
  var nh='';
  var ss=d.sentiment_summary;
  if(ss){
    var cls=ss.label==='Bullish'?'bull':ss.label==='Bearish'?'bear':'neut';
    var score=ss.avg_compound!=null?(parseFloat(ss.avg_compound)*100).toFixed(0):'';
    var trendH='';
    if(ss.trend_emoji&&ss.trend_label)trendH='<span class="trend-pill trend-'+escHtml(ss.trend)+'">'+escHtml(ss.trend_emoji)+' '+escHtml(ss.trend_label)+'</span>';
    nh+='<div class="news-sent-row"><span class="sent-lbl">Sentiment</span><span class="ss-label '+cls+'">'+escHtml(ss.emoji)+' '+escHtml(ss.label)+(score?' ('+score+')':'')+'</span>'+trendH+'</div>';
    var items=d.news||[],bull=0,bear=0,neut=0;
    items.forEach(function(n){var s=n.sentiment;if(s){if(s.compound>0.05)bull++;else if(s.compound<-0.05)bear++;else neut++;}});
    if(bull+bear+neut>0){
      nh+='<div class="news-counts">';
      if(bull)nh+='<span>+ '+bull+'</span>';if(neut)nh+='<span>o '+neut+'</span>';if(bear)nh+='<span>- '+bear+'</span>';
      nh+='</div>';
    }
  }
  (d.news||[]).forEach(function(n){
    var sc='';if(n.sentiment){var c=n.sentiment.compound;sc=c>0.05?' [+]':c<-0.05?' [-]':' [o]';}
    nh+='<li class="news-item"><a href="'+escHtml(n.url||'#')+'" target="_blank">'+escHtml(n.title||'')+sc+'</a><div class="meta">'+escHtml(n.date||'')+'</div></li>';
  });
  if(!nh)nh='<li class="news-item" style="color:var(--text-muted)">No recent news</li>';
  $('nList').innerHTML=nh;
}

// Preview panel
var prevTimer=null;
function loadPreview(ticker){
  if(!ticker||ticker.length<1)return;
  var panel=$('previewPanel'),content=$('previewContent');
  panel.classList.add('open');
  $('prevTicker').textContent=ticker;$('prevName').textContent='';$('prevPrice').textContent='';
  content.innerHTML='<div style="color:var(--text-muted);font-size:12px;padding:6px 0"><span class="spinner"></span> Loading...</div>';
  fetch('/api/preview/'+encodeURIComponent(ticker)+'?data_range='+encodeURIComponent(val('dataRange')))
    .then(function(r){return r.json();}).then(function(d){
      if(d.error){content.innerHTML='<div style="color:var(--danger);padding:6px 0">'+escHtml(d.error)+'</div>';return;}
      $('prevTicker').textContent=ticker;$('prevName').textContent=d.name||'';
      $('prevPrice').textContent=d.current_price?'$'+d.current_price:'';
      var h='';
      if(d.ai_score)h+=renderAIBand(d.ai_score);
      h+='<div class="tech-grid">';
      var rsiCls=d.rsi>70?'warn':d.rsi<30?'up':'';
      h+='<div class="tech-item"><span class="tk">RSI 14</span><span class="tv '+rsiCls+'">'+(d.rsi!=null?d.rsi.toFixed(1):'--')+'</span></div>';
      h+='<div class="tech-item"><span class="tk">EMA 50</span><span class="tv '+(d.ema_label==='BEARISH'?'down':d.ema_label==='BULLISH'?'up':'')+'">'+escHtml(d.ema_label||'--')+'</span></div>';
      h+='<div class="tech-item"><span class="tk">TD Setup</span><span class="tv '+(d.td_count===9?'warn':'')+'">'+escHtml(d.td_label||'--')+'</span></div>';
      h+='<div class="tech-item"><span class="tk">Volatility</span><span class="tv '+(d.vol_label==='OVEREXTENDED'?'warn':'')+'">'+escHtml(d.vol_label||'--')+'</span></div>';
      h+='<div class="tech-item"><span class="tk">52wk High</span><span class="tv">'+(d['52wk_high']?'$'+d['52wk_high'].toFixed(2):'--')+'</span></div>';
      h+='<div class="tech-item"><span class="tk">52wk Low</span><span class="tv">'+(d['52wk_low']?'$'+d['52wk_low'].toFixed(2):'--')+'</span></div>';
      h+='</div>';
      if(d['52wk_high']&&d['52wk_low']&&d.current_price){
        var lo=d['52wk_low'],hi=d['52wk_high'],cur=d.current_price;
        var pct=((cur-lo)/(hi-lo)*100).toFixed(0);
        h+='<div class="wk52"><span class="wlo">$'+lo.toFixed(2)+'</span>'
          +'<div class="wk52-bar"><div class="wk52-pos" style="left:'+pct+'%"></div></div>'
          +'<span class="whi">$'+hi.toFixed(2)+'</span><span class="wpct">'+pct+'% of range</span></div>';
      }
      if(d.sentiment_summary){
        var ss=d.sentiment_summary;
        var cls=ss.label==='Bullish'?'bull':ss.label==='Bearish'?'bear':'neut';
        var sc=ss.avg_compound!=null?(parseFloat(ss.avg_compound)*100).toFixed(0):'';
        var tH='';if(ss.trend_emoji&&ss.trend_label)tH='<span class="trend-pill trend-'+escHtml(ss.trend)+'">'+escHtml(ss.trend_emoji)+' '+escHtml(ss.trend_label)+'</span>';
        h+='<div class="sent-row"><span class="sent-lbl">Sentiment</span><span class="ss-label '+cls+'">'+escHtml(ss.emoji)+' '+escHtml(ss.label)+(sc?' ('+sc+')':'')+'</span>'+tH+'</div>';
      }
      if(d.fundamentals&&d.fundamentals.available){
        var f=d.fundamentals;
        h+='<div class="fund-hdr">Fundamentals</div><div class="fund-grid">';
        if(f.pe_trailing!=null)h+='<div class="fund-item"><span class="fk">P/E</span><span class="fv">'+f.pe_trailing.toFixed(1)+'</span></div>';
        if(f.pe_forward!=null)h+='<div class="fund-item"><span class="fk">Fwd P/E</span><span class="fv '+(f.pe_forward<0?'down':'')+'">'+f.pe_forward.toFixed(1)+'</span></div>';
        if(f.eps!=null)h+='<div class="fund-item"><span class="fk">EPS</span><span class="fv '+(f.eps<0?'down':'up')+'">$'+escHtml(String(f.eps))+'</span></div>';
        if(f.market_cap_fmt)h+='<div class="fund-item"><span class="fk">Mkt Cap</span><span class="fv">'+escHtml(f.market_cap_fmt)+'</span></div>';
        if(f.revenue_fmt)h+='<div class="fund-item"><span class="fk">Revenue</span><span class="fv">'+escHtml(f.revenue_fmt)+'</span></div>';
        if(f.profit_margin_fmt)h+='<div class="fund-item"><span class="fk">Margin</span><span class="fv '+(f.profit_margin!=null&&f.profit_margin<0?'down':f.profit_margin!=null&&f.profit_margin>0.15?'up':'')+'">'+escHtml(f.profit_margin_fmt)+'</span></div>';
        if(f.revenue_growth_fmt)h+='<div class="fund-item"><span class="fk">Rev Growth</span><span class="fv '+(f.revenue_growth!=null&&f.revenue_growth>0?'up':f.revenue_growth!=null&&f.revenue_growth<0?'down':'')+'">'+escHtml(f.revenue_growth_fmt)+'</span></div>';
        if(f.beta!=null)h+='<div class="fund-item"><span class="fk">Beta</span><span class="fv '+(f.beta>2?'warn':f.beta<0.5?'up':'')+'">'+f.beta.toFixed(2)+'</span></div>';
        h+='</div>';
      }
      content.innerHTML=h;syncWatchBtn(ticker);
    }).catch(function(){content.innerHTML='<div style="color:var(--danger);padding:6px 0">Preview failed</div>';});
}

function onTargetChange(){
  var t=val('target');
  if(t&&t.length>=1){clearTimeout(prevTimer);prevTimer=setTimeout(function(){loadPreview(t);},500);}
  else $('previewPanel').classList.remove('open');
}

// Watchlist
var _wl=[];
function loadWatchlist(){try{_wl=JSON.parse(localStorage.getItem('wl2')||'[]');}catch(e){_wl=[];}renderWatchlist();}
function saveWatchlist(){localStorage.setItem('wl2',JSON.stringify(_wl));renderWatchlist();}
function isWatched(t){return _wl.indexOf(t.toUpperCase())>=0;}
function addToWatchlist(t){t=t.toUpperCase();if(_wl.indexOf(t)<0){_wl.push(t);saveWatchlist();}}
function removeFromWatchlist(t){t=t.toUpperCase();var i=_wl.indexOf(t);if(i>=0){_wl.splice(i,1);saveWatchlist();}}
function syncWatchBtn(t){renderWatchlist();}
function renderWatchlist(){
  var bar=$('watchlistBar');if(!bar)return;bar.innerHTML='';
  var t=val('target');
  if(t&&!isWatched(t)){
    var a=document.createElement('div');a.className='wl-add';a.textContent='+ Watch '+t;
    a.onclick=function(){addToWatchlist(t);syncWatchBtn(t);};bar.appendChild(a);
  }else if(t&&isWatched(t)){
    var a=document.createElement('div');a.className='wl-add watching';a.textContent='* '+t;
    a.onclick=function(){removeFromWatchlist(t);syncWatchBtn(t);};bar.appendChild(a);
  }
  _wl.forEach(function(sym){
    var chip=document.createElement('div');chip.className='wl-chip'+(sym===t?' cur':'');
    chip.onclick=function(){$('target').value=sym;onTargetChange();};
    var lbl=document.createElement('span');lbl.textContent=sym;chip.appendChild(lbl);
    var x=document.createElement('span');x.className='x';x.textContent='x';
    x.onclick=function(e){e.stopPropagation();removeFromWatchlist(sym);syncWatchBtn(sym);};
    chip.appendChild(x);bar.appendChild(chip);
  });
}

// Autocomplete
var acTimers={},acSel={};
function setupAC(inputId,listId){
  var input=$(inputId),list=$(listId),idx=-1;
  input.addEventListener('input',function(){
    var q=input.value.trim();
    clearTimeout(acTimers[inputId]);idx=-1;
    if(q.length<2){list.classList.remove('open');list.innerHTML='';return;}
    acTimers[inputId]=setTimeout(function(){
      fetch('/api/search?q='+encodeURIComponent(q)).then(function(r){return r.json();}).then(function(d){
        var results=d.results||[];
        if(!results.length){list.classList.remove('open');list.innerHTML='';return;}
        list.innerHTML='';
        results.forEach(function(item,i){
          var div=document.createElement('div');div.className='acitem';
          div.innerHTML='<span class="sym">'+escHtml(item.symbol)+'</span>'+escHtml(item.name||'')+'<span class="exch">'+escHtml(item.exchange||'')+'</span>';
          div.addEventListener('mousedown',function(e){
            e.preventDefault();input.value=item.symbol;list.classList.remove('open');list.innerHTML='';
            if(inputId==='target')onTargetChange();
          });
          list.appendChild(div);
        });
        list.classList.add('open');
      }).catch(function(){});
    },200);
  });
  input.addEventListener('keydown',function(e){
    var items=list.querySelectorAll('.acitem');if(!items.length)return;
    if(e.key==='ArrowDown'){e.preventDefault();idx=Math.min(idx+1,items.length-1);hilite(items);}
    else if(e.key==='ArrowUp'){e.preventDefault();idx=Math.max(idx-1,-1);hilite(items);}
    else if(e.key==='Enter'&&idx>=0){e.preventDefault();items[idx].dispatchEvent(new MouseEvent('mousedown'));}
    else if(e.key==='Escape'){list.classList.remove('open');list.innerHTML='';}
  });
  input.addEventListener('blur',function(){setTimeout(function(){list.classList.remove('open');},200);});
  input.addEventListener('focus',function(){if(list.innerHTML.trim())list.classList.add('open');});
  function hilite(items){items.forEach(function(it,i){it.classList.toggle('sel',i===idx);});if(idx>=0&&items[idx])items[idx].scrollIntoView({block:'nearest'});}
}

setupAC('target','targetAC');
setupAC('helper','helperAC');
loadWatchlist();
var _init=val('target');if(_init)loadPreview(_init);
document.getElementById('target').addEventListener('change',onTargetChange);
loadCacheStatus();
</script>
</div>
</body></html>"""


if __name__ == "__main__":
    print(f"⚡ Forecast GUI v2 → http://localhost:{PORT}")
    print(f"   WebSocket + Model Caching + Error Display")
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False, allow_unsafe_werkzeug=True)