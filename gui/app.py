#!/usr/bin/env python3
"""
Stock Forecast GUI v2 — WebSocket + Model Caching + Error Display

Changes from v1:
  - Flask-SocketIO for real-time progress (replaces 3s polling)
  - Chronos-2 model cached in-process (subprocess for data download only)
  - Error events and 2-min timeout in frontend

Run:  python gui/app.py [port]
"""
import os, sys, json, re, time, subprocess, threading, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_socketio import SocketIO, emit as socketio_emit
import numpy as np

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
            df_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            target_vals = df_data['close'].values.astype(np.float32)
            X_cal = np.arange(len(target_vals)-30, len(target_vals)).reshape(-1, 1)
            y_cal = target_vals[-30:].ravel()
            mapie = SplitConformalRegressor(
                estimator=LinearRegression().fit(X_cal, y_cal),
                confidence_level=0.9, prefit=True
            )
            mapie.conformalize(X_cal, y_cal)
            _, y_pis = mapie.predict_interval(
                np.arange(len(target_vals), len(target_vals)+horizon).reshape(-1, 1)
            )
            band_lo = y_pis[:, 0, 0]
            band_hi = y_pis[:, 1, 0]

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
    except Exception as e:
        _emit_progress(jid, 'news', 'failed', f'[STATUS:news:failed] {e}')
        result["news"] = []
        result["analyst"] = {"available": False}
        result["earnings"] = {"available": False}

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
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart-ready/<path:chart_name>")
def api_chart_ready(chart_name):
    p = WORKSPACE / chart_name
    return jsonify({"ready": p.exists() and p.stat().st_size > 1000})


@app.route("/test")
def test():
    return send_file(WORKSPACE / "gui" / "test_standalone.html")


# ── HTML (dark cyberpunk theme, WebSocket-based) ──────────────
PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>⚡ Forecast v2</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'SF Mono','Fira Code','Cascadia Code',monospace;background:#0c0c0c;color:#c0c0c0;min-height:100vh;font-size:13px}
.container{max-width:1080px;margin:0 auto;padding:16px}
h1{font-size:20px;padding:16px 0 4px;color:#00ff88;font-weight:400;letter-spacing:2px;text-transform:uppercase}
h1 span{color:#ff6b35}
.subtitle{color:#555;margin-bottom:20px;font-size:11px;letter-spacing:1px}
.form{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:10px;align-items:flex-start}
.field{display:flex;flex-direction:column;gap:3px}
.field label{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:0.5px;padding-left:2px}
.form input,.form select{background:#1a1a1a;border:1px solid #333;color:#00ff88;padding:8px 12px;border-radius:2px;font-size:13px;font-family:inherit;outline:none;width:110px}
.form input:focus,.form select:focus{border-color:#00ff88}
.form input::placeholder{color:#444}
.form button{background:#00ff88;color:#0c0c0c;border:none;padding:8px 20px;border-radius:2px;font-size:13px;font-family:inherit;font-weight:700;cursor:pointer;text-transform:uppercase;letter-spacing:1px}
.form button:hover{opacity:0.85}
.form button:disabled{opacity:0.3;cursor:not-allowed}
.chart-toggle{display:flex;gap:4px;justify-content:center;margin-bottom:10px}
.chart-toggle button{background:#1a1a1a;border:1px solid #333;color:#666;padding:4px 12px;border-radius:2px;font-size:11px;font-family:inherit;cursor:pointer;text-transform:uppercase;letter-spacing:0.5px}
.chart-toggle button.active{background:#00ff88;color:#0c0c0c;border-color:#00ff88}
.toggles{display:flex;gap:12px;justify-content:center;margin-bottom:14px;flex-wrap:wrap}
.toggles label{display:flex;align-items:center;gap:5px;color:#666;font-size:11px;cursor:pointer;text-transform:uppercase;letter-spacing:0.5px}
.toggles input[type="checkbox"]{accent-color:#00ff88}
.status{text-align:center;padding:8px;min-height:30px;font-size:12px;color:#00ff88}
.results{display:none}
.results.active{display:block}
.chart-box{background:#111;border:1px solid #222;border-radius:2px;padding:10px;margin-bottom:16px;text-align:center}
.chart-box img{max-width:100%;border-radius:2px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin-bottom:16px}
.card{background:#111;border:1px solid #222;border-radius:2px;padding:14px}
.card h3{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px;border-bottom:1px solid #1a1a1a;padding-bottom:6px}
.row{display:flex;justify-content:space-between;padding:4px 0;font-size:12px;border-bottom:1px solid #151515}
.row:last-child{border-bottom:none}
.val{font-weight:600}.up{color:#00ff88}.down{color:#ff4444}
.news-list{list-style:none}
.news-list li{padding:8px 0;border-bottom:1px solid #151515}
.news-list li:last-child{border-bottom:none}
.news-list a{color:#4af;text-decoration:none;font-size:12px;line-height:1.4}
.news-list a:hover{color:#00ff88}
.news-list .meta{color:#444;font-size:10px;margin-top:2px}
.analyst-bar{display:flex;height:18px;border-radius:2px;overflow:hidden;margin:8px 0 4px}
.analyst-bar div{display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;color:#0c0c0c}
.bar-buy{background:#00ff88}.bar-hold{background:#555}.bar-sell{background:#ff4444}
.legend{display:flex;gap:10px;font-size:10px;color:#555}
.legend span::before{content:'';display:inline-block;width:8px;height:8px;border-radius:1px;margin-right:3px;vertical-align:middle}
.l-buy::before{background:#00ff88}.l-hold::before{background:#555}.l-sell::before{background:#ff4444}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid #333;border-top-color:#00ff88;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.error-msg{color:#ff4444;background:#1a0a0a;border:1px solid #444;border-radius:2px;padding:10px 14px;font-size:12px;margin:8px 0;text-align:left;font-family:inherit;line-height:1.5}
.timeout-warn{color:#ff6b35;font-size:11px;margin-top:4px}
.cache-status{font-size:10px;color:#555;text-align:center;margin-bottom:8px}
.cache-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle}
.cache-dot.on{background:#00ff88;box-shadow:0 0 4px #00ff88}
.cache-dot.off{background:#444}
.ws-status{font-size:10px;color:#333;text-align:right;float:right;margin-top:-20px;margin-bottom:10px}
.ws-status.connected{color:#00ff88}
.ws-status.disconnected{color:#ff4444}
</style>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
</head><body>
<div class="container">
  <h1>⚡ <span>Forecast</span> <span style="font-size:10px;color:#444">v2</span></h1>
  <p class="subtitle">TIMESFM 2.5 · CHRONOS-2 · LSTM · CONFORMAL CALIBRATION · TD SEQUENTIAL · <span style="color:#00ff88">REAL-TIME WS</span></p>

  <div id="wsStatus" class="ws-status disconnected">⬤ offline</div>

  <div class="form">
    <div class="field"><label>Target</label><input id="target" placeholder="e.g. APLD" value="APLD"></div>
    <div class="field"><label>Helper</label><input id="helper" placeholder="e.g. NVDA" value="NVDA"></div>
    <div class="field"><label>Horizon</label><select id="horizon"><option value="5">5d</option><option value="12" selected>12d</option><option value="20">20d</option><option value="30">30d</option></select></div>
    <div class="field"><label>Data Range</label><select id="dataRange" onchange="checkResources()"><option value="3m" selected>3mo</option><option value="6m">6mo</option><option value="1y">1yr ⚠️</option><option value="2y">2yr ⚠️</option><option value="5y">5yr 🔴</option><option value="max">Max 🔴</option></select></div>
    <div class="field"><label>&nbsp;</label><button id="btn" onclick="run()">▶ Run</button></div>
    <div class="field"><label>&nbsp;</label><button onclick="toggleHelp()" style="background:#222;color:#888;border:1px solid #444;padding:8px 14px;border-radius:2px;font-size:12px;cursor:pointer;font-family:inherit">? Help</button></div>
  </div>
  <div class="chart-toggle">
    <button id="btnCandle" class="active" onclick="setChartType('candle')">🕯 Candlestick</button>
    <button id="btnLine" onclick="setChartType('line')">📈 Line</button>
  </div>

  <div id="helpPanel" style="display:none;background:#111;border:1px solid #333;border-radius:2px;padding:16px;margin-bottom:16px;font-size:12px;line-height:1.6">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
      <h3 style="color:#00ff88;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin:0">How to Use</h3>
      <button onclick="toggleHelp()" style="background:none;border:1px solid #444;color:#666;padding:2px 8px;border-radius:2px;cursor:pointer;font-size:11px">✕</button>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div>
        <p style="color:#888;margin-bottom:6px"><b style="color:#aaa">Target</b> — the stock to forecast. e.g. <code style="background:#1a1a1a;padding:1px 4px">APLD</code>, <code style="background:#1a1a1a;padding:1px 4px">TSLA</code>, <code style="background:#1a1a1a;padding:1px 4px">BTC-USD</code></p>
        <p style="color:#888;margin-bottom:6px"><b style="color:#aaa">Helper</b> — macro signal. e.g. <code style="background:#1a1a1a;padding:1px 4px">NVDA</code> for AI stocks, <code style="background:#1a1a1a;padding:1px 4px">SPY</code> for broad market.</p>
        <p style="color:#888;margin-bottom:6px"><b style="color:#aaa">Horizon</b> — forecast days ahead. 5d is more accurate.</p>
      </div>
      <div>
        <p style="color:#888;margin-bottom:6px"><b style="color:#aaa">Models:</b></p>
        <ul style="color:#666;margin:0 0 8px 16px;padding:0">
          <li><b style="color:#ff7f0e">TimesFM 2.5</b> — Google's time series model (~2min)</li>
          <li><b style="color:#00d4ff">Chronos-2</b> — Amazon's probabilistic forecaster (cached in-memory!)</li>
          <li><b style="color:#00ff88">LSTM</b> — neural network with MC dropout</li>
        </ul>
        <p style="color:#888;margin-bottom:0"><b style="color:#aaa">v2:</b> WebSocket real-time progress · Model caching = faster subsequent runs · Error display + timeout alerts</p>
      </div>
    </div>
  </div>

  <div class="toggles">
    <label><input type="checkbox" id="oT" onchange="checkResources()"> TimesFM</label>
    <label><input type="checkbox" id="oC" checked onchange="checkResources()"> Chronos-2</label>
    <label><input type="checkbox" id="oL" onchange="checkResources()"> LSTM</label>
  </div>

  <div id="cacheInfo" class="cache-status"></div>

  <div class="status" id="st"></div>
  <div id="errBox" style="display:none"></div>
  <div id="warnBox" style="display:none;text-align:center;padding:6px;color:#ff6b35;font-size:11px"></div>

  <div class="results" id="res">
    <div class="chart-box"><img id="chart" src=""></div>
    <div class="cards">
      <div class="card"><h3>Predictions</h3><div id="pBody"></div></div>
      <div class="card"><h3>Technicals</h3><div id="tBody"></div></div>
      <div class="card"><h3>Analyst</h3><div id="aBody"></div></div>
    </div>
    <div class="card"><h3>News</h3><ul class="news-list" id="nList"></ul></div>
  </div>
</div>
<script>
// ── WebSocket connection ──
var socket = io();
var currentJobId = null;
var lastProgressTs = Date.now();
var timeoutInterval = null;
var completedStages = [];
var currentStage = '';
var chartType = 'candle';

socket.on('connect', function() {
  console.log('[WS] Connected:', socket.id);
  document.getElementById('wsStatus').className = 'ws-status connected';
  document.getElementById('wsStatus').textContent = '⬤ connected';
  loadCacheStatus();
});

socket.on('disconnect', function() {
  console.log('[WS] Disconnected');
  document.getElementById('wsStatus').className = 'ws-status disconnected';
  document.getElementById('wsStatus').textContent = '⬤ offline';
});

socket.on('started', function(data) {
  console.log('[WS] Job started:', data.job_id);
  currentJobId = data.job_id;
  lastProgressTs = Date.now();
  startTimeoutWatch();
});

socket.on('progress', function(data) {
  console.log('[WS] Progress:', data.stage, data.state);
  lastProgressTs = Date.now();
  clearError();

  if (data.stage === 'log') {
    document.getElementById('st').innerHTML = '<span class="spinner"></span> ' + escHtml(data.line);
    return;
  }

  // Track completed stages
  if (data.state === 'done' && data.stage !== 'log') {
    if (!completedStages.includes(data.stage)) completedStages.push(data.stage);
  }
  if (data.state === 'running' && data.stage !== 'log') {
    currentStage = data.stage;
  }
  if (data.state === 'failed') {
    if (!completedStages.includes(data.stage)) completedStages.push('❌ ' + data.stage);
  }

  // Build progress display
  var totalStages = 3 + (document.getElementById('oT').checked?1:0) + (document.getElementById('oC').checked?1:0) + (document.getElementById('oL').checked?1:0);
  var parts = completedStages.map(m => '✅ ' + m);
  if (currentStage && !completedStages.includes(currentStage) && !completedStages.includes('❌ ' + currentStage))
    parts.push('🔄 ' + currentStage);
  var txt = parts.length ? parts.join(' → ') + ' (' + completedStages.length + '/' + totalStages + ')' : 'starting...';
  document.getElementById('st').innerHTML = '<span class="spinner"></span> ' + txt;
});

socket.on('done', function(data) {
  console.log('[WS] Done:', data.job_id);
  stopTimeoutWatch();
  showPredictions(data.result);
  document.getElementById('st').innerHTML = '';
  document.getElementById('btn').disabled = false;
  currentJobId = null;
  loadCacheStatus();
});

socket.on('error', function(data) {
  console.log('[WS] Error:', data);
  stopTimeoutWatch();
  showError(data.message || 'Unknown server error');
  document.getElementById('btn').disabled = false;
  currentJobId = null;
});

// ── Timeout watch: if no progress for 2 minutes, show warning ──
function startTimeoutWatch() {
  stopTimeoutWatch();
  lastProgressTs = Date.now();
  timeoutInterval = setInterval(function() {
    var elapsed = Date.now() - lastProgressTs;
    if (elapsed > 120000) {
      document.getElementById('st').innerHTML = '<span style="color:#ff6b35">⚠️ No response for 2 min. The forecast may be stuck.</span>';
    }
  }, 10000);
}

function stopTimeoutWatch() {
  if (timeoutInterval) { clearInterval(timeoutInterval); timeoutInterval = null; }
}

// ── Form handlers ──
function toggleHelp(){var p=document.getElementById('helpPanel');p.style.display=p.style.display==='none'?'block':'none'}
function checkResources(){
  var dr=val('dataRange'), tfm=document.getElementById('oT').checked, lstm=document.getElementById('oL').checked, w=document.getElementById('warnBox');
  var msgs=[];
  if(dr==='2y'||dr==='5y'||dr==='max')msgs.push('Large data range — may be slow');
  if(dr==='max')msgs.push('⚠️ Max data may cause OOM');
  if(tfm)msgs.push('TimesFM is heavy (~2 min)');
  if(lstm)msgs.push('LSTM runs 30 MC passes (~2 min)');
  w.innerHTML=msgs.length?msgs.map(m=>'⚠️ '+m).join(' · '):'';
  w.style.display=msgs.length?'block':'none';
}
function setChartType(t){chartType=t;document.getElementById('btnCandle').className=t==='candle'?'active':'';document.getElementById('btnLine').className=t==='line'?'active':'';if(document.getElementById('res').classList.contains('active'))run()}

function $(id){return document.getElementById(id)}
function val(id){return $(id).value.trim().toUpperCase()}
function chk(id){return $(id).checked}

function loadCacheStatus(){
  fetch('/api/cache-status').then(function(r){return r.json()}).then(function(d){
    var chronos = d.chronos_cached ? '<span class="cache-dot on"></span>Cached' : '<span class="cache-dot off"></span>Not cached';
    document.getElementById('cacheInfo').innerHTML = 'Chronos-2: ' + chronos;
  }).catch(function(){});
}

function run(){
  var t=val('target'), h=val('helper'), hz=val('horizon');
  if(!t){alert('Enter ticker');return}
  document.getElementById('btn').disabled=true;
  document.getElementById('st').innerHTML='<span class="spinner"></span> connecting...';
  document.getElementById('errBox').style.display='none';
  document.getElementById('res').classList.remove('active');
  document.getElementById('chart').src='';
  document.getElementById('pBody').innerHTML='<div class="row" style="color:#444">waiting...</div>';
  document.getElementById('tBody').innerHTML='<div class="row" style="color:#444">waiting...</div>';
  document.getElementById('aBody').innerHTML='<div class="row" style="color:#444">loading...</div>';
  document.getElementById('nList').innerHTML='<li style="color:#444">loading...</li>';
  document.getElementById('res').classList.add('active');
  completedStages = [];
  currentStage = '';
  lastProgressTs = Date.now();

  // Fetch news/analyst immediately
  fetch('/api/news/'+encodeURIComponent(t)).then(function(r){return r.json()}).then(function(d){
    showAnalyst(d); showNews(d);
  }).catch(function(){});

  // Start forecast via WebSocket
  socket.emit('start_forecast', {
    target: t, helper: h, horizon: hz,
    dataRange: val('dataRange'), chartType: chartType,
    use_timesfm: document.getElementById('oT').checked,
    use_chronos: document.getElementById('oC').checked,
    use_lstm: document.getElementById('oL').checked
  });
}

function showError(msg){
  document.getElementById('errBox').innerHTML='<div class="error-msg">❌ '+escHtml(msg)+'</div>';
  document.getElementById('errBox').style.display='block';
  document.getElementById('st').innerHTML='';
  document.getElementById('st').style.color='';
}
function clearError(){
  document.getElementById('errBox').style.display='none';
  document.getElementById('st').style.color='#00ff88';
}

function showPredictions(d){
  clearError();
  if(d.chart_url){document.getElementById('chart').src=d.chart_url+'?t='+Date.now();}
  var ph='';
  (d.predictions||[]).forEach(function(p){var c=p.delta>=0?'up':'down';ph+='<div class="row"><span>'+escHtml(p.model)+'</span><span class="val '+c+'">$'+p.target.toFixed(2)+' ('+(p.delta>=0?'+':'')+p.delta.toFixed(2)+'%)</span></div>'});
  if(ph)ph+='<div class="row" style="border-top:1px solid #333;margin-top:4px;padding-top:4px"><span>consensus</span><span class="val">'+escHtml(d.consensus||'N/A')+'</span></div><div class="row"><span>avg Δ</span><span class="val '+(d.avg_delta>=0?'up':'down')+'">'+(d.avg_delta||0).toFixed(2)+'%</span></div>';
  if(!ph)ph='<div class="row" style="color:#444">no data</div>';
  document.getElementById('pBody').innerHTML=ph;
  var th='';var t2=d.technicals||{};
  th+='<div class="row"><span>RSI 14</span><span class="val">'+(t2.rsi!=null?t2.rsi.toFixed(2):'—')+' '+(t2.rsi_label||'')+'</span></div>';
  th+='<div class="row"><span>EMA 50</span><span class="val">'+(t2.ema_label||'—')+' '+(t2.ema!=null?'$'+t2.ema.toFixed(2):'')+'</span></div>';
  th+='<div class="row"><span>TD</span><span class="val">'+(t2.td_count!=null?'TD '+t2.td_count:'—')+' '+(t2.td_label||'')+'</span></div>';
  th+='<div class="row"><span>VOL</span><span class="val">'+(t2.vol_label||'—')+' '+(t2.bb_upper!=null?'BB $'+t2.bb_upper.toFixed(2):'')+'</span></div>';
  document.getElementById('tBody').innerHTML=th;
}
function showAnalyst(d){
  var ah='';var e=d.earnings||{};
  if(e.available){var dc=e.days_until<=7?'#ff4444':e.days_until<=14?'#ff6b35':e.days_until<=30?'#ffaa00':'#00ff88';ah+='<div class="row"><span>Earnings</span><span class="val" style="color:'+dc+'">'+escHtml(e.date)+' ('+e.days_until+'d)</span></div>'}
  var a=d.analyst||{};
  if(a.available){var tot=(a.total||1);var b2=((a.strong_buy+a.buy)/tot*100).toFixed(0),h2=(a.hold/tot*100).toFixed(0),s=((a.sell+a.strong_sell)/tot*100).toFixed(0);ah+='<div class="analyst-bar"><div class="bar-buy" style="width:'+b2+'%">'+(b2>12?b2+'%':'')+'</div><div class="bar-hold" style="width:'+h2+'%">'+(h2>12?h2+'%':'')+'</div><div class="bar-sell" style="width:'+s+'%">'+(s>12?s+'%':'')+'</div></div>';ah+='<div class="legend"><span class="l-buy">buy '+(a.strong_buy+a.buy)+'</span><span class="l-hold">hold '+a.hold+'</span><span class="l-sell">sell '+(a.sell+a.strong_sell)+'</span></div>'}
  else if(!e.available)ah='<div class="row" style="color:#444">no data</div>';
  document.getElementById('aBody').innerHTML=ah;
}
function showNews(d){
  var nh='';(d.news||[]).forEach(function(n){nh+='<li><a href="'+escHtml(n.url||'#')+'" target="_blank">'+escHtml(n.title||'')+'</a><div class="meta">'+escHtml(n.date||'')+'</div></li>'});
  if(!nh)nh='<li style="color:#444">no news</li>';
  document.getElementById('nList').innerHTML=nh;
}
function escHtml(s){if(!s)return '';var d=document.createElement('div');d.appendChild(document.createTextNode(s));return d.innerHTML}

// Load cache status on page load
loadCacheStatus();
</script></body></html>"""


if __name__ == "__main__":
    print(f"⚡ Forecast GUI v2 → http://localhost:{PORT}")
    print(f"   WebSocket + Model Caching + Error Display")
    socketio.run(app, host="0.0.0.0", port=PORT, debug=False, allow_unsafe_werkzeug=True)