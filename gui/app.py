#!/usr/bin/env python3
"""
Stock Forecast GUI — runs forecast as independent subprocess (no threads).
Run:  python gui/app.py [port]
"""
import os, sys, json, re, time, subprocess, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flask import Flask, render_template_string, request, jsonify, send_file

WORKSPACE  = Path(__file__).resolve().parent.parent
FORECAST   = WORKSPACE / "forecast.py"
PYTHON     = "/opt/homebrew/bin/python3.11"
PORT       = int(sys.argv[1]) if len(sys.argv) > 1 else 9876

app = Flask(__name__)
_jobs = {}
_jobs_lock = threading.Lock()

def _submit_job(target, helper, horizon, use_tfm, use_chr, use_lstm, data_range="1y", chart_type="candle"):
    data_range = data_range.lower()  # normalize: frontend uppercases values
    flags = f"{'T' if use_tfm else ''}{'C' if use_chr else ''}{'L' if use_lstm else ''}"
    jid = f"{target}_{helper}_{horizon}_{data_range}_{chart_type}_{flags}"

    # Check cached/complete jobs — release lock before calling _complete_job
    # to avoid deadlock (threading.Lock is NOT reentrant, and _complete_job
    # acquires _jobs_lock internally for final update).
    needs_complete = None
    with _jobs_lock:
        j = _jobs.get(jid)
        if j and j["status"] == "done" and (time.time() - j["ts"]) < 300:
            return jid, True
        if j and j["status"] == "running":
            proc = j.get("proc")
            if proc and proc.poll() is not None:
                needs_complete = (jid, j)
            else:
                return jid, False

    if needs_complete:
        return _complete_job(*needs_complete)

    # Start new forecast as independent subprocess
    chart_path = WORKSPACE / f"{target}_{helper}_{horizon}_{data_range}_{chart_type}_forecast.png"
    cmd = [PYTHON, str(FORECAST), target, helper,
           "--horizon", str(horizon), "--data-range", data_range,
           "--chart-type", chart_type, "--output", str(chart_path)]
    if not use_tfm:  cmd.append("--no-timesfm")
    if not use_chr:  cmd.append("--no-chronos")
    if not use_lstm: cmd.append("--no-lstm")

    # Launch process (non-blocking, independent from Flask)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Background thread to read status lines from stdout
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
        _jobs[jid] = {"status": "running", "target": target, "ts": time.time(),
                       "proc": proc, "chart_path": str(chart_path),
                       "helper": helper, "horizon": horizon, "data_range": data_range,
                       "chart_type": chart_type,
                       "status_lines": status_lines, "reader_thread": reader_thread,
                       "progress": "Starting..."}
    return jid, False


def _complete_job(jid, job):
    """Called when the subprocess has finished — parses output and news."""
    try:
        proc = job.get("proc")
        reader_thread = job.get("reader_thread")
        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=2)

        # Check exit code: non-zero means subprocess crashed
        exit_code = proc.poll() if proc else None
        all_output = "\n".join(job.get("status_lines", []))

        result = _parse(all_output, job["target"], job["helper"])

        # If subprocess exited with error, surface it
        if exit_code is not None and exit_code != 0:
            err_lines = [l for l in job.get("status_lines", [])
                        if "error" in l.lower() or "traceback" in l.lower()
                        or "exception" in l.lower()][-3:]
            err_detail = "; ".join(err_lines[-2:]) if err_lines else f"exit code {exit_code}"
            result["error"] = err_detail
            result["_exit_code"] = exit_code

        chart_name = f"{job['target']}_{job['helper']}_{job['horizon']}_{job['data_range']}_{job['chart_type']}_forecast.png"
        chart = WORKSPACE / chart_name
        if chart.exists():
            result["chart_url"] = f"/chart/{chart_name}"
        else:
            import glob as _glob
            pattern = f"{job['target']}_{job['helper']}_*_forecast.png"
            matches = _glob.glob(str(WORKSPACE / pattern))
            if matches:
                result["chart_url"] = f"/chart/{os.path.basename(matches[-1])}"

        from news import get_news_and_ratings
        job["status_lines"].append("[STATUS:news:running]")
        nd = get_news_and_ratings(job["target"])
        job["status_lines"].append("[STATUS:news:done]")
        result["news"] = nd.get("news", [])[:10]
        result["analyst"] = nd.get("analyst", {"available": False})
        result["earnings"] = nd.get("earnings", {"available": False})

        job["proc"] = None  # Release proc reference
        with _jobs_lock:
            _jobs[jid] = {"status": "done", "result": result, "ts": time.time()}
        return jid, True
    except Exception as e:
        with _jobs_lock:
            _jobs[jid] = {"status": "error", "result": {"error": str(e)}, "ts": time.time()}
        return jid, False


def _parse(text, target, helper):
    lines = text.splitlines()
    res = {"target": target, "helper": helper, "predictions": [], "consensus": "N/A", "avg_delta": 0}
    in_table = False
    for line in lines:
        if "MODEL" in line and "BIAS" in line:
            in_table = True; continue
        if in_table and line.strip().startswith("─"): continue
        if in_table and line.strip().startswith("="): in_table = False; continue
        if in_table and "|" in line:
            p = [x.strip() for x in line.split("|")]
            if len(p) >= 4 and p[0] and p[0] != "MODEL":
                try:
                    price = float(p[1].replace("$", "").replace(",", ""))
                    delta = float(p[2].replace("%", "").strip())
                    res["predictions"].append({"model": p[0], "target": price, "delta": delta})
                except: pass
    for line in lines:
        s = line.strip()
        if "CONSENSUS" in s and ":" in s:
            res["consensus"] = s.split(":", 1)[-1].strip()
        if "AVG DELTA" in s and ":" in s:
            try: res["avg_delta"] = float(s.split(":", 1)[-1].strip().replace("%","").replace("+",""))
            except: pass
    tech = {}
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3: continue
        label = parts[1]; val_col = parts[2] if len(parts) > 2 else ""
        if "RSI" in label:
            m = re.search(r":\s*([\d.]+)", label)
            if m:
                try: tech["rsi"] = float(m.group(1))
                except: pass
            tech["rsi_label"] = val_col.strip().strip("[]")
        if "EMA" in label and "50" in label:
            m = re.search(r":\s*(\w+)", label)
            if m: tech["ema_label"] = m.group(1)
            m2 = re.search(r"\$([\d,.]+)", val_col)
            if m2:
                try: tech["ema"] = float(m2.group(1).replace(",",""))
                except: tech["ema"] = 0
        if "TD" in label and "SETUP" in label:
            m = re.search(r":\s*(\d+)", label)
            if m:
                try: tech["td_count"] = int(m.group(1)) or None
                except: tech["td_count"] = None
            tech["td_label"] = val_col
        if "VOLATILITY" in label:
            m = re.search(r":\s*(\w+)", label)
            if m: tech["vol_label"] = m.group(1)
            m2 = re.search(r"\$([\d,.]+)", val_col)
            if m2:
                try: tech["bb_upper"] = float(m2.group(1).replace(",",""))
                except: tech["bb_upper"] = 0
    if tech: res["technicals"] = tech
    return res

# ── HTML (same as before) ────────────────────────────────────
PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>⚡ Forecast</title>
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
</style></head><body>
<div class="container">
  <h1>⚡ <span>Forecast</span></h1>
  <p class="subtitle">TIMESFM 2.5 · CHRONOS-2 · LSTM · CONFORMAL CALIBRATION · TD SEQUENTIAL</p>

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
          <li><b style="color:#00d4ff">Chronos-2</b> — Amazon's probabilistic forecaster</li>
          <li><b style="color:#00ff88">LSTM</b> — neural network with MC dropout</li>
        </ul>
        <p style="color:#888;margin-bottom:0"><b style="color:#aaa">Chart:</b> Green/red = candlesticks. Colored bands = model confidence. Orange band = ensemble. TD numbers = DeMark setup (⚠9 = reversal). Volume = middle panel.</p>
      </div>
    </div>
  </div>

  <div class="toggles">
    <label><input type="checkbox" id="oT" onchange="checkResources()"> TimesFM</label>
    <label><input type="checkbox" id="oC" checked onchange="checkResources()"> Chronos-2</label>
    <label><input type="checkbox" id="oL" onchange="checkResources()"> LSTM</label>
  </div>

  <div class="status" id="st"></div>

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
function toggleHelp(){var p=document.getElementById('helpPanel');p.style.display=p.style.display==='none'?'block':'none'}
var chartType='candle';
function checkResources(){
  var dr=val('dataRange'), tfm=chk('oT'), lstm=chk('oL'), w=$('warnBox');
  var msgs=[];
  if(dr==='2Y'||dr==='5Y'||dr==='MAX')msgs.push('Large data range ('+dr+') — may be slow');
  if(dr==='MAX')msgs.push('⚠️ Max data may cause out-of-memory crash');
  if(tfm)msgs.push('TimesFM is heavy (~2 min, jax/torch)');
  if(lstm)msgs.push('LSTM runs 30 MC passes (~2 min, tensorflow)');
  w.innerHTML=msgs.length?msgs.map(m=>'⚠️ '+m).join(' · '):'';
  w.style.display=msgs.length?'block':'none';
}
function setChartType(t){chartType=t;document.getElementById('btnCandle').className=t==='candle'?'active':'';document.getElementById('btnLine').className=t==='line'?'active':'';if(document.getElementById('res').classList.contains('active'))run()}
function $(id){return document.getElementById(id)}
function val(id){return $(id).value.trim().toUpperCase()}
function chk(id){return $(id).checked}

function run(){
  const t=val('target'),h=val('helper'),hz=val('horizon');
  if(!t){alert('Enter ticker');return}
  $('btn').disabled=true;$('st').innerHTML='<span class="spinner"></span> starting...';
  $('res').classList.remove('active');
  // Clear previous results
  $('chart').src='';$('pBody').innerHTML='<div class="row" style="color:#444">waiting...</div>';
  $('tBody').innerHTML='<div class="row" style="color:#444">waiting...</div>';
  $('aBody').innerHTML='<div class="row" style="color:#444">loading...</div>';
  $('nList').innerHTML='<li style="color:#444">loading...</li>';
  $('res').classList.add('active');
  // Fetch news/analyst immediately — no waiting for forecast
  fetch('/api/news/'+encodeURIComponent(t)).then(r=>r.json()).then(d=>{
    showAnalyst(d);
    showNews(d);
  }).catch(()=>{});
  // Start forecast + polling
  const b=new URLSearchParams({target:t,helper:h,horizon:hz,dataRange:val('dataRange'),chartType:chartType,use_timesfm:chk('oT'),use_chronos:chk('oC'),use_lstm:chk('oL')});
  fetch('/api/start',{method:'POST',body:b}).then(r=>r.json()).then(d=>{
    if(d.status==='done'){showPredictions(d.result);$('st').innerHTML='';$('btn').disabled=false;return}
    if(d.status==='error'){handleError(d);return}
    if(d.status==='done'&&d.result&&d.result.chart_url){$('chart').src=d.result.chart_url+'?t='+Date.now();}
    poll();
  }).catch(e=>{handleError({result:{error:'Network: '+e.message}});$('btn').disabled=false});
}
function poll(){
  const t=val('target'),h=val('helper'),hz=val('horizon'),dr=val('dataRange');
  let n=0, chartLoaded=false, completed=[], current='';
  const totalModels=1+(chk('oT')?1:0)+(chk('oL')?1:0);
  const iv=setInterval(()=>{
    n++;if(n>180){clearInterval(iv);$('st').textContent='timeout (6min)';$('btn').disabled=false;return}
    const b=new URLSearchParams({target:t,helper:h,horizon:hz,dataRange:dr,chartType:chartType,use_timesfm:chk('oT'),use_chronos:chk('oC'),use_lstm:chk('oL')});
    fetch('/api/start',{method:'POST',body:b}).then(r=>r.json()).then(d=>{
      if(d.status==='done'){clearInterval(iv);showPredictions(d.result);$('st').innerHTML='';$('btn').disabled=false;return}
      if(d.status==='error'){clearInterval(iv);handleError(d);return}
      if(d.completed_stages&&d.completed_stages.length>0){
        completed=d.completed_stages;
      }
      if(d.model&&d.model_state){
        current=d.model;
        if(d.model_state==='done'&&!completed.includes(d.model))completed.push(d.model);
      }
      var totalStages=4+(chk('oT')||chk('oL')?1:0)+(chk('oC')?1:0); // data+indicators+conformal+chart + models
      var txt='';
      var parts=[];
      if(completed.length>0)parts=completed.map(m=>'✅ '+m);
      if(current&&!parts.includes('✅ '+current))parts.push('🔄 '+current);
      if(parts.length)txt=parts.join(' → ')+' ('+completed.length+'/'+totalStages+' stages)';
      else txt='downloading... ('+n+')';
      $('st').innerHTML='<span class="spinner"></span> '+txt;
      if(d.status==='done'&&d.result&&d.result.chart_url&&!chartLoaded){
        chartLoaded=true;$('chart').src=d.result.chart_url+'?t='+Date.now();
      }
    }).catch(e=>{$('st').textContent='ERR: '+e;clearInterval(iv);$('btn').disabled=false});
  },3000);
}
function handleError(d){
  var msg='❌ ';
  if(d.result&&d.result.error)msg+=d.result.error;
  else msg+='Unknown server error';
  $('st').innerHTML=msg;
  $('st').style.color='#ff4444';
  $('btn').disabled=false;
}
function showPredictions(d){
  if(d.chart_url){$('chart').src=d.chart_url+'?t='+Date.now();}
  let ph='';
  (d.predictions||[]).forEach(p=>{const c=p.delta>=0?'up':'down';ph+=`<div class="row"><span>${p.model}</span><span class="val ${c}">$${p.target.toFixed(2)} (${p.delta>=0?'+':''}${p.delta.toFixed(2)}%)</span></div>`});
  if(ph)ph+=`<div class="row" style="border-top:1px solid #333;margin-top:4px;padding-top:4px"><span>consensus</span><span class="val">${d.consensus||'N/A'}</span></div><div class="row"><span>avg Δ</span><span class="val ${(d.avg_delta||0)>=0?'up':'down'}">${(d.avg_delta||0).toFixed(2)}%</span></div>`;
  if(!ph)ph='<div class="row" style="color:#444">no data</div>';
  $('pBody').innerHTML=ph;
  let th='';const t2=d.technicals||{};
  th+=`<div class="row"><span>RSI 14</span><span class="val">${t2.rsi!=null?t2.rsi.toFixed(2):'—'} ${t2.rsi_label||''}</span></div>`;
  th+=`<div class="row"><span>EMA 50</span><span class="val">${t2.ema_label||'—'} ${t2.ema!=null?'$'+t2.ema.toFixed(2):''}</span></div>`;
  th+=`<div class="row"><span>TD</span><span class="val">${t2.td_count!=null?'TD '+t2.td_count:'—'} ${t2.td_label||''}</span></div>`;
  th+=`<div class="row"><span>VOL</span><span class="val">${t2.vol_label||'—'} ${t2.bb_upper!=null?'BB $'+t2.bb_upper.toFixed(2):''}</span></div>`;
  $('tBody').innerHTML=th;
}
function showAnalyst(d){
  let ah='';const e=d.earnings||{};
  if(e.available){const dc=e.days_until<=7?'#ff4444':e.days_until<=14?'#ff6b35':e.days_until<=30?'#ffaa00':'#00ff88';ah+=`<div class="row"><span>Earnings</span><span class="val" style="color:${dc}">${e.date} (${e.days_until}d)</span></div>`}
  const a=d.analyst||{};
  if(a.available){const tot=(a.total||1);const b2=((a.strong_buy+a.buy)/tot*100).toFixed(0),h2=(a.hold/tot*100).toFixed(0),s=((a.sell+a.strong_sell)/tot*100).toFixed(0);ah+=`<div class="analyst-bar"><div class="bar-buy" style="width:${b2}%">${b2>12?b2+'%':''}</div><div class="bar-hold" style="width:${h2}%">${h2>12?h2+'%':''}</div><div class="bar-sell" style="width:${s}%">${s>12?s+'%':''}</div></div>`;ah+=`<div class="legend"><span class="l-buy">buy ${a.strong_buy+a.buy}</span><span class="l-hold">hold ${a.hold}</span><span class="l-sell">sell ${a.sell+a.strong_sell}</span></div>`}
  else if(!e.available)ah='<div class="row" style="color:#444">no data</div>';
  $('aBody').innerHTML=ah;
}
function showNews(d){
  let nh='';(d.news||[]).forEach(n=>{nh+=`<li><a href="${n.url||'#'}" target="_blank">${n.title||''}</a><div class="meta">${n.date||''}</div></li>`});
  if(!nh)nh='<li style="color:#444">no news</li>';
  $('nList').innerHTML=nh;
}
</script></body></html>"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/api/start", methods=["POST"])
def api_start():
    jid, cached = _submit_job(
        request.form.get("target","APLD").upper(),
        request.form.get("helper","NVDA").upper(),
        int(request.form.get("horizon", 12)),
        request.form.get("use_timesfm")=="true",
        request.form.get("use_chronos")=="true",
        request.form.get("use_lstm")=="true",
        request.form.get("dataRange", "1y"),
        request.form.get("chartType", "candle"),
    )
    if cached:
        with _jobs_lock:
            j = _jobs.get(jid, {})
        return jsonify({"status": "done", "result": j.get("result", {})})

    # Check if the running job has completed (proc.poll() detects exit)
    with _jobs_lock:
        j = _jobs.get(jid, {})
    if j.get("status") == "running":
        proc = j.get("proc")
        if proc and proc.poll() is not None:
            # Subprocess has exited but wasn't caught by _submit_job's check
            # (race condition: exited between check and response)
            _complete_job(jid, j)
            with _jobs_lock:
                j = _jobs.get(jid, {})
        else:
            # Still running — return progress info with ALL completed stages
            resp = {"status": "started", "job_id": jid}
            lines = j.get("status_lines", [])
            if lines:
                resp["last_line"] = lines[-1][:100]
            completed_stages = []
            current_stage = None
            current_state = None
            for line in lines:
                if "[STATUS:" in line:
                    parts = line.strip("[]").split(":")
                    if len(parts) >= 3:
                        stage = parts[1].strip()
                        state = parts[2].strip()
                        if state == "done":
                            if stage not in completed_stages:
                                completed_stages.append(stage)
                        elif state == "running":
                            current_stage = stage
                            current_state = state
            resp["completed_stages"] = completed_stages
            if current_stage:
                resp["model"] = current_stage
                resp["model_state"] = current_state or "running"
            return jsonify(resp)

    # Return final status (done or error)
    if j.get("status") == "done":
        return jsonify({"status": "done", "result": j.get("result", {})})
    elif j.get("status") == "error":
        return jsonify({"status": "error", "result": j.get("result", {})})
    else:
        return jsonify({"status": "running", "job_id": jid})

@app.route("/api/status")
def api_status():
    jid = request.args.get("jid","")
    with _jobs_lock:
        j = _jobs.get(jid)
    if not j:
        return jsonify({"error":"Job not found"}), 404
    if j["status"] == "running" and j.get("proc") and j["proc"].poll() is not None:
        _complete_job(jid, j)
        with _jobs_lock:
            j = _jobs.get(jid, {})
    resp = {"status": j.get("status", "?")}
    # Include progress info while running
    if j.get("status") == "running":
        # Extract latest [STATUS:...] marker
        for line in reversed(j.get("status_lines", [])):
            if "[STATUS:" in line:
                resp["progress"] = line
                # Parse model name and state from [STATUS:Name:state]
                parts = line.strip("[]").split(":")
                if len(parts) >= 3:
                    resp["model"] = parts[1].strip()
                    resp["model_state"] = parts[2].strip()
                break
    if j.get("status") in ("done","error"):
        resp["result"] = j.get("result", {})
    return jsonify(resp)

@app.route("/chart/<path:chart_name>")
def chart(chart_name):
    p = WORKSPACE / chart_name
    return send_file(p, mimetype="image/png") if p.exists() else ("",404)

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
    """Check if a chart file exists on disk."""
    p = WORKSPACE / chart_name
    return jsonify({"ready": p.exists() and p.stat().st_size > 1000})

@app.route("/test")
def test():
    return send_file(WORKSPACE / "gui" / "test_standalone.html")

if __name__ == "__main__":
    print(f"Forecast GUI -> http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
