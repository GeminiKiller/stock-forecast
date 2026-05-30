#!/usr/bin/env python3
"""Fetch analyst ratings and news for a ticker."""
import json
import urllib.request
import re
from datetime import datetime

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _sid = SentimentIntensityAnalyzer()
except ImportError:
    _sid = None

# FinBERT (PyTorch) — much better financial sentiment
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _finbert_tokenizer = None
    _finbert_model = None
    _finbert_available = True
except ImportError:
    _finbert_available = False

def _get_finbert():
    global _finbert_tokenizer, _finbert_model
    if _finbert_model is not None:
        return _finbert_tokenizer, _finbert_model
    _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    _finbert_model.eval()
    return _finbert_tokenizer, _finbert_model

def _finbert_sentiment(text):
    tokenizer, model = _get_finbert()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    # FinBERT labels: 0=positive, 1=negative, 2=neutral
    pos = probs[0].item()
    neg = probs[1].item()
    neu = probs[2].item()
    compound = pos - neg
    return {
        "compound": round(compound, 4),
        "pos": round(pos, 4),
        "neu": round(neu, 4),
        "neg": round(neg, 4),
    }


def fetch_yahoo_rss(ticker: str, max_items: int = 15) -> list[dict]:
    """Fetch latest news from Yahoo Finance RSS feed."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        return [{"error": str(e)}]

    items = []
    for match in re.finditer(r"<item>(.*?)</item>", data, re.DOTALL):
        block = match.group(1)
        def tag(name):
            m = re.search(rf"<{name}[^>]*>(.*?)</{name}>", block, re.DOTALL)
            return m.group(1).strip() if m else ""
        title = re.sub(r"<[^>]+>", "", tag("title")).strip()
        desc  = re.sub(r"<[^>]+>", "", tag("description")).strip()
        link  = tag("link")
        pub   = tag("pubDate")
        if title:
            items.append({"title": title, "description": desc, "url": link, "date": pub})
        if len(items) >= max_items:
            break
    return items


def fetch_analyst_ratings(ticker: str) -> dict:
    """Fetch analyst recommendation summary via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        rec = t.recommendations
        if rec is None or len(rec) == 0:
            return {"available": False, "message": "No analyst data available"}

        latest = rec.iloc[0]
        return {
            "available": True,
            "period": str(latest.get("period", "current")),
            "strong_buy":  int(latest.get("strongBuy", 0)),
            "buy":         int(latest.get("buy", 0)),
            "hold":        int(latest.get("hold", 0)),
            "sell":        int(latest.get("sell", 0)),
            "strong_sell": int(latest.get("strongSell", 0)),
            "total": int(sum([
                latest.get("strongBuy", 0), latest.get("buy", 0),
                latest.get("hold", 0), latest.get("sell", 0),
                latest.get("strongSell", 0)
            ])),
        }
    except Exception as e:
        return {"available": False, "message": str(e)}


def fetch_earnings_date(ticker: str) -> dict:
    """Fetch next earnings date and days until."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        # Try to get earnings calendar
        try:
            cal = t.calendar
            if cal is not None and len(cal) > 0:
                # Get the next earnings date
                for idx, row in cal.iterrows():
                    if 'Earnings Date' in str(idx) or 'earnings' in str(idx).lower():
                        date_val = row.iloc[0] if hasattr(row, 'iloc') else row
                        if hasattr(date_val, 'date'):
                            from datetime import date as dt_date
                            days_until = (date_val.date() - dt_date.today()).days
                            return {
                                "available": True,
                                "date": date_val.strftime("%Y-%m-%d"),
                                "days_until": days_until,
                            }
        except:
            pass

        # Fallback: try earnings_dates
        try:
            ed = t.earnings_dates
            if ed is not None and len(ed) > 0:
                from datetime import datetime as dt
                now = dt.now()
                future = ed[ed.index > now]
                if len(future) > 0:
                    next_date = future.index[0]
                    days_until = (next_date - now).days
                    return {
                        "available": True,
                        "date": next_date.strftime("%Y-%m-%d"),
                        "days_until": days_until,
                    }
        except:
            pass

        # Fallback: use earningsTimestampStart from info (next earnings date)
        try:
            info = t.info
            ts = info.get('earningsTimestampStart') or info.get('earningsTimestamp')
            if ts and isinstance(ts, (int, float)) and ts > 0:
                from datetime import datetime, date
                next_dt = datetime.fromtimestamp(ts)
                days_until = (next_dt.date() - date.today()).days
                if days_until >= -60:  # include recent past earnings
                    return {
                        "available": True,
                        "date": next_dt.strftime("%Y-%m-%d"),
                        "days_until": max(days_until, 0),
                    }
        except:
            pass

        return {"available": False, "message": "Earnings date not available"}
    except Exception as e:
        return {"available": False, "message": str(e)}


# Financial keyword lexicon for boosting sentiment scores
FIN_POS = {
    'surge':0.15,'surges':0.15,'surged':0.15,'rally':0.15,'rallies':0.15,'rallied':0.15,
    'soar':0.2,'soared':0.2,'soaring':0.2,'gaining':0.1,'gain':0.1,'gains':0.1,
    'up':0.05,'rises':0.1,'rising':0.1,'rose':0.1,'high':0.1,'highs':0.1,'record':0.1,
    'breakthrough':0.2,'deal':0.1,'agreement':0.1,'partnership':0.1,'raised':0.15,
    'upgrade':0.2,'upgraded':0.2,'outperform':0.2,'beat':0.15,'beats':0.15,'beating':0.15,
    'profit':0.1,'profits':0.1,'growth':0.1,'bullish':0.3,'strong':0.1,'momentum':0.1,
    'breakout':0.15,'jump':0.1,'jumps':0.1,'jumped':0.1,'spike':0.1,'spikes':0.1,
    'spiked':0.1,'rocket':0.2,'rockets':0.2,'rocketed':0.2,'climb':0.1,'climbs':0.1,
    'climbing':0.1,'advance':0.1,'advances':0.1,'buy':0.15,'overweight':0.15,
    'accumulate':0.1,
    'lease':0.15,'leases':0.15,'leasing':0.15,'signed':0.1,'signs':0.1,'signing':0.1,
    'major':0.1,'expansion':0.15,'expanding':0.15,'expanded':0.15,
    'facility':0.1,'facilities':0.1,'campus':0.05,'campuses':0.05,'new':0.05,
    'building':0.1,'builds':0.1,'built':0.1,'construction':0.05,
    'launch':0.15,'launches':0.15,'launched':0.15,'launching':0.15,
    'deploy':0.1,'deploys':0.1,'deployed':0.1,'deployment':0.1,
    'acquire':0.15,'acquires':0.15,'acquired':0.15,'acquisition':0.15,
    'merge':0.1,'merges':0.1,'merged':0.1,'merger':0.1,
    'contract':0.1,'contracts':0.1,'contracted':0.1,'contracting':0.1,
    'order':0.1,'orders':0.1,'ordered':0.1,'ordering':0.1,
    'booking':0.1,'bookings':0.1,'backlog':0.1,'pipeline':0.1,
    'revenue':0.1,'revenues':0.1,'sales':0.1,'earnings':0.1,
    'margin':0.05,'margins':0.05,'profitability':0.1,
    'beat':0.15,'beats':0.15,'beating':0.15,'beaten':0.15,
    'exceed':0.15,'exceeds':0.15,'exceeded':0.15,'exceeding':0.15,
    'outperform':0.2,'outperforms':0.2,'outperformed':0.2,'outperforming':0.2,
    'dominant':0.1,'leading':0.1,'leader':0.1,'top':0.1,'best':0.15,
    'milestone':0.1,'achievement':0.1,'achieved':0.1,'achieves':0.1,'bounce':0.1,'bounces':0.1,'bounced':0.1,'rebound':0.1,
    'rebounds':0.1,'rebounded':0.1,'lift':0.1,'lifts':0.1,'lifted':0.1,
}
FIN_NEG = {
    'drop':-0.15,'drops':-0.15,'dropping':-0.15,'fall':-0.15,'falls':-0.15,
    'falling':-0.15,'down':-0.05,'crash':-0.3,'crashes':-0.3,'crashed':-0.3,
    'bearish':-0.3,'weak':-0.15,'miss':-0.2,'misses':-0.2,'missed':-0.2,
    'cut':-0.15,'cuts':-0.15,'downgrade':-0.2,'downgraded':-0.2,'lawsuit':-0.2,
    'investigation':-0.2,'debt':-0.1,'loss':-0.15,'losses':-0.15,'declining':-0.15,
    'plunge':-0.25,'plummet':-0.25,'tank':-0.25,'underperform':-0.2,'sell':-0.15,
    'underweight':-0.15,'reduce':-0.1,'avoid':-0.15,'warning':-0.15,'concern':-0.1,
    'risk':-0.1,'risks':-0.1,'volatile':-0.1,'volatility':-0.1,'tumble':-0.2,
    'tumbles':-0.2,'tumbled':-0.2,'slide':-0.15,'slides':-0.15,'sliding':-0.15,
    'slumped':-0.2,'slumps':-0.2,'slumping':-0.2,'plunged':-0.25,'plunging':-0.25,
    'plummeted':-0.25,'plummeting':-0.25,'crashing':-0.3,'collapse':-0.3,
    'collapses':-0.3,'collapsed':-0.3,'dive':-0.15,'dives':-0.15,'dived':-0.15,
    'plunge':-0.25,'nosedive':-0.25,'nosedives':-0.25,'nosedived':-0.25,
    'stagnant':-0.1,'stagnation':-0.1,'struggle':-0.1,'struggles':-0.1,
    'struggling':-0.1,'headwind':-0.1,'headwinds':-0.1,'layoff':-0.2,
    'layoffs':-0.2,'fired':-0.15,'firing':-0.15,'delay':-0.1,'delays':-0.1,
    'delayed':-0.1,'postpone':-0.1,'postponed':-0.1,'cancel':-0.15,
    'cancels':-0.15,'cancelled':-0.15,'cancelling':-0.15,'halt':-0.15,
    'halts':-0.15,'halted':-0.15,'suspend':-0.15,'suspends':-0.15,
    'suspended':-0.15,'fraud':-0.3,'scandal':-0.25,'bankrupt':-0.3,
    'bankruptcy':-0.3,'default':-0.25,'defaults':-0.25,'recession':-0.2,
    'inflation':-0.1,'tariff':-0.1,'tariffs':-0.1,'sanction':-0.15,
    'sanctions':-0.15,'ban':-0.15,'bans':-0.15,'banned':-0.15,'fine':-0.15,
    'fines':-0.15,'fined':-0.15,'penalty':-0.15,'penalties':-0.15,
    'losing':-0.15,'loses':-0.15,'lose':-0.15,'lost':-0.15,'dip':-0.1,
    'dips':-0.1,'dipped':-0.1,'retreat':-0.1,'retreats':-0.1,'retreated':-0.1,
}

def _fin_keyword_boost(text: str) -> float:
    """Compute financial keyword boost for a headline."""
    if not text:
        return 0.0
    text_lower = text.lower()
    boost = 0.0
    # Check multi-word phrases first
    for phrase, val in list(FIN_POS.items()) + list(FIN_NEG.items()):
        if ' ' in phrase and phrase in text_lower:
            boost += val
    # Then single words
    words = set(re.findall(r'\b\w+\b', text_lower))
    for word in words:
        boost += FIN_POS.get(word, 0.0)
        boost += FIN_NEG.get(word, 0.0)
    return max(-0.5, min(0.5, boost))  # clamp boost to ±0.5

def analyze_sentiment(headlines: list[dict]) -> list[dict]:
    """Add FinBERT sentiment scores to each headline (VADER fallback if unavailable).
    
    Adds: sentiment.compound, sentiment.pos, sentiment.neu, sentiment.neg, sentiment.emoji
    Returns the same list with sentiment field added to each dict.
    """
    if not _finbert_available:
        # Fallback to keyword-boosted VADER
        if _sid is None:
            for item in headlines:
                item["sentiment"] = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "emoji": "⚪"}
            return headlines
        for item in headlines:
            text = item.get("title", "") or ""
            if not text:
                item["sentiment"] = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "emoji": "⚪"}
                continue
            scores = _sid.polarity_scores(text)
            boost = _fin_keyword_boost(text)
            compound = max(-1.0, min(1.0, scores["compound"] + boost))
            if compound > 0.0:
                emoji = "🟢"
            elif compound < 0.0:
                emoji = "🔴"
            else:
                emoji = "⚪"
            item["sentiment"] = {
                "compound": round(compound, 4),
                "pos": round(scores["pos"], 4),
                "neu": round(scores["neu"], 4),
                "neg": round(scores["neg"], 4),
                "emoji": emoji,
            }
        return headlines
    
    # FinBERT + keyword boost hybrid
    for item in headlines:
        text = item.get("title", "") or ""
        if not text:
            item["sentiment"] = {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "emoji": "⚪"}
            continue
        scores = _finbert_sentiment(text)
        boost = _fin_keyword_boost(text)
        compound = max(-1.0, min(1.0, scores["compound"] + boost))
        if compound > 0.0:
            emoji = "🟢"
        elif compound < 0.0:
            emoji = "🔴"
        else:
            emoji = "⚪"
        item["sentiment"] = {**scores, "emoji": emoji}
    return headlines

def compute_sentiment_summary(headlines: list[dict]) -> dict:
    """Compute an overall sentiment summary from a list of news dicts with sentiment scores.
    
    Returns: {"label": "Bullish"/"Bearish"/"Neutral", "emoji": "🟢"/"🔴"/"⚪", "avg_compound": float}
    """
    compounds = []
    for item in headlines:
        s = item.get("sentiment", {})
        c = s.get("compound")
        if c is not None:
            compounds.append(c)
    if not compounds:
        return {"label": "Neutral", "emoji": "⚪", "avg_compound": 0.0}
    avg = sum(compounds) / len(compounds)
    if avg > 0.05:
        label, emoji = "Bullish", "🟢"
    elif avg < -0.05:
        label, emoji = "Bearish", "🔴"
    else:
        label, emoji = "Neutral", "⚪"
    return {"label": label, "emoji": emoji, "avg_compound": round(avg, 4)}


def get_news_and_ratings(ticker: str) -> dict:
    """Combined news + analyst data + earnings + sentiment."""
    news = fetch_yahoo_rss(ticker)
    # Filter out error-only items before sentiment analysis
    valid_news = [n for n in news if "error" not in n]
    analyze_sentiment(valid_news)
    summary = compute_sentiment_summary(valid_news)
    # Merge valid news back (error items go at the end unchanged)
    processed = valid_news + [n for n in news if "error" in n]
    return {
        "ticker": ticker.upper(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analyst": fetch_analyst_ratings(ticker),
        "earnings": fetch_earnings_date(ticker),
        "news": processed,
        "sentiment_summary": summary,
    }


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    result = get_news_and_ratings(t)
    print(json.dumps(result, indent=2))
