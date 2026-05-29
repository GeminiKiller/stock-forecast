#!/usr/bin/env python3
"""Fetch analyst ratings and news for a ticker."""
import json
import urllib.request
import re
from datetime import datetime


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


def get_news_and_ratings(ticker: str) -> dict:
    """Combined news + analyst data + earnings."""
    return {
        "ticker": ticker.upper(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analyst": fetch_analyst_ratings(ticker),
        "earnings": fetch_earnings_date(ticker),
        "news": fetch_yahoo_rss(ticker),
    }


if __name__ == "__main__":
    import sys
    t = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    result = get_news_and_ratings(t)
    print(json.dumps(result, indent=2))
