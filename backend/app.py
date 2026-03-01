# =============================================================================
# backend/app.py — FastAPI Application
# =============================================================================
# Start locally:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Render deploy:  Set start command to: uvicorn app:app --host 0.0.0.0 --port $PORT
# Docs:           http://localhost:8000/docs
# WebSocket test: ws://localhost:8000/ws
# =============================================================================

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn

from model import get_model
from trading_engine import get_engine, TICKERS

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("app")

# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="Quantitative Trading Intelligence System",
    description="""
## Live XGBoost Trading Signal Engine

Real-time streaming prediction system powered by Polygon.io market data.

### Features
- Live Polygon WebSocket streaming (minute aggregates)
- Rolling feature computation (24 econometric features)
- XGBoost probability prediction with Sharpe-optimized threshold
- Demo portfolio with transaction cost accounting
- Continuous WebSocket broadcast to all connected clients

### Quick Start
1. `POST /start_auto` — start live data streaming
2. `GET /status` — check engine + portfolio status
3. `WS /ws` — connect to receive live prediction stream
4. `GET /history` — retrieve recent prediction log

### Architecture
```
Polygon WS → Rolling Buffer → Feature Engine → XGBoost → Demo PnL → /ws broadcast
```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# STARTUP / SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Quantitative Trading System...")

    # Load model
    try:
        model = get_model()
        if model.loaded:
            logger.info(f"Model ready. Threshold={model.threshold:.4f}")
        else:
            logger.warning("Model not loaded — running without predictions.")
    except Exception as e:
        logger.error(f"Model load error: {e}")

    # Initialize engine (do NOT auto-start streaming; wait for /start_auto)
    engine = get_engine()
    logger.info("Engine initialized. Call /start_auto to begin streaming.")


@app.on_event("shutdown")
async def shutdown_event():
    engine = get_engine()
    engine.stop()
    logger.info("Engine stopped.")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse, summary="Dashboard")
async def root():
    """Interactive HTML status page."""
    engine = get_engine()
    status = engine.status()
    model  = get_model()

    portfolio = status["portfolio"]
    ticker_rows = ""
    for ticker, ts in status["ticker_states"].items():
        prob_str = f"{ts['last_probability']:.4f}" if ts["last_probability"] is not None else "–"
        signal   = "🟢 BUY" if ts["last_signal"] == 1 else "🔴 FLAT"
        ticker_rows += f"""
        <tr>
            <td><strong>{ticker}</strong></td>
            <td>{ts['buffer_size']}/50</td>
            <td>{prob_str}</td>
            <td>{signal}</td>
            <td>{ts['last_position']}</td>
            <td>${ts['realized_pnl']:,.4f}</td>
            <td>{ts['n_trades']}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<title>QT Trading System</title>
<style>
  body {{ font-family: monospace; background: #0a0a0a; color: #00ff88; padding: 20px; }}
  h1 {{ color: #00aaff; }} h2 {{ color: #ffaa00; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
  th {{ background: #111; color: #ffaa00; }}
  tr:hover {{ background: #111; }}
  .status {{ color: {'#00ff88' if status['running'] else '#ff4444'}; }}
  .btn {{ background: #003366; color: white; border: none; padding: 10px 20px;
          cursor: pointer; margin: 5px; border-radius: 4px; }}
  .equity {{ font-size: 1.5em; color: #00aaff; }}
  pre {{ background: #111; padding: 10px; border-radius: 4px; }}
</style>
<meta http-equiv="refresh" content="5">
</head>
<body>
<h1>⚡ Quantitative Trading Intelligence System</h1>
<p>Auto-refreshes every 5 seconds. Use <a href="/docs" style="color:#00aaff">/docs</a> for API.</p>

<h2>System Status</h2>
<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Engine Running</td><td class="status">{'● LIVE' if status['running'] else '○ STOPPED'}</td></tr>
<tr><td>Model Loaded</td><td>{'✓ Yes' if status['model_loaded'] else '✗ No (upload model.pkl)'}</td></tr>
<tr><td>Threshold τ</td><td>{model.threshold:.4f}</td></tr>
<tr><td>Connected Clients</td><td>{status['n_connected_clients']}</td></tr>
<tr><td>Log Entries</td><td>{status['log_entries']}</td></tr>
</table>

<h2>Portfolio</h2>
<div class="equity">Equity: ${portfolio['equity']:,.2f}</div>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total PnL (log)</td><td>{portfolio['total_pnl']:,.6f}</td></tr>
<tr><td>Max Drawdown</td><td>{portfolio['max_drawdown']:.4%}</td></tr>
<tr><td>Total Trades</td><td>{portfolio['n_trades']}</td></tr>
<tr><td>Bars Processed</td><td>{portfolio['bars_processed']}</td></tr>
</table>

<h2>Per-Ticker State</h2>
<table>
<tr><th>Ticker</th><th>Buffer</th><th>Probability</th><th>Signal</th><th>Position</th><th>Realized PnL</th><th>Trades</th></tr>
{ticker_rows}
</table>

<h2>Controls</h2>
<button class="btn" onclick="fetch('/start_auto', {{method:'POST'}}).then(r=>r.json()).then(d=>alert(JSON.stringify(d)))">
▶ Start Live Stream
</button>
<button class="btn" onclick="fetch('/start_simulation', {{method:'POST'}}).then(r=>r.json()).then(d=>alert(JSON.stringify(d)))">
🔄 Start Simulation Mode
</button>
<button class="btn" onclick="fetch('/stop', {{method:'POST'}}).then(r=>r.json()).then(d=>alert(JSON.stringify(d)))">
■ Stop Engine
</button>

<h2>WebSocket Test</h2>
<pre id="ws_log" style="height:200px;overflow:auto;"></pre>
<script>
  const log = document.getElementById('ws_log');
  const wsUrl = (location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws';
  const ws = new WebSocket(wsUrl);
  ws.onmessage = e => {{
    const d = JSON.parse(e.data);
    const line = `[${{d.ticker||d.event}}] close=${{d.close||''}} p=${{d.probability||''}} sig=${{d.signal||''}} equity=${{d.portfolio?.equity||''}}\n`;
    log.textContent = line + log.textContent.slice(0, 2000);
  }};
  ws.onerror = () => log.textContent = 'WS connection error\n' + log.textContent;
</script>
</body>
</html>"""


@app.post("/start_auto", summary="Start live Polygon WebSocket streaming")
async def start_auto():
    """
    Start the live Polygon WebSocket connection.
    Authenticates, subscribes to AM channels, and begins processing bars.
    Model must be loaded (upload model.pkl, scaler.pkl, features.json, threshold.json).
    """
    engine = get_engine()
    model  = get_model()

    if engine._running:
        return {"status": "already_running", "message": "Engine is already streaming."}

    if not model.loaded:
        return {
            "status": "model_missing",
            "message": "Model not loaded. Upload model.pkl, scaler.pkl, features.json, threshold.json."
        }

    engine.start(simulate=False)
    return {
        "status":    "started",
        "mode":      "live",
        "tickers":   TICKERS,
        "threshold": model.threshold,
        "message":   "Polygon WebSocket streaming started. Connect to /ws for live feed."
    }


@app.post("/start_simulation", summary="Start synthetic data simulation (no API key required)")
async def start_simulation():
    """
    Start simulation mode with synthetic random-walk bars.
    Useful for testing predictions, PnL logic, and WebSocket output
    without requiring a live market session.
    """
    engine = get_engine()

    if engine._running:
        engine.stop()
        await asyncio.sleep(0.5)

    engine.start(simulate=True)
    return {
        "status":  "started",
        "mode":    "simulation",
        "tickers": TICKERS,
        "message": "Synthetic bars running. Connect to /ws for live feed."
    }


@app.post("/stop", summary="Stop the trading engine")
async def stop_engine():
    """Stop Polygon WebSocket and all bar processing."""
    engine = get_engine()
    engine.stop()
    return {"status": "stopped", "message": "Engine stopped."}


@app.get("/status", summary="Full system and portfolio status")
async def get_status():
    """
    Returns:
    - Engine running state
    - Model loaded state + metrics
    - Per-ticker buffer sizes and last predictions
    - Portfolio equity, PnL, drawdown, trades
    """
    engine = get_engine()
    model  = get_model()
    return {
        "engine":    engine.status(),
        "model":     model.info(),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/portfolio", summary="Current portfolio metrics")
async def portfolio():
    """Equity curve, PnL, max drawdown, trade count."""
    engine = get_engine()
    return {
        "portfolio":     engine.portfolio.__dict__,
        "ticker_states": {
            t: {
                "realized_pnl": round(s.realized_pnl, 6),
                "n_trades":     s.n_trades,
                "last_signal":  s.last_signal,
                "last_prob":    s.last_probability,
            }
            for t, s in engine.ticker_states.items()
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/history", summary="Recent prediction log")
async def history(n: int = Query(default=100, le=500, description="Number of recent events")):
    """
    Returns last N prediction events including:
    ticker, close, probability, signal, features, portfolio snapshot.
    """
    engine = get_engine()
    recent = engine.live_outputs[-n:]
    return {
        "count":  len(recent),
        "events": recent,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model_info", summary="Model metadata and performance metrics")
async def model_info():
    """
    Returns model validation metrics, test metrics, feature list,
    optimal threshold, and training metadata.
    """
    model = get_model()
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return model.info()


@app.get("/features", summary="Feature list and current values per ticker")
async def features():
    """Returns current feature vector for each ticker from their rolling buffers."""
    engine = get_engine()
    result = {}
    for ticker, state in engine.ticker_states.items():
        from trading_engine import compute_features
        feats = compute_features(state)
        result[ticker] = {
            "ready":    feats is not None,
            "features": feats or {},
            "buffer_size": len(state.buffer)
        }
    return {"tickers": result, "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict/{ticker}", summary="On-demand prediction for a single bar")
async def predict_bar(
    ticker: str,
    close: float = Query(..., description="Close price"),
    open_: float = Query(alias="open", default=None),
    high:  float = Query(default=None),
    low:   float = Query(default=None),
    volume: float = Query(default=100000),
    vwap:  float = Query(default=None),
):
    """
    Inject a synthetic bar for an on-demand prediction.
    Useful for testing the feature → model → signal pipeline.
    The bar is added to the ticker's rolling buffer.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Unknown ticker. Choose from {TICKERS}")

    engine = get_engine()
    model  = get_model()
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    import time as _time
    from trading_engine import Bar, compute_features

    bar = Bar(
        ticker    = ticker,
        timestamp = int(_time.time() * 1000),
        open      = open_ or close,
        high      = high  or close * 1.001,
        low       = low   or close * 0.999,
        close     = close,
        volume    = volume,
        vwap      = vwap  or close,
        n_trades  = 100,
        datetime  = datetime.utcnow().isoformat(),
    )

    state = engine.ticker_states[ticker]
    state.buffer.append(bar)
    feats = compute_features(state)

    if feats is None:
        return {
            "ready":       False,
            "buffer_size": len(state.buffer),
            "message":     f"Buffer needs {50 - len(state.buffer)} more bars"
        }

    result = model.predict(feats)
    return {
        "ticker":      ticker,
        "close":       close,
        "probability": result["probability"],
        "signal":      result["signal"],
        "threshold":   result["threshold"],
        "buffer_size": len(state.buffer),
        "features":    {k: round(v, 6) for k, v in feats.items()},
    }


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    ## Live WebSocket Feed

    Connect to receive real-time JSON events:

    **Bar event** (per minute per ticker):
    ```json
    {
      "event": "bar",
      "ticker": "AAPL",
      "datetime": "2024-01-15T14:30:00",
      "close": 185.42,
      "probability": 0.6123,
      "signal": 1,
      "threshold": 0.512,
      "position": 1,
      "realized_pnl": 0.0234,
      "portfolio": { "equity": 100234.12, "total_pnl": 234.12, "max_drawdown": -0.001 },
      "features": { "log_ret": 0.0012, "rvol_20": 0.0089, ... }
    }
    ```

    **Connection event:**
    ```json
    { "event": "connected", "message": "Live stream active: AM.AAPL,AM.MSFT,..." }
    ```

    Connect with:
    ```js
    const ws = new WebSocket("ws://your-app.onrender.com/ws");
    ws.onmessage = e => console.log(JSON.parse(e.data));
    ```
    """
    await websocket.accept()
    engine = get_engine()
    await engine.register_client(websocket)

    # Send welcome with current state
    await websocket.send_text(json.dumps({
        "event":     "welcome",
        "tickers":   TICKERS,
        "running":   engine._running,
        "portfolio": engine.portfolio.__dict__,
        "timestamp": datetime.utcnow().isoformat(),
        "message":   "Connected to QT Trading System. Use POST /start_auto or /start_simulation to begin streaming."
    }, default=str))

    try:
        while True:
            # Keep alive — client can optionally send pings
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data.strip().lower() in ("ping", "status"):
                    await websocket.send_text(json.dumps({
                        "event":     "pong",
                        "status":    engine.status(),
                        "timestamp": datetime.utcnow().isoformat()
                    }, default=str))
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({
                    "event":     "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "running":   engine._running,
                    "equity":    round(engine.portfolio.equity, 2)
                }))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await engine.unregister_client(websocket)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
