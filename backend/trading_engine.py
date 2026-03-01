# =============================================================================
# backend/trading_engine.py — Live Streaming + Rolling Feature Engine
# =============================================================================

import os
import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Deque

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosedError, WebSocketException

from model import get_model

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"
API_KEY        = os.environ.get("POLYGON_API_KEY", "MoyLn951WdZAozaSClrOGar9xgYjy0pR")
TICKERS        = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
SUBSCRIBE_CHANNELS = ",".join(f"AM.{t}" for t in TICKERS)

BUFFER_SIZE     = 50     # rolling window length (must cover max feature window = 26 EMA)
INITIAL_CAPITAL = 100_000.0
EWMA_LAMBDA     = 0.94
TRANSACTION_COST = 0.0005

# Reconnection settings
RECONNECT_DELAY_INIT = 2.0
RECONNECT_DELAY_MAX  = 60.0
RECONNECT_JITTER     = 1.0

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Bar:
    """A single minute bar, matching Polygon aggregate schema."""
    ticker:    str
    timestamp: int        # Unix ms
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    vwap:      float
    n_trades:  int
    datetime:  str = ""


@dataclass
class Position:
    ticker:     str
    shares:     float = 0.0
    entry_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_trades:  int = 0


@dataclass
class PortfolioState:
    cash:        float = INITIAL_CAPITAL
    equity:      float = INITIAL_CAPITAL
    total_pnl:   float = 0.0
    n_trades:    int   = 0
    peak_equity: float = INITIAL_CAPITAL
    max_drawdown: float = 0.0
    bars_processed: int = 0
    started_at:  str = ""


@dataclass
class TickerState:
    """Per-ticker rolling buffers and last known state."""
    ticker: str
    buffer: Deque[Bar] = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))

    # Cached values
    last_probability: Optional[float] = None
    last_signal:      int = 0
    last_position:    int = 0    # 0 or 1
    last_bar:         Optional[Bar] = None

    # EWMA variance (recursive)
    ewma_var:         float = 0.0

    # EMA accumulators
    ema12:  float = 0.0
    ema26:  float = 0.0
    ema9:   float = 0.0          # MACD signal EMA

    # Position for PnL
    position_open:    bool = False
    entry_price:      float = 0.0
    realized_pnl:     float = 0.0
    n_trades:         int = 0


# =============================================================================
# ROLLING FEATURE COMPUTATION
# =============================================================================

TICKER_ID_MAP = {t: float(i) for i, t in enumerate(sorted(TICKERS))}


def _log_ret_from_buffer(buf: Deque[Bar]) -> np.ndarray:
    """Compute log returns from close prices in buffer."""
    closes = np.array([b.close for b in buf])
    if len(closes) < 2:
        return np.array([])
    return np.log(closes[1:] / closes[:-1])


def _rolling_realized_vol(log_rets: np.ndarray, window: int) -> Optional[float]:
    if len(log_rets) < window:
        return None
    return float(np.sqrt(np.sum(log_rets[-window:] ** 2)))


def _rolling_moment(log_rets: np.ndarray, window: int, moment: int) -> Optional[float]:
    if len(log_rets) < window:
        return None
    r = log_rets[-window:]
    mu = r.mean()
    std = r.std()
    if std < 1e-12:
        return None
    return float(((r - mu) ** moment).mean() / (std ** moment))


def _ema_update(prev: float, current: float, span: int) -> float:
    alpha = 2.0 / (span + 1)
    return alpha * current + (1 - alpha) * prev


def compute_features(state: TickerState) -> Optional[Dict[str, float]]:
    """
    Compute feature vector from rolling buffer.
    Returns None if buffer is insufficient (NaNs not allowed).
    Matches the feature set from training exactly.
    """
    buf = state.buffer
    if len(buf) < 27:   # need at least 27 bars for EMA-26
        return None

    bars   = list(buf)
    closes = np.array([b.close for b in bars])
    vols   = np.array([b.volume for b in bars])
    highs  = np.array([b.high for b in bars])
    lows   = np.array([b.low for b in bars])
    vwaps  = np.array([b.vwap for b in bars])

    log_rets = _log_ret_from_buffer(buf)
    if len(log_rets) == 0:
        return None

    last_ret = float(log_rets[-1])
    last_close = closes[-1]
    last_high  = highs[-1]
    last_low   = lows[-1]
    last_vol   = vols[-1]
    last_vwap  = vwaps[-1]

    # Realized vols
    rvol_5  = _rolling_realized_vol(log_rets, 5)
    rvol_10 = _rolling_realized_vol(log_rets, 10)
    rvol_20 = _rolling_realized_vol(log_rets, 20)

    if rvol_5 is None or rvol_10 is None or rvol_20 is None:
        return None

    # EWMA volatility
    state.ewma_var = EWMA_LAMBDA * state.ewma_var + (1 - EWMA_LAMBDA) * last_ret ** 2
    ewma_vol = float(np.sqrt(state.ewma_var))

    # Momentum
    def momentum(n):
        if len(closes) <= n:
            return None
        ref = closes[-n - 1]
        return float((last_close - ref) / ref) if ref > 0 else None

    mom_5  = momentum(5)
    mom_10 = momentum(10)
    mom_20 = momentum(20)
    if mom_5 is None or mom_10 is None or mom_20 is None:
        return None

    # EMA-12, EMA-26 (incremental via state)
    if state.ema12 == 0.0:
        state.ema12 = last_close
        state.ema26 = last_close
    state.ema12 = _ema_update(state.ema12, last_close, 12)
    state.ema26 = _ema_update(state.ema26, last_close, 26)
    macd = state.ema12 - state.ema26

    if state.ema9 == 0.0:
        state.ema9 = macd
    state.ema9   = _ema_update(state.ema9, macd, 9)
    macd_signal  = state.ema9
    macd_hist    = macd - macd_signal

    # Price vs VWAP
    price_vwap_dev = float((last_close - last_vwap) / last_vwap) if last_vwap > 0 else 0.0

    # Volume z-score
    if len(vols) >= 5:
        vol_mean = float(vols[-20:].mean())
        vol_std  = float(vols[-20:].std())
        vol_zscore = float((last_vol - vol_mean) / vol_std) if vol_std > 0 else 0.0
    else:
        vol_zscore = 0.0

    # High-Low range
    hl_range = float((last_high - last_low) / last_close) if last_close > 0 else 0.0

    # Volatility ratio
    vol_ratio = float(rvol_5 / rvol_20) if rvol_20 > 0 else 1.0

    # Skewness and kurtosis (20-window)
    skew_20 = _rolling_moment(log_rets, 20, 3)
    kurt_20 = _rolling_moment(log_rets, 20, 4)
    if skew_20 is None or kurt_20 is None:
        skew_20 = 0.0
        kurt_20 = 0.0

    # RSI (14)
    if len(log_rets) >= 14:
        gains  = np.where(log_rets[-14:] > 0, log_rets[-14:], 0.0)
        losses = np.where(log_rets[-14:] < 0, -log_rets[-14:], 0.0)
        avg_g  = gains.mean()
        avg_l  = losses.mean()
        rs     = avg_g / avg_l if avg_l > 0 else 100.0
        rsi_14 = float(100 - 100 / (1 + rs))
    else:
        rsi_14 = 50.0

    # Bollinger Band position
    if len(closes) >= 20:
        sma20 = float(closes[-20:].mean())
        std20 = float(closes[-20:].std())
        bb_pos = float((last_close - sma20) / (2 * std20)) if std20 > 0 else 0.0
    else:
        bb_pos = 0.0

    # Autocorrelation lag-1
    if len(log_rets) >= 20:
        r = log_rets[-20:]
        n = len(r)
        mu = r.mean()
        denom = np.sum((r - mu) ** 2)
        numer = np.sum((r[1:] - mu) * (r[:-1] - mu))
        autocorr_1 = float(numer / denom) if denom > 0 else 0.0
    else:
        autocorr_1 = 0.0

    # Intraday time features (use last bar datetime)
    last_bar = bars[-1]
    try:
        dt = datetime.fromisoformat(last_bar.datetime) if last_bar.datetime else datetime.utcnow()
    except Exception:
        dt = datetime.utcnow()

    minutes_since_open = max(0, dt.hour * 60 + dt.minute - 9 * 60 - 30)
    T = 390
    time_sin = float(np.sin(2 * np.pi * minutes_since_open / T))
    time_cos = float(np.cos(2 * np.pi * minutes_since_open / T))
    day_of_week = float(dt.weekday())

    # Ticker id
    ticker_id = TICKER_ID_MAP.get(state.ticker, 0.0)

    return {
        "log_ret":      last_ret,
        "rvol_5":       rvol_5,
        "rvol_10":      rvol_10,
        "rvol_20":      rvol_20,
        "ewma_vol":     ewma_vol,
        "mom_5":        mom_5,
        "mom_10":       mom_10,
        "mom_20":       mom_20,
        "macd":         float(macd),
        "macd_signal":  float(macd_signal),
        "macd_hist":    float(macd_hist),
        "price_vwap_dev": price_vwap_dev,
        "vol_zscore":   vol_zscore,
        "hl_range":     hl_range,
        "vol_ratio":    vol_ratio,
        "skew_20":      skew_20,
        "kurt_20":      kurt_20,
        "rsi_14":       rsi_14,
        "bb_pos":       bb_pos,
        "autocorr_1":   autocorr_1,
        "time_sin":     time_sin,
        "time_cos":     time_cos,
        "day_of_week":  day_of_week,
        "ticker_id":    ticker_id,
    }


# =============================================================================
# DEMO TRADING LOGIC
# =============================================================================

def update_demo_position(
    ticker_state: TickerState,
    portfolio: PortfolioState,
    new_signal: int,
    current_bar: Bar,
):
    """
    Section 13 PnL logic:
    pos_t = 1 if p_hat > tau else 0
    PnL_t = pos_{t-1} * (c_t - c_{t-1}) - cost * |pos_t - pos_{t-1}| * c_t
    Updates portfolio equity in place.
    """
    close = current_bar.close
    prev_pos = ticker_state.last_position
    new_pos  = new_signal

    # Trade PnL from previous position
    if prev_pos == 1 and ticker_state.last_bar is not None:
        prev_close = ticker_state.last_bar.close
        price_change = close - prev_close
        trade_pnl = price_change  # per-share; we allocate 1 share for simplicity
    else:
        trade_pnl = 0.0

    # Transaction cost
    trade = abs(new_pos - prev_pos)
    cost  = TRANSACTION_COST * trade * close

    net_pnl = trade_pnl - cost

    ticker_state.realized_pnl += net_pnl
    if trade > 0:
        ticker_state.n_trades += 1

    # Update portfolio
    portfolio.total_pnl  += net_pnl
    portfolio.equity      = portfolio.cash + portfolio.total_pnl
    portfolio.n_trades   += trade
    portfolio.bars_processed += 1

    # Drawdown tracking
    if portfolio.equity > portfolio.peak_equity:
        portfolio.peak_equity = portfolio.equity
    current_dd = (portfolio.equity - portfolio.peak_equity) / portfolio.peak_equity
    if current_dd < portfolio.max_drawdown:
        portfolio.max_drawdown = current_dd

    # Update state
    ticker_state.last_position = new_pos
    ticker_state.last_bar      = current_bar


# =============================================================================
# TRADING ENGINE
# =============================================================================

class TradingEngine:
    """
    Manages:
    - Polygon WebSocket connection with exponential-backoff reconnection
    - Per-ticker rolling feature buffers
    - XGBoost live prediction
    - Demo portfolio PnL tracking
    - Broadcasting results to FastAPI WebSocket clients
    """

    def __init__(self):
        self.model     = get_model()
        self.ticker_states: Dict[str, TickerState] = {
            t: TickerState(ticker=t) for t in TICKERS
        }
        self.portfolio = PortfolioState(
            started_at=datetime.utcnow().isoformat()
        )
        self.live_outputs: List[Dict[str, Any]] = []    # circular log
        self.max_log = 500
        self._ws_clients: List[Any] = []                # FastAPI ws connections
        self._running = False
        self._ws_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ── Client management ────────────────────────────────────────────────────

    async def register_client(self, ws):
        async with self._lock:
            self._ws_clients.append(ws)

    async def unregister_client(self, ws):
        async with self._lock:
            try:
                self._ws_clients.remove(ws)
            except ValueError:
                pass

    async def _broadcast(self, payload: Dict[str, Any]):
        """Send JSON payload to all connected FastAPI WebSocket clients."""
        if not self._ws_clients:
            return
        message = json.dumps(payload, default=str)
        dead = []
        for ws in list(self._ws_clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.unregister_client(ws)

    # ── Bar processing ───────────────────────────────────────────────────────

    async def process_bar(self, bar: Bar):
        """
        Core pipeline per incoming bar:
        1. Update rolling buffer
        2. Compute features
        3. Predict
        4. Update demo position
        5. Build output payload
        6. Broadcast
        """
        state = self.ticker_states[bar.ticker]
        state.buffer.append(bar)

        features = compute_features(state)
        if features is None:
            logger.debug(f"[{bar.ticker}] Buffer not ready ({len(state.buffer)}/{BUFFER_SIZE})")
            return

        result = self.model.predict(features)
        prob   = result["probability"]
        signal = result["signal"]

        # Update demo PnL
        update_demo_position(state, self.portfolio, signal, bar)

        state.last_probability = prob
        state.last_signal      = signal

        output = {
            "event":       "bar",
            "ticker":      bar.ticker,
            "timestamp":   bar.timestamp,
            "datetime":    bar.datetime,
            "close":       bar.close,
            "volume":      bar.volume,
            "probability": prob,
            "signal":      signal,
            "threshold":   self.model.threshold,
            "position":    state.last_position,
            "realized_pnl": round(state.realized_pnl, 4),
            "n_trades":    state.n_trades,
            "portfolio": {
                "equity":        round(self.portfolio.equity, 2),
                "total_pnl":     round(self.portfolio.total_pnl, 4),
                "max_drawdown":  round(self.portfolio.max_drawdown, 6),
                "n_trades":      self.portfolio.n_trades,
                "bars_processed": self.portfolio.bars_processed,
            },
            "features": {k: round(v, 6) for k, v in features.items()},
        }

        # Rolling log
        self.live_outputs.append(output)
        if len(self.live_outputs) > self.max_log:
            self.live_outputs.pop(0)

        await self._broadcast(output)
        logger.info(
            f"[{bar.ticker}] close={bar.close:.2f} "
            f"p={prob:.4f} sig={signal} "
            f"equity=${self.portfolio.equity:,.2f}"
        )

    # ── Polygon WebSocket ────────────────────────────────────────────────────

    async def _parse_polygon_message(self, raw: str):
        """Parse Polygon aggregate minute message (AM.*) into Bar."""
        try:
            msgs = json.loads(raw)
        except json.JSONDecodeError:
            return

        if not isinstance(msgs, list):
            msgs = [msgs]

        for msg in msgs:
            ev = msg.get("ev")
            if ev == "AM":
                ticker = msg.get("sym", "")
                if ticker not in TICKERS:
                    continue

                ts = msg.get("s", 0)  # start timestamp in ms
                try:
                    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                except Exception:
                    dt = datetime.utcnow().isoformat()

                bar = Bar(
                    ticker    = ticker,
                    timestamp = ts,
                    open      = float(msg.get("o", 0)),
                    high      = float(msg.get("h", 0)),
                    low       = float(msg.get("l", 0)),
                    close     = float(msg.get("c", 0)),
                    volume    = float(msg.get("av", msg.get("v", 0))),
                    vwap      = float(msg.get("vw", msg.get("c", 0))),
                    n_trades  = int(msg.get("z", 0)),
                    datetime  = dt,
                )

                if bar.close <= 0:
                    continue

                await self.process_bar(bar)

            elif ev == "status":
                status  = msg.get("status", "")
                message = msg.get("message", "")
                logger.info(f"Polygon status: {status} — {message}")

                if status == "auth_success":
                    logger.info("Polygon authenticated. Subscribing to channels...")
                    return {"action": "subscribe", "params": SUBSCRIBE_CHANNELS}

    async def _polygon_connect(self):
        """
        Connect to Polygon WebSocket, authenticate, subscribe.
        Returns the websocket object.
        """
        ws = await websockets.connect(
            POLYGON_WS_URL,
            ping_interval=20,
            ping_timeout=30,
        )
        return ws

    async def _polygon_session(self):
        """
        One full Polygon WebSocket session.
        Handles auth handshake, subscription, and continuous message loop.
        """
        ws = await self._polygon_connect()
        logger.info("Polygon WebSocket connected.")

        subscribed = False
        async for raw in ws:
            reply = await self._parse_polygon_message(raw)
            if reply and not subscribed:
                await ws.send(json.dumps(reply))
                subscribed = True
                logger.info(f"Subscribed to: {SUBSCRIBE_CHANNELS}")

                # Announce connection
                await self._broadcast({
                    "event": "connected",
                    "message": f"Streaming {SUBSCRIBE_CHANNELS}",
                    "timestamp": datetime.utcnow().isoformat()
                })

        await ws.close()

    async def run_polygon_ws(self):
        """
        Outer reconnection loop with exponential backoff + jitter.
        Runs indefinitely.
        """
        self._running = True
        delay = RECONNECT_DELAY_INIT

        while self._running:
            try:
                logger.info(f"Connecting to Polygon WebSocket: {POLYGON_WS_URL}")

                # Auth first
                ws = await self._polygon_connect()

                # Wait for connected message then send auth
                auth_sent = False
                subscribed = False

                async for raw in ws:
                    try:
                        msgs = json.loads(raw)
                    except Exception:
                        continue

                    if not isinstance(msgs, list):
                        msgs = [msgs]

                    for msg in msgs:
                        ev     = msg.get("ev")
                        status = msg.get("status", "")
                        text   = msg.get("message", "")

                        if ev == "status" and status == "connected" and not auth_sent:
                            logger.info("Polygon connected — authenticating...")
                            await ws.send(json.dumps({
                                "action": "auth",
                                "params": API_KEY
                            }))
                            auth_sent = True

                        elif ev == "status" and status == "auth_success" and not subscribed:
                            logger.info("Polygon authenticated — subscribing...")
                            await ws.send(json.dumps({
                                "action": "subscribe",
                                "params": SUBSCRIBE_CHANNELS
                            }))
                            subscribed = True
                            await self._broadcast({
                                "event": "connected",
                                "message": f"Live stream active: {SUBSCRIBE_CHANNELS}",
                                "timestamp": datetime.utcnow().isoformat()
                            })

                        elif ev == "status" and status == "auth_failed":
                            raise RuntimeError("Polygon auth failed — check API key.")

                        elif ev == "AM":
                            # Process bar (re-route through standard path)
                            sym = msg.get("sym", "")
                            if sym not in TICKERS:
                                continue
                            ts = msg.get("s", 0)
                            try:
                                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
                            except Exception:
                                dt = datetime.utcnow().isoformat()

                            bar = Bar(
                                ticker    = sym,
                                timestamp = ts,
                                open      = float(msg.get("o", 0)),
                                high      = float(msg.get("h", 0)),
                                low       = float(msg.get("l", 0)),
                                close     = float(msg.get("c", 0)),
                                volume    = float(msg.get("av", msg.get("v", 0))),
                                vwap      = float(msg.get("vw", msg.get("c", 0))),
                                n_trades  = int(msg.get("z", 0)),
                                datetime  = dt,
                            )
                            if bar.close > 0:
                                await self.process_bar(bar)

                # If we exit the loop cleanly, reconnect
                delay = RECONNECT_DELAY_INIT

            except asyncio.CancelledError:
                logger.info("Trading engine cancelled — shutting down.")
                break

            except Exception as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in {delay:.1f}s...")
                await self._broadcast({
                    "event": "disconnected",
                    "error": str(e),
                    "reconnect_in": delay,
                    "timestamp": datetime.utcnow().isoformat()
                })
                await asyncio.sleep(delay)
                import random
                delay = min(delay * 2 + random.uniform(0, RECONNECT_JITTER),
                            RECONNECT_DELAY_MAX)

    # ── Simulation mode (market closed / testing) ───────────────────────────

    async def run_simulation(self):
        """
        Generate synthetic bars for testing when market is closed.
        Maintains same output format as live mode.
        """
        logger.info("Starting SIMULATION mode (synthetic bars)...")
        prices = {t: 150.0 + i * 50 for i, t in enumerate(TICKERS)}
        bar_count = 0

        while self._running:
            for ticker in TICKERS:
                # Random walk
                ret = np.random.normal(0, 0.001)
                prices[ticker] *= np.exp(ret)
                price = prices[ticker]

                bar = Bar(
                    ticker    = ticker,
                    timestamp = int(time.time() * 1000),
                    open      = price * (1 + np.random.uniform(-0.0005, 0.0005)),
                    high      = price * (1 + abs(np.random.uniform(0, 0.001))),
                    low       = price * (1 - abs(np.random.uniform(0, 0.001))),
                    close     = price,
                    volume    = float(np.random.randint(10000, 500000)),
                    vwap      = price * (1 + np.random.uniform(-0.0002, 0.0002)),
                    n_trades  = np.random.randint(100, 2000),
                    datetime  = datetime.utcnow().isoformat(),
                )
                await self.process_bar(bar)

            bar_count += 1
            await asyncio.sleep(2)   # 2-second cadence in simulation

    # ── Start ────────────────────────────────────────────────────────────────

    def start(self, simulate: bool = False):
        """Launch background task."""
        if self._ws_task and not self._ws_task.done():
            return

        self._running = True
        loop = asyncio.get_event_loop()

        if simulate:
            self._ws_task = loop.create_task(self.run_simulation())
        else:
            self._ws_task = loop.create_task(self.run_polygon_ws())

        logger.info("TradingEngine background task started.")

    def stop(self):
        self._running = False
        if self._ws_task:
            self._ws_task.cancel()

    def status(self) -> Dict[str, Any]:
        return {
            "running":    self._running,
            "model_loaded": self.model.loaded,
            "tickers":    TICKERS,
            "portfolio":  asdict(self.portfolio),
            "ticker_states": {
                t: {
                    "buffer_size":    len(s.buffer),
                    "last_probability": s.last_probability,
                    "last_signal":     s.last_signal,
                    "last_position":   s.last_position,
                    "realized_pnl":   round(s.realized_pnl, 4),
                    "n_trades":        s.n_trades,
                }
                for t, s in self.ticker_states.items()
            },
            "n_connected_clients": len(self._ws_clients),
            "log_entries":         len(self.live_outputs),
        }


# Singleton engine
_engine_instance: Optional[TradingEngine] = None


def get_engine() -> TradingEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TradingEngine()
    return _engine_instance
