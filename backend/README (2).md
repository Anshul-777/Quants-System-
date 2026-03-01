# Quantitative Trading Intelligence System

## Architecture

```
Polygon WebSocket (AM.AAPL, AM.MSFT, AM.NVDA, AM.AMZN, AM.TSLA)
  └─▶ Rolling Buffer (50 bars per ticker)
       └─▶ 24 Econometric Features (log ret, vols, MACD, RSI, etc.)
            └─▶ XGBoost Classifier (predict_proba)
                 └─▶ Sharpe-Optimized Threshold τ
                      └─▶ Demo PnL Engine
                           └─▶ FastAPI WebSocket Broadcast → /ws


Training Pipeline (Colab):
Polygon REST API → Clean/Validate → Feature Engineering →
Walk-Forward Split → XGBoost + Early Stopping →
Threshold Optimization (Sharpe) → model.pkl + scaler.pkl
```

## Files

| File | Purpose |
|------|---------|
| `colab_training.py` | Full training pipeline — run in Google Colab |
| `colab_backend_test.py` | Colab testing script for the live backend |
| `backend/app.py` | FastAPI application with all endpoints |
| `backend/trading_engine.py` | WebSocket engine + rolling feature buffers |
| `backend/model.py` | XGBoost wrapper + artifact loading |
| `backend/requirements.txt` | Python dependencies |

## Training (Google Colab)

```python
# 1. Upload colab_training.py to Colab
# 2. Install deps
!pip install xgboost scikit-learn pandas numpy requests scipy joblib

# 3. Run — fetches ~3 years of minute data, trains, saves artifacts
%run colab_training.py

# 4. Download: model.pkl, scaler.pkl, features.json, threshold.json
from google.colab import files
for f in ["model.pkl","scaler.pkl","features.json","threshold.json"]:
    files.download(f)
```

## Local Testing (Colab or local)

```bash
cd backend
pip install -r requirements.txt
# Copy model artifacts into backend/
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Visit http://localhost:8000/docs
# POST /start_simulation → connect ws://localhost:8000/ws
```

## Render Deployment

1. Push `backend/` directory to GitHub
2. Create **Web Service** on Render
3. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add **Environment Variable**: `POLYGON_API_KEY=MoyLn951WdZAozaSClrOGar9xgYjy0pR`
5. Upload `model.pkl`, `scaler.pkl`, `features.json`, `threshold.json` into the repo root
6. Deploy → visit `https://your-app.onrender.com/docs`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | HTML dashboard (auto-refresh) |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/start_auto` | Start live Polygon stream |
| `POST` | `/start_simulation` | Start synthetic data stream |
| `POST` | `/stop` | Stop engine |
| `GET` | `/status` | Full system status |
| `GET` | `/portfolio` | Portfolio metrics |
| `GET` | `/history?n=100` | Last N prediction events |
| `GET` | `/model_info` | Model metadata + metrics |
| `GET` | `/features` | Live feature values per ticker |
| `POST` | `/predict/{ticker}` | On-demand single bar prediction |
| `WS` | `/ws` | Live JSON prediction stream |

## WebSocket Message Format

```json
{
  "event": "bar",
  "ticker": "AAPL",
  "datetime": "2024-03-01T14:32:00+00:00",
  "close": 185.42,
  "probability": 0.6123,
  "signal": 1,
  "threshold": 0.512,
  "position": 1,
  "realized_pnl": 0.0234,
  "n_trades": 12,
  "portfolio": {
    "equity": 100234.12,
    "total_pnl": 234.12,
    "max_drawdown": -0.0021,
    "n_trades": 47,
    "bars_processed": 1240
  },
  "features": {
    "log_ret": 0.0012,
    "rvol_5": 0.0034,
    "rvol_20": 0.0089,
    "ewma_vol": 0.0071,
    "macd": 0.124,
    "rsi_14": 58.3
  }
}
```

## Feature Engineering (24 Features)

| Feature | Definition |
|---------|-----------|
| `log_ret` | ln(c_t / c_{t-1}) |
| `rvol_5/10/20` | √(Σr²) over window |
| `ewma_vol` | EWMA variance (λ=0.94) |
| `mom_5/10/20` | (c_t - c_{t-n}) / c_{t-n} |
| `macd` | EMA12 - EMA26 |
| `macd_signal` | EMA9 of MACD |
| `macd_hist` | MACD - Signal |
| `price_vwap_dev` | (c_t - vwap) / vwap |
| `vol_zscore` | Volume z-score (20-window) |
| `hl_range` | (H - L) / C |
| `vol_ratio` | rvol_5 / rvol_20 |
| `skew_20` | Rolling skewness |
| `kurt_20` | Rolling kurtosis |
| `rsi_14` | RSI (14-period) |
| `bb_pos` | Bollinger Band position |
| `autocorr_1` | Lag-1 autocorrelation |
| `time_sin/cos` | Intraday periodicity |
| `day_of_week` | 0–4 |
| `ticker_id` | Integer encoding |

## Walk-Forward Validation

```
Train:  2021-01-01 → 2022-12-31  (~200k rows × 5 tickers)
Val:    2023-01-01 → 2023-12-31  (threshold optimization)
Test:   2024-01-01 → 2024-12-31  (out-of-sample PnL)
```

No shuffle. No overlap. No leakage (forward return excluded from features).

## Target Definition

```
y_t = 1  if  ln(c_{t+5}/c_t) > α × σ_t^(20)
y_t = 0  otherwise

α = 0.1 (volatility multiplier)
```

Removes micro-noise; targets economically meaningful moves only.

## Transaction Cost Model

```
PnL_t = pos_{t-1} × (c_t - c_{t-1}) - 0.0005 × |pos_t - pos_{t-1}| × c_t
```
