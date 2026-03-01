# =============================================================================
# QUANTITATIVE TRADING INTELLIGENCE SYSTEM — COLAB TRAINING PIPELINE
# =============================================================================
# Run this entire file in Google Colab cell by cell (or as one block).
# After execution, model.pkl, scaler.pkl, features.json, threshold.json
# will be saved and ready for backend deployment.
# =============================================================================

# ── CELL 1: Install dependencies ─────────────────────────────────────────────
# !pip install xgboost scikit-learn pandas numpy requests scipy joblib --quiet

import os
import json
import time
import warnings
import requests
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

API_KEY        = "MoyLn951WdZAozaSClrOGar9xgYjy0pR"
TICKERS        = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
TRAIN_START    = "2021-01-01"
TRAIN_END      = "2022-12-31"
VAL_START      = "2023-01-01"
VAL_END        = "2023-12-31"
TEST_START     = "2024-01-01"
TEST_END       = "2024-12-31"

ALPHA          = 0.1        # volatility-adjustment multiplier for target
VOL_WINDOW     = 20         # rolling volatility window for target
FORWARD_STEPS  = 5          # forward return horizon
EWMA_LAMBDA    = 0.94       # EWMA decay
EMA_SHORT      = 12
EMA_LONG       = 26
BASE_DIR       = "/content"  # change to "." for local execution

MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH  = os.path.join(BASE_DIR, "features.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")

# =============================================================================
# SECTION 2 — DATA ACQUISITION
# =============================================================================

def fetch_polygon_minute_bars(ticker: str, from_date: str, to_date: str,
                               api_key: str, max_retries: int = 5) -> pd.DataFrame:
    """
    Fetch all minute bars from Polygon v2 aggregates with pagination.
    Handles Polygon's 50,000-row limit by paginating via 'next_url'.
    """
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute"
        f"/{from_date}/{to_date}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    all_results = []
    url = base_url

    while url:
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed fetching {ticker}: {e}")
                time.sleep(2 ** attempt)

        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)
        print(f"  [{ticker}] fetched {len(all_results):,} rows so far...")

        # Polygon pagination
        next_url = data.get("next_url")
        if next_url:
            url = next_url + f"&apiKey={api_key}"
        else:
            url = None

    if not all_results:
        raise ValueError(f"No data returned for {ticker} from {from_date} to {to_date}")

    df = pd.DataFrame(all_results)
    df["ticker"] = ticker
    return df


def load_all_tickers(tickers, from_date, to_date, api_key):
    frames = []
    for ticker in tickers:
        print(f"\nFetching {ticker} [{from_date} → {to_date}]...")
        try:
            df = fetch_polygon_minute_bars(ticker, from_date, to_date, api_key)
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: Skipping {ticker} due to error: {e}")
    if not frames:
        raise RuntimeError("No data fetched for any ticker.")
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# SECTION 3 — DATA CLEANING AND VALIDATION
# =============================================================================

def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Section 3 compliance:
    - Convert timestamp
    - Sort chronologically
    - Remove duplicates
    - Remove zero/negative prices
    - Validate high-low consistency
    - Drop zero-volume rows
    """
    df = df.copy()

    # Rename Polygon fields to standard names
    rename_map = {"t": "timestamp", "o": "open", "h": "high",
                  "l": "low",       "c": "close", "v": "volume",
                  "vw": "vwap",     "n": "n_trades"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    required = ["timestamp", "open", "high", "low", "close", "volume", "ticker"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert timestamp (milliseconds → datetime)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("America/New_York")

    # Chronological sort per ticker
    df.sort_values(["ticker", "datetime"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove duplicates
    before = len(df)
    df.drop_duplicates(subset=["ticker", "datetime"], inplace=True)
    print(f"  Removed {before - len(df):,} duplicate rows")

    # Validate prices > 0
    price_mask = (df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)
    dropped_prices = (~price_mask).sum()
    df = df[price_mask].copy()
    print(f"  Removed {dropped_prices:,} rows with zero/negative prices")

    # Validate high >= low, close within range
    consistency_mask = (df["high"] >= df["low"]) & \
                       (df["close"] <= df["high"] * 1.001) & \
                       (df["close"] >= df["low"] * 0.999)
    dropped_consistency = (~consistency_mask).sum()
    df = df[consistency_mask].copy()
    print(f"  Removed {dropped_consistency:,} rows with OHLC inconsistency")

    # Drop zero-volume rows during market hours
    market_open_hour = 9
    market_close_hour = 16
    in_market = (df["datetime"].dt.hour >= market_open_hour) & \
                (df["datetime"].dt.hour < market_close_hour)
    zero_vol = (df["volume"] == 0) & in_market
    df = df[~zero_vol].copy()
    print(f"  Removed {zero_vol.sum():,} zero-volume market-hours rows")

    df.reset_index(drop=True, inplace=True)
    print(f"  Clean shape: {df.shape}")
    return df


# =============================================================================
# SECTION 4 — FEATURE ENGINEERING
# =============================================================================

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Recursive EWMA variance → volatility (Section 4)."""
    var = np.zeros(len(returns))
    r = returns.fillna(0).values
    var[0] = r[0] ** 2
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i] ** 2
    return pd.Series(np.sqrt(var), index=returns.index)


def rolling_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    """Section 4: σ_t^(n) = sqrt(sum r^2 over window)."""
    return returns.pow(2).rolling(window, min_periods=window).sum().apply(np.sqrt)


def rolling_skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).skew()


def rolling_kurt(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).kurt()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features per ticker without cross-ticker contamination.
    Features align with Section 6 specification.
    """
    frames = []
    ticker_map = {t: i for i, t in enumerate(sorted(df["ticker"].unique()))}

    for ticker, grp in df.groupby("ticker"):
        g = grp.copy().sort_values("datetime").reset_index(drop=True)

        close = g["close"]
        volume = g["volume"]
        high   = g["high"]
        low    = g["low"]
        vwap   = g["vwap"] if "vwap" in g.columns else close

        # ── Log returns ─────────────────────────────────────────────────────
        g["log_ret"] = np.log(close / close.shift(1))

        # ── Realized volatility ──────────────────────────────────────────────
        g["rvol_5"]  = rolling_realized_vol(g["log_ret"], 5)
        g["rvol_10"] = rolling_realized_vol(g["log_ret"], 10)
        g["rvol_20"] = rolling_realized_vol(g["log_ret"], 20)

        # ── EWMA volatility ──────────────────────────────────────────────────
        g["ewma_vol"] = compute_ewma_vol(g["log_ret"], EWMA_LAMBDA)

        # ── Momentum ─────────────────────────────────────────────────────────
        g["mom_5"]  = (close - close.shift(5))  / close.shift(5).replace(0, np.nan)
        g["mom_10"] = (close - close.shift(10)) / close.shift(10).replace(0, np.nan)
        g["mom_20"] = (close - close.shift(20)) / close.shift(20).replace(0, np.nan)

        # ── MACD ─────────────────────────────────────────────────────────────
        ema12 = compute_ema(close, EMA_SHORT)
        ema26 = compute_ema(close, EMA_LONG)
        g["macd"]        = ema12 - ema26
        g["macd_signal"] = compute_ema(g["macd"], 9)
        g["macd_hist"]   = g["macd"] - g["macd_signal"]

        # ── Price vs VWAP ─────────────────────────────────────────────────────
        g["price_vwap_dev"] = (close - vwap) / vwap.replace(0, np.nan)

        # ── Volume z-score ────────────────────────────────────────────────────
        vol_mean = volume.rolling(20, min_periods=5).mean()
        vol_std  = volume.rolling(20, min_periods=5).std()
        g["vol_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

        # ── High-Low range ────────────────────────────────────────────────────
        g["hl_range"] = (high - low) / close.replace(0, np.nan)

        # ── Volatility ratio ─────────────────────────────────────────────────
        g["vol_ratio"] = g["rvol_5"] / g["rvol_20"].replace(0, np.nan)

        # ── Rolling skewness and kurtosis ─────────────────────────────────────
        g["skew_20"] = rolling_skew(g["log_ret"], 20)
        g["kurt_20"] = rolling_kurt(g["log_ret"], 20)

        # ── RSI (14) ──────────────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs    = gain / loss.replace(0, np.nan)
        g["rsi_14"] = 100 - (100 / (1 + rs))

        # ── Bollinger Band position ───────────────────────────────────────────
        sma20      = close.rolling(20, min_periods=20).mean()
        std20      = close.rolling(20, min_periods=20).std()
        g["bb_pos"] = (close - sma20) / (2 * std20.replace(0, np.nan))

        # ── Autocorrelation lag-1 return ──────────────────────────────────────
        g["autocorr_1"] = g["log_ret"].rolling(20, min_periods=10).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )

        # ── Intraday time features ────────────────────────────────────────────
        minutes_since_open = (
            g["datetime"].dt.hour * 60 + g["datetime"].dt.minute - 9 * 60 - 30
        ).clip(lower=0)
        T = 390  # trading minutes per day
        g["time_sin"] = np.sin(2 * np.pi * minutes_since_open / T)
        g["time_cos"] = np.cos(2 * np.pi * minutes_since_open / T)

        # ── Day of week ───────────────────────────────────────────────────────
        g["day_of_week"] = g["datetime"].dt.dayofweek.astype(float)

        # ── Ticker integer encoding ───────────────────────────────────────────
        g["ticker_id"] = float(ticker_map[ticker])

        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["ticker", "datetime"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# =============================================================================
# SECTION 5 — TARGET DEFINITION (volatility-adjusted forward return)
# =============================================================================

def compute_target(df: pd.DataFrame, alpha: float = ALPHA,
                   forward: int = FORWARD_STEPS,
                   vol_window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    y_t = 1  if  R^(5)_{t} > alpha * sigma^(20)_t
    Computed per ticker to prevent cross-contamination.
    CRITICAL: shift(-forward) is the forward look. This column is EXCLUDED
    from features — it is only the label.
    """
    frames = []
    for ticker, grp in df.groupby("ticker"):
        g = grp.copy().sort_values("datetime").reset_index(drop=True)

        # Forward return = sum of next `forward` log returns
        forward_ret = g["log_ret"].shift(-1).rolling(forward, min_periods=forward).sum()
        # This is equivalent to shift(-forward) accumulated — use shift(-forward) directly
        # R^(5)_t = ln(c_{t+5}/c_t)
        forward_ret = np.log(g["close"].shift(-forward) / g["close"])

        sigma_20 = rolling_realized_vol(g["log_ret"], vol_window)
        threshold = alpha * sigma_20

        g["target"] = (forward_ret > threshold).astype(float)
        # Mark rows where target cannot be computed (last `forward` rows)
        g.loc[g.index[-forward:], "target"] = np.nan

        frames.append(g)

    result = pd.concat(frames, ignore_index=True)
    return result


# =============================================================================
# SECTION 6 — FEATURE LIST AND CLEANING
# =============================================================================

FEATURE_COLS = [
    "log_ret", "rvol_5", "rvol_10", "rvol_20", "ewma_vol",
    "mom_5", "mom_10", "mom_20",
    "macd", "macd_signal", "macd_hist",
    "price_vwap_dev", "vol_zscore", "hl_range", "vol_ratio",
    "skew_20", "kurt_20", "rsi_14", "bb_pos", "autocorr_1",
    "time_sin", "time_cos", "day_of_week", "ticker_id"
]


def prepare_dataset(df: pd.DataFrame):
    """
    Drop NaN rows (never fill rolling NaNs with zero).
    Remove constant and high-correlation features.
    Returns X, y, datetime index.
    """
    cols_needed = FEATURE_COLS + ["target", "datetime", "ticker"]
    df_clean = df[cols_needed].copy()

    # Drop rows with NaN in features or target
    before = len(df_clean)
    df_clean.dropna(subset=FEATURE_COLS + ["target"], inplace=True)
    print(f"  Dropped {before - len(df_clean):,} NaN rows → {len(df_clean):,} remaining")

    # Remove constant columns
    constant_cols = [c for c in FEATURE_COLS if df_clean[c].nunique() <= 1]
    if constant_cols:
        print(f"  Removing constant columns: {constant_cols}")
    active_features = [c for c in FEATURE_COLS if c not in constant_cols]

    # Remove highly correlated features (|ρ| > 0.95)
    corr_matrix = df_clean[active_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop_corr:
        print(f"  Removing high-correlation columns: {to_drop_corr}")
    active_features = [c for c in active_features if c not in to_drop_corr]

    X = df_clean[active_features].values.astype(np.float32)
    y = df_clean["target"].values.astype(np.int32)
    datetimes = df_clean["datetime"].values

    print(f"  Final features: {len(active_features)} | Samples: {len(X):,}")
    print(f"  Target balance: {y.mean():.3f} positive rate")

    return X, y, datetimes, active_features, df_clean


# =============================================================================
# SECTION 7 — WALK-FORWARD SPLIT
# =============================================================================

def walk_forward_split(df_clean: pd.DataFrame, active_features: list):
    """
    Section 8: Train 2021-2022 | Val 2023 | Test 2024
    No shuffle. No overlap. No leakage.
    """
    dt = pd.to_datetime(df_clean["datetime"])

    train_mask = (dt >= TRAIN_START) & (dt <= TRAIN_END)
    val_mask   = (dt >= VAL_START)   & (dt <= VAL_END)
    test_mask  = (dt >= TEST_START)  & (dt <= TEST_END)

    X_all = df_clean[active_features].values.astype(np.float32)
    y_all = df_clean["target"].values.astype(np.int32)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# SECTION 8 — XGBOOST TRAINING
# =============================================================================

def train_model(X_train, y_train, X_val, y_val):
    """
    Section 9: XGBoost with early stopping on validation Logloss.
    """
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\nscale_pos_weight = {scale_pos_weight:.3f}")

    model = XGBClassifier(
        objective         = "binary:logistic",
        eval_metric       = "logloss",
        max_depth         = 6,
        learning_rate     = 0.03,
        n_estimators      = 1200,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        gamma             = 0.1,
        reg_alpha         = 0.01,
        reg_lambda        = 1.0,
        scale_pos_weight  = scale_pos_weight,
        tree_method       = "hist",
        use_label_encoder = False,
        random_state      = 42,
        n_jobs            = -1,
        early_stopping_rounds = 50,
    )

    model.fit(
        X_train, y_train,
        eval_set        = [(X_train, y_train), (X_val, y_val)],
        verbose         = 100,
    )

    print(f"\nBest iteration: {model.best_iteration}")
    return model


# =============================================================================
# SECTION 9 — THRESHOLD OPTIMIZATION (Sharpe maximization)
# =============================================================================

def simulate_pnl(proba: np.ndarray, returns: np.ndarray, tau: float,
                 cost: float = 0.0005) -> np.ndarray:
    """
    Section 13 trading logic applied to validation set.
    pos_t = 1 if p_hat > tau else 0
    PnL_t = pos_{t-1} * r_t - cost * |pos_t - pos_{t-1}|
    """
    pos = (proba > tau).astype(float)
    pos_lag = np.concatenate([[0.0], pos[:-1]])
    trade_cost = cost * np.abs(pos - pos_lag)
    pnl = pos_lag * returns - trade_cost
    return pnl


def optimize_threshold(model, X_val: np.ndarray, df_val: pd.DataFrame,
                        active_features: list) -> float:
    """
    Section 10: grid search τ ∈ [0.45, 0.55] step 0.005
    Maximize annualized Sharpe ratio on validation set.
    """
    proba_val = model.predict_proba(X_val)[:, 1]
    ret_val   = df_val["log_ret"].values

    best_tau    = 0.5
    best_sharpe = -np.inf
    results = []

    taus = np.arange(0.45, 0.555, 0.005)
    for tau in taus:
        pnl = simulate_pnl(proba_val, ret_val, tau)
        if pnl.std() < 1e-10:
            continue
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 390)  # annualized
        results.append((tau, sharpe))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_tau = tau

    print(f"\nThreshold optimization results:")
    for tau, sharpe in results:
        marker = " ◄ BEST" if abs(tau - best_tau) < 1e-9 else ""
        print(f"  τ={tau:.3f}  Sharpe={sharpe:.4f}{marker}")

    return float(best_tau), float(best_sharpe)


# =============================================================================
# SECTION 10 — EVALUATION METRICS
# =============================================================================

def evaluate(model, X: np.ndarray, y: np.ndarray,
             df_sub: pd.DataFrame, tau: float, label: str):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba > tau).astype(int)

    auc    = roc_auc_score(y, proba)
    prec   = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1     = f1_score(y, pred, zero_division=0)

    ret = df_sub["log_ret"].values
    pnl = simulate_pnl(proba, ret, tau)
    cum_pnl = np.cumsum(pnl)
    total_return = cum_pnl[-1]
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(252 * 390) if pnl.std() > 0 else 0.0

    # Max drawdown
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown    = cum_pnl - running_max
    max_dd      = drawdown.min()

    print(f"\n── {label} ──────────────────────────────────")
    print(f"  AUC-ROC:      {auc:.4f}")
    print(f"  Precision:    {prec:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1:           {f1:.4f}")
    print(f"  Sharpe:       {sharpe:.4f}")
    print(f"  Total PnL:    {total_return:.4f}  (log return units)")
    print(f"  Max Drawdown: {max_dd:.4f}")
    print(f"  Trades:       {pred.sum():,} / {len(pred):,}")

    return {
        "auc": auc, "precision": prec, "recall": recall,
        "f1": f1, "sharpe": sharpe, "total_pnl": total_return,
        "max_drawdown": max_dd, "n_trades": int(pred.sum())
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 65)
    print("  QUANTITATIVE TRADING SYSTEM — TRAINING PIPELINE")
    print("=" * 65)

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    print("\n[1] Fetching historical minute bars from Polygon...")
    raw_train = load_all_tickers(TICKERS, TRAIN_START, TEST_END, API_KEY)
    print(f"  Total raw rows: {len(raw_train):,}")

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    print("\n[2] Cleaning and validating data...")
    clean_df = clean_and_validate(raw_train)

    # ── 3. Features ──────────────────────────────────────────────────────────
    print("\n[3] Engineering features...")
    feat_df = add_features(clean_df)
    print(f"  Feature frame shape: {feat_df.shape}")

    # ── 4. Target ────────────────────────────────────────────────────────────
    print("\n[4] Computing volatility-adjusted targets...")
    full_df = compute_target(feat_df)
    target_rate = full_df["target"].mean()
    print(f"  Global target positive rate: {target_rate:.4f}")

    # ── 5. Prepare dataset ───────────────────────────────────────────────────
    print("\n[5] Preparing dataset (removing NaN rows)...")
    X, y, datetimes, active_features, df_clean_final = prepare_dataset(full_df)

    # ── 6. Walk-forward split ────────────────────────────────────────────────
    print("\n[6] Walk-forward splitting...")
    X_train, y_train, X_val, y_val, X_test, y_test = walk_forward_split(
        df_clean_final, active_features
    )

    # Corresponding raw-return series for PnL simulation
    dt_col = pd.to_datetime(df_clean_final["datetime"])
    val_mask   = (dt_col >= VAL_START)   & (dt_col <= VAL_END)
    test_mask  = (dt_col >= TEST_START)  & (dt_col <= TEST_END)
    df_val_sub  = df_clean_final[val_mask].reset_index(drop=True)
    df_test_sub = df_clean_final[test_mask].reset_index(drop=True)

    # ── 7. Scale ─────────────────────────────────────────────────────────────
    print("\n[7] Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # ── 8. Train XGBoost ─────────────────────────────────────────────────────
    print("\n[8] Training XGBoost with early stopping...")
    model = train_model(X_train_s, y_train, X_val_s, y_val)

    # ── 9. Threshold optimization ─────────────────────────────────────────────
    print("\n[9] Optimizing decision threshold on validation set...")
    best_tau, best_sharpe = optimize_threshold(model, X_val_s, df_val_sub, active_features)
    print(f"\n  → Optimal threshold τ = {best_tau:.4f}  (Sharpe = {best_sharpe:.4f})")

    # ── 10. Evaluate ─────────────────────────────────────────────────────────
    print("\n[10] Evaluating model performance...")
    train_metrics = evaluate(model, X_train_s, y_train,
                             df_clean_final[dt_col <= TRAIN_END].reset_index(drop=True),
                             best_tau, "TRAIN SET")
    val_metrics   = evaluate(model, X_val_s, y_val, df_val_sub,  best_tau, "VALIDATION SET")
    test_metrics  = evaluate(model, X_test_s, y_test, df_test_sub, best_tau, "TEST SET (OUT-OF-SAMPLE)")

    # ── 11. Save artifacts ────────────────────────────────────────────────────
    print("\n[11] Saving model artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump({"features": active_features, "n_features": len(active_features)}, f, indent=2)

    threshold_data = {
        "threshold": best_tau,
        "val_sharpe": best_sharpe,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "alpha": ALPHA,
        "forward_steps": FORWARD_STEPS,
        "vol_window": VOL_WINDOW,
        "tickers": TICKERS,
        "trained_at": datetime.utcnow().isoformat()
    }
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(threshold_data, f, indent=2)

    print(f"\n  ✓ model.pkl     → {MODEL_PATH}")
    print(f"  ✓ scaler.pkl    → {SCALER_PATH}")
    print(f"  ✓ features.json → {FEATURES_PATH}")
    print(f"  ✓ threshold.json→ {THRESHOLD_PATH}")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE — Deploy backend/ with these 4 files")
    print("=" * 65)

    return model, scaler, active_features, best_tau


if __name__ == "__main__":
    model, scaler, features, tau = main()
