# =============================================================================
# backend/model.py — Model Loading and Inference
# =============================================================================

import os
import json
import logging
import numpy as np
import joblib
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default artifact paths — override via env vars for flexibility
BASE_DIR       = os.environ.get("MODEL_DIR", os.path.dirname(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH    = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH  = os.path.join(BASE_DIR, "features.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")


class TradingModel:
    """
    Wraps XGBoost model + StandardScaler + feature registry + threshold.
    Thread-safe for concurrent requests (read-only inference).
    """

    def __init__(self):
        self.model     = None
        self.scaler    = None
        self.features  = []
        self.threshold = 0.5
        self.metadata  = {}
        self.loaded    = False

    def load(self):
        """Load all artifacts from disk. Call once at startup."""
        logger.info("Loading model artifacts...")

        # Model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"model.pkl not found at {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        logger.info("  ✓ XGBoost model loaded")

        # Scaler
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"scaler.pkl not found at {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)
        logger.info("  ✓ StandardScaler loaded")

        # Features
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"features.json not found at {FEATURES_PATH}")
        with open(FEATURES_PATH) as f:
            feat_data = json.load(f)
        self.features = feat_data["features"]
        logger.info(f"  ✓ {len(self.features)} features loaded")

        # Threshold
        if not os.path.exists(THRESHOLD_PATH):
            raise FileNotFoundError(f"threshold.json not found at {THRESHOLD_PATH}")
        with open(THRESHOLD_PATH) as f:
            thresh_data = json.load(f)
        self.threshold = thresh_data["threshold"]
        self.metadata  = thresh_data
        logger.info(f"  ✓ Threshold τ = {self.threshold:.4f} loaded")

        self.loaded = True
        return self

    def predict(self, feature_vector: Dict[str, float]) -> Dict[str, Any]:
        """
        Accepts a dict mapping feature_name → float value.
        Returns prediction dict with probability, signal, and raw values.
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Build ordered feature array
        x = np.array(
            [feature_vector.get(f, 0.0) for f in self.features],
            dtype=np.float32
        ).reshape(1, -1)

        # Check for all-zero / NaN (model not ready)
        if np.isnan(x).any():
            return {
                "probability": None,
                "signal": 0,
                "threshold": self.threshold,
                "ready": False,
                "error": "Feature vector contains NaN"
            }

        # Scale
        x_scaled = self.scaler.transform(x)

        # Predict
        prob = float(self.model.predict_proba(x_scaled)[0, 1])
        signal = 1 if prob > self.threshold else 0

        return {
            "probability": round(prob, 6),
            "signal": signal,
            "threshold": self.threshold,
            "ready": True,
            "error": None
        }

    def feature_names(self):
        return list(self.features)

    def info(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "n_features": len(self.features),
            "features": self.features,
            "threshold": self.threshold,
            "val_sharpe": self.metadata.get("val_sharpe"),
            "val_metrics": self.metadata.get("val_metrics"),
            "test_metrics": self.metadata.get("test_metrics"),
            "tickers": self.metadata.get("tickers"),
            "trained_at": self.metadata.get("trained_at"),
        }


# Singleton
_model_instance: Optional[TradingModel] = None


def get_model() -> TradingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = TradingModel()
    if not _model_instance.loaded:
        try:
            _model_instance.load()
        except Exception as e:
            logger.warning(f"Model load failed: {e}. Running in demo mode.")
    return _model_instance
