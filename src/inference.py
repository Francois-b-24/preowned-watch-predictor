from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / CFG["model"]["filename"]

_FEATURES = CFG["features"]["columns"]

_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add any missing feature columns as NA/empty defaults
    for col in _FEATURES:
        if col not in df.columns:
            df[col] = np.nan if col not in ("marque", "modele", "matiere_boitier", "matiere_bracelet") else ""
    return df[_FEATURES]


def predict_with_interval(df_feats: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    model = load_model()
    X = ensure_feature_columns(df_feats)
    y_log = model.predict(X)
    point = np.exp(y_log) - 1.0
    out = df_feats.copy()
    out["predicted_price"] = point
    # Try to load conformal q if present
    try:
        q = float(np.load(MODEL_PATH.with_name("conformal_q.npy")).ravel()[0])
        out["pi_low"] = np.maximum(0.0, point - q)
        out["pi_high"] = point + q
        out["pi_width"] = out["pi_high"] - out["pi_low"]
    except Exception:
        pass
    return out


def fit_conformal(calib_X: pd.DataFrame, calib_y_eur: pd.Series, alpha: float = 0.1) -> float:
    # calib_y_eur doit être en EUROS (pas en log)
    model = load_model()
    Xc = ensure_feature_columns(calib_X)
    y_log = model.predict(Xc)
    point = np.exp(y_log) - 1.0
    abs_res = np.abs(point - calib_y_eur.values)
    n = len(abs_res)
    if n == 0:
        return float("nan")
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = int(np.clip(idx, 0, n - 1))
    q = float(np.sort(abs_res)[idx])
    try:
        import numpy as _np
        _np.save(MODEL_PATH.with_name("conformal_q.npy"), _np.array([q], dtype=float))
    except Exception:
        pass
    return q
