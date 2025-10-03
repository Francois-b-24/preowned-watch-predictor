from __future__ import annotations
from pathlib import Path
import yaml

from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))


def build_feature_selector():
    fs_cfg = CFG.get("feature_selection", {})
    if not fs_cfg.get("enabled", False):
        return "passthrough"

    method = str(fs_cfg.get("method", "kbest")).lower()

    if method == "kbest":
        k = int(fs_cfg.get("k", 20))
        return SelectKBest(score_func=f_regression, k=k)

    if method == "lasso":
        alpha = float(fs_cfg.get("lasso_alpha", 0.001))
        return SelectFromModel(Lasso(alpha=alpha, max_iter=10000, random_state=CFG["training"]["random_state"]))

    if method == "rf_importance":
        n = int(fs_cfg.get("rf_n_estimators", 300))
        thresh = fs_cfg.get("threshold", "median")
        est = RandomForestRegressor(n_estimators=n, random_state=CFG["training"]["random_state"], n_jobs=-1)
        return SelectFromModel(estimator=est, threshold=thresh)

    return "passthrough"
