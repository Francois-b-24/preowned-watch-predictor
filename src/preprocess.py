from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))


def build_preprocessor() -> ColumnTransformer:
    material_cats = CFG["preprocess"]["material_categories"]
    return ColumnTransformer([
        ("brand", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["marque"]),
        ("model_te", Pipeline([("te", TargetEncoder(min_samples_leaf=50, smoothing=10))]), ["modele"]),
        ("seg", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), ["segment_prix"]),
        ("mat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="")),
            ("ord", OrdinalEncoder(categories=material_cats, handle_unknown="use_encoded_value", unknown_value=-1))
        ]), ["matiere_boitier", "matiere_bracelet"]),
        ("tech", Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("sc", StandardScaler())
        ]), ["reserve_de_marche", "diametre", "etencheite", "comptage_fonctions"])], remainder="drop")
