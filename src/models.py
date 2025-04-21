#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automatisation des étapes d'analyse et de modélisation pour la prédiction
des prix de montres de luxe d'occasion sur Chrono24.
"""

import warnings
warnings.filterwarnings("ignore", message=".*unknown categories.*", category=UserWarning)

import pandas as pd
import numpy as np
import joblib
import shap
import optuna
import os
from pathlib import Path

# Base directory for project (one level up from src)
BASE_DIR = Path(__file__).resolve().parent.parent
import matplotlib.pyplot as plt
import sys


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split, cross_validate, GridSearchCV, KFold,
    learning_curve
)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from category_encoders import TargetEncoder

sys.path.append(os.path.abspath("../src"))

# --- Constants
DATA_PATH = BASE_DIR / "data" / "processed" / "propre.xlsx"
MODEL_PATH = BASE_DIR / "models" / "final_pipeline.joblib"
# Ensure model directory exists
os.makedirs(MODEL_PATH.parent, exist_ok=True)
RANDOM_STATE = 42

# Reports directory for saving figures
REPORTS_PATH = "reports"
os.makedirs(REPORTS_PATH, exist_ok=True)

# Global Optuna study reference
STUDY = None

# --- Helpers
def segment_prix(prix: float) -> str:
    """Classement métier par prix."""
    if prix < 5000:
        return "Entrée de gamme"
    if prix <= 20000:
        return "Moyen de gamme"
    return "Haut de gamme"

def load_data() -> pd.DataFrame:
    """Charge, nettoie et prépare le DataFrame."""
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=['prix', 'marque', 'modele'])
    df['segment_prix'] = df['prix'].apply(segment_prix)
    df['log_prix']     = np.log(df['prix'] + 1)
    return df

def build_preprocessor() -> ColumnTransformer:
    """Construit le ColumnTransformer commun à la plupart des modèles."""
    material_cats = [['acier','or/acier','or massif'],
                     ['acier','or/acier','or massif']]
    return ColumnTransformer([
        ('brand',   OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['marque']),
        ('model_te', Pipeline([('te', TargetEncoder(min_samples_leaf=50, smoothing=10))]), ['modele']),
        ('seg',     OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), ['segment_prix']),
        ('mat', Pipeline([
            ('imp', SimpleImputer(strategy='constant', fill_value='')),
            ('ord', OrdinalEncoder(categories=material_cats,
                                   handle_unknown='use_encoded_value', unknown_value=-1))
        ]), ['matiere_boitier','matiere_bracelet']),
        ('tech', Pipeline([
            ('imp', SimpleImputer(strategy='mean')),
            ('sc',  StandardScaler())
        ]), ['reserve_de_marche','diametre','etencheite','comptage_fonctions'])
    ], remainder='drop')

# --- Étape 1: Personas
def etape1_personas():
    """1. Personas et périmètre."""
    personas = {
        "Revendeur pro": {
            "But": "Estimer valeur d'achat ±5%",
            "Format": "Valeur + intervalle",
            "Usage": "Quotidien", "Support": "API/Web"
        },
        "Particulier": {
            "But": "Vérifier si surpayé ±10%",
            "Sortie": "Score + conseils",
            "Usage": "Ponctuel", "Support": "Mobile/Web"
        },
        "Plateforme": {
            "But": "Détecter outliers ±20%",
            "Sortie": "Flag outlier",
            "Usage": "Automatique", "Support": "Dashboard"
        }
    }
    print("=== Personas ===")
    for k, v in personas.items():
        print(f"\n{k}:")
        for a, b in v.items():
            print(f"  - {a}: {b}")

# --- Étape 2: Analyse segment_prix
def etape2_segment_prix():
    """2. Analyse segment_prix."""
    df = load_data()
    print("\n=== Counts segment_prix ===")
    print(df['segment_prix'].value_counts().to_string())
    print("\n=== Stats par segment ===")
    print(df.groupby('segment_prix')['prix'].describe().to_string())

# --- Étape 3: Baseline linéaire
def etape3_baseline():
    """3. Baseline linéaire (marque + modèle)."""
    df = load_data()
    X = pd.get_dummies(df[['marque','modele']], drop_first=True)
    y = df['log_prix']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = LinearRegression().fit(Xtr, ytr)
    yp = model.predict(Xte)
    print(f"\n=== Étape 3 === R2={r2_score(yte, yp):.3f}, RMSE={np.sqrt(mean_squared_error(yte, yp)):.3f}, MAE={mean_absolute_error(yte, yp):.3f}")

# --- Étape 4: Ridge + target encoding
def etape4_ridge_te():
    """4. Ridge + target encoding sur 'modele'."""
    df = load_data()
    Xb = pd.get_dummies(df[['marque']], drop_first=True)
    te = TargetEncoder(min_samples_leaf=50, smoothing=10)
    df['enc_modele'] = te.fit_transform(df['modele'], df['log_prix'])
    X = pd.concat([Xb, df[['enc_modele']]], axis=1)
    Xtr, Xte, ytr, yte = train_test_split(X, df['log_prix'], test_size=0.2, random_state=RANDOM_STATE)
    model = Ridge(alpha=1.0).fit(Xtr, ytr)
    yp = model.predict(Xte)
    print(f"\n=== Étape 4 === R2={r2_score(yte, yp):.3f}, RMSE={np.sqrt(mean_squared_error(yte, yp)):.3f}, MAE={mean_absolute_error(yte, yp):.3f}")

# --- Étape 5: Ridge avec features étendues
def etape5_additional():
    """5. Ridge avec features étendues via préprocesseur."""
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    y = df['log_prix']
    pipe = Pipeline([('pre', build_preprocessor()), ('ridge', Ridge(alpha=1.0))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    print(f"\n=== Étape 5 === R2={r2_score(yte, yp):.3f}, RMSE={np.sqrt(mean_squared_error(yte, yp)):.3f}, MAE={mean_absolute_error(yte, yp):.3f}")

# --- Étape 6: Validation croisée 5-fold
def etape6_cv():
    """6. Validation croisée 5‑fold."""
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    y = df['log_prix']
    pipe = Pipeline([('pre', build_preprocessor()), ('ridge', Ridge(alpha=1.0))])
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sc = cross_validate(pipe, X, y, cv=cv, scoring=['r2','neg_root_mean_squared_error','neg_mean_absolute_error'], n_jobs=-1)
    print(f"\n=== Étape 6 === R2={sc['test_r2'].mean():.3f}±{sc['test_r2'].std():.3f}")

# --- Étape 7: Hyperparameter tuning Ridge
def etape7_tuning_ridge():
    """7. Tuning d'alpha pour Ridge."""
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    y = df['log_prix']
    pipe = Pipeline([('pre', build_preprocessor()), ('ridge', Ridge())])
    grid = GridSearchCV(pipe, {'ridge__alpha':[0.01, 0.1, 1, 10, 100]}, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    print(f"\n=== Étape 7 === Best α={grid.best_params_['ridge__alpha']}, R2={grid.best_score_:.3f}")

# --- Étape 8: Évaluation HistGradientBoostingRegressor
def etape8_tree():
    """8. HGB CV 5‑fold."""
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    y = df['log_prix']
    pipe = Pipeline([('pre', build_preprocessor()), ('hgb', HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sc = cross_validate(pipe, X, y, cv=cv, scoring=['r2','neg_root_mean_squared_error','neg_mean_absolute_error'], n_jobs=-1)
    print(f"\n=== Étape 8 === R2={sc['test_r2'].mean():.3f}±{sc['test_r2'].std():.3f}")

# --- Étape 9: Interprétabilité SHAP
def etape9_shap():
    """9. Interprétabilité SHAP."""
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    pipe = Pipeline([('pre', build_preprocessor()), ('hgb', HistGradientBoostingRegressor(random_state=RANDOM_STATE))])
    pipe.fit(X, df['log_prix'])
    expl = shap.TreeExplainer(pipe.named_steps['hgb'])
    Xs = pipe.named_steps['pre'].transform(X.sample(1000, random_state=RANDOM_STATE))
    sv = expl.shap_values(Xs)
    fn = pipe.named_steps['pre'].get_feature_names_out()
    imp = pd.DataFrame({'feature': fn, 'importance': np.abs(sv).mean(axis=0)}).sort_values('importance', ascending=False)
    print("\n=== Étape 9 === Top 10 features SHAP")
    print(imp.head(10).to_string(index=False))
    # Save top-10 SHAP importance as bar chart
    top10 = imp.head(10)[::-1]
    fig, ax = plt.subplots()
    ax.barh(top10['feature'], top10['importance'])
    ax.set_xlabel("SHAP importance")
    ax.set_title("Top 10 SHAP feature importances")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_PATH, "shap_top10.png"))
    plt.close(fig)

# --- Étape 10: Learning curves
def etape10_curves():
    """10. Learning & validation curves."""
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    y = df['log_prix']
    pipe = Pipeline([('pre', build_preprocessor()), ('hgb', HistGradientBoostingRegressor(random_state= RANDOM_STATE))])
    sizes, tr_s, val_s = learning_curve(pipe, X, y, cv=5, train_sizes=np.linspace(0.1,1,5), scoring='r2', n_jobs=-1)
    df_lc = pd.DataFrame({'train_size': sizes, 'train_r2': tr_s.mean(axis=1), 'val_r2': val_s.mean(axis=1)})
    print("\n=== Étape 10 === Learning Curve")
    print(df_lc.to_string(index=False))
    # Save learning curves plot
    fig, ax = plt.subplots()
    ax.plot(df_lc['train_size'], df_lc['train_r2'], marker='o', label='Train R2')
    ax.plot(df_lc['train_size'], df_lc['val_r2'], marker='o', label='Validation R2')
    ax.set_xlabel("Train set size")
    ax.set_ylabel("R2 score")
    ax.set_title("Learning Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_PATH, "learning_curve.png"))
    plt.close(fig)

# --- Étape 11: Optuna tuning
def etape11_optuna():
    """11. Optuna tuning pour HGB."""
    global STUDY
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    y = df['log_prix']
    pre = build_preprocessor()
    def objective(trial):
        params = {
            'max_iter': trial.suggest_int('max_iter', 100, 1000),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 255),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 20)
        }
        pipe = Pipeline([('pre', pre), ('hgb', HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params))])
        cv_results = cross_validate(pipe, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                                    scoring=['r2'], n_jobs=-1)
        return cv_results['test_r2'].mean()
    STUDY = optuna.create_study(direction='maximize')
    STUDY.optimize(objective, n_trials=30)
    print(f"\n=== Étape 11 === Best params={STUDY.best_params}, R2={STUDY.best_value:.3f}")

# --- Étape 12: Retrain et sauvegarde final
def etape12_retrain_and_save():
    """12. Retrain complet et sauvegarde du modèle."""
    df = load_data()
    X = df[['marque','modele','segment_prix','matiere_boitier','matiere_bracelet','reserve_de_marche','diametre','etencheite','comptage_fonctions']]
    y = df['log_prix']
    params = STUDY.best_params
    pipe = Pipeline([('pre', build_preprocessor()), ('hgb', HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params))])
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Modèle final enregistré → {MODEL_PATH}")

# --- Main execution
if __name__ == "__main__":
    etape1_personas()
    etape2_segment_prix()
    etape3_baseline()
    etape4_ridge_te()
    etape5_additional()
    etape6_cv()
    etape7_tuning_ridge()
    etape8_tree()
    etape9_shap()
    etape10_curves()
    etape11_optuna()
    etape12_retrain_and_save()