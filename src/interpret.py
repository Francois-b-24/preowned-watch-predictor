from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import learning_curve

from .preprocess import build_preprocessor

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_personas() -> dict:
    return {
        "Revendeur pro": {
            "Objectif": "Estimer la valeur d'achat au plus juste (±5%)",
            "Usage": "Quotidien",
            "Sortie attendue": "Prix attendu + intervalle de confiance",
            "Canal": "API / Web",
        },
        "Particulier": {
            "Objectif": "Vérifier si une annonce est surpayée (±10%)",
            "Usage": "Ponctuel",
            "Sortie attendue": "Score de fair‑price + conseils",
            "Canal": "Mobile / Web",
        },
        "Plateforme": {
            "Objectif": "Détecter les annonces outliers (±20%)",
            "Usage": "Automatique",
            "Sortie attendue": "Flag outlier + explications",
            "Canal": "Dashboard",
        },
    }


def log_and_save_personas(personas: dict):
    print("=== Personas ===")
    for name, attrs in personas.items():
        print(f"{name}:")
        for k, v in attrs.items():
            print(f"  - {k}: {v}")
    md = ["# Personas"]
    for n, attrs in personas.items():
        md.append(f"{n}")
        for k, v in attrs.items():
            md.append(f"- **{k}**: {v}")
    (REPORTS_DIR / "personas.md").write_text("".join(md), encoding="utf-8")


def analyze_price_segments(df: pd.DataFrame):
    print("=== Segments de prix (counts) ===")
    counts = df["segment_prix"].value_counts()
    print(counts.to_string())

    print("=== Statistiques prix par segment ===")
    stats = df.groupby("segment_prix")["prix"].describe().round(2)
    print(stats.to_string())

    counts.to_csv(REPORTS_DIR / "segment_counts.csv", header=["count"])
    stats.to_csv(REPORTS_DIR / "segment_price_stats.csv")

    fig, ax = plt.subplots()
    df.boxplot(column="prix", by="segment_prix", ax=ax)
    ax.set_title("Distribution du prix par segment")
    ax.set_ylabel("Prix affiché")
    fig.suptitle("")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "segment_price_boxplot.png")
    plt.close(fig)


def shap_top_features(best_pipe, X_sample, topn: int = 10):
    model = best_pipe.named_steps.get("hgb")
    if model is None:
        print("[SHAP] Modèle non arborescent — skip")
        return
    pre = best_pipe.named_steps["pre"]
    Xs = pre.transform(X_sample)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(Xs)
    fn = pre.get_feature_names_out()
    imp = (
        pd.DataFrame({"feature": fn, "importance": np.abs(sv).mean(axis=0)})
        .sort_values("importance", ascending=False)
    )
    top = imp.head(topn)[::-1]
    fig, ax = plt.subplots()
    ax.barh(top["feature"], top["importance"])  # couleurs par défaut
    ax.set_xlabel("SHAP importance")
    ax.set_title(f"Top {topn} SHAP feature importances")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "shap_top10.png")
    plt.close(fig)


def learning_curve_plot(best_pipe, X, y, tag: str = "best"):
    sizes, tr_s, val_s = learning_curve(best_pipe, X, y, cv=5, train_sizes=np.linspace(0.1, 1, 5), scoring="r2", n_jobs=-1)
    import pandas as pd
    df_lc = pd.DataFrame({"train_size": sizes, "train_r2": tr_s.mean(axis=1), "val_r2": val_s.mean(axis=1)})
    fig, ax = plt.subplots()
    ax.plot(df_lc["train_size"], df_lc["train_r2"], marker="o", label="Train R2")
    ax.plot(df_lc["train_size"], df_lc["val_r2"], marker="o", label="Validation R2")
    ax.set_xlabel("Train set size")
    ax.set_ylabel("R2 score")
    ax.set_title(f"Learning Curves — {tag}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "learning_curve.png")
    plt.close(fig)