from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_personas() -> dict:
    return {
        "Revendeur pro": {"Objectif": "Estimer la valeur d'achat au plus juste (±5%)"},
        "Particulier": {"Objectif": "Vérifier si une annonce est surpayée (±10%)"},
        "Plateforme": {"Objectif": "Détecter les annonces outliers (±20%)"},
    }


def log_and_save_personas(personas: dict):
    md = ["# Personas\n"]
    for n, attrs in personas.items():
        md.append(f"\n## {n}\n")
        for k, v in attrs.items():
            md.append(f"- **{k}**: {v}")
    (REPORTS_DIR / "personas.md").write_text("\n".join(md), encoding="utf-8")


def analyze_price_segments(df: pd.DataFrame):
    counts = df["segment_prix"].value_counts()
    stats = df.groupby("segment_prix")["prix"].describe().round(2)
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
