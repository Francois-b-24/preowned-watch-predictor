from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import streamlit as st

from src.inference import predict_with_interval

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
FEATURES = CFG["features"]["columns"]
CHOICES = CFG.get("choices", {})
RANGES = CFG.get("ranges", {})

st.set_page_config(page_title="Watch Price Predictor", page_icon="⌚", layout="centered")
st.title("⌚ Prévision de prix de montres d'occasion")
st.caption("Modèle entraîné hors-ligne — choisissez des caractéristiques pour estimer le prix.")

mode = st.sidebar.radio("Mode", ["Formulaire", "Fichier (CSV/Excel)"])

if mode == "Fichier (CSV/Excel)":
    up = st.file_uploader("Déposez un CSV ou un Excel contenant les colonnes features", type=["csv", "xlsx"])
    if up is not None:
        try:
            df = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier: {e}")
            df = None
        if df is not None:
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.warning("Colonnes manquantes ajoutées par défaut : " + ", ".join(missing))
            preds = predict_with_interval(df, alpha=0.1)
            st.success(f"Prédictions effectuées sur {len(preds)} ligne(s).")
            st.dataframe(preds)
            st.download_button("Télécharger les prédictions (CSV)", data=preds.to_csv(index=False).encode("utf-8"), file_name="watch_predictions.csv", mime="text/csv")
else:
    with st.form("price_form"):
        col1, col2 = st.columns(2)

        def _build_range_list(key, default_min, default_max, default_step):
            r = RANGES.get(key, {"min": default_min, "max": default_max, "step": default_step})
            start = int(r.get("min", default_min)); stop = int(r.get("max", default_max)); step = int(r.get("step", default_step))
            return list(range(start, stop + step, step))

        with col1:
            marque = st.selectbox("Marque", CHOICES.get("marque", []))
            modele = st.selectbox("Modèle", CHOICES.get("modele", []))
            matiere_boitier = st.selectbox("Matière boîtier", CHOICES.get("matiere_boitier", []))
            matiere_bracelet = st.selectbox("Matière bracelet", CHOICES.get("matiere_bracelet", []))

        with col2:
            reserve_de_marche = st.select_slider("Réserve de marche (h)", options=_build_range_list("reserve_de_marche", 0, 120, 1), value=48)
            diametre = st.select_slider("Diamètre (mm)", options=_build_range_list("diametre", 30, 45, 1), value=40)
            etencheite = st.select_slider("Étanchéité (m)", options=_build_range_list("etencheite", 0, 300, 10), value=100)
            comptage_fonctions = st.select_slider("# Complications/fonctions", options=_build_range_list("comptage_fonctions", 0, 10, 1), value=1)

        submitted = st.form_submit_button("Estimer le prix")
    if submitted:
        row = {
            "marque": marque,
            "modele": modele,
            "matiere_boitier": matiere_boitier,
            "matiere_bracelet": matiere_bracelet,
            "reserve_de_marche": reserve_de_marche,
            "diametre": diametre,
            "etencheite": etencheite,
            "comptage_fonctions": comptage_fonctions,
        }
        pred = predict_with_interval(pd.DataFrame([row]), alpha=0.1)
        price = float(pred.loc[0, "predicted_price"])
        st.subheader("Prix estimé (intervalle 90%)")
        if "pi_low" in pred.columns:
            lo = float(pred.loc[0, "pi_low"]); hi = float(pred.loc[0, "pi_high"])
            st.metric("€", f"{price:,.0f}".replace(",", " "), help=f"Intervalle: {lo:,.0f} – {hi:,.0f} €".replace(",", " "))
        else:
            st.metric("€", f"{price:,.0f}".replace(",", " "))
        st.caption("Confiance empirique basée sur calibration split-conformal (alpha=0.1).")
