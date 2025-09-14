from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import pandas as pd
import yaml
import streamlit as st

from .inference import predict_prices, ensure_feature_columns

BASE_DIR = Path(__file__).resolve().parent.parent
CFG = yaml.safe_load(open(BASE_DIR / "config" / "params.yaml", "r", encoding="utf-8"))
FEATURES = CFG["features"]["columns"]

st.set_page_config(page_title="Watch Price Predictor", page_icon="⌚", layout="centered")
st.title("⌚ Prévision de prix de montres d'occasion")
st.caption("Modèle entraîné hors-ligne — fournissez des caractéristiques pour estimer le prix.")

mode = st.sidebar.radio("Mode", ["Formulaire", "Fichier (CSV/Excel)"])

SEG_OPTS = ["Entrée de gamme", "Moyen de gamme", "Haut de gamme"]

if mode == "Fichier (CSV/Excel)":
    up = st.file_uploader("Déposez un CSV ou un Excel contenant les colonnes features", type=["csv", "xlsx"])
    if up is not None:
        try:
            if up.name.lower().endswith(".csv"):
                df = pd.read_csv(up)
            else:
                df = pd.read_excel(up)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier: {e}")
            df = None
        if df is not None:
            missing = [c for c in FEATURES if c not in df.columns]
            if missing:
                st.warning("Colonnes manquantes ajoutées par défaut : " + ", ".join(missing))
            preds = predict_prices(df)
            st.success(f"Prédictions effectuées sur {len(preds)} ligne(s).")
            st.dataframe(preds)
            csv = preds.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les prédictions (CSV)", data=csv, file_name="watch_predictions.csv", mime="text/csv")
else:
    with st.form("price_form"):
        col1, col2 = st.columns(2)
        with col1:
            marque = st.text_input("Marque", "Rolex")
            modele = st.text_input("Modèle", "Submariner")
            segment_prix = st.selectbox("Segment de prix (estimé)", SEG_OPTS)
            matiere_boitier = st.text_input("Matière boîtier", "acier")
            matiere_bracelet = st.text_input("Matière bracelet", "acier")
        with col2:
            reserve_de_marche = st.number_input("Réserve de marche (h)", min_value=0.0, value=48.0, step=1.0)
            diametre = st.number_input("Diamètre (mm)", min_value=0.0, value=40.0, step=0.5)
            etencheite = st.number_input("Étanchéité (m)", min_value=0.0, value=100.0, step=10.0)
            comptage_fonctions = st.number_input("# Complications/fonctions", min_value=0, value=1, step=1)
        submitted = st.form_submit_button("Estimer le prix")
    if submitted:
        row = {
            "marque": marque,
            "modele": modele,
            "segment_prix": segment_prix,
            "matiere_boitier": matiere_boitier,
            "matiere_bracelet": matiere_bracelet,
            "reserve_de_marche": reserve_de_marche,
            "diametre": diametre,
            "etencheite": etencheite,
            "comptage_fonctions": comptage_fonctions,
        }
        df = pd.DataFrame([row])
        pred = predict_prices(df)
        price = float(pred.loc[0, "predicted_price"])
        st.subheader("Prix estimé")
        st.metric("€", f"{price:,.0f}".replace(",", " "))
        st.caption("Note : estimation ponctuelle. Ajouter un intervalle d'incertitude au besoin.")
