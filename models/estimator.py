import streamlit as st
st.set_page_config(
    page_title="✨ Luxury Watch Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "final_pipeline.joblib"

# Load trained pipeline (cached for performance)
@st.cache_resource
def load_model(path):
    return joblib.load(path)

pipeline = load_model(MODEL_PATH)


st.sidebar.header("🔧 Configuration")
st.sidebar.markdown(
    """
    **Segments de prix**
    - 🟢 Entrée de gamme: < 5 000 €
    - 🟡 Moyen de gamme: 5 000–20 000 €
    - 🔴 Haut de gamme: > 20 000 €

    **Instructions**
    1. Sélectionnez ou saisissez les caractéristiques.
    2. Cliquez sur **Estimer le prix**.
    """
)

# --- Main title & description
st.title("🔮 Prédiction de prix de montres de luxe d'occasion")
st.write(
    "Cette application est un outil interactif de prédiction de prix listé (Chrono24) pour montres de seconde main. "
    "Choisissez les attributs et découvrez une estimation instantanée !"
)

# --- Form inputs
with st.form(key='prediction_form'):
    marque = st.selectbox(
        "Marque", 
        options=sorted(pipeline.named_steps['pre'].named_transformers_['brand'].categories_[0])
    )
    modele = st.text_input("Modèle (ex: Submariner)")

    segment = st.radio(
        "Segment de prix",
        ('Entrée de gamme (<5 000 €)', 'Moyen de gamme (5 000–20 000 €)', 'Haut de gamme (>20 000 €)')
    )
    # normalize input back to labels
    segment_map = {
        'Entrée de gamme (<5 000 €)': 'Entrée de gamme',
        'Moyen de gamme (5 000–20 000 €)': 'Moyen de gamme',
        'Haut de gamme (>20 000 €)': 'Haut de gamme'
    }
    matieres = ['acier', 'or/acier', 'or massif', 'or rose', 'or blanc', 'palladium', 'titane', 'céramique']
    matiere_boitier = st.selectbox("Matière boîtier", options=matieres)
    matiere_bracelet = st.selectbox("Matière bracelet", options=matieres)

    reserve_marche = st.number_input(
        "Réserve de marche (heures)", 
        min_value=0, max_value=120, value=48, step=1, format="%d"
    )
    diametre = st.number_input(
        "Diamètre (mm)", 
        min_value=20, max_value=60, value=40, step=1, format="%d"
    )
    etancheite = st.number_input(
        "Étanchéité (m)", 
        min_value=0, max_value=2000, value=100, step=10, format="%d"
    )
    fonctions = st.number_input(
        "Nombre de fonctions", 
        min_value=1, max_value=20, value=1, step=1, format="%d"
    )

    submit = st.form_submit_button("🧮 Estimer le prix")

# --- Prediction logic
def predict_price():
    data = pd.DataFrame([{  
        'marque': marque,
        'modele': modele or 'Unknown',
        'segment_prix': segment_map[segment],
        'matiere_boitier': matiere_boitier,
        'matiere_bracelet': matiere_bracelet,
        'reserve_de_marche': reserve_marche,
        'diametre': diametre,
        'etencheite': etancheite,
        'comptage_fonctions': fonctions
    }])
    log_pred = pipeline.predict(data)[0]
    prix = float(np.expm1(log_pred))
    return prix

# --- Display result
if submit:
    prix_estime = predict_price()
    st.success(f"💰 Prix estimé : {prix_estime:,.2f} €")
    st.balloons()

# --- Footer
st.markdown("---")
st.write("© 2025 - Data Science Project | Déployé via Streamlit")
