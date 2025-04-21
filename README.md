# luxury-watch-price-predictor

**Prédiction de prix de montres de luxe d'occasion**

## 🚀 Objectif
Ce projet fournit un pipeline complet de Machine Learning pour estimer le **prix affiché** de montres de luxe de seconde main, en s’appuyant sur les données de Chrono24. Il propose :
- Extraction, nettoyage et préparation des données
- Modélisation (régression linéaire, Ridge, HGB, tuning Optuna)
- Interprétabilité (SHAP)
- Validation croisée et learning curves
- Packaging du modèle final

## 📁 Structure du projet
```
├── data/
│   ├── raw/               # Données brutes collectées
│   └── processed/         # Données prêtes à l’usage (propre.xlsx)
├── models/                # Modèle entraîné (final_pipeline.joblib)
├── reports/               # Graphiques générés (SHAP, learning curves)
├── src/
│   ├── data_utils.py      # Fonctions de chargement et segmentation
│   ├── preprocess.py      # build_preprocessor()
│   ├── models.py          # Étapes 1 à 12 du pipeline
│   └── run_pipeline.py    # Script principal d’exécution
├── app/                   # (Optionnel) Micro‑service FastAPI
│   ├── main.py
│   └── Dockerfile
├── requirements.txt       # Dépendances Python
├── Dockerfile             # Containerisation du service
├── .gitignore
└── README.md
```

## 🛠️ Installation
1. **Cloner le dépôt**
   ```bash
   git clone https://github.com/<utilisateur>/luxury-watch-price-predictor.git
   cd luxury-watch-price-predictor
   ```

2. **Créer un environnement virtuel et installer**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Préparer les données**
   - Placer `propre.xlsx` dans `data/processed/`


## ▶️ Exécution du pipeline
Pour lancer toutes les étapes (1→12) et générer : personas, analyses, modèles, rapports et sauvegarde finale :
```bash
python src/run_pipeline.py
```

## 🔍 Résultats
- **Modèle final** enregistré dans `models/final_pipeline.joblib`
- **Graphiques** : `reports/shap_top10.png`, `reports/learning_curve.png`

## 📦 Déploiement API (FastAPI)
1. Positionner-vous à la racine du projet
2. Construire l’image Docker :
3. Lancer le conteneur :
4. Tester l’endpoint :
   

## 🤝 Contribuer
1. Forker le dépôt
2. Créer une branche `feature/ma-fonction`
3. Commit & Push
4. Ouvrir un Pull Request

## 📄 Licence
Ce projet est sous licence MIT. Voir [LICENSE](LICENSE) pour plus de détails.



