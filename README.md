# AirParadis — Analyse de sentiment (MLOps)

## Objectif
Prédire si un tweet exprime un sentiment négatif afin d’anticiper un bad buzz.

## Livrables
- Préparation des données : `00_prepare_data.ipynb` → `data/processed/` (non versionné)
- Modèle classique : `01_modele_classique_tfidf.ipynb` (TF-IDF + LogisticRegression) → `artifacts/`
- Modèle sur mesure avancé : `02_modele_sur_mesure_avance_keras.ipynb` (Keras) → `exported_model/` + `artifacts/`
- API FastAPI : `api/main.py` (`/health`, `/predict`, `/feedback`)
- Tests unitaires : `tests/` (pytest)
- Interface : `interface/app_streamlit.py` (Streamlit)
- Tracking : `mlruns/` (MLflow local, non versionné)

## Installation
Activer l’environnement conda `sentiment`, puis :

```bash
pip install -r api/requirements.txt
pip install -r interface/requirements.txt
pip install pytest httpx