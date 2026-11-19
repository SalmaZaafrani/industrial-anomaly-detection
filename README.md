# 🔧 Bearing Fault Detection

Détection de défauts de roulements industriels à partir de signaux, avec trois classes : **Normal**, **Inner Race Fault** et **Outer Race Fault**.

Le projet compare plusieurs approches de deep learning :
- **CNN 1D** sur les signaux temporels
- **CNN 2D** sur des images GASF / GADF générées à partir des signaux
- **Réseau siamois** en few-shot learning, avec peu d'exemples par classe

## 📊 Dataset

[SUBF v2.0 — Bearing Faults Sound Data (Kaggle)](https://www.kaggle.com/datasets/sumairaziz/subf-v2-0-dataset-bearing-faults-sound-data)

Un petit échantillon est inclus dans `data/demo/` pour tester le projet sans télécharger le dataset.

## 🚀 Lancer le projet

```bash
pip install -r requirements.txt
jupyter notebook
```

Ouvrir `main.ipynb`.
