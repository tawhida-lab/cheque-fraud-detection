# 🏦 Détection de Chèques Falsifiés par Computer Vision

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-red)

> Projet de Computer Vision appliqué à la détection de fraude documentaire bancaire.  
> Modèle ResNet50 entraîné par Transfer Learning sur un dataset synthétique de chèques tunisiens.

---

## 📌 Résultats

| Métrique | Authentique | Falsifié | Global |
|----------|-------------|----------|--------|
| Precision | 97% | 95% | 96% |
| Recall | 95% | 97% | 96% |
| F1-Score | 96% | 96% | 96% |
| **Accuracy** | — | — | **96%** |

---

## 🎯 Objectif

Détecter automatiquement si un chèque bancaire est **authentique ou falsifié** à partir de son image, en identifiant visuellement les zones suspectes grâce à **Grad-CAM**.

### Types de falsifications détectées
- Montant altéré (effacement et réécriture)
- Signature remplacée
- Chèque photocopié
- Fond de sécurité retouché

---

## 🗂️ Structure du projet

```
cheque-fraud-detection/
├── cheque_dataset_generator.py  # Génération du dataset synthétique (Pillow + OpenCV)
├── cheque_training.py           # Entraînement ResNet50 + Grad-CAM (PyTorch)
├── app.py                       # Application Streamlit (démo interactive)
├── requirements.txt             # Dépendances Python
└── README.md
```

---

## 🔧 Pipeline technique

```
Images synthétiques → Prétraitement → ResNet50 (Transfer Learning) → Grad-CAM → Streamlit
```

### Étape 1 — Dataset synthétique
- **2 000 images** générées avec Pillow et OpenCV
- Répartition : 1 000 authentiques / 1 000 falsifiés
- Split : 70% train / 15% val / 15% test
- Aucune donnée bancaire réelle utilisée

### Étape 2 — Prétraitement
- Resize 224×224 (format ResNet50)
- Normalisation ImageNet (mean/std obligatoires pour Transfer Learning)
- Augmentation : rotation ±5°, ColorJitter, GaussianBlur

### Étape 3 — Modèle ResNet50
- Poids pré-entraînés ImageNet1K
- **Phase A** : entraînement fc layer seulement (5 epochs)
- **Phase B** : fine-tuning layer4 avec lr=1e-5 (10 epochs)
- Architecture fc : `Linear(2048→256) → ReLU → Dropout → Linear(256→2)`

### Étape 4 — Grad-CAM
- Hook sur `layer4[-1]` (dernière couche convolutive)
- Heatmap superposée sur l'image originale
- Rouge = zone très influente dans la décision

---

## 🚀 Lancer la démo en local

```bash
# Cloner le repo
git clone https://github.com/Linda-BA-data/cheque-fraud-detection.git
cd cheque-fraud-detection

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'app
streamlit run app.py
```

> Le fichier `best_model.pt` doit être placé à la racine du projet.  
> Téléchargeable depuis la section [Releases](https://github.com/Linda-BA-data/cheque-fraud-detection/releases).

---

## 🛠️ Technologies utilisées

| Outil | Usage |
|-------|-------|
| PyTorch | Entraînement du modèle |
| ResNet50 | Architecture CNN (Transfer Learning) |
| OpenCV | Génération dataset + Grad-CAM |
| Pillow | Génération des chèques synthétiques |
| Streamlit | Application web interactive |
| Grad-CAM | Explicabilité du modèle |
| Kaggle GPU | Entraînement (T4 GPU) |

---

## 💡 Cas d'usage réel

Ce projet répond à un besoin concret dans le secteur bancaire tunisien :
- Traitement automatisé des chèques à grande échelle
- Détection de fraude documentaire en temps réel
- Explicabilité de la décision (Grad-CAM) pour la conformité réglementaire

---

## 👩‍💻 Auteure

**Linda Trimeche**  
Data Scientist — Tunis, Tunisie  
[![GitHub](https://img.shields.io/badge/GitHub-tawhida--lab-black)](https://github.com/tawhida-lab)

---

## 📄 Licence

MIT License — libre d'utilisation avec attribution.
