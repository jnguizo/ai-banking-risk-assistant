# 🏦 AI Banking Risk Assistant

> **Système intelligent d'analyse du risque et de détection de fraude bancaire, combinant Machine Learning et Intelligence Artificielle générative.**

[![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![Groq](https://img.shields.io/badge/LLM-Groq-black)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 Présentation

**AI Banking Risk Assistant** est une solution de Data Science appliquée au secteur bancaire permettant d'analyser le risque associé aux transactions et d'assister l'analyse de fraude.

Le système combine :

- un modèle de **Machine Learning XGBoost** pour la classification des transactions ;
- **SMOTE** pour traiter le déséquilibre important entre transactions normales et frauduleuses ;
- un **score de risque** associé à chaque transaction ;
- une interface interactive développée avec **Streamlit** ;
- un agent basé sur un **LLM via Groq** pour générer des analyses et recommandations à partir des résultats du modèle.

L'objectif est de rapprocher la modélisation prédictive d'un **cas d'usage bancaire concret**, avec une chaîne allant de l'analyse des données jusqu'à l'aide à la décision.

---

## 🎯 Objectifs

Le projet répond à plusieurs objectifs :

1. Identifier automatiquement les transactions présentant un risque de fraude.
2. Construire un modèle robuste face au déséquilibre des classes.
3. Produire un score de risque pour faciliter l'analyse des transactions.
4. Fournir des indicateurs de performance permettant d'évaluer le modèle.
5. Générer automatiquement une analyse textuelle à l'aide d'un LLM.
6. Proposer une interface interactive destinée à l'exploration et à l'analyse du risque.

---

## 🧠 Architecture fonctionnelle

```text
                    ┌──────────────────────┐
                    │   Données bancaires  │
                    │   Credit Card Fraud  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prétraitement        │
                    │ • Nettoyage          │
                    │ • Scaling             │
                    │ • Feature handling    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Gestion du déséquilibre│
                    │       SMOTE           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      XGBoost         │
                    │ Classification fraude│
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌─────────────────┐    ┌──────────────────┐
          │ Score de risque │    │ Décision modèle  │
          │ par transaction │    │ Fraude / Normale │
          └────────┬────────┘    └────────┬─────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Agent IA / LLM Groq  │
                    │ Analyse & rapport    │
                    └──────────────────────┘
```

---

## 🔍 Fonctionnalités

### 1. Détection de fraude

Classification des transactions à l'aide de **XGBoost** avec gestion du déséquilibre des classes via **SMOTE**.

### 2. Score de risque

Le modèle produit un score associé à la classe fraude permettant de catégoriser le niveau de risque de manière indicative :

- 🟢 **Faible**
- 🟠 **Modéré**
- 🔴 **Élevé**

> Les niveaux de risque présentés dans l'application sont indicatifs et ne constituent pas à eux seuls une décision bancaire définitive.

### 3. Analyse par Intelligence Artificielle générative

Un agent LLM exploite les résultats disponibles du modèle pour générer :

- une analyse du niveau de risque ;
- une interprétation des résultats ;
- des recommandations prudentes ;
- une synthèse décisionnelle.

L'agent est volontairement limité aux informations réellement disponibles afin de réduire les risques d'hallucination.

### 4. Dashboard interactif

L'application Streamlit permet notamment de visualiser :

- le volume de transactions ;
- les transactions frauduleuses labellisées ;
- le taux de fraude ;
- les montants associés aux transactions frauduleuses labellisées ;
- les métriques du modèle ;
- la matrice de confusion ;
- la courbe ROC ;
- l'importance des variables ;
- l'analyse d'une transaction individuelle.

---

## 📊 Données

Le projet s'appuie sur le dataset public **Credit Card Fraud Detection**, contenant :

- **284 807 transactions**
- **492 transactions frauduleuses**
- **284 315 transactions normales**
- un taux de fraude d'environ **0,17 %**

Les variables `V1` à `V28` sont anonymisées.

Le dataset complet n'est pas stocké dans ce dépôt en raison de sa taille.  
Un **échantillon de données** est fourni dans :

```text
data/creditcard_sample.csv
```

Cet échantillon est utilisé pour le fonctionnement et la démonstration du dashboard.

---

##  Modèle de Machine Learning

### Algorithme

**XGBoost Classifier**

Le modèle est entraîné avec une stratégie de traitement du déséquilibre des classes basée sur :

```text
Données
   ↓
Séparation Train / Test
   ↓
SMOTE sur l'ensemble d'entraînement
   ↓
XGBoost
   ↓
Évaluation
```

### Pourquoi SMOTE ?

La fraude bancaire constitue généralement une classe minoritaire. Un modèle entraîné directement sur des données fortement déséquilibrées peut privilégier la classe normale.

SMOTE permet de générer des observations synthétiques de la classe minoritaire sur l'ensemble d'entraînement afin d'améliorer la capacité du modèle à identifier les fraudes.

---

## 📈 Performances du modèle

Évaluation actuelle du modèle :

| Métrique | Valeur |
|---|---:|
| Accuracy | 98,73 % |
| Precision — Fraude | 94,68 % |
| Recall — Fraude | 90,82 % |
| F1-score — Fraude | 92,71 % |
| AUC-ROC | **98,78 %** |

### Visualisations

Les principaux résultats sont disponibles dans le dossier :

```text
reports/
├── confusion_matrix.png
├── roc_curve.png
└── feature_importance.png
```

---

## 🖥️ Application

L'application interactive est développée avec **Streamlit**.

Elle permet de saisir notamment :

- le montant de la transaction ;
- le temps relatif de la transaction.

Les autres variables du jeu de données sont conservées selon la configuration de démonstration du projet.

Le système retourne ensuite :

```text
Transaction
      ↓
Prétraitement
      ↓
XGBoost
      ↓
Score de risque
      ↓
Fraude / Normale
      ↓
Analyse générative
```

> **Important :** cette version constitue une application interactive de démonstration. Une architecture de production avec ingestion streaming, API temps réel, monitoring et déploiement distribué constituerait une évolution future du projet.

---

## 📁 Structure du projet

```text
ai-banking-risk-assistant/
│
├── app/
│   ├── agent.py
│   └── dashboard.py
│
├── data/
│   └── creditcard_sample.csv
│
├── models/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   ├── metrics.pkl
│   └── auc_roc.pkl
│
├── notebooks/
│   └── 02_ML_model.ipynb
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies utilisées

### Data Science & Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- imbalanced-learn / SMOTE
- Joblib

### Visualisation

- Matplotlib
- Seaborn

### Intelligence Artificielle générative

- Groq API
- LLM `openai/gpt-oss-20b`

### Application

- Streamlit

### Outils

- Jupyter Notebook
- Git
- GitHub
- VS Code

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone https://github.com/jnguizo/ai-banking-risk-assistant.git
cd ai-banking-risk-assistant
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer la clé API Groq

Créer un fichier `.env` à la racine du projet :

```text
GROQ_API_KEY=votre_cle_api
```

> Ne jamais publier le fichier `.env` sur GitHub.

### 4. Lancer l'application

```bash
python -m streamlit run app/dashboard.py
```

---

## 🔐 Sécurité

La clé API Groq est stockée dans une variable d'environnement :

```text
.env
```

Le fichier `.env` est exclu du dépôt Git grâce au `.gitignore`.

Aucune clé secrète ne doit être intégrée directement dans le code source.

---

## 🚀 Perspectives d'évolution

Le projet peut être étendu vers une architecture bancaire plus proche d'un environnement de production :

- API REST avec FastAPI ;
- ingestion de transactions en streaming ;
- Apache Kafka ;
- traitement temps réel avec Spark Structured Streaming ;
- base de données PostgreSQL ;
- monitoring du modèle ;
- détection de dérive des données ;
- réentraînement automatique ;
- gestion des alertes fraude ;
- authentification et contrôle d'accès ;
- déploiement Docker ;
- orchestration Kubernetes ;
- CI/CD ;
- architecture microservices.

---

## ⚠️ Limites actuelles

Cette version présente plusieurs limites :

- les variables `V1` à `V28` sont anonymisées ;
- le dashboard utilise un échantillon du dataset ;
- l'application actuelle est interactive et ne constitue pas encore une plateforme de streaming bancaire de production ;
- le score produit par le modèle ne doit pas être interprété comme une probabilité parfaitement calibrée ;
- l'analyse générative est basée uniquement sur les informations transmises à l'agent ;
- une validation sur des données bancaires réelles d'une institution serait nécessaire avant toute utilisation opérationnelle.

---

##  Contexte académique

Projet réalisé dans le cadre du **Master 2 Data Science & Intelligence Artificielle**.

### Sujet de mémoire

> **Conception et réalisation d'un système de détection de fraude à la carte bancaire en temps réel basé sur l'intelligence artificielle.**

Ce projet constitue une base expérimentale pour l'étude de la détection automatisée de fraude bancaire et de son évolution vers une architecture de traitement temps réel.

---

##  Auteur

**Jordy Nguizo**

Master 2 Data Science & Intelligence Artificielle  
Data Scientist | ML Engineer | AI Engineer

📍 Dakar, Sénégal

🔗 GitHub : https://github.com/jnguizo

---

##  Projet

Si ce projet vous intéresse, n'hésitez pas à consulter le dépôt, explorer le code et proposer des améliorations.