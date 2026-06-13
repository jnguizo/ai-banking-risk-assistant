# \# AI Banking Risk Assistant

# 

# Système intelligent d'analyse du risque client et de détection de fraude bancaire en temps réel, combinant Machine Learning et Intelligence Artificielle générative.

# 

# \## Objectif

# 

# Détecter automatiquement les transactions bancaires frauduleuses et générer des rapports décisionnels grâce à un agent IA, à partir de 284 807 transactions réelles.

# 

# \## Fonctionnalités

# 

# \-Détection de fraude en temps réel (XGBoost + SMOTE)

# \-Génération automatique de rapports IA (Groq LLM)

# \-Dashboard interactif (Streamlit)

# \- Analyse exploratoire complète (EDA)

# \- Score de risque par transaction

# 

# \## Structure du projet



\##Technologies utilisées



\- \*\*Python\*\* — Core

\- \*\*XGBoost + SMOTE\*\* — Détection de fraude

\- \*\*Groq (LLaMA 3.1)\*\* — Agent IA \& rapports

\- \*\*Streamlit\*\* — Dashboard interactif

\- \*\*Pandas, Scikit-learn\*\* — Data Science

\- \*\*Matplotlib, Seaborn\*\* — Visualisation



\## Résultats ML



| Métrique | Valeur |

|----------|--------|

| Dataset | 284 807 transactions |

| Taux de fraude | 0.17% |

| Modèle | XGBoost + SMOTE |

| AUC-ROC | \~0.95 |



\## Installation



```bash

git clone https://github.com/jnguizo/ai-banking-risk-assistant.git

cd ai-banking-risk-assistant

pip install -r requirements.txt

```



Créez un fichier `.env` :



Lancez l'application :

```bash

streamlit run app/dashboard.py

```







