import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Ajouter le dossier app au path
sys.path.append(os.path.dirname(__file__))
from agent import analyser_transaction, generer_rapport_global

# Configuration de la page
st.set_page_config(
    page_title="AI Banking Risk Assistant",
    page_icon="🏦",
    layout="wide"
)

# Titre principal
st.title("AI Banking Risk Assistant")
st.markdown("**Détection de fraude bancaire en temps réel powered by ML & IA**")
st.divider()

# Charger le modèle et les données
@st.cache_resource
def charger_modele():
    model = joblib.load('models/fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

@st.cache_data
def charger_donnees():
    df = pd.read_csv('data/creditcard_sample.csv')
    return df

model, scaler = charger_modele()
df = charger_donnees()

# KPIs

st.subheader("Vue Globale")
col1, col2, col3, col4 = st.columns(4)

total = len(df)
fraudes = df['Class'].sum()
taux = fraudes / total * 100
montant_moyen = df[df['Class']==1]['Amount'].mean()

col1.metric("Total Transactions", f"{total:,}")
col2.metric("Fraudes Détectées", f"{fraudes:,}")
col3.metric("Taux de Fraude", f"{taux:.2f}%")
col4.metric("Montant Moyen Fraude", f"{montant_moyen:.2f} €")

st.divider()

# Graphiques
st.subheader("Analyse Visuelle")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6,4))
    df['Class'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71','#e74c3c'])
    ax.set_xticklabels(['Normal', 'Fraude'], rotation=0)
    ax.set_title('Distribution des transactions')
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df[df['Class']==0]['Amount'], bins=50, alpha=0.6, color='#2ecc71', label='Normal')
    ax.hist(df[df['Class']==1]['Amount'], bins=50, alpha=0.6, color='#e74c3c', label='Fraude')
    ax.set_title('Distribution des montants')
    ax.legend()
    st.pyplot(fig)

st.divider()

# Analyse d'une transaction

st.subheader("Analyser une Transaction")

col1, col2 = st.columns(2)

with col1:
    montant = st.number_input("Montant de la transaction (€)", min_value=0.0, value=250.0)
    analyser = st.button("Analyser", type="primary")

if analyser:
    with st.spinner("Analyse en cours..."):
        # Préparer les features
        sample = df.drop(columns=['Class']).iloc[0].copy()
        sample['Amount'] = montant

        # Normaliser
        sample_scaled = sample.copy()
        sample_scaled['Amount'] = scaler.transform([[montant]])[0][0]

        # Prédiction
        features = sample_scaled.values.reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        # Résultat
        with col2:
            if prediction == 1:
                st.error(f"FRAUDE DÉTECTÉE — Score : {proba:.2%}")
            else:
                st.success(f"TRANSACTION NORMALE — Score : {proba:.2%}")
            st.progress(float(proba))

        # Rapport IA
        st.subheader("Rapport IA")
        transaction_data = {
            "Amount": montant,
            "risk_score": proba,
            "prediction": int(prediction)
        }
        rapport = analyser_transaction(transaction_data)
        st.markdown(rapport)

st.divider()

# Rapport Global IA

st.subheader(" Rapport Global IA")

if st.button("Générer Rapport Global", type="secondary"):
    with st.spinner("Génération du rapport en cours..."):
        stats = {
            "total": total,
            "fraudes": int(fraudes),
            "taux_fraude": taux/100,
            "montant_risque": df[df['Class']==1]['Amount'].sum(),
            "auc_roc": 0.95
        }
        rapport_global = generer_rapport_global(stats)
        st.markdown(rapport_global)