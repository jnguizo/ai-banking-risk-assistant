import os
import sys
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

# Permet d'importer agent.py situé dans le même dossier
sys.path.append(os.path.dirname(__file__))

from agent import analyser_transaction, generer_rapport_global


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="AI Banking Risk Assistant",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITRE
# ============================================================

st.title("🏦 AI Banking Risk Assistant")

st.markdown(
    "**Détection de fraude bancaire interactive "
    "powered by Machine Learning & Generative AI**"
)

st.caption(
    "Système de démonstration basé sur un échantillon "
    "du dataset Credit Card Fraud Detection."
)

st.divider()


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def charger_modele():

    model = joblib.load(
        "models/fraud_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    # Features utilisées pendant l'entraînement
    features_path = "models/features.pkl"

    if os.path.exists(features_path):

        features = joblib.load(
            features_path
        )

    elif hasattr(model, "feature_names_in_"):

        features = list(
            model.feature_names_in_
        )

    else:

        features = None


    # Métriques du modèle
    metrics_path = "models/metrics.pkl"

    if os.path.exists(metrics_path):

        metrics = joblib.load(
            metrics_path
        )

    else:

        metrics = {}


    return model, scaler, features, metrics


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data
def charger_donnees():

    return pd.read_csv(
        "data/creditcard_sample.csv"
    )


# ============================================================
# CHARGEMENT
# ============================================================

model, scaler, features, metrics = charger_modele()

df = charger_donnees()


# ============================================================
# VÉRIFICATION
# ============================================================

colonnes_requises = [
    "Time",
    "Amount",
    "Class"
]

for colonne in colonnes_requises:

    if colonne not in df.columns:

        st.error(
            f"La colonne '{colonne}' est absente du dataset."
        )

        st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Informations du modèle")

    st.write(
        "**Algorithme**"
    )

    st.write(
        "XGBoost"
    )

    st.divider()

    st.write(
        "**Dataset**"
    )

    st.write(
        "Échantillon de démonstration"
    )

    st.write(
        f"**{len(df):,} transactions**"
    )

    st.divider()

    st.write(
        "**Performance**"
    )

    if metrics:

        st.metric(
            "AUC-ROC",
            f"{metrics.get('auc_roc', 0):.2%}"
        )

        st.metric(
            "Recall fraude",
            f"{metrics.get('recall_fraud', 0):.2%}"
        )

        st.metric(
            "Precision fraude",
            f"{metrics.get('precision_fraud', 0):.2%}"
        )

    st.divider()

    st.caption(
        "Les seuils d'interprétation du score sont "
        "indicatifs et ne constituent pas une politique "
        "bancaire réelle."
    )


# ============================================================
# KPIs DATASET
# ============================================================

st.subheader("📊 Vue Globale de l'Échantillon")

col1, col2, col3, col4 = st.columns(4)


total = len(df)

fraudes = int(
    df["Class"].sum()
)

taux = (
    fraudes / total * 100
    if total > 0
    else 0
)

fraud_amounts = df.loc[
    df["Class"] == 1,
    "Amount"
]

montant_moyen = (
    fraud_amounts.mean()
    if not fraud_amounts.empty
    else 0
)


col1.metric(
    "Transactions",
    f"{total:,}"
)

col2.metric(
    "Fraudes labellisées",
    f"{fraudes:,}"
)

col3.metric(
    "Taux de fraude",
    f"{taux:.2f}%"
)

col4.metric(
    "Montant moyen fraude",
    f"{montant_moyen:.2f} €"
)

st.caption(
    "⚠️ Les fraudes affichées correspondent aux labels "
    "présents dans l'échantillon, et non aux prédictions "
    "effectuées par le modèle."
)

st.divider()


# ============================================================
# PERFORMANCE DU MODÈLE
# ============================================================

st.subheader("🎯 Performance du modèle")

if metrics:

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Accuracy",
        f"{metrics.get('accuracy', 0):.2%}"
    )

    col2.metric(
        "Precision fraude",
        f"{metrics.get('precision_fraud', 0):.2%}"
    )

    col3.metric(
        "Recall fraude",
        f"{metrics.get('recall_fraud', 0):.2%}"
    )

    col4.metric(
        "F1-score fraude",
        f"{metrics.get('f1_fraud', 0):.2%}"
    )

    col5.metric(
        "AUC-ROC",
        f"{metrics.get('auc_roc', 0):.2%}"
    )

else:

    st.warning(
        "Les métriques du modèle ne sont pas disponibles."
    )


st.divider()


# ============================================================
# ANALYSE VISUELLE
# ============================================================

st.subheader("📈 Analyse Visuelle")

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Distribution des classes
# ------------------------------------------------------------

with col1:

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    distribution = (
        df["Class"]
        .value_counts()
        .sort_index()
    )

    labels = [
        "Normal",
        "Fraude"
    ]

    values = [
        distribution.get(0, 0),
        distribution.get(1, 0)
    ]

    ax.bar(
        labels,
        values
    )

    ax.set_title(
        "Distribution des transactions"
    )

    ax.set_xlabel(
        "Classe"
    )

    ax.set_ylabel(
        "Nombre de transactions"
    )

    st.pyplot(fig)

    plt.close(fig)


# ------------------------------------------------------------
# Distribution des montants
# ------------------------------------------------------------

with col2:

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    transactions_normales = df.loc[
        df["Class"] == 0,
        "Amount"
    ]

    transactions_fraudeuses = df.loc[
        df["Class"] == 1,
        "Amount"
    ]

    ax.hist(
        transactions_normales,
        bins=50,
        alpha=0.6,
        label="Normal"
    )

    ax.hist(
        transactions_fraudeuses,
        bins=50,
        alpha=0.6,
        label="Fraude"
    )

    ax.set_title(
        "Distribution des montants"
    )

    ax.set_xlabel(
        "Montant (€)"
    )

    ax.set_ylabel(
        "Nombre de transactions"
    )

    ax.legend()

    st.pyplot(fig)

    plt.close(fig)


st.divider()


# ============================================================
# RAPPORTS DE PERFORMANCE
# ============================================================

st.subheader(
    "📊 Rapports de performance"
)

col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Matrice de confusion
# ------------------------------------------------------------

with col1:

    confusion_path = (
        "reports/confusion_matrix.png"
    )

    if os.path.exists(confusion_path):

        st.image(
            confusion_path,
            caption="Matrice de confusion — XGBoost",
            use_container_width=True
        )

    else:

        st.info(
            "Matrice de confusion non disponible."
        )


# ------------------------------------------------------------
# Courbe ROC
# ------------------------------------------------------------

with col2:

    roc_path = (
        "reports/roc_curve.png"
    )

    if os.path.exists(roc_path):

        st.image(
            roc_path,
            caption="Courbe ROC — XGBoost",
            use_container_width=True
        )

    else:

        st.info(
            "Courbe ROC non disponible."
        )


st.divider()


# ============================================================
# ANALYSE D'UNE TRANSACTION
# ============================================================

st.subheader(
    "🔍 Analyser une Transaction"
)

st.info(
    "Pour cette démonstration, vous pouvez modifier "
    "le montant et le temps. Les autres variables "
    "sont conservées depuis une transaction de référence "
    "de l'échantillon."
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# Paramètres utilisateur
# ------------------------------------------------------------

with col1:

    montant = st.number_input(
        "💰 Montant de la transaction (€)",
        min_value=0.0,
        value=250.0,
        step=10.0
    )

    temps_reference = float(
        df["Time"].iloc[0]
    )

    temps = st.number_input(
        "⏱️ Temps de la transaction (secondes)",
        min_value=0.0,
        value=temps_reference,
        step=60.0
    )

    analyser = st.button(
        "🔎 Analyser la transaction",
        type="primary"
    )


# ============================================================
# ANALYSE ML
# ============================================================

if analyser:

    with st.spinner(
        "Analyse ML en cours..."
    ):

        try:

            # ------------------------------------------------
            # Transaction de référence
            # ------------------------------------------------

            sample = (
                df.drop(
                    columns=["Class"]
                )
                .iloc[0]
                .copy()
            )


            # ------------------------------------------------
            # Variables modifiées
            # ------------------------------------------------

            sample["Amount"] = montant

            sample["Time"] = temps


            # ------------------------------------------------
            # Vérification des features
            # ------------------------------------------------

            if features is None:

                st.error(
                    "Impossible de déterminer l'ordre "
                    "des variables du modèle."
                )

                st.stop()


            # ------------------------------------------------
            # DataFrame
            # ------------------------------------------------

            sample_df = pd.DataFrame(
                [sample]
            )


            # Respect de l'ordre d'entraînement
            sample_df = sample_df[
                features
            ]


            # ------------------------------------------------
            # Standardisation
            # ------------------------------------------------

            sample_scaled = sample_df.copy()

            sample_scaled[
                ["Amount", "Time"]
            ] = scaler.transform(
                sample_df[
                    ["Amount", "Time"]
                ]
            )


            # ------------------------------------------------
            # Prédiction
            # ------------------------------------------------

            prediction = model.predict(
                sample_scaled
            )[0]

            proba = model.predict_proba(
                sample_scaled
            )[0][1]


            # ------------------------------------------------
            # Niveau de risque indicatif
            # ------------------------------------------------

            if proba < 0.20:

                niveau_risque = "FAIBLE"

            elif proba < 0.50:

                niveau_risque = "MODÉRÉ"

            else:

                niveau_risque = "ÉLEVÉ"


            # ------------------------------------------------
            # Résultat
            # ------------------------------------------------

            with col2:

                st.subheader(
                    " Résultat du modèle"
                )

                if prediction == 1:

                    st.error(
                        "🚨 FRAUDE POTENTIELLE"
                    )

                else:

                    st.success(
                        "✅ TRANSACTION CLASSÉE NORMALE"
                    )


                st.metric(
                    "Score de risque",
                    f"{proba:.2%}"
                )


                st.metric(
                    "Niveau de risque",
                    niveau_risque
                )


                st.progress(
                    min(
                        max(
                            float(proba),
                            0.0
                        ),
                        1.0
                    )
                )


                st.caption(
    "Le score correspond au score de risque produit "
    "par le modèle pour la classe fraude."
)


            # ------------------------------------------------
            # Données pour l'IA générative
            # ------------------------------------------------

            transaction_data = {

                "Amount": float(
                    montant
                ),

                "risk_score": float(
                    proba
                ),

                "prediction": int(
                    prediction
                )

            }


            # ------------------------------------------------
            # Rapport IA
            # ------------------------------------------------

            st.divider()

            st.subheader(
                " Rapport IA"
            )

            rapport = analyser_transaction(
                transaction_data
            )

            st.markdown(
                rapport
            )


        except Exception as e:

            st.error(
                "Une erreur est survenue pendant "
                "l'analyse de la transaction."
            )

            st.exception(e)


st.divider()


# ============================================================
# RAPPORT GLOBAL IA
# ============================================================

st.subheader(
    " Rapport Global IA"
)

st.write(
    "L'IA générative produit une synthèse "
    "des risques observés dans l'échantillon."
)


if st.button(
    "📄 Générer Rapport Global",
    type="secondary"
):

    with st.spinner(
        "Génération du rapport global..."
    ):

        try:

            stats = {

                "total": total,

                "fraudes": int(
                    fraudes
                ),

                "taux_fraude": taux / 100,

                "montant_risque": float(
                    df.loc[
                        df["Class"] == 1,
                        "Amount"
                    ].sum()
                ),

                "auc_roc": float(
                    metrics.get(
                        "auc_roc",
                        0
                    )
                )

            }


            rapport_global = (
                generer_rapport_global(
                    stats
                )
            )


            st.markdown(
                rapport_global
            )


        except Exception as e:

            st.error(
                "Une erreur est survenue pendant "
                "la génération du rapport global."
            )

            st.exception(e)