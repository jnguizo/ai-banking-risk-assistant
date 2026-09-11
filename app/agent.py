import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "La variable d'environnement GROQ_API_KEY "
        "n'est pas configurée."
    )

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# MODÈLE IA
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"


# ============================================================
# ANALYSE D'UNE TRANSACTION
# ============================================================

def analyser_transaction(transaction):
    """
    Génère une analyse IA à partir des résultats du modèle ML.

    Données disponibles :
    - Amount : montant de la transaction
    - risk_score : score produit par le modèle ML
    - prediction : classe prédite par le modèle

    L'IA ne doit utiliser aucune information
    qui n'est pas présente dans ces données.
    """

    amount = float(
        transaction.get("Amount", 0)
    )

    risk_score = float(
        transaction.get("risk_score", 0)
    )

    prediction = int(
        transaction.get("prediction", 0)
    )


    # --------------------------------------------------------
    # Conversion de la prédiction
    # --------------------------------------------------------

    prediction_label = (
        "FRAUDE POTENTIELLE"
        if prediction == 1
        else "NORMALE"
    )


    # --------------------------------------------------------
    # Niveau de risque indicatif
    # --------------------------------------------------------

    if risk_score < 0.20:

        risk_level = "FAIBLE"

    elif risk_score < 0.50:

        risk_level = "MODÉRÉ"

    else:

        risk_level = "ÉLEVÉ"


    # --------------------------------------------------------
    # Prompt sécurisé
    # --------------------------------------------------------

    prompt = f"""
Tu es un assistant d'analyse des risques spécialisé
dans la détection de fraude bancaire.

Tu dois analyser UNIQUEMENT les informations fournies
par le système de Machine Learning.

DONNÉES DISPONIBLES :

- Montant de la transaction : {amount:.2f} €
- Score de risque ML : {risk_score:.2%}
- Prédiction du modèle : {prediction_label}
- Niveau de risque indicatif : {risk_level}

RÈGLES STRICTES :

1. N'invente aucune information.
2. Tu ne connais pas l'identité du client.
3. Tu ne connais pas son historique bancaire.
4. Tu ne connais pas sa localisation.
5. Tu ne connais pas son âge ou son profil.
6. Tu ne connais pas son moyen de paiement.
7. Tu ne connais pas son comportement habituel.
8. Tu ne dois pas inventer une date ou une heure.
9. Tu ne dois pas inventer de seuil réglementaire ou bancaire.
10. Tu ne dois pas prétendre avoir accès à des données
    qui ne sont pas fournies.
11. Le score de risque est produit par le modèle ML.
12. La classification du modèle ne constitue pas à elle
    seule une décision bancaire définitive.
13. Ne présente pas le score comme une probabilité
    parfaitement calibrée.
14. Utilise l'expression "score de risque ML" lorsque
    cela est pertinent.
15. Ne crée aucune information supplémentaire.

Rédige le rapport exactement avec les quatre sections
suivantes :

## 1. Niveau de risque

Indique le niveau : FAIBLE, MODÉRÉ ou ÉLEVÉ.

Explique brièvement ce que signifie le score produit
par le modèle.

## 2. Analyse

Analyse uniquement :

- le montant ;
- le score de risque ML ;
- la prédiction du modèle.

Précise que les autres informations nécessaires à une
analyse complète ne sont pas disponibles.

## 3. Recommandations

Donne 2 recommandations génériques et prudentes
adaptées à un système de surveillance de fraude.

Ne propose pas de règle bancaire spécifique.

## 4. Décision du modèle

Indique :

- "CLASSÉE NORMALE" si la prédiction est 0 ;
- "FRAUDE POTENTIELLE" si la prédiction est 1.

Explique que cette classification correspond
uniquement à la sortie du modèle ML et ne constitue
pas une décision bancaire définitive.

N'utilise pas de tableau Markdown.
Réponds en français.
"""


    # --------------------------------------------------------
    # Appel Groq
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant spécialisé dans "
                        "l'analyse des risques bancaires. "
                        "Tu dois être factuel, prudent et "
                        "ne jamais halluciner de données."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=900
        )


        rapport = response.choices[0].message.content

        return rapport


    except Exception as e:

        return (
            "### Erreur lors de la génération du rapport IA\n\n"
            f"Une erreur est survenue : `{str(e)}`"
        )


# ============================================================
# RAPPORT GLOBAL
# ============================================================

def generer_rapport_global(stats):
    """
    Génère une synthèse IA des statistiques globales
    de l'échantillon et des performances du modèle.
    """

    total = int(
        stats.get("total", 0)
    )

    fraudes = int(
        stats.get("fraudes", 0)
    )

    taux_fraude = float(
        stats.get("taux_fraude", 0)
    )

    montant_risque = float(
        stats.get("montant_risque", 0)
    )

    auc_roc = float(
        stats.get("auc_roc", 0)
    )


    # --------------------------------------------------------
    # Prompt du rapport global
    # --------------------------------------------------------

    prompt = f"""
Tu es un assistant spécialisé dans l'analyse
des risques de fraude bancaire.

Tu dois produire une synthèse professionnelle
à partir des statistiques fournies.

STATISTIQUES DISPONIBLES :

- Nombre total de transactions : {total:,}
- Nombre de transactions labellisées fraude : {fraudes:,}
- Taux de fraude dans l'échantillon : {taux_fraude:.2%}
- Montant cumulé des transactions labellisées fraude :
  {montant_risque:.2f} €
- AUC-ROC du modèle : {auc_roc:.2%}

CONTEXTE IMPORTANT :

Les données correspondent à un échantillon de
démonstration.

Les transactions labellisées fraude représentent
les labels présents dans les données.

Ne dis pas que le modèle a détecté toutes ces fraudes.

Le montant cumulé correspond uniquement aux montants
associés aux transactions portant le label fraude.

Ne présente pas nécessairement ce montant comme une
perte financière réelle.

RÈGLES STRICTES :

1. N'invente aucune donnée.
2. N'invente aucune période temporelle.
3. N'invente aucune date.
4. N'invente aucune localisation.
5. N'invente aucun profil client.
6. N'invente aucun historique bancaire.
7. N'invente aucun seuil bancaire.
8. Ne prétends pas connaître les causes réelles
   de la fraude à partir de ces seules statistiques.
9. Ne prétends pas que l'AUC garantit les performances
   en production.
10. Précise que l'échantillon de démonstration peut avoir
    une distribution différente du dataset complet.
11. Utilise uniquement les statistiques fournies.

Structure exactement le rapport ainsi :

## Synthèse

Présente les principaux résultats observés.

## Analyse des risques

Analyse le nombre de transactions labellisées fraude,
le taux de fraude et le montant associé aux labels fraude.

## Performance du modèle

Présente l'AUC-ROC et explique brièvement ce qu'elle
indique sur la capacité discriminante du modèle.

## Points d'attention

Présente les principales limites liées à l'utilisation
d'un échantillon de démonstration et à l'interprétation
des résultats.

## Recommandations

Donne des recommandations générales pour améliorer
un système de détection de fraude en environnement
bancaire.

N'utilise pas de tableau Markdown.
Réponds en français.
"""


    # --------------------------------------------------------
    # Appel Groq
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant expert en "
                        "analyse des risques bancaires. "
                        "Tu dois produire des rapports "
                        "factuels et ne jamais inventer "
                        "d'informations."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=800
        )


        rapport = response.choices[0].message.content

        return rapport


    except Exception as e:

        return (
            "### Erreur lors de la génération du rapport global\n\n"
            f"Une erreur est survenue : `{str(e)}`"
        )


# ============================================================
# TEST LOCAL
# ============================================================

if __name__ == "__main__":

    transaction_test = {
        "Amount": 250.0,
        "risk_score": 0.0858,
        "prediction": 0
    }

    print(
        analyser_transaction(
            transaction_test
        )
    )