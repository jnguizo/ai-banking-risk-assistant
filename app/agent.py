import os
from groq import Groq
from dotenv import load_dotenv

# Charger la clé API
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyser_transaction(transaction: dict) -> str:
    """
    Analyse une transaction bancaire et génère un rapport de risque.
    """
    prompt = f"""
    Tu es un expert en détection de fraude bancaire.
    Analyse cette transaction et génère un rapport de risque détaillé.
    
    Transaction :
    - Montant : {transaction.get('Amount', 0):.2f} €
    - Score de risque ML : {transaction.get('risk_score', 0):.2%}
    - Prédiction : {'FRAUDE' if transaction.get('prediction') == 1 else 'NORMALE'}
    
    Génère un rapport structuré avec :
    1. Niveau de risque (Faible/Moyen/Élevé/Critique)
    2. Analyse de la transaction
    3. Recommandations
    4. Décision finale (Approuver/Bloquer/Investiguer)
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def generer_rapport_global(stats: dict) -> str:
    """
    Génère un rapport global d'analyse des risques.
    """
    prompt = f"""
    Tu es un expert en analyse de risque bancaire.
    Génère un rapport exécutif basé sur ces statistiques :
    
    - Total transactions analysées : {stats.get('total', 0)}
    - Transactions frauduleuses détectées : {stats.get('fraudes', 0)}
    - Taux de fraude : {stats.get('taux_fraude', 0):.2%}
    - Montant total à risque : {stats.get('montant_risque', 0):.2f} €
    - AUC-ROC du modèle : {stats.get('auc_roc', 0):.3f}
    
    Génère un rapport exécutif avec :
    1. Résumé exécutif
    2. Analyse des risques
    3. Points d'attention
    4. Recommandations stratégiques
    """
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test rapide
    transaction_test = {
        "Amount": 250.00,
        "risk_score": 0.85,
        "prediction": 1
    }
    print(analyser_transaction(transaction_test))