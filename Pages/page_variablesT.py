import streamlit as st
from PIL import Image

def app():

    st.title("📊 Types de variables")

    st.markdown("""
    La première étape de toute analyse statistique consiste à **identifier correctement le type de chaque variable**.  
    Cela permet de déterminer quelles méthodes statistiques sont adaptées : tests, visualisations, modèles, etc.

    Votre application détecte automatiquement les variables **numériques**, **catégorielles** et **binaires** 
    à partir d’un fichier CSV ou Excel, ou directement depuis un DataFrame.

    ---

    ## 🔍 Pourquoi identifier les types de variables ?
    Le type d’une variable détermine :
    - **Quels tests statistiques sont autorisés**
    - **Quelles visualisations sont pertinentes**
    - **Comment nettoyer ou transformer les données**
    - **Comment interpréter les résultats**

    Par exemple :
    - une variable *numérique* pourra être utilisée pour des tests paramétriques (t-test, ANOVA),
    - une variable *catégorielle* pour des tests du chi-deux,
    - une variable *binaire* pour de la régression logistique ou des comparaisons de proportions.

    ---

    ## 🧩 Les trois types détectés automatiquement

    ### 1️⃣ Variable numérique
    Une variable est considérée comme **numérique** si ses valeurs sont des nombres.  
    Exemples :
    - taille
    - poids
    - âge
    - concentration d’un composé

    👉 **Utilisations possibles :** statistiques descriptives, corrélations, ANOVA, régressions.

    ---

    ### 2️⃣ Variable catégorielle
    Une variable est dite **catégorielle** si ses valeurs correspondent à des groupes ou des labels.  
    Exemples :
    - type de traitement (`Placebo`, `Dose1`, `Dose2`)
    - couleur (`Rouge`, `Bleu`, `Vert`)
    - espèce (`Chat`, `Chien`, `Lapin`)

    👉 **Les catégories ne représentent pas des quantités**, mais des classes.

    ---

    ### 3️⃣ Variable binaire
    Une variable binaire possède **exactement 2 valeurs distinctes**.  
    Exemples :
    - `0 / 1`
    - `Oui / Non`
    - `Succès / Échec`
    - `Homme / Femme`

    👉 Votre application les détecte automatiquement dès qu’il y a **2 valeurs uniques**, quel que soit leur format.

    Ces variables sont souvent utilisées pour :
    - comparaisons de proportions  
    - modèles logistiques  
    - tests exacts de Fisher  

    ---

    ## 📁 Formats pris en charge
    Vous pouvez importer :
    - fichiers **CSV**
    - fichiers **Excel (.xls, .xlsx)**  
    - ou un DataFrame Pandas déjà chargé

    ---

    ## 🎯 Exemple d’interprétation
    Si une variable contient :

    - `['Homme', 'Femme']` → **binaire**
    - `[12.5, 14.0, 15.8]` → **numérique**
    - `['Chat', 'Chien', 'Lapin']` → **catégorielle**

    Votre application affichera pour chaque variable :
    - son type détecté  
    - le nombre de valeurs uniques  
    - quelques exemples de valeurs  

    ---

    ## 🧠 Bonnes pratiques
    - Vérifiez toujours si les types détectés correspondent à votre logique métier  
    - Attention aux nombres codés en texte : `"10"` reste numérique pour l’analyse, mais peut nécessiter un nettoyage  
    - Une variable numérique avec très peu de valeurs uniques (ex. `0, 1, 2`) peut être recodée en catégorie si nécessaire

    ---

    © 2025 Corvus Analytics - Tous droits réservés
    """)

