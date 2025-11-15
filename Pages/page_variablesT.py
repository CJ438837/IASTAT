import streamlit as st
from PIL import Image

def app():

    # --- Titre principal ---
    st.title("📊 Types de variables")
    
    # --- Introduction ---
    st.markdown("""
    La première étape de toute analyse statistique consiste à **identifier correctement le type de chaque variable**.  
    Cela permet de déterminer quelles méthodes statistiques sont adaptées : tests, visualisations, modèles, etc.
    """)

    # --- Illustration/logo section (optionnel) ---
    try:
        img = Image.open("assets/types_variables.png")
        st.image(img, use_column_width=True)
    except:
        pass

    st.markdown("---")

    # --- Pourquoi identifier les types de variables ---
    st.subheader("🔍 Pourquoi identifier les types de variables ?")
    st.markdown("""
    Le type d’une variable détermine :
    - **Quels tests statistiques sont autorisés**
    - **Quelles visualisations sont pertinentes**
    - **Comment nettoyer ou transformer les données**
    - **Comment interpréter les résultats**

    **Exemples :**
    - une variable *numérique* pourra être utilisée pour des tests paramétriques (t-test, ANOVA),
    - une variable *catégorielle* pour des tests du chi-deux,
    - une variable *binaire* pour de la régression logistique ou des comparaisons de proportions.
    """)

    st.markdown("---")

    # --- Les types détectés automatiquement ---
    st.subheader("🧩 Les trois types détectés automatiquement")

    # 1️⃣ Numérique
    st.markdown("### 1️⃣ Variable numérique")
    st.markdown("""
    Une variable est considérée comme **numérique** si ses valeurs sont des nombres.  
    **Exemples :**
    - taille
    - poids
    - âge
    - concentration d’un composé

    **Utilisations possibles :**
    - statistiques descriptives  
    - corrélations  
    - ANOVA  
    - régressions
    """)

    st.markdown("---")

    # 2️⃣ Catégorielle
    st.markdown("### 2️⃣ Variable catégorielle")
    st.markdown("""
    Une variable est dite **catégorielle** si ses valeurs correspondent à des groupes ou des labels.  
    **Exemples :**
    - type de traitement (`Placebo`, `Dose1`, `Dose2`)
    - couleur (`Rouge`, `Bleu`, `Vert`)
    - espèce (`Chat`, `Chien`, `Lapin`)

    ⚠️ Les catégories **ne représentent pas des quantités**, mais des classes.
    """)

    st.markdown("---")

    # 3️⃣ Binaire
    st.markdown("### 3️⃣ Variable binaire")
    st.markdown("""
    Une variable binaire possède **exactement 2 valeurs distinctes**.  
    **Exemples :**
    - `0 / 1`
    - `Oui / Non`
    - `Succès / Échec`
    - `Homme / Femme`

    **Utilisations fréquentes :**
    - comparaisons de proportions  
    - modèles logistiques  
    - tests exacts de Fisher
    """)

    st.markdown("---")

    # --- Formats pris en charge ---
    st.subheader("📁 Formats pris en charge")
    st.markdown("""
    Vous pouvez importer :
    - fichiers **CSV**
    - fichiers **Excel (.xls, .xlsx)**  
    - ou un **DataFrame Pandas déjà chargé**
    """)

    st.markdown("---")

    # --- Exemple pratique ---
    st.subheader("🎯 Exemple d’interprétation")
    st.markdown("""
    Si une variable contient :  

    - `['Homme', 'Femme']` → **binaire**  
    - `[12.5, 14.0, 15.8]` → **numérique**  
    - `['Chat', 'Chien', 'Lapin']` → **catégorielle**

    L’application affichera pour chaque variable :
    - son type détecté  
    - le nombre de valeurs uniques  
    - quelques exemples de valeurs
    """)

    st.markdown("---")

    # --- Bonnes pratiques ---
    st.subheader("🧠 Bonnes pratiques")
    st.markdown("""
    - Vérifiez toujours si les types détectés correspondent à votre logique métier  
    - Attention aux nombres codés en texte : `"10"` reste numérique pour l’analyse, mais peut nécessiter un nettoyage  
    - Une variable numérique avec très peu de valeurs uniques (ex. `0, 1, 2`) peut être recodée en catégorie si nécessaire
    """)

    st.markdown("---")

    # --- Footer ---
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
