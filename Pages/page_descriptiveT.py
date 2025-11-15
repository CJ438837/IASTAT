import streamlit as st
from PIL import Image

def app():

    # --- Titre principal ---
    st.title("📊 Analyse Descriptive — Théorie")

    # --- Introduction ---
    st.markdown("""
    L’analyse descriptive constitue la première étape essentielle de toute étude statistique.  
    Son objectif est de **résumer, structurer et comprendre les données** avant d’entreprendre des analyses plus complexes.

    Cette section décrit les principes théoriques derrière le fonctionnement du module d'analyse descriptive de l’application.
    """)

    # --- Illustration/logo section (optionnel) ---
    try:
        img = Image.open("assets/analyse_descriptive.png")
        st.image(img, use_column_width=True)
    except:
        pass

    st.markdown("---")

    # --- Variables numériques ---
    st.subheader("🔢 1. Variables numériques")
    st.markdown("""
    Pour les variables numériques, plusieurs statistiques sont automatiquement calculées :

    ### Statistiques de tendance centrale
    - **Moyenne** : mesure la valeur centrale moyenne.
    - **Médiane** : valeur centrale robuste aux valeurs extrêmes.

    ### Statistiques de dispersion
    - **Min / Max** : étendue des valeurs.
    - **Écart-type (std)** : variabilité autour de la moyenne.
    - **Variance** : carré de l’écart-type.
    - **Quartiles (Q1, Q2, Q3)** : répartition de la distribution.
    - **Coefficient de variation (CV)** : `std / moyenne` — utile pour comparer des variables de natures différentes.

    ### Mesures de forme
    - **Asymétrie (Skewness)** : indique si la distribution est inclinée à gauche ou à droite.
    - **Kurtosis (Aplatissement)** : indique si la distribution est plus ou moins concentrée que la normale.

    Ces mesures permettent de **diagnostiquer la distribution**, notamment :
    - la présence de valeurs extrêmes,
    - la symétrie ou non des données,
    - la régularité ou dispersion d’une variable.
    """)

    st.markdown("---")

    # --- Variables catégorielles ou binaires ---
    st.subheader("🧩 2. Variables catégorielles ou binaires")
    st.markdown("""
    Pour les variables catégorielles ou binaires, l’analyse descriptive repose sur les **comptages** et **fréquences**.

    ### Statistiques calculées
    - **Effectifs de chaque modalité**
    - **Fréquences relatives (%)**
    - **Détection des modalités rares** :  
      Une modalité est considérée **rare** si elle apparaît dans **moins de 5%** des observations.

    L’identification des modalités rares est importante car :
    - elles peuvent biaiser certains tests statistiques,
    - elles peuvent indiquer un regroupement nécessaire,
    - elles impactent la stabilité des modèles prédictifs.
    """)

    st.markdown("---")

    # --- Traitement automatique selon le type de variable ---
    st.subheader("🧪 3. Traitement automatique selon le type de variable")
    st.markdown("""
    L’application détecte automatiquement le type de chaque variable et applique les règles suivantes :

    - **Numérique → calcul complet des statistiques**  
    - **Catégorielle / Binaire → comptages, pourcentages et détection des modalités rares**  
    - **Autre → indication qu’aucune analyse standard n’est disponible**

    Cette automatisation permet d’obtenir rapidement :
    - un **résumé clair des données**,  
    - une **vue d’ensemble fiable** avant de poursuivre vers des tests statistiques.
    """)

    st.markdown("---")

    # --- Exemple pratique / conseil ---
    st.subheader("💡 Conseils pratiques")
    st.markdown("""
    - Vérifiez toujours si les résultats calculés correspondent à votre logique métier  
    - Attention aux variables numériques codées en texte (ex. `"10"`)  
    - Les variables numériques avec peu de valeurs uniques peuvent être recodées en catégories si nécessaire  
    - Utilisez ces analyses descriptives pour détecter anomalies et tendances avant tests statistiques ou modélisation
    """)

    st.markdown("---")

    # --- Footer ---
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
