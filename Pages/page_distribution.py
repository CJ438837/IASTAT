import streamlit as st
from PIL import Image

def app():

    st.title("📈 Analyse Avancée des Distributions — Théorie")

    st.markdown("""
    L’analyse des distributions vise à comprendre **la forme**, **la normalité** et **le comportement statistique** des variables
    numériques dans un jeu de données.  
    C’est une étape essentielle avant toute modélisation, car elle influence le choix des tests statistiques et des transformations éventuelles.
    """)

    st.markdown("---")
    st.header("🧪 1. Les tests de normalité")

    st.markdown("""
    Évaluer si une variable suit une distribution normale est fondamental pour décider si des méthodes 
    **paramétriques** ou **non paramétriques** doivent être utilisées.

    ### 🔹 **Test de Shapiro–Wilk**
    - Très adapté aux petits et moyens échantillons  
    - Hypothèse nulle : *la variable suit une distribution normale*  
    - Interprétation :  
      - **p > 0.05** → normalité plausible  
      - **p ≤ 0.05** → normalité rejetée

    ### 🔹 **Test de Kolmogorov–Smirnov**
    - Plus approprié pour les grands échantillons  
    - Compare les données à une distribution normale de référence  
    - Interprétation identique :  
      - **p > 0.05** → normalité plausible  
      - **p ≤ 0.05** → normalité rejetée

    Une variable normalement distribuée permettra l’utilisation de tests
    comme le t-test, l’ANOVA ou les corrélations de Pearson.
    """)

    st.markdown("---")
    st.header("📊 2. Identification de la distribution la plus probable")

    st.markdown("""
    Comprendre la distribution d’une variable permet d’interpréter correctement les phénomènes qu’elle représente.

    ### 🔹 Variables discrètes
    Les distributions les plus fréquentes sont :
    - **Poisson** : modélise des comptages (nombre d’événements).  
    - **Binomiale** : modélise un nombre de succès dans une série d’essais.

    ### 🔹 Variables continues
    Certaines distributions reviennent régulièrement :
    - **Normale** : symétrique, en cloche, très répandue en biologie et en physique.  
    - **Exponentielle** : décroissante, utilisée pour modéliser des durées d’attente ou des phénomènes de survie.  
    - **Log-normale** : asymétrique, fréquente lorsque les valeurs sont multipliées plutôt qu’additionnées.  
    - **Uniforme** : absence de structure, toutes les valeurs sont équiprobables.

    Identifier la bonne distribution permet :
    - d’appliquer des tests adaptés,
    - de comprendre l’origine d’une asymétrie,
    - d’anticiper les transformations nécessaires avant modélisation.
    """)

    st.markdown("---")
    st.header("📉 3. Visualisations essentielles")

    st.markdown("""
    Pour interpréter la distribution d’une variable, deux graphiques sont particulièrement importants :

    ### **1️⃣ Histogramme et courbe de densité**
    Ils permettent de visualiser :
    - la forme globale de la distribution,  
    - la symétrie ou asymétrie,  
    - les éventuelles valeurs extrêmes,  
    - l’homogénéité ou la dispersion des observations.

    ### **2️⃣ QQ-Plot (Quantile–Quantile Plot)**
    Cet outil compare les quantiles des données à ceux d’une distribution normale.  
    - Si les points suivent une diagonale → la variable est compatible avec une loi normale.  
    - Des écarts marqués traduisent une asymétrie ou une distribution différente.

    Ces représentations graphiques sont essentielles pour valider visuellement l’hypothèse de normalité.
    """)

    st.markdown("---")
    st.header("📋 4. Synthèse interprétative")

    st.markdown("""
    L’analyse d’une distribution permet de conclure sur :

    - **La normalité ou non-normalité** d’une variable  
    - **L’éventuelle transformation** à appliquer (log, standardisation…)  
    - **La famille de distributions la plus cohérente**  
    - **Le choix des futurs tests statistiques**  

    Cette étape constitue un socle indispensable pour toute analyse bivariée, multivariée ou modélisation prédictive.
    """)

    st.markdown("---")

    st.markdown("""
    Retrouvez l’application dédiée dans l’onglet :  
    👉 *Analyse → Distribution*  
    """)

    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
