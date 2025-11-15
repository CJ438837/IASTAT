import streamlit as st

def app():
    st.title("📊 Analyse des Distributions")
    st.markdown("---")

    st.subheader("🎯 Objectif")
    st.markdown("""
    L’analyse des distributions permet de comprendre **la forme et le comportement des variables numériques** avant toute analyse statistique avancée.  
    Elle aide à :
    - Vérifier l’hypothèse de normalité pour appliquer les tests paramétriques
    - Identifier des distributions sous-jacentes pour modélisation ou simulation
    - Détecter des valeurs extrêmes ou anomalies
    """)

    st.markdown("---")
    st.header("🔹 1. Tests de normalité")
    st.markdown("""
    Plusieurs tests permettent de vérifier si une variable suit une **distribution normale** :

    - **Shapiro-Wilk** : recommandé pour des échantillons de petite taille (< 5000 observations)  
      - H0 : les données suivent une distribution normale  
      - H1 : les données ne sont pas normales  
      - p-value > 0.05 → normalité acceptée
          
    - **Kolmogorov-Smirnov (KS)** : utilisé pour des échantillons plus grands  
      - Compare la distribution empirique avec une distribution théorique (ex. normale)  
      - Même interprétation pour la p-value

    **Verdict** : Normal / Non Normal
    """)

    st.markdown("---")
    st.header("🔹 2. Détection de la distribution probable")
    st.markdown("""
    Une fois la normalité évaluée, il est utile de proposer la **distribution statistique la plus probable** :

    - **Variables discrètes** : Poisson, Binomiale  
    - **Variables continues** : Normale, Exponentielle, Log-normale, Uniforme  

    Le choix de la distribution permet de :
    - Adapter les modèles statistiques et simulations
    - Générer des données synthétiques réalistes
    - Comprendre la variabilité et la forme des données

    ⚡ **Outils utilisés** : bibliothèques de fit automatique comme `Fitter` en Python.
    """)

    st.markdown("---")
    st.header("🔹 3. Visualisations")
    st.markdown("""
    Pour chaque variable numérique, l’application produit deux types de graphiques :

    1. **Histogramme + KDE (Kernel Density Estimate)**  
       - Histogramme : distribution empirique des données  
       - KDE : estimation de la densité de probabilité continue  
       - Permet de visualiser asymétrie, pics et étendue des valeurs

    2. **QQ-plot (Quantile-Quantile plot)**  
       - Compare les quantiles des données avec ceux d’une distribution normale  
       - Les points proches de la diagonale → normalité approximative  
       - Détecte les écarts et valeurs extrêmes

    Ces visualisations aident à **valider visuellement la normalité** et à détecter des anomalies.
    """)

    st.markdown("---")
    st.subheader("💡 Bonnes pratiques")
    st.markdown("""
    - Toujours examiner à la fois les **tests statistiques** et les **visualisations**  
    - Pour des grands échantillons (>5000), privilégier KS plutôt que Shapiro  
    - Pour des variables discrètes, adapter les distributions testées (Poisson, Binomiale)  
    - Vérifier les valeurs aberrantes qui peuvent biaiser les tests et les fit
    """)

    st.markdown("---")
    st.subheader("🎓 Conclusion")
    st.markdown("""
    L’analyse avancée des distributions permet de **diagnostiquer la forme des données numériques**, de détecter des anomalies et de proposer la distribution statistique la plus probable.  
    Ces informations sont essentielles pour :
    - choisir les tests statistiques appropriés  
    - préparer des modèles de simulation ou prédiction  
    - interpréter correctement les résultats des analyses
    """)

    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
