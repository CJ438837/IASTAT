import streamlit as st

def app():
    # --- Titre principal ---
    st.title("📊 Analyse Multivariée")
    st.markdown("---")

    # --- Objectif ---
    st.subheader("🎯 Objectif de l’analyse multivariée")
    st.markdown("""
    L’analyse multivariée permet d’examiner **simultanément plusieurs variables** afin de détecter des structures, relations ou influences.  
    Selon le type de variables (numériques, catégorielles ou mixtes), différents outils statistiques et graphiques sont utilisés.

    Elle sert à :
    - Identifier des patterns ou clusters dans les données
    - Comprendre les relations entre variables
    - Préparer les analyses multivariées ou les modèles prédictifs
    """)

    st.markdown("---")

    # --- 1. PCA ---
    st.subheader("🔷 1. Analyse en Composantes Principales (PCA)")
    st.markdown("""
    - Destinée aux variables **numériques**  
    - Réduit la dimensionnalité tout en conservant la variance maximale  
    - Les axes principaux (PC1, PC2…) représentent des combinaisons linéaires des variables initiales  
    - Permet de visualiser clusters et tendances dans les données  
    - **Interprétation** : pourcentage de variance expliquée et loadings des variables
    """)

    st.markdown("---")

    # --- 2. MCA ---
    st.subheader("🔷 2. Analyse des Correspondances Multiples (MCA)")
    st.markdown("""
    - Destinée aux variables **catégorielles**  
    - Identifie les associations entre modalités  
    - Réduit la dimensionnalité pour visualiser les relations  
    - Utile pour explorer questionnaires ou tableaux de contingence complexes  
    - **Interprétation** : coordonnées des individus et des modalités sur les axes factoriels
    """)

    st.markdown("---")

    # --- 3. FAMD ---
    st.subheader("🔷 3. Analyse Factorielle Mixte (FAMD)")
    st.markdown("""
    - Destinée aux jeux de données **mixtes** (numériques + catégorielles)  
    - Combine PCA et MCA pour représenter toutes les variables sur un plan commun  
    - Permet d’identifier des groupes ou patterns globaux  
    - **Interprétation** : corrélation des variables numériques et contribution des modalités catégorielles
    """)

    st.markdown("---")

    # --- 4. MANOVA ---
    st.subheader("🔷 4. MANOVA (Analyse Multivariée de Variance)")
    st.markdown("""
    - Étend l’ANOVA à **plusieurs variables dépendantes simultanément**  
    - Vérifie si les groupes définis par les variables explicatives ont un effet global significatif  
    - Statistiques multivariées utilisées : Wilks’ Lambda, Pillai’s Trace, Hotelling-Lawley Trace  
    - **Interprétation** : p-value < 0.05 → effet global significatif des facteurs
    """)

    st.markdown("---")

    # --- 5. Régression multiple et diagnostic des résidus ---
    st.subheader("🔷 5. Régression multiple et diagnostic des résidus")
    st.markdown("""
    **Régression multiple :**
    - Modélise l’influence de **plusieurs variables explicatives numériques** sur une variable cible  
    - Fournit coefficients, intervalles de confiance et p-values  
    - Permet de prédire et d’évaluer l’effet relatif des variables

    **Analyse des résidus :**
    - Vérifie les hypothèses du modèle :  
        - Normalité des résidus (Shapiro-Wilk)  
    - Graphique résidus vs fitted pour détecter outliers et tendances  
    - Permet de valider la qualité de la régression
    """)

    st.markdown("---")

    # --- 6. Corrélations multiples ---
    st.subheader("🔷 6. Corrélations multiples")
    st.markdown("""
    - Étudie les relations entre toutes les variables numériques simultanément  
    - Matrice de corrélation visualisée par carte de chaleur (heatmap)  
    - Permet d’identifier des variables fortement liées ou redondantes  
    - **Interprétation** : coefficients proches de ±1 indiquent une forte corrélation
    """)

    st.markdown("---")

    # --- 7. Normalité multivariée (Mardia) ---
    st.subheader("🔷 7. Normalité multivariée (Mardia)")
    st.markdown("""
    - Vérifie si les variables numériques suivent une distribution normale multivariée  
    - Utilisé pour valider les hypothèses des analyses multivariées (PCA, MANOVA, régression multiple)  
    - Résultat : statistique HZ et p-value → normalité si p > 0.05
    """)

    st.markdown("---")

    # --- Bonnes pratiques ---
    st.subheader("💡 Bonnes pratiques")
    st.markdown("""
    - Vérifiez toujours le type des variables avant d’appliquer chaque méthode  
    - Utilisez les visualisations pour compléter l’interprétation statistique  
    - Pour PCA/MCA/FAMD, examinez la variance expliquée et la contribution des variables  
    - Pour MANOVA et régressions multiples, vérifiez les hypothèses et la significativité globale
    """)

    st.markdown("---")

    # --- Conclusion ---
    st.subheader("🎓 Conclusion")
    st.markdown("""
    L’analyse multivariée offre une **vision globale et intégrée** des données.  
    Elle combine :

    - **Réduction de dimension** : PCA, MCA, FAMD  
    - **Tests d’influence multivariés** : MANOVA  
    - **Modélisation et diagnostic** : régression multiple et résidus  
    - **Exploration des corrélations et normalité multivariée**  

    Cette page résume les concepts théoriques essentiels pour comprendre les résultats générés par le module d’analyse multivariée.
    """)

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
