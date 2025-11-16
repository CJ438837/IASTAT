import streamlit as st

def app():
    st.title("🏠 Accueil")

    st.markdown("""
    **Bienvenue sur Corvus Analytics, votre application d'analyse statistique automatisée.**

    Cette application a été conçue pour **faciliter l’exploration, la compréhension et l’analyse de vos données**, qu’il s’agisse de fichiers CSV, Excel ou de DataFrames Pandas déjà chargés.  
    L’objectif principal est de fournir un environnement **intuitif, rapide et fiable** pour analyser vos données sans perdre de temps à configurer des scripts ou des calculs manuels.

    ---
    
    ## 🎯 Objectifs du projet

    - Détecter automatiquement le **type de chaque variable** (numérique, catégorielle, binaire).  
    - Fournir des **analyses descriptives détaillées**, avec tendances centrales, dispersion et mesures de forme pour les variables numériques, ainsi que comptages et fréquences pour les variables catégorielles.  
    - Réaliser des **tests bivariés** adaptés aux types de variables pour explorer les relations entre deux variables.  
    - Effectuer des **analyses multivariées** (PCA, MCA, FAMD, MANOVA, régressions multiples, corrélations) pour identifier des patterns et relations complexes dans vos données.  
    - Proposer une **analyse approfondie des distributions** pour détecter les distributions les plus probables et vérifier la normalité des variables.

    ---
    
    ## 🚀 Fonctionnalités principales

    1. **Importation facile** : CSV, Excel ou DataFrame Pandas.  
    2. **Détection automatique des types de variables** pour guider vos analyses.  
    3. **Analyse descriptive complète** avec graphiques et statistiques adaptées à chaque type de variable.  
    4. **Tests statistiques bivariés et multivariés** avec recommandations théoriques et graphiques.  
    5. **Diagnostic et visualisation des distributions** pour un aperçu clair des données.  
    6. **Interface intuitive et interactive**, avec navigation simple entre Accueil, Théorie et Analyse.

    ---
    
    ## 💡 Pourquoi utiliser Corvus Analytics ?

    - **Gagnez du temps** sur la préparation et l’analyse des données.  
    - **Minimisez les erreurs** grâce à l’automatisation des tests et calculs statistiques.  
    - **Comprenez mieux vos données** avant d’appliquer des modèles complexes ou de tirer des conclusions.  
    - **Formation et théorie intégrées** : chaque test et analyse est accompagné d’explications claires pour apprendre en pratiquant.

    ---
    
    **Commencez dès maintenant !** Cliquez sur le bouton ci-dessous pour accéder à vos données et lancer votre première analyse.

    """)

    # Bouton de redirection vers Analyse → Fichier
    if st.button("📈 Démarrer mon analyse"):
        st.session_state.main_page = "Analyse"
        st.session_state.analyse_subpage = "Fichier"

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")



