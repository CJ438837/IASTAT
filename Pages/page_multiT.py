
import streamlit as st

# ---------------------------------------------
# PAGE : Tests multivariés — Théorie & Interprétation
# ---------------------------------------------

st.title("📊 Analyse Multivariée — Théorie & Interprétation")
st.write("---")

st.header("🎯 Objectif de l’analyse multivariée")
st.write("""
L’analyse multivariée permet d’examiner **simultanément plusieurs variables** afin de détecter des structures, relations ou influences.  
Selon le type de variables (numériques, catégorielles ou mixtes), différents outils statistiques et graphiques sont utilisés.
""")

st.write("---")
st.header("🔷 1. Analyse en Composantes Principales (PCA)")

with st.expander("📌 Théorie PCA"):
    st.write("""
    - Destinée aux variables **numériques**  
    - Réduit la dimensionnalité tout en conservant la variance maximale  
    - Les axes principaux (PC1, PC2…) représentent les combinaisons linéaires des variables initiales  
    - Permet de visualiser les clusters et tendances dans les données
    """)

st.write("---")
st.header("🔷 2. Analyse des Correspondances Multiples (MCA)")

with st.expander("📌 Théorie MCA"):
    st.write("""
    - Destinée aux variables **catégorielles**  
    - Identifie les associations entre modalités  
    - Réduit la dimensionnalité pour visualiser les relations  
    - Utile pour explorer des questionnaires ou des tables de contingence complexes
    """)

st.write("---")
st.header("🔷 3. Analyse Factorielle Mixte (FAMD)")

with st.expander("📌 Théorie FAMD"):
    st.write("""
    - Destinée aux jeux de données **mixtes** (numériques + catégorielles)  
    - Combine PCA et MCA pour représenter toutes les variables sur un plan commun  
    - Permet d’identifier des groupes ou des patterns globaux
    """)

st.write("---")
st.header("🔷 4. MANOVA (Analyse Multivariée de Variance)")

with st.expander("📌 Théorie MANOVA"):
    st.write("""
    - Étend l’ANOVA à **plusieurs variables dépendantes simultanément**  
    - Vérifie si les groupes définis par des variables explicatives ont un effet global significatif  
    - Utilise des statistiques multivariées comme Wilks’ Lambda ou Pillai’s Trace
    """)

st.write("---")
st.header("🔷 5. Régression multiple et diagnostic des résidus")

with st.expander("📌 Régression multiple"):
    st.write("""
    - Modélise l’influence de **plusieurs variables explicatives** sur une variable cible  
    - Fournit coefficients, intervalles de confiance et p-values  
    """)

with st.expander("📌 Analyse des résidus"):
    st.write("""
    - Vérifie les hypothèses du modèle :  
        - Normalité des résidus (Shapiro-Wilk)  
        - Homoscédasticité (Breusch-Pagan)  
        - QQ-plot pour détecter des écarts aux hypothèses  
    - Permet de valider la qualité de la régression
    """)

st.write("---")
st.header("🔷 6. Corrélations multiples")

with st.expander("📌 Théorie corrélations"):
    st.write("""
    - Étudie les relations entre toutes les variables numériques simultanément  
    - Matrice de corrélation visualisée par carte de chaleur (heatmap)  
    - Permet d’identifier des variables fortement liées ou redondantes
    """)

st.write("---")
st.header("🎓 Conclusion")

st.write("""
L’analyse multivariée offre une vision globale et intégrée des données.  
Elle combine :

- **Réduction de dimension** (PCA, MCA, FAMD)  
- **Tests d’influence et MANOVA**  
- **Modélisation prédictive et diagnostic** (régression multiple et résidus)  
- **Exploration des corrélations entre variables**

Cette page résume les concepts théoriques essentiels pour comprendre les résultats générés par le module d’analyse multivariée.
""")
