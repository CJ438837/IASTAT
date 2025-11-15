
import streamlit as st

# ---------------------------------------------
# PAGE : Tests bivariés — Théorie & Interprétation
# ---------------------------------------------

st.title("🔍 Analyse Bivariée — Théorie & Interprétation")
st.write("---")

st.header("🎯 Objectif de l’analyse bivariée")
st.write("""
L’analyse bivariée examine la relation entre **deux variables**.
Selon leur nature (numérique ou catégorielle), différents tests sont utilisés pour déterminer :

- si deux variables numériques sont corrélées,
- si une variable numérique diffère selon des groupes,
- si deux variables catégorielles sont associées.

Cette page présente **uniquement la théorie**, sans exécution de code.
""")

st.write("---")
st.header("🔷 1. Numérique vs Numérique")

with st.expander("📌 Corrélations possibles"):
    st.subheader("🔹 Corrélation de Pearson")
    st.write("""
    - Suppose une distribution normale  
    - Mesure une relation **linéaire**  
    - Valeurs entre -1 et 1  
    """)

    st.subheader("🔹 Corrélation de Spearman")
    st.write("""
    - Test **non paramétrique**  
    - Mesure une relation **monotone**  
    - Plus robuste aux valeurs extrêmes  
    """)

    st.subheader("🔹 Tau de Kendall")
    st.write("""
    - Alternative non paramétrique stricte  
    - Basée sur la concordance des paires  
    """)

st.write("---")
st.header("🔷 2. Numérique vs Catégoriel")

with st.expander("📌 Comparaison de moyennes ou distributions"):
    st.subheader("Cas : 2 groupes")

    st.write("### 🔹 Test t de Student")
    st.write("""
    Conditions :
    - Normalité dans chaque groupe
    - Variances homogènes

    Ce test vérifie si les moyennes sont significativement différentes.
    """)

    st.write("### 🔹 Test t apparié")
    st.write("""
    Utilisé lorsque les mesures proviennent des **mêmes individus** (avant/après).
    """)

    st.write("### 🔹 Test de Mann-Whitney")
    st.write("""
    - Alternative non paramétrique au test t  
    - Aucune hypothèse de normalité  
    - Compare les **distributions** plutôt que les moyennes
    """)

    st.subheader("Cas : plus de 2 groupes")
    st.write("""
    ### 🔹 ANOVA
    - Requiert normalité + homogénéité des variances  
    - Vérifie si **au moins une** moyenne diffère des autres

    ### 🔹 Test de Kruskal-Wallis  
    - Version non paramétrique  
    - Analyse les **rangs** plutôt que les valeurs brutes  
    """)

st.write("---")
st.header("🔷 3. Catégoriel vs Catégoriel")

with st.expander("📌 Tests d'indépendance"):
    st.write("### 🔹 Test du Chi-Deux")
    st.write("""
    - Compare les fréquences observées vs attendues  
    - Hypothèse : les variables sont indépendantes  
    - Requiert des effectifs attendus ≥ 5  
    """)

    st.write("### 🔹 Test exact de Fisher")
    st.write("""
    - Idéal pour les petits effectifs  
    - Fonctionne sur les tableaux 2×2  
    - Aucune hypothèse de normalité  
    """)

st.write("---")
st.header("🔷 4. Taille d’effet")

with st.expander("📌 Importance réelle de la relation"):
    st.write("### 🔹 Cohen's d")
    st.write("""
    - <0.2 : très faible  
    - 0.2–0.5 : faible  
    - 0.5–0.8 : modéré  
    - >0.8 : fort  
    """)

    st.write("### 🔹 Eta² (ANOVA)")
    st.write("""
    Proportion de la variance expliquée par la variable catégorielle.
    """)

    st.write("### 🔹 V de Cramér (catégoriel vs catégoriel)")
    st.write("""
    - 0–0.1 : très faible  
    - 0.1–0.3 : faible  
    - 0.3–0.5 : modérée  
    - >0.5 : forte  
    """)

    st.write("### 🔹 Corrélation bisérielle des rangs (Mann-Whitney)")
    st.write("""
    Mesure l'intensité de la différence entre deux distributions.
    """)

st.write("---")
st.header("🔷 5. Conditions statistiques")

with st.expander("📌 Normalité & Homogénéité"):
    st.write("### 🔹 Normalité")
    st.write("""
    - Test de Shapiro-Wilk  
    - QQ-plots  
    """)

    st.write("### 🔹 Homogénéité des variances")
    st.write("""
    - Test de Levene  
    - Requis pour test t et ANOVA  
    """)

st.write("---")
st.header("🎓 Conclusion")

st.write("""
L'analyse bivariée consiste à sélectionner automatiquement le test adapté selon :

- le type des variables,
- la distribution des données,
- le nombre de groupes,
- les hypothèses statistiques.

L’interprétation combine :
- la significativité,
- la taille d’effet,
- un diagnostic de cohérence statistique.

Cette page résume les **concepts théoriques fondamentaux** utilisés par les modules d’analyse automatiques.
""")
