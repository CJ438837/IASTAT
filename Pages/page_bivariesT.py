import streamlit as st

def app():
    st.title("📊 Tests Bivariés — Théorie & Interprétation")
    st.write("---")

    st.header("🎯 Objectif")
    st.write("""
    Les tests bivariés permettent d’explorer les **relations entre deux variables**.  
    Selon les types de variables (numérique, catégorielle, binaire), différents tests statistiques sont utilisés.
    """)

    st.write("---")
    st.header("🔷 Variables numériques vs numériques")
    with st.expander("📌 Théorie"):
        st.write("""
        - Corrélation de Pearson : si les deux variables sont **normales**  
        - Corrélation de Spearman : si les distributions ne sont pas normales  
        - Kendall Tau : robuste aux valeurs aberrantes  
        - Visualisation : scatter plot
        """)

    st.write("---")
    st.header("🔷 Variables numériques vs catégorielles")
    with st.expander("📌 Théorie"):
        st.write("""
        - T-test ou ANOVA : si normalité et homogénéité des variances  
        - Mann-Whitney / Kruskal-Wallis : si non-normalité  
        - Taille d’effet : Cohen’s d ou eta²  
        - Visualisation : boxplot
        """)

    st.write("---")
    st.header("🔷 Variables catégorielles vs catégorielles")
    with st.expander("📌 Théorie"):
        st.write("""
        - Test du Chi² ou test exact de Fisher  
        - Mesure de force de l’association : Cramér’s V  
        - Visualisation : heatmap (tableau de contingence)
        """)

    st.write("---")
    st.header("🎓 Conclusion")
    st.write("""
    Les tests bivariés permettent de détecter **significativité et force des relations**.  
    Ils sont essentiels pour orienter l’analyse multivariée ou la modélisation prédictive.
    """)
