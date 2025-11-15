import streamlit as st

def app():
    # --- Titre principal ---
    st.title("📊 Tests Bivariés")
    st.markdown("---")

    # --- Objectif ---
    st.subheader("🎯 Objectif")
    st.markdown("""
    Les tests bivariés permettent d’explorer les **relations entre deux variables**.  
    Selon les types de variables (numérique, catégorielle, binaire) et les propriétés des données (normalité, variance), différents tests statistiques sont utilisés.
    
    Ils servent à :
    - Identifier les corrélations ou associations significatives
    - Mesurer la force de ces relations via **effect sizes** et Cramér’s V
    - Fournir des intervalles de confiance pour les corrélations
    - Orienter l’analyse multivariée ou la modélisation prédictive
    """)

    st.markdown("---")

    # --- Variables numériques vs numériques ---
    st.subheader("🔷 Variables numériques vs numériques")
    st.markdown("""
    - **Corrélation de Pearson** : si les deux variables sont **normales**  
    - **Corrélation de Spearman** : si l’une ou les deux variables ne sont pas normales  
    - **Kendall Tau** : robuste aux valeurs aberrantes  
    - **Bootstrap IC** : intervalles de confiance pour les corrélations  
    - **Pente robuste Theil-Sen** : estimation de la tendance linéaire robuste  
    - **Visualisation** : scatter plot avec ligne de tendance  
    - **Interprétation** : coefficient entre -1 et 1, proche de 0 = pas de relation
    """)

    st.markdown("---")

    # --- Variables numériques vs catégorielles ---
    st.subheader("🔷 Variables numériques vs catégorielles")
    st.markdown("""
    - **T-test** (2 groupes) ou **ANOVA** (≥3 groupes) : si normalité et homogénéité des variances  
    - **T-test apparié** ou **Wilcoxon** : si données appariées  
    - **Mann-Whitney / Kruskal-Wallis** : si non-normalité  
    - **Taille d’effet** : Cohen’s d (2 groupes), eta² (≥3 groupes), rank-biserial pour tests non paramétriques  
    - **Visualisation** : boxplot, violin plot  
    - **Interprétation** : p-value < 0.05 → différence statistiquement significative, taille d’effet interprétée en plus de la p-value
    """)

    st.markdown("---")

    # --- Variables catégorielles vs catégorielles ---
    st.subheader("🔷 Variables catégorielles vs catégorielles")
    st.markdown("""
    - **Test du Chi²** : si effectifs suffisants  
    - **Test exact de Fisher** : si effectifs faibles ou tableau 2x2  
    - **Cramér’s V** : mesure de la force de l’association (0 = pas d’association, 1 = association parfaite)  
    - **Visualisation** : heatmap (tableau de contingence)  
    - **Interprétation** : p-value < 0.05 → association significative, Cramér’s V décrit la force de l’association
    """)

    st.markdown("---")

    # --- Bonnes pratiques ---
    st.subheader("💡 Bonnes pratiques")
    st.markdown("""
    - Vérifiez la normalité et l’homogénéité avant d’appliquer des tests paramétriques  
    - Toujours représenter graphiquement les relations pour mieux interpréter  
    - Tenir compte de la taille d’effet et des intervalles de confiance, pas seulement de la p-value  
    - Pour les variables catégorielles avec modalités rares, envisagez un regroupement  
    - Utiliser la correction FDR (p-value corrigée) lorsqu’on teste plusieurs relations simultanément
    """)

    st.markdown("---")

    # --- Conclusion ---
    st.subheader("🎓 Conclusion")
    st.markdown("""
    Les tests bivariés permettent de détecter **significativité, force et robustesse des relations**.  
    Ils sont essentiels pour orienter l’analyse multivariée ou la modélisation prédictive, et pour comprendre les mécanismes sous-jacents dans vos données.
    """)

    st.markdown("---")
    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
