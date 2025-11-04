import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_interactif_auto import propose_tests_interactif_auto
import numpy as np

def app():
    st.title("📊 Tests statistiques automatiques")

    # --- 1️⃣ Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser la distribution des données dans la page Distribution.")
        st.stop()

    # --- 2️⃣ Récupération des données depuis la session ---
    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    st.markdown("### ⚙️ Sélection et exécution des tests")

    # Liste des tests disponibles
    tests_disponibles = [
        "t-test / Mann-Whitney",
        "ANOVA / Kruskal-Wallis",
        "Corrélation (Pearson / Spearman)",
        "Chi² / Fisher",
    ]

    for test in tests_disponibles:
        with st.expander(f"🧪 {test}"):
            apparie = st.radio(
                f"Les groupes sont-ils appariés pour {test} ?",
                ("Non", "Oui"),
                key=f"apparie_{test}"
            ) == "Oui"

            if st.button(f"▶️ Lancer {test}", key=f"lancer_{test}"):
                with st.spinner(f"Exécution de {test} en cours... ⏳"):
                    try:
                        # Exécution du test unique
                        summary_df, all_results = propose_tests_interactif_auto(
                            types_df, distribution_df, df, mots_cles, apparie=apparie, test_selectionne=test
                        )

                        st.success(f"✅ Test {test} exécuté avec succès !")

                        # --- Résumé ---
                        st.markdown(f"### 📄 Résumé des résultats - {test}")
                        st.dataframe(summary_df)

                        # --- Graphiques associés ---
                        num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
                        cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

                        if test in ["t-test / Mann-Whitney", "ANOVA / Kruskal-Wallis"]:
                            st.markdown("#### 📊 Boxplots des groupes")
                            for num in num_vars:
                                for cat in cat_vars:
                                    if df[cat].nunique() > 1:
                                        fig, ax = plt.subplots()
                                        sns.boxplot(x=cat, y=num, data=df, ax=ax)
                                        ax.set_title(f"{num} vs {cat}")
                                        st.pyplot(fig)
                                        plt.close(fig)

                        elif test == "Corrélation (Pearson / Spearman)":
                            st.markdown("#### 🔗 Matrice de corrélation")
                            corr = df[num_vars].corr(method="pearson")
                            fig, ax = plt.subplots()
                            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                            ax.set_title("Matrice de corrélation (Pearson)")
                            st.pyplot(fig)
                            plt.close(fig)

                        elif test == "Chi² / Fisher":
                            st.markdown("#### 📊 Tableau de contingence (exemple)")
                            if len(cat_vars) >= 2:
                                contingency = pd.crosstab(df[cat_vars[0]], df[cat_vars[1]])
                                st.dataframe(contingency)

                    except Exception as e:
                        st.error(f"❌ Erreur pendant l'exécution de {test} : {e}")

    # --- PCA et MCA en bas de page ---
    st.markdown("---")
    st.markdown("## 🔬 Analyses multivariées")

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    if len(num_vars) > 1:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        st.markdown("### 📈 PCA")
        X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
        pca = PCA()
        components = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.cumsum()

        fig, ax = plt.subplots()
        ax.scatter(components[:, 0], components[:, 1], alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Projection individus PC1 vs PC2")
        st.pyplot(fig)
        plt.close(fig)

    if len(cat_vars) > 1:
        try:
            import prince
            st.markdown("### 📊 MCA")
            df_cat = df[cat_vars].dropna()
            mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
            coords = mca.column_coordinates(df_cat)

            fig, ax = plt.subplots()
            ind_coords = mca.row_coordinates(df_cat)
            ax.scatter(ind_coords[0], ind_coords[1], alpha=0.6)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_title("Projection individus (MCA)")
            st.pyplot(fig)
            plt.close(fig)

            fig, ax = plt.subplots()
            ax.scatter(coords[0], coords[1], color='red', alpha=0.7)
            for i, label in enumerate(coords.index):
                ax.text(coords.iloc[i, 0], coords.iloc[i, 1], label, fontsize=9, color='darkred')
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_title("Projection catégories (MCA)")
            st.pyplot(fig)
            plt.close(fig)

        except ImportError:
            st.warning("Module 'prince' non installé. Pour MCA, exécutez : pip install prince")
