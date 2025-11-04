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

    # --- 3️⃣ Sélection des options utilisateur ---
    st.markdown("### ⚙️ Options des tests")
    apparie = st.radio(
        "Les tests à deux groupes sont-ils appariés ?",
        ("Non", "Oui"),
        index=0
    ) == "Oui"

    lancer_tests = st.button("🧠 Exécuter tous les tests")

    if lancer_tests:
        with st.spinner("Exécution des tests en cours... ⏳"):
            try:
                summary_df, all_results = propose_tests_interactif_auto(
                    types_df, distribution_df, df, mots_cles, apparie=apparie
                )
                st.success("✅ Tous les tests ont été exécutés avec succès !")

                # --- 4️⃣ Affichage du résumé des tests ---
                st.markdown("### 📄 Résumé des tests")
                st.dataframe(summary_df)

                # --- 5️⃣ Graphiques numériques vs catégorielles (boxplots) ---
                st.markdown("### 📊 Boxplots Numérique vs Catégoriel")
                num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
                cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

                for num, cat in [(n, c) for n in num_vars for c in cat_vars]:
                    fig, ax = plt.subplots()
                    sns.boxplot(x=cat, y=num, data=df, ax=ax)
                    ax.set_title(f"{num} vs {cat}")
                    st.pyplot(fig)
                    plt.close(fig)

                # --- 6️⃣ PCA pour variables numériques ---
                if len(num_vars) > 1:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.decomposition import PCA

                    st.markdown("### 📈 PCA")
                    X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
                    pca = PCA()
                    components = pca.fit_transform(X_scaled)
                    explained_var = pca.explained_variance_ratio_.cumsum()

                    # Projection individus PC1 vs PC2
                    fig, ax = plt.subplots()
                    ax.scatter(components[:,0], components[:,1], alpha=0.6)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("Projection individus PC1 vs PC2")
                    st.pyplot(fig)
                    plt.close(fig)

                    # Biplot
                    fig, ax = plt.subplots()
                    ax.scatter(components[:,0], components[:,1], alpha=0.5)
                    for i, var in enumerate(num_vars):
                        ax.arrow(0, 0,
                                 pca.components_[0,i]*max(components[:,0]),
                                 pca.components_[1,i]*max(components[:,1]),
                                 color='red', alpha=0.7, head_width=0.05)
                        ax.text(pca.components_[0,i]*max(components[:,0])*1.1,
                                pca.components_[1,i]*max(components[:,1])*1.1,
                                var, color='darkred', ha='center', va='center')
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("Biplot PCA")
                    st.pyplot(fig)
                    plt.close(fig)

                # --- 7️⃣ MCA pour variables catégorielles ---
                if len(cat_vars) > 1:
                    try:
                        import prince
                        st.markdown("### 📊 MCA")
                        df_cat = df[cat_vars].dropna()
                        mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
                        coords = mca.column_coordinates(df_cat)

                        # Projection des individus
                        fig, ax = plt.subplots()
                        ind_coords = mca.row_coordinates(df_cat)
                        ax.scatter(ind_coords[0], ind_coords[1], alpha=0.6)
                        ax.set_xlabel("Dim 1")
                        ax.set_ylabel("Dim 2")
                        ax.set_title("Projection individus (MCA)")
                        st.pyplot(fig)
                        plt.close(fig)

                        # Projection des catégories
                        fig, ax = plt.subplots()
                        ax.scatter(coords[0], coords[1], color='red', alpha=0.7)
                        for i, label in enumerate(coords.index):
                            ax.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=9, color='darkred')
                        ax.set_xlabel("Dim 1")
                        ax.set_ylabel("Dim 2")
                        ax.set_title("Projection catégories (MCA)")
                        st.pyplot(fig)
                        plt.close(fig)

                    except ImportError:
                        st.warning("Module 'prince' non installé. Pour MCA, exécutez : pip install prince")

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution des tests : {e}")
