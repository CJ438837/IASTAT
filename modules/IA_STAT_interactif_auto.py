import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import itertools
import numpy as np

def propose_tests_interactif_auto(types_df, distribution_df, df, mots_cles):
    """
    Parcourt tous les tests statistiques possibles et permet
    de choisir pour chaque test si les données sont appariées,
    puis exécute le test et affiche les résultats et graphiques.
    Inclut ANOVA pour >2 groupes et MCA pour variables catégorielles multiples.
    """
    if "results_tests" not in st.session_state:
        st.session_state.results_tests = []

    # Identifier variables numériques et catégorielles
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    st.info("Parcours des tests statistiques proposés...")

    # --- 1️⃣ Numérique vs Catégoriel ---
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable'] == num, 'verdict'].values[0]

        if n_modalites == 2:
            test_name = "t-test" if verdict == "Normal" else "Mann-Whitney"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict == "Normal" else "Kruskal-Wallis"
        else:
            test_name = "unknown"

        st.subheader(f"Test proposé : {test_name} ({num} vs {cat})")

        apparie = False
        if test_name in ["t-test", "Mann-Whitney"]:
            apparie = st.radio(
                "Données appariées ?", 
                ["Non", "Oui"], 
                key=f"app_{num}_{cat}"
            ) == "Oui"

        run_test = st.button("Exécuter ce test", key=f"run_{num}_{cat}")
        if run_test:
            groupes = df.groupby(cat)[num].apply(list)
            try:
                if test_name == "t-test":
                    stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
                elif test_name == "Mann-Whitney":
                    stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
                elif test_name == "ANOVA":
                    stat, p = stats.f_oneway(*groupes)
                elif test_name == "Kruskal-Wallis":
                    stat, p = stats.kruskal(*groupes)
                else:
                    stat, p = None, None

                st.write(f"Statistique = {stat}, p-value = {p}")
                if p is not None:
                    if p < 0.05:
                        st.success(f"→ La variable '{num}' a un impact significatif sur '{cat}'")
                    else:
                        st.info(f"→ Aucun impact significatif détecté")

                # Graphique
                plt.figure()
                sns.boxplot(x=cat, y=num, data=df)
                plt.title(f"{test_name} : {num} vs {cat}")
                st.pyplot(plt.gcf())
                plt.close()

                st.session_state.results_tests.append({
                    "type": test_name,
                    "variables": (num, cat),
                    "apparie": apparie,
                    "stat": stat,
                    "pvalue": p
                })
            except Exception as e:
                st.error(f"Erreur lors du test : {e}")

    # --- 2️⃣ Deux variables continues ---
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable'] == var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable'] == var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1 == "Normal" and verdict2 == "Normal" else "Spearman"

        st.subheader(f"Corrélation ({test_type}) : {var1} vs {var2}")
        run_test = st.button("Exécuter ce test", key=f"run_{var1}_{var2}")
        if run_test:
            if test_type == "Pearson":
                corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
            else:
                corr, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
            st.write(f"Corrélation = {corr}, p-value = {p}")

            plt.figure()
            sns.scatterplot(x=var1, y=var2, data=df)
            plt.title(f"Corrélation ({test_type}) : {var1} vs {var2}")
            st.pyplot(plt.gcf())
            plt.close()

            st.session_state.results_tests.append({
                "type": f"{test_type} corr",
                "variables": (var1, var2),
                "stat": corr,
                "pvalue": p
            })

    # --- 3️⃣ Deux variables catégorielles ---
    for var1, var2 in itertools.combinations(cat_vars, 2):
        st.subheader(f"Khi² / Fisher : {var1} vs {var2}")
        run_test = st.button("Exécuter ce test", key=f"run_cat_{var1}_{var2}")
        if run_test:
            contingency_table = pd.crosstab(df[var1], df[var2])
            if contingency_table.size <= 4:
                stat, p = stats.fisher_exact(contingency_table)
                test_name = "Fisher exact"
            else:
                stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                test_name = "Chi²"

            st.write(f"{test_name} : statistique = {stat}, p-value = {p}")
            plt.figure()
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm")
            plt.title(f"{test_name} : {var1} vs {var2}")
            st.pyplot(plt.gcf())
            plt.close()

            st.session_state.results_tests.append({
                "type": test_name,
                "variables": (var1, var2),
                "stat": stat,
                "pvalue": p
            })

    # --- 4️⃣ MCA pour variables catégorielles multiples ---
    if len(cat_vars) > 1:
        st.subheader("Analyse des correspondances multiples (MCA)")
        run_mca = st.button("Exécuter MCA", key="run_mca")
        if run_mca:
            try:
                import prince
                df_cat = df[cat_vars].fillna("Missing")
                mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)

                coords = mca.column_coordinates(df_cat)
                ind_coords = mca.row_coordinates(df_cat)
                st.write(f"Variance expliquée : {mca.explained_inertia_}")

                # Graphiques
                plt.figure(figsize=(7,6))
                plt.scatter(ind_coords[0], ind_coords[1], alpha=0.6)
                plt.xlabel("Dim 1")
                plt.ylabel("Dim 2")
                plt.title("Projection des individus (MCA)")
                st.pyplot(plt.gcf())
                plt.close()

                plt.figure(figsize=(7,6))
                plt.scatter(coords[0], coords[1], color='red', alpha=0.7)
                for i, label in enumerate(coords.index):
                    plt.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=9, color='darkred')
                plt.xlabel("Dim 1")
                plt.ylabel("Dim 2")
                plt.title("Projection des catégories (MCA)")
                st.pyplot(plt.gcf())
                plt.close()

            except ImportError:
                st.warning("Le module 'prince' n'est pas installé. Installer avec : pip install prince")

    # --- 5️⃣ Régression linéaire et PCA multivariée ---
    if len(num_vars) > 1:
        st.subheader("Régression linéaire multiple")
        cible = st.selectbox("Variable dépendante pour régression linéaire", num_vars, key="cible_reg")
        run_reg = st.button("Exécuter régression", key="run_reg")
        if run_reg:
            X = df[num_vars].drop(columns=[cible]).dropna()
            y = df[cible].loc[X.index]
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residus = y - y_pred
            st.write(f"R² : {model.score(X, y):.4f}")
            st.write(pd.DataFrame({"variable": X.columns, "coef": model.coef_}))

            plt.figure(figsize=(10,5))
            sns.scatterplot(x=y_pred, y=residus)
            plt.axhline(0, color='red', linestyle='--')
            plt.title("Résidus vs Prédit")
            st.pyplot(plt.gcf())
            plt.close()

        st.subheader("PCA")
        run_pca = st.button("Exécuter PCA", key="run_pca")
        if run_pca:
            X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            plt.figure()
            plt.scatter(components[:,0], components[:,1])
            plt.title("Projection individus (PC1 vs PC2)")
            st.pyplot(plt.gcf())
            plt.close()

    st.success("✅ Tous les tests interactifs terminés.")
