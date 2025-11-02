import streamlit as st
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from Bio import Entrez
import numpy as np

# --- Fonction de recherche PubMed ---
def rechercher_pubmed_test(test_name, mots_cles, email="votre.email@example.com", max_results=3):
    Entrez.email = email
    query = f"{test_name} AND (" + " OR ".join(mots_cles) + ")"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()
    pmids = record['IdList']
    if pmids:
        return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]
    return []

# --- Fonction principale interactive Streamlit ---
def propose_tests_interactif_streamlit(df, types_df, distribution_df, mots_cles):
    st.title("🔬 Tests statistiques interactifs")
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    # --- 1️⃣ Numérique vs Catégoriel ---
    st.header("Numérique vs Catégoriel")
    for num, cat in itertools.product(num_vars, cat_vars):
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]
        n_modalites = df[cat].dropna().nunique()

        if n_modalites == 2:
            test_name = "t-test" if verdict=="Normal" else "Mann-Whitney"
            justification = f"{num} {'normal' if verdict=='Normal' else 'non normal'} vs {cat} à 2 modalités"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict=="Normal" else "Kruskal-Wallis"
            justification = f"{num} {'normal' if verdict=='Normal' else 'non normal'} vs {cat} >2 modalités"
        else:
            test_name = "unknown"
            justification = "Impossible de déterminer le test"

        with st.expander(f"{num} vs {cat} → {test_name}"):
            st.write("Justification :", justification)
            links = rechercher_pubmed_test(test_name, mots_cles)
            for l in links:
                st.markdown(f"[Article PubMed]({l})")
            
            if test_name in ["t-test", "Mann-Whitney"]:
                apparie = st.radio("Données appariées ?", ("Non", "Oui")) == "Oui"
            else:
                apparie = False

            run_test = st.checkbox("Exécuter le test", key=f"{num}_{cat}")
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
                    if stat is not None:
                        st.write(f"Statistique = {stat:.4f}, p-value = {p:.4g}")
                        st.write("→ Impact significatif" if p<0.05 else "→ Pas d'impact significatif")
                        fig, ax = plt.subplots()
                        sns.boxplot(x=cat, y=num, data=df, ax=ax)
                        ax.set_title(f"{test_name}: {num} vs {cat}")
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur lors du test : {e}")

    # --- 2️⃣ Numérique vs Numérique ---
    st.header("Deux variables numériques")
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        with st.expander(f"{var1} vs {var2} → Corrélation {test_type}"):
            links = rechercher_pubmed_test(f"{test_type} correlation", mots_cles)
            for l in links:
                st.markdown(f"[Article PubMed]({l})")
            run_test = st.checkbox("Exécuter la corrélation", key=f"{var1}_{var2}")
            if run_test:
                corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna()) if test_type=="Pearson" else stats.spearmanr(df[var1].dropna(), df[var2].dropna())
                st.write(f"Corrélation = {corr:.4f}, p-value = {p:.4g}")
                st.write("→ Corrélation significative" if p<0.05 else "→ Pas de corrélation significative")
                fig, ax = plt.subplots()
                sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
                ax.set_title(f"Corrélation {test_type} : {var1} vs {var2}")
                st.pyplot(fig)

    # --- 3️⃣ Catégoriel vs Catégoriel ---
    st.header("Deux variables catégorielles")
    for var1, var2 in itertools.combinations(cat_vars, 2):
        with st.expander(f"{var1} vs {var2} → Khi² / Fisher"):
            links = rechercher_pubmed_test("Chi-square test", mots_cles)
            for l in links:
                st.markdown(f"[Article PubMed]({l})")
            run_test = st.checkbox("Exécuter le test", key=f"{var1}_{var2}_cat")
            if run_test:
                contingency_table = pd.crosstab(df[var1], df[var2])
                if contingency_table.size <= 4:
                    stat, p = stats.fisher_exact(contingency_table)
                    test_name = "Fisher exact"
                else:
                    stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                    test_name = "Chi²"
                st.write(f"{test_name} : statistique = {stat:.4f}, p-value = {p:.4g}")
                st.write("→ Dépendance significative" if p<0.05 else "→ Pas de dépendance significative")
                fig, ax = plt.subplots()
                sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                ax.set_title(f"{test_name}: {var1} vs {var2}")
                st.pyplot(fig)

    # --- 4️⃣ Régression linéaire multiple ---
    st.header("Régression linéaire multiple")
    if len(num_vars) > 1:
        dep_var = st.selectbox("Variable dépendante", num_vars)
        run_reg = st.checkbox("Exécuter régression", key="regression")
        if run_reg:
            X = df[num_vars].drop(columns=[dep_var]).dropna()
            y = df[dep_var].loc[X.index]
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            residus = y - y_pred
            st.write("R² :", model.score(X, y))
            st.write(pd.DataFrame({"Variable": X.columns, "Coefficient": model.coef_}))
            st.write("Intercept :", model.intercept_)
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            sns.scatterplot(x=y_pred, y=residus, ax=axes[0,0])
            axes[0,0].axhline(0, color='red', linestyle='--')
            axes[0,0].set_title("Résidus vs Prédit")
            sns.histplot(residus, kde=True, ax=axes[0,1])
            axes[0,1].set_title("Distribution des résidus")
            stats.probplot(residus, dist="norm", plot=axes[1,0])
            axes[1,0].set_title("QQ-Plot")
            sns.scatterplot(x=y, y=y_pred, ax=axes[1,1])
            axes[1,1].plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
            axes[1,1].set_title("Observé vs Prédit")
            st.pyplot(fig)

    # --- 5️⃣ PCA ---
    st.header("Analyse en composantes principales (PCA)")
    run_pca = st.checkbox("Exécuter PCA", key="pca")
    if run_pca:
        X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
        pca = PCA().fit(X_scaled)
        components = pca.transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.cumsum()
        n_comp = (explained_var < 0.8).sum() + 1
        st.write(f"{n_comp} composante(s) principales expliquent ~80% variance")
        fig, ax = plt.subplots()
        ax.scatter(components[:,0], components[:,1])
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Projection individus (PC1 vs PC2)")
        st.pyplot(fig)

    # --- 6️⃣ Régression logistique pour variable binaire ---
    st.header("Régression logistique (binaire)")
    for cat in cat_vars:
        if df[cat].dropna().nunique() == 2:
            with st.expander(f"Régression logistique : {cat}"):
                run_log = st.checkbox(f"Exécuter régression pour {cat}", key=f"log_{cat}")
                if run_log:
                    X = df[num_vars].dropna()
                    y = df[cat].loc[X.index]
                    model = LogisticRegression(max_iter=1000).fit(X, y)
                    st.write("Coefficients :", dict(zip(num_vars, model.coef_[0])))
                    st.write("Intercept :", model.intercept_[0])
