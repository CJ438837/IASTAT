import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import re
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from Bio import Entrez

Entrez.email = "ton.email@example.com"

def _safe_key(*parts):
    """Génère une clé Streamlit sûre (alphanumérique et underscore)."""
    s = "_".join(str(p) for p in parts)
    return re.sub(r'\W+', '_', s)

def _append_result(result):
    """Ajoute un résultat à st.session_state['tests_results'] en créant la liste si besoin."""
    if "tests_results" not in st.session_state:
        st.session_state["tests_results"] = []
    st.session_state["tests_results"].append(result)

def rechercher_pubmed_links(test_name, mots_cles, max_results=3):
    """Retourne une liste de liens PubMed (liens uniquement)."""
    try:
        query = f"{test_name} AND (" + " OR ".join(mots_cles) + ")"
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids = record.get("IdList", [])
        return [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids]
    except Exception:
        return []

def propose_tests_interactif_streamlit(df, types_df, distribution_df, mots_cles):
    """
    Interface Streamlit pour proposer et exécuter des tests statistiques.
    - df : DataFrame des données
    - types_df : DataFrame avec colonnes 'variable' et 'type' (numérique / catégorielle / binaire)
    - distribution_df : DataFrame avec au moins colonnes 'variable' et 'verdict' (Normal / Non Normal)
    - mots_cles : liste de mots-clés pour PubMed
    """

    st.header("🧮 Tests statistiques interactifs (Streamlit)")

    # Prépare listes de variables
    try:
        num_vars = types_df.loc[types_df['type'] == "numérique", 'variable'].tolist()
        cat_vars = types_df.loc[types_df['type'].isin(['catégorielle', 'binaire']), 'variable'].tolist()
    except Exception as e:
        st.error(f"Erreur lecture types_df: {e}")
        return

    # Section explicative
    st.write(f"Variables numériques détectées : {num_vars}")
    st.write(f"Variables catégorielles / binaires détectées : {cat_vars}")

    # --- 1) Numérique vs Catégoriel (boucle) ---
    st.subheader("1️⃣ Numérique vs Catégoriel")
    for num, cat in itertools.product(num_vars, cat_vars):
        title = f"{num}  vs  {cat}"
        with st.expander(title):
            form_key = _safe_key("form_num_cat", num, cat)
            with st.form(key=form_key):
                # choix test en fonction du verdict si distrib dispo
                verdict_row = distribution_df.loc[distribution_df['variable'] == num]
                verdict = verdict_row['verdict'].values[0] if (not verdict_row.empty and 'verdict' in distribution_df.columns) else None

                if verdict == "Normal":
                    default_test = "t-test" if df[cat].dropna().nunique() == 2 else "ANOVA"
                else:
                    default_test = "Mann-Whitney" if df[cat].dropna().nunique() == 2 else "Kruskal-Wallis"

                select_key = _safe_key("select_numcat", num, cat)
                test_choice = st.selectbox("Test proposé", options=[default_test], key=select_key)

                apparie = False
                if test_choice in ["t-test", "Mann-Whitney"]:
                    radio_key = _safe_key("radio_apparie", num, cat)
                    apparie = st.radio("Données appariées ?", options=("Non", "Oui"), index=0, key=radio_key) == "Oui"

                # PubMed links display (non bloquant)
                liens = rechercher_pubmed_links(test_choice, mots_cles)
                if liens:
                    st.markdown("Articles PubMed suggérés :")
                    for l in liens:
                        st.markdown(f"- [{l}]({l})")

                submitted = st.form_submit_button("Exécuter le test")
                if submitted:
                    # Exécute test
                    try:
                        groupes = df.groupby(cat)[num].apply(list)
                        stat = None; p = None
                        if test_choice == "t-test":
                            if apparie:
                                stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1])
                            else:
                                stat, p = stats.ttest_ind(groupes.iloc[0], groupes.iloc[1], nan_policy='omit')
                        elif test_choice == "Mann-Whitney":
                            if apparie:
                                stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1])
                            else:
                                stat, p = stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
                        elif test_choice == "ANOVA":
                            stat, p = stats.f_oneway(*groupes)
                        elif test_choice == "Kruskal-Wallis":
                            stat, p = stats.kruskal(*groupes)
                        # affiche résultats
                        if stat is not None:
                            st.write(f"Statistique = {stat:.4f}, p-value = {p:.4g}")
                            st.write("→ Impact significatif" if p < 0.05 else "→ Pas d'impact significatif")

                            fig, ax = plt.subplots()
                            sns.boxplot(x=cat, y=num, data=df, ax=ax)
                            ax.set_title(f"{test_choice} : {num} vs {cat}")
                            st.pyplot(fig)

                            # enregistrer résultat
                            _append_result({
                                "test": test_choice,
                                "var_x": num,
                                "var_y": cat,
                                "apparie": apparie,
                                "stat": float(stat) if hasattr(stat, "__float__") else str(stat),
                                "p_value": float(p) if p is not None else None
                            })
                    except Exception as e:
                        st.error(f"Erreur exécution test : {e}")

    # --- 2) Corrélations numériques ---
    st.subheader("2️⃣ Corrélations (numérique vs numérique)")
    for var1, var2 in itertools.combinations(num_vars, 2):
        with st.expander(f"Corrélation : {var1} vs {var2}"):
            form_key = _safe_key("form_corr", var1, var2)
            with st.form(key=form_key):
                # decide test type by distribution verdict
                v1 = distribution_df.loc[distribution_df['variable'] == var1, 'verdict']
                v2 = distribution_df.loc[distribution_df['variable'] == var2, 'verdict']
                verdict1 = v1.values[0] if not v1.empty else None
                verdict2 = v2.values[0] if not v2.empty else None
                test_type = "Pearson" if verdict1 == "Normal" and verdict2 == "Normal" else "Spearman"

                submitted = st.form_submit_button(f"Exécuter corrélation ({test_type})")
                if submitted:
                    try:
                        if test_type == "Pearson":
                            corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                        else:
                            corr, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())

                        st.write(f"Corrélation = {corr:.4f}, p-value = {p:.4g}")
                        st.write("→ Corrélation significative" if p < 0.05 else "→ Pas de corrélation significative")

                        fig, ax = plt.subplots()
                        sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
                        ax.set_title(f"Corrélation ({test_type}) : {var1} vs {var2}")
                        st.pyplot(fig)

                        _append_result({
                            "test": f"Correlation_{test_type}",
                            "var_x": var1,
                            "var_y": var2,
                            "stat": float(corr),
                            "p_value": float(p)
                        })
                    except Exception as e:
                        st.error(f"Erreur corrélation : {e}")

    # --- 3) Catégorielle vs Catégorielle ---
    st.subheader("3️⃣ Variables catégorielles")
    for var1, var2 in itertools.combinations(cat_vars, 2):
        with st.expander(f"{var1} vs {var2}"):
            form_key = _safe_key("form_cat", var1, var2)
            with st.form(key=form_key):
                submitted = st.form_submit_button("Exécuter test catégoriel")
                if submitted:
                    try:
                        contingency_table = pd.crosstab(df[var1], df[var2])
                        if contingency_table.size <= 4:
                            # fisher_exact expects a 2x2 table; convert
                            if contingency_table.shape == (2, 2):
                                stat, p = stats.fisher_exact(contingency_table)
                                test_name = "Fisher exact"
                            else:
                                st.warning("Tableau non 2x2 pour Fisher ; utilisation de Chi².")
                                stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                                test_name = "Chi²"
                        else:
                            stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                            test_name = "Chi²"

                        st.write(f"{test_name} : statistique={stat:.4g}, p-value={p:.4g}")
                        st.write("→ Dépendance significative" if p < 0.05 else "→ Pas de dépendance significative")

                        fig, ax = plt.subplots()
                        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                        ax.set_title(f"{test_name} : {var1} vs {var2}")
                        st.pyplot(fig)

                        _append_result({
                            "test": test_name,
                            "var_x": var1,
                            "var_y": var2,
                            "stat": float(stat) if hasattr(stat, "__float__") else str(stat),
                            "p_value": float(p)
                        })
                    except Exception as e:
                        st.error(f"Erreur test catégoriel : {e}")

    # --- 4) Régression linéaire multiple (optionnelle) ---
    st.subheader("4️⃣ Régression linéaire multiple (optionnel)")
    if len(num_vars) > 1:
        with st.expander("Régression linéaire multiple"):
            form_key = _safe_key("form_linreg")
            with st.form(key=form_key):
                execute = st.checkbox("Exécuter régression linéaire multiple", value=False)
                cible = None
                if execute:
                    cible = st.selectbox("Variable dépendante", num_vars, key=_safe_key("select_linreg_cible"))
                submitted = st.form_submit_button("Calculer régression")
                if submitted and execute and cible:
                    try:
                        X = df[num_vars].dropna()
                        y = X[cible]
                        X_pred = X.drop(columns=[cible])
                        model = LinearRegression()
                        model.fit(X_pred, y)
                        y_pred = model.predict(X_pred)
                        residus = y - y_pred

                        st.write(f"R² = {model.score(X_pred, y):.4f}")
                        stat, p = stats.shapiro(residus)
                        st.write(f"Shapiro-Wilk résidus : stat={stat:.4f}, p={p:.4g}")
                        coef_df = pd.DataFrame({"Variable": X_pred.columns, "Coefficient": model.coef_})
                        st.table(coef_df)
                        st.write(f"Intercept : {model.intercept_:.4f}")

                        fig, axes = plt.subplots(2,2, figsize=(10,8))
                        sns.scatterplot(x=y_pred, y=residus, ax=axes[0,0])
                        axes[0,0].axhline(0, color='red', linestyle='--')
                        sns.histplot(residus, kde=True, ax=axes[0,1])
                        stats.probplot(residus, dist="norm", plot=axes[1,0])
                        sns.scatterplot(x=y, y=y_pred, ax=axes[1,1])
                        plt.tight_layout()
                        st.pyplot(fig)

                        _append_result({
                            "test": "LinearRegression",
                            "target": cible,
                            "r2": float(model.score(X_pred, y))
                        })
                    except Exception as e:
                        st.error(f"Erreur régression : {e}")

    # --- 5) PCA (optionnel) ---
    st.subheader("5️⃣ PCA (optionnel)")
    if len(num_vars) > 1:
        with st.expander("PCA"):
            form_key = _safe_key("form_pca")
            with st.form(key=form_key):
                execute = st.checkbox("Exécuter PCA", value=False, key=_safe_key("check_pca"))
                submitted = st.form_submit_button("Calculer PCA")
                if submitted and execute:
                    try:
                        X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
                        pca = PCA()
                        components = pca.fit_transform(X_scaled)
                        explained_variance = pca.explained_variance_ratio_
                        cum_var = explained_variance.cumsum()
                        n_comp = (cum_var < 0.8).sum() + 1
                        st.write(f"{n_comp} composantes expliquent ~80% de la variance")
                        loading_matrix = pd.DataFrame(pca.components_.T, index=num_vars,
                                                      columns=[f"PC{i+1}" for i in range(len(num_vars))])
                        st.write(loading_matrix.iloc[:, :n_comp])

                        fig, ax = plt.subplots()
                        ax.scatter(components[:,0], components[:,1])
                        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
                        st.pyplot(fig)

                        _append_result({"test": "PCA", "n_components_80pct": int(n_comp)})
                    except Exception as e:
                        st.error(f"Erreur PCA : {e}")

    # --- 6) Régression logistique (optionnelle si binaires) ---
    st.subheader("6️⃣ Régression logistique")
    binary_vars = [v for v in cat_vars if df[v].dropna().nunique() == 2]
    for cat in binary_vars:
        with st.expander(f"Logistique : {cat}"):
            form_key = _safe_key("form_log", cat)
            with st.form(key=form_key):
                execute = st.checkbox(f"Exécuter régression logistique pour {cat}", key=_safe_key("check_log", cat))
                submitted = st.form_submit_button("Calculer régression logistique")
                if submitted and execute:
                    try:
                        X = df[num_vars].dropna()
                        y = df[cat].loc[X.index]
                        model = LogisticRegression(max_iter=2000)
                        model.fit(X, y)
                        st.write("Coefficients :", dict(zip(X.columns, model.coef_[0])))
                        st.write(f"Intercept : {model.intercept_[0]}")
                        _append_result({"test": "LogisticRegression", "target": cat})
                    except Exception as e:
                        st.error(f"Erreur logistique : {e}")

    st.success("Interface des tests prête — les résultats sont ajoutés dans `st.session_state['tests_results']`.")
