import streamlit as st
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression

def propose_tests_interactif_streamlit(df: pd.DataFrame,
                                       types_df: pd.DataFrame,
                                       distribution_df: pd.DataFrame,
                                       mots_cles: list|None = None):
    """
    Interface Streamlit complète pour proposer et exécuter tests statistiques.
    Enregistre les résultats dans st.session_state['tests_results'] (liste de dicts).

    Inputs:
        - df: DataFrame des données
        - types_df: DataFrame avec colonnes au minimum ['variable','type']
                   types attendus: 'numérique', 'catégorielle', 'binaire'
        - distribution_df: DataFrame avec colonne 'variable' et idéalement 'verdict' ou 'best_fit_distribution'
        - mots_cles: liste optionnelle de mots-clés pour contexte / PubMed (non utilisé ici mais reçu)
    """
    st.header("🧪 Tests statistiques interactifs (version complète)")

    # validation
    if df is None or df.empty:
        st.warning("Importe d'abord des données (page Fichier).")
        return
    if types_df is None or types_df.empty:
        st.warning("Détecte d'abord les types de variables (page Variables).")
        return

    # normalize distribution_df access
    def get_verdict(var):
        if distribution_df is None:
            return "unknown"
        for col in ("verdict", "best_fit_distribution", "distribution"):
            if col in distribution_df.columns:
                row = distribution_df.loc[distribution_df["variable"] == var, col]
                if not row.empty:
                    return str(row.iloc[0]).lower()
        return "unknown"

    # prepare lists
    vars_all = list(df.columns)
    # map types (if multiple sheets use same names)
    try:
        var_to_type = dict(zip(types_df['variable'], types_df['type']))
    except Exception:
        st.error("Le DataFrame types_df doit contenir les colonnes 'variable' et 'type'.")
        return

    num_vars = [v for v in vars_all if var_to_type.get(v) == "numérique"]
    cat_vars = [v for v in vars_all if var_to_type.get(v) in ("catégorielle", "binaire")]
    st.write(f"Variables numériques détectées: {num_vars}")
    st.write(f"Variables catégorielles détectées: {cat_vars}")

    if "tests_results" not in st.session_state:
        st.session_state["tests_results"] = []

    # utility for saving a result
    def save_result(entry: dict):
        st.session_state["tests_results"].append(entry)

    # --- PART A: Numérique vs Catégorielle (univarié outcome numeric) ---
    st.subheader("1) Numérique ← vs → Catégorielle")
    for i, (num, cat) in enumerate(itertools.product(num_vars, cat_vars)):
        with st.expander(f"{num}  vs  {cat}", expanded=False):
            # quick checks
            cats = df[cat].dropna().unique()
            if len(cats) < 2:
                st.info(f"`{cat}` n'a qu'une modalité valide, impossible de tester.")
                continue

            dist = get_verdict(num)
            is_norm = "norm" in dist or "normal" in dist

            st.write(f"Distribution détectée pour `{num}` : **{dist}**")
            st.write("Choisir le test et si les données sont appariées :")

            # unique keys to avoid duplicate IDs
            key_testsel = f"testsel_num_cat_{i}_{num}_{cat}"
            key_apparie = f"apparie_num_cat_{i}_{num}_{cat}"
            test_options = []
            if len(cats) == 2:
                test_options = ["t-test (param)", "Mann-Whitney (non param)", "Wilcoxon (apparié)"]
                if is_norm:
                    default = "t-test (param)"
                else:
                    default = "Mann-Whitney (non param)"
            else:
                # >2 categories
                test_options = ["ANOVA (param)", "Kruskal-Wallis (non param)"]
                default = "ANOVA (param)" if is_norm else "Kruskal-Wallis (non param)"

            test_choice = st.selectbox("Test proposé :", test_options, index=max(0, test_options.index(default)), key=key_testsel)
            apparie = False
            if test_choice in ("Wilcoxon (apparié)", "t-test (param)"):
                # allow user to toggle pairing explicitly
                apparie = st.checkbox("Données appariées ?", key=key_apparie)

            # run button
            key_run = f"run_num_cat_{i}_{num}_{cat}"
            if st.button(f"Exécuter {test_choice}", key=key_run):
                try:
                    # prepare groups
                    groups = [df.loc[df[cat] == val, num].dropna() for val in np.sort(cats)]
                    # handle tests
                    if test_choice.startswith("t-test"):
                        if len(groups) == 2 and apparie:
                            stat, p = stats.ttest_rel(groups[0], groups[1])
                        elif len(groups) == 2:
                            stat, p = stats.ttest_ind(groups[0], groups[1], nan_policy='omit')
                        else:
                            stat, p = stats.f_oneway(*groups)
                    elif test_choice == "Mann-Whitney (non param)":
                        stat, p = stats.mannwhitneyu(groups[0], groups[1])
                    elif test_choice == "Wilcoxon (apparié)":
                        stat, p = stats.wilcoxon(groups[0], groups[1])
                    elif test_choice == "ANOVA (param)":
                        stat, p = stats.f_oneway(*groups)
                    elif test_choice == "Kruskal-Wallis (non param)":
                        stat, p = stats.kruskal(*groups)
                    else:
                        stat, p = (None, None)

                    st.write(f"Résultat : statistic={stat}, p-value={p:.4g}" if stat is not None else "Test non exécuté.")
                    # boxplot
                    fig, ax = plt.subplots()
                    sns.boxplot(x=cat, y=num, data=df, ax=ax)
                    ax.set_title(f"{test_choice} — {num} vs {cat}")
                    st.pyplot(fig)

                    # save
                    save_result({
                        "type": "num_vs_cat",
                        "num": num, "cat": cat,
                        "test": test_choice,
                        "apparie": apparie,
                        "statistic": float(stat) if stat is not None else None,
                        "p_value": float(p) if p is not None else None
                    })
                except Exception as e:
                    st.error(f"Erreur exécution test : {e}")

    # --- PART B: Numérique vs Numérique (corrélations) ---
    st.subheader("2) Numérique ↔ Numérique (corrélations)")
    for j, (v1, v2) in enumerate(itertools.combinations(num_vars, 2)):
        with st.expander(f"{v1}  vs  {v2}", expanded=False):
            d1, d2 = get_verdict(v1), get_verdict(v2)
            is_norm1 = "norm" in d1 or "normal" in d1
            is_norm2 = "norm" in d2 or "normal" in d2

            default_corr = "Pearson" if is_norm1 and is_norm2 else "Spearman"
            key_corr_choice = f"corr_choice_{j}_{v1}_{v2}"
            corr_choice = st.selectbox("Type de corrélation :", ["Pearson", "Spearman"], index=0 if default_corr=="Pearson" else 1, key=key_corr_choice)

            key_corr_run = f"run_corr_{j}_{v1}_{v2}"
            if st.button(f"Calculer corrélation {corr_choice}", key=key_corr_run):
                try:
                    x = df[v1].dropna()
                    y = df[v2].dropna()
                    # align indices
                    common_index = x.index.intersection(y.index)
                    x, y = x.loc[common_index], y.loc[common_index]
                    if corr_choice == "Pearson":
                        r, p = stats.pearsonr(x, y)
                    else:
                        r, p = stats.spearmanr(x, y)

                    st.write(f"{corr_choice} : r = {r:.4f}, p = {p:.4g}")
                    fig, ax = plt.subplots()
                    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'s':20})
                    ax.set_title(f"{corr_choice} entre {v1} et {v2}")
                    st.pyplot(fig)

                    save_result({
                        "type": "num_vs_num",
                        "var1": v1, "var2": v2,
                        "test": f"{corr_choice} correlation",
                        "r": float(r), "p_value": float(p)
                    })
                except Exception as e:
                    st.error(f"Erreur corrélation : {e}")

    # --- PART C: Catégorielle vs Catégorielle (Chi² / Fisher) ---
    st.subheader("3) Catégorielle ↔ Catégorielle")
    for k, (c1, c2) in enumerate(itertools.combinations(cat_vars, 2)):
        with st.expander(f"{c1}  vs  {c2}", expanded=False):
            contingency = pd.crosstab(df[c1].fillna("Missing"), df[c2].fillna("Missing"))
            st.dataframe(contingency)
            key_cat_run = f"run_cat_{k}_{c1}_{c2}"
            if st.button(f"Exécuter Chi² / Fisher ({c1} vs {c2})", key=key_cat_run):
                try:
                    if contingency.size <= 4:
                        # Fisher expects 2x2 -- ensure shape
                        if contingency.shape == (2,2):
                            oddsratio, p = stats.fisher_exact(contingency)
                            st.write(f"Fisher exact : p={p:.4g}, oddsratio={oddsratio:.4g}")
                            save_result({"type":"cat_vs_cat","var1":c1,"var2":c2,"test":"Fisher","p_value":float(p)})
                        else:
                            st.warning("Fisher exact applicable uniquement sur table 2x2.")
                    else:
                        chi2, p, dof, exp = stats.chi2_contingency(contingency)
                        st.write(f"Chi² : chi2={chi2:.4f}, p={p:.4g}, dof={dof}")
                        save_result({"type":"cat_vs_cat","var1":c1,"var2":c2,"test":"Chi2","chi2":float(chi2),"p_value":float(p)})
                except Exception as e:
                    st.error(f"Erreur test catégoriel : {e}")

    # --- PART D: Régression linéaire multiple (optionnelle) ---
    st.subheader("4) Régression linéaire multiple (option)")
    if len(num_vars) > 1:
        key_linrun = "linreg_run"
        execute_lin = st.checkbox("Exécuter régression linéaire multiple", key="linreg_checkbox")
        if execute_lin:
            # select target
            key_target = "linreg_target"
            target = st.selectbox("Variable dépendante (target)", options=num_vars, key=key_target)
            predictors = [v for v in num_vars if v != target]
            if st.button("Lancer régression linéaire", key=key_linrun):
                try:
                    X = df[predictors].dropna()
                    y = df[target].loc[X.index]
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    residus = y - y_pred
                    r2 = model.score(X, y)
                    st.write(f"R² = {r2:.4f}")
                    coef_df = pd.DataFrame({"variable": predictors, "coef": model.coef_})
                    st.dataframe(coef_df)
                    # resid plots
                    fig, axes = plt.subplots(1,2, figsize=(10,4))
                    sns.scatterplot(x=y_pred, y=residus, ax=axes[0]); axes[0].axhline(0, ls="--")
                    sns.histplot(residus, kde=True, ax=axes[1])
                    st.pyplot(fig)
                    save_result({"type":"linear_regression","target":target,"r2":float(r2),"coef":coef_df.to_dict(orient="list")})
                except Exception as e:
                    st.error(f"Erreur régression linéaire : {e}")

    # --- PART E: Régression logistique pour variable binaire dépendante ---
    st.subheader("5) Régression logistique (variable binaire)")
    binaries = [c for c in cat_vars if df[c].dropna().nunique()==2]
    if binaries:
        key_log_choice = "log_reg_choice"
        target_log = st.selectbox("Choisir variable binaire dépendante :", options=["(none)"] + binaries, key=key_log_choice)
        if target_log != "(none)":
            predictors_log = st.multiselect("Choisir prédicteurs numériques :", options=[v for v in num_vars if v != target_log], default=[v for v in num_vars if v != target_log][:3], key=f"preds_log_{target_log}")
            key_log_run = f"run_logreg_{target_log}"
            if st.button("Lancer régression logistique", key=key_log_run):
                try:
                    X = df[predictors_log].dropna()
                    y = df[target_log].loc[X.index]
                    model = LogisticRegression(max_iter=2000)
                    model.fit(X, y)
                    probs = model.predict_proba(X)[:,1]
                    st.write("Coefficients :", dict(zip(predictors_log, model.coef_[0])))
                    save_result({"type":"logistic_regression","target":target_log,"coef":dict(zip(predictors_log, model.coef_[0]))})
                except Exception as e:
                    st.error(f"Erreur regression logistique : {e}")
    else:
        st.info("Aucune variable binaire détectée pour la régression logistique.")

    # --- PART F: PCA (option) ---
    st.subheader("6) PCA (option)")
    if len(num_vars) > 1:
        key_pca_chk = "pca_checkbox"
        if st.checkbox("Exécuter PCA", key=key_pca_chk):
            try:
                X = df[num_vars].dropna()
                Xs = StandardScaler().fit_transform(X)
                pca = PCA()
                comps = pca.fit_transform(Xs)
                explained = pca.explained_variance_ratio_.cumsum()
                n_comp = int(np.searchsorted(explained, 0.8) + 1)
                st.write(f"{n_comp} composantes expliquent ~80% de la variance")
                loadings = pd.DataFrame(pca.components_.T, index=num_vars, columns=[f"PC{i+1}" for i in range(len(num_vars))])
                st.dataframe(loadings.iloc[:, :n_comp])
                fig, ax = plt.subplots()
                ax.scatter(comps[:,0], comps[:,1], alpha=0.6)
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Projection individus (PC1 vs PC2)")
                st.pyplot(fig)
                save_result({"type":"pca","n_comp":n_comp,"explained_var": explained.tolist()})
            except Exception as e:
                st.error(f"Erreur PCA : {e}")

    # --- PART G: MCA (optionnel si prince installé) ---
    st.subheader("7) MCA (pour variables catégorielles, optionnel)")
    if len(cat_vars) > 1:
        if st.checkbox("Exécuter MCA (prince requis)", key="mca_checkbox"):
            try:
                import prince
                df_cat = df[cat_vars].fillna("Missing")
                mca = prince.MCA(n_components=2, random_state=42)
                mca = mca.fit(df_cat)
                coords = mca.column_coordinates(df_cat)
                ind_coords = mca.row_coordinates(df_cat)
                fig, ax = plt.subplots()
                ax.scatter(ind_coords[0], ind_coords[1], alpha=0.6)
                st.pyplot(fig)
                save_result({"type":"mca","coords_shape": coords.shape})
            except ImportError:
                st.error("Le module 'prince' n'est pas installé. pip install prince")
            except Exception as e:
                st.error(f"Erreur MCA : {e}")

    st.success("✅ Interface de tests terminée — résultats stockés dans st.session_state['tests_results']")

    # Option: afficher et proposer téléchargement des résultats
    if st.session_state.get("tests_results"):
        st.subheader("Historique des résultats (session)")
        st.dataframe(pd.DataFrame(st.session_state["tests_results"]))
        if st.button("Télécharger résultats (CSV)"):
            out_df = pd.DataFrame(st.session_state["tests_results"])
            st.download_button("Télécharger CSV", out_df.to_csv(index=False).encode('utf-8'), file_name="tests_results.csv", mime="text/csv")
