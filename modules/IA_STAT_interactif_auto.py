import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

def propose_tests_interactif_auto(
    types_df, distribution_df, df, mots_cles=None,
    apparie=False, execute_regression=True,
    execute_pca=True, execute_mca=True
):
    """
    Exécution automatique de tous les tests statistiques avec tableaux séparés.
    Retourne un dictionnaire de DataFrames par type de test + tous les résultats détaillés.
    """

    # --- Normalisation des colonnes de types_df ---
    rename_dict = {}
    for col in types_df.columns:
        lower = col.lower()
        if lower in ["var", "variable_name", "nom", "column"]:
            rename_dict[col] = "variable"
        elif lower in ["var_type", "type_var", "variable_type", "kind"]:
            rename_dict[col] = "type"
    types_df = types_df.rename(columns=rename_dict)

    expected_cols = {"variable", "type"}
    if not expected_cols.issubset(types_df.columns):
        raise ValueError(f"Le tableau des types doit contenir {expected_cols}, "
                         f"colonnes actuelles : {types_df.columns.tolist()}")

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    all_results = {}
    results_tables = {}

    # Dictionnaires temporaires pour stocker les résultats par test
    ttest_records, anova_records, corr_records, chi2_records = [], [], [], []
    reg_records, pca_records, mca_records, logit_records = [], [], [], []

    # -------------------------------
    # 1️⃣ Numérique vs Catégoriel
    # -------------------------------
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable'] == num, 'verdict'].values[0]
        test_name = None

        if n_modalites == 2:
            test_name = "t-test" if verdict == "Normal" else "Mann-Whitney"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict == "Normal" else "Kruskal-Wallis"

        if not test_name:
            continue

        groupes = df.groupby(cat)[num].apply(list)
        stat, p = None, None
        try:
            if test_name == "t-test":
                stat, p = (stats.ttest_rel(groupes.iloc[0], groupes.iloc[1])
                           if apparie else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1]))
                ttest_records.append({"Variable_num": num, "Variable_cat": cat, "Stat": stat, "p-value": p})
            elif test_name == "Mann-Whitney":
                stat, p = (stats.wilcoxon(groupes.iloc[0], groupes.iloc[1])
                           if apparie else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1]))
                ttest_records.append({"Variable_num": num, "Variable_cat": cat, "Stat": stat, "p-value": p})
            elif test_name == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
                anova_records.append({"Variable_num": num, "Variable_cat": cat, "Stat": stat, "p-value": p})
            elif test_name == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)
                anova_records.append({"Variable_num": num, "Variable_cat": cat, "Stat": stat, "p-value": p})

            # Graphique
            sns.boxplot(x=cat, y=num, data=df)
            plt.title(f"{test_name} : {num} vs {cat}")
            plt.show()

        except Exception as e:
            print(f"Erreur {num} vs {cat} : {e}")

        all_results[f"{num}_vs_{cat}"] = {"test": test_name, "stat": stat, "p": p}

    # -------------------------------
    # 2️⃣ Corrélations entre variables numériques
    # -------------------------------
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable'] == var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable'] == var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1 == "Normal" and verdict2 == "Normal" else "Spearman"

        corr, p = (stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                   if test_type == "Pearson"
                   else stats.spearmanr(df[var1].dropna(), df[var2].dropna()))

        corr_records.append({"Var1": var1, "Var2": var2, "Test": test_type, "Corr": corr, "p-value": p})
        all_results[f"{var1}_vs_{var2}"] = {"test": test_type, "corr": corr, "p": p}

        sns.scatterplot(x=var1, y=var2, data=df)
        plt.title(f"Corrélation ({test_type}) : {var1} vs {var2}")
        plt.show()

    # -------------------------------
    # 3️⃣ Catégoriel vs Catégoriel
    # -------------------------------
    for var1, var2 in itertools.combinations(cat_vars, 2):
        contingency_table = pd.crosstab(df[var1], df[var2])
        try:
            if contingency_table.size <= 4:
                stat, p = stats.fisher_exact(contingency_table)
                test_name = "Fisher exact"
            else:
                stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                test_name = "Chi²"

            chi2_records.append({"Var1": var1, "Var2": var2, "Test": test_name, "Stat": stat, "p-value": p})
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm")
            plt.title(f"{test_name} : {var1} vs {var2}")
            plt.show()

        except Exception as e:
            print(f"Erreur test catégoriel {var1} vs {var2} : {e}")

    # -------------------------------
    # 4️⃣ Régression linéaire multiple
    # -------------------------------
    if execute_regression and len(num_vars) > 1:
        X = df[num_vars].dropna()
        for cible_col in num_vars:
            y = X[cible_col]
            X_pred = X.drop(columns=[cible_col])
            model = LinearRegression()
            model.fit(X_pred, y)
            y_pred = model.predict(X_pred)
            residus = y - y_pred
            stat, p = stats.shapiro(residus)
            reg_records.append({
                "Variable_cible": cible_col,
                "R²": model.score(X_pred, y),
                "Shapiro_p": p
            })
            all_results[f"regression_{cible_col}"] = {"R²": model.score(X_pred, y), "p": p}

    # -------------------------------
    # 5️⃣ PCA
    # -------------------------------
    if execute_pca and len(num_vars) > 1:
        X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
        pca = PCA()
        pca.fit(X_scaled)
        explained_var = pca.explained_variance_ratio_
        cum_var = explained_var.cumsum()
        n_comp = (cum_var < 0.8).sum() + 1
        pca_records.append({"Nb composantes 80% variance": n_comp})
        all_results["PCA"] = {"explained_var": explained_var, "n_comp": n_comp}

    # -------------------------------
    # 6️⃣ MCA
    # -------------------------------
    if execute_mca and len(cat_vars) > 1:
        try:
            import prince
            df_cat = df[cat_vars].dropna()
            mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
            mca_records.append({"Nb catégories": len(cat_vars), "Inertie globale": mca.total_inertia_})
            all_results["MCA"] = {"inertia": mca.total_inertia_}
        except ImportError:
            print("⚠️ Module 'prince' non installé, MCA ignorée")

    # -------------------------------
    # 7️⃣ Régression logistique
    # -------------------------------
    for cat in cat_vars:
        if df[cat].dropna().nunique() == 2:
            X = df[num_vars].dropna()
            y = df[cat].loc[X.index]
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            logit_records.append({"Variable_cible": cat, "Coefficients": dict(zip(num_vars, model.coef_[0]))})
            all_results[f"logit_{cat}"] = {"coef": dict(zip(num_vars, model.coef_[0]))}

    # -------------------------------
    # 8️⃣ Assemblage des résultats
    # -------------------------------
    results_tables = {
        "T-tests & Mann-Whitney": pd.DataFrame(ttest_records),
        "ANOVA & Kruskal-Wallis": pd.DataFrame(anova_records),
        "Corrélations": pd.DataFrame(corr_records),
        "Chi² / Fisher": pd.DataFrame(chi2_records),
        "Régressions linéaires": pd.DataFrame(reg_records),
        "PCA": pd.DataFrame(pca_records),
        "MCA": pd.DataFrame(mca_records),
        "Régressions logistiques": pd.DataFrame(logit_records)
    }

    return results_tables, all_results
