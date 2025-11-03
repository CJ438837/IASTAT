import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

def propose_tests_interactif_auto(types_df, distribution_df, df, mots_cles=None, apparie=False, execute_regression=True, execute_pca=True, execute_mca=True):
    """
    Exécution automatique de tous les tests statistiques pour Streamlit.
    Arguments :
        types_df : DataFrame avec colonnes 'variable' et 'type'
        distribution_df : DataFrame avec colonnes 'variable' et 'verdict' (Normal / Non normal)
        df : DataFrame principal
        mots_cles : liste de mots-clés (optionnel)
        apparie : bool, True si les tests à 2 groupes doivent être appariés
        execute_regression : bool, True pour exécuter régression linéaire
        execute_pca : bool, True pour exécuter PCA
        execute_mca : bool, True pour exécuter MCA
    Retour :
        summary_df : DataFrame récapitulatif des tests
        all_results : dictionnaire avec résultats détaillés
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
        raise ValueError(f"Le tableau des types doit contenir les colonnes {expected_cols}, "
                         f"colonnes actuelles : {types_df.columns.tolist()}")

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    summary_records = []
    all_results = {}

    # -------------------------------
    # 1️⃣ Numérique vs Catégoriel
    # -------------------------------
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable'] == num, 'verdict'].values[0]

        if n_modalites == 2:
            if verdict == "Normal":
                test_name = "t-test"
            else:
                test_name = "Mann-Whitney"
        elif n_modalites > 2:
            if verdict == "Normal":
                test_name = "ANOVA"
            else:
                test_name = "Kruskal-Wallis"
        else:
            test_name = "unknown"

        groupes = df.groupby(cat)[num].apply(list)
        stat, p = None, None
        try:
            if test_name == "t-test":
                stat, p = (stats.ttest_rel(groupes.iloc[0], groupes.iloc[1])
                           if apparie else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1]))
            elif test_name == "Mann-Whitney":
                stat, p = (stats.wilcoxon(groupes.iloc[0], groupes.iloc[1])
                           if apparie else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1]))
            elif test_name == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
            elif test_name == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)

            sns.boxplot(x=cat, y=num, data=df)
            plt.title(f"{test_name} : {num} vs {cat}")
            plt.show()

        except Exception as e:
            print(f"Erreur {num} vs {cat} : {e}")

        summary_records.append({"Variable_num": num, "Variable_cat": cat, "Test": test_name, "Statistique": stat, "p-value": p})
        all_results[f"{num}_vs_{cat}"] = {"stat": stat, "p": p}

    # -------------------------------
    # 2️⃣ Deux variables numériques
    # -------------------------------
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable'] == var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable'] == var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1 == "Normal" and verdict2 == "Normal" else "Spearman"

        if test_type == "Pearson":
            corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
        else:
            corr, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())

        sns.scatterplot(x=var1, y=var2, data=df)
        plt.title(f"Corrélation ({test_type}) : {var1} vs {var2}")
        plt.show()

        summary_records.append({"Variable_num1": var1, "Variable_num2": var2, "Test": f"Corrélation ({test_type})", "Statistique": corr, "p-value": p})
        all_results[f"{var1}_vs_{var2}"] = {"stat": corr, "p": p}

    # -------------------------------
    # 3️⃣ Deux variables catégorielles
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

            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm")
            plt.title(f"{test_name} : {var1} vs {var2}")
            plt.show()

        except Exception as e:
            print(f"Erreur test catégoriel {var1} vs {var2} : {e}")

        summary_records.append({"Variable_cat1": var1, "Variable_cat2": var2, "Test": test_name, "Statistique": stat, "p-value": p})
        all_results[f"{var1}_vs_{var2}"] = {"stat": stat, "p": p}

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

            summary_records.append({"Regression_var_dep": cible_col, "R²": model.score(X_pred, y), "Shapiro_stat": stat, "Shapiro_p": p})
            all_results[f"regression_{cible_col}"] = {"R²": model.score(X_pred, y), "residus_shapiro": (stat, p)}

    # -------------------------------
    # 5️⃣ PCA
    # -------------------------------
    if execute_pca and len(num_vars) > 1:
        X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
        pca = PCA()
        components = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_
        cum_var = explained_variance.cumsum()
        n_comp = (cum_var < 0.8).sum() + 1

        summary_records.append({"PCA_n_comp_80%": n_comp})
        all_results["PCA"] = {"explained_variance": explained_variance, "components": components}

    # -------------------------------
    # 6️⃣ MCA
    # -------------------------------
    if execute_mca and len(cat_vars) > 1:
        try:
            import prince
            df_cat = df[cat_vars].dropna()
            mca = prince.MCA(n_components=2, random_state=42)
            mca = mca.fit(df_cat)
            all_results["MCA"] = {"row_coordinates": mca.row_coordinates(df_cat), "col_coordinates": mca.column_coordinates(df_cat)}
        except ImportError:
            print("Module 'prince' non installé pour MCA")

    # -------------------------------
    # 7️⃣ Régression logistique
    # -------------------------------
    for cat in cat_vars:
        if df[cat].dropna().nunique() == 2:
            X = df[num_vars].dropna()
            y = df[cat].loc[X.index]
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            all_results[f"logistic_{cat}"] = {"coef": dict(zip(num_vars, model.coef_[0])), "intercept": model.intercept_[0]}

    summary_df = pd.DataFrame(summary_records)
    return summary_df, all_results
