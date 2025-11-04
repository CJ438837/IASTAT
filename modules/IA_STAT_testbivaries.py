import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def propose_tests_bivariés(df, types_df, distribution_df, mots_cles=None):
    """
    Exécution des tests bivariés pour Streamlit.
    Retour : liste de dicts avec 'result_df' et 'fig' pour chaque test.
    """
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    results_list = []

    # --- Numérique vs Catégoriel ---
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]

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

        # --- Demande d'appariement si pertinent ---
        apparie = False
        if test_name in ["t-test", "Mann-Whitney"]:
            apparie = None  # la page Streamlit doit demander

        groupes = df.groupby(cat)[num].apply(list)
        stat, p = None, None
        try:
            if test_name == "t-test":
                if apparie is True:
                    stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1])
                elif apparie is False:
                    stat, p = stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "Mann-Whitney":
                if apparie is True:
                    stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1])
                elif apparie is False:
                    stat, p = stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
            elif test_name == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)

            # --- Figure ---
            fig, ax = plt.subplots()
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"{test_name} : {num} vs {cat}")

            # --- Résultat DataFrame ---
            result_df = pd.DataFrame([{
                "Variable_num": num,
                "Variable_cat": cat,
                "Test": test_name,
                "Apparié": apparie,
                "Statistique": stat,
                "p-value": p
            }])

            results_list.append({"result_df": result_df, "fig": fig, "test_name": test_name,
                                 "num": num, "cat": cat, "apparie_needed": test_name in ["t-test", "Mann-Whitney"]})

        except Exception as e:
            print(f"Erreur {num} vs {cat} : {e}")

    # --- 2️⃣ Deux variables numériques ---
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        corr, p = (stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                    if test_type=="Pearson" else stats.spearmanr(df[var1].dropna(), df[var2].dropna()))

        fig, ax = plt.subplots()
        sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
        ax.set_title(f"Corrélation ({test_type}) : {var1} vs {var2}")

        result_df = pd.DataFrame([{
            "Variable_num1": var1,
            "Variable_num2": var2,
            "Test": f"Corrélation ({test_type})",
            "Statistique": corr,
            "p-value": p
        }])

        results_list.append({"result_df": result_df, "fig": fig, "test_name": f"Corrélation ({test_type})",
                             "var1": var1, "var2": var2, "apparie_needed": False})

    # --- 3️⃣ Deux variables catégorielles ---
    for var1, var2 in itertools.combinations(cat_vars, 2):
        contingency_table = pd.crosstab(df[var1], df[var2])
        try:
            if contingency_table.size <= 4:
                stat, p = stats.fisher_exact(contingency_table)
                test_name = "Fisher exact"
            else:
                stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                test_name = "Chi²"

            fig, ax = plt.subplots()
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            ax.set_title(f"{test_name} : {var1} vs {var2}")

            result_df = pd.DataFrame([{
                "Variable_cat1": var1,
                "Variable_cat2": var2,
                "Test": test_name,
                "Statistique": stat,
                "p-value": p
            }])

            results_list.append({"result_df": result_df, "fig": fig, "test_name": test_name,
                                 "var1": var1, "var2": var2, "apparie_needed": False})
        except Exception as e:
            print(f"Erreur {var1} vs {var2} : {e}")

    return results_list
