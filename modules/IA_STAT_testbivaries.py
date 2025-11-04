import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def propose_tests_bivariés(df, types_df, distribution_df, mots_cles=None):
    """
    Génère tous les tests bivariés (numérique vs cat / num vs num / cat vs cat)
    Chaque test renvoie un dict avec tableau, graphique et fonction de recalcul pour apparié/non apparié
    """
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    tests = []

    # -------------------------------
    # 1️⃣ Numérique vs Catégoriel
    # -------------------------------
    for num, cat in [(n, c) for n in num_vars for c in cat_vars]:
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

        def recalc(apparie=False, num=num, cat=cat, test_name=test_name):
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
            except Exception as e:
                print(f"Erreur {num} vs {cat} : {e}")

            result_df = pd.DataFrame({"Variable_num": [num],
                                      "Variable_cat": [cat],
                                      "Test": [test_name],
                                      "Statistique": [stat],
                                      "p-value": [p]})
            fig, ax = plt.subplots()
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"{test_name} : {num} vs {cat}")
            plt.close(fig)
            return result_df, fig

        result_df, fig = recalc(apparie=False)
        tests.append({
            "var_num": num,
            "var_cat": cat,
            "test_name": test_name,
            "test_type": test_name,
            "apparie": False,
            "result_df": result_df,
            "fig": fig,
            "recalc_func": recalc
        })

    # -------------------------------
    # 2️⃣ Deux variables numériques
    # -------------------------------
    for var1, var2 in [(v1, v2) for v1, v2 in itertools.combinations(num_vars, 2)]:
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        def recalc_num(apparie=None, var1=var1, var2=var2, test_type=test_type):
            if test_type == "Pearson":
                stat, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
            else:
                stat, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
            result_df = pd.DataFrame({"Variable_num1":[var1], "Variable_num2":[var2],
                                      "Test":[f"Corrélation ({test_type})"], "Statistique":[stat], "p-value":[p]})
            fig, ax = plt.subplots()
            sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
            ax.set_title(f"Corrélation ({test_type}) : {var1} vs {var2}")
            plt.close(fig)
            return result_df, fig

        result_df, fig = recalc_num()
        tests.append({
            "var1": var1,
            "var2": var2,
            "test_name": f"Corrélation ({test_type})",
            "test_type": f"Corrélation ({test_type})",
            "result_df": result_df,
            "fig": fig,
            "recalc_func": recalc_num
        })

    # -------------------------------
    # 3️⃣ Deux variables catégorielles
    # -------------------------------
    for var1, var2 in [(v1, v2) for v1, v2 in itertools.combinations(cat_vars, 2)]:
        def recalc_cat(var1=var1, var2=var2):
            contingency_table = pd.crosstab(df[var1], df[var2])
            try:
                if contingency_table.size <= 4:
                    stat, p = stats.fisher_exact(contingency_table)
                    test_name = "Fisher exact"
                else:
                    stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                    test_name = "Chi²"
            except Exception as e:
                print(f"Erreur {var1} vs {var2} : {e}")
            result_df = pd.DataFrame({"Variable_cat1":[var1], "Variable_cat2":[var2],
                                      "Test":[test_name], "Statistique":[stat], "p-value":[p]})
            fig, ax = plt.subplots()
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            ax.set_title(f"{test_name} : {var1} vs {var2}")
            plt.close(fig)
            return result_df, fig

        result_df, fig = recalc_cat()
        tests.append({
            "var1": var1,
            "var2": var2,
            "test_name": "Chi² / Fisher",
            "test_type": "Chi² / Fisher",
            "result_df": result_df,
            "fig": fig,
            "recalc_func": recalc_cat
        })

    return tests
