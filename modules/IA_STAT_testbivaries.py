# modules/IA_STAT_bivarie_auto.py
import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def propose_tests_bivariés(df, types_df, distribution_df):
    """
    Propose et exécute tous les tests bivariés automatiquement pour Streamlit.
    Retourne une liste de dictionnaires contenant pour chaque test :
        - result_df : DataFrame du résultat
        - fig : figure matplotlib du test
        - test_name : nom du test
        - num, cat : variables testées
        - groupes : données groupées pour test apparié/non
        - apparie_needed : bool indiquant si choix apparié est nécessaire
    """
    # Normalisation colonnes
    types_df = types_df.rename(columns={col: col.lower() for col in types_df.columns})
    if 'variable' not in types_df.columns or 'type' not in types_df.columns:
        raise ValueError("Le tableau des types doit contenir 'variable' et 'type'")
    
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    results_list = []

    # 1️⃣ Numérique vs Catégoriel
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]

        if n_modalites == 2:
            test_name = "t-test" if verdict=="Normal" else "Mann-Whitney"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict=="Normal" else "Kruskal-Wallis"
        else:
            test_name = "unknown"

        groupes = df.groupby(cat)[num].apply(list)
        apparie_needed = test_name in ["t-test","Mann-Whitney"]
        stat, p = (None, None) if apparie_needed else (None, None)

        fig, ax = plt.subplots()
        sns.boxplot(x=cat, y=num, data=df, ax=ax)
        ax.set_title(f"{test_name} : {num} vs {cat}")

        result_df = pd.DataFrame([{
            "Variable_num": num,
            "Variable_cat": cat,
            "Test": test_name,
            "Apparié": None,
            "Statistique": stat,
            "p-value": p
        }])

        results_list.append({
            "result_df": result_df,
            "fig": fig,
            "test_name": test_name,
            "num": num,
            "cat": cat,
            "groupes": groupes,
            "apparie_needed": apparie_needed
        })

    # 2️⃣ Numérique vs Numérique
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1,'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2,'verdict'].values[0]
        test_name = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        if test_name=="Pearson":
            stat, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
        else:
            stat, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())

        fig, ax = plt.subplots()
        sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
        ax.set_title(f"Corrélation ({test_name}) : {var1} vs {var2}")

        result_df = pd.DataFrame([{
            "Variable_num1": var1,
            "Variable_num2": var2,
            "Test": f"Corrélation ({test_name})",
            "Statistique": stat,
            "p-value": p
        }])

        results_list.append({
            "result_df": result_df,
            "fig": fig,
            "test_name": test_name,
            "num": var1,
            "cat": var2,  # juste pour cohérence
            "groupes": None,
            "apparie_needed": False
        })

    # 3️⃣ Catégoriel vs Catégoriel
    for var1, var2 in itertools.combinations(cat_vars, 2):
        contingency_table = pd.crosstab(df[var1], df[var2])
        try:
            if contingency_table.size <= 4:
                stat, p = stats.fisher_exact(contingency_table)
                test_name = "Fisher exact"
            else:
                stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                test_name = "Chi²"
        except Exception as e:
            stat, p = None, None
            print(f"Erreur {var1} vs {var2} : {e}")

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

        results_list.append({
            "result_df": result_df,
            "fig": fig,
            "test_name": test_name,
            "num": var1,
            "cat": var2,
            "groupes": None,
            "apparie_needed": False
        })

    return results_list
