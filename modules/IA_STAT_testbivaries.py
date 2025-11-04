import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def propose_tests_bivaries(df, types_df, distribution_df, mots_cles=None, apparie_dict=None):
    """
    Exécution automatique des tests bivariés pour Streamlit.
    
    Arguments :
        df : DataFrame principal
        types_df : DataFrame avec colonnes 'variable' et 'type' ('numérique', 'catégorielle', 'binaire')
        distribution_df : DataFrame avec colonnes 'variable' et 'verdict' (Normal / Non normal)
        mots_cles : liste de mots-clés (optionnel, pour PubMed ou rapport)
        apparie_dict : dictionnaire optionnel {'var_num_vs_var_cat': True/False} pour tests appariés
    Retour :
        results_list : liste de DataFrames, un par test exécuté
    """

    # Normalisation des colonnes de types_df
    rename_dict = {col: col.lower() for col in types_df.columns}
    types_df = types_df.rename(columns=rename_dict)
    types_df.rename(columns={"var": "variable", "variable_name": "variable",
                             "nom": "variable", "column": "variable",
                             "var_type": "type", "type_var": "type",
                             "variable_type": "type", "kind": "type"}, inplace=True)

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    results_list = []

    # -------------------------------
    # Numérique vs Catégorielle
    # -------------------------------
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable'] == num, 'verdict'].values[0]

        if n_modalites == 2:
            test_name = "t-test" if verdict=="Normal" else "Mann-Whitney"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict=="Normal" else "Kruskal-Wallis"
        else:
            test_name = "unknown"

        groupes = df.groupby(cat)[num].apply(list)
        stat, p = None, None

        # Déterminer apparié/non apparié pour ce test
        key = f"{num}_vs_{cat}"
        apparie = apparie_dict.get(key, False) if apparie_dict else False

        try:
            if test_name == "t-test":
                stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "Mann-Whitney":
                stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
            elif test_name == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)

            # Graphique
            fig, ax = plt.subplots()
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"{test_name} : {num} vs {cat}")
            plt.close(fig)  # fermeture pour Streamlit

        except Exception as e:
            print(f"Erreur t-test / Mann-Whitney / ANOVA / Kruskal {num} vs {cat} : {e}")

        # Tableau résultat par test
        result_df = pd.DataFrame([{
            "Variable_num": num,
            "Variable_cat": cat,
            "Test": test_name,
            "Apparié": apparie,
            "Statistique": stat,
            "p-value": p
        }])
        results_list.append({"result_df": result_df, "fig": fig})

    # -------------------------------
    # Numérique vs Numérique
    # -------------------------------
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        try:
            if test_type=="Pearson":
                stat, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
            else:
                stat, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())

            # Graphique
            fig, ax = plt.subplots()
            sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
            ax.set_title(f"{test_type} corrélation : {var1} vs {var2}")
            plt.close(fig)

        except Exception as e:
            print(f"Erreur corrélation {var1} vs {var2} : {e}")

        result_df = pd.DataFrame([{
            "Variable_num1": var1,
            "Variable_num2": var2,
            "Test": f"Corrélation ({test_type})",
            "Statistique": stat,
            "p-value": p
        }])
        results_list.append({"result_df": result_df, "fig": fig})

    # -------------------------------
    # Catégorielle vs Catégorielle
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

            # Graphique
            fig, ax = plt.subplots()
            sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            ax.set_title(f"{test_name} : {var1} vs {var2}")
            plt.close(fig)

        except Exception as e:
            print(f"Erreur Chi² / Fisher {var1} vs {var2} : {e}")

        result_df = pd.DataFrame([{
            "Variable_cat1": var1,
            "Variable_cat2": var2,
            "Test": test_name,
            "Statistique": stat,
            "p-value": p
        }])
        results_list.append({"result_df": result_df, "fig": fig})

    return results_list
