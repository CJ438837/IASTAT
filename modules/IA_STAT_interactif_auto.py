import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

def propose_tests_interactif_auto(types_df, distribution_df, df, mots_cles=None, apparie=False):
    """
    Version non interactive : exécute automatiquement les tests statistiques adaptés.
    Retourne un dictionnaire des résultats.
    """
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()
    results = []

    # --- 1️⃣ Numérique vs Catégoriel ---
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable'] == num, 'verdict'].values[0]

        # Choix du test
        if n_modalites == 2:
            if verdict == "Normal":
                test_name = "t-test apparié" if apparie else "t-test indépendant"
            else:
                test_name = "Wilcoxon" if apparie else "Mann-Whitney"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict == "Normal" else "Kruskal-Wallis"
        else:
            continue

        groupes = df.groupby(cat)[num].apply(list)

        try:
            if test_name == "t-test apparié":
                stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "t-test indépendant":
                stat, p = stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "Wilcoxon":
                stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "Mann-Whitney":
                stat, p = stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
            elif test_name == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)
            else:
                continue

            # Graphique
            fig, ax = plt.subplots()
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"{test_name} : {num} vs {cat}")

            # Résumé
            results.append({
                "type": "num_vs_cat",
                "variables": (num, cat),
                "test": test_name,
                "stat": stat,
                "pvalue": p,
                "significatif": p < 0.05,
                "interpretation": (
                    f"La variable '{num}' a un impact significatif sur '{cat}'."
                    if p < 0.05 else f"Aucun impact significatif entre '{num}' et '{cat}'."
                ),
                "figure": fig
            })

        except Exception as e:
            results.append({
                "type": "num_vs_cat",
                "variables": (num, cat),
                "test": test_name,
                "erreur": str(e)
            })

    # --- 2️⃣ Deux variables continues ---
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable'] == var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable'] == var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1 == "Normal" and verdict2 == "Normal" else "Spearman"

        x, y = df[var1].dropna(), df[var2].dropna()
        corr, p = (stats.pearsonr(x, y) if test_type == "Pearson"
                   else stats.spearmanr(x, y))

        fig, ax = plt.subplots()
        sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
        ax.set_title(f"Corrélation ({test_type}) : {var1} vs {var2}")

        results.append({
            "type": "num_vs_num",
            "variables": (var1, var2),
            "test": f"Corrélation {test_type}",
            "stat": corr,
            "pvalue": p,
            "significatif": p < 0.05,
            "interpretation": (
                f"{var1} et {var2} sont significativement corrélés."
                if p < 0.05 else f"Aucune corrélation significative entre {var1} et {var2}."
            ),
            "figure": fig
        })

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

            results.append({
                "type": "cat_vs_cat",
                "variables": (var1, var2),
                "test": test_name,
                "stat": stat,
                "pvalue": p,
                "significatif": p < 0.05,
                "interpretation": (
                    f"'{var1}' dépend significativement de '{var2}'."
                    if p < 0.05 else f"Aucune dépendance significative entre '{var1}' et '{var2}'."
                ),
                "figure": fig
            })
        except Exception as e:
            results.append({
                "type": "cat_vs_cat",
                "variables": (var1, var2),
                "test": "Chi²/Fisher",
                "erreur": str(e)
            })

    # --- 4️⃣ Résumé final ---
    return pd.DataFrame([{
        "Test": r["test"],
        "Variables": f"{r['variables'][0]} vs {r['variables'][1]}",
        "Statistique": round(r.get("stat", np.nan), 4) if "stat" in r else None,
        "p-value": round(r.get("pvalue", np.nan), 4) if "pvalue" in r else None,
        "Significatif": r.get("significatif", False),
        "Interprétation": r.get("interpretation", r.get("erreur", "")),
    } for r in results]), results
