import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import prince  # pour ACM (analyse des correspondances multiples)

def propose_tests_interactif_auto(df, types_df, distribution_df, mots_cles=None, apparie=False):
    """
    Version automatique de IA_STAT_interactif2 : exécute tous les tests statistiques
    sans interaction utilisateur.
    Retourne un (DataFrame résumé, liste de résultats détaillés).
    """
    results = []
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    # ======================
    # 1️⃣ NUMÉRIQUE vs CATÉGORIEL
    # ======================
    for num, cat in itertools.product(num_vars, cat_vars):
        try:
            n_modalites = df[cat].dropna().nunique()
            verdict = distribution_df.loc[distribution_df['variable'] == num, 'verdict'].values[0]
            groupes = df.groupby(cat)[num].apply(list)
            if len(groupes) < 2:
                continue

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

            # Exécution du test
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

            results.append({
                "type": "num_vs_cat",
                "variables": (num, cat),
                "test": test_name,
                "stat": stat,
                "pvalue": p,
                "significatif": p < 0.05,
                "figure": fig,
                "interpretation": (
                    f"Différences significatives entre les groupes de '{cat}' sur '{num}'."
                    if p < 0.05 else f"Aucune différence significative détectée entre groupes de '{cat}'."
                )
            })
        except Exception as e:
            results.append({"type": "num_vs_cat", "variables": (num, cat), "erreur": str(e)})

    # ======================
    # 2️⃣ NUMÉRIQUE vs NUMÉRIQUE (corrélation + régression)
    # ======================
    for var1, var2 in itertools.combinations(num_vars, 2):
        try:
            verdict1 = distribution_df.loc[distribution_df['variable'] == var1, 'verdict'].values[0]
            verdict2 = distribution_df.loc[distribution_df['variable'] == var2, 'verdict'].values[0]
            test_type = "Pearson" if verdict1 == "Normal" and verdict2 == "Normal" else "Spearman"

            x, y = df[var1].dropna(), df[var2].dropna()
            corr, p = (stats.pearsonr(x, y) if test_type == "Pearson"
                       else stats.spearmanr(x, y))

            # Graphique
            fig, ax = plt.subplots()
            sns.regplot(x=var1, y=var2, data=df, ax=ax)
            ax.set_title(f"Corrélation ({test_type}) : {var1} vs {var2}")

            results.append({
                "type": "num_vs_num",
                "variables": (var1, var2),
                "test": f"Corrélation {test_type}",
                "stat": corr,
                "pvalue": p,
                "significatif": p < 0.05,
                "figure": fig,
                "interpretation": (
                    f"{var1} et {var2} sont corrélés (r={corr:.2f})."
                    if p < 0.05 else f"Aucune corrélation significative entre {var1} et {var2}."
                )
            })

            # Régression linéaire
            model = LinearRegression().fit(x.values.reshape(-1, 1), y)
            r2 = model.score(x.values.reshape(-1, 1), y)
            results.append({
                "type": "regression",
                "variables": (var1, var2),
                "test": "Régression linéaire",
                "stat": r2,
                "pvalue": None,
                "significatif": r2 > 0.3,
                "figure": fig,
                "interpretation": f"R² = {r2:.3f} → {var1} explique {r2*100:.1f}% de la variance de {var2}."
            })
        except Exception as e:
            results.append({"type": "num_vs_num", "variables": (var1, var2), "erreur": str(e)})

    # ======================
    # 3️⃣ CATÉGORIEL vs CATÉGORIEL
    # ======================
    for var1, var2 in itertools.combinations(cat_vars, 2):
        try:
            contingency = pd.crosstab(df[var1], df[var2])
            if contingency.size <= 4:
                stat, p = stats.fisher_exact(contingency)
                test_name = "Fisher exact"
            else:
                stat, p, _, _ = stats.chi2_contingency(contingency)
                test_name = "Chi²"

            fig, ax = plt.subplots()
            sns.heatmap(contingency, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            ax.set_title(f"{test_name} : {var1} vs {var2}")

            results.append({
                "type": "cat_vs_cat",
                "variables": (var1, var2),
                "test": test_name,
                "stat": stat,
                "pvalue": p,
                "significatif": p < 0.05,
                "figure": fig,
                "interpretation": (
                    f"Association significative entre '{var1}' et '{var2}'."
                    if p < 0.05 else f"Aucune association significative entre '{var1}' et '{var2}'."
                )
            })
        except Exception as e:
            results.append({"type": "cat_vs_cat", "variables": (var1, var2), "erreur": str(e)})

    # ======================
    # 4️⃣ ACP (PCA)
    # ======================
    try:
        if len(num_vars) >= 2:
            X = df[num_vars].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)

            fig, ax = plt.subplots()
            sns.scatterplot(x=components[:, 0], y=components[:, 1])
            ax.set_title("Analyse en composantes principales (ACP)")

            results.append({
                "type": "ACP",
                "test": "PCA",
                "stat": pca.explained_variance_ratio_.sum(),
                "pvalue": None,
                "significatif": True,
                "figure": fig,
                "interpretation": f"Les deux premières composantes expliquent {pca.explained_variance_ratio_.sum()*100:.1f}% de la variance totale."
            })
    except Exception as e:
        results.append({"type": "ACP", "erreur": str(e)})

    # ======================
    # 5️⃣ ACM (MCA)
    # ======================
    try:
        if len(cat_vars) >= 2:
            mca = prince.MCA(n_components=2, random_state=42)
            mca = mca.fit(df[cat_vars].astype(str))
            coords = mca.row_coordinates(df[cat_vars].astype(str))

            fig, ax = plt.subplots()
            sns.scatterplot(x=coords[0], y=coords[1])
            ax.set_title("Analyse des correspondances multiples (ACM)")

            results.append({
                "type": "ACM",
                "test": "MCA",
                "stat": mca.explained_inertia_.sum(),
                "pvalue": None,
                "significatif": True,
                "figure": fig,
                "interpretation": f"Les deux premières dimensions expliquent {mca.explained_inertia_.sum()*100:.1f}% de l'inertie totale."
            })
    except Exception as e:
        results.append({"type": "ACM", "erreur": str(e)})

    # ======================
    # 6️⃣ RÉSUMÉ FINAL
    # ======================
    summary = pd.DataFrame([{
        "Test": r.get("test"),
        "Variables": f"{r.get('variables', ('', ''))[0]} vs {r.get('variables', ('', ''))[1]}",
        "Statistique": round(r.get("stat", np.nan), 4) if "stat" in r else None,
        "p-value": round(r.get("pvalue", np.nan), 4) if "pvalue" in r and r["pvalue"] is not None else None,
        "Significatif": r.get("significatif", False),
        "Interprétation": r.get("interpretation", r.get("erreur", ""))
    } for r in results])

    return summary, results
