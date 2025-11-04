import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# =====================================================================
# 🧠 1. ANOVA
# =====================================================================
def propose_tests_interactif_auto_anova(types_df, distribution_df, df, mots_cles, apparie=False):
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    results = []
    for num_var in num_vars:
        for cat_var in cat_vars:
            try:
                model = ols(f"{num_var} ~ C({cat_var})", data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                pval = anova_table["PR(>F)"][0]
                results.append({
                    "Test": "ANOVA (apparié)" if apparie else "ANOVA",
                    "Variable numérique": num_var,
                    "Variable catégorielle": cat_var,
                    "p-value": round(pval, 5)
                })
            except Exception as e:
                results.append({
                    "Test": "ANOVA",
                    "Variable numérique": num_var,
                    "Variable catégorielle": cat_var,
                    "Erreur": str(e)
                })

    summary_df = pd.DataFrame(results)
    return summary_df, results


# =====================================================================
# 🧮 2. Kruskal-Wallis
# =====================================================================
def propose_tests_interactif_auto_kruskal(types_df, distribution_df, df, mots_cles, apparie=False):
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    results = []
    for num_var in num_vars:
        for cat_var in cat_vars:
            try:
                groups = [group[num_var].dropna() for name, group in df.groupby(cat_var)]
                stat, pval = stats.kruskal(*groups)
                results.append({
                    "Test": "Kruskal-Wallis (apparié)" if apparie else "Kruskal-Wallis",
                    "Variable numérique": num_var,
                    "Variable catégorielle": cat_var,
                    "p-value": round(pval, 5)
                })
            except Exception as e:
                results.append({
                    "Test": "Kruskal-Wallis",
                    "Variable numérique": num_var,
                    "Variable catégorielle": cat_var,
                    "Erreur": str(e)
                })

    summary_df = pd.DataFrame(results)
    return summary_df, results


# =====================================================================
# 📊 3. t-test (Student)
# =====================================================================
def propose_tests_interactif_auto_ttest(types_df, distribution_df, df, mots_cles, apparie=False):
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    bin_vars = types_df[types_df['type'] == "binaire"]['variable'].tolist()

    results = []
    for num_var in num_vars:
        for bin_var in bin_vars:
            try:
                groups = [group[num_var].dropna() for name, group in df.groupby(bin_var)]
                if len(groups) == 2:
                    if apparie:
                        stat, pval = stats.ttest_rel(groups[0], groups[1])
                    else:
                        stat, pval = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    results.append({
                        "Test": "t-test apparié" if apparie else "t-test",
                        "Variable numérique": num_var,
                        "Variable binaire": bin_var,
                        "p-value": round(pval, 5)
                    })
            except Exception as e:
                results.append({
                    "Test": "t-test",
                    "Variable numérique": num_var,
                    "Variable binaire": bin_var,
                    "Erreur": str(e)
                })

    summary_df = pd.DataFrame(results)
    return summary_df, results


# =====================================================================
# ⚖️ 4. Mann-Whitney U
# =====================================================================
def propose_tests_interactif_auto_mannwhitney(types_df, distribution_df, df, mots_cles, apparie=False):
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    bin_vars = types_df[types_df['type'] == "binaire"]['variable'].tolist()

    results = []
    for num_var in num_vars:
        for bin_var in bin_vars:
            try:
                groups = [group[num_var].dropna() for name, group in df.groupby(bin_var)]
                if len(groups) == 2:
                    stat, pval = stats.mannwhitneyu(groups[0], groups[1])
                    results.append({
                        "Test": "Mann-Whitney (apparié)" if apparie else "Mann-Whitney",
                        "Variable numérique": num_var,
                        "Variable binaire": bin_var,
                        "p-value": round(pval, 5)
                    })
            except Exception as e:
                results.append({
                    "Test": "Mann-Whitney",
                    "Variable numérique": num_var,
                    "Variable binaire": bin_var,
                    "Erreur": str(e)
                })

    summary_df = pd.DataFrame(results)
    return summary_df, results


# =====================================================================
# 🔢 5. Khi²
# =====================================================================
def propose_tests_interactif_auto_chi2(types_df, distribution_df, df, mots_cles):
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    results = []
    for i, var1 in enumerate(cat_vars):
        for var2 in cat_vars[i+1:]:
            try:
                contingency = pd.crosstab(df[var1], df[var2])
                chi2, pval, _, _ = stats.chi2_contingency(contingency)
                results.append({
                    "Test": "Chi²",
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "p-value": round(pval, 5)
                })
            except Exception as e:
                results.append({
                    "Test": "Chi²",
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "Erreur": str(e)
                })

    summary_df = pd.DataFrame(results)
    return summary_df, results


# =====================================================================
# 🔗 6. Corrélations (Pearson/Spearman)
# =====================================================================
def propose_tests_interactif_auto_correlation(types_df, distribution_df, df, mots_cles):
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()

    results = []
    for i, var1 in enumerate(num_vars):
        for var2 in num_vars[i+1:]:
            try:
                corr_pearson, p_pearson = stats.pearsonr(df[var1], df[var2])
                corr_spearman, p_spearman = stats.spearmanr(df[var1], df[var2])
                results.append({
                    "Test": "Corrélation",
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "Pearson r": round(corr_pearson, 3),
                    "p Pearson": round(p_pearson, 5),
                    "Spearman rho": round(corr_spearman, 3),
                    "p Spearman": round(p_spearman, 5)
                })
            except Exception as e:
                results.append({
                    "Test": "Corrélation",
                    "Variable 1": var1,
                    "Variable 2": var2,
                    "Erreur": str(e)
                })

    summary_df = pd.DataFrame(results)
    return summary_df, results
