# ===========================================
# 📊 IA-STAT - Tests bivariés avec interprétation automatique
# ===========================================
import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, chi2_contingency, fisher_exact

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSmodels_AVAILABLE = True
except Exception:
    STATSmodels_AVAILABLE = False

plt.style.use('seaborn-v0_8-muted')


# === Fonctions utilitaires ===
def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _cohens_d(x, y, paired=False):
    x = np.asarray(x)
    y = np.asarray(y)
    if paired:
        d = (np.mean(x - y)) / np.std(x - y, ddof=1)
        return float(d)
    nx = len(x)
    ny = len(y)
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = math.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


def _rank_biserial_from_u(u, n1, n2):
    try:
        return 1.0 - (2.0 * u) / (n1 * n2)
    except Exception:
        return np.nan


def _eta_squared_anova(groups):
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
    ss_total = sum((all_vals - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan
    return float(ss_between / ss_total)


def _cramers_v(table):
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    denom = min((r - 1), (k - 1))
    if denom == 0:
        return np.nan
    return float(np.sqrt(phi2 / denom))


def _kendall_tau(x, y):
    try:
        tau, p = stats.kendalltau(x, y)
        return float(tau), float(p)
    except Exception:
        return np.nan, np.nan


def _normality_test(series):
    arr = np.asarray(series.dropna())
    if len(arr) < 3:
        return {"test": "shapiro", "stat": np.nan, "p": np.nan, "normal": False}
    try:
        stat, p = shapiro(arr)
    except Exception:
        stat, p = np.nan, np.nan
    return {"test": "Shapiro-Wilk", "stat": float(stat), "p": float(p),
            "normal": (p is not None and p > 0.05)}


def _levene_test(*groups):
    try:
        stat, p = levene(*groups)
        return float(stat), float(p), (p > 0.05)
    except Exception:
        return np.nan, np.nan, False


# === Interprétation automatique ===
def interpret_bivariate_result(test_name, p_value, effect_size=None, cramers_v=None):
    alpha = 0.05
    if np.isnan(p_value):
        return "⚠️ Résultat non calculable (valeur p manquante)."

    # Significativité
    if p_value < alpha:
        significance = "✅ Résultat significatif (p < 0.05)."
    else:
        significance = "❌ Aucune différence significative (p ≥ 0.05)."

    test = test_name.lower()
    interpretation = significance

    # Taille d'effet
    if "t-test" in test or "anova" in test:
        if effect_size is not None and not np.isnan(effect_size):
            if abs(effect_size) < 0.2:
                eff = "effet très faible"
            elif abs(effect_size) < 0.5:
                eff = "effet faible"
            elif abs(effect_size) < 0.8:
                eff = "effet modéré"
            else:
                eff = "effet fort"
            interpretation += f" Taille d'effet = {effect_size:.2f} ({eff})."

    elif "mann" in test or "wilcoxon" in test:
        interpretation += " Test non paramétrique adapté en cas de non-normalité."

    elif "pearson" in test or "spearman" in test:
        if effect_size is not None and not np.isnan(effect_size):
            interpretation += f" Corrélation de {effect_size:.2f}."
        else:
            interpretation += " Corrélation détectée."

    elif "chi2" in test or "fisher" in test:
        if cramers_v is not None and not np.isnan(cramers_v):
            if cramers_v < 0.1:
                strength = "relation très faible"
            elif cramers_v < 0.3:
                strength = "relation faible"
            elif cramers_v < 0.5:
                strength = "relation modérée"
            else:
                strength = "relation forte"
            interpretation += f" Force de l'association (Cramér's V = {cramers_v:.2f}) → {strength}."

    return interpretation


# === Fonction principale ===
def propose_tests_bivaries(types_df, distribution_df, df,
                           output_folder="tests_bivaries_plots",
                           assume_apparie_map=None,
                           default_apparie=False,
                           verbose=False):

    _ensure_dir(output_folder)
    if assume_apparie_map is None:
        assume_apparie_map = {}

    # Normaliser les noms de colonnes
    rename_dict = {}
    for col in types_df.columns:
        lc = col.lower()
        if lc in {"variable", "var", "nom", "column", "name"}:
            rename_dict[col] = "variable"
        if lc in {"type", "var_type", "variable_type", "kind"}:
            rename_dict[col] = "type"
    types_df = types_df.rename(columns=rename_dict)

    if "variable" not in types_df.columns or "type" not in types_df.columns:
        raise ValueError("types_df must contain 'variable' and 'type' columns")

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].astype(str).tolist()
    cat_vars = types_df[types_df['type'].isin(["catégorielle", "binaire"])]['variable'].astype(str).tolist()

    summary_records = []
    details = {}

    # === Numérique vs Numérique ===
    for v1, v2 in itertools.combinations(num_vars, 2):
        arr1 = df[v1].dropna()
        arr2 = df[v2].dropna()
        norm1 = _normality_test(arr1)
        norm2 = _normality_test(arr2)

        if norm1["normal"] and norm2["normal"]:
            test_name = "Corrélation de Pearson"
            stat, p = stats.pearsonr(arr1, arr2)
        else:
            test_name = "Corrélation de Spearman"
            stat, p = stats.spearmanr(arr1, arr2)

        tau, p_tau = _kendall_tau(arr1, arr2)

        # Graphique
        fig, ax = plt.subplots()
        sns.scatterplot(x=v1, y=v2, data=df, ax=ax)
        ax.set_title(f"{v1} vs {v2} ({test_name})")
        fname = f"{v1}__{v2}_scatter.png"
        fig.savefig(os.path.join(output_folder, fname), bbox_inches="tight")
        plt.close(fig)

        interpretation = interpret_bivariate_result(test_name, p, effect_size=stat)

        summary_records.append({
            "var1": v1, "var2": v2, "test": test_name,
            "stat": stat, "p_value": p, "interpretation": interpretation
        })

    # === Numérique vs Catégoriel ===
    for num, cat in itertools.product(num_vars, cat_vars):
        modalities = df[cat].dropna().unique()
        n_mod = len(modalities)
        groups = [df[df[cat] == m][num].dropna().values for m in modalities]

        lv_stat, lv_p, equal_var = _levene_test(*groups)
        groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in groups])

        if n_mod == 2:
            if groups_norm and equal_var:
                test_name = "t-test"
                stat, p = stats.ttest_ind(*groups, equal_var=True)
                effect = _cohens_d(groups[0], groups[1])
            else:
                test_name = "Mann-Whitney"
                stat, p = stats.mannwhitneyu(*groups)
                effect = _rank_biserial_from_u(stat, len(groups[0]), len(groups[1]))
        else:
            if groups_norm and equal_var:
                test_name = "ANOVA"
                stat, p = stats.f_oneway(*groups)
                effect = _eta_squared_anova(groups)
            else:
                test_name = "Kruskal-Wallis"
                stat, p = stats.kruskal(*groups)
                effect = np.nan

        # Graphique
        fig, ax = plt.subplots()
        sns.boxplot(x=cat, y=num, data=df, ax=ax)
        ax.set_title(f"{num} par {cat} ({test_name})")
        fname = f"{num}__{cat}_boxplot.png"
        fig.savefig(os.path.join(output_folder, fname), bbox_inches="tight")
        plt.close(fig)

        interpretation = interpret_bivariate_result(test_name, p, effect_size=effect)

        summary_records.append({
            "var_num": num, "var_cat": cat, "test": test_name,
            "stat": stat, "p_value": p, "effect": effect, "interpretation": interpretation
        })

    # === Catégoriel vs Catégoriel ===
    for v1, v2 in itertools.combinations(cat_vars, 2):
        table = pd.crosstab(df[v1], df[v2])
        chi2, p_chi, dof, expected = chi2_contingency(table)
        small_expected = (expected < 5).sum() > 0
        if small_expected or table.shape == (2, 2):
            test_name = "Fisher exact" if table.shape == (2, 2) else "Chi2 avec prudence"
            if table.shape == (2, 2):
                stat, p = fisher_exact(table)
            else:
                stat, p = chi2, p_chi
        else:
            test_name = "Chi2"
            stat, p = chi2, p_chi

        cram = _cramers_v(table)

        # Graphique
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
        ax.set_title(f"{v1} vs {v2} ({test_name})")
        fname = f"{v1}__{v2}_heatmap.png"
        fig.savefig(os.path.join(output_folder, fname), bbox_inches="tight")
        plt.close(fig)

        interpretation = interpret_bivariate_result(test_name, p, cramers_v=cram)

        summary_records.append({
            "var1": v1, "var2": v2, "test": test_name,
            "stat": stat, "p_value": p, "cramers_v": cram, "interpretation": interpretation
        })

    summary_df = pd.DataFrame(summary_records)
    return summary_df
