# modules/IA_STAT_testbivaries.py (fonction mise à jour)
# ===========================================
# 📊 IA-STAT - Tests bivariés avec interprétation automatique (corrigé)
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
    return {"test": "Shapiro-Wilk", "stat": float(stat) if not np.isnan(stat) else np.nan,
            "p": float(p) if not np.isnan(p) else np.nan,
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
    if pd.isna(p_value):
        return "⚠️ Résultat non calculable (valeur p manquante)."

    # Significativité
    if p_value < alpha:
        significance = "✅ Résultat significatif (p < 0.05)."
    else:
        significance = "❌ Aucune différence significative (p ≥ 0.05)."

    test = str(test_name).lower()
    interpretation = significance

    # Taille d'effet
    if "t-test" in test or "anova" in test:
        if effect_size is not None and not pd.isna(effect_size):
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
        if effect_size is not None and not pd.isna(effect_size):
            interpretation += f" Corrélation = {effect_size:.2f}."
        else:
            interpretation += " Corrélation détectée."

    elif "chi2" in test or "fisher" in test:
        if cramers_v is not None and not pd.isna(cramers_v):
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
        key = f"{v1}__{v2}"
        # align on same rows to avoid shape mismatch
        pair_df = df[[v1, v2]].dropna()
        arr1 = pair_df[v1]
        arr2 = pair_df[v2]

        norm1 = _normality_test(arr1)
        norm2 = _normality_test(arr2)

        try:
            if norm1["normal"] and norm2["normal"]:
                test_name = "Corrélation de Pearson"
                stat, p = stats.pearsonr(arr1, arr2)
            else:
                test_name = "Corrélation de Spearman"
                stat, p = stats.spearmanr(arr1, arr2)
        except Exception:
            stat, p = (np.nan, np.nan)

        tau, p_tau = _kendall_tau(arr1, arr2)

        # Graphique
        plot_path = None
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=arr1, y=arr2, ax=ax)
            ax.set_xlabel(v1)
            ax.set_ylabel(v2)
            ax.set_title(f"{v1} vs {v2} ({test_name})")
            fname = f"{key}_scatter.png"
            plot_path = os.path.join(output_folder, fname)
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plot_path = None

        interpretation = interpret_bivariate_result(test_name, p, effect_size=stat)

        summary_records.append({
            "test_id": key, "test": test_name, "var1": v1, "var2": v2,
            "stat": stat, "p_value": p, "interpretation": interpretation
        })
        details[key] = {
            "type": "num-num",
            "var1": v1, "var2": v2,
            "test": test_name,
            "statistic": float(stat) if not pd.isna(stat) else np.nan,
            "p_value": float(p) if not pd.isna(p) else np.nan,
            "kendall_tau": tau,
            "kendall_p": p_tau,
            "plot": plot_path,
            "interpretation": interpretation,
            "normality_var1": norm1,
            "normality_var2": norm2
        }

    # === Numérique vs Catégoriel ===
    for num, cat in itertools.product(num_vars, cat_vars):
        key = f"{num}__{cat}"
        modalities = df[cat].dropna().unique()
        n_mod = len(modalities)
        groups = [df[df[cat] == m][num].dropna().values for m in modalities]

        lv_stat, lv_p, equal_var = _levene_test(*[g for g in groups if len(g) > 0])
        groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in groups if len(g) > 0])

        stat, p, effect = (np.nan, np.nan, np.nan)
        test_name = "unknown"

        try:
            if n_mod == 2:
                # check pairedness map / default
                paired = assume_apparie_map.get(key, default_apparie)
                if paired:
                    # align rows where both variables present
                    df_pair = df[[num, cat]].dropna()
                    # build paired arrays by taking rows where cat==modalities[0] or modalities[1]
                    grp1 = df_pair[df_pair[cat] == modalities[0]][num].values
                    grp2 = df_pair[df_pair[cat] == modalities[1]][num].values
                    # if lengths differ, cut to min to avoid errors
                    min_len = min(len(grp1), len(grp2))
                    grp1 = grp1[:min_len]
                    grp2 = grp2[:min_len]
                    try:
                        stat, p = stats.ttest_rel(grp1, grp2)
                        test_name = "t-test (paired)"
                    except Exception:
                        try:
                            stat, p = stats.wilcoxon(grp1, grp2)
                            test_name = "Wilcoxon"
                        except Exception:
                            stat, p = (np.nan, np.nan)
                    effect = _cohens_d(grp1, grp2, paired=True)
                else:
                    # independent two groups
                    if groups_norm and equal_var:
                        test_name = "t-test"
                        try:
                            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var)
                        except Exception:
                            stat, p = (np.nan, np.nan)
                        effect = _cohens_d(groups[0], groups[1], paired=False)
                    else:
                        test_name = "Mann-Whitney"
                        try:
                            stat, p = stats.mannwhitneyu(groups[0], groups[1])
                        except Exception:
                            stat, p = (np.nan, np.nan)
                        effect = _rank_biserial_from_u(stat, len(groups[0]), len(groups[1]))
            elif n_mod > 2:
                if groups_norm and equal_var:
                    test_name = "ANOVA"
                    try:
                        stat, p = stats.f_oneway(*groups)
                    except Exception:
                        stat, p = (np.nan, np.nan)
                    effect = _eta_squared_anova(groups)
                else:
                    test_name = "Kruskal-Wallis"
                    try:
                        stat, p = stats.kruskal(*groups)
                    except Exception:
                        stat, p = (np.nan, np.nan)
                    effect = np.nan
            else:
                test_name = "unknown"
                stat, p, effect = (np.nan, np.nan, np.nan)
        except Exception:
            stat, p, effect = (np.nan, np.nan, np.nan)

        # Graphique
        plot_path = None
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"{num} par {cat} ({test_name})")
            fname = f"{key}_boxplot.png"
            plot_path = os.path.join(output_folder, fname)
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plot_path = None

        interpretation = interpret_bivariate_result(test_name, p, effect_size=effect)

        summary_records.append({
            "test_id": key, "test": test_name, "var_num": num, "var_cat": cat,
            "stat": stat, "p_value": p, "effect": effect, "interpretation": interpretation
        })
        details[key] = {
            "type": "num-cat",
            "var_num": num, "var_cat": cat,
            "n_modalities": n_mod,
            "levene_stat": lv_stat, "levene_p": lv_p, "equal_var": equal_var,
            "test": test_name,
            "statistic": float(stat) if not pd.isna(stat) else np.nan,
            "p_value": float(p) if not pd.isna(p) else np.nan,
            "effect_size": float(effect) if not pd.isna(effect) else np.nan,
            "plot_boxplot": plot_path,
            "interpretation": interpretation
        }

    # === Catégoriel vs Catégoriel ===
    for v1, v2 in itertools.combinations(cat_vars, 2):
        key = f"{v1}__{v2}"
        table = pd.crosstab(df[v1], df[v2])
        try:
            chi2, p_chi, dof, expected = chi2_contingency(table)
        except Exception:
            chi2, p_chi, dof, expected = (np.nan, np.nan, np.nan, None)
        small_expected = (expected is not None) and ((expected < 5).sum() > 0)
        if small_expected or table.shape == (2, 2):
            test_name = "Fisher exact" if table.shape == (2, 2) else "Chi2 avec prudence"
            if table.shape == (2, 2):
                try:
                    stat, p = fisher_exact(table.values)
                except Exception:
                    stat, p = (np.nan, np.nan)
            else:
                stat, p = chi2, p_chi
        else:
            test_name = "Chi2"
            stat, p = chi2, p_chi

        try:
            cram = _cramers_v(table)
        except Exception:
            cram = np.nan

        # Heatmap
        plot_path = None
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            ax.set_title(f"{v1} vs {v2} ({test_name})")
            fname = f"{key}_heatmap.png"
            plot_path = os.path.join(output_folder, fname)
            fig.savefig(plot_path, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            plot_path = None

        interpretation = interpret_bivariate_result(test_name, p, cramers_v=cram)

        summary_records.append({
            "test_id": key, "test": test_name, "var1": v1, "var2": v2,
            "stat": stat, "p_value": p, "cramers_v": cram, "interpretation": interpretation
        })
        details[key] = {
            "type": "cat-cat",
            "var1": v1, "var2": v2,
            "contingency_table": table,
            "test": test_name,
            "statistic": float(stat) if not pd.isna(stat) else np.nan,
            "p_value": float(p) if not pd.isna(p) else np.nan,
            "cramers_v": float(cram) if not pd.isna(cram) else np.nan,
            "plot": plot_path,
            "interpretation": interpretation
        }

    summary_df = pd.DataFrame(summary_records)
    return summary_df, details
