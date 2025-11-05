# modules/IA_STAT_testbivaries.py
import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, chi2_contingency, fisher_exact

# Optional import
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    STATSmodels_AVAILABLE = True
except Exception:
    STATSmodels_AVAILABLE = False

plt.style.use('seaborn-v0_8-muted')


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
            "p": float(p) if not np.isnan(p) else np.nan, "normal": (p is not None and p > 0.05)}


def _levene_test(*groups):
    try:
        stat, p = levene(*groups)
        return float(stat), float(p), (p > 0.05)
    except Exception:
        return np.nan, np.nan, False


def propose_tests_bivaries(types_df, distribution_df, df,
                           output_folder="tests_bivaries_plots",
                           assume_apparie_map=None,
                           default_apparie=False,
                           verbose=False):
    """
    Exécute une batterie complète de tests bivariés avec gestion des tests appariés.
    """
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

    # Construire les listes
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].astype(str).tolist()
    cat_vars = types_df[types_df['type'].isin(["catégorielle", "binaire"])]['variable'].astype(str).tolist()
    ord_vars = types_df[types_df['type'] == "ordinale"]['variable'].astype(str).tolist()

    summary_records = []
    details = {}

    # ---------- Numérique vs Numérique ----------
    for v1, v2 in itertools.combinations(num_vars, 2):
        key = f"{v1}__{v2}"
        arr1 = df[v1].dropna()
        arr2 = df[v2].dropna()
        res = {"type": "num-num", "var1": v1, "var2": v2}
        # Normality
        norm1 = _normality_test(arr1)
        norm2 = _normality_test(arr2)
        res["normality_var1"] = norm1
        res["normality_var2"] = norm2
        # Choose correlation
        if norm1["normal"] and norm2["normal"]:
            corr_name = "Pearson"
            try:
                stat, p = stats.pearsonr(arr1, arr2)
            except Exception:
                stat, p = (np.nan, np.nan)
        else:
            corr_name = "Spearman"
            try:
                stat, p = stats.spearmanr(arr1, arr2)
            except Exception:
                stat, p = (np.nan, np.nan)
        res["test"] = f"Correlation ({corr_name})"
        res["statistic"] = float(stat) if not pd.isna(stat) else np.nan
        res["p_value"] = float(p) if not pd.isna(p) else np.nan
        tau, p_tau = _kendall_tau(arr1, arr2)
        res["kendall_tau"] = tau
        res["kendall_p"] = p_tau
        # Scatter plot
        try:
            fig, ax = plt.subplots()
            sns.scatterplot(x=v1, y=v2, data=df, ax=ax)
            ax.set_title(f"{v1} vs {v2} ({corr_name})")
            # regression line
            try:
                m, b = np.polyfit(arr1, arr2, 1)
                xs = np.array(ax.get_xlim())
                ax.plot(xs, m*xs+b, linestyle='--')
            except Exception:
                pass
            fname = f"{key}_scatter.png"
            ppath = os.path.join(output_folder, fname)
            fig.savefig(ppath, bbox_inches="tight")
            plt.close(fig)
            res["plot"] = ppath
        except Exception:
            res["plot"] = None

        summary_records.append({
            "test_id": key, "test": res["test"], "var1": v1, "var2": v2,
            "stat": res["statistic"], "p_value": res["p_value"]
        })
        details[key] = res

    # ---------- Numérique vs Catégoriel ----------
    for num, cat in itertools.product(num_vars, cat_vars):
        key = f"{num}__{cat}"
        res = {"type": "num-cat", "var_num": num, "var_cat": cat}

        modalities = df[cat].dropna().unique()
        n_modalites = len(modalities)
        res["n_modalites"] = n_modalites

        # Vérifier appariement
        paired = assume_apparie_map.get(key, default_apparie)

        if n_modalites == 2 and paired:
            # Apparié : supprimer les lignes avec NA dans l'une ou l'autre variable
            df_pair = df[df[cat].isin(modalities)][[num, cat]].dropna()
            grp1 = df_pair[df_pair[cat]==modalities[0]][num].values
            grp2 = df_pair[df_pair[cat]==modalities[1]][num].values
            min_len = min(len(grp1), len(grp2))
            grp1, grp2 = grp1[:min_len], grp2[:min_len]
            try:
                stat, p = stats.ttest_rel(grp1, grp2)
                test_name = "t-test (paired)"
            except Exception:
                try:
                    stat, p = stats.wilcoxon(grp1, grp2)
                    test_name = "Wilcoxon"
                except Exception:
                    stat, p = np.nan, np.nan
                    test_name = "paired unknown"
            effect = _cohens_d(grp1, grp2, paired=True)
        else:
            # Indépendant (ou >2 modalités)
            groups = [series.dropna().values for _, series in df.groupby(cat)[num]]
            groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in groups])
            lv_stat, lv_p, equal_var = _levene_test(*groups)
            if n_modalites == 2:
                if groups_norm and equal_var:
                    test_name = "t-test"
                    stat, p = stats.ttest_ind(*groups, equal_var=equal_var)
                    effect = _cohens_d(groups[0], groups[1], paired=False)
                else:
                    test_name = "Mann-Whitney"
                    stat, p = stats.mannwhitneyu(*groups)
                    effect = _rank_biserial_from_u(stat, len(groups[0]), len(groups[1]))
            elif n_modalites > 2:
                if groups_norm and equal_var:
                    test_name = "ANOVA"
                    stat, p = stats.f_oneway(*groups)
                    effect = _eta_squared_anova(groups)
                else:
                    test_name = "Kruskal-Wallis"
                    stat, p = stats.kruskal(*groups)
                    effect = np.nan
            else:
                test_name = "unknown"
                stat, p, effect = np.nan, np.nan, np.nan

        res["test"] = test_name
        res["statistic"] = float(stat) if not pd.isna(stat) else np.nan
        res["p_value"] = float(p) if not pd.isna(p) else np.nan
        res["effect_size"] = float(effect) if not pd.isna(effect) else np.nan

        # Boxplot
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=cat, y=num, data=df, ax=ax)
            ax.set_title(f"{num} by {cat} ({test_name})")
            fname = f"{key}_boxplot.png"
            ppath = os.path.join(output_folder, fname)
            fig.savefig(ppath, bbox_inches="tight")
            plt.close(fig)
            res["plot_boxplot"] = ppath
        except Exception:
            res["plot_boxplot"] = None

        summary_records.append({
            "test_id": key, "test": test_name, "var_num": num, "var_cat": cat,
            "stat": res["statistic"], "p_value": res["p_value"], "effect": res["effect_size"]
        })
        details[key] = res

    # ---------- Catégoriel vs Catégoriel ----------
    for v1, v2 in itertools.combinations(cat_vars, 2):
        key = f"{v1}__{v2}"
        res = {"type": "cat-cat", "var1": v1, "var2": v2}
        table = pd.crosstab(df[v1], df[v2])
        res["contingency_table"] = table
        n = table.to_numpy().sum()
        try:
            chi2, p_chi, dof, expected = chi2_contingency(table)
            small_expected = (expected < 5).sum() > 0
        except Exception:
            chi2, p_chi, dof, expected = (np.nan, np.nan, np.nan, None)
            small_expected = False
        if small_expected or table.size <= 4:
            test_name = "Fisher exact" if table.shape == (2, 2) else "Chi2 with caution"
            if table.shape == (2, 2):
                try:
                    stat, p = fisher_exact(table)
                except Exception:
                    stat, p = (np.nan, np.nan)
            else:
                stat, p = chi2, p_chi
        else:
            test_name = "Chi2"
            stat, p = chi2, p_chi
        res["test"] = test_name
        res["statistic"] = float(stat) if not pd.isna(stat) else np.nan
        res["p_value"] = float(p) if not pd.isna(p) else np.nan
        try:
            cram = _cramers_v(table)
        except Exception:
            cram = np.nan
        res["cramers_v"] = cram
        # Heatmap
        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
            ax.set_title(f"{v1} vs {v2} ({test_name})")
            fname = f"{key}_heatmap.png"
            ppath = os.path.join(output_folder, fname)
            fig.savefig(ppath, bbox_inches="tight")
            plt.close(fig)
            res["plot"] = ppath
        except Exception:
            res["plot"] = None

        summary_records.append({
            "test_id": key, "test": test_name, "var1": v1, "var2": v2,
            "stat": res["statistic"], "p_value": res["p_value"], "cramers_v": res["cramers_v"]
        })
        details[key] = res

    summary_df = pd.DataFrame(summary_records)
    return summary_df, details
