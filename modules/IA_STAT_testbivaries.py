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

# Try optional imports
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
    nx = len(x); ny = len(y)
    sx = np.var(x, ddof=1); sy = np.var(y, ddof=1)
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
    try:
        chi2, p, dof, expected = chi2_contingency(table)
        n = table.to_numpy().sum()
        phi2 = chi2 / n
        r, k = table.shape
        denom = min((r - 1), (k - 1))
        if denom == 0:
            return np.nan
        return float(np.sqrt(phi2 / denom))
    except Exception:
        return np.nan


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
    Exécute et retourne une batterie de tests bivariés.
    Tests exécutés un par un, avec gestion des appariés.
    """
    _ensure_dir(output_folder)
    if assume_apparie_map is None:
        assume_apparie_map = {}

    # normalize column names
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
    ord_vars = types_df[types_df['type'] == "ordinale"]['variable'].astype(str).tolist()

    summary_records = []
    details = {}

    # ---------- Numeric vs Numeric ----------
    for v1, v2 in itertools.combinations(num_vars, 2):
        key = f"{v1}__{v2}"
        paired = assume_apparie_map.get(key, default_apparie)
        if paired:
            paired_df = df[[v1, v2]].dropna()
            arr1 = paired_df[v1]
            arr2 = paired_df[v2]
        else:
            arr1 = df[v1].dropna()
            arr2 = df[v2].dropna()
        res = {"type": "num-num", "var1": v1, "var2": v2, "paired": paired}
        norm1 = _normality_test(arr1)
        norm2 = _normality_test(arr2)
        res["normality_var1"] = norm1
        res["normality_var2"] = norm2
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
        # plot
        try:
            fig, ax = plt.subplots()
            sns.scatterplot(x=v1, y=v2, data=df, ax=ax)
            ax.set_title(f"{v1} vs {v2} ({corr_name})")
            m, b = np.polyfit(arr1, arr2, 1)
            xs = np.array(ax.get_xlim())
            ax.plot(xs, m*xs + b, linestyle='--')
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

    # ---------- Numeric vs Categorical ----------
    for num, cat in itertools.product(num_vars, cat_vars):
        key = f"{num}__{cat}"
        paired = assume_apparie_map.get(key, default_apparie)
        res = {"type": "num-cat", "var_num": num, "var_cat": cat, "paired": paired}
        # drop NA if paired
        if paired and df.shape[0] > 0:
            paired_df = df[[num, cat]].dropna()
        else:
            paired_df = df[[num, cat]].copy()
        grouped = [g.dropna().values for _, g in paired_df.groupby(cat)[num]]
        n_modalites = len(grouped)
        res["n_modalites"] = n_modalites
        # normality
        res["normality_groups"] = {str(label): _normality_test(series)
                                   for label, series in paired_df.groupby(cat)[num]}
        lv_stat, lv_p, equal_var = _levene_test(*grouped)
        res["levene_stat"] = lv_stat
        res["levene_p"] = lv_p
        res["equal_var"] = equal_var
        # decide test
        try:
            if n_modalites == 2:
                groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in grouped])
                if groups_norm and equal_var:
                    test_name = "t-test (paired)" if paired else "t-test (indep)"
                    stat, p = stats.ttest_rel(*grouped) if paired else stats.ttest_ind(*grouped, equal_var=equal_var)
                    effect = _cohens_d(grouped[0], grouped[1], paired=paired)
                else:
                    test_name = "Wilcoxon" if paired else "Mann-Whitney"
                    stat, p = stats.wilcoxon(*grouped) if paired else stats.mannwhitneyu(*grouped)
                    effect = _rank_biserial_from_u(stat, len(grouped[0]), len(grouped[1])) if not paired else np.nan
            elif n_modalites > 2:
                groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in grouped])
                if groups_norm and equal_var:
                    test_name = "ANOVA"
                    stat, p = stats.f_oneway(*grouped)
                    effect = _eta_squared_anova(grouped)
                else:
                    test_name = "Kruskal-Wallis"
                    stat, p = stats.kruskal(*grouped)
                    effect = np.nan
            else:
                test_name = "unknown"
                stat, p, effect = np.nan, np.nan, np.nan
        except Exception:
            stat, p, effect = np.nan, np.nan, np.nan
            test_name = "error"

        res["test"] = test_name
        res["statistic"] = stat
        res["p_value"] = p
        res["effect_size"] = effect

        # plot
        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=cat, y=num, data=paired_df, ax=ax)
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
            "stat": stat, "p_value": p, "effect": effect
        })
        details[key] = res

    # ---------- Categorical vs Categorical ----------
    for v1, v2 in itertools.combinations(cat_vars, 2):
        key = f"{v1}__{v2}"
        res = {"type": "cat-cat", "var1": v1, "var2": v2}
        table = pd.crosstab(df[v1], df[v2])
        res["contingency_table"] = table
        try:
            chi2, p_chi, dof, expected = chi2_contingency(table)
            small_expected = (expected < 5).sum() > 0
        except Exception:
            chi2, p_chi, dof, expected = np.nan, np.nan, np.nan, None
            small_expected = False
        if small_expected or table.size <= 4:
            test_name = "Fisher exact" if table.shape == (2, 2) else "Chi2 (small counts)"
            try:
                stat, p = fisher_exact(table) if table.shape == (2, 2) else (chi2, p_chi)
            except Exception:
                stat, p = np.nan, np.nan
        else:
            test_name = "Chi2"
            stat, p = chi2, p_chi
        res["test"] = test_name
        res["statistic"] = stat
        res["p_value"] = p
        res["cramers_v"] = _cramers_v(table)
        # heatmap
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
            "stat": stat, "p_value": p, "cramers_v": res["cramers_v"]
        })
        details[key] = res

    summary_df = pd.DataFrame(summary_records)
    return summary_df, details
