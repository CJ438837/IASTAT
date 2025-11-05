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

plt.style.use('seaborn-muted')


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
    # rank-biserial r_rb = 1 - (2U) / (n1*n2)
    try:
        return 1.0 - (2.0 * u) / (n1 * n2)
    except Exception:
        return np.nan


def _eta_squared_anova(groups):
    # groups: list of arrays
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
    ss_total = sum((all_vals - grand_mean) ** 2)
    if ss_total == 0:
        return np.nan
    return float(ss_between / ss_total)


def _cramers_v(table):
    """Cramer's V for contingency table"""
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
    # For large n (>=5000) Shapiro may be inappropriate; still use if small.
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
    - types_df : DataFrame avec colonnes 'variable' et 'type' (type in: "numérique","catégorielle","binaire","ordinale" possible)
    - distribution_df : DataFrame avec colonnes 'variable' et 'verdict' ("Normal" / "Non Normal") (optionnel mais utilisé pour choix)
    - df : DataFrame de données
    - assume_apparie_map : dict optional mapping ("var1__var2": True/False) to force pairedness per test
    - default_apparie : default boolean if not specified in map
    - output_folder : where plots are saved
    Returns:
        summary_df (pandas.DataFrame) : résumé compact par test
        details (dict) : résultats détaillés et chemins des figures
    """
    _ensure_dir(output_folder)
    if assume_apparie_map is None:
        assume_apparie_map = {}

    # -- normalize column names in types_df if needed
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

    # build lists
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].astype(str).tolist()
    cat_vars = types_df[types_df['type'].isin(["catégorielle", "binaire"])]['variable'].astype(str).tolist()
    ord_vars = types_df[types_df['type'] == "ordinale"]['variable'].astype(str).tolist()

    summary_records = []
    details = {}

    # ---------- Numeric vs Numeric ----------
    for v1, v2 in itertools.combinations(num_vars, 2):
        key = f"{v1}__{v2}"
        arr1 = df[v1].dropna()
        arr2 = df[v2].dropna()
        res = {"type": "num-num", "var1": v1, "var2": v2}
        # normality checks
        norm1 = _normality_test(arr1)
        norm2 = _normality_test(arr2)
        res["normality_var1"] = norm1
        res["normality_var2"] = norm2
        # choose correlation
        if norm1["normal"] and norm2["normal"]:
            corr_name = "Pearson"
            try:
                stat, p = stats.pearsonr(arr1.dropna(), arr2.dropna())
            except Exception:
                stat, p = (np.nan, np.nan)
        else:
            # prefer Spearman; also offer Kendall
            corr_name = "Spearman"
            try:
                stat, p = stats.spearmanr(arr1.dropna(), arr2.dropna())
            except Exception:
                stat, p = (np.nan, np.nan)
        res["test"] = f"Correlation ({corr_name})"
        res["statistic"] = float(stat) if not pd.isna(stat) else np.nan
        res["p_value"] = float(p) if not pd.isna(p) else np.nan
        # Kendall effect optionally
        tau, p_tau = _kendall_tau(arr1.dropna(), arr2.dropna())
        res["kendall_tau"] = tau
        res["kendall_p"] = p_tau
        # plot scatter + regression line
        try:
            fig, ax = plt.subplots()
            sns.scatterplot(x=v1, y=v2, data=df, ax=ax)
            ax.set_title(f"{v1} vs {v2} ({corr_name})")
            # regression line
            try:
                m, b = np.polyfit(df[v1].dropna(), df[v2].dropna(), 1)
                xs = np.array(ax.get_xlim())
                ax.plot(xs, m * xs + b, linestyle='--')
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

    # ---------- Numeric vs Categorical ----------
    for num, cat in itertools.product(num_vars, cat_vars):
        key = f"{num}__{cat}"
        res = {"type": "num-cat", "var_num": num, "var_cat": cat}
        groups = []
        group_labels = []
        grouped = df.groupby(cat)[num]
        for label, series in grouped:
            group_labels.append(label)
            groups.append(series.dropna().values)

        n_modalites = len(groups)
        res["n_modalites"] = n_modalites
        # normality per group
        res["normality_groups"] = {str(label): _normality_test(series) for label, series in df.groupby(cat)[num]}
        # homogeneity of variances
        try:
            lv_stat, lv_p, equal_var = _levene_test(*groups)
        except Exception:
            lv_stat, lv_p, equal_var = (np.nan, np.nan, False)
        res["levene_stat"] = lv_stat
        res["levene_p"] = lv_p
        res["equal_var"] = equal_var

        # decide test
        if n_modalites == 2:
            # choose between t-test (indep/paired) or Mann-Whitney
            # pairedness from map
            paired = assume_apparie_map.get(key, assume_apparie_map.get(f"{cat}__{num}", default_apparie))
            # normality of both groups?
            groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in groups])
            if groups_norm and equal_var:
                test_name = "t-test (independent)" if not paired else "t-test (paired)"
                if paired:
                    try:
                        stat, p = stats.ttest_rel(groups[0], groups[1])
                    except Exception:
                        stat, p = (np.nan, np.nan)
                else:
                    stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var)
                effect = _cohens_d(groups[0], groups[1], paired=paired)
            else:
                # consider Mann-Whitney / Wilcoxon
                test_name = "Mann-Whitney" if not paired else "Wilcoxon"
                try:
                    if paired:
                        stat, p = stats.wilcoxon(groups[0], groups[1])
                    else:
                        stat, p = stats.mannwhitneyu(groups[0], groups[1])
                except Exception:
                    stat, p = (np.nan, np.nan)
                # rank-biserial
                try:
                    if not paired:
                        u = stat if hasattr(stat, "__float__") else stat
                        effect = _rank_biserial_from_u(u, len(groups[0]), len(groups[1]))
                    else:
                        effect = np.nan
                except Exception:
                    effect = np.nan
        elif n_modalites > 2:
            # ANOVA vs Kruskal-Wallis; if unequal var -> suggest Welch when possible
            groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in groups])
            if groups_norm and equal_var:
                test_name = "ANOVA (one-way)"
                try:
                    stat, p = stats.f_oneway(*groups)
                except Exception:
                    stat, p = (np.nan, np.nan)
                effect = _eta_squared_anova(groups)
            else:
                # try Kruskal-Wallis
                test_name = "Kruskal-Wallis"
                try:
                    stat, p = stats.kruskal(*groups)
                except Exception:
                    stat, p = (np.nan, np.nan)
                effect = np.nan
        else:
            test_name = "unknown"
            stat, p, effect = (np.nan, np.nan, np.nan)

        res["test"] = test_name
        res["statistic"] = float(stat) if not pd.isna(stat) else np.nan
        res["p_value"] = float(p) if not pd.isna(p) else np.nan
        res["effect_size"] = float(effect) if not pd.isna(effect) else np.nan
        # plot boxplot + histogram per modality
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

    # ---------- Categorical vs Categorical ----------
    for v1, v2 in itertools.combinations(cat_vars, 2):
        key = f"{v1}__{v2}"
        res = {"type": "cat-cat", "var1": v1, "var2": v2}
        table = pd.crosstab(df[v1], df[v2])
        res["contingency_table"] = table
        n = table.to_numpy().sum()
        # choose Fisher if small expected counts
        try:
            chi2, p_chi, dof, expected = chi2_contingency(table)
            small_expected = (expected < 5).sum() > 0
        except Exception:
            chi2, p_chi, dof, expected = (np.nan, np.nan, np.nan, None)
            small_expected = False
        if small_expected or table.size <= 4:
            # Fisher exact maybe only for 2x2
            test_name = "Fisher exact" if table.shape == (2, 2) else "Chi2 with caution (small counts)"
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
        # effect size: Cramer's V
        try:
            cram = _cramers_v(table)
        except Exception:
            cram = np.nan
        res["cramers_v"] = cram
        # plot heatmap
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

    # Return summary DataFrame and details dict
    summary_df = pd.DataFrame(summary_records)
    return summary_df, details
