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
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    denom = min(r - 1, k - 1)
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
        return {"test": "Shapiro", "stat": np.nan, "p": np.nan, "normal": False}
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
    - types_df : DataFrame avec colonnes 'variable' et 'type'
    - distribution_df : DataFrame avec colonnes 'variable' et 'verdict'
    - df : DataFrame de données
    - assume_apparie_map : dict pour forcer appariement par test
    - default_apparie : bool par défaut si non précisé
    - output_folder : dossier pour les plots
    Returns:
        summary_df : résumé compact par test
        details : dictionnaire avec résultats et plots
    """
    _ensure_dir(output_folder)
    if assume_apparie_map is None:
        assume_apparie_map = {}

    # Normalisation des colonnes
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

    # Variables
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].astype(str).tolist()
    cat_vars = types_df[types_df['type'].isin(["catégorielle", "binaire"])]['variable'].astype(str).tolist()
    ord_vars = types_df[types_df['type'] == "ordinale"]['variable'].astype(str).tolist()

    summary_records = []
    details = {}

    # ---------- Numeric vs Numeric ----------
    for v1, v2 in itertools.combinations(num_vars, 2):
        key = f"{v1}__{v2}"
        res = {"type": "num-num", "var1": v1, "var2": v2}

        # Gestion appariement + NA
        paired = assume_apparie_map.get(key, default_apparie)
        if paired:
            df_pair = df[[v1, v2]].dropna()
            arr1 = df_pair[v1].values
            arr2 = df_pair[v2].values
        else:
            arr1 = df[v1].dropna().values
            arr2 = df[v2].dropna().values

        # Normalité
        norm1 = _normality_test(pd.Series(arr1))
        norm2 = _normality_test(pd.Series(arr2))
        res["normality_var1"] = norm1
        res["normality_var2"] = norm2

        # Test statistique
        if paired:
            try:
                stat, p = stats.ttest_rel(arr1, arr2)
                test_name = "t-test (paired)"
            except Exception:
                try:
                    stat, p = stats.wilcoxon(arr1, arr2)
                    test_name = "Wilcoxon"
                except Exception:
                    stat, p = (np.nan, np.nan)
                    test_name = "paired unknown"
            effect = _cohens_d(arr1, arr2, paired=True)
        else:
            try:
                stat, p = stats.ttest_ind(arr1, arr2, equal_var=True)
                test_name = "t-test (independent)"
            except Exception:
                try:
                    stat, p = stats.mannwhitneyu(arr1, arr2)
                    test_name = "Mann-Whitney"
                except Exception:
                    stat, p = (np.nan, np.nan)
                    test_name = "independent unknown"
            effect = _cohens_d(arr1, arr2, paired=False)

        res["test"] = test_name
        res["statistic"] = float(stat) if not pd.isna(stat) else np.nan
        res["p_value"] = float(p) if not pd.isna(p) else np.nan
        res["effect_size"] = float(effect) if not pd.isna(effect) else np.nan

        # Scatter plot
        try:
            fig, ax = plt.subplots()
            ax.scatter(arr1, arr2)
            ax.set_xlabel(v1)
            ax.set_ylabel(v2)
            ax.set_title(f"{v1} vs {v2} ({test_name})")
            fname = f"{key}_scatter.png"
            ppath = os.path.join(output_folder, fname)
            fig.savefig(ppath, bbox_inches="tight")
            plt.close(fig)
            res["plot"] = ppath
        except Exception:
            res["plot"] = None

        summary_records.append({
            "test_id": key, "test": test_name, "var1": v1, "var2": v2,
            "stat": res["statistic"], "p_value": res["p_value"], "effect": res["effect_size"]
        })
        details[key] = res

    # ---------- Numeric vs Categorical ----------
    for num, cat in itertools.product(num_vars, cat_vars):
        key = f"{num}__{cat}"
        res = {"type": "num-cat", "var_num": num, "var_cat": cat}

        # Groupes
        grouped = df.groupby(cat)[num]
        groups = [g.dropna().values for _, g in grouped]
        group_labels = [label for label, _ in grouped]
        n_modalites = len(groups)
        res["n_modalites"] = n_modalites
        res["normality_groups"] = {str(label): _normality_test(pd.Series(g)) for label, g in grouped}

        # Levene
        try:
            lv_stat, lv_p, equal_var = _levene_test(*groups)
        except Exception:
            lv_stat, lv_p, equal_var = (np.nan, np.nan, False)
        res["levene_stat"] = lv_stat
        res["levene_p"] = lv_p
        res["equal_var"] = equal_var

        # Paired?
        paired = assume_apparie_map.get(key, default_apparie)
        if n_modalites == 2:
            if paired:
                df_pair = df[[num, cat]].dropna()
                grp_vals = [df_pair[df_pair[cat]==label][num].values for label in df_pair[cat].unique()]
                arr1, arr2 = grp_vals[:2]
                try:
                    stat, p = stats.ttest_rel(arr1, arr2)
                    test_name = "t-test (paired)"
                except Exception:
                    try:
                        stat, p = stats.wilcoxon(arr1, arr2)
                        test_name = "Wilcoxon"
                    except Exception:
                        stat, p = (np.nan, np.nan)
                        test_name = "paired unknown"
                effect = _cohens_d(arr1, arr2, paired=True)
            else:
                arr1, arr2 = groups[:2]
                groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in groups])
                if groups_norm and equal_var:
                    try:
                        stat, p = stats.ttest_ind(arr1, arr2, equal_var=equal_var)
                        test_name = "t-test (independent)"
                    except Exception:
                        stat, p = (np.nan, np.nan)
                        test_name = "independent unknown"
                    effect = _cohens_d(arr1, arr2, paired=False)
                else:
                    try:
                        stat, p = stats.mannwhitneyu(arr1, arr2)
                        test_name = "Mann-Whitney"
                    except Exception:
                        stat, p = (np.nan, np.nan)
                        test_name = "Mann-Whitney unknown"
                    effect = _rank_biserial_from_u(stat, len(arr1), len(arr2))
        elif n_modalites > 2:
            groups_norm = all([_normality_test(pd.Series(g))["normal"] for g in groups])
            if groups_norm and equal_var:
                try:
                    stat, p = stats.f_oneway(*groups)
                    test_name = "ANOVA (one-way)"
                except Exception:
                    stat, p = (np.nan, np.nan)
                    test_name = "ANOVA unknown"
                effect = _eta_squared_anova(groups)
            else:
                try:
                    stat, p = stats.kruskal(*groups)
                    test_name = "Kruskal-Wallis"
                except Exception:
                    stat, p = (np.nan, np.nan)
                    test_name = "Kruskal-Wallis unknown"
                effect = np.nan
        else:
            stat, p, effect = np.nan, np.nan, np.nan
            test_name = "unknown"

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

    # ---------- Categorical vs Categorical ----------
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
            test_name = "Fisher exact" if table.shape == (2, 2) else "Chi2 (small counts)"
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
