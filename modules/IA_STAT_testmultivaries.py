# modules/IA_STAT_testmultivaries.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from prince import MCA, FAMD
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import shapiro, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests
from scipy import stats
import math
import warnings

plt.style.use("seaborn-v0_8-muted")


# -----------------------
# Helpers
# -----------------------
def _kmo(X):
    """Compute KMO measure of sampling adequacy. Returns (kmo_total, kmo_per_variable)."""
    try:
        corr = np.corrcoef(X.T)
        inv_corr = np.linalg.pinv(corr)
        partial = -inv_corr.copy()
        d = np.sqrt(np.abs(np.diag(partial)))
        partial = (partial / d).T / d
        np.fill_diagonal(partial, 0.0)
        a = corr.copy()
        np.fill_diagonal(a, 0.0)
        denom = np.sum(a**2) + np.sum(partial**2)
        if denom == 0:
            return np.nan, np.full(X.shape[1], np.nan)
        kmo_total = np.sum(a**2) / denom
        kmo_per_var = np.sum(a**2, axis=0) / (np.sum(a**2, axis=0) + np.sum(partial**2, axis=0))
        return float(kmo_total), np.array(kmo_per_var, dtype=float)
    except Exception:
        return np.nan, np.full(X.shape[1], np.nan)


def _ensure_df(obj):
    """Return DataFrame if possible, else None"""
    try:
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj
        return pd.DataFrame(obj)
    except Exception:
        return None


def _safe_fig(fig):
    """Return fig if it's a matplotlib Figure, else None"""
    try:
        import matplotlib
        if isinstance(fig, matplotlib.figure.Figure):
            return fig
    except Exception:
        pass
    return None


def _box_m_own(X, group):
    """
    Own implementation of Box's M (approximation).
    X: numeric DataFrame
    group: categorical Series aligned with X
    Returns (M_corr, df, pval, error_message_or_None)
    """
    try:
        groups = [X[group == g].values for g in pd.Series(group).dropna().unique()]
        g = len(groups)
        p = X.shape[1]
        ns = [len(gg) for gg in groups]
        if g < 2:
            return None, None, None, "Besoin d'au moins 2 groupes pour Box's M."
        if any(n <= p for n in ns):
            return None, None, None, "Taille de groupe insuffisante (n <= p) pour Box's M."
        cov_mats = [np.cov(gg.T, bias=False) for gg in groups]
        pooled = sum([(ns[i] - 1) * cov_mats[i] for i in range(g)]) / (sum(ns) - g)
        # ensure invertible
        if np.linalg.matrix_rank(pooled) < p:
            return None, None, None, "Matrice de covariance poolée singulière."
        ln_det_pooled = np.log(np.linalg.det(pooled))
        M = 0.0
        for i in range(g):
            det_i = np.linalg.det(cov_mats[i])
            if det_i <= 0:
                return None, None, None, "Détérminant non strictement positif pour un groupe."
            M += (ns[i] - 1) * (np.log(det_i) - ln_det_pooled)
        c = 0.0
        for n in ns:
            c += 1.0 / (n - 1)
        correction = ((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) * (c - 1.0 / (sum(ns) - g))
        M_corr = (1 - correction) * M
        df = (g - 1) * p * (p + 1) / 2.0
        pval = 1 - chi2.cdf(M_corr, df)
        return float(M_corr), float(df), float(pval), None
    except Exception as e:
        return None, None, None, str(e)


# -----------------------
# Wrapper Box's M: tries pingouin first, else own implementation
# -----------------------
def test_box_m(df, group_col, variables):
    """
    Returns uniform dict with keys: test, result_df, fig, additional_info, interpretation, error
    """
    entry = {"test": "Box's M", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
    try:
        data = df[variables + [group_col]].dropna()
        groups_ser = data[group_col]
        # try pingouin if available
        try:
            import pingouin as pg
            res = pg.box_m(data[variables], data[group_col])
            # pg.box_m may return tuple or DataFrame-like depending on version
            if isinstance(res, tuple) or isinstance(res, list):
                # expect (stat, pval, df1, df2)
                try:
                    stat, pval, df1, df2 = res
                except Exception:
                    # fallback if ordering different
                    stat, pval, df1, df2 = res[0], res[1], None, None
            elif isinstance(res, dict):
                stat = res.get("stat") or res.get("M") or None
                pval = res.get("pval") or res.get("p-value") or None
                df1 = res.get("df1")
                df2 = res.get("df2")
            else:
                # DataFrame-like
                try:
                    stat = float(res.loc["stat", 0])
                    pval = float(res.loc["pval", 0])
                    df1 = None
                    df2 = None
                except Exception:
                    stat, pval, df1, df2 = None, None, None, None
            entry["result_df"] = pd.DataFrame({"BoxM": [stat], "p-value": [pval], "df1": [df1], "df2": [df2]})
            entry["interpretation"] = ("✔️ Homogénéité des matrices de covariance (p ≥ 0.05)." if pval is not None and pval >= 0.05
                                       else "⚠️ Hétérogénéité des matrices de covariance (p < 0.05).")
            entry["additional_info"] = {"method": "pingouin.box_m"}
            return entry
        except Exception:
            # pingouin absent or failed => fallback
            M_corr, df_box, pval_box, err = _box_m_own(data[variables], data[group_col])
            if err:
                entry["error"] = err
                entry["additional_info"] = {"method": "fallback_internal", "note": err}
                return entry
            entry["result_df"] = pd.DataFrame({"BoxM": [M_corr], "df": [df_box], "p-value": [pval_box]})
            entry["interpretation"] = ("✔️ Homogénéité des matrices de covariance (p ≥ 0.05)." if pval_box is not None and pval_box >= 0.05
                                       else "⚠️ Hétérogénéité des matrices de covariance (p < 0.05).")
            entry["additional_info"] = {"method": "internal_approx"}
            return entry
    except Exception as e:
        entry["error"] = str(e)
        return entry


# -----------------------
# propose_tests_multivariés (version harmonisée)
# -----------------------
def propose_tests_multivariés(df, types_df, target_var, explicatives):
    """
    Analyse multivariée complète : PCA, MCA, FAMD, MANOVA, régression+diagnostics,
    corrélations, normalité multivariée, Box's M.
    Retour : liste d'entrées uniformisées.
    """
    results = []

    def _info_wrap(obj):
        if isinstance(obj, dict):
            return obj
        elif obj is None:
            return {}
        else:
            return {"detail": obj}

    # Validate inputs quickly
    if target_var not in df.columns:
        return [{"test": "Input validation", "result_df": None, "fig": None,
                 "additional_info": {"error": f"target_var '{target_var}' absent du DataFrame"}, "interpretation": None, "error": None}]
    for v in explicatives:
        if v not in df.columns:
            return [{"test": "Input validation", "result_df": None, "fig": None,
                     "additional_info": {"error": f"explicative '{v}' absent du DataFrame"}, "interpretation": None, "error": None}]

    # determine types
    try:
        try:
            target_type = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
        except Exception:
            target_type = "numérique" if pd.api.types.is_numeric_dtype(df[target_var]) else "catégorielle"
        explicative_types = []
        for c in explicatives:
            try:
                explicative_types.append(types_df.loc[types_df["variable"] == c, "type"].values[0])
            except Exception:
                explicative_types.append("numérique" if pd.api.types.is_numeric_dtype(df[c]) else "catégorielle")
        all_numeric = all(t == "numérique" for t in [target_type] + explicative_types)
        all_categorical = all(t == "catégorielle" for t in [target_type] + explicative_types)
        mixte = not all_numeric and not all_categorical
        subset = df[[target_var] + explicatives].dropna()
        numeric_subset = subset.select_dtypes(include=np.number)
    except Exception as e:
        results.append({"test": "Global", "result_df": None, "fig": None, "additional_info": {"error": str(e)}, "interpretation": None, "error": None})
        return results

    # 1) PCA
    if all_numeric:
        entry = {"test": "PCA", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
        try:
            X = numeric_subset[explicatives]
            if X.shape[1] >= 2 and X.shape[0] >= 2:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                n_comp = min(2, X.shape[1])
                pca = PCA(n_components=n_comp)
                pcs = pca.fit_transform(Xs)
                explained = pca.explained_variance_ratio_
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                contrib_df = pd.DataFrame(loadings, index=X.columns, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])
                fig = None
                try:
                    if loadings.shape[1] >= 2:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        circle = plt.Circle((0, 0), 1, color='black', fill=False)
                        ax.add_artist(circle)
                        for i_var, var in enumerate(X.columns):
                            x = float(contrib_df.iloc[i_var, 0])
                            y = float(contrib_df.iloc[i_var, 1])
                            ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
                            ax.text(x*1.05, y*1.05, var, fontsize=9)
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_xlabel("PC1")
                        ax.set_ylabel("PC2")
                        ax.set_title("Cercle des corrélations (PCA)")
                except Exception:
                    fig = None
                try:
                    kmo_total, kmo_per_var = _kmo(X.values)
                    kmo_info = {"KMO_total": kmo_total, "KMO_per_variable": dict(zip(X.columns, kmo_per_var.tolist()))}
                except Exception as e:
                    kmo_info = {"error": str(e)}
                entry["result_df"] = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
                entry["fig"] = _safe_fig(fig)
                entry["additional_info"] = {"explained_variance": explained.tolist(), "contributions": contrib_df.reset_index().rename(columns={"index": "variable"}).to_dict(orient="records"), "kmo": kmo_info}
            else:
                entry["error"] = "Trop peu de données/variables pour PCA."
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

    # 2) MCA
    if all_categorical:
        entry = {"test": "MCA", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
        try:
            subset_cat = subset.astype(str)
            mca = MCA(n_components=2, random_state=42)
            coords = mca.fit_transform(subset_cat)
            fig = None
            try:
                fig, ax = plt.subplots(figsize=(6,5))
                if hasattr(coords, "iloc"):
                    ax.scatter(coords.iloc[:,0], coords.iloc[:,1], alpha=0.7)
                else:
                    ax.scatter(coords[0], coords[1], alpha=0.7)
                ax.set_xlabel("Dimension 1"); ax.set_ylabel("Dimension 2"); ax.set_title("MCA - individus")
            except Exception:
                fig = None
            inertia = getattr(mca, "explained_inertia_", None)
            entry["result_df"] = _ensure_df(coords)
            entry["fig"] = _safe_fig(fig)
            entry["additional_info"] = {"inertia": inertia}
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

    # 3) FAMD
    if mixte:
        entry = {"test": "FAMD", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
        try:
            famd = FAMD(n_components=2, random_state=42)
            coords = famd.fit_transform(subset)
            fig = None
            try:
                fig, ax = plt.subplots(figsize=(6,5))
                if hasattr(coords, "iloc"):
                    ax.scatter(coords.iloc[:,0], coords.iloc[:,1], alpha=0.7)
                else:
                    ax.scatter(coords[0], coords[1], alpha=0.7)
                ax.set_xlabel("Dimension 1"); ax.set_ylabel("Dimension 2"); ax.set_title("FAMD - individus")
            except Exception:
                fig = None
            try:
                contribs = famd.column_correlations(subset) if hasattr(famd, "column_correlations") else None
            except Exception:
                contribs = None
            entry["result_df"] = _ensure_df(coords)
            entry["fig"] = _safe_fig(fig)
            entry["additional_info"] = {"contributions": contribs}
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

    # 4) MANOVA
    entry = {"test": "MANOVA", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
    try:
        formula = f"{target_var} ~ " + " + ".join(explicatives)
        manova = MANOVA.from_formula(formula, data=subset)
        try:
            manova_text = str(manova.mv_test())
        except Exception:
            manova_text = None
        entry["additional_info"] = {"manova_summary": manova_text}
    except Exception as e:
        entry["error"] = str(e)
    results.append(entry)

    # 5) Régression multiple + diagnostics
    entry = {"test": "Régression multiple (OLS)", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
    try:
        X = subset[explicatives].select_dtypes(include=np.number)
        if not X.empty:
            X_const = sm.add_constant(X)
            y = subset[target_var]
            model = sm.OLS(y, X_const).fit()
            summary_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values,
                "IC Inf": model.conf_int()[0].values,
                "IC Sup": model.conf_int()[1].values
            })
            try:
                pvals = model.pvalues.values
                _, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
                summary_df["p-value FDR"] = pvals_corr
            except Exception:
                summary_df["p-value FDR"] = np.nan
            try:
                vif_df = pd.DataFrame({"Variable": X.columns, "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
            except Exception:
                vif_df = pd.DataFrame()
            entry["result_df"] = summary_df
            entry["additional_info"] = {"vif": vif_df.to_dict(orient="records")}
        else:
            entry["error"] = "Aucune variable explicative numérique disponible pour régression."
    except Exception as e:
        entry["error"] = str(e)
    results.append(entry)

    # Residuals diagnostics
    entry = {"test": "Analyse des résidus (diagnostic)", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
    try:
        X = subset[explicatives].select_dtypes(include=np.number)
        if not X.empty:
            X_const = sm.add_constant(X)
            y = subset[target_var]
            model = sm.OLS(y, X_const).fit()
            residuals = model.resid
            fitted = model.fittedvalues
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.scatter(fitted, residuals, alpha=0.7)
            ax1.axhline(0, color='red', linestyle='--')
            ax1.set_xlabel("Valeurs ajustées"); ax1.set_ylabel("Résidus"); ax1.set_title("Résidus vs Valeurs ajustées")
            fig2 = None
            try:
                fig2 = sm.qqplot(residuals, line='s')
            except Exception:
                fig2 = None
            shapiro_test = shapiro(residuals) if len(residuals) >= 3 else (np.nan, np.nan)
            bp_test = het_breuschpagan(residuals, model.model.exog) if model.model.exog.shape[1] > 0 else (np.nan, np.nan, np.nan, np.nan)
            norm_test = normal_ad(residuals) if len(residuals) >= 8 else (np.nan, np.nan)
            resid_summary = pd.DataFrame({
                "Test": ["Shapiro-Wilk", "Breusch-Pagan", "Anderson-Darling"],
                "Statistique": [shapiro_test[0] if shapiro_test is not None else np.nan,
                                bp_test[0] if isinstance(bp_test, tuple) else np.nan,
                                norm_test[0] if norm_test is not None else np.nan],
                "p-value": [shapiro_test[1] if shapiro_test is not None else np.nan,
                            bp_test[1] if isinstance(bp_test, tuple) else np.nan,
                            norm_test[1] if norm_test is not None else np.nan]
            })
            entry["result_df"] = resid_summary
            entry["fig"] = _safe_fig(fig1)
            entry["additional_info"] = {"qqplot_fig_available": bool(fig2)}
        else:
            entry["error"] = "Pas de diagnostics de résidus (pas de variables numériques explicatives)."
    except Exception as e:
        entry["error"] = str(e)
    results.append(entry)

    # 6) Corrélations multiples
    entry = {"test": "Corrélations multiples", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
    try:
        corr_df = numeric_subset.corr(numeric_only=True)
        fig_corr, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(corr_df, cmap="coolwarm")
        fig_corr.colorbar(cax)
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
        plt.yticks(range(len(corr_df.columns)), corr_df.columns)
        ax.set_title("Matrice de corrélation")
        entry["result_df"] = corr_df
        entry["fig"] = _safe_fig(fig_corr)
    except Exception as e:
        entry["error"] = str(e)
    results.append(entry)

    # 7) Normalité multivariée (Mardia) - try pingouin, else fallback to univariate checks
    entry = {"test": "Normalité multivariée (Mardia)", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
    try:
        X = numeric_subset.dropna()
        if X.shape[1] < 2:
            entry["additional_info"] = {"note": "Impossible : au moins 2 variables numériques nécessaires."}
        else:
            try:
                import pingouin as pg
                mardia = pg.multivariate_normality(X, alpha=0.05)
                # mardia may be tuple or dict-like
                if isinstance(mardia, tuple) or isinstance(mardia, list):
                    # Many pingouin versions return (normal, p, skew, kurt, p_skew, p_kurt) or similar
                    # Try to extract best-effort
                    try:
                        # try common formats
                        if len(mardia) >= 6:
                            _, _, skew_stat, kurt_stat, p_skew, p_kurt = mardia[:6]
                        elif len(mardia) >= 3:
                            skew_stat = mardia[0]; kurt_stat = None; p_skew = None; p_kurt = None
                        else:
                            skew_stat = None; kurt_stat = None; p_skew = None; p_kurt = None
                    except Exception:
                        skew_stat = None; kurt_stat = None; p_skew = None; p_kurt = None
                    # Build df if possible
                    mardia_df = pd.DataFrame({
                        "Statistique": [skew_stat, kurt_stat],
                        "p-value": [p_skew, p_kurt]
                    }, index=["Skewness", "Kurtosis"])
                    entry["result_df"] = mardia_df
                    entry["additional_info"] = {"note": "pingouin utilisé (tuple result)."}
                    entry["interpretation"] = None
                elif isinstance(mardia, dict):
                    # dict-like output (older versions)
                    sk = mardia.get("skewness"); ku = mardia.get("kurtosis")
                    psk = None; pku = None
                    try:
                        psk = mardia.get("p_skew") or (mardia.get("skew_pval") if "skew_pval" in mardia else None)
                        pku = mardia.get("p_kurt") or (mardia.get("kurt_pval") if "kurt_pval" in mardia else None)
                    except Exception:
                        psk = pku = None
                    mardia_df = pd.DataFrame({"Statistique": [sk, ku], "p-value": [psk, pku]}, index=["Skewness", "Kurtosis"])
                    entry["result_df"] = mardia_df
                    entry["additional_info"] = {"note": "pingouin utilisé (dict result)."}
                    entry["interpretation"] = None
                else:
                    entry["additional_info"] = {"note": "Format de retour pingouin inattendu; affichage partiel."}
                # If pingouin provided a clear normal flag, try to use it
                try:
                    normal_flag = None
                    if isinstance(mardia, dict):
                        normal_flag = mardia.get("normal")
                    elif isinstance(mardia, (tuple, list)) and len(mardia) >= 1:
                        # some versions return (normal_bool, pval, ...)
                        normal_flag = bool(mardia[0])
                    if normal_flag is not None:
                        entry["interpretation"] = "Normale" if normal_flag else "Non normale"
                except Exception:
                    pass
            except Exception:
                # pingouin not available: fallback to univariate Shapiro per variable + note
                uni = []
                pvals = []
                for col in X.columns:
                    try:
                        stat, pval = shapiro(X[col]) if X[col].dropna().shape[0] >= 3 else (np.nan, np.nan)
                    except Exception:
                        stat, pval = (np.nan, np.nan)
                    uni.append({"variable": col, "shapiro_stat": float(stat) if not pd.isna(stat) else np.nan, "p_value": float(pval) if not pd.isna(pval) else np.nan})
                    pvals.append(pval if not pd.isna(pval) else 1.0)
                entry["result_df"] = pd.DataFrame(uni)
                entry["additional_info"] = {"note": "pingouin non installé — test multivarié non réalisé; résultats univariés Shapiro fournis."}
        # append
    except Exception as e:
        entry["error"] = str(e)
        entry["additional_info"] = {"note": "Erreur pendant normalité multivariée."}
    results.append(entry)

    # 8) Box's M - only meaningful if grouping variable provided (we reuse target_var as grouping if categorical)
    entry = {"test": "Box's M", "result_df": None, "fig": None, "additional_info": None, "interpretation": None, "error": None}
    try:
        # meaningful if target_var is categorical and there are numeric explicatives
        if not pd.api.types.is_numeric_dtype(subset[target_var]) and not numeric_subset.empty:
            # call wrapper which will try pingouin then fallback
            box_entry = test_box_m(subset, target_var, numeric_subset.columns.tolist())
            # normalize keys into our format
            entry.update({
                "result_df": box_entry.get("result_df"),
                "fig": None,
                "additional_info": {"method": box_entry.get("additional_info", {})} if box_entry.get("additional_info") is not None else {},
                "interpretation": box_entry.get("interpretation"),
                "error": box_entry.get("error")
            })
        else:
            entry["additional_info"] = {"note": "Box's M non applicable (target numérique ou pas de variables numériques explicatives)."}
    except Exception as e:
        entry["error"] = str(e)
    results.append(entry)

    return results
