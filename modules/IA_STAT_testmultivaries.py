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

plt.style.use("seaborn-v0_8-muted")


# ---- Helpers supplémentaires ----
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


def _box_m_test(X, group):
    """
    Compute Box's M test for equality of covariance matrices across groups.
    X: numeric DataFrame of predictors
    group: categorical Series with group labels
    Returns M_stat_corrected, df, pvalue
    Implementation based on standard Box's M statistic with small-sample correction.
    """
    try:
        groups = [X[group == g] for g in group.dropna().unique()]
        g = len(groups)
        p = X.shape[1]
        ns = [len(gg) for gg in groups]
        if any([n <= p for n in ns]):
            # not enough observations to compute cov reliably
            return None, None, None, "Taille de groupe insuffisante pour Box's M"
        cov_mats = [np.cov(gg.T, bias=False) for gg in groups]
        pooled = sum([(ns[i] - 1) * cov_mats[i] for i in range(g)]) / (sum(ns) - g)
        ln_det_pooled = np.log(np.linalg.det(pooled))
        M = 0.0
        for i in range(g):
            M += (ns[i] - 1) * (np.log(np.linalg.det(cov_mats[i])) - ln_det_pooled)
        # correction factor
        c = 0.0
        for i in range(g):
            c += 1.0 / (ns[i] - 1)
        correction = ((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) * (c - 1.0 / (sum(ns) - g))
        M_corr = (1 - correction) * M
        df = (g - 1) * p * (p + 1) / 2.0
        pval = 1 - chi2.cdf(M_corr, df)
        return float(M_corr), float(df), float(pval), None
    except Exception as e:
        return None, None, None, str(e)


# === Fonction harmonisée et complète ===
def propose_tests_multivariés(df, types_df, target_var, explicatives):
    """
    Retourne une liste d'entrées harmonisées décrivant analyses multivariées.
    Output : list[ dict ] with keys:
      - test (str)
      - result_df (pd.DataFrame | None)
      - fig (matplotlib.figure.Figure | None)
      - additional_info (dict | None)
      - error (str | None)
    """
    results = []

    try:
        # --- Validate inputs ---
        if target_var not in df.columns:
            results.append({"test": "Input validation", "result_df": None, "fig": None,
                            "additional_info": None, "error": f"target_var '{target_var}' absent du DataFrame"})
            return results

        for v in explicatives:
            if v not in df.columns:
                results.append({"test": "Input validation", "result_df": None, "fig": None,
                                "additional_info": None, "error": f"explicative '{v}' absent du DataFrame"})
                return results

        # --- Détection types ---
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

        # ----------------
        # 1) PCA (numérique)
        # ----------------
        if all_numeric:
            entry = {"test": "PCA", "result_df": None, "fig": None, "additional_info": None, "error": None}
            try:
                X = numeric_subset[explicatives].copy()
                if X.shape[1] >= 2 and X.shape[0] >= 2:
                    scaler = StandardScaler()
                    Xs = scaler.fit_transform(X)
                    n_comp = min(min(X.shape[0], X.shape[1]), 2)
                    pca = PCA(n_components=n_comp)
                    pcs = pca.fit_transform(Xs)
                    explained = pca.explained_variance_ratio_
                    explained_cum = np.cumsum(explained)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    contrib_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(loadings.shape[1])], index=X.columns)
                    explained_df = pd.DataFrame({
                        "PC": [f"PC{i+1}" for i in range(len(explained))],
                        "ExplainedVariance": explained,
                        "ExplainedVarianceCumulative": explained_cum
                    })

                    # circle plot if possible
                    fig = None
                    try:
                        if loadings.shape[1] >= 2:
                            fig, ax = plt.subplots(figsize=(6,6))
                            circle = plt.Circle((0,0), 1, color='black', fill=False)
                            ax.add_artist(circle)
                            for i_var, var in enumerate(X.columns):
                                x = float(contrib_df.iloc[i_var, 0])
                                y = float(contrib_df.iloc[i_var, 1])
                                ax.arrow(0, 0, x, y, head_width=0.02, length_includes_head=True)
                                ax.text(x*1.05, y*1.05, var, fontsize=9)
                            ax.set_xlim(-1,1)
                            ax.set_ylim(-1,1)
                            ax.set_xlabel("PC1")
                            ax.set_ylabel("PC2")
                            ax.set_title("Cercle des corrélations (PCA)")
                            ax.axhline(0, color='grey', lw=0.5)
                            ax.axvline(0, color='grey', lw=0.5)
                    except Exception:
                        fig = None

                    # KMO
                    try:
                        kmo_total, kmo_per_var = _kmo(X.values)
                        kmo_info = {"KMO_total": kmo_total, "KMO_per_variable": dict(zip(X.columns, kmo_per_var.tolist()))}
                    except Exception as e:
                        kmo_info = {"error": str(e)}

                    entry["result_df"] = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
                    entry["fig"] = _safe_fig(fig)
                    entry["additional_info"] = {
                        "explained_variance": explained_df.to_dict(orient="records"),
                        "contributions": contrib_df.reset_index().rename(columns={"index":"variable"}).to_dict(orient="records"),
                        "kmo": kmo_info
                    }
                else:
                    entry["error"] = "Trop peu de données/variables pour PCA."
            except Exception as e:
                entry["error"] = str(e)
            results.append(entry)

        # ----------------
        # 2) MCA (catégoriel)
        # ----------------
        if all_categorical:
            entry = {"test": "MCA", "result_df": None, "fig": None, "additional_info": None, "error": None}
            try:
                subset_cat = subset.astype(str)
                mca = MCA(n_components=2, random_state=42)
                coords = mca.fit_transform(subset_cat)
                try:
                    inertias = mca.explained_inertia_
                    inertia_df = pd.DataFrame({"dimension": list(range(1, len(inertias)+1)), "explained_inertia": inertias})
                except Exception:
                    inertia_df = None
                fig = None
                try:
                    fig, ax = plt.subplots(figsize=(6,5))
                    if hasattr(coords, "iloc"):
                        ax.scatter(coords.iloc[:,0], coords.iloc[:,1], alpha=0.7)
                    else:
                        ax.scatter(coords[0], coords[1], alpha=0.7)
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.set_title("MCA - individus")
                except Exception:
                    fig = None

                entry["result_df"] = _ensure_df(coords)
                entry["fig"] = _safe_fig(fig)
                entry["additional_info"] = {"inertia": _ensure_df(inertia_df).to_dict(orient="records") if inertia_df is not None else None}
            except Exception as e:
                entry["error"] = str(e)
            results.append(entry)

        # ----------------
        # 3) FAMD (mixte)
        # ----------------
        if mixte:
            entry = {"test": "FAMD", "result_df": None, "fig": None, "additional_info": None, "error": None}
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
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.set_title("FAMD - individus")
                except Exception:
                    fig = None
                # contributions optional
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

        # ----------------
        # 4) MANOVA
        # ----------------
        entry = {"test": "MANOVA", "result_df": None, "fig": None, "additional_info": None, "error": None}
        try:
            if (all_numeric or mixte) and subset.shape[0] > 2:
                formula = f"{target_var} ~ " + " + ".join(explicatives)
                manova = MANOVA.from_formula(formula, data=subset)
                try:
                    manova_text = str(manova.mv_test())
                except Exception:
                    manova_text = None
                entry["additional_info"] = {"manova_summary": manova_text}
            else:
                entry["additional_info"] = {"note": "MANOVA non applicable (données insuffisantes ou type incompatible)."}
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

        # ----------------
        # 5) Régression multiple + diagnostics
        # ----------------
        entry = {"test": "Régression multiple (OLS)", "result_df": None, "fig": None, "additional_info": None, "error": None}
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

                # p-values FDR
                try:
                    pvals = model.pvalues.values
                    _, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
                    summary_df["p-value FDR"] = pvals_corr
                except Exception:
                    summary_df["p-value FDR"] = np.nan

                # VIF
                try:
                    vif_df = pd.DataFrame({
                        "Variable": X.columns,
                        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    })
                except Exception as e:
                    vif_df = pd.DataFrame({"Variable": [], "VIF": []})

                entry["result_df"] = summary_df
                entry["additional_info"] = {"vif": vif_df.to_dict(orient="records")}
            else:
                entry["error"] = "Aucune variable explicative numérique disponible pour régression."
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

        # Residual diagnostics as separate entry
        entry = {"test": "Analyse des résidus (diagnostic)", "result_df": None, "fig": None, "additional_info": None, "error": None}
        try:
            if not subset[explicatives].select_dtypes(include=np.number).empty:
                X = subset[explicatives].select_dtypes(include=np.number)
                X_const = sm.add_constant(X)
                y = subset[target_var]
                model = sm.OLS(y, X_const).fit()
                residuals = model.resid
                fitted = model.fittedvalues

                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.scatter(fitted, residuals, alpha=0.7)
                ax1.axhline(0, color='red', linestyle='--')
                ax1.set_xlabel("Valeurs ajustées")
                ax1.set_ylabel("Résidus")
                ax1.set_title("Résidus vs Valeurs ajustées")

                fig2 = None
                try:
                    fig2 = sm.qqplot(residuals, line='s')
                    plt.title("QQ-plot des résidus")
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

        # ----------------
        # 6) Corrélations multiples
        # ----------------
        entry = {"test": "Corrélations multiples", "result_df": None, "fig": None, "additional_info": None, "error": None}
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

        # ----------------
        # 7) Normalité multivariée (Henze-Zirkler via pingouin) + Mardia if available
        # ----------------
        entry = {"test": "Normalité multivariée", "result_df": None, "fig": None, "additional_info": None, "error": None}
        try:
            try:
                import pingouin as pg
                numeric_for_mvn = numeric_subset.dropna()
                if numeric_for_mvn.shape[0] >= 10 and numeric_for_mvn.shape[1] >= 2:
                    hz_stat, hz_p, hz_norm = pg.multivariate_normality(numeric_for_mvn, alpha=0.05)
                    entry["result_df"] = pd.DataFrame({"HZ_stat": [hz_stat], "p-value": [hz_p], "normal": [hz_norm]})
                    entry["additional_info"] = {"conclusion": "multinormalité" if hz_norm else "non multinormale"}
                else:
                    entry["additional_info"] = {"note": "Données insuffisantes pour normalité multivariée (min 10 obs)."}
            except Exception:
                # fallback: try to compute Mardia via pingouin or skip
                try:
                    import pingouin as pg2
                    numeric_for_mvn = numeric_subset.dropna()
                    if numeric_for_mvn.shape[0] >= 10 and numeric_for_mvn.shape[1] >= 2:
                        mardia = pg2.multivariate_normality(numeric_for_mvn, alpha=0.05)
                        entry["result_df"] = pd.DataFrame({"Mardia_skew": [mardia[0]], "Mardia_kurt": [mardia[1]], "p-value": [mardia[2]]})
                        entry["additional_info"] = {"conclusion": mardia[3]}
                    else:
                        entry["additional_info"] = {"note": "Données insuffisantes pour normalité multivariée."}
                except Exception:
                    entry["additional_info"] = {"note": "pingouin non installé — normalité multivariée non réalisée"}
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

        # ----------------
        # 8) Box's M (homogénéité matrices covariance) - requires categorical grouping variable
        # ----------------
        entry = {"test": "Box's M", "result_df": None, "fig": None, "additional_info": None, "error": None}
        try:
            # Box's M is meaningful if target_var is categorical (groups) and explicatives numeric
            if not pd.api.types.is_numeric_dtype(subset[target_var]) and not numeric_subset.empty:
                M_corr, df_box, pval_box, box_err = _box_m_test(numeric_subset, subset[target_var])
                if box_err:
                    entry["error"] = box_err
                else:
                    entry["result_df"] = pd.DataFrame({"BoxM": [M_corr], "df": [df_box], "p-value": [pval_box]})
                    entry["additional_info"] = {"conclusion": "homogène" if pval_box > 0.05 else "hétérogénéité détectée"}
            else:
                entry["additional_info"] = {"note": "Box's M non applicable (target numérique ou pas de variables numériques explicatives)."}
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)

    except Exception as e:
        results.append({"test": "Global", "result_df": None, "fig": None, "additional_info": None, "error": str(e)})

    return results
