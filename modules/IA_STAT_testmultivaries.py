import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from prince import MCA, FAMD
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, normal_ad
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

plt.style.use("seaborn-v0_8-muted")

# ---- Helpers supplémentaires ----
def _kmo(X):
    """
    Compute KMO measure of sampling adequacy.
    Returns overall_kmo, kmo_per_variable (array)
    Implementation based on correlation and partial correlation matrices.
    """
    try:
        corr = np.corrcoef(X.T)
        inv_corr = np.linalg.pinv(corr)
        partial = -inv_corr.copy()
        d = np.sqrt(np.abs(np.diag(partial)))
        partial = (partial / d).T / d
        np.fill_diagonal(partial, 0.0)
        a = corr.copy()
        np.fill_diagonal(a, 0.0)
        # squared sums
        denom = np.sum(a**2) + np.sum(partial**2)
        if denom == 0:
            return np.nan, np.full(X.shape[1], np.nan)
        kmo_total = np.sum(a**2) / denom
        kmo_per_var = np.sum(a**2, axis=0) / (np.sum(a**2, axis=0) + np.sum(partial**2, axis=0))
        return float(kmo_total), np.array(kmo_per_var, dtype=float)
    except Exception as e:
        return np.nan, np.full(X.shape[1], np.nan)

def _safe_fig_to_none(fig):
    """
    If fig is matplotlib Figure, return it; otherwise return None.
    (keeps previous behavior where sometimes figures are returned)
    """
    try:
        if fig is None:
            return None
        # accept matplotlib.figure.Figure
        import matplotlib
        if isinstance(fig, matplotlib.figure.Figure):
            return fig
        return None
    except Exception:
        return None

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

# === Fonction améliorée ===
def propose_tests_multivariés(df, types_df, target_var, explicatives):
    """
    Retourne une liste 'results' contenant des dictionnaires décrivant les tests
    et visualisations multivariés. Conserve la structure existante et ajoute des diagnostics.
    Inputs / outputs inchangés : retourne 'results' (list).
    """
    results = []

    try:
        # --- Détection du type des variables ---
        target_type = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
        explicative_types = types_df.loc[types_df["variable"].isin(explicatives), "type"].tolist()

        all_numeric = all(t == "numérique" for t in [target_type] + explicative_types)
        all_categorical = all(t == "catégorielle" for t in [target_type] + explicative_types)
        mixte = not all_numeric and not all_categorical

        subset = df[[target_var] + explicatives].dropna()

        # Common numeric subset for many analyses
        numeric_subset = subset.select_dtypes(include=np.number)

        # =========================================
        # 1️⃣ PCA (si tout numérique)
        # =========================================
        if all_numeric:
            try:
                X = numeric_subset[explicatives].copy()
                # Safety: need at least 2 variables
                if X.shape[1] >= 2 and X.shape[0] >= 2:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    pca = PCA(n_components=min( min(X.shape[0], X.shape[1]), 2 ))
                    pcs = pca.fit_transform(X_scaled)
                    explained = pca.explained_variance_ratio_
                    explained_cum = np.cumsum(explained)

                    # contributions / loadings (correlations between vars and PCs)
                    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                    contrib_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(loadings.shape[1])], index=X.columns)
                    # percentage explained
                    explained_df = pd.DataFrame({
                        "PC": [f"PC{i+1}" for i in range(len(explained))],
                        "ExplainedVariance": explained,
                        "ExplainedVarianceCumulative": explained_cum
                    })

                    # Cercle des corrélations (si n_components >=2)
                    fig_circle = None
                    try:
                        if loadings.shape[1] >= 2:
                            fig_circle, ax = plt.subplots(figsize=(6,6))
                            circle = plt.Circle((0,0), 1, color='black', fill=False)
                            ax.add_artist(circle)
                            for i, var in enumerate(X.columns):
                                x = contrib_df.iloc[i,0]
                                y = contrib_df.iloc[i,1]
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
                        fig_circle = None

                    # KMO
                    try:
                        kmo_total, kmo_per_var = _kmo(X.values)
                        kmo_df = pd.DataFrame({
                            "Variable": X.columns,
                            "KMO_per_var": kmo_per_var
                        })
                        kmo_info = {"KMO_total": kmo_total, "KMO_per_variable": kmo_per_var.tolist()}
                    except Exception as e:
                        kmo_df = None
                        kmo_info = {"error": str(e)}

                    results.append({
                        "test": "Analyse en Composantes Principales (PCA)",
                        "result_df": _ensure_df(pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1]) ] )),
                        "fig": fig_circle,
                        "info": {
                            "explained_variance": explained_df.to_dict(orient="list"),
                            "contributions": contrib_df.reset_index().rename(columns={"index":"variable"}).to_dict(orient="records"),
                            "kmo": kmo_info
                        }
                    })
                else:
                    results.append({"test": "PCA", "error": "Trop peu de données/variables pour PCA."})
            except Exception as e:
                results.append({"test": "PCA", "error": str(e)})

        # =========================================
        # 2️⃣ MCA (si tout catégoriel)
        # =========================================
        if all_categorical:
            try:
                subset_cat = subset.astype(str)
                mca = MCA(n_components=2, random_state=42)
                coords = mca.fit_transform(subset_cat)
                # inertie / explained inertia
                try:
                    inertias = mca.explained_inertia_
                    inertia_df = pd.DataFrame({
                        "Dimension": [1,2][:len(inertias)],
                        "ExplainedInertia": list(inertias)
                    })
                except Exception:
                    inertia_df = None

                fig_mca, ax = plt.subplots(figsize=(6,5))
                # coords may be: DataFrame with shape (n_samples, n_components)
                if hasattr(coords, "shape"):
                    ax.scatter(coords.iloc[:,0], coords.iloc[:,1], alpha=0.7)
                else:
                    ax.scatter(coords[0], coords[1], alpha=0.7)
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_title("MCA - individus")
                results.append({
                    "test": "Analyse des Correspondances Multiples (MCA)",
                    "result_df": _ensure_df(coords),
                    "fig": fig_mca,
                    "info": {"inertia": _ensure_df(inertia_df)}
                })
            except Exception as e:
                results.append({"test": "MCA", "error": str(e)})

        # =========================================
        # 3️⃣ FAMD (si mixte)
        # =========================================
        if mixte:
            try:
                famd = FAMD(n_components=2, random_state=42)
                coords = famd.fit_transform(subset)
                fig_famd, ax = plt.subplots(figsize=(6,5))
                # coords may be DataFrame-like
                if hasattr(coords, "__len__") and coords.shape[1] >= 2:
                    ax.scatter(coords.iloc[:,0], coords.iloc[:,1], alpha=0.7)
                else:
                    ax.scatter(coords[0], coords[1], alpha=0.7)
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_title("Analyse Factorielle Mixte (FAMD)")

                # contributions if available
                try:
                    contribs = famd.column_correlations(subset) if hasattr(famd, "column_correlations") else None
                except Exception:
                    contribs = None

                results.append({
                    "test": "Analyse Factorielle Mixte (FAMD)",
                    "result_df": _ensure_df(coords),
                    "fig": fig_famd,
                    "info": {"contributions": contribs}
                })
            except Exception as e:
                results.append({"test": "FAMD", "error": str(e)})

        # =========================================
        # 4️⃣ MANOVA
        # =========================================
        try:
            if (all_numeric or mixte) and subset.shape[0] > 2:
                formula = f"{target_var} ~ " + " + ".join(explicatives)
                manova = MANOVA.from_formula(formula, data=subset)
                manova_res = manova.mv_test()
                # Try to extract simple metric (Wilks' lambda) and approximate effect
                info = {}
                try:
                    # manova_res is an object with .results; try to parse
                    for key, val in manova_res.items() if isinstance(manova_res, dict) else []:
                        pass
                except Exception:
                    pass
                # attempt generic extraction (may vary by statsmodels version)
                try:
                    # manova_res.results is often a dict-like
                    resdict = getattr(manova, "mv_test", None)
                    # we simply add textual summary
                    manova_text = str(manova.mv_test())
                except Exception:
                    manova_text = None

                results.append({
                    "test": "MANOVA (Analyse multivariée de variance)",
                    "result_df": None,
                    "fig": None,
                    "info": {"manova_summary": manova_text}
                })
        except Exception as e:
            results.append({"test": "MANOVA", "error": str(e)})

        # =========================================
        # 5️⃣ Régression multiple + diagnostics (améliorée)
        # =========================================
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
                    "IC Inf": model.conf_int()[0],
                    "IC Sup": model.conf_int()[1]
                })

                # p-values corrigées (FDR)
                try:
                    pvals = model.pvalues.values
                    _, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
                    summary_df["p-value FDR"] = pvals_corr
                except Exception:
                    summary_df["p-value FDR"] = np.nan

                # VIF (skip constant)
                try:
                    vif_df = pd.DataFrame({
                        "Variable": X.columns,
                        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    })
                except Exception as e:
                    vif_df = pd.DataFrame({"Variable": [], "VIF": []})

                # Résidus
                residuals = model.resid
                fitted = model.fittedvalues

                fig1, ax1 = plt.subplots(figsize=(6, 4))
                ax1.scatter(fitted, residuals, alpha=0.7)
                ax1.axhline(0, color='red', linestyle='--')
                ax1.set_xlabel("Valeurs ajustées")
                ax1.set_ylabel("Résidus")
                ax1.set_title("Résidus vs Valeurs ajustées")

                fig2 = sm.qqplot(residuals, line='s')
                plt.title("QQ-plot des résidus")

                # Tests statistiques sur les résidus
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

                results.append({
                    "test": "Régression multiple (OLS)",
                    "result_df": summary_df,
                    "fig": None,
                    "info": {"vif": _ensure_df(vif_df)}
                })
                results.append({
                    "test": "Analyse des résidus (diagnostic)",
                    "result_df": resid_summary,
                    "fig": fig1
                })
                results.append({
                    "test": "QQ-plot des résidus",
                    "result_df": None,
                    "fig": fig2
                })
            else:
                results.append({"test": "Régression multiple (OLS)", "error": "Aucune variable explicative numérique disponible."})
        except Exception as e:
            results.append({"test": "Régression / Résidus", "error": str(e)})

        # =========================================
        # 6️⃣ Corrélations multiples
        # =========================================
        try:
            corr_df = numeric_subset.corr(numeric_only=True)
            fig_corr, ax = plt.subplots(figsize=(6, 5))
            cax = ax.matshow(corr_df, cmap="coolwarm")
            plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
            plt.yticks(range(len(corr_df.columns)), corr_df.columns)
            fig_corr.colorbar(cax)
            ax.set_title("Matrice de corrélation")

            results.append({
                "test": "Corrélations multiples",
                "result_df": corr_df,
                "fig": fig_corr
            })
        except Exception as e:
            results.append({"test": "Corrélations multiples", "error": str(e)})

        # =========================================
        # 7️⃣ Diagnostics multivariés additionnels (optionnels)
        # =========================================
        # Multivariate normality (pingouin if available)
        try:
            import pingouin as pg
            try:
                numeric_for_mvn = numeric_subset.dropna()
                if numeric_for_mvn.shape[0] >= 10 and numeric_for_mvn.shape[1] >= 2:
                    mardia = pg.multivariate_normality(numeric_for_mvn, alpha=0.05)
                    mvn_df = pd.DataFrame({
                        "Skew": [mardia[0]],
                        "Kurtosis": [mardia[1]],
                        "p-value": [mardia[2]],
                        "Normal": [mardia[3]]
                    })
                    results.append({
                        "test": "Normalité multivariée (Pingouin Mardia)",
                        "result_df": mvn_df,
                        "fig": None
                    })
            except Exception as e_m:
                results.append({"test": "Normalité multivariée", "error": f"pingouin présent mais erreur: {e_m}"})
        except Exception:
            # pingouin absent -> skip quietly but add note
            results.append({"test": "Normalité multivariée", "info": "pingouin non installé — test multivarié non réalisé"})

        # Box's M (homogénéité des matrices covariance) - optional: try to compute via external lib or skip
        try:
            # Attempt to use bioinfokit if available
            from bioinfokit.analys import stat
            try:
                # requires categorical grouping, here we don't have multiple groups for MANOVA directly
                # we skip unless user provided grouping variable (not always the case)
                results.append({"test": "Box's M", "info": "Box's M not computed by default (requires grouping variable)."})
            except Exception as e_box:
                results.append({"test": "Box's M", "error": str(e_box)})
        except Exception:
            results.append({"test": "Box's M", "info": "bioinfokit non installé — test Box's M non réalisé"})

    except Exception as e:
        results.append({"test": "Global", "error": str(e)})

    return results
