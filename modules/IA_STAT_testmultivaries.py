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
    Analyse multivariée complète :
    - PCA, MCA, FAMD
    - MANOVA
    - Régression multiple + diagnostics
    - Corrélations
    - Normalité multivariée (Mardia)
    - Box’s M (si disponible)
    Retour : liste de dictionnaires uniformisés.
    """

    results = []

    # =========================================
    # Helpers internes
    # =========================================
    def safe_info(obj):
        """Transforme toute info non-dict en dict, pour éviter info.items() errors."""
        if isinstance(obj, dict):
            return obj
        elif isinstance(obj, str):
            return {"Information": obj}
        elif obj is None:
            return {"Information": "Aucune information disponible"}
        else:
            return {"Détail": str(obj)}

    # =========================================
    # Détermination des types
    # =========================================
    try:
        target_type = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
        explicative_types = types_df.loc[types_df["variable"].isin(explicatives), "type"].tolist()

        all_numeric = all(t == "numérique" for t in [target_type] + explicative_types)
        all_categorical = all(t == "catégorielle" for t in [target_type] + explicative_types)
        mixte = not all_numeric and not all_categorical

        subset = df[[target_var] + explicatives].dropna()
        numeric_subset = subset.select_dtypes(include=np.number)

    except Exception as e:
        return [{"test": "Global", "error": str(e)}]

    # ======================================================
    # 1️⃣ PCA (si toutes les variables sont numériques)
    # ======================================================
    if all_numeric:
        try:
            X = numeric_subset[explicatives]
            if X.shape[1] >= 2 and X.shape[0] >= 2:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                pca = PCA(n_components=min(2, X.shape[1]))
                pcs = pca.fit_transform(X_scaled)
                explained = pca.explained_variance_ratio_

                # Loadings
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
                contrib_df = pd.DataFrame(
                    loadings,
                    index=X.columns,
                    columns=[f"PC{i+1}" for i in range(loadings.shape[1])]
                )

                fig_circle, ax = plt.subplots(figsize=(6, 6))
                circle = plt.Circle((0, 0), 1, color="black", fill=False)
                ax.add_patch(circle)
                for i, var in enumerate(X.columns):
                    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                             head_width=0.02, length_includes_head=True)
                    ax.text(loadings[i, 0] * 1.05, loadings[i, 1] * 1.05, var)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_title("Cercle des corrélations (PCA)")

                # KMO
                try:
                    kmo_total, kmo_var = _kmo(X.values)
                    kmo_info = {
                        "KMO total": float(kmo_total),
                        "KMO par variable": kmo_var.tolist()
                    }
                except Exception as e_kmo:
                    kmo_info = {"Erreur": str(e_kmo)}

                results.append({
                    "test": "Analyse en Composantes Principales (PCA)",
                    "result_df": pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])]),
                    "fig": fig_circle,
                    "info": safe_info({
                        "Variance expliquée": explained.tolist(),
                        "Contributions": contrib_df.reset_index().to_dict(orient="records"),
                        "KMO": kmo_info
                    })
                })

        except Exception as e:
            results.append({"test": "PCA", "error": str(e)})

    # ======================================================
    # 2️⃣ MCA (si tout catégoriel)
    # ======================================================
    if all_categorical:
        try:
            subset_cat = subset.astype(str)
            mca = MCA(n_components=2, random_state=42)
            coords = mca.fit_transform(subset_cat)

            fig_mca, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1])
            ax.set_title("MCA - individus")

            results.append({
                "test": "Analyse des Correspondances Multiples (MCA)",
                "result_df": coords,
                "fig": fig_mca,
                "info": safe_info({"Inerties": getattr(mca, "explained_inertia_", None)})
            })
        except Exception as e:
            results.append({"test": "MCA", "error": str(e)})

    # ======================================================
    # 3️⃣ FAMD (variables mixtes)
    # ======================================================
    if mixte:
        try:
            famd = FAMD(n_components=2, random_state=42)
            coords = famd.fit_transform(subset)

            fig_famd, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1])
            ax.set_title("FAMD - individus")

            # contributions si disponibles
            try:
                contribs = famd.column_correlations(subset)
            except:
                contribs = None

            results.append({
                "test": "Analyse Factorielle Mixte (FAMD)",
                "result_df": coords,
                "fig": fig_famd,
                "info": safe_info({"Contributions": contribs})
            })
        except Exception as e:
            results.append({"test": "FAMD", "error": str(e)})

    # ======================================================
    # 4️⃣ MANOVA
    # ======================================================
    try:
        formula = f"{target_var} ~ " + " + ".join(explicatives)
        manova = MANOVA.from_formula(formula, data=subset)
        manova_text = str(manova.mv_test())

        results.append({
            "test": "MANOVA",
            "result_df": None,
            "fig": None,
            "info": safe_info({"Résumé": manova_text})
        })

    except Exception as e:
        results.append({"test": "MANOVA", "error": str(e)})

    # ======================================================
    # 5️⃣ Régression multiple + diagnostics
    # ======================================================
    try:
        X = subset[explicatives].select_dtypes(include=np.number)
        if not X.empty:
            Xc = sm.add_constant(X)
            y = subset[target_var]

            model = sm.OLS(y, Xc).fit()

            summary_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values,
                "IC Inf": model.conf_int()[0],
                "IC Sup": model.conf_int()[1]
            })

            # VIF
            try:
                vif_df = pd.DataFrame({
                    "Variable": X.columns,
                    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                })
            except:
                vif_df = pd.DataFrame()

            results.append({
                "test": "Régression multiple (OLS)",
                "result_df": summary_df,
                "fig": None,
                "info": safe_info({"VIF": vif_df.to_dict(orient="records")})
            })

            # Résidus
            residuals = model.resid
            fitted = model.fittedvalues

            fig_res, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(fitted, residuals)
            ax.axhline(0, color="red")
            ax.set_title("Résidus vs Valeurs ajustées")

            fig_qq = sm.qqplot(residuals, line='s')
            plt.title("QQ-plot")

            results.append({
                "test": "Analyse des résidus",
                "result_df": None,
                "fig": fig_res
            })
            results.append({
                "test": "QQ-plot des résidus",
                "result_df": None,
                "fig": fig_qq
            })

    except Exception as e:
        results.append({"test": "Régression / Résidus", "error": str(e)})

    # ======================================================
    # 6️⃣ Corrélations multiples
    # ======================================================
    try:
        corr_df = numeric_subset.corr()

        fig_corr, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(corr_df, cmap="coolwarm")
        fig_corr.colorbar(cax)
        ax.set_title("Matrice des corrélations")

        results.append({
            "test": "Corrélations multiples",
            "result_df": corr_df,
            "fig": fig_corr
        })
    except Exception as e:
        results.append({"test": "Corrélations multiples", "error": str(e)})

    # ======================================================
    # 7️⃣ Normalité multivariée (Mardia) **corrigée**
    # ======================================================
    try:
        X = numeric_subset.dropna()
        if X.shape[1] < 2:
            results.append({
                "test": "Normalité multivariée (Mardia)",
                "info": safe_info("Impossible : au moins 2 variables numériques nécessaires.")
            })
        else:
            from pingouin import multivariate_normality
            mardia = multivariate_normality(X, alpha=0.05)

            mardia_df = pd.DataFrame({
                "Statistique": [mardia['skewness'], mardia['kurtosis']],
                "p-value": [mardia['skew_kurt']['p_skew'], mardia['skew_kurt']['p_kurt']]
            }, index=["Skewness", "Kurtosis"])

            results.append({
                "test": "Normalité multivariée (Mardia)",
                "result_df": mardia_df,
                "fig": None,
                "info": safe_info({
                    "Décision": "Normale" if mardia["normal"] else "Non normale"
                })
            })

    except Exception as e:
        results.append({
            "test": "Normalité multivariée (Mardia)",
            "error": str(e),
            "info": safe_info("Erreur courante : colinéarité ou matrice de covariance non inversible.")
        })

    # ======================================================
    # 8️⃣ Box's M
    # ======================================================
    try:
        from bioinfokit.analys import stat
        results.append({
            "test": "Box's M",
            "info": safe_info("Box's M non calculé : aucune variable de regroupement fournie.")
        })
    except:
        results.append({
            "test": "Box's M",
            "info": safe_info("bioinfokit non installé.")
        })

    return results
