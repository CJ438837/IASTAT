# modules/IA_STAT_testmultivaries.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
import warnings

plt.style.use("seaborn-v0_8-muted")

# optional libraries
try:
    from prince import MCA, FAMD
    PRINCE_AVAILABLE = True
except Exception:
    MCA = None
    FAMD = None
    PRINCE_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.multivariate.manova import MANOVA
    STATSMODELS_AVAILABLE = True
except Exception:
    sm = None
    MANOVA = None
    STATSMODELS_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    VIF_AVAILABLE = True
except Exception:
    variance_inflation_factor = None
    VIF_AVAILABLE = False

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except Exception:
    pg = None
    PINGOUIN_AVAILABLE = False

warnings.filterwarnings("ignore")


# --- Helpers ---
def _ensure_df(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None


def _safe_info(obj):
    """Retourne un dict uniforme pour 'info' afin d'éviter les erreurs côté UI."""
    if obj is None:
        return {"info": "Aucune information supplémentaire."}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.DataFrame):
        return {"table": obj.to_dict(orient="records")}
    return {"info": str(obj)}


def _kmo(X):
    """Kaiser-Meyer-Olkin measure of sampling adequacy. Returns (kmo_total, kmo_per_variable)."""
    try:
        corr = np.corrcoef(X.T)
        inv_corr = np.linalg.pinv(corr)
        partial = -inv_corr.copy()
        d = np.sqrt(np.abs(np.diag(partial)))
        partial = (partial / d).T / d
        np.fill_diagonal(partial, 0.0)
        a = corr.copy()
        np.fill_diagonal(a, 0.0)
        denom = np.sum(a ** 2) + np.sum(partial ** 2)
        if denom == 0:
            return np.nan, np.full(X.shape[1], np.nan)
        kmo_total = np.sum(a ** 2) / denom
        kmo_per_var = np.sum(a ** 2, axis=0) / (np.sum(a ** 2, axis=0) + np.sum(partial ** 2, axis=0))
        return float(kmo_total), np.array(kmo_per_var, dtype=float)
    except Exception:
        return np.nan, np.full(X.shape[1], np.nan)


# === Fonction principale ===
def propose_tests_multivariés(df, types_df, target_var, explicatives):
    """
    Analyse multivariée complète.
    Retourne une LISTE de dicts :
      - 'test', 'result_df', 'fig', 'info', 'interpretation' (optionnel), 'error' (optionnel)
    """
    results = []
    try:
        # Validation minimale
        if target_var not in df.columns:
            return [{"test": "Global", "error": f"Variable cible '{target_var}' introuvable."}]
        for v in explicatives:
            if v not in df.columns:
                return [{"test": "Global", "error": f"Variable explicative '{v}' introuvable."}]

        subset = df[[target_var] + explicatives].dropna()
        numeric_subset = subset.select_dtypes(include=np.number)

        # === PCA (tout numérique) ===
        all_numeric = all(
            types_df.loc[types_df["variable"] == v, "type"].values[0] == "numérique"
            for v in [target_var] + explicatives if v in types_df["variable"].values
        )
        if all_numeric and numeric_subset.shape[1] >= 2 and numeric_subset.shape[0] >= 2:
            try:
                X = numeric_subset.copy()
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                n_comp = min(2, X.shape[1])
                pca = PCA(n_components=n_comp)
                pcs = pca.fit_transform(Xs)
                explained = pca.explained_variance_ratio_.tolist()
                loadings = (pca.components_.T * np.sqrt(pca.explained_variance_)).tolist()

                # Cercle des corrélations
                fig = None
                try:
                    if n_comp >= 2:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        circle = plt.Circle((0, 0), 1, color="black", fill=False)
                        ax.add_patch(circle)
                        load = np.array(loadings)
                        for i, col in enumerate(X.columns):
                            ax.arrow(0, 0, load[i, 0], load[i, 1],
                                     head_width=0.02, length_includes_head=True)
                            ax.text(load[i, 0] * 1.05, load[i, 1] * 1.05, col)
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                        ax.set_title("Cercle des corrélations (PCA)")
                except Exception:
                    fig = None

                kmo_total, kmo_per_var = _kmo(X.values)
                info = {
                    "explained_variance": explained,
                    "loadings": [{ "variable": c, "loadings": list(loadings[i]) } for i, c in enumerate(X.columns)],
                    "kmo_total": float(kmo_total) if not np.isnan(kmo_total) else None,
                    "kmo_per_variable": (kmo_per_var.tolist() if not np.all(np.isnan(kmo_per_var)) else None)
                }
                results.append({
                    "test": "PCA",
                    "result_df": pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])]),
                    "fig": fig,
                    "info": _safe_info(info)
                })
            except Exception as e:
                results.append({"test": "PCA", "error": str(e)})

        # === MCA (tout catégoriel) ===
        all_categorical = all(
            types_df.loc[types_df["variable"] == v, "type"].values[0] in ["catégorielle", "binaire"]
            for v in [target_var] + explicatives if v in types_df["variable"].values
        )
        if all_categorical and PRINCE_AVAILABLE:
            try:
                subset_cat = subset.astype(str)
                mca = MCA(n_components=2, random_state=42)
                coords = mca.fit_transform(subset_cat)
                fig = None
                try:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], alpha=0.7)
                    ax.set_title("MCA - individus")
                except Exception:
                    fig = None
                info = {"explained_inertia": getattr(mca, "explained_inertia_", None)}
                results.append({
                    "test": "MCA",
                    "result_df": _ensure_df(coords),
                    "fig": fig,
                    "info": _safe_info(info)
                })
            except Exception as e:
                results.append({"test": "MCA", "error": str(e)})

        # === FAMD (mixte) ===
        mixed = not all_numeric and not all_categorical
        if mixed and PRINCE_AVAILABLE:
            try:
                famd = FAMD(n_components=2, random_state=42)
                coords = famd.fit_transform(subset)
                fig = None
                try:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], alpha=0.7)
                    ax.set_title("FAMD - individus")
                except Exception:
                    fig = None
                info = {"note": "FAMD exécuté (prince)."}
                results.append({
                    "test": "FAMD",
                    "result_df": _ensure_df(coords),
                    "fig": fig,
                    "info": _safe_info(info)
                })
            except Exception as e:
                results.append({"test": "FAMD", "error": str(e)})

        # === MANOVA (si possible) ===
        if STATSMODELS_AVAILABLE:
            try:
                formula = f"{target_var} ~ " + " + ".join(explicatives)
                manova = MANOVA.from_formula(formula, data=subset)
                results.append({
                    "test": "MANOVA",
                    "result_df": None,
                    "fig": None,
                    "info": _safe_info({"manova_summary": str(manova.mv_test())})
                })
            except Exception as e:
                results.append({"test": "MANOVA", "error": str(e)})

        # === Régression multiple (OLS) + diagnostics ===
        try:
            if STATSMODELS_AVAILABLE:
                X = subset[explicatives].select_dtypes(include=np.number)
                if X.shape[1] > 0:
                    Xc = sm.add_constant(X)
                    y = subset[target_var]
                    model = sm.OLS(y, Xc).fit()
                    summary_df = pd.DataFrame({
                        "Variable": model.params.index,
                        "Coefficient": model.params.values,
                        "p-value": model.pvalues.values,
                        "IC_low": model.conf_int().iloc[:, 0].values,
                        "IC_high": model.conf_int().iloc[:, 1].values
                    })
                    # VIF si dispo
                    vif_df = None
                    if VIF_AVAILABLE:
                        try:
                            vif = [float(variance_inflation_factor(X.values, i)) for i in range(X.shape[1])]
                            vif_df = pd.DataFrame({"Variable": X.columns, "VIF": vif})
                        except Exception:
                            vif_df = None

                    # Graphique résidus
                    fig_res = None
                    try:
                        resid = model.resid
                        fitted = model.fittedvalues
                        fig_res, ax = plt.subplots(figsize=(6, 4))
                        ax.scatter(fitted, resid, alpha=0.7)
                        ax.axhline(0, color="red", linestyle="--")
                        ax.set_xlabel("Fitted")
                        ax.set_ylabel("Residuals")
                        ax.set_title("Residuals vs Fitted")
                    except Exception:
                        fig_res = None

                    # Test Shapiro sur résidus
                    sh = shapiro(model.resid) if len(model.resid) >= 3 else (np.nan, np.nan)

                    info = {
                        "vif_table": vif_df.to_dict(orient="records") if isinstance(vif_df, pd.DataFrame) else None,
                        "residual_tests": {"shapiro_stat": float(sh[0]), "shapiro_p": float(sh[1])}
                    }

                    results.append({
                        "test": "Régression multiple (OLS)",
                        "result_df": _ensure_df(summary_df),
                        "fig": fig_res,
                        "info": _safe_info(info)
                    })
                else:
                    results.append({"test": "Régression multiple (OLS)", "info": _safe_info("Pas de variable explicative numérique.")})
            else:
                results.append({"test": "Régression multiple (OLS)", "info": _safe_info("statsmodels non disponible.")})
        except Exception as e:
            results.append({"test": "Régression / Résidus", "error": str(e)})

        # === Corrélations multiples ===
        try:
            corr_df = numeric_subset.corr()
            fig_corr = None
            try:
                fig_corr, ax = plt.subplots(figsize=(6, 5))
                cax = ax.matshow(corr_df, cmap="coolwarm")
                fig_corr.colorbar(cax)
                ax.set_title("Matrice de corrélation")
            except Exception:
                fig_corr = None
            results.append({
                "test": "Corrélations multiples",
                "result_df": _ensure_df(corr_df),
                "fig": fig_corr,
                "info": _safe_info({"note": "Matrice de corrélation entre variables numériques."})
            })
        except Exception as e:
            results.append({"test": "Corrélations multiples", "error": str(e)})

        # === Normalité multivariée (Mardia) ===
        if PINGOUIN_AVAILABLE:
            try:
                numeric_for_mvn = numeric_subset.dropna()
                if numeric_for_mvn.shape[0] >= 8 and numeric_for_mvn.shape[1] >= 2:
                    m = pg.multivariate_normality(numeric_for_mvn, alpha=0.05)
                    if hasattr(m, 'normal'):  # objet HZResults
                        mvn_info = {
                            "normal": bool(m.normal),
                            "HZ_stat": float(getattr(m, "HZ", np.nan)),
                            "pvalue": float(getattr(m, "pval", np.nan))
                        }
                    elif isinstance(m, dict):  # dict dans certaines versions
                        mvn_info = {
                            "normal": bool(m.get("normal", False)),
                            "skewness": float(m.get("skew", np.nan)),
                            "kurtosis": float(m.get("kurtosis", np.nan)),
                            "pvalue": float(m.get("pval", np.nan))
                        }
                    else:
                        mvn_info = {"note": str(m)}
                else:
                    mvn_info = {"note": "Taille insuffisante pour test Mardia."}
                results.append({
                    "test": "Normalité multivariée (Mardia)",
                    "result_df": None,
                    "fig": None,
                    "info": _safe_info(mvn_info)
                })
            except Exception as e:
                results.append({"test": "Normalité multivariée (Mardia)", "error": str(e)})
        else:
            results.append({
                "test": "Normalité multivariée (Mardia)",
                "info": _safe_info("pingouin non installé.")
            })

    except Exception as e:
        results = [{"test": "Global", "error": str(e)}]

    return results
