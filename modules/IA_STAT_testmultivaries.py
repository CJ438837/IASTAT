# modules/IA_STAT_testmultivaries.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import shapiro
from scipy import stats
import warnings

plt.style.use("seaborn-v0_8-muted")
warnings.filterwarnings("ignore")

# optional libs
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


# --- Helpers ---------------------------------------------------------

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
    """
    Garantit un format uniforme (dict) pour éviter .items() errors côté interface.
    """
    if obj is None:
        return {"info": "Aucune information supplémentaire."}
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.DataFrame):
        return {"table": obj.to_dict(orient="records")}
    return {"info": str(obj)}


def _kmo(X):
    """Kaiser-Meyer-Olkin measure."""
    try:
        corr = np.corrcoef(X.T)
        inv_corr = np.linalg.pinv(corr)
        partial = -inv_corr.copy()
        d = np.sqrt(np.abs(np.diag(partial)))
        partial = (partial / d).T / d
        np.fill_diagonal(partial, 0)
        a = corr.copy()
        np.fill_diagonal(a, 0)
        denom = np.sum(a ** 2) + np.sum(partial ** 2)
        if denom == 0:
            return np.nan, np.full(X.shape[1], np.nan)
        kmo_total = np.sum(a ** 2) / denom
        kmo_per_var = np.sum(a ** 2, axis=0) / (
            np.sum(a ** 2, axis=0) + np.sum(partial ** 2, axis=0)
        )
        return float(kmo_total), np.array(kmo_per_var, dtype=float)
    except Exception:
        return np.nan, np.full(X.shape[1], np.nan)


# --- MAIN FUNCTION ----------------------------------------------------

def propose_tests_multivariés(df, types_df, target_var, explicatives):

    results = []

    # Vérif variables
    if target_var not in df.columns:
        return [{"test": "Global", "error": f"Variable cible '{target_var}' introuvable."}]

    for v in explicatives:
        if v not in df.columns:
            return [{"test": "Global", "error": f"Variable explicative '{v}' introuvable."}]

    # Sous-ensemble propre
    subset = df[[target_var] + explicatives].dropna()
    numeric_subset = subset.select_dtypes(include=np.number)

    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------
    all_numeric = all(
        types_df.loc[types_df["variable"] == v, "type"].values[0] == "numérique"
        for v in [target_var] + explicatives
    )

    if all_numeric and numeric_subset.shape[1] >= 2:
        try:
            X = numeric_subset.copy()
            Xs = StandardScaler().fit_transform(X)

            pca = PCA(n_components=min(2, X.shape[1]))
            pcs = pca.fit_transform(Xs)

            # Graphique cercle corrélations
            fig = None
            if pca.n_components >= 2:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.add_patch(plt.Circle((0, 0), 1, fill=False))
                loadings = pca.components_.T
                for i, col in enumerate(X.columns):
                    ax.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                             head_width=0.02, length_includes_head=True)
                    ax.text(loadings[i, 0] * 1.1, loadings[i, 1] * 1.1, col)
                ax.set_title("Cercle des corrélations (PCA)")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

            kmo_total, kmo_per_var = _kmo(X.values)

            info = {
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "kmo_total": float(kmo_total),
                "kmo_per_variable": kmo_per_var.tolist(),
            }

            results.append({
                "test": "PCA",
                "result_df": pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])]),
                "fig": fig,
                "info": _safe_info(info)
            })

        except Exception as e:
            results.append({"test": "PCA", "error": str(e)})

    else:
        results.append({
            "test": "PCA",
            "info": _safe_info("PCA non applicable (variables non toutes numériques).")
        })

    # ---------------------------------------------------------
    # MCA
    # ---------------------------------------------------------
    all_cat = all(
        types_df.loc[types_df["variable"] == v, "type"].values[0] in ["catégorielle", "binaire"]
        for v in [target_var] + explicatives
    )

    if all_cat and PRINCE_AVAILABLE:
        try:
            subset_cat = subset.astype(str)
            mca = MCA(n_components=2, random_state=42)
            coords = mca.fit_transform(subset_cat)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], alpha=0.7)
            ax.set_title("MCA - individus")

            results.append({
                "test": "MCA",
                "result_df": coords,
                "fig": fig,
                "info": _safe_info({"inertia": getattr(mca, "explained_inertia_", None)})
            })
        except Exception as e:
            results.append({"test": "MCA", "error": str(e)})

    else:
        if all_cat:
            results.append({
                "test": "MCA",
                "info": _safe_info("Package 'prince' non installé.")
            })
        else:
            results.append({
                "test": "MCA",
                "info": _safe_info("MCA non applicable (variables non toutes catégorielles).")
            })

    # ---------------------------------------------------------
    # FAMD (mixte)
    # ---------------------------------------------------------
    mixed = not all_numeric and not all_cat
    if mixed and PRINCE_AVAILABLE:
        try:
            famd = FAMD(n_components=2, random_state=42)
            coords = famd.fit_transform(subset)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(coords.iloc[:, 0], coords.iloc[:, 1], alpha=0.7)
            ax.set_title("FAMD - individus")

            results.append({
                "test": "FAMD",
                "result_df": coords,
                "fig": fig,
                "info": _safe_info({"note": "Analyse FAMD (mixte)"})
            })
        except Exception as e:
            results.append({"test": "FAMD", "error": str(e)})
    else:
        results.append({
            "test": "FAMD",
            "info": _safe_info("FAMD non applicable (pas de mélange).")
        })

    # ---------------------------------------------------------
    # MANOVA
    # ---------------------------------------------------------
    if STATSMODELS_AVAILABLE:
        try:
            formula = f"{target_var} ~ " + " + ".join(explicatives)
            manova = MANOVA.from_formula(formula, data=subset)
            results.append({
                "test": "MANOVA",
                "result_df": None,
                "fig": None,
                "info": _safe_info({"summary": str(manova.mv_test())})
            })
        except Exception as e:
            results.append({"test": "MANOVA", "error": str(e)})
    else:
        results.append({
            "test": "MANOVA",
            "info": _safe_info("statsmodels non disponible.")
        })

    # ---------------------------------------------------------
    # Régression multiple + graphiques
    # ---------------------------------------------------------
    if STATSMODELS_AVAILABLE:

        X = subset[explicatives].select_dtypes(include=np.number)
        if X.shape[1] >= 1:
            try:
                Xc = sm.add_constant(X)
                y = subset[target_var]
                model = sm.OLS(y, Xc).fit()

                # Summary table
                summary_df = pd.DataFrame({
                    "Variable": model.params.index,
                    "Coefficient": model.params.values,
                    "p-value": model.pvalues.values,
                    "IC_low": model.conf_int().iloc[:, 0].values,
                    "IC_high": model.conf_int().iloc[:, 1].values,
                })

                # VIF
                vif_df = None
                if VIF_AVAILABLE:
                    vif_values = [
                        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
                    ]
                    vif_df = pd.DataFrame({"Variable": X.columns, "VIF": vif_values})

                # Graphiques OLS : Résidus + QQ-plot
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                # Residus vs fitted
                axes[0].scatter(model.fittedvalues, model.resid, alpha=0.7)
                axes[0].axhline(0, color="red", linestyle="--")
                axes[0].set_title("Résidus vs valeurs ajustées")
                axes[0].set_xlabel("Fitted")
                axes[0].set_ylabel("Residus")

                # QQ-plot
                sm.qqplot(model.resid, line="45", ax=axes[1])
                axes[1].set_title("QQ-plot des résidus")

                info = {
                    "vif": vif_df.to_dict(orient="records") if isinstance(vif_df, pd.DataFrame) else None,
                    "r2": float(model.rsquared),
                    "adj_r2": float(model.rsquared_adj),
                }

                results.append({
                    "test": "Régression multiple (OLS)",
                    "result_df": summary_df,
                    "fig": fig,
                    "info": _safe_info(info)
                })

            except Exception as e:
                results.append({"test": "Régression multiple (OLS)", "error": str(e)})

    else:
        results.append({
            "test": "Régression multiple (OLS)",
            "info": _safe_info("statsmodels non disponible")
        })

    # ---------------------------------------------------------
    # Corrélations multiples
    # ---------------------------------------------------------
    try:
        corr = numeric_subset.corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax)
        ax.set_title("Matrice de corrélation")

        results.append({
            "test": "Corrélations multiples",
            "result_df": corr,
            "fig": fig,
            "info": _safe_info({"note": "Corrélations entre variables numériques"})
        })
    except Exception as e:
        results.append({"test": "Corrélations multiples", "error": str(e)})

    # ---------------------------------------------------------
    # Normalité multivariée (Mardia)
    # ---------------------------------------------------------
    # Normalité multivariée (Mardia)
    if PINGOUIN_AVAILABLE:
       try:
          numeric_for_mvn = numeric_subset.dropna()
          if numeric_for_mvn.shape[0] >= 8 and numeric_for_mvn.shape[1] >= 2:
              m = pg.multivariate_normality(numeric_for_mvn, alpha=0.05)
            
            # Extraction sécurisée selon type renvoyé par pingouin
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

    return results
