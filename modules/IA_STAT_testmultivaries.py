# modules/IA_STAT_testmultivaries.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

plt.style.use("seaborn-v0_8-muted")


def propose_tests_multivariés(df, types_df, distribution_df=None, mots_cles=None):
    """
    Exécute une série de tests multivariés et retourne une liste de résultats.
    Chaque résultat : {"test": ..., "result_df": ..., "fig": ...}
    """
    results = []

    # Colonnes par type
    num_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
    bin_vars = types_df[types_df["type"] == "binaire"]["variable"].tolist()
    cat_vars = types_df[types_df["type"] == "catégorielle"]["variable"].tolist()

    # --- 1️⃣ Régression linéaire multiple ---
    if len(num_vars) >= 2:
        for target in num_vars:
            predictors = [v for v in num_vars if v != target]
            data = df[predictors + [target]].dropna()
            if len(data) > 5:
                X = sm.add_constant(data[predictors])
                y = data[target]
                model = sm.OLS(y, X).fit()

                result_df = pd.DataFrame({
                    "Variable": model.params.index,
                    "Coefficient": model.params.values,
                    "p-value": model.pvalues.values,
                    "R² ajusté": [model.rsquared_adj] + [None]*(len(model.params)-1)
                })

                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(model.fittedvalues, y, alpha=0.7)
                ax.plot(y, y, color="red", linestyle="--")
                ax.set_xlabel("Valeurs prédites")
                ax.set_ylabel("Valeurs réelles")
                ax.set_title(f"Régression linéaire multiple ({target})")

                results.append({
                    "test": f"Régression linéaire multiple ({target})",
                    "result_df": result_df,
                    "fig": fig
                })

    # --- 2️⃣ Régression logistique ---
    if len(bin_vars) > 0 and len(num_vars) > 0:
        for target in bin_vars:
            predictors = num_vars.copy()
            data = df[predictors + [target]].dropna()
            if len(data) > 10 and data[target].nunique() == 2:
                X = sm.add_constant(data[predictors])
                y = data[target].astype("category").cat.codes
                model = sm.Logit(y, X).fit(disp=False)

                result_df = pd.DataFrame({
                    "Variable": model.params.index,
                    "Coefficient": model.params.values,
                    "p-value": model.pvalues.values,
                    "Pseudo R²": [model.prsquared] + [None]*(len(model.params)-1)
                })

                fig, ax = plt.subplots(figsize=(6,4))
                sns.histplot(model.predict(X), kde=True, ax=ax)
                ax.set_title(f"Régression logistique ({target})")
                ax.set_xlabel("Probabilité prédite")

                results.append({
                    "test": f"Régression logistique ({target})",
                    "result_df": result_df,
                    "fig": fig
                })

    # --- 3️⃣ Régression multinomiale ---
    for target in cat_vars:
        predictors = num_vars.copy()
        data = df[predictors + [target]].dropna()
        if len(data[target].unique()) > 2 and len(data) > 10:
            try:
                X = sm.add_constant(data[predictors])
                y = data[target].astype("category")
                model = sm.MNLogit(y.cat.codes, X).fit(disp=False)

                result_df = model.params.reset_index()
                fig, ax = plt.subplots(figsize=(6,4))
                ax.axis("off")
                ax.set_title(f"Régression multinomiale ({target})")

                results.append({
                    "test": f"Régression multinomiale ({target})",
                    "result_df": result_df,
                    "fig": fig
                })
            except Exception as e:
                print(f"Erreur régression multinomiale {target}: {e}")

    # --- 4️⃣ PCA ---
    if len(num_vars) > 1:
        try:
            data = df[num_vars].dropna()
            X_scaled = StandardScaler().fit_transform(data)
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_.cumsum()

            result_df = pd.DataFrame({
                "Composante": [f"PC{i+1}" for i in range(len(explained_var))],
                "Variance cumulée": explained_var
            })

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(range(1, len(explained_var)+1), explained_var, marker="o")
            ax.set_title("PCA - Variance expliquée cumulée")
            ax.set_xlabel("Composantes principales")
            ax.set_ylabel("Variance expliquée cumulée")

            results.append({
                "test": "Analyse en composantes principales (PCA)",
                "result_df": result_df,
                "fig": fig
            })
        except Exception as e:
            print(f"Erreur PCA: {e}")

    # --- 5️⃣ MCA ---
    cat_all = cat_vars + bin_vars
    if len(cat_all) > 1:
        try:
            import prince
            df_cat = df[cat_all].dropna()
            mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
            coords = mca.column_coordinates(df_cat)

            result_df = coords.reset_index().rename(columns={"index": "Catégorie"})
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(coords[0], coords[1], alpha=0.7)
            for i, label in enumerate(coords.index):
                ax.text(coords.iloc[i, 0], coords.iloc[i, 1], label, fontsize=8)
            ax.set_title("Analyse des correspondances multiples (MCA)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

            results.append({
                "test": "Analyse des correspondances multiples (MCA)",
                "result_df": result_df,
                "fig": fig
            })
        except ImportError:
            print("Module prince non installé pour MCA.")
        except Exception as e:
            print(f"Erreur MCA: {e}")

    return results
