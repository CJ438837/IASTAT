# modules/IA_STAT_testmultivaries.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

plt.style.use("seaborn-v0_8-muted")


def propose_tests_multivariés(df, types_df, distribution_df, mots_cles=None,
                              target_var=None, predictor_vars=None):
    """
    Exécute automatiquement une série de tests multivariés pour une variable cible et ses prédicteurs :
    - Régression linéaire multiple si target_var numérique
    - Régression logistique si target_var binaire
    - ACP (PCA) sur variables numériques
    - ACM (MCA) sur variables catégorielles
    """

    results = []

    if target_var is None or predictor_vars is None or len(predictor_vars) == 0:
        raise ValueError("Veuillez fournir target_var et au moins un predictor_var")

    # Récupérer types des variables
    type_dict = types_df.set_index("variable")["type"].to_dict()
    target_type = type_dict.get(target_var, "numérique")

    # Sélection des prédicteurs
    predictor_types = [type_dict.get(p, "numérique") for p in predictor_vars]

    # --- Régression linéaire ---
    if target_type == "numérique":
        X = df[predictor_vars].dropna()
        y = df[target_var].dropna()
        # garder seulement les lignes présentes dans X et y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        if len(X) > 5:
            import statsmodels.api as sm
            Xc = sm.add_constant(X)
            model = sm.OLS(y, Xc).fit()
            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values,
                "R² ajusté": [model.rsquared_adj] + [None]*(len(model.params)-1)
            })

            # Graphique
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(model.fittedvalues, y, alpha=0.7)
            ax.plot(y, y, color="red", linestyle="--")
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Valeurs réelles")
            ax.set_title(f"Régression linéaire multiple ({target_var})")

            results.append({"test": f"Régression linéaire multiple ({target_var})",
                            "result_df": result_df, "fig": fig})

    # --- Régression logistique ---
    elif target_type == "binaire":
        X = df[predictor_vars].dropna()
        y = df[target_var].dropna()
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        if len(X) > 5:
            import statsmodels.api as sm
            Xc = sm.add_constant(X)
            y_bin = y.astype("category").cat.codes
            model = sm.Logit(y_bin, Xc).fit(disp=False)
            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values
            })
            result_df["Pseudo R²"] = model.prsquared

            # Graphique
            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(model.predict(Xc), kde=True, ax=ax)
            ax.set_title(f"Régression logistique ({target_var})")
            ax.set_xlabel("Probabilité prédite")

            results.append({"test": f"Régression logistique ({target_var})",
                            "result_df": result_df, "fig": fig})

    # --- PCA sur variables numériques si plus d'une ---
    num_vars = [v for v, t in type_dict.items() if t=="numérique"]
    if len(num_vars) > 1:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            X_num = df[num_vars].dropna()
            X_scaled = StandardScaler().fit_transform(X_num)
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

            results.append({"test": "Analyse en composantes principales (PCA)",
                            "result_df": result_df, "fig": fig})
        except Exception as e:
            print(f"Erreur PCA : {e}")

    # --- MCA sur variables catégorielles si plus d'une ---
    cat_vars = [v for v, t in type_dict.items() if t in ["catégorielle","binaire"]]
    if len(cat_vars) > 1:
        try:
            import prince
            df_cat = df[cat_vars].dropna()
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

            results.append({"test": "Analyse des correspondances multiples (MCA)",
                            "result_df": result_df, "fig": fig})
        except Exception as e:
            print(f"Erreur MCA : {e}")

    return results

