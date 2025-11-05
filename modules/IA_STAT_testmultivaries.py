# modules/IA_STAT_testsmulti.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

plt.style.use('seaborn-v0_8-muted')

def propose_tests_multivariés(df, types_df, distribution_df=None,
                              target_var=None, predictor_vars=None):
    """
    Exécute des tests multivariés :
    - Régression linéaire multiple pour target numérique
    - Régression logistique pour target binaire
    - PCA sur variables numériques
    - MCA sur variables catégorielles/binaire
    """

    results = []

    # --- Détection des types ---
    types_df = types_df.rename(columns=lambda x: x.lower())
    if "variable" not in types_df.columns or "type" not in types_df.columns:
        raise ValueError("types_df doit contenir 'variable' et 'type'")

    num_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
    cat_vars = types_df[types_df["type"].isin(["catégorielle", "binaire"])]["variable"].tolist()
    bin_vars = types_df[types_df["type"] == "binaire"]["variable"].tolist()

    # --- Si target_var non défini, prendre la première variable numérique ou binaire ---
    if target_var is None:
        if len(num_vars) > 0:
            target_var = num_vars[0]
        elif len(bin_vars) > 0:
            target_var = bin_vars[0]
        else:
            raise ValueError("Aucune variable cible valide trouvée")

    # --- Si predictor_vars non défini, prendre toutes sauf target_var ---
    if predictor_vars is None:
        predictor_vars = [v for v in df.columns if v != target_var]

    # --- Régression linéaire ---
    if target_var in num_vars:
        X = df[predictor_vars].copy()
        y = pd.to_numeric(df[target_var], errors='coerce')

        # Conversion forcée en numérique
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # Supprimer lignes avec NaN
        common_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        if len(X) > 5:
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
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

            results.append({
                "test": f"Régression linéaire multiple ({target_var})",
                "result_df": result_df,
                "fig": fig
            })

    # --- Régression logistique ---
    elif target_var in bin_vars:
        data = df[predictor_vars + [target_var]].copy()
        for col in predictor_vars:
            if col in num_vars:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        data = data.dropna()
        if len(data) > 10:
            X = sm.add_constant(data[predictor_vars])
            y = data[target_var].astype("category").cat.codes
            model = sm.Logit(y, X).fit(disp=False)
            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values
            })
            result_df["Pseudo R²"] = model.prsquared

            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(model.predict(X), kde=True, ax=ax)
            ax.set_title(f"Régression logistique ({target_var})")
            ax.set_xlabel("Probabilité prédite")

            results.append({
                "test": f"Régression logistique ({target_var})",
                "result_df": result_df,
                "fig": fig
            })

    # --- PCA ---
    num_vars_for_pca = [v for v in num_vars if v in df.columns]
    if len(num_vars_for_pca) > 1:
        X_num = df[num_vars_for_pca].copy()
        for col in X_num.columns:
            X_num[col] = pd.to_numeric(X_num[col], errors='coerce')
        X_num = X_num.dropna()
        if len(X_num) > 1:
            pca = PCA()
            components = pca.fit_transform(StandardScaler().fit_transform(X_num))
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

    # --- MCA ---
    cat_vars_for_mca = [v for v in cat_vars if v in df.columns]
    if len(cat_vars_for_mca) > 1:
        try:
            import prince
            df_cat = df[cat_vars_for_mca].dropna()
            if len(df_cat) > 1:
                mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
                coords = mca.column_coordinates(df_cat)
                result_df = coords.reset_index().rename(columns={"index": "Catégorie"})

                fig, ax = plt.subplots(figsize=(6,4))
                ax.scatter(coords[0], coords[1], alpha=0.7)
                for i, label in enumerate(coords.index):
                    ax.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=8)
                ax.set_title("Analyse des correspondances multiples (MCA)")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")

                results.append({
                    "test": "Analyse des correspondances multiples (MCA)",
                    "result_df": result_df,
                    "fig": fig
                })
        except ImportError:
            print("Module prince non installé.")
        except Exception as e:
            print(f"Erreur MCA : {e}")

    return results
