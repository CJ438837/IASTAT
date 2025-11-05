# modules/IA_STAT_testsmulti.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
import os


plt.style.use('seaborn-v0_8-muted')

def _ensure_numeric(df, columns):
    """Convertit les colonnes en numériques, remplace non-convertibles par NaN"""
    df_numeric = df[columns].copy()
    for col in columns:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    return df_numeric

def propose_tests_multivariés(df, types_df, target_var=None, predictor_vars=None, mots_cles=None, output_folder="tests_multivariés"):
    """
    Exécute automatiquement une série de tests multivariés pour une variable cible et des variables explicatives choisies :
    - Régression linéaire multiple (numérique cible)
    - Régression logistique (binaire cible)
    - PCA (variables numériques)
    - MCA (variables catégorielles ou binaires)
    
    Args:
        df (pd.DataFrame) : données
        types_df (pd.DataFrame) : colonnes "variable" et "type"
        target_var (str) : variable expliquée
        predictor_vars (list) : variables explicatives
        mots_cles (list) : mots clés facultatifs
        output_folder (str) : dossier de sauvegarde des figures
        
    Returns:
        results (list) : liste de dictionnaires {"test":..., "result_df":..., "fig":...}
    """
    import os
    _ensure_dir(output_folder)
    results = []

    # --- Normalisation noms colonnes ---
    rename_dict = {}
    for col in types_df.columns:
        lc = col.lower()
        if lc in {"variable", "var", "nom", "column", "name"}:
            rename_dict[col] = "variable"
        if lc in {"type", "var_type", "variable_type", "kind"}:
            rename_dict[col] = "type"
    types_df = types_df.rename(columns=rename_dict)

    # --- Vérifications ---
    if target_var is None:
        raise ValueError("Veuillez fournir une variable cible 'target_var'.")
    if predictor_vars is None or len(predictor_vars) == 0:
        raise ValueError("Veuillez fournir au moins une variable explicative 'predictor_vars'.")

    # --- Déterminer type de la cible ---
    target_type = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
    predictor_types = types_df.set_index("variable").loc[predictor_vars, "type"].to_dict()

    # --- Conversion en numérique si nécessaire ---
    num_predictors = [v for v in predictor_vars if predictor_types[v] == "numérique"]
    cat_predictors = [v for v in predictor_vars if predictor_types[v] in ["catégorielle", "binaire"]]

    if target_type == "numérique":
        # Régression linéaire multiple
        X = _ensure_numeric(df, num_predictors)
        y = pd.to_numeric(df[target_var], errors='coerce')
        common_index = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        if len(X) > 5:
            X = sm.add_constant(X)
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
            ax.set_title(f"Régression linéaire multiple ({target_var})")

            results.append({"test": f"Régression linéaire multiple ({target_var})",
                            "result_df": result_df,
                            "fig": fig})

    elif target_type in ["binaire", "catégorielle"]:
        # Régression logistique si cible binaire
        if df[target_var].nunique() == 2:
            X = _ensure_numeric(df, num_predictors)
            data = df[[target_var]+cat_predictors].copy()
            data = pd.get_dummies(data, drop_first=True)
            X = pd.concat([X, data.drop(columns=target_var)], axis=1)
            y = pd.to_numeric(df[target_var], errors='coerce')
            common_index = X.dropna().index.intersection(y.dropna().index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            if len(X) > 10:
                X = sm.add_constant(X)
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

                results.append({"test": f"Régression logistique ({target_var})",
                                "result_df": result_df,
                                "fig": fig})

    # --- PCA pour variables numériques ---
    num_vars = types_df[types_df["type"]=="numérique"]["variable"].tolist()
    if len(num_vars) > 1:
        X = _ensure_numeric(df, num_vars).dropna()
        try:
            X_scaled = StandardScaler().fit_transform(X)
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

            results.append({"test": "PCA",
                            "result_df": result_df,
                            "fig": fig})
        except Exception as e:
            print(f"Erreur PCA : {e}")

    # --- MCA pour variables catégorielles ---
    cat_vars = types_df[types_df["type"].isin(["catégorielle","binaire"])]["variable"].tolist()
    if len(cat_vars) > 1:
        try:
            import prince
            df_cat = df[cat_vars].dropna()
            mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
            coords = mca.column_coordinates(df_cat)
            result_df = coords.reset_index().rename(columns={"index":"Catégorie"})

            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(coords[0], coords[1], alpha=0.7)
            for i, label in enumerate(coords.index):
                ax.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=8)
            ax.set_title("MCA")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

            results.append({"test": "MCA",
                            "result_df": result_df,
                            "fig": fig})
        except Exception as e:
            print(f"Erreur MCA : {e}")

    return results

def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
