# modules/IA_STAT_testsmulti.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

plt.style.use('seaborn-v0_8-muted')


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def propose_tests_multivariés(df, types_df, target_var=None, predictor_vars=None, output_folder="tests_multivariés"):
    """
    Exécute une série de tests multivariés :
    - Régression linéaire multiple si target_var numérique
    - Régression logistique si target_var binaire
    - PCA pour variables numériques
    - MCA pour variables catégorielles/binaire

    Args:
        df (pd.DataFrame) : données
        types_df (pd.DataFrame) : colonnes 'variable' et 'type'
        target_var (str) : variable à expliquer
        predictor_vars (list) : variables explicatives
        output_folder (str) : dossier pour sauvegarder les figures

    Returns:
        results (list of dict) : chaque dict contient {"test":..., "result_df":..., "fig":...}
    """
    _ensure_dir(output_folder)
    results = []

    # Normaliser noms de colonnes
    types_df = types_df.rename(columns={col: col.lower() for col in types_df.columns})
    types_map = dict(zip(types_df['variable'], types_df['type']))

    # Sélection des variables
    if target_var is None:
        raise ValueError("Veuillez spécifier target_var (variable à expliquer).")
    if predictor_vars is None:
        predictor_vars = [v for v in df.columns if v != target_var]

    # Filtrer les colonnes existantes
    predictor_vars = [v for v in predictor_vars if v in df.columns]

    # Nettoyage : suppression des lignes avec NaN dans target ou prédicteurs
    data = df[[target_var] + predictor_vars].copy().dropna()
    if len(data) == 0:
        raise ValueError("Pas de données valides après suppression des NaN.")

    target_type = types_map.get(target_var, "numérique")

    # ---------------- Régression linéaire ----------------
    if target_type == "numérique":
        X = data[predictor_vars].select_dtypes(include=[np.number]).astype(float)
        y = data[target_var].astype(float)

        if len(X) > 0:
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
            ax.set_title(f"Régression linéaire ({target_var})")

            results.append({
                "test": f"Régression linéaire multiple ({target_var})",
                "result_df": result_df,
                "fig": fig
            })

    # ---------------- Régression logistique ----------------
    elif target_type == "binaire":
        X = pd.get_dummies(data[predictor_vars], drop_first=True)
        y = data[target_var].astype("category").cat.codes

        if len(X) > 0 and y.nunique() == 2:
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

            results.append({
                "test": f"Régression logistique ({target_var})",
                "result_df": result_df,
                "fig": fig
            })

    # ---------------- PCA ----------------
    num_vars = types_df[types_df['type']=="numérique"]["variable"].tolist()
    num_vars = [v for v in num_vars if v in df.columns]
    if len(num_vars) > 1:
        try:
            data_num = df[num_vars].dropna().astype(float)
            X_scaled = StandardScaler().fit_transform(data_num)
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_.cumsum()

            result_df = pd.DataFrame({
                "Composante": [f"PC{i+1}" for i in range(len(explained_var))],
                "Variance cumulée": explained_var
            })

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(range(1,len(explained_var)+1), explained_var, marker="o")
            ax.set_title("PCA - Variance expliquée cumulée")
            ax.set_xlabel("Composantes principales")
            ax.set_ylabel("Variance expliquée cumulée")

            results.append({"test":"PCA", "result_df":result_df, "fig":fig})
        except Exception as e:
            print(f"Erreur PCA : {e}")

    # ---------------- MCA ----------------
    cat_vars = types_df[types_df['type'].isin(["catégorielle","binaire"])]['variable'].tolist()
    cat_vars = [v for v in cat_vars if v in df.columns]
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
                ax.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=8)
            ax.set_title("MCA - Projection catégories")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

            results.append({"test":"MCA","result_df":result_df,"fig":fig})
        except ImportError:
            print("Module prince non installé.")
        except Exception as e:
            print(f"Erreur MCA : {e}")

    return results
