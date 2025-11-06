import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

plt.style.use("seaborn-v0_8-muted")

def propose_tests_multivariés(df, types_df, distribution_df=None, target_var=None, explicatives=None):
    """
    Propose et exécute automatiquement les tests multivariés les plus pertinents :
    - Régression linéaire multiple
    - Régression logistique
    - PCA (analyse en composantes principales)
    - MCA (analyse des correspondances multiples)
    - FAMD (analyse factorielle mixte)
    """

    results = []

    # Vérification des entrées
    if df is None or types_df is None or target_var is None or explicatives is None:
        return [{"test": "Erreur", "result_df": pd.DataFrame(), "fig": None, 
                 "message": "Paramètres manquants (df, types_df, target_var, explicatives)"}]

    # Nettoyage des données
    df = df.dropna(subset=[target_var] + explicatives)
    if df.empty:
        return [{"test": "Erreur", "result_df": pd.DataFrame(), "fig": None,
                 "message": "Aucune donnée disponible après suppression des valeurs manquantes."}]

    # Détection des types
    type_target = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
    explicatives_types = {var: types_df.loc[types_df["variable"] == var, "type"].values[0] for var in explicatives}

    num_vars = [v for v, t in explicatives_types.items() if t == "numérique"]
    cat_vars = [v for v, t in explicatives_types.items() if t in ["catégorielle", "binaire"]]

    # ---------------------------------------------------------------------
    # 1️⃣ Régression linéaire multiple
    # ---------------------------------------------------------------------
    if type_target == "numérique" and len(num_vars) > 0:
        try:
            X = df[num_vars].apply(pd.to_numeric, errors="coerce").dropna()
            y = pd.to_numeric(df[target_var], errors="coerce")
            y = y.loc[X.index]
            X = sm.add_constant(X)

            model = sm.OLS(y, X).fit()

            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values
            })
            result_df["R² ajusté"] = model.rsquared_adj

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(x=model.fittedvalues, y=y, ax=ax, alpha=0.7)
            ax.plot(y, y, color="red", linestyle="--")
            ax.set_title(f"Régression linéaire multiple ({target_var})")
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Valeurs réelles")

            results.append({
                "test": "Régression linéaire multiple",
                "result_df": result_df,
                "fig": fig
            })
        except Exception as e:
            results.append({"test": "Régression linéaire multiple", "error": str(e)})

    # ---------------------------------------------------------------------
    # 2️⃣ Régression logistique
    # ---------------------------------------------------------------------
    if type_target == "binaire" and (len(num_vars) + len(cat_vars)) > 0:
        try:
            df_enc = df.copy()
            for v in cat_vars:
                df_enc[v] = df_enc[v].astype("category").cat.codes
            X = df_enc[num_vars + cat_vars]
            y = df_enc[target_var].astype("category").cat.codes
            X = sm.add_constant(X)

            model = sm.Logit(y, X).fit(disp=False)
            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values
            })
            result_df["Pseudo R²"] = model.prsquared

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(model.predict(X), kde=True, ax=ax)
            ax.set_title(f"Régression logistique ({target_var})")
            ax.set_xlabel("Probabilité prédite")

            results.append({
                "test": "Régression logistique",
                "result_df": result_df,
                "fig": fig
            })
        except Exception as e:
            results.append({"test": "Régression logistique", "error": str(e)})

    # ---------------------------------------------------------------------
    # 3️⃣ PCA (Analyse en composantes principales)
    # ---------------------------------------------------------------------
    if len(num_vars) > 1:
        try:
            X_scaled = StandardScaler().fit_transform(df[num_vars].apply(pd.to_numeric, errors="coerce").dropna())
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_.cumsum()

            result_df = pd.DataFrame({
                "Composante": [f"PC{i+1}" for i in range(len(explained_var))],
                "Variance cumulée": explained_var
            })

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(range(1, len(explained_var)+1), explained_var, marker="o")
            ax.set_title("Analyse en composantes principales (PCA)")
            ax.set_xlabel("Composantes principales")
            ax.set_ylabel("Variance expliquée cumulée")

            results.append({
                "test": "Analyse en composantes principales (PCA)",
                "result_df": result_df,
                "fig": fig
            })
        except Exception as e:
            results.append({"test": "PCA", "error": str(e)})

    # ---------------------------------------------------------------------
    # 4️⃣ MCA (Analyse des correspondances multiples)
    # ---------------------------------------------------------------------
    if len(cat_vars) > 1:
        try:
            import prince
            df_cat = df[cat_vars].dropna()
            mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
            coords = mca.column_coordinates(df_cat)

            result_df = coords.reset_index().rename(columns={"index": "Catégorie"})

            fig, ax = plt.subplots(figsize=(5, 4))
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
        except Exception as e:
            results.append({"test": "MCA", "error": str(e)})

    # ---------------------------------------------------------------------
    # 5️⃣ FAMD (Analyse Factorielle Mixte)
    # ---------------------------------------------------------------------
    if len(num_vars) > 0 and len(cat_vars) > 0:
        try:
            import prince
            df_mix = df[num_vars + cat_vars].dropna()
            famd = prince.FAMD(n_components=2, random_state=42).fit(df_mix)
            coords = famd.row_coordinates(df_mix)
            explained_var = famd.explained_inertia_

            result_df = pd.DataFrame({
                "Composante": [f"Dim{i+1}" for i in range(len(explained_var))],
                "Variance expliquée": explained_var
            })

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(coords[0], coords[1], alpha=0.7)
            ax.set_title("Analyse Factorielle Mixte (FAMD)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")

            results.append({
                "test": "Analyse Factorielle Mixte (FAMD)",
                "result_df": result_df,
                "fig": fig
            })
        except Exception as e:
            results.append({"test": "FAMD", "error": str(e)})

    # ---------------------------------------------------------------------
    # Fin
    # ---------------------------------------------------------------------
    if not results:
        results.append({"test": "Aucun test applicable", "result_df": pd.DataFrame(), "fig": None})

    return results
