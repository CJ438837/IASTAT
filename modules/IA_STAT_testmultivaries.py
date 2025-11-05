# modules/IA_STAT_testsmultivaries.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

# Optional: prince pour MCA
try:
    import prince
    PRINCE_AVAILABLE = True
except ImportError:
    PRINCE_AVAILABLE = False

plt.style.use('seaborn-v0_8-muted')


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def propose_tests_multivariés(df, types_df, distribution_df, target_var, predictors, output_folder="tests_multivariés"):
    """
    Réalise un test multivarié à la fois en fonction du type de variable cible.
    - df : DataFrame des données
    - types_df : colonnes 'variable' et 'type' ("numérique", "binaire", "catégorielle", "ordinale")
    - target_var : variable dépendante
    - predictors : liste de variables explicatives
    - output_folder : dossier pour sauvegarder les figures
    Retourne : dict {"test":..., "result_df":..., "fig":...}
    """
    _ensure_dir(output_folder)
    results = {}

    # Déterminer le type de la variable cible
    target_type = types_df.loc[types_df['variable'] == target_var, 'type'].values[0]

    # Nettoyer les données pour lignes complètes
    cols = predictors + [target_var]
    data = df[cols].dropna()

    if data.empty:
        raise ValueError("Pas de données complètes pour les variables sélectionnées.")

    # ------------------ Régression linéaire ------------------
    if target_type == "numérique":
        X = sm.add_constant(data[predictors])
        y = data[target_var]
        model = sm.OLS(y, X).fit()
        result_df = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "p-value": model.pvalues.values,
            "R² ajusté": [model.rsquared_adj] + [None]*(len(model.params)-1)
        })

        # Graphique valeurs réelles vs prédites
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(model.fittedvalues, y, alpha=0.7)
        ax.plot(y, y, color="red", linestyle="--")
        ax.set_xlabel("Valeurs prédites")
        ax.set_ylabel("Valeurs réelles")
        ax.set_title(f"Régression linéaire multiple ({target_var})")

        results = {"test": f"Régression linéaire multiple ({target_var})", "result_df": result_df, "fig": fig}

    # ------------------ Régression logistique binaire ------------------
    elif target_type == "binaire":
        if data[target_var].nunique() != 2:
            raise ValueError("Variable binaire doit avoir exactement 2 modalités.")
        X = sm.add_constant(data[predictors])
        y = data[target_var].astype("category").cat.codes
        model = sm.Logit(y, X).fit(disp=False)
        result_df = pd.DataFrame({
            "Variable": model.params.index,
            "Coefficient": model.params.values,
            "p-value": model.pvalues.values
        })
        result_df["Pseudo R²"] = model.prsquared

        # Graphique probas prédites
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(model.predict(X), kde=True, ax=ax)
        ax.set_title(f"Régression logistique ({target_var})")
        ax.set_xlabel("Probabilité prédite")

        results = {"test": f"Régression logistique ({target_var})", "result_df": result_df, "fig": fig}

    # ------------------ Régression logistique multinomiale ------------------
    elif target_type == "catégorielle" and PRINCE_AVAILABLE:
        if data[target_var].nunique() < 2:
            raise ValueError("Variable catégorielle doit avoir au moins 2 modalités.")
        X = sm.add_constant(pd.get_dummies(data[predictors], drop_first=True))
        y = pd.Categorical(data[target_var])
        model = sm.MNLogit(y.codes, X).fit(disp=False)
        result_df = pd.DataFrame({
            "Variable": X.columns,
            **{f"Category_{cat}": model.params.iloc[:, i].values for i, cat in enumerate(y.categories)}
        })

        # Graphique probas prédites pour chaque catégorie
        pred_probs = model.predict(X)
        fig, ax = plt.subplots(figsize=(6,4))
        pred_probs.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title(f"Régression logistique multinomiale ({target_var})")
        ax.set_xlabel("Observations")
        ax.set_ylabel("Probabilité prédite")

        results = {"test": f"Régression logistique multinomiale ({target_var})", "result_df": result_df, "fig": fig}

    else:
        # PCA pour toutes les variables numériques si cible n'est pas sélectionnable
        num_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
        if len(num_vars) > 1:
            X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
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

            results = {"test": "Analyse en composantes principales (PCA)", "result_df": result_df, "fig": fig}

    # ------------------ MCA pour variables catégorielles ------------------
    cat_vars = types_df[types_df["type"].isin(["catégorielle","binaire"])]["variable"].tolist()
    if PRINCE_AVAILABLE and len(cat_vars) > 1:
        df_cat = df[cat_vars].dropna()
        mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
        coords = mca.column_coordinates(df_cat)
        result_df = coords.reset_index().rename(columns={"index":"Catégorie"})

        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(coords[0], coords[1], alpha=0.7)
        for i, label in enumerate(coords.index):
            ax.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=8)
        ax.set_title("Analyse des correspondances multiples (MCA)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        results["MCA"] = {"test":"MCA", "result_df":result_df, "fig":fig}

    return results
