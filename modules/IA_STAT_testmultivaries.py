# modules/IA_STAT_testsmultivaries.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

plt.style.use('seaborn-v0_8-muted')

def _ensure_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def propose_tests_multivariés(df, types_df, distribution_df=None, mots_cles=None,
                              output_folder="tests_multivariés"):
    """
    Exécute une série de tests multivariés et retourne les résultats avec graphiques.
    - df : DataFrame des données
    - types_df : DataFrame avec colonnes 'variable' et 'type' ("numérique","binaire","catégorielle","ordinale")
    - distribution_df : optional, non utilisé ici
    - mots_cles : optional, non utilisé ici
    - output_folder : dossier pour enregistrer les figures
    """
    _ensure_dir(output_folder)
    results = []

    # Listes de variables selon type
    num_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
    bin_vars = types_df[types_df["type"] == "binaire"]["variable"].tolist()
    cat_vars = types_df[types_df["type"].isin(["catégorielle","ordinale"])].tolist()

    # --------------------------------------
    # 1️⃣ Régression : variable cible numérique
    # --------------------------------------
    for target in num_vars:
        predictors = [v for v in df.columns if v != target]
        data = df[[target]+predictors].dropna()

        if len(data) < 5:
            continue

        X = data[predictors].copy()
        y = data[target].astype(float)

        # Convertir colonnes numériques en float
        for col in X.select_dtypes(include=['number']).columns:
            X[col] = X[col].astype(float)
        # Encoder les colonnes catégorielles
        X = pd.get_dummies(X, drop_first=True)
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values,
                "R² ajusté": [model.rsquared_adj]+[None]*(len(model.params)-1)
            })

            # Graphique : réel vs prédit
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(model.fittedvalues, y, alpha=0.7)
            ax.plot(y, y, color="red", linestyle="--")
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Valeurs réelles")
            ax.set_title(f"Régression linéaire multiple ({target})")

            results.append({"test": f"Régression linéaire ({target})",
                            "result_df": result_df, "fig": fig})
        except Exception as e:
            print(f"Erreur régression linéaire {target}: {e}")

    # --------------------------------------
    # 2️⃣ Régression logistique binaire
    # --------------------------------------
    for target in bin_vars:
        predictors = [v for v in df.columns if v != target]
        data = df[[target]+predictors].dropna()
        if len(data) < 10 or data[target].nunique() != 2:
            continue

        X = data[predictors].copy()
        y = data[target].astype("category").cat.codes

        # Encoder colonnes catégorielles
        X = pd.get_dummies(X, drop_first=True)
        X = sm.add_constant(X)

        try:
            model = sm.Logit(y, X).fit(disp=False)
            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values,
                "Pseudo R²": [model.prsquared]+[None]*(len(model.params)-1)
            })

            fig, ax = plt.subplots(figsize=(6,4))
            sns.histplot(model.predict(X), kde=True, ax=ax)
            ax.set_xlabel("Probabilité prédite")
            ax.set_title(f"Régression logistique ({target})")

            results.append({"test": f"Régression logistique binaire ({target})",
                            "result_df": result_df, "fig": fig})
        except Exception as e:
            print(f"Erreur régression logistique {target}: {e}")

    # --------------------------------------
    # 3️⃣ Régression logistique multinomiale
    # --------------------------------------
    for target in cat_vars:
        if target in bin_vars:  # skip binary, déjà fait
            continue
        predictors = [v for v in df.columns if v != target]
        data = df[[target]+predictors].dropna()
        if len(data[target].unique()) < 3:
            continue

        X = pd.get_dummies(data[predictors], drop_first=True)
        X = sm.add_constant(X)
        y = data[target].astype("category")

        try:
            model = sm.MNLogit(y.cat.codes, X).fit(disp=False)
            result_df = pd.DataFrame(model.params)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.set_title(f"Régression multinomiale ({target})")
            results.append({"test": f"Régression multinomiale ({target})",
                            "result_df": result_df, "fig": fig})
        except Exception as e:
            print(f"Erreur régression multinomiale {target}: {e}")

    # --------------------------------------
    # 4️⃣ PCA pour variables numériques
    # --------------------------------------
    if len(num_vars) > 1:
        try:
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
            results.append({"test": "PCA", "result_df": result_df, "fig": fig})
        except Exception as e:
            print(f"Erreur PCA: {e}")

    # --------------------------------------
    # 5️⃣ MCA pour variables catégorielles
    # --------------------------------------
    cat_all = types_df[types_df["type"].isin(["catégorielle","binaire","ordinale"])]["variable"].tolist()
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
                ax.text(coords.iloc[i,0], coords.iloc[i,1], label, fontsize=8)
            ax.set_title("Analyse des correspondances multiples (MCA)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            results.append({"test": "MCA", "result_df": result_df, "fig": fig})
        except ImportError:
            print("Module prince non installé pour MCA")
        except Exception as e:
            print(f"Erreur MCA: {e}")

    return results
