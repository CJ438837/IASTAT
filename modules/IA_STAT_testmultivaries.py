import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use("seaborn-v0_8-muted")

def propose_tests_multivariés(df, types_df, distribution_df=None, target_var=None, explicatives=None):
    """
    Propose et exécute automatiquement les tests multivariés les plus pertinents :
    - Régression linéaire multiple
    - Régression logistique
    - PCA (analyse en composantes principales)
    - MCA (analyse des correspondances multiples)
    - FAMD (analyse factorielle mixte)
    
    ➕ Fournit des indicateurs de qualité (R² ajusté, VIF, intervalles de confiance, interprétation)
    ➕ Graphiques esthétiques et annotations claires
    ➕ Résumé interprétatif automatique
    """

    results = []

    # =============================
    # 🔍 Vérifications et nettoyage
    # =============================
    if df is None or types_df is None or target_var is None or explicatives is None:
        return [{"test": "Erreur", "result_df": pd.DataFrame(), "fig": None,
                 "message": "Paramètres manquants (df, types_df, target_var, explicatives)"}]

    df = df.dropna(subset=[target_var] + explicatives)
    if df.empty:
        return [{"test": "Erreur", "result_df": pd.DataFrame(), "fig": None,
                 "message": "Aucune donnée disponible après suppression des valeurs manquantes."}]

    # Détection des types
    type_target = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
    explicatives_types = {var: types_df.loc[types_df["variable"] == var, "type"].values[0] for var in explicatives}

    num_vars = [v for v, t in explicatives_types.items() if t == "numérique"]
    cat_vars = [v for v, t in explicatives_types.items() if t in ["catégorielle", "binaire"]]

    # Fonction utilitaire
    def summarize_significance(pval):
        if pval < 0.001: return "*** (hautement significatif)"
        elif pval < 0.01: return "** (significatif)"
        elif pval < 0.05: return "* (tendance)"
        else: return "ns (non significatif)"

    # =====================================================
    # 1️⃣ RÉGRESSION LINÉAIRE MULTIPLE (cible numérique)
    # =====================================================
    if type_target == "numérique" and len(num_vars) > 0:
        try:
            X = df[num_vars].apply(pd.to_numeric, errors="coerce").dropna()
            y = pd.to_numeric(df[target_var], errors="coerce")
            y = y.loc[X.index]
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()

            # VIF (colinéarité)
            vif_data = pd.DataFrame({
                "Variable": X.columns,
                "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            })

            result_df = pd.DataFrame({
                "Variable": model.params.index,
                "Coefficient": model.params.values,
                "p-value": model.pvalues.values,
                "IC-95% Inf": model.conf_int()[0],
                "IC-95% Sup": model.conf_int()[1]
            })
            result_df["Signification"] = result_df["p-value"].apply(summarize_significance)
            result_df["R² ajusté"] = round(model.rsquared_adj, 3)

            # Graphique : valeurs prédites vs réelles
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=model.fittedvalues, y=y, ax=ax, alpha=0.7)
            ax.plot(y, y, color="red", linestyle="--")
            ax.set_title(f"Régression linéaire multiple : {target_var}")
            ax.set_xlabel("Valeurs prédites")
            ax.set_ylabel("Valeurs réelles")
            ax.grid(True, alpha=0.3)

            interpretation = (
                f"La régression linéaire explique {model.rsquared_adj*100:.1f}% de la variance de {target_var}. "
                f"Les variables les plus significatives sont : "
                f"{', '.join(result_df.loc[result_df['p-value'] < 0.05, 'Variable'].tolist()) or 'aucune'}."
            )

            results.append({
                "test": "Régression linéaire multiple",
                "result_df": result_df.merge(vif_data, on="Variable", how="left"),
                "fig": fig,
                "interpretation": interpretation
            })

        except Exception as e:
            results.append({"test": "Régression linéaire multiple", "error": str(e)})

    # =====================================================
    # 2️⃣ RÉGRESSION LOGISTIQUE (cible binaire)
    # =====================================================
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
                "p-value": model.pvalues.values,
                "Odds Ratio": np.exp(model.params.values)
            })
            result_df["Signification"] = result_df["p-value"].apply(summarize_significance)
            result_df["Pseudo R² (McFadden)"] = round(model.prsquared, 3)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(model.predict(X), kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Régression logistique : {target_var}")
            ax.set_xlabel("Probabilité prédite")
            ax.set_ylabel("Densité")

            interpretation = (
                f"La régression logistique montre un pseudo R² de {model.prsquared*100:.1f}%. "
                f"Les variables avec effet significatif sont : "
                f"{', '.join(result_df.loc[result_df['p-value'] < 0.05, 'Variable'].tolist()) or 'aucune'}."
            )

            results.append({
                "test": "Régression logistique",
                "result_df": result_df,
                "fig": fig,
                "interpretation": interpretation
            })
        except Exception as e:
            results.append({"test": "Régression logistique", "error": str(e)})

    # =====================================================
    # 3️⃣ PCA - Analyse en composantes principales
    # =====================================================
    if len(num_vars) > 1:
        try:
            X_scaled = StandardScaler().fit_transform(df[num_vars].apply(pd.to_numeric, errors="coerce").dropna())
            pca = PCA()
            components = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_.cumsum()

            result_df = pd.DataFrame({
                "Composante": [f"PC{i+1}" for i in range(len(explained_var))],
                "Variance cumulée (%)": np.round(explained_var * 100, 2)
            })

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.lineplot(x=range(1, len(explained_var)+1), y=explained_var*100, marker="o", ax=ax)
            ax.set_title("Analyse en composantes principales (PCA)")
            ax.set_xlabel("Composantes principales")
            ax.set_ylabel("Variance expliquée cumulée (%)")
            ax.grid(True, alpha=0.3)

            interpretation = (
                f"Les {np.argmax(explained_var >= 0.8) + 1} premières composantes expliquent "
                f"{explained_var[np.argmax(explained_var >= 0.8)]*100:.1f}% de la variance totale."
            )

            results.append({
                "test": "Analyse en composantes principales (PCA)",
                "result_df": result_df,
                "fig": fig,
                "interpretation": interpretation
            })
        except Exception as e:
            results.append({"test": "PCA", "error": str(e)})

    # =====================================================
    # 4️⃣ MCA - Analyse des correspondances multiples
    # =====================================================
    if len(cat_vars) > 1:
        try:
            import prince
            df_cat = df[cat_vars].dropna()
            mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
            coords = mca.column_coordinates(df_cat)

            result_df = coords.reset_index().rename(columns={"index": "Catégorie"})
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(coords[0], coords[1], alpha=0.7, color="teal")
            for i, label in enumerate(coords.index):
                ax.text(coords.iloc[i, 0], coords.iloc[i, 1], label, fontsize=8)
            ax.set_title("Analyse des correspondances multiples (MCA)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True, alpha=0.3)

            interpretation = (
                "L'analyse des correspondances multiples révèle les associations entre catégories. "
                "Les modalités proches sur le graphique sont souvent co-observées dans les données."
            )

            results.append({
                "test": "Analyse des correspondances multiples (MCA)",
                "result_df": result_df,
                "fig": fig,
                "interpretation": interpretation
            })
        except Exception as e:
            results.append({"test": "MCA", "error": str(e)})

    # =====================================================
    # 5️⃣ FAMD - Analyse factorielle mixte
    # =====================================================
    if len(num_vars) > 0 and len(cat_vars) > 0:
        try:
            import prince
            df_mix = df[num_vars + cat_vars].dropna()
            famd = prince.FAMD(n_components=2, random_state=42).fit(df_mix)
            coords = famd.row_coordinates(df_mix)
            explained_var = getattr(famd, "explained_inertia_", None)

            result_df = pd.DataFrame({
                "Composante": [f"Dim{i+1}" for i in range(len(explained_var))],
                "Variance expliquée (%)": np.round(np.array(explained_var)*100, 2)
            })

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(coords[0], coords[1], alpha=0.7, color="darkorange")
            ax.set_title("Analyse Factorielle Mixte (FAMD)")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True, alpha=0.3)

            interpretation = (
                "L'analyse factorielle mixte permet de visualiser la structure conjointe des variables "
                "numériques et catégorielles. Les individus proches ont un profil similaire."
            )

            results.append({
                "test": "Analyse Factorielle Mixte (FAMD)",
                "result_df": result_df,
                "fig": fig,
                "interpretation": interpretation
            })
        except Exception as e:
            results.append({"test": "FAMD", "error": str(e)})

    # =====================================================
    # ✅ Aucun test applicable
    # =====================================================
    if not results:
        results.append({"test": "Aucun test applicable", "result_df": pd.DataFrame(), "fig": None,
                        "interpretation": "Aucun test n'a pu être réalisé avec les variables fournies."})

    return results
