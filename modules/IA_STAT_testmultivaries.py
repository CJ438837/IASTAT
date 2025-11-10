import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-muted")

def propose_tests_multivariés(df, types_df, target_var, explicatives):
    """
    Exécute et propose automatiquement les tests multivariés pertinents
    selon les types des variables sélectionnées.
    Retourne une liste de dictionnaires :
    - test : nom du test
    - result_df : tableau de résultats
    - fig : figure matplotlib
    - interpretation : résumé en français
    - error / message : messages éventuels
    """

    results = []

    try:
        # --- Validation des entrées ---
        if df is None or len(df) == 0:
            return [{"error": "Le DataFrame est vide ou non défini."}]
        if target_var not in df.columns:
            return [{"error": f"La variable cible '{target_var}' est absente du DataFrame."}]
        if not explicatives:
            return [{"error": "Aucune variable explicative n’a été sélectionnée."}]

        df = df.copy().dropna(subset=[target_var] + explicatives)

        # --- Déterminer les types ---
        target_type = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
        explicative_types = {
            v: types_df.loc[types_df["variable"] == v, "type"].values[0] for v in explicatives
        }

        num_vars = [v for v, t in explicative_types.items() if t == "numérique"]
        cat_vars = [v for v, t in explicative_types.items() if t == "catégorielle"]

        # ---------------------------------------------------------------------
        # 1️⃣ Régression linéaire multiple
        # ---------------------------------------------------------------------
        if target_type == "numérique" and len(num_vars) > 0:
            try:
                formula = f"{target_var} ~ " + " + ".join(num_vars)
                model = smf.ols(formula, data=df).fit()

                fig, ax = plt.subplots()
                sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax, color="steelblue")
                ax.set_title("Résidus vs Valeurs ajustées")
                ax.set_xlabel("Valeurs ajustées")
                ax.set_ylabel("Résidus")

                result_df = pd.DataFrame(model.summary2().tables[1])
                interpretation = (
                    f"Le modèle explique **{model.rsquared_adj*100:.1f}%** de la variance de {target_var}. "
                    "Les p-values permettent de juger la significativité des coefficients."
                )

                results.append({
                    "test": "Régression Linéaire Multiple",
                    "result_df": result_df,
                    "fig": fig,
                    "interpretation": interpretation
                })
            except Exception as e:
                results.append({"test": "Régression Linéaire Multiple", "error": str(e)})

        # ---------------------------------------------------------------------
        # 2️⃣ Régression logistique binaire
        # ---------------------------------------------------------------------
        if target_type == "catégorielle" and df[target_var].nunique() == 2 and len(num_vars) > 0:
            try:
                df[target_var] = df[target_var].astype("category").cat.codes
                formula = f"{target_var} ~ " + " + ".join(num_vars)
                model = smf.logit(formula, data=df).fit(disp=False)

                fig, ax = plt.subplots()
                sns.histplot(model.predict(), bins=20, kde=True, ax=ax)
                ax.set_title("Distribution des probabilités prédites")
                ax.set_xlabel("Probabilité prédite")

                result_df = pd.DataFrame(model.summary2().tables[1])
                interpretation = (
                    f"Les coefficients représentent l’influence sur la probabilité que {target_var}=1. "
                    f"Pseudo R² = {model.prsquared:.3f}. Une p-valeur < 0.05 indique un effet significatif."
                )

                results.append({
                    "test": "Régression Logistique Binaire",
                    "result_df": result_df,
                    "fig": fig,
                    "interpretation": interpretation
                })
            except Exception as e:
                results.append({"test": "Régression Logistique", "error": str(e)})

        # ---------------------------------------------------------------------
        # 3️⃣ ANOVA / MANOVA
        # ---------------------------------------------------------------------
        if target_type == "numérique" and len(cat_vars) > 0:
            try:
                formula = f"{target_var} ~ " + " + ".join(cat_vars)
                model = smf.ols(formula, data=df).fit()
                anova_res = sm.stats.anova_lm(model, typ=2)

                fig, ax = plt.subplots()
                sns.boxplot(data=df, x=cat_vars[0], y=target_var, ax=ax)
                ax.set_title("ANOVA - Effet des catégories")
                interpretation = (
                    "L’ANOVA teste si les moyennes diffèrent significativement entre les groupes. "
                    "Une p-valeur < 0.05 suggère une différence significative."
                )

                results.append({
                    "test": "ANOVA (Analyse de la Variance)",
                    "result_df": anova_res,
                    "fig": fig,
                    "interpretation": interpretation
                })
            except Exception as e:
                results.append({"test": "ANOVA", "error": str(e)})

        # ---------------------------------------------------------------------
        # 4️⃣ Corrélations
        # ---------------------------------------------------------------------
        if len(num_vars) > 1:
            try:
                corr = df[num_vars].corr()
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                ax.set_title("Matrice de corrélation (Pearson)")
                results.append({
                    "test": "Corrélations (Pearson)",
                    "result_df": corr,
                    "fig": fig,
                    "interpretation": "Les coefficients proches de ±1 indiquent une forte corrélation."
                })
            except Exception as e:
                results.append({"test": "Corrélations", "error": str(e)})

        # ---------------------------------------------------------------------
        # 5️⃣ Chi² pour variables catégorielles
        # ---------------------------------------------------------------------
        if target_type == "catégorielle" and len(cat_vars) > 0:
            for v in cat_vars:
                try:
                    table = pd.crosstab(df[target_var], df[v])
                    chi2, p, dof, _ = stats.chi2_contingency(table)
                    interpretation = (
                        f"Le test du Chi² entre {target_var} et {v} donne une p-valeur de {p:.4f}. "
                        f"{'→ Association significative' if p < 0.05 else '→ Pas d’association significative'}."
                    )

                    results.append({
                        "test": f"Test du Chi² ({target_var} ~ {v})",
                        "result_df": table,
                        "fig": None,
                        "interpretation": interpretation
                    })
                except Exception as e:
                    results.append({"test": "Chi²", "error": str(e)})

        # ---------------------------------------------------------------------
        # 6️⃣ PCA (Analyse en Composantes Principales)
        # ---------------------------------------------------------------------
        if len(num_vars) > 1:
            try:
                X_scaled = StandardScaler().fit_transform(df[num_vars])
                pca = PCA()
                pca_fit = pca.fit(X_scaled)
                explained_var = pca.explained_variance_ratio_.cumsum()

                result_df = pd.DataFrame({
                    "Composante": [f"PC{i+1}" for i in range(len(explained_var))],
                    "Variance cumulée": explained_var
                })

                fig, ax = plt.subplots()
                ax.plot(range(1, len(explained_var)+1), explained_var, marker="o")
                ax.set_title("Analyse en Composantes Principales (PCA)")
                ax.set_xlabel("Composantes")
                ax.set_ylabel("Variance expliquée cumulée")

                results.append({
                    "test": "Analyse en Composantes Principales (PCA)",
                    "result_df": result_df,
                    "fig": fig,
                    "interpretation": f"Les {min(2, len(num_vars))} premières composantes expliquent "
                                     f"{pca.explained_variance_ratio_[:2].sum()*100:.1f}% de la variance."
                })
            except Exception as e:
                results.append({"test": "PCA", "error": str(e)})

        # ---------------------------------------------------------------------
        # 7️⃣ MCA (Analyse des Correspondances Multiples)
        # ---------------------------------------------------------------------
        if len(cat_vars) > 1:
            try:
                import prince
                df_cat = df[cat_vars].dropna()
                mca = prince.MCA(n_components=2, random_state=42).fit(df_cat)
                coords = mca.column_coordinates(df_cat)

                fig, ax = plt.subplots()
                ax.scatter(coords[0], coords[1], alpha=0.7)
                for i, label in enumerate(coords.index):
                    ax.text(coords.iloc[i, 0], coords.iloc[i, 1], label, fontsize=8)
                ax.set_title("Analyse des Correspondances Multiples (MCA)")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")

                result_df = coords.reset_index().rename(columns={"index": "Catégorie"})
                results.append({
                    "test": "Analyse des Correspondances Multiples (MCA)",
                    "result_df": result_df,
                    "fig": fig,
                    "interpretation": "La MCA révèle les associations entre modalités catégorielles."
                })
            except Exception as e:
                results.append({"test": "MCA", "error": str(e)})

        # ---------------------------------------------------------------------
        # 8️⃣ FAMD (Analyse Factorielle Mixte)
        # ---------------------------------------------------------------------
        if len(num_vars) > 0 and len(cat_vars) > 0:
            try:
                import prince
                df_mix = df[num_vars + cat_vars].dropna()
                famd = prince.FAMD(n_components=2, random_state=42).fit(df_mix)
                coords = famd.row_coordinates(df_mix)

                fig, ax = plt.subplots()
                ax.scatter(coords[0], coords[1], alpha=0.6)
                ax.set_title("Analyse Factorielle Mixte (FAMD)")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")

                explained_var = famd.explained_inertia_
                result_df = pd.DataFrame({
                    "Composante": [f"Dim{i+1}" for i in range(len(explained_var))],
                    "Variance expliquée": explained_var
                })

                results.append({
                    "test": "Analyse Factorielle Mixte (FAMD)",
                    "result_df": result_df,
                    "fig": fig,
                    "interpretation": "La FAMD combine variables numériques et catégorielles dans un même espace."
                })
            except Exception as e:
                results.append({"test": "FAMD", "error": str(e)})

        if not results:
            results.append({"message": "Aucun test multivarié applicable à cette configuration."})

    except Exception as e:
        results.append({"error": f"Erreur globale : {e}"})

    return results
