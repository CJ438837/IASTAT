import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from prince import MCA, FAMD
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import pearsonr

def propose_tests_multivariés(df, types_df, target_var, explicatives):
    results = []

    try:
        # --- Vérification et typage des variables ---
        target_type = types_df.loc[types_df["variable"] == target_var, "type"].values[0]
        explicative_types = types_df.loc[types_df["variable"].isin(explicatives), "type"].tolist()

        all_numeric = all(t == "numérique" for t in [target_type] + explicative_types)
        all_categorical = all(t == "catégorielle" for t in [target_type] + explicative_types)
        mixte = not all_numeric and not all_categorical

        # --- Nettoyage des données ---
        subset = df[[target_var] + explicatives].dropna()

        # =========================================
        # 🧮 1️⃣ PCA - Analyse en Composantes Principales
        # =========================================
        if all_numeric:
            try:
                X = subset[explicatives].select_dtypes(include=np.number)
                X_scaled = StandardScaler().fit_transform(X)

                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(X_scaled)
                explained_var = pca.explained_variance_ratio_

                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.6)
                ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
                ax.set_title("Analyse en Composantes Principales (PCA)")

                results.append({
                    "test": "Analyse en Composantes Principales (PCA)",
                    "result_df": pd.DataFrame({
                        "Composante 1": principal_components[:, 0],
                        "Composante 2": principal_components[:, 1],
                        "Variable cible": subset[target_var].values
                    }),
                    "fig": fig
                })
            except Exception as e:
                results.append({"test": "PCA", "error": str(e)})

        # =========================================
        # 🧩 2️⃣ MCA - Analyse des Correspondances Multiples
        # =========================================
        elif all_categorical:
            try:
                subset_cat = subset.astype(str)
                mca = MCA(n_components=2, random_state=42)
                mca_result = mca.fit(subset_cat)

                coords = mca.transform(subset_cat)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(coords[0], coords[1], alpha=0.6)
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_title("Analyse des Correspondances Multiples (MCA)")

                results.append({
                    "test": "Analyse des Correspondances Multiples (MCA)",
                    "result_df": coords,
                    "fig": fig
                })
            except Exception as e:
                results.append({"test": "MCA", "error": str(e)})

        # =========================================
        # ⚗️ 3️⃣ FAMD - Analyse Factorielle Mixte
        # =========================================
        elif mixte:
            try:
                famd = FAMD(n_components=2, random_state=42)
                famd_result = famd.fit(subset)

                coords = famd.transform(subset)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(coords[0], coords[1], alpha=0.6)
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_title("Analyse Factorielle Mixte (FAMD)")

                results.append({
                    "test": "Analyse Factorielle Mixte (FAMD)",
                    "result_df": coords,
                    "fig": fig
                })
            except Exception as e:
                results.append({"test": "FAMD", "error": str(e)})

        # =========================================
        # 🧪 4️⃣ MANOVA - Variance multivariée
        # =========================================
        try:
            if all_numeric or mixte:
                formula = f"{target_var} ~ " + " + ".join(explicatives)
                manova = MANOVA.from_formula(formula, data=subset)
                manova_res = manova.mv_test()

                results.append({
                    "test": "MANOVA (Analyse multivariée de variance)",
                    "result_df": pd.DataFrame(manova_res.results),
                    "fig": None
                })
        except Exception as e:
            results.append({"test": "MANOVA", "error": str(e)})

        # =========================================
        # 📊 5️⃣ Régression multiple (OLS)
        # =========================================
        try:
            X = subset[explicatives].select_dtypes(include=np.number)
            if not X.empty:
                X = sm.add_constant(X)
                y = subset[target_var]

                model = sm.OLS(y, X).fit()
                summary_df = pd.DataFrame({
                    "Variable": model.params.index,
                    "Coefficient": model.params.values,
                    "p-value": model.pvalues.values,
                    "IC Inf": model.conf_int()[0],
                    "IC Sup": model.conf_int()[1]
                })

                results.append({
                    "test": "Régression multiple (OLS)",
                    "result_df": summary_df,
                    "fig": None
                })
        except Exception as e:
            results.append({"test": "Régression multiple", "error": str(e)})

        # =========================================
        # 🔗 6️⃣ Corrélations multiples
        # =========================================
        try:
            corr_df = subset.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            cax = ax.matshow(corr_df, cmap="coolwarm")
            plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45)
            plt.yticks(range(len(corr_df.columns)), corr_df.columns)
            fig.colorbar(cax)
            ax.set_title("Matrice de corrélation")

            results.append({
                "test": "Corrélations multiples",
                "result_df": corr_df,
                "fig": fig
            })
        except Exception as e:
            results.append({"test": "Corrélations multiples", "error": str(e)})

    except Exception as e:
        results.append({"test": "Global", "error": str(e)})

    return results
