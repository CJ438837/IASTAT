import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd

def executer_test(df, test_dict, apparie=False):
    """
    Exécute un test statistique et génère les graphiques associés.
    
    df : DataFrame
    test_dict : dictionnaire d'un test généré par propose_tests_interactif_auto
    apparie : bool (si applicable)
    """
    t_type = test_dict["type"]
    vars_ = test_dict["variables"]

    # --- Numérique vs Catégoriel ---
    if t_type in ["t-test", "Mann-Whitney", "ANOVA", "Kruskal-Wallis"]:
        num = vars_[0]
        cat = vars_[1]
        groupes = df.groupby(cat)[num].apply(list)

        if t_type == "t-test":
            stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
        elif t_type == "Mann-Whitney":
            stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
        elif t_type == "ANOVA":
            stat, p = stats.f_oneway(*groupes)
        elif t_type == "Kruskal-Wallis":
            stat, p = stats.kruskal(*groupes)
        else:
            stat, p = None, None

        # Graphiques
        sns.boxplot(x=cat, y=num, data=df)
        plt.title(f"{t_type} : {num} vs {cat}")
        plt.show()

        return {"stat": stat, "p": p}

    # --- Corrélation ---
    elif "Corrélation" in t_type:
        var1, var2 = vars_
        if "Pearson" in t_type:
            corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
        else:
            corr, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
        sns.scatterplot(x=var1, y=var2, data=df)
        plt.title(t_type)
        plt.show()
        return {"stat": corr, "p": p}

    # --- Chi2 / Fisher ---
    elif t_type == "Chi2/Fisher":
        var1, var2 = vars_
        table = pd.crosstab(df[var1], df[var2])
        if table.size <= 4:
            stat, p = stats.fisher_exact(table)
            test_name = "Fisher exact"
        else:
            stat, p, dof, expected = stats.chi2_contingency(table)
            test_name = "Chi²"
        sns.heatmap(table, annot=True, fmt="d", cmap="coolwarm")
        plt.title(test_name)
        plt.show()
        return {"stat": stat, "p": p}

    # --- Régression linéaire multiple ---
    elif t_type == "Régression linéaire multiple":
        cible = vars_[0]
        pred_vars = vars_[1:]
        X = df[pred_vars].dropna()
        y = df[cible].loc[X.index]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residus = y - y_pred

        # Graphiques résidus
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        sns.scatterplot(x=y_pred, y=residus, ax=axes[0])
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_title("Résidus vs Valeurs prédites")
        sns.histplot(residus, kde=True, ax=axes[1])
        axes[1].set_title("Distribution des résidus")
        plt.show()
        return {"R2": model.score(X, y), "coef": model.coef_}

    # --- PCA ---
    elif t_type == "PCA":
        X_scaled = StandardScaler().fit_transform(df[vars_].dropna())
        pca = PCA()
        comps = pca.fit_transform(X_scaled)
        plt.scatter(comps[:,0], comps[:,1])
        plt.title("Projection PCA")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()
        return {"explained_variance_ratio": pca.explained_variance_ratio_}

    # --- MCA (catégoriel) ---
    elif t_type == "MCA":
        try:
            import prince
            df_cat = df[vars_].dropna()
            mca = prince.MCA(n_components=2, random_state=42)
            mca = mca.fit(df_cat)
            coords = mca.row_coordinates(df_cat)
            plt.scatter(coords[0], coords[1])
            plt.title("Projection MCA")
            plt.show()
            return {"explained_variance_ratio": mca.explained_inertia_}
        except ImportError:
            print("Module 'prince' manquant")
            return None

    # --- Régression logistique ---
    elif t_type == "Régression logistique":
        cible = vars_[0]
        pred_vars = vars_[1:]
        X = df[pred_vars].dropna()
        y = df[cible].loc[X.index]
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        return {"coef": model.coef_, "intercept": model.intercept_}
