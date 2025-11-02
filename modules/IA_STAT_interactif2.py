import pandas as pd
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from Bio import Entrez
import numpy as np


# --- Fonction de recherche PubMed ---
def rechercher_pubmed_test(test_name, mots_cles, email="votre.email@example.com", max_results=3):
    """
    Recherche des articles PubMed pertinents pour un test statistique donné
    et des mots-clés liés à l'étude.
    """
    Entrez.email = email
    query = f"{test_name} AND (" + " OR ".join(mots_cles) + ")"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()

    pmids = record['IdList']
    if not pmids:
        print("\nAucun article trouvé pour ces mots-clés et ce test.")
    else:
        print(f"\n=== Articles PubMed suggérés pour le test '{test_name}' ===")
        for i, pmid in enumerate(pmids, 1):
            print(f"{i}. https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

# --- Fonction principale interactive ---
def propose_tests_interactif(types_df, distribution_df, df, mots_cles):
    """
    Propose et exécute les tests statistiques de manière interactive,
    avec graphiques, interprétation simple et articles PubMed pertinents.
    """
    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    # --- 1️⃣ Numérique vs Catégoriel ---
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]

        if n_modalites == 2:
            if verdict == "Normal":
                test_name = "t-test"
                justification = "Numérique normal vs Catégoriel à 2 modalités"
            else:
                test_name = "Mann-Whitney"
                justification = "Numérique non normal vs Catégoriel à 2 modalités"
        elif n_modalites > 2:
            if verdict == "Normal":
                test_name = "ANOVA"
                justification = "Numérique normal vs Catégoriel >2 modalités"
            else:
                test_name = "Kruskal-Wallis"
                justification = "Numérique non normal vs Catégoriel >2 modalités"
        else:
            test_name = "unknown"
            justification = "Impossible de déterminer le test"

        print("\n--- Suggestion ---")
        print(f"Test proposé : {test_name}")
        print(f"Variables : {num} vs {cat}")
        print(f"Justification : {justification}")

        # Question appariement pour les tests à 2 groupes
        apparie = None
        if test_name in ["t-test", "Mann-Whitney"]:
            rep_app = input("Les données sont-elles appariées ? (oui/non) : ").strip().lower()
            apparie = rep_app == "oui"

        # Articles PubMed
        rechercher_pubmed_test(test_name, mots_cles)

        rep = input("Voulez-vous exécuter ce test ? (oui/non) : ").strip().lower()
        if rep != "oui":
            continue

        groupes = df.groupby(cat)[num].apply(list)
        try:
            if test_name == "t-test":
                if apparie:
                    stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1])
                else:
                    stat, p = stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "Mann-Whitney":
                if apparie:
                    stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1])
                else:
                    stat, p = stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
            elif test_name == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)
            else:
                stat, p = None, None

            if stat is not None:
                print(f"Statistique = {stat:.4f}, p-value = {p:.4g}")
                if p < 0.05:
                    print(f"→ Interprétation : La variable '{num}' a un impact significatif sur '{cat}'.")
                else:
                    print(f"→ Interprétation : Aucun impact significatif détecté entre '{num}' et '{cat}'.")

            sns.boxplot(x=cat, y=num, data=df)
            plt.title(f"{test_name} : {num} vs {cat}")
            plt.show()
        except Exception as e:
            print(f"Erreur lors du test : {e}")

    # --- 2️⃣ Deux variables continues ---
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        print("\n--- Suggestion ---")
        print(f"Test proposé : Corrélation ({test_type})")
        print(f"Variables : {var1} vs {var2}")
        rechercher_pubmed_test(f"{test_type} correlation", mots_cles)

        rep = input("Voulez-vous exécuter ce test ? (oui/non) : ").strip().lower()
        if rep != "oui":
            continue

        if test_type == "Pearson":
            corr, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
        else:
            corr, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())

        print(f"Corrélation = {corr:.4f}, p-value = {p:.4g}")
        if p < 0.05:
            print(f"→ Interprétation : {var1} et {var2} sont significativement corrélés.")
        else:
            print(f"→ Interprétation : Pas de corrélation significative détectée.")

        sns.scatterplot(x=var1, y=var2, data=df)
        plt.title(f"Corrélation ({test_type}) : {var1} vs {var2}")
        plt.show()

    # --- 3️⃣ Deux variables catégorielles ---
    for var1, var2 in itertools.combinations(cat_vars, 2):
        print("\n--- Suggestion ---")
        print(f"Test proposé : Khi² / Fisher")
        print(f"Variables : {var1} vs {var2}")
        rechercher_pubmed_test("Chi-square test", mots_cles)

        rep = input("Voulez-vous exécuter ce test ? (oui/non) : ").strip().lower()
        if rep != "oui":
            continue

        contingency_table = pd.crosstab(df[var1], df[var2])
        if contingency_table.size <= 4:
            stat, p = stats.fisher_exact(contingency_table)
            test_name = "Fisher exact"
        else:
            stat, p, dof, expected = stats.chi2_contingency(contingency_table)
            test_name = "Chi²"

        print(f"{test_name} : statistique = {stat:.4f}, p-value = {p:.4g}")
        if p < 0.05:
            print(f"→ Interprétation : '{var1}' dépend significativement de '{var2}'.")
        else:
            print(f"→ Interprétation : Pas de dépendance significative détectée.")

        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm")
        plt.title(f"{test_name} : {var1} vs {var2}")
        plt.show()

    # --- 4️⃣ Multivarié : Régression linéaire et PCA ---
    if len(num_vars) > 1:
        print("\n--- Suggestion ---")
        print("Analyse multivariée : Régression linéaire / PCA")
        rechercher_pubmed_test("multiple regression", mots_cles)
        rechercher_pubmed_test("principal component analysis", mots_cles)

        # Régression linéaire multiple
        rep = input("Voulez-vous exécuter une régression linéaire multiple ? (oui/non) : ").strip().lower()
        if rep == "oui":
           X = df[num_vars].dropna()
           cible = input(f"Quelle variable souhaitez-vous expliquer avec la régression ? {num_vars} : ").strip()

    # --- Gestion insensible à la casse ---
           cible_col = None
           for col in num_vars:
             if col.lower() == cible.lower():
               cible_col = col
             break

           if cible_col is None:
              print(f"⚠️ La variable '{cible}' n'a pas été trouvée dans les données numériques.")
           else:
              y = df[cible_col].loc[X.index]
              X_pred = X.drop(columns=[cible_col])

        # --- Modèle ---
              model = LinearRegression()
              model.fit(X_pred, y)
              y_pred = model.predict(X_pred)
              residus = y - y_pred

        # --- Résultats numériques ---
              print("\n=== Résultats de la régression linéaire multiple ===")
              print(f"Variable dépendante : {cible_col}")
              print(f"R² : {model.score(X_pred, y):.4f}")

              stat, p = stats.shapiro(residus)
              print(f"Test de Shapiro-Wilk sur les résidus : stat={stat:.4f}, p-value={p:.4g}")
              if p > 0.05:
                 print("→ Résidus normalement distribués (hypothèse respectée).")
              else:
                 print("→ ⚠️ Résidus non normaux, hypothèse de normalité violée.")

              print("\nCoefficients :")
              for var, coef in zip(X_pred.columns, model.coef_):
                 print(f"  {var} : {coef:.4f}")
                 print(f"Intercept : {model.intercept_:.4f}")

        # --- 🔢 Formule complète de la régression ---
              formule = f"{cible_col} = {model.intercept_:.4f}"
              for var, coef in zip(X_pred.columns, model.coef_):
                  signe = " + " if coef >= 0 else " - "
                  formule += f"{signe}{abs(coef):.4f} × {var}"
              print(f"\n🧮 Formule de la régression :\n{formule}")

        # --- Graphiques associés ---
              fig, axes = plt.subplots(2, 2, figsize=(12, 10))
              plt.suptitle("Analyse graphique de la régression linéaire multiple", fontsize=14, fontweight="bold")

        # 1️⃣ Résidus vs Valeurs prédites
              sns.scatterplot(x=y_pred, y=residus, ax=axes[0, 0])
              axes[0, 0].axhline(0, color='red', linestyle='--')
              axes[0, 0].set_xlabel("Valeurs prédites")
              axes[0, 0].set_ylabel("Résidus")
              axes[0, 0].set_title("Résidus vs Valeurs prédites")

        # 2️⃣ Histogramme des résidus
              sns.histplot(residus, kde=True, ax=axes[0, 1], color='skyblue')
              axes[0, 1].set_title("Distribution des résidus")
              axes[0, 1].set_xlabel("Résidus")

        # 3️⃣ QQ-plot
              stats.probplot(residus, dist="norm", plot=axes[1, 0])
              axes[1, 0].set_title("QQ-Plot des résidus")

        # 4️⃣ Observé vs Prédit
              sns.scatterplot(x=y, y=y_pred, ax=axes[1, 1])
              axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
              axes[1, 1].set_xlabel("Valeurs observées")
              axes[1, 1].set_ylabel("Valeurs prédites")
              axes[1, 1].set_title("Observé vs Prédit")

              plt.tight_layout(rect=[0, 0, 1, 0.96])
              plt.show()

        # PCA
        # --- Analyse PCA améliorée ---
    rep = input("Voulez-vous exécuter une PCA ? (oui/non) : ").strip().lower()
    if rep == "oui":
         X_scaled = StandardScaler().fit_transform(df[num_vars].dropna())
         pca = PCA()
         components = pca.fit_transform(X_scaled)
         explained_variance = pca.explained_variance_ratio_
         cum_var = explained_variance.cumsum()
         n_comp = (cum_var < 0.8).sum() + 1
         print(f"→ {n_comp} composante(s) principales expliquent ~80% de la variance")

    # --- Matrice de contributions ---
         loading_matrix = pd.DataFrame(pca.components_.T,
                                  index=num_vars,
                                  columns=[f"PC{i+1}" for i in range(len(num_vars))])

         print("\n→ Variables qui contribuent le plus aux premières composantes :")
         print(loading_matrix.iloc[:, :n_comp].abs().sort_values(by="PC1", ascending=False).head(10))

    # --- 🔢 Interprétation automatique ---
         print("\n=== Interprétation automatique ===")
         for i in range(n_comp):
             top_vars = loading_matrix.iloc[:, i].abs().sort_values(ascending=False).head(3).index.tolist()
             print(f"PC{i+1} : fortement influencée par {', '.join(top_vars)}")

    # --- Graphiques associés ---
         plt.figure(figsize=(8,6))
         plt.scatter(components[:,0], components[:,1])
         plt.xlabel("PC1")
         plt.ylabel("PC2")
         plt.title("Projection des individus sur les 2 premières composantes principales")
         plt.grid(True)
         plt.show()

    # --- Biplot : variables et individus ---
         plt.figure(figsize=(8,6))
         plt.title("Biplot PCA : variables et individus")
         plt.scatter(components[:,0], components[:,1], alpha=0.5)
         for i, var in enumerate(num_vars):
             plt.arrow(0, 0,
                  pca.components_[0, i]*max(components[:,0]),
                  pca.components_[1, i]*max(components[:,1]),
                  color='red', alpha=0.7, head_width=0.05)
             plt.text(pca.components_[0, i]*max(components[:,0])*1.1,
                 pca.components_[1, i]*max(components[:,1])*1.1,
                 var, color='darkred', ha='center', va='center')
         plt.xlabel("PC1")
         plt.ylabel("PC2")
         plt.grid(True)
         plt.show()

    if len(cat_vars) > 1:
       print("\n--- Suggestion ---")
       print("Analyse multivariée : Analyse des correspondances multiples (MCA)")
       rechercher_pubmed_test("multiple correspondence analysis", "biological data")

       rep = input("Voulez-vous exécuter une MCA ? (oui/non) : ").strip().lower()
       if rep == "oui":
            try:
               import prince
               df_cat = df[cat_vars].fillna("Missing")
               df_cat = df[cat_vars].dropna()
               mca = prince.MCA(n_components=2, random_state=42)
               mca = mca.fit(df_cat)

               if hasattr(mca, "explained_inertia_"):
                  var_expl = mca.explained_inertia_
               elif hasattr(mca, "explained_variance_ratio_"):
                  var_expl = mca.explained_variance_ratio_
               else:
                  var_expl = [np.nan, np.nan]
               print(f"\n=== Résultats MCA ===")
               print(f"Variance expliquée : {var_expl[0]*100:.2f}% et {var_expl[1]*100:.2f}%")
               print(f"→ Ensemble, ~{sum(var_expl[:2])*100:.2f}% de la variabilité totale.")
               coords = mca.column_coordinates(df_cat)
               print("\n→ Catégories les plus contributrices :")
               print(coords.head(10))

            # Graphique individus
               plt.figure(figsize=(7, 6))
               ind_coords = mca.row_coordinates(df_cat)
               plt.scatter(ind_coords[0], ind_coords[1], alpha=0.6)
               plt.xlabel("Dim 1")
               plt.ylabel("Dim 2")
               plt.title("Projection des individus (MCA)")
               plt.grid(True)
               plt.show()

            # Graphique catégories
               plt.figure(figsize=(7, 6))
               plt.scatter(coords[0], coords[1], color='red', alpha=0.7)
               for i, label in enumerate(coords.index):
                   plt.text(coords.iloc[i, 0], coords.iloc[i, 1], label, fontsize=9, color='darkred')
               plt.xlabel("Dim 1")
               plt.ylabel("Dim 2")
               plt.title("Projection des catégories (MCA)")
               plt.grid(True)
               plt.show()

            # Cercle des corrélations (MCA)
               plt.figure(figsize=(6, 6))
               circle = plt.Circle((0, 0), 1, color='gray', fill=False)
               plt.gca().add_artist(circle)
               for i, label in enumerate(coords.index):
                   plt.arrow(0, 0, coords.iloc[i, 0], coords.iloc[i, 1],
                          color='blue', alpha=0.5, head_width=0.03)
                   plt.text(coords.iloc[i, 0]*1.1, coords.iloc[i, 1]*1.1, label,
                         color='blue', ha='center', va='center', fontsize=8)
               plt.xlim(-1.1, 1.1)
               plt.ylim(-1.1, 1.1)
               plt.axhline(0, color='gray', lw=0.5)
               plt.axvline(0, color='gray', lw=0.5)
               plt.title("Cercle des corrélations (MCA)")
               plt.show()

            except ImportError:
               print("\n⚠️ Le module 'prince' n'est pas installé. Exécutez : pip install prince")
            except Exception as e:
               print(f"Erreur lors de la MCA : {e}")
    # --- 5️⃣ Variable binaire dépendante : Régression logistique ---
    for cat in cat_vars:
        if df[cat].dropna().nunique() == 2:
            print("\n--- Suggestion ---")
            print(f"Régression logistique pour la variable binaire dépendante : {cat}")
            rechercher_pubmed_test("logistic regression", mots_cles)
            rep = input("Voulez-vous exécuter ce test ? (oui/non) : ").strip().lower()
            if rep != "oui":
                continue
            X = df[num_vars].dropna()
            y = df[cat].loc[X.index]
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            print(f"Coefficients : {dict(zip(num_vars, model.coef_[0]))}")
            print(f"Intercept : {model.intercept_[0]}")

    print("\n✅ Tous les tests interactifs terminés.")
