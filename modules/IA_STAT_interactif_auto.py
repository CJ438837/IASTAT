import itertools
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

def propose_tests_interactif_auto(types_df, distribution_df, df, mots_cles):
    """
    Génère tous les tests statistiques possibles sur le dataset,
    avec les informations nécessaires pour l'interface Streamlit.
    
    Retourne :
        - summary_df : résumé des tests générés
        - all_results : liste de dictionnaires décrivant chaque test
    """
    all_results = []

    # --- Normalisation des colonnes ---
    rename_dict = {}
    for col in types_df.columns:
        lower = col.lower()
        if lower in ["var", "variable_name", "nom", "column"]:
            rename_dict[col] = "variable"
        elif lower in ["var_type", "type_var", "variable_type", "kind"]:
            rename_dict[col] = "type"
    types_df = types_df.rename(columns=rename_dict)

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle','binaire'])]['variable'].tolist()

    # =========================
    # 1️⃣ Numérique vs Catégoriel
    # =========================
    for num, cat in itertools.product(num_vars, cat_vars):
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable']==num, 'verdict'].values[0]

        if n_modalites == 2:
            test_type = "t-test" if verdict=="Normal" else "Mann-Whitney"
            needs_apparie = True
        elif n_modalites > 2:
            test_type = "ANOVA" if verdict=="Normal" else "Kruskal-Wallis"
            needs_apparie = False
        else:
            test_type = "unknown"
            needs_apparie = False

        all_results.append({
            "type": test_type,
            "variables": [num, cat],
            "apparie_possible": needs_apparie,
            "fonction": None  # sera exécuté plus tard via Streamlit
        })

    # =========================
    # 2️⃣ Numérique vs Numérique (Corrélation)
    # =========================
    for var1, var2 in itertools.combinations(num_vars, 2):
        verdict1 = distribution_df.loc[distribution_df['variable']==var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable']==var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1=="Normal" and verdict2=="Normal" else "Spearman"

        all_results.append({
            "type": f"Corrélation ({test_type})",
            "variables": [var1, var2],
            "apparie_possible": False,
            "fonction": None
        })

    # =========================
    # 3️⃣ Catégoriel vs Catégoriel
    # =========================
    for var1, var2 in itertools.combinations(cat_vars, 2):
        all_results.append({
            "type": "Chi2/Fisher",
            "variables": [var1, var2],
            "apparie_possible": False,
            "fonction": None
        })

    # =========================
    # 4️⃣ Régression linéaire multiple
    # =========================
    if len(num_vars) > 1:
        for cible in num_vars:
            pred_vars = [v for v in num_vars if v != cible]
            all_results.append({
                "type": "Régression linéaire multiple",
                "variables": [cible] + pred_vars,
                "apparie_possible": False,
                "fonction": None
            })

    # =========================
    # 5️⃣ PCA
    # =========================
    if len(num_vars) > 1:
        all_results.append({
            "type": "PCA",
            "variables": num_vars,
            "apparie_possible": False,
            "fonction": None
        })

    # =========================
    # 6️⃣ MCA (catégoriel)
    # =========================
    if len(cat_vars) > 1:
        all_results.append({
            "type": "MCA",
            "variables": cat_vars,
            "apparie_possible": False,
            "fonction": None
        })

    # =========================
    # 7️⃣ Régression logistique (binaire)
    # =========================
    for cat in cat_vars:
        if df[cat].dropna().nunique() == 2:
            all_results.append({
                "type": "Régression logistique",
                "variables": [cat] + num_vars,
                "apparie_possible": False,
                "fonction": None
            })

    # =========================
    # Résumé DataFrame
    # =========================
    summary_df = pd.DataFrame([{
        "Test": t["type"],
        "Variables": ", ".join(t["variables"]),
        "Apparié possible": t["apparie_possible"]
    } for t in all_results])

    return summary_df, all_results
