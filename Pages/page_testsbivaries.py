import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np

# -----------------------------
# Fonction principale tests bivariés
# -----------------------------
def propose_tests_bivaries(df, types_df, distribution_df, mots_cles=None, interactive=False):
    """
    Exécute tous les tests bivariés (num vs cat, num vs num, cat vs cat) avec graphiques.
    Retourne un dictionnaire avec un tableau de résultats par test.
    """
    # --- Préparer les variables ---
    rename_dict = {}
    for col in types_df.columns:
        lower = col.lower()
        if lower in ["var", "variable_name", "nom", "column"]:
            rename_dict[col] = "variable"
        elif lower in ["var_type", "type_var", "variable_type", "kind"]:
            rename_dict[col] = "type"
    types_df = types_df.rename(columns=rename_dict)

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    all_results = {}

    # -------------------------------
    # 1️⃣ Numérique vs Catégoriel
    # -------------------------------
    for num, cat in [(n, c) for n in num_vars for c in cat_vars]:
        n_modalites = df[cat].dropna().nunique()
        verdict = distribution_df.loc[distribution_df['variable'] == num, 'verdict'].values[0]

        # Choix du test
        if n_modalites == 2:
            test_name = "t-test" if verdict == "Normal" else "Mann-Whitney"
        elif n_modalites > 2:
            test_name = "ANOVA" if verdict == "Normal" else "Kruskal-Wallis"
        else:
            test_name = "unknown"

        # Option apparié
        apparie = False
        if test_name in ["t-test", "Mann-Whitney"] and interactive:
            apparie = st.radio(f"Les données {num} vs {cat} sont-elles appariées ?", ("Non", "Oui")) == "Oui"

        groupes = df.groupby(cat)[num].apply(list)
        stat, p = None, None
        try:
            if test_name == "t-test":
                stat, p = stats.ttest_rel(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.ttest_ind(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "Mann-Whitney":
                stat, p = stats.wilcoxon(groupes.iloc[0], groupes.iloc[1]) if apparie else stats.mannwhitneyu(groupes.iloc[0], groupes.iloc[1])
            elif test_name == "ANOVA":
                stat, p = stats.f_oneway(*groupes)
            elif test_name == "Kruskal-Wallis":
                stat, p = stats.kruskal(*groupes)
        except Exception as e:
            st.warning(f"Erreur {num} vs {cat} : {e}")

        # Graphique
        fig, ax = plt.subplots()
        sns.boxplot(x=cat, y=num, data=df, ax=ax)
        ax.set_title(f"{test_name} : {num} vs {cat}")
        st.pyplot(fig)
        plt.close(fig)

        # Résultat
        all_results[f"{num}_vs_{cat}"] = pd.DataFrame([{
            "Variable_num": num,
            "Variable_cat": cat,
            "Test": test_name,
            "Statistique": stat,
            "p-value": p,
            "Apparié": apparie
        }])

    # -------------------------------
    # 2️⃣ Numérique vs Numérique
    # -------------------------------
    for var1, var2 in [(v1, v2) for v1, v2 in itertools.combinations(num_vars, 2)]:
        verdict1 = distribution_df.loc[distribution_df['variable'] == var1, 'verdict'].values[0]
        verdict2 = distribution_df.loc[distribution_df['variable'] == var2, 'verdict'].values[0]
        test_type = "Pearson" if verdict1 == "Normal" and verdict2 == "Normal" else "Spearman"

        try:
            if test_type == "Pearson":
                stat, p = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
            else:
                stat, p = stats.spearmanr(df[var1].dropna(), df[var2].dropna())
        except Exception as e:
            st.warning(f"Erreur corrélation {var1} vs {var2} : {e}")
            stat, p = None, None

        # Graphique
        fig, ax = plt.subplots()
        sns.scatterplot(x=var1, y=var2, data=df, ax=ax)
        ax.set_title(f"Corrélation ({test_type}) : {var1} vs {var2}")
        st.pyplot(fig)
        plt.close(fig)

        # Résultat
        all_results[f"{var1}_vs_{var2}"] = pd.DataFrame([{
            "Variable_num1": var1,
            "Variable_num2": var2,
            "Test": f"Corrélation ({test_type})",
            "Statistique": stat,
            "p-value": p
        }])

    # -------------------------------
    # 3️⃣ Catégoriel vs Catégoriel
    # -------------------------------
    for var1, var2 in [(v1, v2) for v1, v2 in itertools.combinations(cat_vars, 2)]:
        contingency_table = pd.crosstab(df[var1], df[var2])
        try:
            if contingency_table.size <= 4:
                stat, p = stats.fisher_exact(contingency_table)
                test_name = "Fisher exact"
            else:
                stat, p, dof, expected = stats.chi2_contingency(contingency_table)
                test_name = "Chi²"
        except Exception as e:
            st.warning(f"Erreur test catégoriel {var1} vs {var2} : {e}")
            stat, p, test_name = None, None, "Erreur"

        # Graphique
        fig, ax = plt.subplots()
        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
        ax.set_title(f"{test_name} : {var1} vs {var2}")
        st.pyplot(fig)
        plt.close(fig)

        # Résultat
        all_results[f"{var1}_vs_{var2}"] = pd.DataFrame([{
            "Variable_cat1": var1,
            "Variable_cat2": var2,
            "Test": test_name,
            "Statistique": stat,
            "p-value": p
        }])

    return all_results

# -----------------------------
# Page Streamlit
# -----------------------------
def app():
    st.title("📊 Tests bivariés automatiques")

    # Vérifications
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser la distribution dans la page Distribution.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    # Option interactive pour appariement
    interactive = st.checkbox("Choisir l'appariement pour chaque test à 2 groupes", value=False)

    if st.button("🧠 Exécuter tous les tests bivariés"):
        with st.spinner("Exécution des tests... ⏳"):
            all_results = propose_tests_bivaries(df, types_df, distribution_df, mots_cles, interactive=interactive)
            st.success("✅ Tests terminés !")

            # Affichage des résultats test par test
            for test_name, df_result in all_results.items():
                st.markdown(f"### {test_name}")
                st.dataframe(df_result)
