import streamlit as st
import pandas as pd

# 🔹 Import de toutes les fonctions de tests
from modules.IA_STAT_interactif_auto import (
    propose_tests_interactif_auto_anova,
    propose_tests_interactif_auto_kruskal,
    propose_tests_interactif_auto_ttest,
    propose_tests_interactif_auto_mannwhitney,
    propose_tests_interactif_auto_chi2,
    propose_tests_interactif_auto_correlation
)


def app():
    st.title("📊 Tests statistiques interactifs")

    # --- 1️⃣ Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser la distribution des données dans la page Distribution.")
        st.stop()

    # --- 2️⃣ Récupération des données ---
    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    # --- 3️⃣ Vérification et renommage des colonnes ---
    rename_dict = {}
    for col in types_df.columns:
        lower = col.lower()
        if lower in ["var", "variable_name", "nom", "column"]:
            rename_dict[col] = "variable"
        elif lower in ["var_type", "type_var", "variable_type", "kind"]:
            rename_dict[col] = "type"
    types_df.rename(columns=rename_dict, inplace=True)

    expected_cols = {"variable", "type"}
    if not expected_cols.issubset(types_df.columns):
        st.error(f"⚠️ Le tableau des types de variables doit contenir les colonnes : {expected_cols}. "
                 f"Colonnes actuelles : {types_df.columns.tolist()}")
        st.stop()

    # --- 4️⃣ Interface utilisateur ---
    st.success("✅ Toutes les données nécessaires ont été chargées.")
    st.markdown("### 💡 Choisis un test à exécuter")

    test_options = {
        "ANOVA": propose_tests_interactif_auto_anova,
        "Kruskal-Wallis": propose_tests_interactif_auto_kruskal,
        "t-test (Student)": propose_tests_interactif_auto_ttest,
        "Mann-Whitney": propose_tests_interactif_auto_mannwhitney,
        "Chi²": propose_tests_interactif_auto_chi2,
        "Corrélations": propose_tests_interactif_auto_correlation
    }

    choix_test = st.selectbox("📈 Sélectionne le test à exécuter :", list(test_options.keys()))
    apparie = st.radio("Données appariées ?", ("Non", "Oui"), key=f"apparie_{choix_test}") == "Oui"

    lancer = st.button("🚀 Exécuter ce test")

    if lancer:
        with st.spinner("Analyse en cours... ⏳"):
            summary_df, all_results = test_options[choix_test](types_df, distribution_df, df, mots_cles, apparie)

        st.success(f"✅ Test {choix_test} exécuté avec succès !")
        st.markdown("### 📊 Résultats du test")
        st.dataframe(summary_df)

        # 📥 Option de téléchargement
        csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les résultats (CSV)",
            data=csv,
            file_name=f"resultats_{choix_test}.csv",
            mime="text/csv"
        )
