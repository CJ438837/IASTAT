import streamlit as st
import pandas as pd
from modules.IA_STAT_interactif2 import propose_tests_interactif_streamlit

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

    # --- 3️⃣ Normalisation des colonnes ---
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

    st.success("✅ Toutes les données nécessaires ont été chargées.")
    st.markdown("### 💡 Propositions de tests statistiques adaptés")

    # --- 4️⃣ Initialisation de l’état ---
    if "tests_generes" not in st.session_state:
        st.session_state.tests_generes = []
    if "test_index" not in st.session_state:
        st.session_state.test_index = 0
    if "tests_resultats" not in st.session_state:
        st.session_state.tests_resultats = {}

    # --- 5️⃣ Lancement de la génération des tests ---
    if st.button("🧠 Générer les propositions de tests"):
        with st.spinner("Analyse en cours... ⏳"):
            st.session_state.tests_generes = propose_tests_interactif_streamlit(
                types_df, distribution_df, df, mots_cles, generation_mode=True
            )
            st.session_state.test_index = 0
            st.session_state.tests_resultats = {}

    # --- 6️⃣ Navigation entre tests ---
    if st.session_state.tests_generes:
        current_index = st.session_state.test_index
        total_tests = len(st.session_state.tests_generes)
        test = st.session_state.tests_generes[current_index]

        st.markdown(f"### 🔍 Test {current_index + 1}/{total_tests}")
        st.write(f"**Proposition :** {test['nom_test']}")
        st.caption(test.get("description", "Aucune description disponible."))

        # --- Options spécifiques au test ---
        with st.form(key=f"form_test_{current_index}"):
            if test.get("options"):
                for opt in test["options"]:
                    if opt["type"] == "radio":
                        st.radio(opt["label"], opt["choices"], key=f"radio_{current_index}_{opt['label']}")
                    elif opt["type"] == "selectbox":
                        st.selectbox(opt["label"], opt["choices"], key=f"select_{current_index}_{opt['label']}")
                    elif opt["type"] == "checkbox":
                        st.checkbox(opt["label"], key=f"check_{current_index}_{opt['label']}")

            run_test = st.form_submit_button("🚀 Exécuter ce test")

        # --- 7️⃣ Exécution du test ---
        if run_test:
            st.session_state.tests_resultats[current_index] = f"Résultats du test {test['nom_test']} ✅"

        # --- 8️⃣ Affichage du résultat si dispo ---
        if current_index in st.session_state.tests_resultats:
            st.success(st.session_state.tests_resultats[current_index])

        # --- 9️⃣ Navigation ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Précédent", disabled=(current_index == 0)):
                st.session_state.test_index = max(0, current_index - 1)
                st.rerun()
        with col2:
            if st.button("Suivant ➡️", disabled=(current_index == total_tests - 1)):
                st.session_state.test_index = min(total_tests - 1, current_index + 1)
                st.rerun()
