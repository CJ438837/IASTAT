import streamlit as st
import pandas as pd
from modules.IA_STAT_interactif_auto import propose_tests_interactif_auto
from modules.IA_STAT_execute_test import executer_test  # fonction que je t'ai donnée

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

    # --- 2️⃣ Récupération des données depuis la session ---
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

    # --- 4️⃣ Génération des tests ---
    if "tests_generes" not in st.session_state:
        st.session_state.tests_generes, _ = propose_tests_interactif_auto(types_df, distribution_df, df, mots_cles)

    tests = st.session_state.tests_generes
    if not tests:
        st.warning("Aucun test n'a été généré.")
        st.stop()

    # --- 5️⃣ Navigation test par test ---
    if "test_index" not in st.session_state:
        st.session_state.test_index = 0

    test_idx = st.session_state.test_index
    test_dict = tests[test_idx]

    st.subheader(f"Test {test_idx+1} / {len(tests)} : {test_dict['type']}")
    st.write(f"Variables : {', '.join(test_dict['variables'])}")
    st.write(f"Justification : {test_dict.get('justification','')}")

    # --- 6️⃣ Choix appariement si applicable ---
    apparie = False
    if test_dict['type'] in ["t-test", "Mann-Whitney"]:
        apparie = st.radio("Les données sont-elles appariées ?", ("Non", "Oui"), key=f"apparie_{test_idx}") == "Oui"

    # --- 7️⃣ Bouton exécuter le test ---
    if st.button("▶️ Exécuter ce test", key=f"exec_{test_idx}"):
        with st.spinner("Calcul en cours... ⏳"):
            resultats = executer_test(df, test_dict, apparie)
            st.success("✅ Test exécuté !")
            st.write("Résultats :", resultats)

    # --- 8️⃣ Flèches navigation ---
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Précédent", key="prev_test"):
            if st.session_state.test_index > 0:
                st.session_state.test_index -= 1
    with col3:
        if st.button("Suivant ➡️", key="next_test"):
            if st.session_state.test_index < len(tests) - 1:
                st.session_state.test_index += 1
