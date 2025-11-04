import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testbivaries import propose_tests_bivariés

def app():
    st.title("📊 Tests bivariés automatiques")

    # --- Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser la distribution des données dans la page Distribution.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    # --- Bouton pour lancer les tests ---
    if "test_index" not in st.session_state:
        st.session_state["test_index"] = 0
    if "test_results" not in st.session_state:
        st.session_state["test_results"] = []

    if st.button("🧠 Générer les tests bivariés"):
        st.session_state["test_results"] = propose_tests_bivaries(df, types_df, distribution_df, mots_cles)
        st.session_state["test_index"] = 0
        st.success(f"✅ {len(st.session_state['test_results'])} tests générés !")

    # --- Navigation des tests ---
    if st.session_state["test_results"]:
        test_index = st.session_state["test_index"]
        test_data = st.session_state["test_results"][test_index]

        # --- Sélection apparié/non apparié pour tests numériques à 2 groupes ---
        if test_data.get("test_type") in ["t-test", "Mann-Whitney"]:
            apparie = st.radio(
                f"Le test {test_data['test_name']} pour {test_data['var_num']} vs {test_data['var_cat']} est-il apparié ?",
                ("Non", "Oui"),
                index=0
            ) == "Oui"
            test_data["apparie"] = apparie
            # Recalcul du test avec la sélection
            test_data["result_df"], test_data["fig"] = test_data["recalc_func"](apparie)

        # Affichage tableau
        st.markdown("### 📄 Résultat du test")
        st.dataframe(test_data["result_df"])

        # Affichage graphique
        st.markdown("### 📊 Graphique associé")
        st.pyplot(test_data["fig"])

        # Navigation test suivant / précédent
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("⬅️ Test précédent") and test_index > 0:
                st.session_state["test_index"] -= 1
        with col3:
            if st.button("Test suivant ➡️") and test_index < len(st.session_state["test_results"]) - 1:
                st.session_state["test_index"] += 1

        # Information de navigation
        st.markdown(f"**Test {test_index+1} / {len(st.session_state['test_results'])}**")

