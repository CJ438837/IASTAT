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

    # --- Initialisation session_state pour navigation ---
    if "test_index" not in st.session_state:
        st.session_state["test_index"] = 0
    if "test_results" not in st.session_state:
        st.session_state["test_results"] = []

    # --- Génération des tests ---
    if st.button("🧠 Générer les tests bivariés"):
        st.session_state["test_results"] = propose_tests_bivariés(df, types_df, distribution_df, mots_cles, interactive=False)
        st.session_state["test_index"] = 0
        st.success(f"✅ {len(st.session_state['test_results'])} tests générés !")

    # --- Navigation test par test ---
    if st.session_state["test_results"]:
        test_index = st.session_state["test_index"]
        test_data = st.session_state["test_results"][test_index]

        st.markdown(f"### Test {test_index+1} / {len(st.session_state['test_results'])}")

        # --- Option apparié/non apparié pour tests 2 groupes ---
        if test_data["type"] == "num_vs_cat" and test_data["n_modalites"] == 2:
            apparie = st.radio(
                f"Test {test_data['num_var']} vs {test_data['cat_var']}: Les groupes sont-ils appariés ?",
                ("Non", "Oui"),
                index=0
            ) == "Oui"
            test_data["apparie"] = apparie

        # --- Exécution du test individuel ---
        if st.button("▶️ Exécuter ce test"):
            try:
                test_result = test_data["execute"](df, test_data)
                st.session_state["test_results"][test_index]["result_df"] = test_result["result_df"]
                st.session_state["test_results"][test_index]["fig"] = test_result["fig"]
            except Exception as e:
                st.error(f"❌ Erreur lors de l'exécution du test : {e}")

        # --- Affichage du tableau et graphique si déjà exécuté ---
        if "result_df" in test_data and "fig" in test_data:
            st.markdown("#### 📄 Résultat du test")
            st.dataframe(test_data["result_df"])

            st.markdown("#### 📊 Graphique associé")
            st.pyplot(test_data["fig"])

        # --- Navigation test précédent / suivant ---
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("⬅️ Test précédent") and test_index > 0:
                st.session_state["test_index"] -= 1
        with col3:
            if st.button("Test suivant ➡️") and test_index < len(st.session_state["test_results"]) - 1:
                st.session_state["test_index"] += 1
