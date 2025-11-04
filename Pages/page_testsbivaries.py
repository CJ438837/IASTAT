import streamlit as st
from modules.IA_STAT_testbivaries import propose_tests_bivariés

def app():
    st.title("📊 Tests bivariés automatiques")

    if "df_selected" not in st.session_state or \
       "types_df" not in st.session_state or \
       "distribution_df" not in st.session_state:
        st.warning("Importez les fichiers dans les pages précédentes.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    if "test_index" not in st.session_state:
        st.session_state["test_index"] = 0
    if "test_results" not in st.session_state:
        st.session_state["test_results"] = []

    if st.button("🧠 Générer tous les tests bivariés"):
        st.session_state["test_results"] = propose_tests_bivariés(df, types_df, distribution_df)
        st.session_state["test_index"] = 0
        st.success(f"{len(st.session_state['test_results'])} tests générés !")

    if st.session_state["test_results"]:
        idx = st.session_state["test_index"]
        test_data = st.session_state["test_results"][idx]

        # --- Apparié si nécessaire ---
        apparie = test_data.get("apparie_needed", False)
        if apparie:
            apparie_choice = st.radio(f"Le test {test_data['test_name']} ({test_data['num']} vs {test_data['cat']}) est-il apparié ?", ("Non","Oui"))
            test_data["result_df"]["Apparié"] = apparie_choice == "Oui"

        st.markdown("### 📄 Résultat")
        st.dataframe(test_data["result_df"])

        st.markdown("### 📊 Graphique")
        st.pyplot(test_data["fig"])

        # --- Navigation ---
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("⬅️ Test précédent") and idx>0:
                st.session_state["test_index"] -= 1
        with col3:
            if st.button("Test suivant ➡️") and idx<len(st.session_state["test_results"])-1:
                st.session_state["test_index"] += 1

        st.markdown(f"**Test {idx+1} / {len(st.session_state['test_results'])}**")
