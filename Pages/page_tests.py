import streamlit as st
from modules.IA_STAT_interactif_auto import propose_tests_interactif_auto

def app():
    st.title("📊 Tests statistiques automatiques")

    df = st.session_state["df_selected"]
    types_df = st.session_state["types_df"]
    distribution_df = st.session_state["distribution_df"]
    mots_cles = st.session_state.get("keywords", [])

    apparie = st.radio("Les données sont-elles appariées ?", ["Non", "Oui"]) == "Oui"

    if st.button("🚀 Lancer l’analyse complète"):
        with st.spinner("Exécution des tests..."):
            summary_df, all_results = propose_tests_interactif_auto(types_df, distribution_df, df, mots_cles, apparie)
        st.success("✅ Analyse terminée")
        st.dataframe(summary_df)

        # Affiche les graphiques
        for r in all_results:
            if "figure" in r:
                st.pyplot(r["figure"])
