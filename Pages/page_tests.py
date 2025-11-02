import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modules.IA_STAT_interactif2 import propose_tests_interactif_streamlit

def app():
    st.title("📊 Tests statistiques interactifs")

    # --- Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()

    if "distribution_df" not in st.session_state:
        st.warning("Pas de distribution")
        st.stop()


    df = st.session_state["df_selected"]
    types_df = st.session_state["types_df"]
    distribution_df = st.session_state["distribution_df"]
    mots_cles = st.session_state.get("keywords", [])

    st.markdown("### 💡 Propositions de tests")

    # --- Lancer les tests avec un bouton ---
    if st.button("Lancer les tests interactifs"):
        with st.spinner("Exécution des tests..."):
            propose_tests_interactif_streamlit(types_df, distribution_df, df, mots_cles)
        st.success("✅ Tous les tests interactifs ont été proposés et exécutés.")

