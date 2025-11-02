import streamlit as st

def app():
    st.header("Statistiques descriptives")
    st.write("Ici nous allons réaliser les statistiques descriptive de ton étude")

import streamlit as st
from modules.IA_STAT_descriptive_251125 import descriptive_analysis
from modules.IA_STAT_Illustrations_251125 import plot_descriptive

def app():
    st.header("📊 Analyse descriptive")

    # Vérification de session_state
    if 'df_selected' not in st.session_state or 'types_results' not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier et détecter les types de variables dans les pages Fichier et Variables.")
        st.stop()

    df = st.session_state['cleaned_data']
    types_df = st.session_state['types_results']

    # Analyse descriptive
    summary = descriptive_analysis(df, types_df)
    st.subheader("Résumé statistique")
    st.write(summary)

    # Graphiques descriptifs
    st.subheader("Graphiques descriptifs")
    plot_descriptive(df, types_df, output_folder="plots")
    st.success("Graphiques générés !")

