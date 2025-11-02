# Pages/2_Variables.py
import streamlit as st
import pandas as pd
from modules.IA_STAT_typevariable_251125 import detect_variable_types

def app():
    st.header("📝 Détection des types de variables")
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page 'Fichier'.")
        st.stop()
    df_selected = st.session_state["df_selected"]
    if st.button("Détecter automatiquement les types de variables"):
        types_results, cleaned_data = detect_variable_types(df_selected)
        df_types = types_results['data'] if 'data' in types_results else list(types_results.values())[0]
        st.session_state["types_df"] = df_types
        st.session_state["data_df"] = cleaned_data['data'] if 'data' in cleaned_data else list(cleaned_data.values())[0]
    if "types_df" in st.session_state:
        st.subheader("Types détectés (modifiable)")
        df_types = st.session_state["types_df"].copy()
        for i, row in df_types.iterrows():
            var = row["variable"]
            current_type = row["type"]
            new_type = st.selectbox(f"Type pour '{var}'", ["numérique", "catégorielle", "binaire"],
                                    index=["numérique", "catégorielle", "binaire"].index(current_type))
            df_types.at[i, "type"] = new_type
        st.session_state["types_df"] = df_types
        st.dataframe(df_types)

