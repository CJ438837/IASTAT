
import streamlit as st
import pandas as pd
from modules.IA_STAT_typevariable_251125 import detect_variable_types

def app():
    # --- 🎨 Thème Corvus ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Impossible de charger le thème Corvus : {e}")

    # --- 🧠 En-tête ---
    st.markdown("<h1 class='corvus-title'> Détection des types de variables</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Le but est de d'identifier les types de variables qui composent votre fichier.**
    **Cette étape est primordiale pour la bonne réalisation de la suite de l'analyse !**
    **N'hésitez pas à modifier les types de variables proposées si nécessaire.**
    """)

    # --- 📦 Vérification des données importées ---
    if "df_selected" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer un fichier dans la page **Fichier**.")
        st.stop()
    
    df_selected = st.session_state["df_selected"]

    # --- 🚀 Détection automatique ---
    with st.container():
        st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Détection automatique")
        
        
        if st.button("📈 Détecter les types de variables", use_container_width=True):
            with st.spinner("Analyse des colonnes en cours..."):
                types_results, cleaned_data = detect_variable_types(df_selected)
                df_types = types_results['data'] if 'data' in types_results else list(types_results.values())[0]
                st.session_state["types_df"] = df_types
                st.session_state["data_df"] = cleaned_data['data'] if 'data' in cleaned_data else list(cleaned_data.values())[0]
        
        st.markdown("</div>", unsafe_allow_html=True)

    # --- 📊 Édition manuelle des types ---
    if "types_df" in st.session_state:
        df_types = st.session_state["types_df"].copy()

        st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Types détectés (modifiable)")
        st.markdown("<p class='corvus-text'>Vous pouvez ajuster le type de chaque variable manuellement si nécessaire.</p>", unsafe_allow_html=True)

        for i, row in df_types.iterrows():
            var = row["variable"]
            current_type = row["type"]
            col1, col2 = st.columns([2, 2])
            with col1:
                st.markdown(f"**{var}**")
            with col2:
                new_type = st.selectbox(
                    f"Type pour {var}",
                    ["numérique", "catégorielle", "binaire"],
                    index=["numérique", "catégorielle", "binaire"].index(current_type),
                    label_visibility="collapsed"
                )
            df_types.at[i, "type"] = new_type
        
        st.session_state["types_df"] = df_types
        st.dataframe(df_types, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- ➡️ Navigation ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("➡️ Passer à la page Analyse Descriptive", use_container_width=True):
            st.session_state.target_page = "Descriptive"

