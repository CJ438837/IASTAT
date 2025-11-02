import streamlit as st
import pandas as pd
from modules.IA_STAT_descriptive_251125 import descriptive_analysis

def app():
    st.title("📊 Analyse Descriptive")

    # --- 1️⃣ Vérification des données ---
    if 'df_selected' not in st.session_state or st.session_state['df_selected'] is None:
        st.warning("⚠️ Veuillez d'abord importer et sélectionner un fichier dans la page Fichier.")
        return

    if 'types_df' not in st.session_state or st.session_state['types_df'] is None:
        st.warning("⚠️ Veuillez d'abord définir les types de variables dans la page Variables.")
        return

    df = st.session_state['df_selected']
    types_df = st.session_state['types_df']

    # --- 2️⃣ Sélection des colonnes à analyser ---
    st.subheader("Colonnes à inclure dans l'analyse")
    cols_selected = st.multiselect("Choisir les variables :", df.columns.tolist(), default=df.columns.tolist())
    if not cols_selected:
        st.warning("⚠️ Veuillez sélectionner au moins une colonne.")
        return

    df = df[cols_selected]
    types_df = types_df[types_df['variable'].isin(cols_selected)]

    # --- 3️⃣ Calcul du summary ---
    summary = descriptive_analysis(df, types_df)

    # --- 4️⃣ Affichage des résultats ---
    st.subheader("Résumé descriptif par variable")
    for var, stats in summary.items():
        st.markdown(f"### {var} ({types_df.loc[types_df['variable']==var,'type'].values[0]})")
        st.json(stats)
