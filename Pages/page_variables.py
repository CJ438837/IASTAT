# Pages/2_Variables.py
import streamlit as st
import pandas as pd
from modules.IA_STAT_typevariable_251125 import detect_variable_types

st.header("📝 Détection des types de variables")

# --- Récupération du DataFrame depuis la page Fichier ---
if "df_selected" not in st.session_state:
    st.warning("Veuillez d'abord importer un fichier dans la page 'Fichier'.")
    st.stop()

df_selected = st.session_state["df_selected"]

# --- 1️⃣ Choix de la feuille (si fichier Excel multi-feuilles) ---
sheet_name = None
if hasattr(df_selected, "sheet_names"):  # si c'est un ExcelFile
    if len(df_selected.sheet_names) > 1:
        sheet_name = st.selectbox("Choisir la feuille à analyser :", df_selected.sheet_names)
    else:
        sheet_name = df_selected.sheet_names[0]

# --- 2️⃣ Bouton pour lancer la détection automatique ---
if st.button("Détecter automatiquement les types de variables"):
    # Ici, si df_selected est déjà un DataFrame, on peut directement l'envoyer
    if isinstance(df_selected, pd.DataFrame):
        types_results, cleaned_data = detect_variable_types(df_selected)
        df_types = types_results['data'] if 'data' in types_results else list(types_results.values())[0]
        st.session_state["types_df"] = df_types
        st.session_state["data_df"] = cleaned_data['data'] if 'data' in cleaned_data else list(cleaned_data.values())[0]
    else:
        types_results, cleaned_data = detect_variable_types(df_selected, sheet_name)
        df_types = types_results[sheet_name]
        st.session_state["types_df"] = df_types
        st.session_state["data_df"] = cleaned_data[sheet_name]

# --- 3️⃣ Affichage et modification des types ---
if "types_df" in st.session_state:
    st.subheader("Types détectés (modifiable)")
    df_types = st.session_state["types_df"].copy()

    for i, row in df_types.iterrows():
        var = row["variable"]
        current_type = row["type"]
        new_type = st.selectbox(f"Type pour '{var}'", options=["numérique", "catégorielle", "binaire"],
                                index=["numérique", "catégorielle", "binaire"].index(current_type))
        df_types.at[i, "type"] = new_type

    st.session_state["types_df"] = df_types
    st.dataframe(df_types)
