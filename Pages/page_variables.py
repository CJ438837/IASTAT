import streamlit as st

def app():
    st.header("Variables")
    st.write("Définissons ensemble le type de variables qui composent ton étude, le premier pas pour des tests de qualité")
# Pages/2_Variables.py
import streamlit as st
import pandas as pd
from modules.IA_STAT_typevariable_251125 import detect_variable_types

# --- 1️⃣ Récupérer le fichier depuis la session ---
if "uploaded_file" not in st.session_state:
    st.warning("Veuillez d'abord importer un fichier dans la page 'Fichier'.")
    st.stop()

uploaded_file = st.session_state["uploaded_file"]

st.header("📝 Détection des types de variables")

# --- 2️⃣ Choix de la feuille (si Excel multi-feuilles) ---
sheet_name = None
if uploaded_file.name.endswith(('.xls', '.xlsx')):
    all_sheets = pd.ExcelFile(uploaded_file)
    if len(all_sheets.sheet_names) > 1:
        sheet_name = st.selectbox("Choisir la feuille à analyser :", all_sheets.sheet_names)
    else:
        sheet_name = all_sheets.sheet_names[0]

# --- 3️⃣ Bouton pour lancer la détection automatique ---
if st.button("Détecter automatiquement les types de variables"):
    types_results, cleaned_data = detect_variable_types(uploaded_file, sheet_name)
    df_types = types_results[sheet_name]
    st.session_state["types_df"] = df_types
    st.session_state["data_df"] = cleaned_data[sheet_name]

# --- 4️⃣ Affichage et modification des types ---
if "types_df" in st.session_state:
    st.subheader("Types détectés (modifiable)")
    df_types = st.session_state["types_df"].copy()

    # Permettre à l'utilisateur de modifier le type
    for i, row in df_types.iterrows():
        var = row["variable"]
        current_type = row["type"]
        new_type = st.selectbox(f"Type pour '{var}'", options=["numérique", "catégorielle", "binaire"], index=["numérique", "catégorielle", "binaire"].index(current_type))
        df_types.at[i, "type"] = new_type

    st.session_state["types_df"] = df_types
    st.dataframe(df_types)
