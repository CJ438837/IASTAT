import streamlit as st
import pandas as pd
from modules.IA_STAT_interactif2 import propose_tests_interactif_streamlit

def app():
    st.title("📊 Tests statistiques interactifs")

    # --- 1️⃣ Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser la distribution des données dans la page Distribution.")
        st.stop()

    # --- 2️⃣ Récupération des données depuis la session ---
    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    # --- 3️⃣ Vérification et normalisation des colonnes de types_df ---
    rename_dict = {}
    for col in types_df.columns:
        lower = col.lower()
        if lower in ["var", "variable_name", "nom", "column"]:
            rename_dict[col] = "variable"
        elif lower in ["var_type", "type_var", "variable_type", "kind"]:
            rename_dict[col] = "type"

    types_df.rename(columns=rename_dict, inplace=True)

    # Vérifie que les colonnes attendues existent
    expected_cols = {"variable", "type"}
    if not expected_cols.issubset(types_df.columns):
        st.error(f"⚠️ Le tableau des types de variables doit contenir les colonnes : {expected_cols}. "
                 f"Colonnes actuelles : {types_df.columns.tolist()}")
        st.stop()

    # --- 4️⃣ Interface utilisateur ---
    st.success("✅ Toutes les données nécessaires ont été chargées.")
    st.markdown("### 💡 Propositions de tests statistiques adaptés")

    lancer_tests = st.button("🧠 Lancer les tests interactifs")


    if lancer_tests:
        with st.spinner("Analyse en cours... ⏳"):
            propose_tests_interactif_streamlit(df, types_df, distribution_df, mots_cles)
        st.success("✅ Les tests interactifs ont été exécutés avec succès.")





