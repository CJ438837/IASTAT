import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modules.IA_STAT_interactif2 import propose_tests_interactif_streamlit

def app():
    st.title("📊 Tests statistiques interactifs")
    st.write("Le dur du sujet ! Voyons ce que tes données ont dans le ventre.")

    # --- 1️⃣ Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser les distributions dans la page Distribution.")
        st.stop()

    df = st.session_state["df_selected"]
    types_df = st.session_state["types_df"]
    distribution_df = st.session_state["distribution_df"]
    mots_cles = st.session_state.get("keywords", [])

    # --- 2️⃣ Normalisation des colonnes ---
    # Si la colonne du type n'existe pas avec le nom 'type', on la renomme
    if 'type' not in types_df.columns:
        possible_names = ['var_type', 'Type', 'variable_type']
        for name in possible_names:
            if name in types_df.columns:
                types_df = types_df.rename(columns={name: 'type'})
                break
    st.session_state["types_df"] = types_df

    st.success("✅ Fichier importé, types détectés et distributions analysées.")

    # --- 3️⃣ Lancer les tests interactifs ---
    st.markdown("### 💡 Propositions de tests")
    if st.button("Lancer les tests interactifs"):
        propose_tests_interactif_streamlit(types_df, distribution_df, df, mots_cles)
        st.success("✅ Tous les tests interactifs ont été proposés et exécutés.")
