import streamlit as st

def app():
    st.header("Statistiques descriptives")
    st.write("Ici nous allons réaliser les statistiques descriptive de ton étude")

import streamlit as st
import pandas as pd
import os
from modules.IA_STAT_descriptive_251125 import descriptive_analysis
from modules.IA_STAT_Illustrations_251125 import plot_descriptive

def app():
    st.title("📊 Analyse Descriptive")

    # --- Récupération des données et types ---
    if 'df_selected' not in st.session_state or 'df_types' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer un fichier et détecter les types de variables dans la page Fichier et Variables.")
        return

    df = st.session_state['df_selected']
    types_results = st.session_state['df_types']  # dictionnaire de feuilles

    # --- Choix de la feuille à analyser ---
    feuille = st.selectbox("Choisir la feuille à analyser :", list(df_types.keys()))
    types_df = df_types[feuille]

    # --- Bouton pour lancer l'analyse ---
    if st.button("🧮 Lancer l'analyse descriptive"):
        # --- 1️⃣ Analyse descriptive ---
        summary = descriptive_analysis(df, types_df)
        st.subheader("Résumé statistique des variables")
        for var, stats_dict in summary.items():
            st.markdown(f"**{var}**")
            st.json(stats_dict)

        # --- 2️⃣ Génération des graphiques ---
        st.subheader("Graphiques descriptifs")
        output_folder = f"plots/{feuille}"
        plot_descriptive(df, types_df, output_folder=output_folder)

        # --- 3️⃣ Affichage des graphiques avec défiler ---
        images = [f for f in os.listdir(output_folder) if f.endswith(".png")]
        images.sort()

        if images:
            selected_img_idx = st.number_input(
                "Sélectionner un graphique",
                min_value=0,
                max_value=len(images)-1,
                value=0,
                step=1
            )
            img_path = os.path.join(output_folder, images[selected_img_idx])
            st.image(img_path, use_column_width=True)
        else:
            st.info("Aucun graphique généré pour cette feuille.")

