import streamlit as st
import os
from modules.IA_STAT_descriptive_251125 import descriptive_analysis
from modules.IA_STAT_Illustrations_251125 import plot_descriptive
import pandas as pd

def app():
    st.title("📊 Analyse Descriptive")

    # --- 1️⃣ Vérification prérequis ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()

    df = st.session_state["df_selected"]
    types_df = st.session_state["types_df"]

    st.success("✅ Fichier importé et types de variables détectés.")

    # --- 2️⃣ Sélection des variables ---
    st.subheader("Sélection des variables")
    numeric_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
    cat_vars = types_df[types_df["type"].isin(["catégorielle", "binaire"])]["variable"].tolist()
    all_vars = numeric_vars + cat_vars

    explicatives = st.multiselect("Choisir les variables à analyser", options=all_vars)

    if not explicatives:
        st.warning("Sélectionnez au moins une variable pour continuer.")
        st.stop()

    # Optionnel : variable catégorielle pour grouper
    group_var = st.selectbox("Optionnel : Grouper par variable catégorielle", options=[None]+cat_vars)

    # --- 3️⃣ Résumé descriptif ---
    st.subheader("Résumé descriptif des variables")

    if group_var:
        grouped = df.groupby(group_var)
        for grp_name, grp_df in grouped:
            st.markdown(f"### Groupe : {grp_name}")
            summary = descriptive_analysis(grp_df, types_df[types_df["variable"].isin(explicatives)])
            for var, stats in summary.items():
                st.markdown(f"**{var}**")
                st.json(stats)
    else:
        summary = descriptive_analysis(df[explicatives], types_df[types_df["variable"].isin(explicatives)])
        for var, stats in summary.items():
            st.markdown(f"**{var}**")
            st.json(stats)

    # --- 4️⃣ Graphiques descriptifs ---
    st.subheader("Visualisations des variables")
    output_folder = "plots"

    # Crée le dossier si nécessaire
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Génération des graphiques uniquement pour les variables sélectionnées
    plot_descriptive(df[explicatives], types_df[types_df["variable"].isin(explicatives)], 
                     output_folder=output_folder)

    # Liste des fichiers générés pour les variables sélectionnées
    plot_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png") and any(v in f for v in explicatives)])
    if not plot_files:
        st.warning("Aucun graphique généré pour les variables sélectionnées.")
        return

    # Initialisation de l'indice du graphique
    if "plot_index" not in st.session_state:
        st.session_state.plot_index = 0

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Précédent"):
            if st.session_state.plot_index > 0:
                st.session_state.plot_index -= 1
    with col3:
        if st.button("Suivant ➡️"):
            if st.session_state.plot_index < len(plot_files) - 1:
                st.session_state.plot_index += 1

    # Affichage du graphique courant
    plot_path = os.path.join(output_folder, plot_files[st.session_state.plot_index])
    st.image(plot_path, use_column_width=True)
    st.caption(f"Graphique {st.session_state.plot_index + 1} / {len(plot_files)} : {plot_files[st.session_state.plot_index]}")
