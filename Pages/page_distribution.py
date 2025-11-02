import streamlit as st
import os
from modules.IA_STAT_distribution_251125 import advanced_distribution_analysis

def app():
    st.title("📈 Analyse de Distribution")

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

    # --- 2️⃣ Tableau récapitulatif des distributions ---
    st.subheader("Résumé des tests de distribution")
    output_folder = "distribution_plots"

    # Exécution de l'analyse
    distribution_df = advanced_distribution_analysis(df, types_df, output_folder=output_folder)
    st.dataframe(distribution_df)

    # --- 2️⃣b Sauvegarde dans la session ---
    st.session_state["distribution_df"] = distribution_df

    # --- 3️⃣ Navigation des graphiques ---
    plot_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png")])
    if not plot_files:
        st.warning("Aucun graphique généré.")
        return

    if "dist_plot_index" not in st.session_state:
        st.session_state.dist_plot_index = 0

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅️ Précédent"):
            if st.session_state.dist_plot_index > 0:
                st.session_state.dist_plot_index -= 1
    with col3:
        if st.button("Suivant ➡️"):
            if st.session_state.dist_plot_index < len(plot_files) - 1:
                st.session_state.dist_plot_index += 1

    # Affichage du graphique courant
    plot_path = os.path.join(output_folder, plot_files[st.session_state.dist_plot_index])
    st.image(plot_path, use_column_width=True)
    st.caption(f"Graphique {st.session_state.dist_plot_index + 1} / {len(plot_files)} : {plot_files[st.session_state.dist_plot_index]}")

