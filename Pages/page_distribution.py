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

    # --- 2️⃣ Sélection de la variable ---
    st.subheader("Choix de la variable numérique à analyser")
    numeric_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
    selected_var = st.selectbox("Variable à analyser", options=numeric_vars)

    if not selected_var:
        st.warning("Sélectionnez une variable pour continuer.")
        st.stop()

    # --- 3️⃣ Bouton pour lancer l'analyse ---
    if st.button("Lancer l'analyse"):
        output_folder = "distribution_plots"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Exécution de l'analyse pour la variable sélectionnée
        distribution_df = advanced_distribution_analysis(df[[selected_var]], types_df[types_df["variable"] == selected_var], 
                                                         output_folder=output_folder)

        st.subheader("Résumé des tests de distribution")
        st.dataframe(distribution_df)

        # Sauvegarde dans la session
        st.session_state["distribution_df"] = distribution_df

        # --- 4️⃣ Navigation des graphiques ---
        plot_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png") and selected_var in f])
        if not plot_files:
            st.warning("Aucun graphique généré pour cette variable.")
            return

        if "dist_plot_index" not in st.session_state:
            st.session_state.dist_plot_index = 0

        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.button("⬅️ Précédent", key="prev_plot"):
                if st.session_state.dist_plot_index > 0:
                    st.session_state.dist_plot_index -= 1
        with col3:
            if st.button("Suivant ➡️", key="next_plot"):
                if st.session_state.dist_plot_index < len(plot_files) - 1:
                    st.session_state.dist_plot_index += 1

        # Affichage du graphique courant
        plot_path = os.path.join(output_folder, plot_files[st.session_state.dist_plot_index])
        st.image(plot_path, use_column_width=True)
        st.caption(f"Graphique {st.session_state.dist_plot_index + 1} / {len(plot_files)} : {plot_files[st.session_state.dist_plot_index]}")
