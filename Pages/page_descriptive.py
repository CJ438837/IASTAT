import streamlit as st
import os
import pandas as pd
from modules.IA_STAT_descriptive_251125 import descriptive_analysis
from modules.IA_STAT_Illustrations_251125 import plot_descriptive

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
    all_vars = types_df["variable"].tolist()
    explicatives = st.multiselect("Variables à analyser :", all_vars)
    if not explicatives:
        st.warning("Veuillez sélectionner au moins une variable.")
        st.stop()

    group_var = st.selectbox("Grouper par (optionnel) :", [None] + all_vars)

    # --- 3️⃣ Lancer l'analyse ---
    if st.button("📈 Lancer l'analyse descriptive") or "result_df" not in st.session_state:

        # Nettoyage des anciens résultats
        st.session_state.result_df = pd.DataFrame()
        st.session_state.summary_dict = {}

        groupes = [None]
        if group_var:
            groupes = df[group_var].dropna().unique()

        records = []
        for g in groupes:
            if g is not None:
                df_grp = df[df[group_var] == g]
                grp_label = str(g)
            else:
                df_grp = df
                grp_label = "Tous"

            summary_dict = descriptive_analysis(df_grp[explicatives],
                                                types_df[types_df["variable"].isin(explicatives)])

            # Transformer en dataframe plat
            for var, stats in summary_dict.items():
                flat = {"Variable": var, "Groupe": grp_label}
                for k, v in stats.items():
                    if isinstance(v, dict):
                        flat[k] = str(v)
                    else:
                        flat[k] = v
                records.append(flat)

        st.session_state.result_df = pd.DataFrame(records)
        st.session_state.summary_dict = summary_dict

    # --- 4️⃣ Affichage du tableau ---
    st.subheader("Résumé descriptif")
    if not st.session_state.result_df.empty:
        st.dataframe(st.session_state.result_df)
    else:
        st.warning("Aucun résultat à afficher. Lancez l'analyse.")

    # --- 5️⃣ Graphiques descriptifs ---
    st.subheader("Visualisations des variables")
    output_folder = "plots"
    plot_descriptive(df, types_df, output_folder=output_folder)

    # Liste des fichiers générés
    plot_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png")])
    if not plot_files:
        st.warning("Aucun graphique généré.")
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
