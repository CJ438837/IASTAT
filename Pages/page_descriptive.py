# Pages/3_Analyse_Descriptive.py
import streamlit as st
import os
import pandas as pd
from modules.IA_STAT_descriptive_251125 import descriptive_analysis
from modules.IA_STAT_Illustrations_251125 import plot_descriptive

def app():
    # --- 🎨 Thème Corvus ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Impossible de charger le thème Corvus : {e}")

    # --- 🧠 En-tête ---
    st.markdown("<h1 class='corvus-title'>Analyse descriptive</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Rentrons dans le concret ! L'analyse descriptive, voici la première vraie analyse.**
    **Voyons la "carte" d'identité de vos variables et vos premières illustrations!**
    """)

    # --- 1️⃣ Vérification des prérequis ---
    if "df_selected" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer un fichier dans la page **Fichier**.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord détecter les types de variables dans la page **Variables**.")
        st.stop()

    df = st.session_state["df_selected"]
    types_df = st.session_state["types_df"]

    st.success("✅ Données et types de variables prêts pour l'analyse.")

    # --- 2️⃣ Sélection des variables ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Sélection des variables")
    st.markdown("<p class='corvus-text'>Choisissez les variables que vous souhaitez analyser.</p>", unsafe_allow_html=True)

    all_vars = types_df["variable"].tolist()
    explicatives = st.multiselect("Variables à analyser :", all_vars)
    group_var = st.selectbox("Variable de regroupement (optionnel) :", [None] + all_vars)

    st.markdown("</div>", unsafe_allow_html=True)

    if not explicatives:
        st.warning("⚠️ Veuillez sélectionner au moins une variable.")
        st.stop()

    # --- 3️⃣ Lancer l'analyse ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Lancer l'analyse descriptive")

    run_analysis = st.button("📈 Démarrer l'analyse descriptive", use_container_width=True)

    if run_analysis:
        with st.spinner("Analyse en cours..."):
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

                summary_dict = descriptive_analysis(
                    df_grp[explicatives],
                    types_df[types_df["variable"].isin(explicatives)]
                )

                for var, stats in summary_dict.items():
                    flat = {"Variable": var, "Groupe": grp_label}
                    for k, v in stats.items():
                        flat[k] = str(v) if isinstance(v, dict) else v
                    records.append(flat)

            st.session_state.result_df = pd.DataFrame(records)
            st.session_state.summary_dict = summary_dict

        st.success("✅ Analyse terminée avec succès !")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- 4️⃣ Résumé descriptif ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🧾 Résumé descriptif")

    if "result_df" in st.session_state and not st.session_state.result_df.empty:
        st.dataframe(st.session_state.result_df, use_container_width=True)
    else:
        st.info("Cliquez sur **Démarrer l'analyse descriptive** pour afficher les résultats.")
        st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

    # --- 5️⃣ Graphiques descriptifs ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📉 Visualisations des variables sélectionnées")

    output_folder = "plots"
    os.makedirs(output_folder, exist_ok=True)

    # ⚙️ Génération des graphiques uniquement après le clic sur "Démarrer"
    if run_analysis:
        plot_descriptive(
            df=df,
            types_df=types_df[types_df["variable"].isin(explicatives)],
            output_folder=output_folder,
            selected_vars=explicatives,
            group_var=group_var
        )

    plot_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".png")])
    if not plot_files:
        st.warning("Aucun graphique généré pour les variables sélectionnées.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # --- Navigation entre les graphiques (sans refresh complet) ---
    if "plot_index" not in st.session_state:
        st.session_state.plot_index = 0

    def prev_plot():
        if st.session_state.plot_index > 0:
            st.session_state.plot_index -= 1

    def next_plot():
        if st.session_state.plot_index < len(plot_files) - 1:
            st.session_state.plot_index += 1

    nav_container = st.container()
    with nav_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.button("⬅️ Précédent", on_click=prev_plot)
        with col3:
            st.button("Suivant ➡️", on_click=next_plot)

    plot_path = os.path.join(output_folder, plot_files[st.session_state.plot_index])
    st.image(plot_path, use_column_width=True)
    st.caption(
        f"Graphique {st.session_state.plot_index + 1} / {len(plot_files)} : "
        f"{plot_files[st.session_state.plot_index]}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- ➡️ Navigation ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Passer à la page Distribution", use_container_width=True):
            st.session_state.main_page = "Analyse"
            st.session_state.analyse_subpage = "Distribution"
    with col3:
        if st.button("➡️ Besoin d'une aide théorique ?", use_container_width=True):
            st.session_state.main_page = "Théorie"
            st.session_state.theorie_subpage = "Descriptive"






