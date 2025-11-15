import streamlit as st
import os
from modules.IA_STAT_distribution_251125 import advanced_distribution_analysis

def app():
    # --- 🎨 Thème Corvus ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Impossible de charger le thème Corvus : {e}")

    # --- 🧠 En-tête ---
    st.markdown("<h1 class='corvus-title'>Analyse de Distribution</h1>", unsafe_allow_html=True)
    st.markdown("""
    **Regardons la distribution de vos variables numériques.**
    **Indispenssable pour le choix adéquat des tests lors des prochaines étapes**
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

    # --- 2️⃣ Sélection de la variable ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Sélection de la variable numérique à analyser")
    st.markdown("<p class='corvus-text'>Choisissez la variable pour laquelle vous souhaitez étudier la distribution.</p>", unsafe_allow_html=True)

    numeric_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
    selected_var = st.selectbox("Variable à analyser", options=numeric_vars)

    st.markdown("</div>", unsafe_allow_html=True)

    if not selected_var:
        st.warning("⚠️ Sélectionnez une variable pour continuer.")
        st.stop()

    # --- 3️⃣ Bouton d'analyse ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Lancer l'analyse de distribution")

    run_analysis = st.button("📈 Démarrer l'analyse", use_container_width=True)

    if run_analysis:
        with st.spinner("Analyse en cours..."):
            output_folder = "distribution_plots"
            os.makedirs(output_folder, exist_ok=True)

            distribution_df = advanced_distribution_analysis(
                df[[selected_var]],
                types_df[types_df["variable"] == selected_var],
                output_folder=output_folder
            )

            st.session_state["distribution_df"] = distribution_df
            st.success("✅ Analyse terminée avec succès !")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- 4️⃣ Résumé des résultats ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🧾 Résumé des tests de distribution")

    if "distribution_df" in st.session_state and not st.session_state["distribution_df"].empty:
        st.dataframe(st.session_state["distribution_df"], use_container_width=True)
    else:
        st.info("Cliquez sur **Démarrer l'analyse** pour afficher les résultats.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

    # --- 5️⃣ Navigation des graphiques ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📉 Visualisations associées")

    output_folder = "distribution_plots"
    plot_files = sorted(
        [f for f in os.listdir(output_folder) if f.endswith(".png") and selected_var in f]
    )

    if not plot_files:
        st.warning("Aucun graphique généré pour cette variable.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "dist_plot_index" not in st.session_state:
        st.session_state.dist_plot_index = 0

    def prev_plot():
        if st.session_state.dist_plot_index > 0:
            st.session_state.dist_plot_index -= 1

    def next_plot():
        if st.session_state.dist_plot_index < len(plot_files) - 1:
            st.session_state.dist_plot_index += 1

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.button("⬅️ Précédent", on_click=prev_plot, key="prev_dist_plot")
    with col3:
        st.button("Suivant ➡️", on_click=next_plot, key="next_dist_plot")

    plot_path = os.path.join(output_folder, plot_files[st.session_state.dist_plot_index])
    st.image(plot_path, use_column_width=True)
    st.caption(
        f"Graphique {st.session_state.dist_plot_index + 1} / {len(plot_files)} : "
        f"{plot_files[st.session_state.dist_plot_index]}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # --- ➡️ Navigation ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Passer à la page Tests bivariés", use_container_width=True):
               st.session_state.main_page = "Analyse"
               st.session_state.analyse_subpage = "Tests bivariés"




