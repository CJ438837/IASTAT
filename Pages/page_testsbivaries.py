# Pages/page_testsbivaries.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def app():
    st.title("📊 Tests statistiques bivariés automatiques")

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

    # --- 3️⃣ Options utilisateur ---
    st.markdown("### ⚙️ Options des tests")
    apparie = st.radio(
        "Les tests à deux groupes sont-ils appariés ?",
        ("Non", "Oui"),
        index=0
    ) == "Oui"

    lancer_tests = st.button("🧠 Exécuter tous les tests bivariés")

    if lancer_tests:
        with st.spinner("Exécution des tests en cours... ⏳"):
            try:
                # Appel de la fonction sans argument 'apparie', on utilise default_apparie
                summary_df, all_results = propose_tests_bivaries(
                    types_df=types_df,
                    distribution_df=distribution_df,
                    df=df,
                    default_apparie=apparie
                )
                st.success("✅ Tous les tests bivariés ont été exécutés avec succès !")

                # --- 4️⃣ Affichage du résumé des tests ---
                st.markdown("### 📄 Résumé des tests")
                st.dataframe(summary_df)

                # --- 5️⃣ Affichage des plots ---
                st.markdown("### 📊 Graphiques associés")
                for test_id, details in all_results.items():
                    st.write(f"**Test : {test_id}**")
                    if "plot" in details and details["plot"] is not None:
                        st.image(details["plot"])
                    elif "plot_boxplot" in details and details["plot_boxplot"] is not None:
                        st.image(details["plot_boxplot"])

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution des tests : {e}")
