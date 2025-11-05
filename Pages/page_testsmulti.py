# Pages/page_testsmulti.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testsmulti import propose_tests_multivariés

plt.style.use('seaborn-v0_8-muted')

def app():
    st.title("📊 Tests statistiques multivariés")

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

    # --- 3️⃣ Sélection des options utilisateur ---
    st.markdown("### 🎯 Sélection du test multivarié à réaliser")

    test_options = [
        "Régression linéaire multiple",
        "Régression logistique",
        "ACP (PCA)",
        "ACM (MCA)"
    ]
    test_selected = st.selectbox("Choisissez le test :", test_options)

    lancer_tests = st.button("🧠 Exécuter le test")

    if lancer_tests:
        with st.spinner("Exécution du test en cours... ⏳"):
            try:
                results = propose_tests_multivariés(df, types_df, distribution_df, mots_cles=mots_cles)

                # --- 4️⃣ Filtrer pour le test choisi ---
                filtered_results = [r for r in results if r["test"].startswith(test_selected)]

                if not filtered_results:
                    st.warning("⚠️ Aucun résultat pour ce test avec les données sélectionnées.")
                    st.stop()

                for res in filtered_results:
                    st.markdown(f"### 📄 {res['test']}")
                    if "result_df" in res and res["result_df"] is not None:
                        st.dataframe(res["result_df"])
                    if "fig" in res and res["fig"] is not None:
                        st.pyplot(res["fig"])
                        plt.close(res["fig"])

                st.success("✅ Test multivarié exécuté avec succès !")

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution du test : {e}")
