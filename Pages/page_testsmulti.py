# Pages/page_testsmulti.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testsmultivaries import propose_test_multivariés

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

    # --- 3️⃣ Sélection des variables ---
    st.markdown("### 🎯 Choix des variables")
    target = st.selectbox("Variable à expliquer :", df.columns)
    possible_predictors = [col for col in df.columns if col != target]
    predictors = st.multiselect("Variables explicatives :", possible_predictors, default=possible_predictors)

    if not predictors:
        st.warning("⚠️ Sélectionnez au moins une variable explicative.")
        st.stop()

    lancer_test = st.button("🧠 Exécuter le test")

    if lancer_test:
        with st.spinner("Exécution du test en cours... ⏳"):
            try:
                results = propose_tests_multivariés(df, types_df, distribution_df, mots_cles=mots_cles,
                                                    target_var=target, predictor_vars=predictors)

                if not results:
                    st.warning("⚠️ Aucun résultat pour la sélection de variables actuelle.")
                    st.stop()

                for res in results:
                    st.markdown(f"### 📄 {res['test']}")
                    if "result_df" in res and res["result_df"] is not None:
                        st.dataframe(res["result_df"])
                    if "fig" in res and res["fig"] is not None:
                        st.pyplot(res["fig"])
                        plt.close(res["fig"])

                st.success("✅ Test multivarié exécuté avec succès !")

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution du test : {e}")
