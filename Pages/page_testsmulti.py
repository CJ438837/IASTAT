# Pages/page_testsmulti.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.IA_STAT_testmultivaries import propose_tests_multivariés

plt.style.use("seaborn-v0_8-muted")  # même style que page bivariées

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

    # --- 2️⃣ Récupération des données ---
    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()

    st.markdown("### 🎯 Sélection de la variable à expliquer et des variables explicatives")

    # --- Choix de la variable cible ---
    target_var = st.selectbox(
        "Variable à expliquer (target) :",
        options=types_df["variable"].tolist()
    )

    # --- Choix des variables explicatives ---
    predictor_vars = st.multiselect(
        "Variables explicatives (predictors) :",
        options=[v for v in types_df["variable"].tolist() if v != target_var]
    )

    if not predictor_vars:
        st.warning("⚠️ Veuillez sélectionner au moins une variable explicative.")
        st.stop()

    lancer_tests = st.button("🧠 Exécuter le test multivarié")

    if lancer_tests:
        with st.spinner("Exécution du test en cours... ⏳"):
            try:
                results = propose_tests_multivariés(
                    df=df,
                    types_df=types_df,
                    target_var=target_var,
                    predictor_vars=predictor_vars
                )

                st.success("✅ Test(s) exécuté(s) avec succès !")

                # --- Affichage des résultats ---
                for res in results:
                    st.markdown(f"### 🧪 {res['test']}")
                    if "result_df" in res:
                        st.dataframe(res["result_df"])
                    if "fig" in res and res["fig"] is not None:
                        st.pyplot(res["fig"])
                        plt.close(res["fig"])

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution du test : {e}")
