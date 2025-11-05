# Pages/page_testsmulti.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

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

    # --- 3️⃣ Sélection de la variable cible et des prédicteurs ---
    st.header("🎯 Sélection de la variable cible et des variables explicatives")

    target_var = st.selectbox("Variable à expliquer :", types_df['variable'].tolist())
    possible_predictors = [v for v in types_df['variable'] if v != target_var]
    predictors = st.multiselect("Variables explicatives :", possible_predictors, default=possible_predictors[:3])

    if not predictors:
        st.warning("⚠️ Veuillez sélectionner au moins une variable explicative.")
        st.stop()

    # --- 4️⃣ Bouton pour lancer le test ---
    lancer_test = st.button("🧠 Exécuter le test")

    if lancer_test:
        with st.spinner("Exécution du test en cours... ⏳"):
            try:
                results = propose_tests_multivariés(df, types_df, distribution_df, target_var, predictors)

                st.success("✅ Test exécuté avec succès !")

                # Affichage des résultats
                for key, res in results.items():
                    st.markdown(f"### 📄 {res['test']}")
                    st.dataframe(res["result_df"])
                    if res.get("fig") is not None:
                        st.pyplot(res["fig"])
                        plt.close(res["fig"])

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution du test : {e}")

    # --- 5️⃣ Conseils et info ---
    st.markdown(
        """
        **Conseils :**
        - Pour une variable numérique cible : régression linéaire multiple
        - Pour une variable binaire : régression logistique
        - Pour une variable catégorielle multi‑modalités : régression logistique multinomiale
        - PCA et MCA sont réalisées automatiquement si applicable pour analyse exploratoire
        """
    )
