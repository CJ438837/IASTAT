# Pages/page_testsmulti.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

def app():
    st.title("📊 Tests statistiques multivariés")

    # --- Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser la distribution des données dans la page Distribution.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    st.markdown("### 🎯 Sélection des tests multivariés")

    # --- 1️⃣ Sélection de la variable cible ---
    num_vars = types_df[types_df["type"] == "numérique"]["variable"].tolist()
    bin_vars = types_df[types_df["type"] == "binaire"]["variable"].tolist()
    cat_vars = types_df[types_df["type"].isin(["catégorielle", "binaire"])]["variable"].tolist()

    target_type = st.radio("Type de variable cible :", ["Numérique", "Binaire", "Catégorielle"])
    if target_type == "Numérique":
        target_var = st.selectbox("Variable dépendante :", num_vars)
        predictor_vars = st.multiselect("Variables explicatives :", [v for v in num_vars if v != target_var])
    elif target_type == "Binaire":
        target_var = st.selectbox("Variable dépendante :", bin_vars)
        predictor_vars = st.multiselect("Variables explicatives :", num_vars)
    else:
        st.info("Pour l'instant, les tests multivariés sont limités aux numériques et binaires.")
        st.stop()

    lancer_tests = st.button("🧠 Exécuter le test")

    if lancer_tests:
        if len(predictor_vars) == 0:
            st.warning("⚠️ Veuillez sélectionner au moins une variable explicative.")
        else:
            with st.spinner("Exécution du test en cours... ⏳"):
                try:
                    # Création d'un sous-DataFrame pour éviter les NaN
                    df_subset = df[[target_var] + predictor_vars].dropna()
                    types_subset = types_df[types_df["variable"].isin([target_var] + predictor_vars)].copy()

                    results = propose_tests_multivariés(df_subset, types_subset, distribution_df, mots_cles)

                    if len(results) == 0:
                        st.warning("Aucun test n'a été exécuté. Vérifiez vos variables sélectionnées.")
                    else:
                        for res in results:
                            st.markdown(f"### 🧪 {res['test']}")
                            st.dataframe(res["result_df"])
                            if res.get("fig") is not None:
                                st.pyplot(res["fig"])
                                plt.close(res["fig"])

                    st.success("✅ Test terminé !")

                except Exception as e:
                    st.error(f"❌ Une erreur est survenue pendant l'exécution du test : {e}")
