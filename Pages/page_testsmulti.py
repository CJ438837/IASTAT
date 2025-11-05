# Pages/page_testsmulti.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

plt.style.use('seaborn-muted')

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

    # --- Récupération des données depuis la session ---
    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    st.markdown("### 🎯 Sélection des variables")
    var_dep = st.selectbox("Variable à expliquer (dépendante) :", df.columns)
    var_ind = st.multiselect(
        "Variables explicatives :", [v for v in df.columns if v != var_dep]
    )

    if not var_ind:
        st.warning("⚠️ Veuillez sélectionner au moins une variable explicative.")
        st.stop()

    lancer_tests = st.button("🧠 Exécuter les tests multivariés")

    if lancer_tests:
        with st.spinner("Exécution des tests en cours... ⏳"):
            try:
                # Filtrage des colonnes choisies
                df_sub = df[[var_dep] + var_ind].copy()
                types_sub = types_df[types_df['variable'].isin([var_dep] + var_ind)].copy()

                results = propose_tests_multivariés(
                    df_sub, types_sub, distribution_df, mots_cles
                )
                st.success("✅ Tests multivariés exécutés avec succès !")

                # Affichage des résultats
                for r in results:
                    st.markdown(f"### 📄 {r['test']}")
                    st.dataframe(r['result_df'])
                    st.pyplot(r['fig'])
                    plt.close(r['fig'])

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution des tests : {e}")
