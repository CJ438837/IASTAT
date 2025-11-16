# Pages/page_testsmulti.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

st.set_page_config(page_title="Tests Multivariés", layout="wide")

st.title("📊 Analyse Multivariée Automatisée")

# ---------------------------
# Upload du fichier
# ---------------------------
uploaded_file = st.file_uploader("Choisir un fichier Excel ou CSV", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        st.stop()

    st.success(f"Fichier chargé avec succès : {uploaded_file.name}")
    st.write("Aperçu des données :")
    st.dataframe(df.head())

    # ---------------------------
    # Définition types variables
    # ---------------------------
    st.sidebar.header("Configuration des variables")
    target_var = st.sidebar.selectbox("Variable cible (target)", df.columns)
    explicatives = st.sidebar.multiselect("Variables explicatives", [c for c in df.columns if c != target_var])

    # Types de variables (optionnel)
    types_df = pd.DataFrame({
        "variable": df.columns,
        "type": ["numérique" if pd.api.types.is_numeric_dtype(df[c]) else "catégorielle" for c in df.columns]
    })

    # ---------------------------
    # Bouton pour lancer l'analyse
    # ---------------------------
    if st.sidebar.button("📈 Lancer l'analyse multivariée"):

        with st.spinner("Analyse en cours…"):
            results = propose_tests_multivariés(df, types_df, target_var, explicatives)

        st.success("✅ Analyse terminée")

        # ---------------------------
        # Affichage des résultats
        # ---------------------------
        for res in results:
            test_name = res.get("test", "Test inconnu")
            st.subheader(f"🧪 {test_name}")

            # Erreurs
            if res.get("error"):
                st.error(f"Erreur : {res['error']}")

            # DataFrame
            df_res = res.get("result_df")
            if df_res is not None:
                st.dataframe(df_res)

            # Figure
            fig = res.get("fig")
            if fig:
                st.pyplot(fig)

            # Additional info
            info = res.get("additional_info")
            if info:
                # On s'assure que c'est un dict
                if isinstance(info, dict):
                    st.write("ℹ️ Informations complémentaires :")
                    for k, v in info.items():
                        st.write(f"- **{k}** : {v}")
                else:
                    st.write("ℹ️ Informations complémentaires :")
                    st.write(info)

            # Interprétation
            interp = res.get("interpretation")
            if interp:
                st.info(f"💡 Interprétation : {interp}")

            st.markdown("---")
