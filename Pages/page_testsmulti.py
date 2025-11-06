import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

plt.style.use("seaborn-v0_8-muted")

def app():
    st.title("📊 Tests Multivariés Avancés")

    # Récupération du fichier chargé dans la page Fichier
    if "df_selected" not in st.session_state or st.session_state["df_selected"] is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier dans l'onglet **Fichier**.")
        return

    df = st.session_state["df_selected"]

    # Chargement des types de variables (s’ils sont déjà détectés)
    if "types_df" not in st.session_state or st.session_state["types_df"] is None:
        types_df = pd.DataFrame({
            "variable": df.columns,
            "type": [
                "numérique" if pd.api.types.is_numeric_dtype(df[col]) else "catégorielle"
                for col in df.columns
            ]
        })
        st.session_state["types_df"] = types_df
    else:
        types_df = st.session_state["types_df"]

    st.success(f"✅ Données disponibles ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
    st.write("### 📋 Aperçu des données")
    st.dataframe(df.head())

    # --- Sélection de la variable à expliquer ---
    st.divider()
    st.subheader("🎯 Sélection des variables")

    target_var = st.selectbox("Variable à expliquer :", df.columns)

    explicatives = st.multiselect(
        "Variables explicatives :",
        [c for c in df.columns if c != target_var],
        default=[]
    )

    if not explicatives:
        st.info("💡 Sélectionnez au moins une variable explicative pour continuer.")
        return

    # --- Bouton d'exécution ---
    if st.button("🚀 Lancer les tests multivariés", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            try:
                results = propose_tests_multivariés(
                    df,
                    types_df,
                    target_var=target_var,
                    explicatives=explicatives
                )

                for res in results:
                    st.divider()
                    st.subheader(f"🧠 {res.get('test', 'Test inconnu')}")

                    # Gestion des erreurs
                    if "error" in res:
                        st.error(f"❌ Erreur : {res['error']}")
                        continue
                    if "message" in res:
                        st.warning(res["message"])
                        continue

                    # Tableau des résultats
                    if isinstance(res.get("result_df"), pd.DataFrame) and not res["result_df"].empty:
                        st.dataframe(res["result_df"], use_container_width=True)

                    # Graphique
                    if res.get("fig") is not None:
                        st.pyplot(res["fig"])

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution : {e}")
