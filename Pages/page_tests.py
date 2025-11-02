import streamlit as st
import pandas as pd
from modules.IA_STAT_interactif2 import propose_tests_interactif_streamlit

# ==============================================================
# PAGE : Tests statistiques interactifs
# ==============================================================

def app():
    st.title("🧮 Page des tests statistiques interactifs")

    # --- 1️⃣ Vérification des données disponibles ---
    st.header("1️⃣ Chargement des données")
    if "data" not in st.session_state or st.session_state["data"] is None:
        st.warning("⚠️ Aucune donnée chargée. Va sur la page **Import** pour importer un fichier.")
        return
    else:
        df = st.session_state["data"]
        st.success(f"✅ Données chargées ({df.shape[0]} lignes × {df.shape[1]} colonnes)")
        st.dataframe(df.head(5))

    # --- 2️⃣ Vérification des types de variables ---
    st.header("2️⃣ Types de variables détectés")
    if "types_df" not in st.session_state or st.session_state["types_df"] is None:
        st.warning("⚠️ Les types de variables ne sont pas encore détectés.")
        st.info("Va sur la page **Types de variables** pour effectuer la détection automatique.")
        return
    else:
        types_df = st.session_state["types_df"]
        st.dataframe(types_df)

    # --- 3️⃣ Vérification des distributions ---
    st.header("3️⃣ Distribution des variables")
    if "distribution_df" not in st.session_state or st.session_state["distribution_df"] is None:
        st.warning("⚠️ Aucune information sur les distributions détectée.")
        st.info("Tu peux continuer, mais certains tests (paramétriques vs non-paramétriques) ne seront pas proposés automatiquement.")
        distribution_df = pd.DataFrame(columns=["variable", "verdict"])
    else:
        distribution_df = st.session_state["distribution_df"]
        st.dataframe(distribution_df)

    # --- 4️⃣ Lancement de l’interface de tests ---
    st.header("4️⃣ Interface de tests statistiques")

    # On récupère les mots-clés optionnels (s’ils existent)
    mots_cles = st.session_state.get("mots_cles", [])

    try:
        propose_tests_interactif_streamlit(
            df=df,
            types_df=types_df,
            distribution_df=distribution_df,
            mots_cles=mots_cles
        )
    except Exception as e:
        st.error(f"❌ Erreur lors de l’exécution des tests : {e}")

    # --- 5️⃣ Résumé des résultats enregistrés ---
    st.header("5️⃣ Résultats enregistrés")
    if "tests_results" in st.session_state and st.session_state["tests_results"]:
        results_df = pd.DataFrame(st.session_state["tests_results"])
        st.dataframe(results_df)
        st.download_button(
            "⬇️ Télécharger les résultats (CSV)",
            results_df.to_csv(index=False).encode("utf-8"),
            file_name="tests_statistiques.csv",
            mime="text/csv"
        )
    else:
        st.info("Aucun test statistique n’a encore été exécuté.")

