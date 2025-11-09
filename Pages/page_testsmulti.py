import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

plt.style.use("seaborn-v0_8-muted")

def app():
    # --- 🎨 Thème Corvus ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Impossible de charger le thème Corvus : {e}")

    # --- 🧠 En-tête ---
    st.markdown("<h1 class='corvus-title'> Tests Multivariés Avancés</h1>", unsafe_allow_html=True)
    st.markdown("<p class='corvus-subtitle'>Analysez les relations complexes entre plusieurs variables simultanément.</p>", unsafe_allow_html=True)

    # --- 1️⃣ Vérification des prérequis ---
    if "df_selected" not in st.session_state or st.session_state["df_selected"] is None:
        st.warning("⚠️ Veuillez d'abord charger un fichier dans l'onglet **Fichier**.")
        st.stop()

    df = st.session_state["df_selected"]

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

    st.success(f"✅ Données disponibles : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # --- 2️⃣ Aperçu des données ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📋 Aperçu du jeu de données")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- 3️⃣ Sélection des variables ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Sélection des variables à inclure dans l'analyse")

    target_var = st.selectbox("Variable à expliquer :", df.columns)

    explicatives = st.multiselect(
        "Variables explicatives :",
        [c for c in df.columns if c != target_var],
        default=[]
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if not explicatives:
        st.info("💡 Sélectionnez au moins une variable explicative pour continuer.")
        st.stop()

    # --- 4️⃣ Lancer les tests ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🚀 Lancer les tests multivariés")

    if st.button("🧠 Démarrer l'analyse multivariée", use_container_width=True):
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
                    st.subheader(f"🧩 {res.get('test', 'Test inconnu')}")

                    if "error" in res:
                        st.error(f"❌ Erreur : {res['error']}")
                        continue
                    if "message" in res:
                        st.warning(res["message"])
                        continue

                    if isinstance(res.get("result_df"), pd.DataFrame) and not res["result_df"].empty:
                        st.dataframe(res["result_df"], use_container_width=True)

                    if res.get("fig") is not None:
                        st.pyplot(res["fig"])

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution : {e}")

    st.markdown("</div>", unsafe_allow_html=True)

   
