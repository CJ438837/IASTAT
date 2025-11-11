import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testbivaries import propose_tests_bivaries

plt.style.use("seaborn-v0_8-muted")

def app():
    # --- 🎨 Thème Corvus (si présent) ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Impossible de charger le thème Corvus : {e}")

    # --- 🧠 En-tête ---
    st.markdown("<h1 class='corvus-title'> Tests Bivariés</h1>", unsafe_allow_html=True)
    st.markdown("<p class='corvus-subtitle'>Analysez les relations entre deux variables selon leur nature.</p>", unsafe_allow_html=True)

    # --- 1️⃣ Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer un fichier dans la page **Fichier**.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord détecter les types de variables dans la page **Variables**.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord analyser la distribution des données dans la page **Distribution**.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()

    st.success("✅ Données et analyses de distribution prêtes.")

    # --- 2️⃣ Sélection des variables ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 🎯 Sélection des variables à comparer")

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Variable dépendante (Y)", df.columns)
    with col2:
        var2 = st.selectbox("Variable explicative (X)", df.columns, index=min(1, len(df.columns) - 1))

    if var1 == var2:
        st.warning("⚠️ Sélectionnez deux variables différentes.")
        st.stop()

    # Détection automatique du type
    type1 = types_df.loc[types_df["variable"] == var1, "type"].values[0]
    type2 = types_df.loc[types_df["variable"] == var2, "type"].values[0]
    st.markdown(f"**Types détectés :** `{var1}` → {type1}, `{var2}` → {type2}")

    # --- 3️⃣ Options de test ---
    apparie = False
    if type1 == "numérique" and type2 == "numérique":
        st.info("Un test de corrélation (Pearson, Spearman ou Kendall) sera appliqué selon la distribution.")
    elif (type1 == "numérique" and type2 in ["catégorielle", "binaire"]) or (type2 == "numérique" and type1 in ["catégorielle", "binaire"]):
        apparie = st.radio(
            "Les deux groupes sont-ils appariés ?",
            ["Non", "Oui"],
            index=0,
            horizontal=True
        ) == "Oui"
    elif type1 in ["catégorielle", "binaire"] and type2 in ["catégorielle", "binaire"]:
        st.info("Un test du Chi² ou de Fisher sera utilisé selon la taille de la table.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- 4️⃣ Lancement du test ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Lancer les tests bivariés")

    if st.button("📊 Démarrer l'analyse bivariée", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            try:
                results = propose_tests_bivaries(
                    df=df,
                    types_df=types_df,
                    distribution_df=distribution_df,
                    var1=var1,
                    var2=var2,
                    default_apparie=apparie
                )

                if not isinstance(results, list) or len(results) == 0:
                    st.error("❌ Format inattendu : la fonction n'a pas renvoyé de résultats exploitables.")
                    st.stop()

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

    # --- 5️⃣ Navigation entre pages ---
    st.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Page suivante : Tests multivariés"):
            st.session_state.target_page = "Tests multivariés"
