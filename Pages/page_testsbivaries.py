import streamlit as st
import pandas as pd
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def app():
    st.title(" Analyse bivariée")

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

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()

    st.success("✅ Données et analyses de distribution prêtes.")

    # --- 2️⃣ Sélection des variables ---
    st.subheader("🎯 Sélection des variables à comparer")

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

    # --- 4️⃣ Lancement du test ---
    if st.button("🧪 Démarrer le test"):
        with st.spinner("Exécution du test... ⏳"):
            try:
                summary_df, details = propose_tests_bivaries(
                    types_df=types_df,
                    distribution_df=distribution_df,
                    df=df,
                    default_apparie=apparie
                )

                key = f"{var1}__{var2}"
                if key not in details:
                    st.warning("Aucun résultat trouvé pour cette paire de variables.")
                    st.stop()

                result = details[key]

                # --- Résumé du test ---
                st.subheader("📋 Résultats du test")
                st.dataframe(pd.DataFrame([{
                    "Test": result.get("test"),
                    "Statistique": result.get("statistic"),
                    "p-value": result.get("p_value"),
                    "Effect size": result.get("effect_size", None),
                    "Cramer's V": result.get("cramers_v", None)
                }]))

                # --- Graphique associé ---
                st.subheader("📊 Visualisation du résultat")
                plot_path = result.get("plot") or result.get("plot_boxplot")
                if plot_path:
                    st.image(plot_path, use_column_width=True)
                else:
                    st.info("Aucun graphique disponible pour ce test.")

            except Exception as e:
                st.error(f"❌ Erreur pendant l'exécution du test : {e}")

    # --- 5️⃣ Navigation entre pages ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Page suivante : Tests multivariés"):
            st.session_state.target_page = "Tests multivariés"


