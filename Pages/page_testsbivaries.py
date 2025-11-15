# Pages/page_testsbivaries.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testbivaries import propose_tests_bivaries

plt.style.use("seaborn-v0_8-muted")


def app():
    st.title("Tests statistiques bivariés")
    st.markdown("""
    **Étudions l'impact des variables les unes sur les autres.**
    **Ici l'étude se fait avec une variable dépendante et une variable explicative.**
    **Voyons ce qu'il en ressort avec les résultats des tests et des illustrations graphiques**
    """)

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

    st.success("✅ Données et analyses de distribution prêtes.")

    # --- Sélection d'une paire de variables ---
    st.header("🎯 Sélection des variables à comparer")
    cols = df.columns.tolist()

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Variable dépendante (Y)", cols)
    with col2:
        var2 = st.selectbox("Variable explicative (X)", cols, index=min(1, len(cols) - 1))

    if var1 == var2:
        st.warning("⚠️ Veuillez sélectionner deux variables différentes.")
        st.stop()

    # --- Détection automatique du type ---
    try:
        type1 = types_df.loc[types_df["variable"] == var1, "type"].values[0]
        type2 = types_df.loc[types_df["variable"] == var2, "type"].values[0]
    except Exception as e:
        st.error(f"Erreur lors de la détection des types : {e}")
        st.stop()

    st.markdown(f"**Types détectés :** `{var1}` → **{type1}**, `{var2}` → **{type2}**")

    # --- Appariement si applicable ---
    apparie = False
    if (type1 == "numérique" and type2 in ["catégorielle", "binaire"]) or (type2 == "numérique" and type1 in ["catégorielle", "binaire"]):
        apparie = st.radio("Les deux groupes sont-ils appariés ?", ["Non", "Oui"], index=0, horizontal=True) == "Oui"
    elif type1 == "numérique" and type2 == "numérique":
        st.info("Test de corrélation (Pearson/Spearman/Kendall) sera appliqué selon la distribution.")
    else:
        st.info("Chi² / Fisher seront considérés pour des variables catégorielles.")

    # --- Exécution du test ---
    if st.button("🧪 Exécuter le test sélectionné"):
        with st.spinner("Exécution du test... ⏳"):
            try:
                summary_df, details = propose_tests_bivaries(
                    types_df=types_df,
                    distribution_df=distribution_df,
                    df=df,
                    default_apparie=apparie
                )
            except Exception as e:
                st.error(f"Erreur lors de l'exécution des tests : {e}")
                return

            # --- Récupération de la clé ---
            key = f"{var1}__{var2}"
            alt_key = f"{var2}__{var1}"
            if key not in details and alt_key in details:
                key = alt_key
            if key not in details:
                st.warning("⚠️ Test non trouvé dans les résultats générés pour cette paire de variables.")
                st.write("Clés disponibles :", list(details.keys())[:20])
                return

            test_detail = details[key]

            # --- 1) Résumé synthétique ---
            st.subheader("📋 Résumé du test")
            summary_record = {
                "Test": test_detail.get("test", None),
                "Statistique": test_detail.get("statistic", test_detail.get("stat", None)),
                "p-value": test_detail.get("p_value", test_detail.get("p", None)),
                "p-value corrigée": test_detail.get("p_value_corrected", None),
                "Effect size": test_detail.get("effect_size", test_detail.get("effect", None)),
                "Cramers V": test_detail.get("cramers_v", None),
                "Test recommandé": test_detail.get("recommended_test", None)
            }
            st.table(pd.DataFrame([summary_record]))

            # --- 2) Détails complémentaires ---
            st.subheader("🔎 Détails")
            if "normality_var1" in test_detail or "normality_var2" in test_detail:
                st.markdown("**Tests de normalité :**")
                st.write(test_detail.get("normality_var1", "—"))
                st.write(test_detail.get("normality_var2", "—"))

            if "ci_low" in test_detail and "ci_high" in test_detail:
                st.markdown("**Intervalle de confiance bootstrap (corrélations) :**")
                st.write(f"[{test_detail['ci_low']:.3f}, {test_detail['ci_high']:.3f}]")

            if "theil_sen" in test_detail and test_detail["theil_sen"]:
                ts = test_detail["theil_sen"]
                st.markdown("**Pente Theil-Sen robuste :**")
                st.write(f"Slope = {ts['slope']:.3f}, Intercept = {ts['intercept']:.3f}, CI slope = {ts['ci_slope']}")

            # --- 3) Graphique associé ---
            st.subheader("📊 Graphique associé")
            plot_path = test_detail.get("plot") or test_detail.get("plot_boxplot")
            if plot_path and os.path.exists(plot_path):
                st.image(plot_path, use_column_width=True)
            else:
                st.info("Aucun graphique disponible pour ce test.")

            # --- 4) Table de contingence si présente ---
            if "contingency_table" in test_detail:
                st.subheader("🧾 Table de contingence")
                st.dataframe(test_detail["contingency_table"])

    # --- Navigation vers multivariés ---
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("➡️ Page suivante : Tests multivariés"):
            st.session_state.main_page = "Analyse"
            st.session_state.analyse_subpage = "Tests multivariés"
