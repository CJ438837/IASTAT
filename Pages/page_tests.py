import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from modules.IA_STAT_interactif_auto import (
    propose_tests_interactif_auto_anova,
    propose_tests_interactif_auto_kruskal,
    propose_tests_interactif_auto_ttest,
    propose_tests_interactif_auto_mannwhitney,
    propose_tests_interactif_auto_chi2,
    propose_tests_interactif_auto_correlation
)

def afficher_boxplots(df, num_vars, cat_vars):
    """Affiche les boxplots Numérique vs Catégoriel"""
    st.markdown("#### 📊 Visualisation : Boxplots")
    for num, cat in [(n, c) for n in num_vars for c in cat_vars]:
        fig, ax = plt.subplots()
        sns.boxplot(x=cat, y=num, data=df, ax=ax)
        ax.set_title(f"{num} vs {cat}")
        st.pyplot(fig)
        plt.close(fig)


def app():
    st.title("🧠 Tests statistiques interactifs")

    if "df_selected" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord définir les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord analyser les distributions dans la page Distribution.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
    cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

    st.divider()
    st.header("📈 Tests de comparaison de moyennes")

    # --- ANOVA ---
    st.subheader("ANOVA")
    apparie_anova = st.radio("Les groupes sont-ils appariés ?", ("Non", "Oui"), key="anova_app", index=0) == "Oui"
    if st.button("Lancer ANOVA"):
        with st.spinner("Exécution du test ANOVA..."):
            try:
                summary_df, all_results = propose_tests_interactif_auto_anova(
                    types_df, distribution_df, df, mots_cles, apparie=apparie_anova
                )
                st.success("✅ ANOVA exécutée avec succès !")
                st.dataframe(summary_df)
                afficher_boxplots(df, num_vars, cat_vars)
            except Exception as e:
                st.error(f"Erreur ANOVA : {e}")

    # --- Kruskal-Wallis ---
    st.subheader("Kruskal-Wallis")
    apparie_kw = st.radio("Les groupes sont-ils appariés ?", ("Non", "Oui"), key="kw_app", index=0) == "Oui"
    if st.button("Lancer Kruskal-Wallis"):
        with st.spinner("Exécution du test de Kruskal-Wallis..."):
            try:
                summary_df, all_results = propose_tests_interactif_auto_kruskal(
                    types_df, distribution_df, df, mots_cles, apparie=apparie_kw
                )
                st.success("✅ Test de Kruskal-Wallis exécuté avec succès !")
                st.dataframe(summary_df)
                afficher_boxplots(df, num_vars, cat_vars)
            except Exception as e:
                st.error(f"Erreur Kruskal-Wallis : {e}")

    # --- t-test ---
    st.subheader("t-test")
    apparie_ttest = st.radio("Les échantillons sont-ils appariés ?", ("Non", "Oui"), key="ttest_app", index=0) == "Oui"
    if st.button("Lancer t-test"):
        with st.spinner("Exécution du test t de Student..."):
            try:
                summary_df, all_results = propose_tests_interactif_auto_ttest(
                    types_df, distribution_df, df, mots_cles, apparie=apparie_ttest
                )
                st.success("✅ t-test exécuté avec succès !")
                st.dataframe(summary_df)
                afficher_boxplots(df, num_vars, cat_vars)
            except Exception as e:
                st.error(f"Erreur t-test : {e}")

    # --- Mann-Whitney ---
    st.subheader("Mann-Whitney")
    apparie_mw = st.radio("Les échantillons sont-ils appariés ?", ("Non", "Oui"), key="mw_app", index=0) == "Oui"
    if st.button("Lancer Mann-Whitney"):
        with st.spinner("Exécution du test de Mann-Whitney..."):
            try:
                summary_df, all_results = propose_tests_interactif_auto_mannwhitney(
                    types_df, distribution_df, df, mots_cles, apparie=apparie_mw
                )
                st.success("✅ Test de Mann-Whitney exécuté avec succès !")
                st.dataframe(summary_df)
                afficher_boxplots(df, num_vars, cat_vars)
            except Exception as e:
                st.error(f"Erreur Mann-Whitney : {e}")

    # --- Khi² ---
    st.divider()
    st.header("📊 Tests de dépendance")
    st.subheader("Chi²")
    if st.button("Lancer Chi²"):
        with st.spinner("Exécution du test du Chi²..."):
            try:
                summary_df, all_results = propose_tests_interactif_auto_chi2(
                    types_df, distribution_df, df, mots_cles
                )
                st.success("✅ Test du Chi² exécuté avec succès !")
                st.dataframe(summary_df)
            except Exception as e:
                st.error(f"Erreur Chi² : {e}")

    # --- Corrélations ---
    st.divider()
    st.header("🔗 Tests de corrélation")
    if st.button("Lancer les corrélations"):
        with st.spinner("Exécution des tests de corrélation..."):
            try:
                summary_df, all_results = propose_tests_interactif_auto_correlation(
                    types_df, distribution_df, df, mots_cles
                )
                st.success("✅ Corrélations calculées avec succès !")
                st.dataframe(summary_df)

                # Heatmap de corrélation
                st.markdown("#### 🔥 Heatmap des corrélations")
                corr = df[num_vars].corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Erreur corrélation : {e}")
