# Pages/page_testsbivaries.py
import streamlit as st
from modules.IA_STAT_testsbivaries import propose_tests_bivariés

def app():
    st.title("📊 Tests statistiques bivariés")

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

    st.markdown("### ⚙️ Exécution des tests bivariés")

    results_list = propose_tests_bivariés(df, types_df, distribution_df)

    for i, test_data in enumerate(results_list):
        st.markdown(f"### Test {i+1} : {test_data['test_name']} ({test_data['num']} vs {test_data['cat']})")

        # Affichage graphique
        st.pyplot(test_data['fig'])

        # Si test nécessite choix apparié/non
        if test_data['apparie_needed']:
            apparie_choice = st.radio(f"Le test {test_data['test_name']} est-il apparié ?", ("Non","Oui"), key=f"apparie_{i}")
            apparié = apparie_choice=="Oui"

            g = test_data["groupes"]
            if test_data["test_name"]=="t-test":
                stat, p = (stats.ttest_rel(g.iloc[0], g.iloc[1]) if apparié else stats.ttest_ind(g.iloc[0], g.iloc[1]))
            elif test_data["test_name"]=="Mann-Whitney":
                stat, p = (stats.wilcoxon(g.iloc[0], g.iloc[1]) if apparié else stats.mannwhitneyu(g.iloc[0], g.iloc[1]))

            # Mise à jour tableau
            test_data["result_df"].at[0, "Apparié"] = apparié
            test_data["result_df"].at[0, "Statistique"] = stat
            test_data["result_df"].at[0, "p-value"] = p

        # Affichage du tableau pour ce test
        st.dataframe(test_data["result_df"])


