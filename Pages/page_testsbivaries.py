import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from modules.IA_STAT_testbivaries import propose_tests_bivariés # version bivariée que tu as créée
import numpy as np

def app():
    st.title("📊 Tests statistiques bivariés")

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

    # --- 2️⃣ Récupération des données depuis la session ---
    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()
    mots_cles = st.session_state.get("keywords", [])

    # --- 3️⃣ Sélection des options utilisateur ---
    st.markdown("### ⚙️ Options des tests")
    apparie = st.radio(
        "Les tests à deux groupes sont-ils appariés ?",
        ("Non", "Oui"),
        index=0
    ) == "Oui"

    lancer_tests = st.button("🧠 Exécuter tous les tests bivariés")

    if lancer_tests:
        with st.spinner("Exécution des tests en cours... ⏳"):
            try:
                # exécution des tests bivariés automatiques
                all_results = propose_tests_bivariés(
                    types_df, distribution_df, df, mots_cles, apparie=apparie
                )
                st.success("✅ Tous les tests bivariés ont été exécutés avec succès !")

                # --- 4️⃣ Affichage des résultats test par test ---
                for test_data in all_results:
                    st.markdown(f"### {test_data['test_name']} : {test_data.get('num', '')} vs {test_data.get('cat', '')}{test_data.get('var1','')} {test_data.get('var2','')}")
                    
                    # Résultats statistiques
                    st.write("Statistique :", test_data.get('stat'))
                    st.write("p-value :", test_data.get('p'))

                    # Graphiques
                    fig, ax = plt.subplots()
                    if test_data['test_type'] == "num_vs_cat":
                        sns.boxplot(x=test_data['cat'], y=test_data['num'], data=df, ax=ax)
                    elif test_data['test_type'] == "num_vs_num":
                        sns.scatterplot(x=test_data['var1'], y=test_data['var2'], data=df, ax=ax)
                    elif test_data['test_type'] == "cat_vs_cat":
                        contingency_table = pd.crosstab(df[test_data['var1']], df[test_data['var2']])
                        sns.heatmap(contingency_table, annot=True, fmt="d", cmap="coolwarm", ax=ax)
                    ax.set_title(f"{test_data['test_name']}")
                    st.pyplot(fig)
                    plt.close(fig)

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution des tests : {e}")
