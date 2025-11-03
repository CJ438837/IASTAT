import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_interactif_auto import propose_tests_interactif_auto

def app():
    st.title("📊 Tests statistiques automatiques")

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

    lancer_tests = st.button("🧠 Exécuter tous les tests")

    if lancer_tests:
        with st.spinner("Exécution des tests en cours... ⏳"):
            try:
                summary_df, all_results = propose_tests_interactif_auto(
                    types_df, distribution_df, df, mots_cles, apparie=apparie
                )
                st.success("✅ Tous les tests ont été exécutés avec succès !")

                # --- 4️⃣ Affichage du résumé des tests ---
                st.markdown("### 📄 Résumé des tests")
                st.dataframe(summary_df)

                # --- 5️⃣ Graphiques générés automatiquement ---
                st.markdown("### 📊 Graphiques principaux")
                # Exemples : boxplots pour tests numériques/catégoriels
                num_vars = types_df[types_df['type'] == "numérique"]['variable'].tolist()
                cat_vars = types_df[types_df['type'].isin(['catégorielle', 'binaire'])]['variable'].tolist()

                for num, cat in [(n, c) for n in num_vars for c in cat_vars]:
                    st.markdown(f"**{num} vs {cat}**")
                    fig, ax = plt.subplots()
                    df.boxplot(column=num, by=cat, ax=ax)
                    plt.title(f"{num} vs {cat}")
                    plt.suptitle("")
                    st.pyplot(fig)
                    plt.close(fig)

                st.success("✅ Graphiques générés.")

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution des tests : {e}")
