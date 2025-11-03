import streamlit as st
import pandas as pd
from modules.IA_STAT_interactif_auto import propose_tests_interactif_auto

def app():
    st.title("📊 Tests statistiques interactifs")

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

    # --- 3️⃣ Initialisation des tests ---
    if "tests_generes" not in st.session_state:
        st.session_state.tests_generes = propose_tests_interactif_auto(
            types_df, distribution_df, df, mots_cles
        )
        st.session_state.index_test = 0  # test courant

    if not st.session_state.tests_generes:
        st.info("Aucun test n'a été généré.")
        st.stop()

    # --- 4️⃣ Navigation test par test ---
    index = st.session_state.index_test
    test_courant = st.session_state.tests_generes[index]

    st.subheader(f"Test {index + 1} / {len(st.session_state.tests_generes)}")
    st.write(f"**Type de test :** {test_courant['type']}")
    st.write(f"**Variables :** {test_courant['variables']}")
    
    # Choix apparié si applicable
    if test_courant['type'] in ["t-test", "Mann-Whitney"]:
        test_courant['apparie'] = st.radio(
            "Données appariées ?", 
            ["Non", "Oui"], 
            index=0 if not test_courant.get('apparie', False) else 1,
            key=f"app_{index}"
        ) == "Oui"

    # Bouton pour exécuter le test courant
    if st.button("Exécuter ce test", key=f"run_{index}"):
        try:
            # Ici on exécute le test (fonction spécifique déjà dans propose_tests_interactif_auto)
            test_courant['resultat'] = st.session_state.tests_generes[index]['fonction'](
                df, test_courant['variables'], apparie=test_courant.get('apparie', False)
            )
            st.success("Test exécuté avec succès !")
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du test : {e}")

    # --- 5️⃣ Flèches navigation ---
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        if st.button("⬅ Précédent") and index > 0:
            st.session_state.index_test -= 1
    with col3:
        if st.button("Suivant ➡") and index < len(st.session_state.tests_generes) - 1:
            st.session_state.index_test += 1

    # --- 6️⃣ Affichage résultats si déjà exécuté ---
    if 'resultat' in test_courant:
        st.write("### Résultat du test :")
        st.write(test_courant['resultat'])
