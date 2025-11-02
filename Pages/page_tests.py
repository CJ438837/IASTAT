import streamlit as st
from modules.IA_STAT_interactif2 import propose_tests_interactif_streamlit

def app():
    st.title("📊 Tests statistiques interactifs")
    st.write("Le dur du sujet ! Voyons ce que tes données ont dans le ventre.")

    # --- 1️⃣ Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page Variables.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("Veuillez d'abord analyser les distributions dans la page Distribution.")
        st.stop()

    df = st.session_state["df_selected"]
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"]
    mots_cles = st.session_state.get("keywords", [])

    # --- 2️⃣ Assurer l’existence de la colonne 'type' ---
    # Renommer automatiquement si elle a un autre nom
    if 'type' not in types_df.columns:
        for col_name in types_df.columns:
            if col_name.lower() in ['type', 'var_type', 'variable_type']:
                types_df = types_df.rename(columns={col_name: 'type'})
                break
        else:
            st.error("Le DataFrame des types ne contient aucune colonne de type valide ('type', 'var_type', etc.).")
            st.stop()

    st.session_state["types_df"] = types_df  # mise à jour

    st.success("✅ Fichier importé, types détectés et distributions analysées.")

    # --- 3️⃣ Lancer les tests interactifs ---
    st.markdown("### 💡 Propositions de tests")
    if st.button("Lancer les tests interactifs"):
        # Passer le types_df corrigé
        propose_tests_interactif_streamlit(types_df, distribution_df, df, mots_cles)
        st.success("✅ Tous les tests interactifs ont été proposés et exécutés.")
