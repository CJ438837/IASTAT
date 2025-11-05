import streamlit as st
import pandas as pd
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def app():
    st.title("🔍 Tests statistiques bivariés")

    # === Chargement du dataset depuis la page Fichier ===
    st.header("📂 Chargement des données")

    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page **Fichier** avant de poursuivre.")
        st.stop()
    else:
        df = st.session_state["df_selected"].copy()
        st.success(f"✅ Données chargées ({df.shape[0]} lignes, {df.shape[1]} colonnes)")
        st.dataframe(df.head())

    # === Détection automatique des types de variables ===
    types_df = pd.DataFrame({
        "variable": df.columns,
        "type": [
            "numérique" if pd.api.types.is_numeric_dtype(df[col]) else "catégorielle"
            for col in df.columns
        ]
    })

    st.markdown("### 📊 Types de variables détectés automatiquement")
    st.dataframe(types_df)

    st.divider()

    # === Sélection des variables ===
    st.header("🎯 Sélection des variables à comparer")

    var1 = st.selectbox("Variable 1 :", df.columns)
    var2 = st.selectbox("Variable 2 :", df.columns, index=min(1, len(df.columns) - 1))

    if var1 == var2:
        st.warning("⚠️ Veuillez sélectionner deux variables différentes.")
        st.stop()

    st.divider()

    # === Options du test ===
    st.markdown("### ⚙️ Paramètres du test")
    apparie = st.radio(
        "Les échantillons sont-ils appariés ?",
        ("Non", "Oui"),
        index=0
    ) == "Oui"

    lancer = st.button("🚀 Exécuter le test")

    if lancer:
        st.info("Analyse en cours... ⏳")

        try:
            results = propose_tests_bivaries(df, var1, var2, apparie)

            if results:
                for nom_test, contenu in results.items():
                    st.markdown(f"## 🧠 {nom_test}")
                    if isinstance(contenu, pd.DataFrame):
                        st.dataframe(contenu)
                    else:
                        st.write(contenu)
            else:
                st.warning("Aucun test applicable pour ces variables.")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'exécution du test : {e}")
