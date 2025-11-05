import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def app():
    st.title("📊 Tests bivariés automatiques")

    # === Chargement des données depuis la page Fichier ===
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page Fichier.")
        st.stop()
    
    df = st.session_state["df_selected"].copy()

    # === Détection automatique des types de variables ===
    types_df = pd.DataFrame({
        "variable": df.columns,
        "type": [
            "numérique" if pd.api.types.is_numeric_dtype(df[col]) else "catégorielle"
            for col in df.columns
        ]
    }).rename(columns=lambda x: x.strip())  # nettoyage des espaces invisibles

    # Vérification stricte des colonnes attendues
    if 'variable' not in types_df.columns or 'type' not in types_df.columns:
        st.error(f"❌ types_df doit contenir les colonnes 'variable' et 'type'. Colonnes actuelles : {types_df.columns.tolist()}")
        st.stop()

    st.write("### 📊 Types de variables détectés automatiquement")
    st.dataframe(types_df)

    st.divider()

    # === Sélection des variables à comparer ===
    st.header("🎯 Sélection des variables à comparer")
    var1 = st.selectbox("Variable 1 :", df.columns)
    var2 = st.selectbox("Variable 2 :", df.columns, index=min(1, len(df.columns) - 1))

    if var1 == var2:
        st.warning("⚠️ Veuillez sélectionner deux variables différentes.")
        return

    # === Options utilisateur pour tests appariés ===
    apparie = st.radio(
        "Les tests à deux groupes sont-ils appariés ?",
        ("Non", "Oui"),
        index=0
    ) == "Oui"

    lancer_tests = st.button("🧠 Exécuter les tests bivariés")

    if lancer_tests:
        with st.spinner("Exécution des tests en cours... ⏳"):
            try:
                summary_df, all_results = propose_tests_bivaries(
                    types_df, df, var1, var2, apparie=apparie
                )
                st.success("✅ Tests exécutés avec succès !")

                # --- Affichage du résumé des tests ---
                st.markdown("### 📄 Résumé des tests")
                st.dataframe(summary_df)

                # --- Graphiques ---
                st.markdown("### 📊 Visualisations")
                for key, res in all_results.items():
                    if "fig" in res:
                        st.pyplot(res["fig"])
                        plt.close(res["fig"])

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution des tests : {e}")
