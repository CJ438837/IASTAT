import streamlit as st
import pandas as pd
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def app():
    st.title("🔍 Analyse Bivariée - Tests statistiques")

    st.markdown("""
    Cette section permet d'explorer les relations entre deux variables à l'aide de tests bivariés adaptés :
    - **Comparaison de moyennes** (t-test, Mann-Whitney, ANOVA)
    - **Corrélations** (Pearson, Spearman, Kendall)
    - **Tests de dépendance** (Chi², Fisher)
    """)

    st.divider()

    # === Chargement du dataset ===
    st.header("📂 Chargement des données")
    uploaded_file = st.file_uploader("Chargez votre fichier CSV :", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
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
        st.write("### 📊 Types de variables détectés automatiquement")
        st.dataframe(types_df)

        st.divider()

        # === Sélection des variables ===
        st.header("🎯 Sélection des variables à comparer")

        var1 = st.selectbox("Variable 1 :", df.columns)
        var2 = st.selectbox("Variable 2 :", df.columns, index=min(1, len(df.columns) - 1))

        if var1 == var2:
            st.warning("⚠️ Veuillez sélectionner deux variables différentes.")
            return

        # === Sélection du test ===
        st.header("⚖️ Choix du test statistique")

        test_options = [
            "t-test / Mann-Whitney",
            "ANOVA / Kruskal-Wallis",
            "Chi² / Fisher",
            "Corrélation (Pearson/Spearman/Kendall)",
        ]
        test_selectionne = st.selectbox("Choisissez un test :", test_options)

        # === Options supplémentaires ===
        apparie = st.checkbox("Données appariées ?", value=False)
        alpha = st.slider("Niveau de signification α :", 0.01, 0.10, 0.05, step=0.01)

        st.divider()

        # === Exécution du test ===
        if st.button("🚀 Lancer le test"):
            with st.spinner("Analyse en cours..."):
                try:
                    resultats_df, graph = propose_tests_bivaries(
                        df=df,
                        var1=var1,
                        var2=var2,
                        test_selectionne=test_selectionne,
                        apparie=apparie,
                        alpha=alpha
                    )

                    st.success("✅ Test effectué avec succès")

                    # Affichage des résultats
                    st.subheader("📋 Résultats du test")
                    st.dataframe(resultats_df)

                    if graph is not None:
                        st.subheader("📈 Visualisation")
                        st.pyplot(graph)

                except Exception as e:
                    st.error(f"❌ Erreur pendant l’exécution : {e}")

    else:
        st.info("💡 Importez un fichier CSV pour commencer l'analyse.")
