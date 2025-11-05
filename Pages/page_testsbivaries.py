# Pages/page_testsbivaries.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def app():
    st.title("📊 Tests statistiques bivariés interactifs")

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

    # --- Sélection d'une paire de variables ---
    st.header("🎯 Sélection des variables à comparer")
    var_type_options = types_df['type'].unique().tolist()

    var1 = st.selectbox("Variable 1 :", df.columns)
    var2 = st.selectbox("Variable 2 :", df.columns, index=min(1, len(df.columns)-1))

    if var1 == var2:
        st.warning("⚠️ Veuillez sélectionner deux variables différentes.")
        return

    # --- Détection automatique du type de test ---
    type1 = types_df.loc[types_df['variable'] == var1, 'type'].values[0]
    type2 = types_df.loc[types_df['variable'] == var2, 'type'].values[0]

    st.markdown(f"**Types détectés : {var1} → {type1}, {var2} → {type2}**")

    # --- Appariement si applicable ---
    apparie = False
    if type1 == "numérique" and type2 == "numérique":
        st.info("Test de corrélation (Pearson/Spearman/Kendall)")
    elif (type1 == "numérique" and type2 in ["catégorielle","binaire"]) or (type2 == "numérique" and type1 in ["catégorielle","binaire"]):
        apparie = st.radio(
            "Les deux groupes sont-ils appariés ?",
            ("Non", "Oui"),
            index=0
        ) == "Oui"
    elif type1 in ["catégorielle","binaire"] and type2 in ["catégorielle","binaire"]:
        st.info("Test Chi² / Fisher selon la taille de la table")

    # --- Bouton pour exécuter le test ---
    if st.button("🧪 Exécuter le test"):
        with st.spinner("Exécution du test... ⏳"):
            try:
                # Exécute la fonction sur une seule paire de variables
                summary_df, details = propose_tests_bivaries(
                    types_df=types_df,
                    distribution_df=distribution_df,
                    df=df,
                    default_apparie=apparie
                )
                
                # Filtrer pour ne garder que le test sélectionné
                key = f"{var1}__{var2}"
                if key in details:
                    test_detail = details[key]
                    st.subheader(f"Résultat : {key}")
                    st.dataframe(pd.DataFrame([{
                        "Test": test_detail.get("test"),
                        "Statistique": test_detail.get("statistic"),
                        "p-value": test_detail.get("p_value"),
                        "Effect size": test_detail.get("effect_size", None),
                        "Cramers V": test_detail.get("cramers_v", None)
                    }]))

                    # Affichage du graphique
                    plot_path = test_detail.get("plot") or test_detail.get("plot_boxplot")
                    if plot_path:
                        st.image(plot_path)

                else:
                    st.warning("⚠️ Test non trouvé dans les résultats.")

            except Exception as e:
                st.error(f"❌ Une erreur est survenue pendant l'exécution du test : {e}")
