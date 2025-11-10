import streamlit as st
import pandas as pd
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def app():
    # --- 🎨 Thème global ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # --- 🧠 En-tête ---
    st.markdown("<h1 class='corvus-title'>Analyse bivariée</h1>", unsafe_allow_html=True)
    st.markdown("<p class='corvus-subtitle'>Explorez la relation entre deux variables à la fois.</p>", unsafe_allow_html=True)

    # --- 1️⃣ Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord importer un fichier dans la page **Fichier**.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord détecter les types de variables dans la page **Variables**.")
        st.stop()
    if "distribution_df" not in st.session_state:
        st.warning("⚠️ Veuillez d'abord analyser la distribution des données dans la page **Distribution**.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()
    distribution_df = st.session_state["distribution_df"].copy()

    st.success("✅ Données et analyses de distribution prêtes.")

    # --- 2️⃣ Sélection des variables ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Sélection des variables à comparer")

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Variable dépendante (Y)", df.columns)
    with col2:
        var2 = st.selectbox("Variable explicative (X)", [c for c in df.columns if c != var1])

    st.markdown("</div>", unsafe_allow_html=True)

    if var1 == var2:
        st.warning("⚠️ Sélectionnez deux variables différentes.")
        st.stop()

    # Détection automatique du type
    def normalize_type(t):
        if isinstance(t, str) and t.lower() in ["bool", "boolean", "binaire"]:
            return "binaire"
        return t

    type1 = normalize_type(types_df.loc[types_df["variable"] == var1, "type"].values[0])
    type2 = normalize_type(types_df.loc[types_df["variable"] == var2, "type"].values[0])

    st.markdown(f"**Types détectés :** `{var1}` → {type1}, `{var2}` → {type2}")

    # --- 3️⃣ Options de test ---
    apparie = False
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Options d'analyse")

    if type1 == "numérique" and type2 == "numérique":
        st.info("💡 Un test de corrélation (Pearson, Spearman ou Kendall) sera appliqué selon la distribution.")
    elif (type1 == "numérique" and type2 in ["catégorielle", "binaire"]) or (type2 == "numérique" and type1 in ["catégorielle", "binaire"]):
        apparie = st.radio(
            "Les deux groupes sont-ils appariés ?",
            ["Non", "Oui"],
            index=0,
            horizontal=True
        ) == "Oui"
    elif type1 in ["catégorielle", "binaire"] and type2 in ["catégorielle", "binaire"]:
        st.info("💡 Un test du Chi² ou de Fisher sera utilisé selon la taille de la table de contingence.")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- 4️⃣ Lancement du test ---
    st.markdown("<div class='corvus-card'>", unsafe_allow_html=True)
    st.subheader("📈 Lancer le test bivarié")

    if st.button("📊 Démarrer le test", use_container_width=True, type="primary"):
        with st.spinner("Analyse en cours... ⏳"):
            try:
                # --- Compatibilité automatique selon le format du retour ---
                result = propose_tests_bivaries(
                    types_df=types_df,
                    distribution_df=distribution_df,
                    df=df,
                    default_apparie=apparie
                )

                # Certains modules renvoient (summary_df, details)
                if isinstance(result, tuple) and len(result) == 2:
                    summary_df, details = result
                # D'autres renvoient un seul dict {summary_df, details}
                elif isinstance(result, dict):
                    summary_df = result.get("summary_df", pd.DataFrame())
                    details = result.get("details", {})
                else:
                    raise ValueError("Format de retour inattendu depuis propose_tests_bivaries().")

                key = f"{var1}__{var2}"
                if key not in details:
                    st.warning("Aucun résultat trouvé pour cette paire de variables.")
                    st.stop()

                result = details[key]

                # --- Résumé du test ---
                st.subheader("📋 Résultats du test")
                st.dataframe(pd.DataFrame([{
                    "Test": result.get("test"),
                    "Statistique": result.get("statistic"),
                    "p-value": result.get("p_value"),
                    "Effect size": result.get("effect_size", None),
                    "Cramer's V": result.get("cramers_v", None)
                }]), use_container_width=True)

                # --- Graphique associé ---
                st.subheader("📊 Visualisation du résultat")
                plot_path = result.get("plot") or result.get("plot_boxplot")
                if plot_path:
                    st.image(plot_path, use_container_width=True)
                else:
                    st.info("Aucun graphique disponible pour ce test.")

                # --- Analyse des résidus ---
                if "residus_plot" in result and result["residus_plot"]:
                    st.subheader("📉 Analyse des résidus")
                    st.image(result["residus_plot"], use_container_width=True)

                if "residus_summary" in result and result["residus_summary"] is not None:
                    st.dataframe(result["residus_summary"], use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erreur pendant l'exécution du test : {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- 5️⃣ Navigation entre pages ---
    st.markdown("<hr>", unsafe_allow_html=True)
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col2:
        if st.button("➡️ Page suivante : Tests multivariés", use_container_width=True):
            st.session_state.target_page = "Tests multivariés"
