# Pages/page_testsbivaries.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from modules.IA_STAT_testbivaries import propose_tests_bivaries

def _load_theme():
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

def app():
    _load_theme()

    st.markdown("<h1 class='corvus-title'>Analyse bivariée</h1>", unsafe_allow_html=True)
    st.markdown("<p class='corvus-subtitle'>Explorez la relation entre deux variables à la fois.</p>", unsafe_allow_html=True)

    # --- prérequis
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

    # --- sélection variables
    st.subheader("🎯 Sélection des variables à comparer")
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Variable dépendante (Y)", df.columns)
    with col2:
        var2 = st.selectbox("Variable explicative (X)", [c for c in df.columns if c != var1])

    if var1 == var2:
        st.warning("⚠️ Sélectionnez deux variables différentes.")
        st.stop()

    # normaliser types simples
    def _norm_type(val):
        if isinstance(val, str):
            v = val.lower()
            if v in ("bool", "boolean", "binaire"):
                return "binaire"
            return val
        return val

    try:
        type1 = _norm_type(types_df.loc[types_df["variable"] == var1, "type"].values[0])
    except Exception:
        type1 = "inconnu"
    try:
        type2 = _norm_type(types_df.loc[types_df["variable"] == var2, "type"].values[0])
    except Exception:
        type2 = "inconnu"

    st.markdown(f"**Types détectés :** `{var1}` → {type1}, `{var2}` → {type2}")

    # options
    apparie = False
    if (type1 == "numérique" and type2 in ["catégorielle", "binaire"]) or (type2 == "numérique" and type1 in ["catégorielle", "binaire"]):
        apparie = st.radio("Les deux groupes sont-ils appariés ?", ["Non", "Oui"], index=0, horizontal=True) == "Oui"

    if st.button("📊 Démarrer le test", use_container_width=True):
        with st.spinner("Exécution du test... ⏳"):
            try:
                raw_result = propose_tests_bivaries(
                    types_df=types_df,
                    distribution_df=distribution_df,
                    df=df,
                    default_apparie=apparie
                )

                # --- ------------- Robust parsing of return value -------------
                summary_df = pd.DataFrame()
                details = {}

                # Case 1: tuple/list like (summary_df, details) or [summary_df, details]
                if isinstance(raw_result, (tuple, list)):
                    if len(raw_result) >= 2:
                        candidate_summary, candidate_details = raw_result[0], raw_result[1]
                        if isinstance(candidate_summary, pd.DataFrame):
                            summary_df = candidate_summary.copy()
                        elif isinstance(candidate_summary, dict):
                            # maybe dict of records -> try to convert
                            try:
                                summary_df = pd.DataFrame(candidate_summary)
                            except Exception:
                                summary_df = pd.DataFrame()
                        if isinstance(candidate_details, dict):
                            details = candidate_details.copy()
                        else:
                            # if second element is dataframe -> try to transform into details map
                            details = {}
                            if isinstance(candidate_details, pd.DataFrame):
                                # attempt to build by test_id
                                if "test_id" in candidate_details.columns:
                                    for _, r in candidate_details.iterrows():
                                        details[str(r["test_id"])] = r.to_dict()
                    elif len(raw_result) == 1:
                        raw0 = raw_result[0]
                        if isinstance(raw0, dict):
                            # either details dict or {summary_df, details}
                            if "details" in raw0 or "summary_df" in raw0:
                                summary_df = pd.DataFrame(raw0.get("summary_df", []))
                                details = raw0.get("details", {})
                            else:
                                # maybe it's details mapping directly
                                details = raw0
                        elif isinstance(raw0, pd.DataFrame):
                            summary_df = raw0.copy()

                # Case 2: dict
                elif isinstance(raw_result, dict):
                    # If it has keys 'summary_df' or 'details', use them
                    if "details" in raw_result or "summary_df" in raw_result:
                        summary_df = pd.DataFrame(raw_result.get("summary_df", []))
                        details = raw_result.get("details", {}) or {}
                    else:
                        # maybe it's directly the details mapping (keys are test ids)
                        # detect keys like "var1__var2"
                        keys = list(raw_result.keys())
                        plausible_keys = [k for k in keys if "__" in k]
                        if plausible_keys:
                            details = raw_result
                        else:
                            # else try to convert to dataframe
                            try:
                                summary_df = pd.DataFrame(raw_result)
                            except Exception:
                                summary_df = pd.DataFrame()

                # Case 3: DataFrame
                elif isinstance(raw_result, pd.DataFrame):
                    summary_df = raw_result.copy()

                # Final fallback: if summary_df contains the test row, try to build details from it
                key = f"{var1}__{var2}"
                if not details and not summary_df.empty and "test_id" in summary_df.columns:
                    # try to find matching row(s)
                    rows = summary_df[summary_df["test_id"] == key]
                    if not rows.empty:
                        # convert first matching row to details dict (best-effort)
                        r = rows.iloc[0].to_dict()
                        details[key] = {
                            "test": r.get("test") or r.get("Test") or r.get("Test name"),
                            "statistic": r.get("stat"),
                            "p_value": r.get("p_value"),
                            "effect_size": r.get("effect") or r.get("effect_size"),
                        }

                # If still no details, try to detect if raw_result itself contained the key
                if not details:
                    if isinstance(raw_result, dict) and key in raw_result:
                        details = {key: raw_result[key]}

                # --- If still no details found, show debug info and graceful error
                if key not in details:
                    st.error("⚠️ Aucun résultat exploitable trouvé pour cette paire de variables.")
                    st.markdown("**Diagnostic (type du retour) :**")
                    st.write(type(raw_result))
                    # show keys/cols for developer debugging
                    try:
                        if isinstance(raw_result, dict):
                            st.markdown("**Clés du dict retourné :**")
                            st.write(list(raw_result.keys())[:50])
                        elif isinstance(raw_result, (tuple, list)):
                            st.markdown("**Structure tuple/list — types des éléments :**")
                            st.write([type(x).__name__ for x in raw_result])
                            # if any element is dataframe show columns
                            for i, x in enumerate(raw_result):
                                if isinstance(x, pd.DataFrame):
                                    st.markdown(f"Element {i} : DataFrame colonnes")
                                    st.write(x.columns.tolist())
                        elif isinstance(raw_result, pd.DataFrame):
                            st.markdown("**DataFrame retourné — colonnes :**")
                            st.write(raw_result.columns.tolist())
                    except Exception:
                        pass

                    st.info("➡️ Probables solutions :\n"
                            "- Mettre à jour `propose_tests_bivaries()` pour renvoyer `(summary_df, details)`.\n"
                            "- Ou me dire ici le format exact retourné et je l'adapte automatiquement.")
                    st.stop()

                # ---------- Si on a les details pour la clé ----------
                result = details.get(key, {})
                st.subheader("📋 Résultats du test")
                df_res = pd.DataFrame([{
                    "Test": result.get("test"),
                    "Statistique": result.get("statistic"),
                    "p-value": result.get("p_value"),
                    "Effect size": result.get("effect_size", None),
                    "Cramer's V": result.get("cramers_v", None)
                }])
                st.dataframe(df_res, use_container_width=True)

                # graphique associé si présent
                plot_path = result.get("plot") or result.get("plot_boxplot")
                if plot_path:
                    if isinstance(plot_path, (list, tuple)):
                        # parfois le module peut renvoyer une liste de chemins -> afficher le premier
                        plot_path = plot_path[0]
                    if os.path.exists(plot_path):
                        st.image(plot_path, use_column_width=True)
                    else:
                        # si c'est une image b64/objet, essayer d'afficher directement
                        try:
                            st.image(plot_path, use_column_width=True)
                        except Exception:
                            st.warning("Graphique fourni mais introuvable ou format non reconnu.")
                else:
                    st.info("Aucun graphique disponible pour ce test.")

                # résidus si disponibles
                if "residus_plot" in result and result["residus_plot"]:
                    st.subheader("📉 Analyse des résidus")
                    try:
                        st.image(result["residus_plot"], use_column_width=True)
                    except Exception:
                        st.write("Résidus fournis mais impossible à afficher.")

            except Exception as e:
                st.error(f"❌ Erreur pendant l'exécution du test : {e}")

    # navigation
    st.markdown("---")
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
    with nav_col2:
        if st.button("➡️ Page suivante : Tests multivariés", use_container_width=True):
            st.session_state.target_page = "Tests multivariés"
