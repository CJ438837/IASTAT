# Pages/page_testsmulti.py
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.IA_STAT_testmultivaries import propose_tests_multivariés

plt.style.use("seaborn-v0_8-muted")


def _render_info_block(info):
    """
    info est attendu comme dict (safe_info dans le module garantit ça).
    On rend proprement selon le contenu.
    """
    if not info:
        return
    if not isinstance(info, dict):
        st.write(info)
        return

    # Si table fournie
    if "table" in info and isinstance(info["table"], list):
        try:
            df = pd.DataFrame(info["table"])
            st.dataframe(df, use_container_width=True)
            return
        except Exception:
            pass

    # Afficher clefs/valeurs de manière lisible
    for k, v in info.items():
        # petites règles d'affichage selon type
        if v is None:
            st.write(f"**{k}** : —")
        elif isinstance(v, (str, int, float, bool)):
            st.write(f"**{k}** : {v}")
        elif isinstance(v, list):
            st.write(f"**{k}** :")
            try:
                # si liste de dicts, tabuler
                if len(v) > 0 and isinstance(v[0], dict):
                    st.dataframe(pd.DataFrame(v), use_container_width=True)
                else:
                    st.write(v)
            except Exception:
                st.write(v)
        elif isinstance(v, dict):
            st.write(f"**{k}** :")
            # afficher paires clef-valeur
            for kk, vv in v.items():
                st.write(f"- **{kk}** : {vv}")
        else:
            # fallback
            st.write(f"**{k}** : {v}")


def app():
    # --- Thème Corvus ---
    try:
        with open("assets/corvus_theme.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    st.title("📊 Tests Multivariés")
    st.markdown("---")

    st.markdown("""
    **Analyse multivariée automatisée — guide & exécution**  
    Ici vous pouvez lancer plusieurs analyses multivariées (PCA, MCA/FAMD, MANOVA, régression multiple, diagnostics, corrélations, tests multivariés).
    """)

    # --- Vérifications préalables ---
    if "df_selected" not in st.session_state:
        st.warning("Veuillez d'abord importer un fichier dans la page **Fichier**.")
        st.stop()
    if "types_df" not in st.session_state:
        st.warning("Veuillez d'abord détecter les types de variables dans la page **Variables**.")
        st.stop()

    df = st.session_state["df_selected"].copy()
    types_df = st.session_state["types_df"].copy()

    st.success(f"✅ Données prêtes — {df.shape[0]} lignes x {df.shape[1]} colonnes")

    # --- Aperçu / sélection variables ---
    st.markdown("---")
    st.subheader("🎯 Sélection des variables")
    st.markdown("Choisissez la variable cible (à expliquer) puis une ou plusieurs variables explicatives.")

    cols = df.columns.tolist()
    col1, col2 = st.columns([2, 3])
    with col1:
        target_var = st.selectbox("Variable cible (Y)", cols)
    with col2:
        explicatives = st.multiselect("Variables explicatives (X)", [c for c in cols if c != target_var])

    if not explicatives:
        st.info("Sélectionnez au moins une variable explicative pour pouvoir lancer l'analyse.")
        st.stop()

    st.markdown("---")

    # Options (ex : appeler avec options futures)
    st.write("Options :")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        run_button = st.button("📈 Exécuter l'analyse multivariée", type="primary")
    with col_opt2:
        show_all = st.checkbox("Afficher tous les tests même si non applicables", value=False)

    if run_button:
        with st.spinner("Exécution des tests multivariés... ⏳"):
            try:
                results = propose_tests_multivariés(df=df, types_df=types_df, target_var=target_var, explicatives=explicatives)
            except Exception as e:
                st.error(f"❌ Erreur lors de l'appel à propose_tests_multivariés : {e}")
                return

            # Résultats : liste de dicts
            if not isinstance(results, list):
                st.error("Le module n'a pas renvoyé la liste attendue de résultats.")
                return

            # Afficher chaque résultat
            for res in results:
                st.markdown("---")
                test_name = res.get("test", "Test inconnu")
                st.subheader(f"🧩 {test_name}")

                # Erreur levée côté module
                if "error" in res:
                    st.error(f"❌ Erreur : {res.get('error')}")
                    # afficher info potentiellement utile
                    if res.get("info"):
                        st.info("Détails:")
                        _render_info_block(res.get("info"))
                    continue

                # Message / info courte
                if res.get("info") and isinstance(res.get("info"), dict) and "info" in res.get("info") and isinstance(res.get("info")["info"], str):
                    # cas simple
                    st.info(res.get("info")["info"])

                # Résultat tabulaire si présent
                result_df = res.get("result_df", None)
                if result_df is not None:
                    try:
                        if isinstance(result_df, pd.DataFrame):
                            if not result_df.empty:
                                st.markdown("**Résultat (aperçu)**")
                                st.dataframe(result_df, use_container_width=True)
                        else:
                            # tenter convertir
                            df_try = pd.DataFrame(result_df)
                            if not df_try.empty:
                                st.markdown("**Résultat (aperçu)**")
                                st.dataframe(df_try, use_container_width=True)
                    except Exception:
                        st.write(result_df)

                # Figure
                fig = res.get("fig", None)
                if fig is not None:
                    try:
                        st.pyplot(fig)
                    except Exception:
                        st.write("Figure fournie mais impossible à afficher.")

                # Interpretation (si fournie)
                if res.get("interpretation"):
                    st.markdown(f"**Interprétation :** {res.get('interpretation')}")

                # Info détaillée (toujours dict grâce à _safe_info)
                if res.get("info"):
                    st.markdown("**Informations complémentaires :**")
                    _render_info_block(res.get("info"))

            st.success("✅ Analyse multivariée terminée.")

    # Navigation rapide vers la page bivariée
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col3:
        if st.button("➡️ Besoin d'une aide théorique ?", use_container_width=True):
            st.session_state.main_page = "Théorie"
            st.session_state.theorie_subpage = "Tests multivariés"
            

    st.markdown("© 2025 Corvus Analytics - Tous droits réservés")
